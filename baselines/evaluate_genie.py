#!/usr/bin/env python3

import argparse
import time
import os
import sys
from collections import defaultdict

import lpips
import numpy as np
import torch
import torchvision.transforms.functional as transforms_f
from einops import rearrange
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from transformers import default_data_collator

# 1xgpt imports
sys.path.append(os.getcwd())
from data import RawTokenDataset
from visualize import decode_latents_wrapper
from evaluate import decode_tokens, compute_loss_and_acc, compute_lpips
from baselines.genie_world_model import LitWorldModel

# Hardcoded values for the final dataset
WINDOW_SIZE = 16
STRIDE = 15  # Data is 30 Hz so with stride 15, video is 2 Hz
LATENT_H, LATENT_W = 20, 20  # Dimensions of the compressed image

DEBUG = False
SAVE_OUTPUTS = "data/genie/outputs_1_160k.pkl"


def parse_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--val_data_dir", type=str, default="data/val_v0",
        help="A directory with video data, should have a `metadata.json` and `video.bin`."
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to a Lightning checkpoint."
    )
    parser.add_argument(
        "--batch_size", type=int, default=16,
        help="Batch size, current script only supports a single GPU."
    )
    parser.add_argument(
        "--num_layers", type=int, default=8, help="Num hidden layers"
    )
    parser.add_argument(
        "--num_heads", type=int, default=16, help="Num attention heads"
    )
    parser.add_argument(
        "--d_model", type=int, default=1024, help="Hidden size"
    )
    parser.add_argument(
        "--maskgit_steps", type=int, default=8, help="Number of maskgit sampling steps."
    )
    parser.add_argument(
        "--temperature", type=float, default=1., help="Sampling temperature."
    )

    return parser.parse_args()


class GenieEvaluator:
    def __init__(self, args, wrapped_decode_latents, device="cuda"):
        super().__init__()
        self.model = LitWorldModel.load_from_checkpoint(args.checkpoint, T=WINDOW_SIZE, S=LATENT_H * LATENT_W,
                                                        image_vocab_size=1001, num_layers=args.num_layers,
                                                        num_heads=args.num_heads, d_model=args.d_model).model
        self.model.eval()

        self.wrapped_decode_latents = wrapped_decode_latents
        self.device = device
        self.maskgit_steps, self.temperature = args.maskgit_steps, args.temperature

    def predict_zframe_logits(self, input_ids: torch.LongTensor) -> torch.Tensor:
        """
        Conditioned on each prefix: [frame_0], [frame_0, frame_1], ..., [frame_0, frame_1, ... frame_{T-1}],
        predict the tokens in the following frame: [pred_frame_1, pred_frame_2, ..., pred_frame_T].

        Image logits are denoised in parallel across spatial dimension and teacher-forced
        across the time dimension. To compute logits, we save both the samples and logits as we do MaskGIT generation.

        Total number of forward passes is (T-1) * maskgit steps.

        Args:
            input_ids: LongTensor of size (B, T*H*W) corresponding to flattened, tokenized images.

        Returns:
            FloatTensor of size (B, C, T-1, H, W) corresponding to the predicted logits,
            where C=1000 is the number of classes.
        """
        inputs_THW = rearrange(input_ids, "b (t h w) -> b t h w", t=WINDOW_SIZE, h=LATENT_H, w=LATENT_W).to(self.device)
        all_samples = []
        all_logits = []
        for timestep in range(1, WINDOW_SIZE):
            print(f"Generating frame {timestep}")
            inputs_masked = inputs_THW.clone()
            inputs_masked[:, timestep:] = self.model.image_mask_token

            # single forward pass from masked image - for debugging purposes on zero-shot prediction
            if DEBUG:
                logits_CTHW = self.model(inputs_masked)
                logits_CHW = logits_CTHW[:, :-1, timestep]
                sample_HW = logits_CHW.argmax(dim=1)
            else:
                # maskgit sampling
                sample_HW, logits_CHW = self.model.maskgit_generate(
                    inputs_masked, out_t=timestep, maskgit_steps=self.maskgit_steps, temperature=self.temperature
                )

            all_samples.append(sample_HW)
            all_logits.append(logits_CHW)

        samples = torch.stack(all_samples, dim=1)
        return samples, torch.stack(all_logits, dim=2)

    def predict_next_frames(self, samples_THW) -> torch.Tensor:
        """
        All model submissions should have this defined.

        Like predict_next_frames, this is teacher-forced along time dimension, autoregressive along spatial dimension.

        Conditioned on each prefix: [frame_0], [frame_0, frame_1], ..., [frame_0, frame_1, ..., frame_{T-1}],
        predict the following frame: [pred_frame_1, pred_frame_2, ..., pred_frame_T].

        For this model, the frames are generated by using the argmax of `predict_zframe_logits`
        and decoding the quantized latent space tokens back to the original image space.

        Args:
            samples_THW: LongTensor of size (B, T, H, W) corresponding to sampled images in the quantized latent space.

        Returns:
            LongTensor of size (B, T-1, 3, 160, 160) corresponding to the predicted frames.
        """
        return decode_tokens(samples_THW.cpu(), self.wrapped_decode_latents)


@torch.no_grad()
def main():
    args = parse_args()

    wrapped_decode_latents = decode_latents_wrapper()
    val_dataset = RawTokenDataset(args.val_data_dir, window_size=WINDOW_SIZE, stride=STRIDE)
    evaluator = GenieEvaluator(args, wrapped_decode_latents)

    # To save time, only evaluate on each chunk once instead of using a sliding window.
    val_dataset = Subset(
        val_dataset,
        [i for chunk_start in range(val_dataset.stride)
         for i in range(chunk_start, len(val_dataset), val_dataset.stride * val_dataset.window_size)]
    )

    dataloader = DataLoader(val_dataset, collate_fn=default_data_collator, batch_size=args.batch_size)

    lpips_alex = lpips.LPIPS(net="alex")  # Calculate LPIPS w/ AlexNet, which is the fastest model out of their options
    metrics = defaultdict(list)

    if SAVE_OUTPUTS is not None:
        all_outputs = []

    for batch in tqdm(dataloader):
        reshaped_input_ids = rearrange(batch["input_ids"], "b (t h w) -> b t h w", t=WINDOW_SIZE, h=LATENT_H,
                                       w=LATENT_W)

        start_time = time.time()

        samples, logits = evaluator.predict_zframe_logits(batch["input_ids"])
        frames_per_batch = (WINDOW_SIZE - 1) * batch["input_ids"].size(0)
        metrics["gen_time"].append((time.time() - start_time) / frames_per_batch)

        loss, _ = compute_loss_and_acc(batch["input_ids"], logits)

        acc = (reshaped_input_ids[:, 1:].to("cuda") == samples).float().mean().item()
        
        metrics["loss"].append(loss)
        metrics["acc"].append(acc)

        start_time = time.time()
        pred_frames = evaluator.predict_next_frames(samples)
        metrics["dec_time"].append((time.time() - start_time) / frames_per_batch)

        decoded_gtruth = decode_tokens(reshaped_input_ids, wrapped_decode_latents)
        metrics["pred_lpips"].extend(compute_lpips(decoded_gtruth[:, 1:], pred_frames, lpips_alex))
        
        print({key: np.mean(val) for key, val in metrics.items() if len(val) > 0})
        if SAVE_OUTPUTS is not None:
            os.makedirs(os.path.dirname(SAVE_OUTPUTS), exist_ok=True)
            all_outputs.append((pred_frames, decoded_gtruth, logits, reshaped_input_ids))

    if SAVE_OUTPUTS is not None:
        with open(SAVE_OUTPUTS, "wb") as f:
            import pickle
            pickle.dump(all_outputs, f)


if __name__ == "__main__":
    main()
