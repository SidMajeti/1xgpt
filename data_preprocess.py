"""
Define a class to convert raw data into tokens. Use MagVit
"""
import json
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
from einops import rearrange
# from magvit2.models import lfqgan
import argparse
from torch.utils.data import Dataset, DataLoader
import torchvision
from magvit2.config import VQConfig
from tqdm import tqdm
import psutil
from torchmetrics.functional import structural_similarity_index_measure
import torch.nn.functional as F  # Also needed for F.mse_loss
from PIL import Image  # Add this with other imports at the top
from cosmos_tokenizer.video_lib import CausalVideoTokenizer


COMPRESSED_WIDTH = 16
COMPRESSED_HEIGHT = 9

COMPRESSION_FACTOR = 0.5

SUB_BATCH_SIZE = 12
SUB_TOKEN_SIZE = 8

#need to make sure that data takes form of num_imgs x s x s 
#also make sure that all frames come right after each other
class DataTokenizer():
    def __init__(self, ckpt_dir, out_file, vq_config, encoder, decoder, tokenizer_type):
        self.encoder = encoder
        self.decoder = decoder
        self.out_file = out_file
        self.tokenizer_type = tokenizer_type
        # Initialize empty output file
        open(self.out_file, 'wb').close()
    def tokenize(self, batch):
        encoded_tokens_list = []
        with torch.no_grad():  # Reduce memory usage during inference
            # Split into smaller batches to avoid OOM
            sub_batch_size = SUB_BATCH_SIZE  # Adjust this based on available GPU memory
            end = batch.shape[2] if self.tokenizer_type == "cosmos" else len(batch)
            for i in range(0, end - sub_batch_size + 1, sub_batch_size):
                if self.tokenizer_type == "magvit":
                    sub_batch = batch[i:i + sub_batch_size].to("cuda")
                    #preprocessing at the image level 
                    sub_batch =  sub_batch.float() / 127.5 - 1.0
                    quantized, _, encoded_tokens,_ = self.encoder.encode(sub_batch)
                    encoded_tokens = rearrange(encoded_tokens, "(b h w) -> b h w", b=sub_batch_size, h=COMPRESSED_HEIGHT, w=COMPRESSED_WIDTH)
                elif self.tokenizer_type == "cosmos":
                    sub_batch = batch[:,:,i:i + sub_batch_size].to("cuda").to(torch.bfloat16)
                    sub_batch = sub_batch.float() / 127.5 - 1.0
                    encoded_tokens, codes = self.encoder.encode(sub_batch)
                    
                
                encoded_tokens_list.append(encoded_tokens.cpu().numpy())
            #what to do when this becomes really large?
            
            
            # Cleanup intermediate tensors
            torch.cuda.empty_cache()  # If using GPU
            
            return encoded_tokens_list, encoded_tokens.shape[1]
                
    def reconstruct(self, token_path, ground_truth):
        """
        Reconstruct images from tokens and compute reconstruction metrics.
        Args:
            tokens: numpy array of shape (num_frames, height, width) containing token indices
        Returns:
            reconstructed_images: torch tensor of shape (num_frames, C, H, W) containing reconstructed images
            metrics: dict containing reconstruction metrics (PSNR, SSIM)
        """
        with torch.no_grad():
            # Load tokens from file if path provided
            if isinstance(token_path, str):
                # First read the shape (3 integers for 3D array)
                shape = np.fromfile(token_path, dtype=np.int32, count=4)
                # Then read the actual data and reshape it
                offset = 4 * np.dtype(np.int32).itemsize
                tokens = np.memmap(token_path, dtype=np.int32, mode="r", shape=shape, offset=offset)

                if self.tokenizer_type == "magvit":
                    tokens = tokens.reshape(-1, COMPRESSED_HEIGHT, COMPRESSED_WIDTH) # Reshape to original dimensions# Reshape to original dimensions
                        
            # Split into smaller batches
            sub_batch_size = SUB_BATCH_SIZE
            sub_token_size = SUB_TOKEN_SIZE
            end = tokens.shape[1] if self.tokenizer_type == "cosmos" else len(tokens)
            for i in range(0, end - sub_token_size + 1, sub_token_size):
                # Convert memmap to torch tensor first
                
                sub_tokens = torch.from_numpy(tokens[:,i:i + sub_token_size]).to("cuda")
                
                
                # Reshape tokens for decoder
                if self.tokenizer_type == "magvit":
                    quantized = self.encoder.quantize.decode(sub_tokens).reshape(sub_batch_size, 18, COMPRESSED_HEIGHT, COMPRESSED_WIDTH)
                    decoded = self.decoder.decode(quantized)
                    decoded = ((decoded + 1) * 127.5).clamp(0, 255)

                elif self.tokenizer_type == "cosmos":
                    decoded = self.decoder.decode(sub_tokens)
                    decoded = ((decoded + 1) * 127.5).clamp(0, 255)
                
                #might need to decode tokens first using quantizer; since decode expects quantized input not discrete indices
                # Decode tokens back to image space
                
                # Compute PSNR and SSIM metrics for this batch
                if ground_truth is not None:
                    # Get corresponding ground truth batch
                    gt_batch = ground_truth[:,:,int(i*sub_batch_size/sub_token_size): int((i+sub_token_size)*sub_batch_size/sub_token_size)].to(decoded.device)
                    # Compute PSNR
                    mse = F.mse_loss(decoded, gt_batch)
                    psnr = 20 * torch.log10(255.0 / torch.sqrt(mse))
                    
                    # Compute SSIM
                    # Convert to format expected by SSIM (B, C, H, W)
                    
                    ssim = structural_similarity_index_measure(
                        decoded, 
                        gt_batch,
                        data_range=255.0
                    )
                    
                    print(f"Batch {i//sub_batch_size} - PSNR: {psnr:.2f}, SSIM: {ssim:.4f}")
                
                

                    # Convert tensors to numpy arrays and correct format for saving
                    decoded_np = decoded.float().squeeze().cpu().permute(1, 0, 2, 3).numpy().astype(np.uint8)
                    gt_np = gt_batch.float().squeeze().cpu().permute(1, 0, 2, 3).numpy().astype(np.uint8)
                    
                    # Create output directory if it doesn't exist
                    os.makedirs("/data_vol/cosmos", exist_ok=True)
                    
                    # Save each image in the batch
                    for j in range(0,len(decoded_np),3):
                        frame_idx = i + j
                        # Save decoded image
                        decoded_img = Image.fromarray(decoded_np[j].transpose(1, 2, 0))
                        decoded_img.save(f"/data_vol/cosmos/cosmos_reconstructed_frame_{frame_idx:04d}.png")
                        
                        # Save ground truth image
                        gt_img = Image.fromarray(gt_np[j].transpose(1, 2, 0))
                        gt_img.save(f"/data_vol/cosmos/cosmos_ground_truth_frame_{frame_idx:04d}.png")
                        
                # Cleanup
                torch.cuda.empty_cache()
                    

class VideoDataset(Dataset):
    def __init__(self, video_dir, max_frames=16, num_videos= 1):
        self.video_dir = Path(video_dir)
        self.video_files = []
        counter = 0
        for video_path in self.video_dir.rglob('*.mp4'):
            if counter >= num_videos:
                break
            if "stereo.mp4" in str(video_path):
                continue
            self.video_files.append(str(video_path))
            counter += 1
        self.max_frames = max_frames
        # self.feature_extractor = VideoMAEFeatureExtractor.from_pretrained("MCG-NJU/videomae-base")
        
    def __len__(self):
        return len(self.video_files)
    
    #TODO: might be easier to just store frames that you want here
        # for example, crop videos to all be same length and then take every 'stride' frames window times
    def __getitem__(self, idx):
        video_path = str(self.video_files[idx])
        
        print("video_path", video_path)
        
        vframes, _, info = torchvision.io.read_video(
            video_path, 
            pts_unit='sec'
        )
        
        #only select window_size * stride frames
        # Select 240 evenly spaced frames
        #~ 4 secs for 60 fps 
        total_frames = len(vframes)
        if total_frames >= 240:
            # Calculate stride to get 240 evenly spaced frames
            stride = total_frames // 240
            vframes = vframes[::stride][:240]  # Take every stride-th frame, up to 240 frames
            

        # Get video fps from info
        fps = info['video_fps']
        print(f"Video FPS: {fps}")
        
        B, H, W, C = vframes.shape
        vframes = vframes.permute(0, 3, 1, 2)  # [B, C, H, W]
        vframes = torch.nn.functional.interpolate(
            vframes, 
            scale_factor=COMPRESSION_FACTOR,
            mode='bilinear',
            align_corners=False
        )
            
        return vframes


# Example usage:
def create_video_dataloader(video_dir, num_videos, batch_size=1, num_workers=2):
    dataset = VideoDataset(video_dir, num_videos=num_videos)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        prefetch_factor=2,  # Limit prefetching
        pin_memory=True,    # Only if using GPU
        persistent_workers=True,
        drop_last=True      # Avoid irregular batch sizes
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Tokenize external data")
    
    parser.add_argument(
        "--external_data_dir", type=str, default="/data/robotics/droid_raw/1.0.1",
        help="Directory containing tokenized data, should have a `video.bin`, `metadata.json` and `segment_ids.json`."
    )
    parser.add_argument(
        "--ckpt_dir", type=str, default="/1xgpt/data/magvit2.ckpt",
        help="Checkpoint directory for the tokenizer."
    )
    parser.add_argument(
        "--out_file", type=str, default="/data_vol/droid_tokens.bin",
        help="Output file for the tokenized data."
    )
    parser.add_argument(
        "--video_lengths_file", type=str, default="/data_vol/droid_video_lengths.bin",
        help="Output file for the video lengths."
    )
    parser.add_argument(
        "--num_videos", type=int, default=1,
        help="Number of videos to process."
    )
    args = parser.parse_args()

    return args
    
def main():
    args = parse_args()
    model_name = "Cosmos-Tokenizer-DV4x8x8"
    encoder = CausalVideoTokenizer(checkpoint_enc=f'/pretrained_ckpts/{model_name}/encoder.jit')
    decoder = CausalVideoTokenizer(checkpoint_dec=f'/pretrained_ckpts/{model_name}/decoder.jit')
    tokenizer = DataTokenizer(args.ckpt_dir, args.out_file, VQConfig(), encoder, decoder, "cosmos")
    
    dataloader = create_video_dataloader(args.external_data_dir, num_videos=args.num_videos)
    
    def print_memory_stats():
        cpu_memory = psutil.Process().memory_info().rss / 1024 / 1024
        gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0
        print(f"\rCPU Memory: {cpu_memory:.2f} MB | GPU Memory: {gpu_memory:.2f} MB", end="")
    
    encoded_tokens_list = []
    video_lengths = []
    for batch in tqdm(dataloader, desc="Processing videos"):
        batch = batch.transpose(1,2)
        
        enc_tokens, video_length = tokenizer.tokenize(batch)
        encoded_tokens_list.extend(enc_tokens)
        video_lengths.append(video_length)
        # tokenizer.reconstruct(tokenizer.out_file, batch)
        print_memory_stats()
    
    encoded_tokens = np.concatenate(encoded_tokens_list, axis=1)
    with open(args.out_file, 'ab') as f:  # 'ab' mode for binary append
        # Save shape information before the data
        shape = np.array(encoded_tokens.shape, dtype=np.int32)
        shape.tofile(f)
        # Save the actual data
        encoded_tokens.tofile(f)
    
    video_lengths = np.array(video_lengths, dtype=np.int32)
    with open(args.video_lengths_file, 'ab') as f:
        shape = np.array(video_lengths, dtype=np.int32)
        shape.tofile(f)
        video_lengths.tofile(f)
        
if __name__ == "__main__":
    main()

