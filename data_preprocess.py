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
from magvit2.models import lfqgan
import argparse
from torch.utils.data import Dataset, DataLoader
import torchvision
from transformers import VideoMAEFeatureExtractor
from magvit2.config import VQConfig
from tqdm import tqdm
import psutil
from torchmetrics.functional import structural_similarity_index_measure
import torch.nn.functional as F  # Also needed for F.mse_loss
from PIL import Image  # Add this with other imports at the top

COMPRESSED_WIDTH = 16
COMPRESSED_HEIGHT = 9

#need to make sure that data takes form of num_imgs x s x s 
#also make sure that all frames come right after each other
class DataTokenizer():
    def __init__(self, ckpt_dir, out_file, vq_config):
        self.tokenizer = lfqgan.VQModel(vq_config).to("cuda")
        self.tokenizer.init_from_ckpt(ckpt_dir)
        self.out_file = out_file
        # Initialize empty output file
        open(self.out_file, 'wb').close()
    def tokenize(self, batch):
        with torch.no_grad():  # Reduce memory usage during inference
            # Split into smaller batches to avoid OOM
            sub_batch_size = 4  # Adjust this based on available GPU memory
            for i in range(0, len(batch), sub_batch_size):
                sub_batch = batch[i:i + sub_batch_size].to("cuda")
                #preprocessing at the image level 
                sub_batch =  sub_batch.float() / 127.5 - 1.0
                quantized, _, encoded_tokens,_ = self.tokenizer.encode(sub_batch)
                encoded_tokens = rearrange(encoded_tokens, "(b h w) -> b h w", b=sub_batch_size, h=COMPRESSED_HEIGHT, w=COMPRESSED_WIDTH)
            
            encoded_tokens = encoded_tokens.cpu().numpy()
            with open(self.out_file, 'ab') as f:  # 'ab' mode for binary append
                encoded_tokens.tofile(f)
            # Cleanup intermediate tensors
            torch.cuda.empty_cache()  # If using GPU
            
            return encoded_tokens
                
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
                tokens = np.fromfile(token_path, dtype=np.int32)
                tokens = tokens.reshape(-1, COMPRESSED_HEIGHT, COMPRESSED_WIDTH) # Reshape to original dimensions
            
            # Convert to tensor and move to GPU
            tokens = torch.from_numpy(tokens).to("cuda")
            
            # Split into smaller batches
            sub_batch_size = 4
            
            for i in range(0, len(tokens), sub_batch_size):
                sub_tokens = tokens[i:i + sub_batch_size]
                
                # Reshape tokens for decoder
                quantized = self.tokenizer.quantize.decode(sub_tokens).reshape(sub_batch_size, 18, COMPRESSED_HEIGHT, COMPRESSED_WIDTH)
                
                #might need to decode tokens first using quantizer; since decode expects quantized input not discrete indices
                # Decode tokens back to image space
                decoded = self.tokenizer.decode(quantized)
                
                
                # Denormalize from [-1,1] to [0,255]
                decoded = ((decoded + 1) * 127.5).clamp(0, 255)
                
                
                # Compute PSNR and SSIM metrics for this batch
                if ground_truth is not None:
                    # Get corresponding ground truth batch
                    gt_batch = ground_truth[i:i + sub_batch_size].to(decoded.device)
                    
                    
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
                
                
                # Save decoded and ground truth images
                if ground_truth is not None:
                    gt_batch = ground_truth[i:i + sub_batch_size].to(decoded.device)
                    
                    # Convert tensors to numpy arrays and correct format for saving
                    decoded_np = decoded.cpu().permute(0, 2, 3, 1).numpy().astype(np.uint8)
                    gt_np = gt_batch.cpu().permute(0, 2, 3, 1).numpy().astype(np.uint8)
                    
                    # Save each image in the batch
                    for j in range(len(decoded_np)):
                        frame_idx = i + j
                        
                        # Save decoded image
                        decoded_img = Image.fromarray(decoded_np[j])
                        decoded_img.save(f"/data_vol/reconstructed_frame_{frame_idx:04d}.png")
                        
                        # Save ground truth image
                        gt_img = Image.fromarray(gt_np[j])
                        gt_img.save(f"/data_vol/ground_truth_frame_{frame_idx:04d}.png")
                        
                # Cleanup
                torch.cuda.empty_cache()
                    

class VideoDataset(Dataset):
    def __init__(self, video_dir, tokenizer, max_frames=16, num_videos= 5):
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
        
        # Only read the frames you need
        start_sec = 0
        end_sec = 8  # Assuming 30fps, this will give you ~240 frames
        vframes, _, info = torchvision.io.read_video(
            video_path, 
            start_pts=start_sec,
            end_pts=end_sec,
            pts_unit='sec'
        )
        #only select window_size * stride frames
        vframes = vframes[:240].clone()

        # Convert to float and normalize to [-1, 1]        
        # Get video fps from info
        fps = info['video_fps']
        print(f"Video FPS: {fps}")
        
        B, H, W, C = vframes.shape
        vframes = vframes.permute(0, 3, 1, 2)  # [B, C, H, W]
        vframes = torch.nn.functional.interpolate(
            vframes, 
            scale_factor=0.20,
            mode='bilinear',
            align_corners=False
        )
        
        #any other preprocessing?
        
        #240, 3, 720, 1280
        
        # Get the pixel values        
        # Tokenize using the provided tokenizer        
        return vframes


# Example usage:
def create_video_dataloader(video_dir, tokenizer, batch_size=1, num_workers=2):
    dataset = VideoDataset(video_dir, tokenizer)
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
    args = parser.parse_args()

    return args
    
def main():
    args = parse_args()
    tokenizer = DataTokenizer(args.ckpt_dir, args.out_file, VQConfig())
    
    dataloader = create_video_dataloader(args.external_data_dir, tokenizer)
    
    def print_memory_stats():
        cpu_memory = psutil.Process().memory_info().rss / 1024 / 1024
        gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0
        print(f"\rCPU Memory: {cpu_memory:.2f} MB | GPU Memory: {gpu_memory:.2f} MB", end="")
    
    for batch in tqdm(dataloader, desc="Processing videos"):
        batch = batch.squeeze(0)
        print("input batch.shape", batch.shape)
        tokenizer.tokenize(batch)
        tokenizer.reconstruct(tokenizer.out_file, batch)
        print_memory_stats()
        
if __name__ == "__main__":
    main()

