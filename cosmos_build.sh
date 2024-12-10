#!/bin/bash

# Create and activate virtual environment
python3 -m venv cosmos
. cosmos/bin/activate

# Install dependencies and clone repo
pip install av
git clone https://github.com/NVIDIA/Cosmos-Tokenizer.git

# Install Cosmos and PyTorch
cd Cosmos-Tokenizer
pip install -e .
pip install torch

pip install lightning && pip install torchvision && pip install psutil && ls && cd .. && ls && python download_cosmos.py