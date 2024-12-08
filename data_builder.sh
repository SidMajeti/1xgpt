#!/usr/bin/bash

. ~/miniconda3/bin/activate
conda create --name droid_policy_learning_env python=3.10
conda activate droid_policy_learning_env
git clone https://github.com/octo-models/octo.git
cd octo 
git checkout 85b83fc19657ab407a7f56558a5384ae56fe453b
pip install -e . && pip install -r requirements.txt && cd ..
git clone https://github.com/droid-dataset/droid_policy_learning.git
cd droid_policy_learning && pip install -e .
pip install --upgrade "jax[cuda11_pip]==0.4.20" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install torch==2.1.0
