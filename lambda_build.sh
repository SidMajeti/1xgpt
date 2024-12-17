#! /bin/bash

git clone https://github.com/1x-technologies/1xgpt
cd 1xgpt && ls && bash ./build.sh
wget https://huggingface.co/1x-technologies/GENIE_138M

# Example of copying files to a remote server using scp
# Replace these variables with your actual values
REMOTE_USER="ubuntu"
REMOTE_HOST="159.54.171.254" 
REMOTE_PATH="/home/ubuntu/sidm23"
LOCAL_FILES="/Users/sid/Documents/Sid/ResearchWork/1xgpt/*"

# Copy files using scp
# -r flag copies directories recursively 
scp -r $LOCAL_FILES $REMOTE_USER@$REMOTE_HOST:$REMOTE_PATH

# Alternative using rsync which is more efficient for large transfers
# rsync -avz $LOCAL_FILES $REMOTE_USER@$REMOTE_HOST:$REMOTE_PATH



