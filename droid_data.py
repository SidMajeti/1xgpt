import modal
import subprocess

cosmos_img = (
    modal.Image.debian_slim(python_version="3.10.12")
    .apt_install(
        "wget",
        "git",
        "ffmpeg",
    )
    .copy_local_file("/Users/sid/Documents/Sid/ResearchWork/1xgpt/download_cosmos.py", "./download_cosmos.py")
    .copy_local_file("/Users/sid/Documents/Sid/ResearchWork/1xgpt/cosmos_build.sh", "./cosmos_build.sh")
    .run_commands(
        "bash ./cosmos_build.sh",
        secrets=[modal.Secret.from_name("huggingface-secret")],
    )
)

cloud_bucket = modal.CloudBucketMount(
            bucket_name="gresearch",
            bucket_endpoint_url="https://storage.googleapis.com",
        )

data_vol = '/data_vol'

app = modal.App("preprocess-droid-data")

volume = modal.Volume.from_name("dataset", create_if_missing=True)

@app.function(image=cosmos_img, volumes={'/data': cloud_bucket, data_vol: volume}, gpu = "a10g", timeout=3600,
              secrets=[modal.Secret.from_name("huggingface-secret")],
              mounts=[modal.Mount.from_local_dir("/Users/sid/Documents/Sid/ResearchWork/1xgpt", remote_path="/1xgpt")])
def get_data():
    command = f"cd .. && pwd && echo 'Current directory:' && pwd && . cosmos/bin/activate && cd /1xgpt && echo 'New directory:' && pwd && python data_preprocess.py \
    --external_data_dir /data/robotics/droid_raw/1.0.1 --out_file /data_vol/droid_tokens.bin --num_videos 1000"
    subprocess.run(command, shell=True)
    
@app.local_entrypoint()
def main():
    get_data.remote()
