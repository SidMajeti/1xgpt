import modal
import subprocess

app  = modal.App()

img = (
    modal.Image.from_registry(
        "ubuntu:20.04", add_python="3.10"
    )
    .env(
        {"DEBIAN_FRONTEND":"noninteractive"}
    )
    .apt_install(
        "wget",
        "git",
        "cmake",
        "build-essential",
    )
    .copy_local_file("/Users/sid/Documents/Sid/ResearchWork/1xgpt/requirements.txt", "/1xgpt/requirements.txt")
    .run_commands(
        "mkdir -p ~/miniconda3",
        "wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh",
        "bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3",
        "rm ~/miniconda3/miniconda.sh",
        ". ~/miniconda3/bin/activate",
        "pip install -r /1xgpt/requirements.txt",
    )
)


volume = modal.Volume.from_name("dataset", create_if_missing=True)

dataset = '/data'

cloud_bucket = modal.CloudBucketMount(
            bucket_name="gresearch",
            bucket_endpoint_url="https://storage.googleapis.com",
        )

@app.function(
    image = img,
    volumes={
        "/data": cloud_bucket
    },
    gpu = "a10g", timeout=3600,
    mounts=[modal.Mount.from_local_dir("/Users/sid/Documents/Sid/ResearchWork/1xgpt", remote_path="/1xgpt")]
)
#preprocess data and then run through tokenizer and store data
def run_magvit():
    command = ". ~/miniconda3/bin/activate && conda activate droid_policy_learning_env && pip install lightning && ls /data/robotics/droid_raw/1.0.1 \
    && python /1xgpt/data_preprocess.py"
    subprocess.run(command, shell=True)


@app.local_entrypoint()
def main():
    run_magvit.remote()