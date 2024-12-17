import modal
import subprocess
import genie

img = (
    modal.Image.debian_slim(python_version="3.10.12")
    .apt_install(
        "wget",
        "git",
        "ffmpeg",
    )
    .run_commands(
        "git clone https://github.com/1x-technologies/1xgpt",
        "cd 1xgpt && ls && bash ./build.sh",
        "wget https://huggingface.co/1x-technologies/GENIE_138M",
    )
)

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
    
volume = modal.Volume.from_name("dataset", create_if_missing=True)

data_vol = '/data_vol'

app = modal.App("run-generate")

output_dir='/data_vol/genie_baseline_generated'

cloud_bucket = modal.CloudBucketMount(
            bucket_name="gresearch",
            bucket_endpoint_url="https://storage.googleapis.com",
        )

@app.function(image=img, volumes={data_vol: volume}, gpu = "a10g")
def generate():
    command = f"cd /1xgpt && . venv/bin/activate && python genie/generate.py --checkpoint_dir 1x-technologies/GENIE_138M \
                --output_dir {output_dir} --example_ind 10 --maskgit_steps 2 --temperature 0.7 && \
                python visualize.py --token_dir {output_dir}"
    subprocess.run(command, shell=True)
    volume.commit()
    
@app.function(image=img, volumes={data_vol: volume}, gpu = "a10g", timeout=3600, mounts=[modal.Mount.from_local_dir("/Users/sid/Documents/Sid/ResearchWork/1xgpt", remote_path="/1xgpt")])
def evaluate():
    command = f"cd /1xgpt && . venv/bin/activate && python genie/evaluate.py --checkpoint_dir 1x-technologies/GENIE_138M"
    subprocess.run(command, shell=True)
    volume.commit()

@app.function(image=img, volumes={data_vol: volume}, gpu = "h100", timeout=10800, mounts=[modal.Mount.from_local_dir("/Users/sid/Documents/Sid/ResearchWork/1xgpt", remote_path="/1xgpt")])
def train():
    command = f"cd /1xgpt && . venv/bin/activate && python train.py --genie_config genie/configs/magvit_n32_h8_d256.json --output_dir /data_vol/genie_model --max_eval_steps 10 \
                --num_warmup_steps 1000 --learning_rate 3e-4 --per_device_train_batch_size 16 --per_device_eval_batch_size 16 --resume_from_checkpoint /data_vol/genie_model/step_27000"
    subprocess.run(command, shell=True)
    volume.commit()
    
# @app.function(image=cosmos_img, volumes={'/data': cloud_bucket, data_vol: volume}, gpu = "a10g", timeout=3600,
#               secrets=[modal.Secret.from_name("huggingface-secret")],
#               mounts=[modal.Mount.from_local_dir("/Users/sid/Documents/Sid/ResearchWork/1xgpt", remote_path="/1xgpt")])
# def get_data():
#     command = f"cd .. && . cosmos/bin/activate && cd /1xgpt && python data_preprocess.py \
#     --external_data_dir /data/robotics/droid_raw/1.0.1 --out_file /data_vol/droid_tokens.bin"
#     subprocess.run(command, shell=True)
    
@app.local_entrypoint()
def main():
    train.remote()
    # generate.remote()
    # evaluate.remote()
