import modal
import subprocess
import genie

img = (
    modal.Image.debian_slim(python_version="3.10.12")
    .apt_install(
        "wget",
        "git"
    )
    .run_commands(
        "git clone https://github.com/SidMajeti/1xgpt",
        "cd 1xgpt && ls && bash ./build.sh",
        "wget https://huggingface.co/1x-technologies/GENIE_138M",
    )
)

volume = modal.Volume.from_name("dataset", create_if_missing=True)

dataset = '/data'

app = modal.App("run-generate")

output_dir='/data/genie_baseline_generated'

@app.function(image=img, volumes={dataset: volume}, gpu = "a10g")
def run():
    command = f"cd /1xgpt && . venv/bin/activate && python genie/generate.py --checkpoint_dir 1x-technologies/GENIE_138M \
                --output_dir {output_dir} --example_ind 0 --maskgit_steps 2 --temperature 0 && \
                python visualize.py --token_dir {output_dir}"
    subprocess.run(command, shell=True)
    volume.commit()
    
@app.function(image=img, volumes={dataset: volume}, gpu = "a10g", timeout=3600, mounts=[modal.Mount.from_local_dir("/Users/sid/Documents/Sid/ResearchWork/1xgpt", remote_path="/1xgpt")])
def evaluate():
    command = f"cd /1xgpt && . venv/bin/activate && python genie/evaluate.py --checkpoint_dir 1x-technologies/GENIE_138M"
    subprocess.run(command, shell=True)
    volume.commit()

    
@app.local_entrypoint()
def main():
    evaluate.remote()
