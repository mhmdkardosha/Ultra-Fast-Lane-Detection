import modal
import os
import subprocess

app = modal.App("ufld-training")

# Define our image with exact dependencies needed by UFLD
# UFLD doesn't have a strict requirements.txt setup that works perfectly in all docker envs out of the box so we install manually
image = (
    modal.Image.debian_slim(python_version="3.9")
    .apt_install("libgl1-mesa-glx", "libglib2.0-0") # OpenCV dependencies
    .pip_install(
        "torch==1.13.1",
        "torchvision==0.14.1",
        "opencv-python",
        "scipy",
        "tqdm",
        "tensorboard",
        "wandb",
        "pathspec"
    )
)

# Reference our specific volumes
# The ufld-dataset volume holds the exact TuLaneConverted structure mapping to CULane format
dataset_volume = modal.Volume.from_name("ufld-dataset", create_if_missing=True)
runs_volume = modal.Volume.from_name("ufld-runs", create_if_missing=True)

@app.function(
    image=image,
    gpu="T4", # Default, could use T4 or A100 based on preference
    timeout=86400, # 24 hours
    volumes={
        "/data": dataset_volume,
        "/runs": runs_volume
    },
    mounts=[modal.Mount.from_local_dir(".", remote_path="/workspace")],
    secrets=[modal.Secret.from_name("my-wandb-secret")] # Ensure you create this secret in Modal Dashboard!
)
def train():
    os.chdir("/workspace")
    
    print("Starting UFLD Framework on Modal with W&B Logging...")

    # Exclusively call the identical train execution with the Modal specific config
    cmd = ["python3", "train.py", "configs/tulane_modal.py"]
    
    # We yield the log output immediately using Popen
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    
    # Continuously print output from the subprocess so we can monitor on Modal dashboard
    for line in iter(process.stdout.readline, ""):
        print(line, end="")
        
    process.stdout.close()
    return_code = process.wait()
    
    if return_code != 0:
        raise RuntimeError(f"Training failed with return code {return_code}")
    print("Training Completed Successfully!")

@app.local_entrypoint()
def main():
    print("Deploying UFLD to Modal Cloud Network in DETACHED MODE...")
    train.spawn()
    print("Training job spawned successfully! It is now running in the background on Modal.")
    print("You can close this terminal. Monitor progress on your WandB dashboard or the Modal web interface.")

