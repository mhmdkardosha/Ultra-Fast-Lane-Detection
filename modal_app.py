import modal
import os
import subprocess

app = modal.App("ufld-training")

# Define our image with exact dependencies needed by UFLD
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("libgl1-mesa-glx", "libglib2.0-0") # OpenCV dependencies
    .pip_install(
        "torch", 
        "torchvision",
        "opencv-python",
        "tqdm",
        "tensorboard",
        "addict",
        "scikit-learn",
        "pathspec",
        "scipy",
        "wandb",
    )
    .add_local_dir(".", remote_path="/workspace")
)

DATASET_DIR = "/data"

# Reference our specific volumes
dataset_vol = modal.Volume.from_name("ufld-dataset", create_if_missing=True)
runs_volume = modal.Volume.from_name("ufld-runs", create_if_missing=True)


@app.function(
    image=image,
    volumes={DATASET_DIR: dataset_vol},
    timeout=7200,  # 2 hours for large extractions
)
def extract_dataset(tar_name: str = "TuLane.tar.gz"):
    """Extract a tar.gz dataset archive inside a Modal container.

    The archive must already be uploaded to the volume. Run:
        modal volume put ufld-dataset ./dataset/TuLane.tar.gz TuLane.tar.gz
        modal run modal_app.py::extract_dataset

    The archive is extracted into /data/ on the volume, producing
    /data/TuLaneConverted/ (or whatever the archive root folder is).
    """
    import tarfile
    import time

    archive_path = os.path.join(DATASET_DIR, tar_name)

    if not os.path.exists(archive_path):
        print(f"❌ Archive not found at {archive_path}")
        print()
        print("Upload it first:")
        print(f"  modal volume put ufld-dataset ./dataset/{tar_name} {tar_name}")
        return False

    size_gb = os.path.getsize(archive_path) / (1024**3)
    print(f"📦 Found {archive_path} ({size_gb:.2f} GB)")
    print(f"📂 Extracting to {DATASET_DIR}/ ...")

    start = time.time()
    with tarfile.open(archive_path, "r:gz") as tar:
        members = tar.getmembers()
        print(f"   Archive contains {len(members)} entries")
        tar.extractall(path=DATASET_DIR)
    elapsed = time.time() - start

    print(f"✅ Extraction complete in {elapsed:.1f}s")

    # Show what was extracted
    entries = sorted(os.listdir(DATASET_DIR))
    print(f"   Volume contents: {entries}")

    # Find TuLaneConverted — it may be nested inside the archive structure
    dataset_path = os.path.join(DATASET_DIR, "TuLaneConverted")
    if not os.path.exists(dataset_path):
        import shutil

        print("   🔍 Searching for TuLaneConverted in extracted tree ...")
        found = None
        for root, dirs, _files in os.walk(DATASET_DIR):
            if "TuLaneConverted" in dirs:
                found = os.path.join(root, "TuLaneConverted")
                break
        if found:
            print(f"   Found at: {found}")
            print(f"   📦 Moving to {dataset_path} ...")
            shutil.move(found, dataset_path)

            # Clean up the leftover empty nested directories
            top_level_leftover = os.path.join(
                DATASET_DIR,
                entries[0] if entries else "",
            )
            for entry in entries:
                entry_path = os.path.join(DATASET_DIR, entry)
                if os.path.isdir(entry_path) and entry != "TuLaneConverted":
                    print(f"   🗑️  Removing leftover directory: {entry}")
                    shutil.rmtree(entry_path)
        else:
            print("   ❌ TuLaneConverted not found anywhere in the archive")
            print(f"   Top-level entries: {entries}")

    if os.path.exists(dataset_path):
        sub_entries = sorted(os.listdir(dataset_path))
        print(f"   TuLaneConverted/ contents: {sub_entries}")

    # Commit moved files to the volume
    print("💾 Committing to volume ...")
    dataset_vol.commit()
    print("✅ Done! Dataset is ready for training.")

    return True


@app.function(
    image=image,
    volumes={DATASET_DIR: dataset_vol},
    timeout=600,
)
def flatten_dataset():
    """Move TuLaneConverted from nested archive folders to /dataset/TuLaneConverted.

    Use this when the archive is already extracted but TuLaneConverted is buried
    in nested directories (e.g. /dataset/teamspace/studios/.../TuLaneConverted).

    Run:  modal run modal_app.py::flatten_dataset
    """
    import shutil

    dataset_path = os.path.join(DATASET_DIR, "TuLaneConverted")

    if os.path.exists(dataset_path):
        sub_entries = sorted(os.listdir(dataset_path))
        print(f"✅ TuLaneConverted already at {dataset_path}")
        print(f"   Contents: {sub_entries}")
        return True

    print("🔍 Searching for TuLaneConverted ...")
    entries = sorted(os.listdir(DATASET_DIR))
    print(f"   Volume contents: {entries}")

    found = None
    for root, dirs, _files in os.walk(DATASET_DIR):
        if "TuLaneConverted" in dirs:
            found = os.path.join(root, "TuLaneConverted")
            break

    if not found:
        print("❌ TuLaneConverted not found anywhere in the volume")
        return False

    print(f"   Found at: {found}")
    print(f"   📦 Moving to {dataset_path} ...")
    shutil.move(found, dataset_path)

    # Clean up leftover empty nested directories
    for entry in entries:
        entry_path = os.path.join(DATASET_DIR, entry)
        if os.path.isdir(entry_path) and entry != "TuLaneConverted":
            print(f"   🗑️  Removing leftover: {entry}")
            shutil.rmtree(entry_path)

    sub_entries = sorted(os.listdir(dataset_path))
    print(f"   TuLaneConverted/ contents: {sub_entries}")

    print("💾 Committing to volume ...")
    dataset_vol.commit()
    print("✅ Done!")

    return True


@app.function(
    image=image,
    volumes={DATASET_DIR: dataset_vol},
    timeout=300,
)
def verify_dataset():
    """Check that the dataset Volume is populated correctly.

    Upload & extract your dataset first:
        modal volume put ufld-dataset ./dataset/TuLane.tar.gz TuLane.tar.gz
        modal run modal_app.py::extract_dataset
    """
    dataset_path = os.path.join(DATASET_DIR, "TuLaneConverted")

    if not os.path.exists(dataset_path):
        print("❌ Dataset not found at /data/TuLaneConverted")
        print()
        print("Upload & extract it:")
        print(
            "  modal volume put ufld-dataset ./dataset/TuLane.tar.gz TuLane.tar.gz"
        )
        print("  modal run modal_app.py::extract_dataset")
        return False

    entries = sorted(os.listdir(dataset_path))
    print(f"✅ Dataset found at {dataset_path}")
    print(f"   Contents: {entries}")

    # Check for list files
    list_dir = os.path.join(dataset_path, "list")
    if os.path.exists(list_dir):
        list_files = os.listdir(list_dir)
        print(f"   List files: {list_files}")
        for lf in list_files:
            path = os.path.join(list_dir, lf)
            with open(path) as f:
                lines = f.readlines()
            print(f"   {lf}: {len(lines)} samples")
    else:
        print("   ⚠️  No list/ directory found")

    return True

@app.function(
    image=image,
    gpu="T4", # Default, could use T4 or A100 based on preference
    timeout=86400, # 24 hours
    volumes={
        DATASET_DIR: dataset_vol,
        "/runs": runs_volume
    },
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
