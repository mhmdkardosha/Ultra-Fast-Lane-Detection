# Modal Deployment and Weights and Biases Logging Walkthrough

You are configured for cloud training on Modal with live logging in Weights and Biases (WandB).

## What Is In Place Now

1. WandB logging is enabled through TensorBoard scalar forwarding in `utils/dist_utils.py`.
2. Training runs remotely through `modal_app.py` with dataset and runs volumes mounted.
3. Modal config uses `configs/tulane_modal.py` paths:
	- dataset root: `/data/TuLaneConverted`
	- logs and checkpoints: `/runs/logs_tulane_modal`
4. Training now includes end-of-epoch validation and end-of-epoch summary output.
5. Checkpoint strategy now saves all of the following:
	- `epNNN.pth` every epoch
	- `latest.pth` every epoch (rolling latest)
	- `best.pth` whenever monitored validation metric improves
6. Resume now works from `epNNN.pth`, `latest.pth`, or `best.pth`.
7. WandB run continuity is supported on resume:
	- run id is stored in checkpoints as `wandb_run_id`
	- resumed training reattaches to the same WandB run automatically

## Execution Sequence

Use these steps to run training cleanly and safely.

### Step 1: Create WandB Secret in Modal

```bash
modal secret create my-wandb-secret WANDB_API_KEY="YOUR_ACTUAL_WANDB_API_KEY_HERE"
```

### Step 2: Upload and Prepare Dataset Volume

```bash
# 1. Compress locally (if needed)
tar -czvf TuLane.tar.gz dataset/TuLaneConverted

# 2. Upload archive to dataset volume
modal volume put ufld-dataset TuLane.tar.gz TuLane.tar.gz

# 3. Extract inside Modal volume
modal run modal_app.py::extract_dataset

# 4. Ensure list files contain image-mask pairs
modal run modal_app.py::format_dataset

# 5. Verify dataset structure
modal run modal_app.py::verify_dataset
```

### Step 3: Launch Training

```bash
modal run --detach modal_app.py
```

Detached mode returns your terminal immediately while training continues in the cloud.

## New Training Behavior (Important)

Each epoch now does:

1. train pass
2. validation pass
3. epoch summary print with train and val loss and metrics
4. checkpoint writes (`epNNN`, `latest`, and optionally `best`)

Monitored metric priority for best checkpoint is:

1. `laneiou`
2. `iou`
3. `top1`
4. fallback to `loss` (minimize)

## Resume Training

You can resume from any saved checkpoint path:

```bash
python train.py configs/tulane_modal.py --resume /runs/logs_tulane_modal/<run_dir>/latest.pth
```

You can also resume from:

1. `/runs/logs_tulane_modal/<run_dir>/best.pth`
2. `/runs/logs_tulane_modal/<run_dir>/ep012.pth` (example)

### WandB Same-Run Resume

For checkpoints created with the latest code, WandB run resume is automatic because the checkpoint stores `wandb_run_id`.

If you resume from an older checkpoint that does not include `wandb_run_id`, set it manually:

```bash
export WANDB_RUN_ID="your_existing_wandb_run_id"
python train.py configs/tulane_modal.py --resume /path/to/older_checkpoint.pth
```

## Download Logs and Checkpoints

To copy run artifacts from Modal volume to local machine:

```bash
modal volume get ufld-runs /logs_tulane_modal ./modal_downloads/
```
