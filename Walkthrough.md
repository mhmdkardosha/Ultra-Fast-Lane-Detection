# Modal Deployment and W&B Logging Walkthrough

This project supports cloud training on Modal with metric visualization in Weights and Biases (W&B), and also supports local Python-only image testing.

## Modal: Current behavior

- Training runs on Modal through `modal_app.py` with mounted volumes at `/data` and `/runs`.
- Modal training config is `configs/tulane_modal.py`:
  - dataset root: `/data/TuLaneConverted`
  - logs/checkpoints root: `/runs/logs_tulane_modal`
  - batch metric logging interval: `batch_log_interval = 1` (every batch)
- Validation runs at the end of every epoch.
- Checkpoints saved during training:
  - `latest.pth` every epoch
  - `best.pth` when monitored validation metric improves
- W&B logging is epoch-oriented:
  - only `epoch/*` metrics are sent to W&B
  - W&B step is set to epoch index (`1, 2, 3, ...`)
- TensorBoard still receives all scalar logs (batch and epoch).

## Modal: One-time setup

### 1) Create W&B secret in Modal

```bash
modal secret create my-wandb-secret WANDB_API_KEY="YOUR_ACTUAL_WANDB_API_KEY_HERE"
```

### 2) Upload and prepare dataset volume

```bash
# Optional: compress locally
tar -czvf TuLane.tar.gz dataset/TuLaneConverted

# Upload archive to Modal volume
modal volume put ufld-dataset TuLane.tar.gz TuLane.tar.gz

# Extract and prepare
modal run modal_app.py::extract_dataset
modal run modal_app.py::format_dataset
modal run modal_app.py::verify_dataset
```

## Modal: Start training

```bash
modal run --detach modal_app.py
```

Detached mode returns your terminal immediately while training continues remotely.

## Modal: Monitor training

```bash
# List active/recent apps
modal app list

# Stream logs (by app name)
modal app logs ufld-training -f
```

## Modal: Resume training

Set `resume` in `configs/tulane_modal.py` to a checkpoint path under `/runs/logs_tulane_modal/<run_dir>/`, then launch training again.

Example checkpoint choices:

1. `/runs/logs_tulane_modal/<run_dir>/latest.pth`
2. `/runs/logs_tulane_modal/<run_dir>/best.pth`

W&B run continuity is automatic for checkpoints that contain `wandb_run_id`.

## Modal: Evaluate the trained model so far

1. Find your run folder:

```bash
modal volume ls ufld-runs /logs_tulane_modal
```

1. Download a checkpoint locally:

```bash
modal volume get ufld-runs /logs_tulane_modal/<run_dir>/latest.pth ./latest_modal.pth
```

1. Run evaluation from repository root:

```bash
python3 test.py configs/tulane_modal.py \
  --data_root ../dataset/TuLaneConverted \
  --test_model ./latest_modal.pth \
  --test_work_dir ./tmp_eval
```

Use `best.pth` instead of `latest.pth` if you want the best validation checkpoint.

## Modal: Download logs and checkpoints

```bash
modal volume get ufld-runs /logs_tulane_modal ./modal_downloads/
```

## Local Python testing examples

### Comparison screenshot testing

```bash
python scripts/infer_images_to_video.py configs/tulane.py \
  --model path_to_your_model.pth \
  --input_dir ../dataset/TuLaneConverted/images/target_test \
  --gt_mask_dir ../dataset/TuLaneConverted/lane_masks/target_test \
  --render_style comparison \
  --output_mode images \
  --output_dir outputs/target_test_comparison \
  --device cpu \
  --dataset CULane
```

### Overlay-only testing

```bash
python scripts/infer_images_to_video.py configs/tulane.py \
  --model path_to_your_model.pth \
  --input_dir ../dataset/TuLaneConverted/images/target_test \
  --render_style overlay \
  --output_mode images \
  --output_dir outputs/target_test_overlay \
  --device cpu \
  --dataset CULane
```
