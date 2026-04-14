# Install

## 1) Clone

```bash
git clone https://github.com/cfzd/Ultra-Fast-Lane-Detection
cd Ultra-Fast-Lane-Detection
```

## 2) Create environment

```bash
conda create -n lane-det python=3.10 -y
conda activate lane-det
```

## 3) Install dependencies

```bash
pip install -r requirements.txt
```

## 4) Prepare your data

Use your converted TuLane dataset in this structure:

- `../dataset/TuLaneConverted/images/<split>`
- `../dataset/TuLaneConverted/lane_masks/<split>`

Example split names: `target_test`, `target_train`, `target_val`, `train`, `val`.

## 5) Python-only testing (no C++ build)

### Comparison screenshot output

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

## 6) Optional: Modal cloud training and W&B

If you are training on Modal, use the cloud workflow in [Walkthrough.md](./Walkthrough.md).

Quick start:

```bash
modal run --detach modal_app.py
```

### Overlay-only output

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
