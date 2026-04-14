import argparse
import glob
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from tqdm import tqdm

# Make local project imports work regardless of where the script is launched from.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.constant import culane_row_anchor, tusimple_row_anchor
from model.model import parsingNet
from utils.config import Config


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run lane detection on an image folder and export overlay images or video."
    )
    parser.add_argument(
        "config", type=str, help="Path to a config file, e.g. configs/tulane.py"
    )
    parser.add_argument(
        "--model", type=str, required=True, help="Path to model checkpoint (.pth)"
    )
    parser.add_argument(
        "--input_dir", type=str, required=True, help="Folder containing input images"
    )
    parser.add_argument(
        "--output_video",
        type=str,
        default=None,
        help="Output video path, e.g. lanes.mp4 (required for video/both mode)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for picture results (required for images/both mode)",
    )
    parser.add_argument(
        "--output_mode",
        type=str,
        default="images",
        choices=["images", "video", "both"],
        help="Export mode: images, video, or both",
    )
    parser.add_argument(
        "--render_style",
        type=str,
        default="comparison",
        choices=["overlay", "comparison"],
        help="overlay: only prediction overlay, comparison: 4-panel screenshot style",
    )
    parser.add_argument(
        "--gt_mask_dir",
        type=str,
        default=None,
        help="Folder with ground-truth masks matching image basenames",
    )
    parser.add_argument(
        "--panel_width",
        type=int,
        default=512,
        help="Panel width for comparison style output",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        choices=["CULane", "Tusimple"],
        help="Dataset protocol for row anchors; defaults to value from config",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Inference device. Use cpu to avoid CUDA.",
    )
    parser.add_argument("--fps", type=float, default=20.0, help="Output video FPS")
    parser.add_argument(
        "--image_glob",
        type=str,
        default="*.jpg",
        help="Glob pattern for images inside input_dir, e.g. '*.png'",
    )
    parser.add_argument(
        "--num_lanes", type=int, default=None, help="Override number of lanes"
    )
    parser.add_argument(
        "--lane_style",
        type=str,
        default="lines",
        choices=["points", "lines", "both"],
        help="How to render predictions: points, lines, or both",
    )
    parser.add_argument(
        "--thickness",
        type=int,
        default=4,
        help="Drawing thickness (line thickness / point radius)",
    )
    return parser.parse_args()


def choose_device(device_arg):
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_protocol_params(dataset_name):
    if dataset_name == "CULane":
        return 18, culane_row_anchor
    if dataset_name == "Tusimple":
        return 56, tusimple_row_anchor
    raise ValueError("dataset must be one of: CULane, Tusimple")


def print_model_info(cfg, ckpt, net, num_lanes, device):
    """Print a human-readable summary of the loaded model and checkpoint metadata."""
    sep = "=" * 56
    print(sep)
    print("  Model loaded")
    print(sep)
    print(f"  Checkpoint  : {cfg.test_model}")
    print(f"  Backbone    : ResNet-{cfg.backbone}")
    print(f"  Griding num : {cfg.griding_num}")
    print(f"  Num lanes   : {num_lanes}")
    print(f"  Device      : {device}")

    total_params = sum(p.numel() for p in net.parameters())
    trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"  Parameters  : {total_params:,}  ({trainable_params:,} trainable)")

    if isinstance(ckpt, dict):
        epoch = ckpt.get("epoch")
        if epoch is not None:
            print(f"  Saved epoch : {int(epoch) + 1}")

        best_name = ckpt.get("best_metric_name")
        best_value = ckpt.get("best_metric_value")
        best_epoch = ckpt.get("best_epoch")
        mon_name = ckpt.get("monitor_name")
        mon_value = ckpt.get("monitor_value")

        if best_name is not None and best_value is not None:
            epoch_str = (
                f"  (epoch {int(best_epoch) + 1})" if best_epoch is not None else ""
            )
            print(f"  Best score  : {best_name} = {float(best_value):.4f}{epoch_str}")
        elif mon_name is not None and mon_value is not None:
            print(f"  Score       : {mon_name} = {float(mon_value):.4f}")

        wandb_run_id = ckpt.get("wandb_run_id")
        if wandb_run_id:
            print(f"  W&B run     : {wandb_run_id}")

    print(sep)


def build_model(cfg, cls_num_per_lane, device, num_lanes):
    net = parsingNet(
        pretrained=False,
        backbone=cfg.backbone,
        cls_dim=(cfg.griding_num + 1, cls_num_per_lane, num_lanes),
        use_aux=False,
    ).to(device)

    ckpt = torch.load(cfg.test_model, map_location="cpu")
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt

    compatible_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            compatible_state_dict[key[7:]] = value
        else:
            compatible_state_dict[key] = value

    net.load_state_dict(compatible_state_dict, strict=False)
    net.eval()

    print_model_info(cfg, ckpt, net, num_lanes, device)
    return net


def decode_lanes(out_tensor, griding_num):
    out = out_tensor[0].detach().cpu().numpy()
    out = out[:, ::-1, :]

    logits = out[:-1, :, :]
    logits = logits - np.max(logits, axis=0, keepdims=True)
    exp_logits = np.exp(logits)
    prob = exp_logits / np.sum(exp_logits, axis=0, keepdims=True)

    idx = np.arange(griding_num) + 1
    idx = idx.reshape(-1, 1, 1)
    loc = np.sum(prob * idx, axis=0)

    max_idx = np.argmax(out, axis=0)
    loc[max_idx == griding_num] = 0
    return loc


def draw_lanes(
    image,
    lane_loc,
    row_anchor,
    cls_num_per_lane,
    griding_num,
    thickness,
    lane_style,
    color=(0, 255, 0),
):
    h, w = image.shape[:2]
    col_sample = np.linspace(0, 800 - 1, griding_num)
    col_sample_w = col_sample[1] - col_sample[0]
    pred_mask = np.zeros((h, w), dtype=np.uint8)

    for lane_i in range(lane_loc.shape[1]):
        if np.sum(lane_loc[:, lane_i] != 0) <= 2:
            continue
        lane_points = []
        for row_i in range(lane_loc.shape[0]):
            if lane_loc[row_i, lane_i] <= 0:
                continue
            x = int(lane_loc[row_i, lane_i] * col_sample_w * w / 800) - 1
            y = int(h * (row_anchor[cls_num_per_lane - 1 - row_i] / 288)) - 1
            if x < 0 or x >= w or y < 0 or y >= h:
                continue
            lane_points.append((x, y))

        if len(lane_points) <= 2:
            continue

        if lane_style in ("points", "both"):
            for x, y in lane_points:
                cv2.circle(image, (x, y), thickness, color, -1)
                cv2.circle(pred_mask, (x, y), thickness, 255, -1)

        if lane_style in ("lines", "both"):
            poly = np.array(lane_points, dtype=np.int32)
            cv2.polylines(
                image, [poly], isClosed=False, color=color, thickness=thickness
            )
            cv2.polylines(
                pred_mask, [poly], isClosed=False, color=255, thickness=thickness
            )

    return image, pred_mask


def infer_gt_mask_dir(input_dir):
    norm = os.path.normpath(input_dir)
    marker = f"{os.sep}images{os.sep}"
    if marker in norm:
        candidate = norm.replace(marker, f"{os.sep}lane_masks{os.sep}", 1)
        if os.path.isdir(candidate):
            return candidate

    parent = os.path.dirname(norm)
    grandparent = os.path.dirname(parent)
    candidate = os.path.join(grandparent, "lane_masks", os.path.basename(norm))
    if os.path.isdir(candidate):
        return candidate
    return None


def find_gt_mask_path(image_path, gt_mask_dir):
    if gt_mask_dir is None:
        return None

    stem = os.path.splitext(os.path.basename(image_path))[0]
    for ext in [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]:
        path = os.path.join(gt_mask_dir, stem + ext)
        if os.path.isfile(path):
            return path
    return None


def load_binary_mask(mask_path, target_hw):
    if mask_path is None:
        return np.zeros(target_hw, dtype=np.uint8)

    mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
    if mask is None:
        return np.zeros(target_hw, dtype=np.uint8)

    if len(mask.shape) == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    h, w = target_hw
    if mask.shape[:2] != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

    bin_mask = (mask > 0).astype(np.uint8) * 255
    return bin_mask


def make_gt_overlay(image, gt_mask, color=(0, 255, 0), alpha=0.45):
    overlay = image.copy()
    idx = gt_mask > 0
    if np.any(idx):
        color_arr = np.array(color, dtype=np.float32)
        overlay[idx] = (
            (1.0 - alpha) * overlay[idx].astype(np.float32) + alpha * color_arr
        ).astype(np.uint8)
    return overlay


def lane_percent(mask):
    return 100.0 * float(np.count_nonzero(mask)) / float(mask.size)


def panel_with_title(image, title, panel_w, color=(255, 255, 255), bar_h=44):
    h, w = image.shape[:2]
    panel_h = int(round(h * float(panel_w) / float(w)))
    panel = cv2.resize(image, (panel_w, panel_h), interpolation=cv2.INTER_LINEAR)

    bar = np.zeros((bar_h, panel_w, 3), dtype=np.uint8)
    cv2.putText(
        bar,
        title,
        (10, 29),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        color,
        2,
        cv2.LINE_AA,
    )
    return np.vstack([bar, panel])


def make_comparison_canvas(
    original, gt_mask, gt_overlay, pred_panel, gt_pct, pred_pct, panel_w
):
    gt_mask_vis = cv2.cvtColor(gt_mask, cv2.COLOR_GRAY2BGR)

    p1 = panel_with_title(original, "Original", panel_w, color=(255, 255, 255))
    p2 = panel_with_title(
        gt_mask_vis,
        f"GT Mask ({gt_pct:.1f}% lane)",
        panel_w,
        color=(255, 255, 255),
    )
    p3 = panel_with_title(gt_overlay, "GT Overlay", panel_w, color=(0, 255, 0))
    p4 = panel_with_title(
        pred_panel,
        f"Prediction ({pred_pct:.1f}% lane)",
        panel_w,
        color=(255, 255, 0),
    )
    return np.hstack([p1, p2, p3, p4])


def collect_image_paths(input_dir, image_glob):
    primary_pattern = os.path.join(input_dir, image_glob)
    image_paths = sorted(glob.glob(primary_pattern))
    if image_paths:
        return image_paths, primary_pattern

    fallback_globs = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]
    fallback_paths = []
    for pattern in fallback_globs:
        fallback_paths.extend(glob.glob(os.path.join(input_dir, pattern)))

    return sorted(set(fallback_paths)), primary_pattern


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)

    cfg.test_model = args.model
    dataset_name = args.dataset if args.dataset is not None else cfg.dataset
    num_lanes = (
        args.num_lanes if args.num_lanes is not None else getattr(cfg, "num_lanes", 4)
    )

    cls_num_per_lane, row_anchor = get_protocol_params(dataset_name)
    device = choose_device(args.device)

    image_paths, image_pattern = collect_image_paths(args.input_dir, args.image_glob)
    if not image_paths:
        raise FileNotFoundError(
            "No images found. Checked pattern: "
            f"{image_pattern}. Also tried: *.jpg, *.jpeg, *.png"
        )

    first = cv2.imread(image_paths[0])
    if first is None:
        raise RuntimeError(f"Failed to read first image: {image_paths[0]}")

    save_images = args.output_mode in ["images", "both"]
    save_video = args.output_mode in ["video", "both"]

    if save_images and not args.output_dir:
        raise ValueError("--output_dir is required when output_mode is images or both")
    if save_video and not args.output_video:
        raise ValueError("--output_video is required when output_mode is video or both")

    if save_images:
        os.makedirs(args.output_dir, exist_ok=True)

    gt_mask_dir = args.gt_mask_dir
    if args.render_style == "comparison" and gt_mask_dir is None:
        gt_mask_dir = infer_gt_mask_dir(args.input_dir)

    writer = None
    if save_video:
        out_dir = os.path.dirname(args.output_video)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        first_h, first_w = first.shape[:2]
        if args.render_style == "comparison":
            panel_h = int(round(first_h * float(args.panel_width) / float(first_w)))
            video_size = (args.panel_width * 4, panel_h + 44)
        else:
            video_size = (first_w, first_h)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.output_video, fourcc, args.fps, video_size)
        if not writer.isOpened():
            raise RuntimeError(f"Failed to create video writer: {args.output_video}")

    net = build_model(cfg, cls_num_per_lane, device, num_lanes)

    img_transforms = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((288, 800)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )

    print(f"Device: {device}")
    print(f"Images: {len(image_paths)}")
    if save_images:
        print(f"Writing pictures to: {args.output_dir}")
    if save_video:
        print(f"Writing video: {args.output_video}")
    if args.render_style == "comparison":
        print(f"Comparison style enabled (GT mask dir: {gt_mask_dir})")
    print(f"Lane render style: {args.lane_style}")

    try:
        with torch.no_grad():
            for image_path in tqdm(image_paths, desc="Inference", unit="img"):
                frame = cv2.imread(image_path)
                if frame is None:
                    print(f"Warning: skipping unreadable image: {image_path}")
                    continue

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                inp = img_transforms(rgb).unsqueeze(0).to(device)

                out = net(inp)
                lane_loc = decode_lanes(out, cfg.griding_num)
                pred_vis, pred_mask = draw_lanes(
                    frame.copy(),
                    lane_loc,
                    row_anchor,
                    cls_num_per_lane,
                    cfg.griding_num,
                    args.thickness,
                    args.lane_style,
                    color=(255, 255, 0),
                )

                if args.render_style == "comparison":
                    mask_path = find_gt_mask_path(image_path, gt_mask_dir)
                    gt_mask = load_binary_mask(mask_path, frame.shape[:2])
                    gt_overlay = make_gt_overlay(frame, gt_mask)
                    final_vis = make_comparison_canvas(
                        frame,
                        gt_mask,
                        gt_overlay,
                        pred_vis,
                        lane_percent(gt_mask),
                        lane_percent(pred_mask),
                        args.panel_width,
                    )
                else:
                    final_vis = pred_vis

                if save_images:
                    base = os.path.basename(image_path)
                    stem, ext = os.path.splitext(base)
                    out_image = os.path.join(args.output_dir, f"{stem}_lanes{ext}")
                    ok = cv2.imwrite(out_image, final_vis)
                    if not ok:
                        print(f"Warning: failed to write image: {out_image}")

                if save_video and writer is not None:
                    writer.write(final_vis)
    finally:
        if writer is not None:
            writer.release()

    print("Done.")


if __name__ == "__main__":
    main()
