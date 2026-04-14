# Ultra-Fast-Lane-Detection

PyTorch implementation of the paper "[Ultra Fast Structure-aware Deep Lane Detection](https://arxiv.org/abs/2004.11757)".

**\[July 18, 2022\] Updates: The new version of our method has been accepted by TPAMI 2022. Code is available [here](https://github.com/cfzd/Ultra-Fast-Lane-Detection-v2)**.

\[June 28, 2021\] Updates: we will release an extended version, which improves **6.3** points of F1 on CULane with the ResNet-18 backbone compared with the ECCV version.

Updates: Our paper has been accepted by ECCV2020.

![alt text](vis.jpg "vis")

The evaluation code is modified from [SCNN](https://github.com/XingangPan/SCNN) and [Tusimple Benchmark](https://github.com/TuSimple/tusimple-benchmark).

Caffe model and prototxt can be found [here](https://github.com/Jade999/caffe_lane_detection).

# Demo

<a href="http://www.youtube.com/watch?feature=player_embedded&v=lnFbAG3GBN4
" target="_blank"><img src="http://img.youtube.com/vi/lnFbAG3GBN4/0.jpg"
alt="Demo" width="240" height="180" border="10" /></a>

# Install

Please see [INSTALL.md](./INSTALL.md)

# Get started

First of all, please modify `data_root` and `log_path` in your `configs/culane.py` or `configs/tusimple.py` config according to your environment.

- `data_root` is the path of your CULane dataset or Tusimple dataset.
- `log_path` is where tensorboard logs, trained models and code backup are stored. ***It should be placed outside of this project.***

***

For single gpu training, run

```Shell
python train.py configs/path_to_your_config
```

For Modal cloud training and W&B logging workflow, see [Walkthrough.md](./Walkthrough.md).

Since our code has auto backup function which will copy all codes to the `log_path` according to the gitignore, additional temp file might also be copied if it is not filtered by gitignore, which may block the execution if the temp files are large. So you should keep the working directory clean.
***

Besides config style settings, we also support command line style one. You can override a setting like

```Shell
python train.py configs/path_to_your_config --batch_size 8
```

The ```batch_size``` will be set to 8 during training.

***

To visualize the log with tensorboard, run

```Shell
tensorboard --logdir log_path --bind_all
```

# Trained models

We provide two trained Res-18 models on CULane and Tusimple.

|  Dataset | Metric paper | Metric This repo | Avg FPS on GTX 1080Ti |    Model    |
|:--------:|:------------:|:----------------:|:-------------------:|:-----------:|
| Tusimple |     95.87    |       95.82      |         306         | [GoogleDrive](https://drive.google.com/file/d/1WCYyur5ZaWczH15ecmeDowrW30xcLrCn/view?usp=sharing)/[BaiduDrive(code:bghd)](https://pan.baidu.com/s/1Fjm5yVq1JDpGjh4bdgdDLA) |
|  CULane  |     68.4     |       69.7       |         324         | [GoogleDrive](https://drive.google.com/file/d/1zXBRTw50WOzvUp6XKsi8Zrk3MUC3uFuq/view?usp=sharing)/[BaiduDrive(code:w9tw)](https://pan.baidu.com/s/19Ig0TrV8MfmFTyCvbSa4ag) |

For testing, use one of the examples in the next section.

# Visualization

We provide a script to visualize the detection results. Run the following commands to visualize on the testing set of CULane and Tusimple.

```Shell
python demo.py configs/culane.py --test_model path_to_culane_18.pth
# or
python demo.py configs/tusimple.py --test_model path_to_tusimple_18.pth
```

Since the testing set of Tusimple is not ordered, the visualized video might look bad and we **do not recommend** doing this.

## Run On Your Own Images (Pictures or Video, Python Only)

If you only want to run inference on a folder of images and export picture results (or a video), you can use the Python script below. This path does not require the C++ evaluation tools.

```Shell
python scripts/infer_images_to_video.py configs/tulane.py \
  --model path_to_your_model.pth \
  --input_dir path_to_your_images \
  --render_style comparison \
  --output_mode images \
  --output_dir outputs/lanes_images \
  --device cpu \
  --dataset CULane
```

Notes:

- Use `--device cpu` to avoid CUDA completely (slower).
- Use `--device auto` to use CUDA when available.
- Use `--render_style comparison` to get a 4-panel screenshot style: Original, GT Mask, GT Overlay, Prediction.
- If needed, set `--gt_mask_dir path_to_lane_masks` explicitly (otherwise the script tries to infer it from `input_dir`).
- Use `--output_mode video` with `--output_video outputs/lanes.mp4` if you still want a video.
- Use `--output_mode both` to save both pictures and a video in one run.
- Change `--image_glob` if your images are not `.jpg` (example: `--image_glob '*.png'`).

## Testing Command Examples

1. Comparison screenshots (Original + GT Mask + GT Overlay + Prediction):

```Shell
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

1. Prediction overlay images only:

```Shell
python scripts/infer_images_to_video.py configs/tulane.py \
  --model path_to_your_model.pth \
  --input_dir ../dataset/TuLaneConverted/images/target_test \
  --render_style overlay \
  --output_mode images \
  --output_dir outputs/target_test_overlay \
  --device cpu \
  --dataset CULane
```

1. Python-only benchmark file generation (no C++ metrics) for CULane using `test.py`:

```Shell
python test.py configs/tulane.py \
  --data_root ../dataset/TuLaneConverted \
  --test_model path_to_your_model.pth \
  --test_work_dir ./tmp_eval
```

# Speed

To test the runtime, please run

```Shell
python speed_simple.py  
# this will test the speed with a simple protocol and requires no additional dependencies

python speed_real.py
# this will test the speed with real video or camera input
```

It will loop 100 times and calculate the average runtime and fps in your environment.

# Citation

```BibTeX
@InProceedings{qin2020ultra,
author = {Qin, Zequn and Wang, Huanyu and Li, Xi},
title = {Ultra Fast Structure-aware Deep Lane Detection},
booktitle = {The European Conference on Computer Vision (ECCV)},
year = {2020}
}

@ARTICLE{qin2022ultrav2,
  author={Qin, Zequn and Zhang, Pengyi and Li, Xi},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
  title={Ultra Fast Deep Lane Detection With Hybrid Anchor Driven Ordinal Classification}, 
  year={2022},
  volume={},
  number={},
  pages={1-14},
  doi={10.1109/TPAMI.2022.3182097}
}
```

# Thanks

Thanks zchrissirhcz for the contribution to the compile tool of CULane, KopiSoftware for contributing to the speed test, and ustclbh for testing on the Windows platform.
