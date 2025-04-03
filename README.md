# Spatial-Temporal Enhanced Transformer Towards Multi-Frame 3D Object Detection 
[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2307.00347)

Implementation of paper: "Spatial-Temporal Enhanced Transformer Towards Multi-Frame 3D Object Detection".  The implementation is mainly based on an Efficient, Flexible, and General deep learning framework, namely [EFG](https://github.com/V2AI/EFG).

## Overview
- [Installation](#1-installation)
- [Data](#2-data)
- [Get Started](#3-get-started)
- [Main Results](#4-main-results)
- [Citation](#5-citation)

# 1. Installation

## 1.1 Prerequisites
* gcc 5 - 7
* python >= 3.6
* cuda >= 10.2
* pytorch >= 1.6

```shell
# spconv
spconv_cu11{X} (set X according to your cuda version)

# waymo_open_dataset
## python 3.6
waymo-open-dataset-tf-2-1-0==1.2.0

## python 3.7, 3.8
waymo-open-dataset-tf-2-4-0==1.3.1

```
## 1.2 Build from source

```shell
git clone https://github.com/Eaphan/STEMD.git
cd STEMD
pip install -v -e .
# set logging path to save model checkpoints, training logs, etc.
echo "export EFG_CACHE_DIR=/path/to/your/logs/dir" >> ~/.bashrc
```

# 2. Data
## 2.1 Data Preparation - Waymo
```shell

# download waymo dataset v1.2.0 (or v1.3.2, etc)
gsutil -m cp -r \
  "gs://waymo_open_dataset_v_1_2_0_individual_files/testing" \
  "gs://waymo_open_dataset_v_1_2_0_individual_files/training" \
  "gs://waymo_open_dataset_v_1_2_0_individual_files/validation" \
  .

# extract frames from tfrecord to pkl
CUDA_VISIBLE_DEVICES=-1 python cli/data_preparation/waymo/waymo_converter.py --record_path "/path/to/waymo/training/*.tfrecord" --root_path "/path/to/waymo/train/"
CUDA_VISIBLE_DEVICES=-1 python cli/data_preparation/waymo/waymo_converter.py --record_path "/path/to/waymo/validation/*.tfrecord" --root_path "/path/to/waymo/val/"

# create softlink to datasets
cd /path/to/STEMD/datasets; ln -s /path/to/waymo/dataset/root waymo; cd ..
# create data summary and gt database from extracted frames
python cli/data_preparation/waymo/create_data.py --root-path datasets/waymo --split train --nsweeps 4
python cli/data_preparation/waymo/create_data.py --root-path datasets/waymo --split val --nsweeps 4

```

## 2.2 Data Preparation - nuScenes
```shell
# create softlink to datasets
cd /path/to/STEMD/datasets; ln -s /path/to/nuscenes/dataset/root nuscenes; cd ..
python cli/data_preparation/nuscenes/create_data.py --root-path datasets/nuscenes --version v1.0-trainval --nsweeps 31  # 1 sample frame + 30 sweeps frame (1.5s)
```

# 3. Get Started
##  3.1 Training & Evaluation

```shell
cd playground/detection.3d/waymo/stemd/STEMD.waymo.resnet18.cdn.epoch12

efg_run --num-gpus x  # default 1
efg_run --num-gpus x task [train | val | test]
efg_run --num-gpus x --resume
efg_run --num-gpus x dataloader.num_workers 0  # dynamically change options in config.yaml
```
Models will be evaluated automatically at the end of training. Or, 
```shell
efg_run --num-gpus x task val
```


# 4. Main Results

All models are trained and evaluated on 8 x NVIDIA A100 GPUs.

## Waymo Open Dataset - 3D Object Detection (val L2- mAP/mAPH)

|    Methods    | Frames | Schedule |  VEHICLE  | PEDESTRIAN |  CYCLIST  |
| :-----------: | :----: | :------: | :-------: | :--------: | :-------: |
|    [STEMD](playground/detection.3d/waymo/stemd/STEMD.waymo.resnet18_wide2x.cdn.epoch12/config.yaml)      |   4    |    12    | 72.4/72.0 | 78.0/74.7  | 78.0/76.9 |

<!--
## nuScenes - 3D Object Detection (val)

|    Methods    | Schedule | mAP  | NDS  | Logs |
| :-----------: | :------: | :--: | :--: | :--: |
|  CenterPoint  |    20    | 59.0 | 66.7 |      |
-->

# 5. Citation
```shell
@article{zhang2024stemd,
  title={Spatial-Temporal Graph Enhanced DETR Towards Multi-Frame 3D Object Detection},
  author={Zhang, Yifan and Zhu, Zhiyu and Hou, Junhui and Wu, Dapeng},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
  volume={46},
  number={12},
  pages={10614--10628},
  year={2024}
}
@misc{zhu2023efg,
    title={EFG: An Efficient, Flexible, and General deep learning framework that retains minimal},
    author={EFG Contributors},
    howpublished = {\url{https://github.com/poodarchu/efg}},
    year={2023}
}
@inproceedings{zhu2023conquer,
  title={Conquer: Query contrast voxel-detr for 3d object detection},
  author={Zhu, Benjin and Wang, Zhe and Shi, Shaoshuai and Xu, Hang and Hong, Lanqing and Li, Hongsheng},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={9296--9305},
  year={2023}
}
```
