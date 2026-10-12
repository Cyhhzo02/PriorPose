# PriorPose: Reference-Guided Joint Deformation and Alignment for Category-Level Object Pose Estimation

Yihan Chen, Huan Ren, Wenfei Yang, Hang Du, Tianzhu Zhang, Feng Wu

ECCV 2026 · [Paper](https://arxiv.org/abs/2609.16727) · [Pretrained models](#pretrained-models)

This repository provides the implementation of **PriorPose** for
category-level 6D object pose estimation on
[NOCS](https://github.com/hughw19/NOCS_CVPR2019) and
[HouseCat6D](https://sites.google.com/view/housecat6d).

PriorPose represents each category with an explicit canonical shape prior and
jointly reasons about observation-to-NOCS correspondence and instance-specific
prior deformation. The two tasks share a seeded two-stream Transformer and are
coupled by canonical-space reconstruction and camera-space consistency.

The repository provides a unified model and training interface for both
datasets, with dataset-specific settings in `config/`.

## Method overview

PriorPose consists of five main components:

1. **RGB-D feature extraction.** A frozen DINOv2-S/14 backbone extracts image
   features, while PointNet++ extracts per-point geometric features.
2. **Explicit category prior.** Each category uses a canonical mean shape and
   a learnable category-conditioned prior feature bank.
3. **Seeded two-stream fusion.** A Transformer exchanges information
   between observation and prior tokens through FPS seeds, local geometry
   routing and geometry-quality gating.
4. **Joint reconstruction and correspondence.** One head predicts dense NOCS
   coordinates for observed points and another deforms the category prior into
   an instance-specific canonical reconstruction.
5. **Pose and size estimation.** A learned head regresses rotation,
   translation and metric size from observed points, predicted NOCS and fused
   features.

## Repository structure

```text
PriorPose/
├── config/                 # NOCS and HouseCat6D configurations
├── docs/                   # release scope and implementation notes
├── model/                  # PriorPose network, loss and PointNet++ extension
├── provider/               # NOCS and HouseCat6D datasets
├── scripts/                # third-party dependency setup
├── third_party/            # local DINOv2 checkout (not vendored)
├── utils/                  # training and official evaluation utilities
├── train.py                # unified training entry point
└── test.py                 # unified inference and evaluation entry point
```

## Environment

A fresh environment with Python 3.10 or newer and a CUDA-enabled PyTorch build
is recommended. Install PyTorch and torchvision using the versions appropriate
for your CUDA driver, then run:

```bash
git clone https://github.com/Cyhhzo02/PriorPose.git
cd PriorPose

python -m pip install -r requirements.txt
python -m pip install --no-build-isolation -v ./model/pointnet2
bash scripts/setup_dinov2.sh
```

The two PointNet++ extensions (native operators and fixed-order CUDA backward)
are compiled locally; a CUDA toolkit with `nvcc` and a C++ compiler are required.
For an RTX PRO 6000 Blackwell or
another `sm_120` GPU, install a PyTorch/CUDA build that explicitly supports
Blackwell before compiling the extension.

Verify the native extension before starting a long run:

```bash
python -c "import torch; from pointnet2 import _ext, _ordered; print(_ext.__file__); print(_ordered.__file__)"
```

DINOv2 downloads its ViT-S/14 checkpoint through Torch Hub when the model is
first constructed. On an offline machine, place an official DINOv2 checkout at
`third_party/dinov2` and populate the Torch Hub checkpoint cache in advance. The setup script pins
DINOv2 to the revision used by this release. See
[runtime settings](docs/RELEASE_SCOPE.md#runtime-settings) for the recorded
NOCS software environment.

## Data preparation

### NOCS

Download and preprocess NOCS following the standard category-level pose
protocol. The preprocessing instructions from
[Self-DPDN](https://github.com/JiehongLin/Self-DPDN) are compatible with the
loader used here. Download the CAMERA25/REAL275 segmentation results following
the public AG-Pose/DualPoseNet evaluation setup.

The expected layout is:

```text
data/nocs/
├── camera/
│   ├── train/
│   ├── val/
│   ├── train_list.txt
│   └── val_list.txt
├── real/
│   ├── train/
│   ├── test/
│   └── train_list.txt
├── segmentation_results/
│   ├── CAMERA25/
│   └── REAL275/
├── camera_full_depths/
├── gts/
└── obj_models/
```

### HouseCat6D

Download HouseCat6D from the
[official project page](https://sites.google.com/view/housecat6d). The expected
layout is:

```text
data/housecat6d/
├── scene*/
├── val_scene*/
├── test_scene*/
└── obj_models_small_size_final/
    └── objects.pkl
```

### Category mean shapes

PriorPose requires precomputed category mean shapes:

```text
data/mean_shapes/
├── nocs_512.npy          # float32, shape [6, 512, 3]
└── housecat6d_512.npy    # float32, shape [10, 512, 3]
```

The [pretrained-model archives](#pretrained-models) include these mean shapes.
Alternatively, prepare them from the original mean shapes:

```bash
python scripts/prepare_mean_shapes.py \
  --dataset nocs --input /path/to/mean_shapes.npy \
  --output data/mean_shapes/nocs_512.npy

python scripts/prepare_mean_shapes.py \
  --dataset housecat6d --input /path/to/mean_shape_housecat6d.npy \
  --output data/mean_shapes/housecat6d_512.npy
```

Dataset, mean-shape and DINOv2 paths can be overridden from the command line:

```text
--data-root PATH --mean-shapes PATH --dino-repo PATH
```

## Training

### NOCS

```bash
PYTHONHASHSEED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python train.py \
  --config config/nocs.yaml \
  --gpu 0 \
  --output-dir runs/priorpose_nocs
```

### HouseCat6D

```bash
python train.py \
  --config config/housecat6d.yaml \
  --gpu 0 \
  --output-dir runs/priorpose_housecat6d
```

Resume training with the same config and output directory:

```bash
python train.py \
  --config config/housecat6d.yaml \
  --gpu 0 \
  --output-dir runs/priorpose_housecat6d \
  --resume runs/priorpose_housecat6d/checkpoints/epoch_50.pth
```

Resume restores model weights, optimizer state and the epoch/step counters.
For NOCS training, start a fresh run with the command above.

## Evaluation

### NOCS REAL275

```bash
python test.py \
  --config config/nocs.yaml \
  --gpu 0 \
  --checkpoint runs/priorpose_nocs/checkpoints/epoch_30.pth \
  --output-dir results/nocs_epoch30
```

### HouseCat6D

```bash
python test.py \
  --config config/housecat6d.yaml \
  --gpu 0 \
  --checkpoint runs/priorpose_housecat6d/checkpoints/epoch_150.pth \
  --output-dir results/housecat6d_epoch150
```

The test entry point saves predictions and runs the corresponding official
evaluator.

## Pretrained models

Download the pretrained models below:

| Dataset | Checkpoint | Config | Download |
| --- | --- | --- | --- |
| NOCS REAL275 | `priorpose_nocs_epoch30.pth` | [nocs.yaml](config/nocs.yaml) | [Download](https://github.com/Cyhhzo02/PriorPose/releases/download/v1.0.0/PriorPose-nocs-assets.tar.gz) |
| HouseCat6D | `priorpose_housecat6d_epoch150.pth` | [housecat6d.yaml](config/housecat6d.yaml) | [Download](https://github.com/Cyhhzo02/PriorPose/releases/download/v1.0.0/PriorPose-housecat6d-assets.tar.gz) |

Each dataset archive includes the pretrained checkpoint and category mean
shapes. Extract it into the repository root, then evaluate:

```bash
tar -xzf PriorPose-nocs-assets.tar.gz
python test.py --config config/nocs.yaml --gpu 0 \
  --checkpoint checkpoints/priorpose_nocs_epoch30.pth \
  --output-dir results/nocs_release
```

For HouseCat6D:

```bash
tar -xzf PriorPose-housecat6d-assets.tar.gz
python test.py --config config/housecat6d.yaml --gpu 0 \
  --checkpoint checkpoints/priorpose_housecat6d_epoch150.pth \
  --output-dir results/housecat6d_release
```

Both checkpoints contain model weights only; training checkpoints written by
`train.py` additionally contain optimizer state for resuming.

The documented NOCS training command uses the
fixed-order CUDA backend; the pretrained checkpoint was trained with native
PointNet++ backward. Both use the same network and loss settings.

## Tests

```bash
python -m unittest discover -s tests -v
```

CUDA operator tests run when a CUDA device is available.

## Citation

If you use PriorPose in your research, please cite:

```bibtex
@misc{chen2026priorpose,
  title={PriorPose: Reference-Guided Joint Deformation and Alignment for Category-Level Object Pose Estimation},
  author={Chen, Yihan and Ren, Huan and Yang, Wenfei and Du, Hang and Zhang, Tianzhu and Wu, Feng},
  year={2026},
  eprint={2609.16727},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2609.16727}
}
```

This implementation builds on AG-Pose; please also cite:

```bibtex
@inproceedings{lin2024instance,
  title={Instance-adaptive and geometric-aware keypoint learning for
         category-level 6D object pose estimation},
  author={Lin, Xiao and Yang, Wenfei and Gao, Yuan and Zhang, Tianzhu},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and
             Pattern Recognition},
  pages={21040--21049},
  year={2024}
}
```

## Acknowledgements

This repository builds on the public implementations and evaluation tools of:

- [AG-Pose](https://github.com/Leeiieeo/AG-Pose)
- [NOCS](https://github.com/hughw19/NOCS_CVPR2019)
- [SPD](https://github.com/mentian/object-deformnet)
- [DualPoseNet](https://github.com/Gorilla-Lab-SCUT/DualPoseNet)
- [Self-DPDN](https://github.com/JiehongLin/Self-DPDN)
- [VI-Net](https://github.com/JiehongLin/VI-Net)
- [HouseCat6D Toolbox](https://github.com/Junggy/HouseCat6D)
- [DINOv2](https://github.com/facebookresearch/dinov2)

Please see [NOTICE](NOTICE) and the source-file headers for attribution details.

## License

PriorPose is released under the [MIT License](LICENSE).

## Contact

Please open a GitHub issue for questions about the released code or data setup.
