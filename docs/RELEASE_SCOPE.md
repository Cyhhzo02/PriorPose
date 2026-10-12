# Implementation and runtime

## Shared model

NOCS and HouseCat6D use the same `model/priorpose.py` network and
`model/loss.py` objective implementation. Both configs select six independent
Transformer blocks, 1,024 observation points, 512 category-prior points,
96/64 observation/prior seeds, four attention heads and 32 neighbors.
The geometry-quality gate and score bias are enabled for both datasets.

Features come from frozen DINOv2-S/14 RGB features and a PointNet++ backbone.
The two-stream Transformer includes a camera token, category-prior features
and coordinate embeddings. Binned soft-expectation heads predict NOCS
coordinates and prior deformation; a learned head estimates pose and size.

Both configs use symmetry-adjusted rotations in the NOCS and camera-consistency
terms. The category lists and term weights are recorded in each config. NOCS
uses raw-camera coordinates for its early embedding; HouseCat6D uses centered,
training-augmented coordinates. Later geometric routing is centered/augmented
in both cases.

The objective combines pose regression, Smooth-L1 NOCS supervision,
unsquared canonical reconstruction and camera-space alignment. Its exact
weights are specified in the YAML configs.

## Runtime settings

`config/nocs.yaml` uses `deterministic: true`,
`pointnet_backend: ordered_cuda`, and `pointnet_reduction_seed: 2`.
The latter controls a private, cached permutation of PointNet++ gradient
contributions; it does not consume the training RNG stream. Gathering,
grouping and interpolation accumulate FP32 gradients in a fixed order on the
GPU. Forward operators and the model architecture are unchanged.

The NOCS runtime also selects strict PyTorch operations, disables TF32 and
cuDNN benchmarking, sets `CUBLAS_WORKSPACE_CONFIG=:4096:8`, and uses separate
seeded CAMERA/REAL loader generators. Keep the software stack, GPU type,
batch sizes, worker counts and both seeds fixed when comparing training runs.

Tested environment for the release implementation and NOCS evaluation:

| Component | Version |
| --- | --- |
| Python | 3.10.19 |
| PyTorch | 2.11.0.dev20260202+cu128 |
| torchvision | 0.25.0.dev20260202+cu128 |
| CUDA toolkit | 12.8 |
| NumPy | 1.26.4 |
| GPU | NVIDIA GeForce RTX 4090 |
| DINOv2 revision | `e1277af2ba9496fbadf7aec6eba56e8d882d1e35` |

The HouseCat6D checkpoint, both FPS modes and the fixed-order backward tests
were also validated on an NVIDIA RTX PRO 6000 Blackwell (`sm_120`) with this
software stack.

Install a matching CUDA-enabled PyTorch/torchvision pair before building the
extensions. Set `TORCH_CUDA_ARCH_LIST` explicitly when building for a different
deployment GPU; otherwise PyTorch targets the visible GPUs.

HouseCat6D retains its native PointNet++ backward, loader policy and training
settings. The NOCS numerical settings are opt-in through its config, not global
changes to the HouseCat6D recipe. Use separate training processes per run.

FPS candidate handling is also dataset-specific. HouseCat6D sets
`pointnet_fps_skip_near_origin: false`, retaining all points in centered clouds.
NOCS retains the default `true`, excluding candidates with squared distance
to the origin at most `1e-3` after the initial seed. This setting applies to
both PointNet++ and Transformer seed selection in training and evaluation.
The model architecture and learned parameters are shared across datasets.

## Files and checkpoints

The source distribution contains the model, data loaders, official evaluation
utilities, two dataset configs, setup scripts and tests. Datasets, build products,
experiment logs and machine-specific launchers are not part of the source tree.
Pretrained weights and mean shapes are distributed separately from source.

Model-only checkpoints support `test.py`; checkpoints written by `train.py`
also include the optimizer, epoch and update count. `--resume` restores those
fields and starts the following epoch. Use fresh training for the documented
NOCS schedule.

The checkpoint asset manifest records its training configuration, evaluation
configuration, epoch and file hashes. The supplied NOCS model uses native
PointNet++ backward during training; `config/nocs.yaml` selects fixed-order
CUDA for new training and strict arithmetic for evaluation. Network and loss
settings are shared. Evaluation uses the standard dataset evaluator.
