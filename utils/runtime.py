"""Configure numerical execution without changing model or data definitions."""

import os

import torch


def configure_runtime(cfg, *, training=False):
    """Apply the selected recipe before constructing a model or CUDA tensors.

    Configs without these options retain the native execution path. Inference
    needs strict arithmetic but does not need to initialize a backward backend.
    """
    deterministic = bool(getattr(cfg, "deterministic", False))
    backend = getattr(cfg, "pointnet_backend", "native")
    if backend not in ("native", "ordered_cuda"):
        raise ValueError(f"Unknown PointNet++ backend: {backend}")
    from pointnet2_utils import configure_fps

    fps_skip_near_origin = getattr(cfg, "pointnet_fps_skip_near_origin", True)
    configure_fps(fps_skip_near_origin)
    if deterministic and backend != "ordered_cuda":
        raise ValueError("deterministic training recipes require pointnet_backend: ordered_cuda")
    if deterministic:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.set_num_threads(1)
        if torch.get_num_interop_threads() != 1:
            torch.set_num_interop_threads(1)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        torch.use_deterministic_algorithms(True)
    if training:
        from pointnet2_utils import configure_backward

        configure_backward(backend, getattr(cfg, "pointnet_reduction_seed", None))
    return {"pointnet_backend": backend,
            "pointnet_reduction_seed": getattr(cfg, "pointnet_reduction_seed", None),
            "pointnet_fps_skip_near_origin": fps_skip_near_origin}
