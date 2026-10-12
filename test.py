#!/usr/bin/env python3
"""Test PriorPose and run the official dataset evaluator."""

import argparse
import os
import sys
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parent
sys.path.insert(0, str(REPOSITORY))
sys.path.insert(0, str(REPOSITORY / "model" / "pointnet2"))

from utils.config import load_config  # noqa: E402


def _absolute(path):
    path = Path(path).expanduser()
    return str(path if path.is_absolute() else (REPOSITORY / path).resolve())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--category-id", type=int, default=-1)
    parser.add_argument("--data-root")
    parser.add_argument("--mean-shapes")
    parser.add_argument("--dino-repo")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    cfg = load_config(args.config)
    cfg.data.root = _absolute(args.data_root or cfg.data.root)
    cfg.data.mean_shape_path = _absolute(
        args.mean_shapes or cfg.data.mean_shape_path
    )
    cfg.model.dino_repository = _absolute(
        args.dino_repo or cfg.model.dino_repository
    )

    from utils.runtime import configure_runtime

    configure_runtime(cfg)
    if bool(getattr(cfg, "deterministic", False)):
        from utils.training import seed_everything

        seed_everything(int(cfg.seed), deterministic=True)

    from utils.inference import test_housecat, test_nocs

    if cfg.data.dataset == "nocs":
        test_nocs(
            cfg,
            _absolute(args.checkpoint),
            _absolute(args.output_dir),
            workers=args.workers,
            category_id=args.category_id,
        )
    elif cfg.data.dataset == "housecat6d":
        test_housecat(
            cfg,
            _absolute(args.checkpoint),
            _absolute(args.output_dir),
            workers=args.workers,
        )
    else:
        raise ValueError(f"unsupported dataset: {cfg.data.dataset}")


if __name__ == "__main__":
    main()
