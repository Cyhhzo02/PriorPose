#!/usr/bin/env python3
"""Train PriorPose on NOCS or HouseCat6D."""

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
    parser.add_argument("--output-dir")
    parser.add_argument("--resume")
    parser.add_argument("--gpu", default="0")
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
    output = _absolute(args.output_dir or f"runs/{cfg.experiment}")

    from utils.training import train

    resume = _absolute(args.resume) if args.resume else None
    train(cfg, output, resume=resume)


if __name__ == "__main__":
    main()
