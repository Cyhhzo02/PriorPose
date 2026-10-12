#!/usr/bin/env python3
"""Deterministically reproduce the 512-point PriorPose category priors."""

import argparse
from pathlib import Path

import numpy as np


START_INDICES = {
    "nocs": [748, 859, 379, 176, 769, 201],
    "housecat6d": [308, 381, 570, 402, 461, 890, 233, 601, 788, 26],
}


def farthest_point_sample(points, count, start_index):
    selected = np.empty(count, dtype=np.int64)
    distances = np.full(points.shape[0], np.inf, dtype=np.float64)
    farthest = int(start_index)
    for index in range(count):
        selected[index] = farthest
        squared = np.square(points - points[farthest]).sum(axis=1)
        distances = np.minimum(distances, squared)
        farthest = int(np.argmax(distances))
    return points[selected]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=sorted(START_INDICES), required=True)
    parser.add_argument("--input", required=True, help="category mean shapes [C,N,3]")
    parser.add_argument("--output", required=True, help="output .npy path")
    args = parser.parse_args()

    source_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    source = np.load(source_path, allow_pickle=False).astype(np.float32)
    starts = START_INDICES[args.dataset]
    expected = (len(starts), 3)
    if source.ndim != 3 or source.shape[0] != expected[0] or source.shape[2] != 3:
        raise ValueError(
            f"expected [{expected[0]},N,3] for {args.dataset}, got {source.shape}"
        )
    if source.shape[1] < 512 or not np.isfinite(source).all():
        raise ValueError("input must contain at least 512 finite points per category")

    sampled = np.stack(
        [
            farthest_point_sample(points, 512, start)
            for points, start in zip(source, starts)
        ],
        axis=0,
    ).astype(np.float32)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, sampled, allow_pickle=False)
    print(f"saved {sampled.shape} {sampled.dtype} to {output_path}")


if __name__ == "__main__":
    main()
