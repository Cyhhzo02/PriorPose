"""Dataloader construction for the two released datasets."""

import random

import numpy as np
import torch

from provider.housecat6d_dataset import HouseCat6DTrainingDataset
from provider.nocs_dataset import TrainingDataset


def _seed_worker(_worker_id):
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def _deterministic_loader_options(seed):
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return {"generator": generator, "worker_init_fn": _seed_worker}


def build_train_loaders(cfg, seed=0, deterministic=False):
    common = dict(
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        persistent_workers=False,
    )
    if cfg.dataset == "nocs":
        synthetic = TrainingDataset(
            cfg.image_size,
            cfg.sample_num,
            cfg.root,
            cfg.mean_shape_path,
            data_type="syn",
            num_img_per_epoch=cfg.steps_per_epoch * cfg.synthetic_batch_size,
            threshold=cfg.outlier_threshold,
        )
        real = TrainingDataset(
            cfg.image_size,
            cfg.sample_num,
            cfg.root,
            cfg.mean_shape_path,
            data_type="real_withLabel",
            num_img_per_epoch=cfg.steps_per_epoch * cfg.real_batch_size,
            threshold=cfg.outlier_threshold,
        )
        synthetic_options = dict(common)
        real_options = dict(common)
        if deterministic:
            synthetic_options.update(_deterministic_loader_options(seed + 1))
            real_options.update(_deterministic_loader_options(seed + 2))
        return {
            "synthetic": torch.utils.data.DataLoader(
                synthetic,
                batch_size=cfg.synthetic_batch_size,
                num_workers=cfg.synthetic_workers,
                **synthetic_options,
            ),
            "real": torch.utils.data.DataLoader(
                real,
                batch_size=cfg.real_batch_size,
                num_workers=cfg.real_workers,
                **real_options,
            ),
        }
    if cfg.dataset == "housecat6d":
        dataset = HouseCat6DTrainingDataset(
            cfg.image_size,
            cfg.sample_num,
            cfg.root,
            cfg.mean_shape_path,
            seq_length=cfg.sequence_count,
            img_length=cfg.images_per_sequence,
            threshold=cfg.outlier_threshold,
        )
        loader_options = dict(common)
        if deterministic:
            loader_options.update(_deterministic_loader_options(seed + 1))
        return {
            "real": torch.utils.data.DataLoader(
                dataset,
                batch_size=cfg.batch_size,
                num_workers=cfg.workers,
                **loader_options,
            )
        }
    raise ValueError(f"unsupported dataset: {cfg.dataset}")
