"""Reproducible single-GPU training loop for PriorPose."""

import json
import logging
import math
import os
import random
import shlex
import sys
from pathlib import Path

import numpy as np
import torch

from model import PriorPose, PriorPoseLoss
from provider.dataloaders import build_train_loaders
from utils.runtime import configure_runtime


def seed_everything(seed, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        torch.use_deterministic_algorithms(True)


def move_to_device(batch, device):
    return {
        key: value.to(device, non_blocking=True)
        if torch.is_tensor(value)
        else value
        for key, value in batch.items()
    }


def _learning_rate(step, cfg):
    if step < cfg.warmup_steps:
        progress = step / max(1, cfg.warmup_steps)
        return cfg.learning_rate * (
            cfg.warmup_factor + (1.0 - cfg.warmup_factor) * progress
        )
    progress = min(
        1.0,
        (step - cfg.warmup_steps)
        / max(1, cfg.total_steps - cfg.warmup_steps),
    )
    return cfg.learning_rate * 0.5 * (1.0 + math.cos(math.pi * progress))


def _merge_model_inputs(first, second):
    keys = ["rgb", "pts", "choose", "category_label", "prior"]
    return {key: torch.cat([first[key], second[key]], dim=0) for key in keys}


def _attach_predictions(batch, predictions):
    return {**batch, **predictions}


def _weighted_nocs_loss(model, objective, synthetic, real):
    synthetic_count = synthetic["rgb"].shape[0]
    real_count = real["rgb"].shape[0]
    predictions = model(_merge_model_inputs(synthetic, real))
    synthetic_predictions = {
        key: value[:synthetic_count] for key, value in predictions.items()
    }
    real_predictions = {
        key: value[synthetic_count:] for key, value in predictions.items()
    }
    synthetic_losses = objective(
        _attach_predictions(synthetic, synthetic_predictions)
    )
    real_losses = objective(_attach_predictions(real, real_predictions))
    denominator = synthetic_count + real_count
    total = (
        synthetic_losses["loss_all"] * synthetic_count
        + real_losses["loss_all"] * real_count
    ) / denominator
    metrics = {
        key: (
            synthetic_losses[key].detach().item() * synthetic_count
            + real_losses[key].detach().item() * real_count
        )
        / denominator
        for key in synthetic_losses
    }
    metrics["loss_synthetic"] = synthetic_losses["loss_all"].detach().item()
    metrics["loss_real"] = real_losses["loss_all"].detach().item()
    return total, metrics


def _save_checkpoint(path, model, optimizer, epoch, step, cfg):
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "step": step,
            "config": cfg.config_path,
        },
        path,
    )


def train(cfg, output_dir, resume=None):
    runtime = configure_runtime(cfg, training=True)
    if not torch.cuda.is_available():
        raise RuntimeError("PriorPose training requires a CUDA GPU")
    output_dir = Path(output_dir).expanduser().resolve()
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("priorpose.train")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(message)s")
    file_handler = logging.FileHandler(output_dir / "train.log")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    deterministic = bool(getattr(cfg, "deterministic", False))
    seed_everything(int(cfg.seed), deterministic=deterministic)
    device = torch.device("cuda")
    model = PriorPose(cfg.model).to(device)
    objective = PriorPoseLoss(cfg.loss).to(device)
    optimizer = torch.optim.Adam(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=cfg.schedule.learning_rate,
        weight_decay=cfg.schedule.weight_decay,
    )
    start_epoch, global_step = 1, 0
    if resume:
        checkpoint = torch.load(resume, map_location="cpu")
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint["epoch"]) + 1
        global_step = int(checkpoint["step"])
        logger.info("resumed from %s at epoch=%d step=%d", resume, start_epoch, global_step)

    loaders = build_train_loaders(
        cfg.data,
        seed=int(cfg.seed),
        deterministic=deterministic,
    )
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    logger.info("config=%s", cfg.config_path)
    logger.info("output=%s parameters=%d", output_dir, parameter_count)
    metrics_path = output_dir / "metrics.jsonl"
    manifest = {
        "command": shlex.join(sys.argv),
        "config": cfg.config_path,
        "seed": int(cfg.seed),
        "dataset": cfg.data.dataset,
        "deterministic": deterministic,
        **runtime,
        "resolved_config": dict(cfg),
        "torch_version": str(torch.__version__),
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(),
        "data_root": cfg.data.root,
        "mean_shape_path": cfg.data.mean_shape_path,
        "output_dir": str(output_dir),
        "resume": str(resume) if resume else None,
    }
    with (output_dir / "run.json").open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2, sort_keys=True)
    logger.info("command=%s", manifest["command"])
    logger.info(
        "seed=%d dataset=%s deterministic=%s",
        cfg.seed,
        cfg.data.dataset,
        deterministic,
    )

    for epoch in range(start_epoch, int(cfg.epochs) + 1):
        model.train()
        for loader in loaders.values():
            loader.dataset.reset()
        iterator = (
            zip(loaders["synthetic"], loaders["real"])
            if cfg.data.dataset == "nocs"
            else loaders["real"]
        )
        running = {}
        batches = 0
        for batch_index, batch in enumerate(iterator, start=1):
            lr = _learning_rate(global_step, cfg.schedule)
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            if cfg.data.dataset == "nocs":
                synthetic, real = batch
                synthetic = move_to_device(synthetic, device)
                real = move_to_device(real, device)
                total, metrics = _weighted_nocs_loss(
                    model, objective, synthetic, real
                )
            else:
                real = move_to_device(batch, device)
                predictions = model(real)
                losses = objective(_attach_predictions(real, predictions))
                total = losses["loss_all"]
                metrics = {
                    key: value.detach().item() for key, value in losses.items()
                }
            total.backward()
            optimizer.step()
            global_step += 1
            batches += 1
            for key, value in metrics.items():
                running[key] = running.get(key, 0.0) + float(value)
            if batch_index % int(cfg.log_every) == 0:
                logger.info(
                    "epoch=%d batch=%d step=%d lr=%.7g loss=%.6f",
                    epoch,
                    batch_index,
                    global_step,
                    lr,
                    metrics["loss_all"],
                )

        epoch_metrics = {
            key: value / max(1, batches) for key, value in running.items()
        }
        record = {
            "epoch": epoch,
            "step": global_step,
            "learning_rate": _learning_rate(global_step, cfg.schedule),
            **epoch_metrics,
        }
        with metrics_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, sort_keys=True) + "\n")
        checkpoint_path = checkpoint_dir / f"epoch_{epoch}.pth"
        _save_checkpoint(
            checkpoint_path, model, optimizer, epoch, global_step, cfg
        )
        logger.info(
            "epoch=%d complete step=%d loss=%.6f checkpoint=%s",
            epoch,
            global_step,
            epoch_metrics.get("loss_all", float("nan")),
            checkpoint_path,
        )
