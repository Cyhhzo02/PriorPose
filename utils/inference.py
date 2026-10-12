"""Prediction serialization and official NOCS/HouseCat6D evaluation."""

import logging
import pickle
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from model import PriorPose
from provider.housecat6d_dataset import HouseCat6DTestDataset
from provider.nocs_dataset import TestDataset
from utils.evaluation_utils import evaluate
from utils.housecat6d_eval_utils import evaluate_housecat


def _logger(output_dir):
    logger = logging.getLogger("priorpose.test")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(message)s")
    file_handler = logging.FileHandler(output_dir / "evaluation.log")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def _load_model(cfg, checkpoint, device):
    model = PriorPose(cfg.model).to(device)
    payload = torch.load(checkpoint, map_location="cpu")
    state_dict = payload["model"] if "model" in payload else payload
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def _pose_results(predictions):
    translation = predictions["pred_translation"]
    size = predictions["pred_size"]
    scale = torch.norm(size, dim=1, keepdim=True).clamp_min(1e-8)
    normalized_size = size / scale
    rotation = predictions["pred_rotation"]
    count = rotation.shape[0]
    transforms = torch.eye(
        4, device=rotation.device, dtype=rotation.dtype
    ).unsqueeze(0).repeat(count, 1, 1)
    transforms[:, :3, 3] = translation
    transforms[:, :3, :3] = rotation * scale.unsqueeze(2)
    return transforms.cpu().numpy(), normalized_size.cpu().numpy()


def _model_inputs(data, device):
    return {
        key: data[key][0].to(device, non_blocking=True)
        for key in ["rgb", "pts", "choose", "category_label", "prior"]
    }


def _base_result(data):
    result = {
        "gt_class_ids": data["gt_class_ids"][0].numpy(),
        "gt_bboxes": data["gt_bboxes"][0].numpy(),
        "gt_RTs": data["gt_RTs"][0].numpy(),
        "gt_scales": data["gt_scales"][0].numpy(),
        "pred_class_ids": data["pred_class_ids"][0].numpy(),
        "pred_bboxes": data["pred_bboxes"][0].numpy(),
        "pred_scores": data["pred_scores"][0].numpy(),
    }
    if "gt_handle_visibility" in data:
        result["gt_handle_visibility"] = data["gt_handle_visibility"][0].numpy()
    return result


def _failed_pose(count):
    transforms = np.repeat(np.eye(4, dtype=np.float32)[None], count, axis=0)
    scales = np.ones((count, 3), dtype=np.float32)
    return transforms, scales


def test_nocs(cfg, checkpoint, output_dir, workers=8, category_id=-1):
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = _logger(output_dir)
    logger.info("config=%s checkpoint=%s", cfg.config_path, checkpoint)
    logger.info("dataset=nocs data_root=%s output=%s", cfg.data.root, output_dir)
    device = torch.device("cuda")
    model = _load_model(cfg, checkpoint, device)
    dataset = TestDataset(
        cfg.data.image_size,
        cfg.data.sample_num,
        cfg.data.root,
        cfg.data.mean_shape_path,
        "supervised",
        "real",
    )
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, num_workers=workers, shuffle=False
    )
    with torch.inference_mode():
        for index, data in enumerate(tqdm(loader, desc="NOCS inference")):
            result = _base_result(data)
            predictions = model(_model_inputs(data, device))
            result["pred_RTs"], result["pred_scales"] = _pose_results(
                predictions
            )
            filename = Path(dataset.result_pkl_list[index]).name
            with (output_dir / filename).open("wb") as file:
                pickle.dump(result, file)
    evaluate(str(output_dir), logger=logger, cat_id=category_id)


def test_housecat(cfg, checkpoint, output_dir, workers=8):
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = _logger(output_dir)
    logger.info("config=%s checkpoint=%s", cfg.config_path, checkpoint)
    logger.info(
        "dataset=housecat6d data_root=%s output=%s", cfg.data.root, output_dir
    )
    device = torch.device("cuda")
    model = _load_model(cfg, checkpoint, device)
    dataset = HouseCat6DTestDataset(
        cfg.data.image_size,
        cfg.data.sample_num,
        cfg.data.root,
        cfg.data.mean_shape_path,
    )
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, num_workers=workers, shuffle=False
    )
    with torch.inference_mode():
        for index, data in enumerate(tqdm(loader, desc="HouseCat6D inference")):
            result = _base_result(data)
            if "pts" in data:
                predictions = model(_model_inputs(data, device))
                result["pred_RTs"], result["pred_scales"] = _pose_results(
                    predictions
                )
            else:
                result["pred_RTs"], result["pred_scales"] = _failed_pose(
                    len(result["pred_class_ids"])
                )
            image_path = Path(dataset.test_img_list[index])
            scene = next(part for part in image_path.parts if "test_scene" in part)
            scene_dir = output_dir / scene
            scene_dir.mkdir(exist_ok=True)
            with (scene_dir / image_path.with_suffix(".pkl").name).open("wb") as file:
                pickle.dump(result, file)
    evaluate_housecat(str(output_dir), logger=logger)
