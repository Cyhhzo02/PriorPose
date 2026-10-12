"""PriorPose training objective with joint canonical reconstruction."""

import torch
import torch.nn as nn

import pointnet2_utils


def smooth_l1_distance(prediction, target, threshold=0.1):
    difference = (prediction - target).abs()
    distance = torch.where(
        difference > threshold,
        difference - threshold / 2.0,
        difference.square() / (2.0 * threshold),
    )
    return distance.sum(dim=2).mean()


def balanced_chamfer(first, second):
    distances = torch.cdist(first, second)
    return 0.5 * (
        distances.min(dim=2).values.mean(dim=1)
        + distances.min(dim=1).values.mean(dim=1)
    ).mean()


def one_sided_masked_distance(source, target, source_mask):
    """Mean target-to-valid-source distance used by the reference objective."""
    valid_batch = source_mask.bool().sum(dim=1) > 0
    if not valid_batch.any():
        return target.sum() * 0.0
    source = source[valid_batch]
    target = target[valid_batch]
    source_mask = source_mask[valid_batch].bool()
    distances = torch.cdist(source, target)
    distances = distances.masked_fill(~source_mask.unsqueeze(-1), float("inf"))
    return distances.min(dim=1).values.mean()


def farthest_point_sample(points, count):
    indices = pointnet2_utils.furthest_point_sample(
        points.contiguous(), int(count)
    )
    return pointnet2_utils.gather_operation(
        points.transpose(1, 2).contiguous(), indices
    ).transpose(1, 2).contiguous()


def symmetry_adjusted_rotations(prediction, target, category, symmetry_ids):
    """Apply the reference y-axis projection without mutating input tensors."""
    category = category.reshape(-1)
    symmetric = torch.zeros_like(category, dtype=torch.bool)
    for category_id in symmetry_ids:
        symmetric |= category == int(category_id)
    adjusted_prediction = prediction.clone()
    adjusted_target = target.clone()
    if symmetric.any():
        y_axis = prediction.new_tensor([0.0, 1.0, 0.0]).view(1, 3, 1)
        predicted_axis = prediction[symmetric] @ y_axis
        target_axis = target[symmetric] @ y_axis
        adjusted_prediction[symmetric] = predicted_axis.expand(-1, -1, 3)
        adjusted_target[symmetric] = target_axis.expand(-1, -1, 3)
    return adjusted_prediction, adjusted_target


class PriorPoseLoss(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.pose_weight = float(cfg.pose)
        self.rotation_extra_weight = float(cfg.rotation_extra)
        self.nocs_weight = float(cfg.nocs)
        self.canonical_weight = float(cfg.canonical)
        self.reconstruction_weight = float(cfg.reconstruction)
        self.diversity_weight = float(cfg.diversity)
        self.camera_pose_weight = float(cfg.camera_pose)
        self.camera_shape_weight = float(cfg.camera_shape)
        self.camera_symmetry_adjusted_rotation = bool(
            cfg.camera_symmetry_adjusted_rotation
        )
        self.nocs_symmetry_adjusted_rotation = bool(
            getattr(cfg, "nocs_symmetry_adjusted_rotation", False)
        )
        self.diversity_threshold = float(cfg.diversity_threshold)
        self.symmetry_ids = list(cfg.symmetry_ids)

    def _diversity(self, reconstruction):
        point_count = reconstruction.shape[1]
        distances = torch.cdist(reconstruction, reconstruction)
        distances = distances + torch.eye(
            point_count, device=distances.device, dtype=distances.dtype
        ).unsqueeze(0)
        distances = distances.clamp(max=self.diversity_threshold)
        penalties = -distances / self.diversity_threshold + 1.0
        return (
            penalties.sum(dim=(1, 2)) / (point_count * (point_count - 1))
        ).mean()

    def forward(self, endpoints):
        points = endpoints["pts"]
        predicted_points = endpoints["pred_kpt_3d"]
        reconstruction = endpoints["recon_model_instance"]
        model = endpoints["model"]
        translation_gt = endpoints["translation_label"]
        rotation_gt = endpoints["rotation_label"]
        size_gt = endpoints["size_label"]
        point_mask = endpoints["pc_mask"].bool()

        pose_rotation, pose_rotation_gt = symmetry_adjusted_rotations(
            endpoints["pred_rotation"],
            rotation_gt,
            endpoints["category_label"],
            self.symmetry_ids,
        )
        rotation_loss = torch.norm(
            pose_rotation - pose_rotation_gt, dim=1
        ).mean()
        rotation_extra_loss = self.rotation_extra_weight * rotation_loss
        translation_loss = torch.norm(
            endpoints["pred_translation"] - translation_gt, dim=1
        ).mean()
        size_loss = torch.norm(
            endpoints["pred_size"] - size_gt, dim=1
        ).mean()
        pose_loss = 3.0 * (
            0.2 * rotation_loss
            + 0.6 * translation_loss
            + 0.2 * size_loss
        )

        canonical_loss = one_sided_masked_distance(
            endpoints["pred_kpt_nocs"], reconstruction, point_mask
        )
        canonical_target = (
            predicted_points.detach() - translation_gt.unsqueeze(1)
        ) / (torch.norm(size_gt, dim=1).view(-1, 1, 1) + 1e-8)
        nocs_rotation = (
            pose_rotation_gt
            if self.nocs_symmetry_adjusted_rotation
            else rotation_gt
        )
        canonical_target = canonical_target @ nocs_rotation
        nocs_loss = smooth_l1_distance(
            endpoints["pred_kpt_nocs"], canonical_target
        )
        diversity_loss = self._diversity(reconstruction)
        downsampled_model = farthest_point_sample(
            model, reconstruction.shape[1]
        )
        reconstruction_loss = balanced_chamfer(
            downsampled_model, reconstruction
        )

        # Select the configured rotation target without modifying predictions.
        camera_rotation = (
            pose_rotation
            if self.camera_symmetry_adjusted_rotation
            else endpoints["pred_rotation"]
        )
        camera_model = torch.bmm(
            model.detach(), camera_rotation.transpose(1, 2)
        )
        isotropic_scale = torch.norm(
            endpoints["pred_size"], dim=1, keepdim=True
        )
        camera_model = (
            camera_model * isotropic_scale.unsqueeze(1)
            + endpoints["pred_translation"].unsqueeze(1)
        )
        camera_pose_loss = one_sided_masked_distance(
            points, camera_model, point_mask
        )

        camera_shape_model = torch.bmm(
            model, camera_rotation.detach().transpose(1, 2)
        )
        camera_shape_model = (
            camera_shape_model * isotropic_scale.detach().unsqueeze(1)
            + endpoints["pred_translation"].detach().unsqueeze(1)
        )
        camera_shape_loss = one_sided_masked_distance(
            points, camera_shape_model, point_mask
        )

        total = (
            self.pose_weight * pose_loss
            + rotation_extra_loss
            + self.nocs_weight * nocs_loss
            + self.canonical_weight * canonical_loss
            + self.diversity_weight * diversity_loss
            + self.reconstruction_weight * reconstruction_loss
            + self.camera_pose_weight * camera_pose_loss
            + self.camera_shape_weight * camera_shape_loss
        )
        return {
            "loss_all": total,
            "loss_pose": self.pose_weight * pose_loss,
            "loss_rotation_extra": rotation_extra_loss,
            "loss_nocs": self.nocs_weight * nocs_loss,
            "loss_canonical": self.canonical_weight * canonical_loss,
            "loss_diversity": self.diversity_weight * diversity_loss,
            "loss_reconstruction": self.reconstruction_weight
            * reconstruction_loss,
            "loss_camera_pose": self.camera_pose_weight * camera_pose_loss,
            "loss_camera_shape": self.camera_shape_weight * camera_shape_loss,
            "loss_rotation": rotation_loss,
            "loss_translation": translation_loss,
            "loss_size": size_loss,
        }
