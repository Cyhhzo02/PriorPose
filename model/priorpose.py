"""Released PriorPose network for NOCS and HouseCat6D."""

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.backbone import PointNet2MSG
from model.heads import NOCSPredictor, PoseSizeEstimator
from model.seeded_transformer import TwoStreamSeededTransformer
from utils.augmentation import generate_pose_augmentation


class PriorPose(nn.Module):
    """RGB-D category-level pose estimation with joint prior reconstruction.

    This class is the single released implementation.  Dataset configs select
    the positional frame and Transformer parameter-sharing recipe;
    all other PriorPose components are shared.
    """

    def __init__(self, cfg):
        super().__init__()
        self.category_count = int(cfg.category_count)
        self.prior_point_count = int(cfg.prior_point_count)
        self.point_position_frame = str(cfg.point_position_frame).lower()
        self.use_input_normalization = bool(
            getattr(cfg, "input_normalization", True)
        )
        if self.point_position_frame not in {
            "camera_raw",
            "centered_augmented",
        }:
            raise ValueError(
                "point_position_frame must be camera_raw or "
                "centered_augmented"
            )
        if self.prior_point_count != 512:
            raise ValueError("the released model expects 512-point mean shapes")

        dino_repository = Path(cfg.dino_repository).expanduser().resolve()
        if not dino_repository.is_dir():
            raise FileNotFoundError(
                f"DINOv2 repository not found: {dino_repository}. "
                "Run scripts/setup_dinov2.sh first."
            )
        self.rgb_extractor = torch.hub.load(
            str(dino_repository), "dinov2_vits14", source="local"
        )
        for parameter in self.rgb_extractor.parameters():
            parameter.requires_grad = False
        self.rgb_projection = nn.Conv1d(384, 128, 1)
        self.point_extractor = PointNet2MSG()

        self.position_encoder = nn.Sequential(
            nn.Conv1d(3, 128, 1), nn.ReLU(), nn.Conv1d(128, 128, 1)
        )
        self.prior_position_encoder = nn.Sequential(
            nn.Conv1d(3, 128, 1), nn.ReLU(), nn.Conv1d(128, 256, 1)
        )
        self.observation_token_position = nn.Sequential(
            nn.Linear(3, 128), nn.ReLU(), nn.Linear(128, 256)
        )
        self.observation_fusion = nn.Sequential(
            nn.Conv1d(384, 256, 1),
            nn.ReLU(),
            nn.Conv1d(256, 256, 1),
            nn.ReLU(),
            nn.Conv1d(256, 256, 1),
        )
        self.rgb_norm = nn.LayerNorm(128)
        self.point_norm = nn.LayerNorm(128)
        self.prior_norm = nn.LayerNorm(256)
        self.input_norm = nn.LayerNorm(256)

        self.prior_features = nn.Parameter(
            torch.randn(self.category_count, self.prior_point_count, 256) * 0.2
        )
        self.camera_token = nn.Parameter(torch.empty(1, 256))
        self.nocs_token = nn.Parameter(torch.empty(1, 256))
        self.deformation_token = nn.Parameter(torch.empty(1, 256))
        nn.init.xavier_normal_(self.camera_token)
        nn.init.xavier_normal_(self.nocs_token)
        nn.init.xavier_normal_(self.deformation_token)

        transformer = cfg.transformer
        self.transformer = TwoStreamSeededTransformer(
            blocks=int(transformer.blocks),
            dim=256,
            heads=int(transformer.heads),
            hidden_dim=int(transformer.hidden_dim),
            observation_seeds=int(transformer.observation_seeds),
            prior_seeds=int(transformer.prior_seeds),
            neighbors=int(transformer.neighbors),
            score_bias_gamma=float(transformer.score_bias_gamma),
            share_parameters=bool(transformer.share_parameters),
        )
        self.nocs_predictor = NOCSPredictor(bins=64, dim=256, blocks=2, heads=4)
        self.deformation_predictor = NOCSPredictor(
            bins=64, dim=256, blocks=2, heads=4
        )
        self.pose_estimator = PoseSizeEstimator(dim=256)

    def _image_features(self, rgb, choose):
        features = self.rgb_extractor.forward_features(rgb)["x_prenorm"][:, 1:]
        batch, patch_count, feature_dim = features.shape
        patch_side = int(patch_count**0.5)
        if patch_side * patch_side != patch_count:
            raise RuntimeError("DINOv2 patch tokens do not form a square grid")
        features = features.reshape(
            batch, patch_side, patch_side, feature_dim
        ).permute(0, 3, 1, 2)
        features = F.interpolate(
            features,
            size=(patch_side * 14, patch_side * 14),
            mode="bilinear",
            align_corners=False,
        ).reshape(batch, feature_dim, -1)
        features = self.rgb_projection(features)
        gather_index = choose.unsqueeze(1).expand(-1, features.shape[1], -1)
        return torch.gather(features, 2, gather_index).contiguous()

    def forward(self, inputs, epoch=None):
        del epoch
        rgb = inputs["rgb"]
        raw_points = inputs["pts"]
        choose = inputs["choose"].long()
        categories = inputs["category_label"].reshape(-1).long()
        prior = inputs["prior"]
        if prior.shape[1] != self.prior_point_count:
            raise ValueError(
                f"expected {self.prior_point_count} prior points, got {prior.shape[1]}"
            )

        batch = raw_points.shape[0]
        center = raw_points.mean(dim=1, keepdim=True)
        centered_points = raw_points - center
        points = centered_points
        if self.training:
            delta_rotation, delta_translation, delta_scale = (
                generate_pose_augmentation(batch, raw_points.device, raw_points.dtype)
            )
            points = (
                (points - delta_translation) / delta_scale.unsqueeze(2)
            ) @ delta_rotation

        image_features = self._image_features(rgb, choose)
        point_features = self.point_extractor(points)
        position_points = (
            raw_points
            if self.point_position_frame == "camera_raw"
            else points
        )
        position_features = self.position_encoder(
            position_points.transpose(1, 2)
        )
        image_features = self.rgb_norm(
            image_features.transpose(1, 2)
        ).transpose(1, 2)
        point_features = self.point_norm(
            point_features.transpose(1, 2)
        ).transpose(1, 2)
        observation_features = self.observation_fusion(
            torch.cat([point_features, image_features, position_features], dim=1)
        ).transpose(1, 2)

        prior_position = self.prior_position_encoder(
            prior.transpose(1, 2)
        ).transpose(1, 2)
        prior_features = self.prior_norm(self.prior_features[categories])
        prior_features = prior_features + prior_position

        observation_tokens = (
            observation_features
            + self.observation_token_position(points.detach())
            + self.nocs_token.expand(batch, observation_features.shape[1], -1)
        )
        prior_tokens = prior_features + self.deformation_token.expand(
            batch, prior_features.shape[1], -1
        )
        tokens = torch.cat(
            [
                self.camera_token.expand(batch, 1, -1),
                observation_tokens,
                prior_tokens,
            ],
            dim=1,
        )
        transformer_tokens = (
            self.input_norm(tokens)
            if self.use_input_normalization
            else tokens
        )
        tokens = self.transformer(transformer_tokens, points, prior)
        observation_count = points.shape[1]
        observation_features = tokens[:, 1 : 1 + observation_count]
        prior_features = tokens[:, 1 + observation_count :]

        # Add geometric position features before the prediction heads.
        observation_features = (
            observation_features + self.observation_token_position(points.detach())
        )
        prior_features = prior_features + self.prior_position_encoder(
            prior.transpose(1, 2)
        ).transpose(1, 2)

        predicted_nocs, _, _ = self.nocs_predictor(
            points, observation_features
        )
        deformation, _, _ = self.deformation_predictor(prior, prior_features)
        reconstructed_instance = prior + deformation
        rotation, translation, size = self.pose_estimator(
            points, predicted_nocs.detach(), observation_features
        )

        if self.training:
            return {
                "pred_kpt_3d": (
                    points @ delta_rotation.transpose(1, 2)
                ) * delta_scale.unsqueeze(2)
                + delta_translation
                + center,
                "pred_kpt_nocs": predicted_nocs,
                "pred_translation": delta_translation.squeeze(1)
                + delta_scale
                * torch.bmm(delta_rotation, translation.unsqueeze(2)).squeeze(2)
                + center.squeeze(1),
                "pred_rotation": delta_rotation @ rotation,
                "pred_size": size * delta_scale,
                "recon_model_instance": reconstructed_instance,
            }
        return {
            "pred_translation": translation + center.squeeze(1),
            "pred_rotation": rotation,
            "pred_size": size,
            "pred_kpt_3d": points + center,
            "pred_kpt_nocs": predicted_nocs,
            "recon_model_instance": reconstructed_instance,
        }
