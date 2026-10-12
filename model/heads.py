"""Minimal prediction heads used by the released PriorPose model."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.rotation_utils import Ortho6d2Mat


class SelfAttentionBlock(nn.Module):
    def __init__(self, dim=256, heads=4, ffn_dim=256, dropout=0.0):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            dim, heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, dim),
        )

    def forward(self, tokens):
        normalized = self.norm1(tokens)
        update, _ = self.attention(
            normalized, normalized, normalized, need_weights=False
        )
        tokens = tokens + self.dropout1(update)
        return tokens + self.dropout2(self.ffn(self.norm2(tokens)))


class SelfAttentionStack(nn.Module):
    def __init__(self, blocks=2, dim=256, heads=4, ffn_dim=256):
        super().__init__()
        self.blocks = nn.ModuleList(
            [SelfAttentionBlock(dim, heads, ffn_dim) for _ in range(blocks)]
        )

    def forward(self, tokens):
        for block in self.blocks:
            tokens = block(tokens)
        return tokens


class NOCSPredictor(nn.Module):
    """Class-agnostic binned coordinate head used for NOCS and deformation."""

    def __init__(self, bins=64, dim=256, blocks=2, heads=4):
        super().__init__()
        self.bins = bins
        bin_length = 1.0 / bins
        centers = torch.linspace(-0.5, 0.5 - bin_length, bins).view(bins, 1)
        self.register_buffer(
            "bins_center", centers + bin_length / 2.0, persistent=False
        )
        self.feature_mlp = nn.Sequential(
            nn.Conv1d(dim * 2, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Conv1d(512, dim, 1),
            nn.BatchNorm1d(dim),
        )
        self.self_attention = SelfAttentionStack(blocks, dim, heads, dim)
        self.logit_head = nn.Sequential(
            nn.Conv1d(dim, 512, 1),
            nn.ReLU(),
            nn.Conv1d(512, 512, 1),
            nn.ReLU(),
            nn.Conv1d(512, 3 * bins, 1),
        )
        self.position_encoder = nn.Sequential(
            nn.Conv1d(3, 64, 1),
            nn.ReLU(),
            nn.Conv1d(64, 128, 1),
            nn.ReLU(),
            nn.Conv1d(128, dim, 1),
        )
        self.score_head = nn.Sequential(
            nn.Linear(dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, xyz, features):
        batch, count, _ = features.shape
        position = self.position_encoder(xyz.transpose(1, 2)).transpose(1, 2)
        features = self.feature_mlp(
            torch.cat([features, position], dim=-1).transpose(1, 2)
        ).transpose(1, 2)
        features = self.self_attention(features)
        logits = self.logit_head(features.transpose(1, 2))
        logits = logits.view(batch, 3, self.bins, count)
        probabilities = F.softmax(logits, dim=2).permute(0, 3, 1, 2)
        coordinates = torch.matmul(probabilities, self.bins_center).squeeze(-1)
        return coordinates, features, self.score_head(features)


class PoseSizeEstimator(nn.Module):
    def __init__(self, dim=256):
        super().__init__()
        self.observed_encoder = nn.Sequential(
            nn.Conv1d(3, 32, 1), nn.ReLU(), nn.Conv1d(32, 64, 1), nn.ReLU()
        )
        self.canonical_encoder = nn.Sequential(
            nn.Conv1d(3, 32, 1), nn.ReLU(), nn.Conv1d(32, 64, 1), nn.ReLU()
        )
        self.local_fusion = nn.Sequential(
            nn.Conv1d(128 + dim, 512, 1),
            nn.ReLU(),
            nn.Conv1d(512, 256, 1),
            nn.ReLU(),
        )
        self.global_fusion = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.ReLU(),
            nn.Conv1d(512, 512, 1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.rotation_head = nn.Sequential(
            nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, 6)
        )
        self.translation_head = nn.Sequential(
            nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, 3)
        )
        self.size_head = nn.Sequential(
            nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, 3)
        )

    def forward(self, observed, canonical, local_features):
        observed = self.observed_encoder(observed.transpose(1, 2))
        canonical = self.canonical_encoder(canonical.transpose(1, 2))
        fused = self.local_fusion(
            torch.cat([observed, local_features.transpose(1, 2), canonical], dim=1)
        )
        global_feature = fused.mean(dim=2, keepdim=True).expand_as(fused)
        global_feature = self.global_fusion(
            torch.cat([fused, global_feature], dim=1)
        ).squeeze(2)
        rotation_6d = self.rotation_head(global_feature)
        rotation = Ortho6d2Mat(
            rotation_6d[:, :3].contiguous(), rotation_6d[:, 3:].contiguous()
        ).view(-1, 3, 3)
        return (
            rotation,
            self.translation_head(global_feature),
            self.size_head(global_feature),
        )
