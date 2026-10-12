"""PointNet++ feature backbone used by PriorPose."""

import torch.nn as nn

from pointnet2_modules import PointnetFPModule, PointnetSAModuleMSG


class PointNet2MSG(nn.Module):
    """Per-point 128-D features from a four-level PointNet++ MSG encoder."""

    def __init__(self):
        super().__init__()
        radii = [[0.01, 0.02], [0.02, 0.04], [0.04, 0.08], [0.08, 0.16]]
        channels = [0, 64, 128, 256, 512]
        widths = [16, 32, 64, 128]
        self.sa_modules = nn.ModuleList()
        for level, (npoint, radius) in enumerate(zip([512, 256, 128, 64], radii)):
            width = widths[level]
            self.sa_modules.append(
                PointnetSAModuleMSG(
                    npoint=npoint,
                    radii=radius,
                    nsamples=[16, 32],
                    mlps=[
                        [channels[level], width, width, width * 2],
                        [channels[level], width, width, width * 2],
                    ],
                    use_xyz=True,
                    bn=True,
                )
            )

        self.fp_modules = nn.ModuleList(
            [
                PointnetFPModule(mlp=[256, 128, 128], bn=True),
                PointnetFPModule(mlp=[320, 256, 256], bn=True),
                PointnetFPModule(mlp=[640, 256, 256], bn=True),
                PointnetFPModule(mlp=[768, 512, 512], bn=True),
            ]
        )

    def forward(self, pointcloud):
        xyz = pointcloud[..., :3].contiguous()
        features = None
        xyz_levels, feature_levels = [xyz], [features]
        for level, module in enumerate(self.sa_modules):
            next_xyz, next_features = module(
                xyz_levels[level], feature_levels[level]
            )
            xyz_levels.append(next_xyz)
            feature_levels.append(next_features)

        for level in range(-1, -len(self.fp_modules) - 1, -1):
            feature_levels[level - 1] = self.fp_modules[level](
                xyz_levels[level - 1],
                xyz_levels[level],
                feature_levels[level - 1],
                feature_levels[level],
            )
        return feature_levels[0]
