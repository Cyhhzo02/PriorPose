"""Continuous 6-D rotation conversion."""

import torch


def _normalize(vector):
    return vector / vector.norm(dim=1, keepdim=True).clamp_min(1e-8)


def Ortho6d2Mat(x_raw, y_raw):
    """Convert two 3-D vectors into the reference orthonormal rotation basis."""
    y_axis = _normalize(y_raw)
    z_axis = _normalize(torch.cross(x_raw, y_axis, dim=1))
    x_axis = torch.cross(y_axis, z_axis, dim=1)
    return torch.stack([x_axis, y_axis, z_axis], dim=2)
