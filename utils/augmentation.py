"""Training-time pose perturbation used by PriorPose."""

import torch


def generate_pose_augmentation(batch_size, device, dtype=torch.float32):
    translation = torch.empty(batch_size, 1, 3, device=device, dtype=dtype)
    translation.uniform_(-0.02, 0.02)

    # The reference implementation draws angles on CPU; keeping that behavior
    # preserves its seeded random-number streams.
    angles = torch.empty(batch_size, 3).uniform_(-20.0, 20.0)
    angles = angles / 180.0 * torch.pi
    rotation_x = torch.eye(3).repeat(batch_size, 1, 1)
    rotation_y = torch.eye(3).repeat(batch_size, 1, 1)
    rotation_z = torch.eye(3).repeat(batch_size, 1, 1)
    rotation_x[:, 1, 1] = torch.cos(angles[:, 0])
    rotation_x[:, 1, 2] = -torch.sin(angles[:, 0])
    rotation_x[:, 2, 1] = torch.sin(angles[:, 0])
    rotation_x[:, 2, 2] = torch.cos(angles[:, 0])
    rotation_y[:, 0, 0] = torch.cos(angles[:, 1])
    rotation_y[:, 0, 2] = torch.sin(angles[:, 1])
    rotation_y[:, 2, 0] = -torch.sin(angles[:, 1])
    rotation_y[:, 2, 2] = torch.cos(angles[:, 1])
    rotation_z[:, 0, 0] = torch.cos(angles[:, 2])
    rotation_z[:, 0, 1] = -torch.sin(angles[:, 2])
    rotation_z[:, 1, 0] = torch.sin(angles[:, 2])
    rotation_z[:, 1, 1] = torch.cos(angles[:, 2])
    rotation = torch.bmm(torch.bmm(rotation_x, rotation_y), rotation_z)
    rotation = rotation.to(device=device, dtype=dtype)

    scale = torch.empty(batch_size, 1, device=device, dtype=dtype)
    scale.uniform_(0.8, 1.2)
    return rotation, translation, scale
