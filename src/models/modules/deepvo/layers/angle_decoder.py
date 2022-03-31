from typing import Tuple

import torch
import torch.nn as nn
from torch.nn import functional as F


class VOPoseDecoder(nn.Module):
    def __init__(self, input_dim: int, coord_dim: int, angle_dim: int):
        super().__init__()

        self._linear_coord = nn.Linear(input_dim, coord_dim)
        self._linear_angles = nn.Linear(input_dim, angle_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._linear_coord(x), self._linear_angles(x)


class VOPoseQuaternionDecoder(VOPoseDecoder):
    def __init__(self, input_dim: int):
        super().__init__(input_dim=input_dim, coord_dim=3, angle_dim=4)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        coords, angles = super().forward(x)
        return coords, F.normalize(angles, dim=-1)


class VOPoseConstrainQuaternionDecoder(VOPoseDecoder):
    def __init__(self, input_dim: int):
        super().__init__(input_dim=input_dim, coord_dim=3, angle_dim=4)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        coords, angles = super().forward(x)
        qw = F.relu(angles[..., :1]).contiguous()
        qx = angles[..., 1:2].contiguous()
        qy = angles[..., 2:3].contiguous()
        qz = angles[..., 3:].contiguous()

        return coords, F.normalize(torch.cat([qw, qx, qy, qz], dim=-1), dim=-1)
