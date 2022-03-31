# https://github.com/JHUVisionLab/multi-modal-regression/blob/master
import torch
from torch import nn


def _compute_geodesic_distance_from_two_matrices(m1: torch.Tensor, m2: torch.Tensor, eps: float) -> torch.Tensor:
    m = torch.matmul(m1, m2.transpose(2, 3))  # batch*3*3

    cos = (m[:, :, 0, 0] + m[:, :, 1, 1] + m[:, :, 2, 2] - 1.) / 2.
    cos = torch.clamp(cos, -1 + eps, 1 - eps)

    theta = torch.acos(cos)

    return torch.mean(theta)


class GeodesicLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self._eps = 1e-6

    def forward(self, ypred: torch.Tensor, ytrue: torch.Tensor) -> torch.Tensor:
        return _compute_geodesic_distance_from_two_matrices(ypred, ytrue, self._eps)
