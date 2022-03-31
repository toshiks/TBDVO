from typing import Mapping

import torch
import torch.nn as nn

from src.losses.geodesic_loss import GeodesicLoss
from src.losses.loss_with_parameters import LossWithParameters
from src.utils.odometry_to_track import odometry2track


class GeometricLoss(LossWithParameters):
    """Loss by article.

    Link: https://arxiv.org/abs/1803.03642
    """

    def __init__(
            self, is_trainable: bool = False, coef_coord: float = 10.0, coef_angle: float = 1000.0
    ):
        """Init loss with params."""
        super().__init__()
        self._s_x = coef_coord
        self._s_q = coef_angle
        self._is_trainable = is_trainable

        if is_trainable:
            self._s_x = nn.Parameter(torch.tensor(coef_coord))
            self._s_q = nn.Parameter(torch.tensor(coef_angle))

        self.mse = nn.MSELoss()
        self.mae = nn.L1Loss()

        self.angle_loss = GeodesicLoss()
        self.odometry2track = odometry2track

    @property
    def params(self) -> Mapping[str, nn.Parameter]:
        if not isinstance(self._s_x, nn.Parameter):
            return {}

        return {"geometry_loss/coord": self._s_x, "geometry_loss/angles": self._s_q}

    def forward(self, odometry, odometry_gt):
        """Calculate loss function.

        Args:
            odometry: pair of predicted tensors of odometry (coords, quaternions)
            odometry_gt:  pair of target tensor of odometry (coords, quaternions)

        Returns:
            calculated value

        """
        n = odometry.shape[0]

        first_pose = torch.tile(torch.eye(4, dtype=odometry.dtype, device=odometry.device).unsqueeze(0), (n, 1, 1))

        poses = self.odometry2track(odometry, first_pose)
        poses_gt = self.odometry2track(odometry_gt, first_pose)

        poses_x = poses[:, :, :3, -1]
        poses_gt_x = poses_gt[:, :, :3, -1]

        loss_linear_x = self.mse(poses_x, poses_gt_x) + self.mae(poses_x, poses_gt_x)
        loss_linear_q = self.angle_loss(poses[:, :, :3, :3], poses_gt[:, :, :3, :3])

        if self._is_trainable:
            return (loss_linear_x * torch.exp(-self._s_x) + self._s_x +
                    loss_linear_q * torch.exp(-self._s_q) + self._s_q)

        return loss_linear_x * self._s_x + loss_linear_q * self._s_q
