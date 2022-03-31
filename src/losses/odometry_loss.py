from typing import Mapping

import torch
import torch.nn as nn

from src.losses.geodesic_loss import GeodesicLoss
from src.losses.loss_with_parameters import LossWithParameters


class OdometryLoss(LossWithParameters):
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

    @property
    def params(self) -> Mapping[str, nn.Parameter]:
        if not isinstance(self._s_x, nn.Parameter):
            return {}

        return {"odometry_loss/coord": self._s_x, "odometry_loss/angles": self._s_q}

    def forward(self, odometry, odometry_gt):
        """Calculate loss function.

        Args:
            odometry: pair of predicted tensors of odometry (coords, quaternions)
            odometry_gt:  pair of target tensor of odometry (coords, quaternions)

        Returns:
            calculated value
        """
        odometry_x = odometry[:, :, :3, -1]
        odometry_gt_x = odometry_gt[:, :, :3, -1]

        loss_odom_x = self.mse(odometry_x, odometry_gt_x) + self.mae(odometry_x, odometry_gt_x)
        loss_odom_q = self.angle_loss(odometry[:, :, :3, :3], odometry_gt[:, :, :3, :3])

        if self._is_trainable:
            return loss_odom_x * torch.exp(-self._s_x) + self._s_x + loss_odom_q * torch.exp(-self._s_q) + self._s_q

        return loss_odom_x * self._s_x + loss_odom_q * self._s_q
