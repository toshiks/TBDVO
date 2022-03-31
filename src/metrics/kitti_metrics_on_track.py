# pylint: skip-file

import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from torchmetrics import Metric

from src.metrics.kitti_odometry_benchmark import KittiEvalOdom
from src.utils.odometry_to_track import odometry2track, to_homogeneous_q


class KittiMetricsOnTrack(Metric):
    """Kitti metrics."""

    def __init__(self, first_pose: torch.Tensor, track_name: str = ""):
        super().__init__(compute_on_step=False, dist_sync_on_step=False)

        self._track_name = track_name
        self._first_pose = first_pose

        self.preds_pose: torch.Tensor
        self.target_pose: torch.Tensor

        self.indices: torch.Tensor

        self.add_state("preds_pose", default=[], dist_reduce_fx="cat")
        self.add_state("target_pose", default=[], dist_reduce_fx="cat")
        self.add_state("indices", default=[], dist_reduce_fx="cat")

        self._eval_odom = KittiEvalOdom()

        self._odometry2track = odometry2track
        self._to_homogeneous = to_homogeneous_q

    def update(
            self, preds_pose: torch.Tensor, target_pose: torch.Tensor, batch_index: torch.Tensor
    ):
        """Update all parameters on one step."""
        self.preds_pose.append(preds_pose)
        self.target_pose.append(target_pose)
        self.indices.append(batch_index)

    @staticmethod
    def _plot_tracks(predict_coord: np.ndarray, target_coord: np.ndarray, track_id: str):
        target_coord_min = np.min(target_coord, axis=0)
        target_coord_max = np.max(target_coord, axis=0)

        ax = plt.subplot()
        plt.plot(target_coord[:, 0], target_coord[:, 2], label="target", c='r')
        plt.plot(predict_coord[:, 0], predict_coord[:, 2], label="predict", c='b')
        plt.xlabel("x (m)")
        plt.ylabel("z (m)")

        diff_x = (target_coord_max[0] - target_coord_min[0]) / 10.
        diff_z = (target_coord_max[2] - target_coord_min[2]) / 10.
        plt.xlim((target_coord_min[0] - diff_x, target_coord_max[0] + diff_x))
        plt.ylim((target_coord_min[2] - diff_z, target_coord_max[2] + diff_z))
        plt.legend()
        plt.title(f"KITTI TRACK: {track_id}")
        image = wandb.Image(ax, caption=f"KITTI TRACK: {track_id}")
        plt.cla()
        plt.clf()
        return image

    def _compute(self):
        first_pose = self._first_pose.to(self.preds_pose.device).unsqueeze(0)

        preds_track = self._odometry2track(self.preds_pose, first_pose)
        target_track = self._odometry2track(self.target_pose, first_pose)

        preds_track = preds_track.cpu().numpy()[0]
        target_track = target_track.cpu().numpy()[0]

        track_plot = self._plot_tracks(preds_track[:, :3, -1], target_track[:, :3, -1], self._track_name)
        metrics, plots = self._eval_odom(
            preds_track, target_track,
            self._track_name
        )

        return metrics, plots, track_plot

    def __sort_by_indices(self):
        indices_data = np.argsort(self.indices.cpu().numpy())

        print(self._track_name)
        print(indices_data, self.indices.cpu().numpy())

        print(self.target_pose.shape)
        print(self.preds_pose.shape)

        self.preds_pose = self.preds_pose[indices_data].permute(1, 0, 2, 3)
        self.target_pose = self.target_pose[indices_data].permute(1, 0, 2, 3)

        print(self.target_pose.shape)
        print(self.preds_pose.shape)

    def compute(self):
        """Compute all metrics."""
        self.__sort_by_indices()
        metric, [tr_plot, rot_plot], track_plot = self._compute()
        return metric, track_plot, tr_plot, rot_plot
