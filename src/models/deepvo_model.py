from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR

from src.losses import GeometricLoss, OdometryLoss
from src.metrics import KittiMetricsOnTrack
from src.metrics.kitti_odometry_benchmark import MetricType
from src.utils.telegram_bot import Bot


class DeepVOModel(pl.LightningModule):
    """Train module for SixDof model."""

    def __init__(
            self,
            torch_model: torch.nn.Module,
            lr: float = 0.0004,
            betas: Tuple[float, float] = (0.99, 0.999),
            **kwargs
    ):
        """Init train by config."""
        super().__init__()

        self.save_hyperparameters()

        self._model = torch_model

        self._vo_loss: Optional[OdometryLoss] = None
        self._geo_loss: Optional[GeometricLoss] = None
        self._metrics: List[KittiMetricsOnTrack] = []

        self._bot_getter: Optional[Callable[[], Bot]] = None

        self._setup_losses()

    def on_fit_start(self):
        for val_dataset in self.trainer.datamodule.val_datasets:
            first_point_pose = torch.tensor(val_dataset.first_pose, dtype=self.dtype)

            self._metrics.append(KittiMetricsOnTrack(
                first_point_pose, val_dataset.name
            ))

    @property
    def bot_getter(self):
        """Get bot."""
        return self._bot_getter

    @bot_getter.setter
    def bot_getter(self, bot_getter: Callable[[], Bot]):
        """Set telegram bot"""
        self._bot_getter = bot_getter

    def _setup_losses(self):
        if self.hparams.odometry_loss.need:
            self._vo_loss = OdometryLoss(
                is_trainable=self.hparams.odometry_loss.is_trainable,
                coef_coord=self.hparams.odometry_loss.coef_coord,
                coef_angle=self.hparams.odometry_loss.coef_angle,
            )

        if self.hparams.geometric_loss.need:
            self._geo_loss = GeometricLoss(
                is_trainable=self.hparams.geometric_loss.is_trainable,
                coef_coord=self.hparams.geometric_loss.coef_coord,
                coef_angle=self.hparams.geometric_loss.coef_angle,
            )

        if self._vo_loss is None and self._geo_loss is None:
            raise AttributeError("One of the losses should be added to model.")

    def configure_optimizers(self):
        """Setup Adam optimizer."""
        optimizer = AdamW(
            self.parameters(), lr=self.hparams.lr, betas=self.hparams.betas
        )

        schedulers = [
            {
                "scheduler": StepLR(optimizer, step_size=self.hparams.epoch_count_optimizer, gamma=0.8),
                "interval": "epoch",
                "frequency": 1,
            }
        ]
        return [optimizer], schedulers

    def forward(self, batch: Dict) -> torch.Tensor:
        images = batch["images"]

        predicted_poses = self._model(images)

        return predicted_poses

    def _calc_loss_on_predicts(self, batch, predicted_poses):
        loss = 0
        if self._vo_loss:
            loss = loss + self._vo_loss(predicted_poses, batch["odometry"])
        if self._geo_loss:
            loss = loss + self._geo_loss(predicted_poses, batch["odometry"])

        return loss

    def training_step(self, batch, batch_idx):
        """Train loop for one batch."""
        predicted_poses = self.forward(batch)
        loss = self._calc_loss_on_predicts(batch, predicted_poses)

        self.log("train/loss", loss, sync_dist=True, add_dataloader_idx=False)

        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        """Validation loop for one batch."""
        predicted_poses = self.forward(batch)
        loss = self._calc_loss_on_predicts(batch, predicted_poses)

        metric = self._metrics[dataloader_idx]

        if batch["first_index"][0] == 0:
            reshaped_predicted_poses = predicted_poses[0].unsqueeze(1)
            reshaped_target_poses = batch["odometry"][0].unsqueeze(1)

            additional_indices = torch.range(
                -reshaped_predicted_poses.shape[0] + 1, 0,
                dtype=batch["first_index"].dtype,
                device=batch["first_index"].device,
            )

            metric(
                torch.cat([reshaped_predicted_poses, predicted_poses[1:, -1:]]),
                torch.cat([reshaped_target_poses, batch["odometry"][1:, -1:]]),
                torch.cat([additional_indices, batch["first_index"][1:]]),
            )
        else:
            metric(
                predicted_poses[:, -1:],
                batch["odometry"][:, -1:],
                batch["first_index"]
            )

        self.log("val/loss", loss)
        return loss

    def validation_epoch_end(self, outputs):
        """Validation epoch finishing."""
        mean_loss = torch.tensor([
            torch.mean(torch.tensor(dataloader_results)) for dataloader_results in outputs
        ]).mean()

        track_plots = []
        trans_err_plots = []
        rot_err_plots = []
        metrics = {}

        for metric in self._metrics:
            metric_dict, track_plot, tr_plot, rot_plot = metric.compute()
            track_plots.append(track_plot)
            trans_err_plots.append(tr_plot)
            rot_err_plots.append(rot_plot)
            metrics.update(metric_dict)

        for metric in self._metrics:
            metric.reset()

        trans_err = []
        rot_err = []

        for key, value in metrics.items():
            if MetricType.TRANS_ERROR.value in key:
                trans_err.append(value)
            elif MetricType.ROT_ERROR.value in key:
                rot_err.append(value)

        self.log(f"{MetricType.TRANS_ERROR.value}/AVG", np.mean(trans_err), sync_dist=True, add_dataloader_idx=False)
        self.log(f"{MetricType.ROT_ERROR.value}/AVG", np.mean(rot_err), sync_dist=True, add_dataloader_idx=False)

        if self.global_rank == 0:
            self.logger.experiment[0].log({
                "tracks": track_plots,
                "translation_error": trans_err_plots,
                "rotation_error_plot": rot_err_plots
            }, commit=False)
            self.logger.experiment[0].log(metrics, commit=False)

            try:
                bot = self._bot_getter()
                if bot is not None:
                    bot.send_images(
                        track_plots,
                        f"Val images. Loss: {mean_loss.item():.4f}. "
                        f"Epoch: {self.current_epoch}. Run url: {self.logger.experiment[0].get_url()}"
                    )
            except Exception as exception:
                print(f"Bot exception: {exception}")
