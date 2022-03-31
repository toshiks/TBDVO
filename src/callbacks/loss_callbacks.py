import logging
from typing import Any, Optional

from pytorch_lightning import Callback, LightningModule, Trainer
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.types import STEP_OUTPUT

from src.losses.loss_with_parameters import LossWithParameters


class LossParamsCallback(Callback):
    def __init__(self):
        self._loss_attributes_names = []

    def on_pretrain_routine_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if not trainer.logger:
            raise MisconfigurationException(
                "Cannot use `LossParamsCallback` callback with `Trainer` that has no logger."
            )

        self._loss_attributes_names = [
            key for key, value in pl_module.named_modules() if isinstance(value, LossWithParameters)
        ]

        logging.getLogger("loss_callbacks.LossParamsCallback").info(
            "Registered names for logging params: %s", self._loss_attributes_names
        )

    def on_train_batch_end(self, trainer: Trainer, pl_module: LightningModule, outputs: STEP_OUTPUT,
                           batch: Any, batch_idx: int, unused: Optional[int] = 0) -> None:
        params = {}
        for name in self._loss_attributes_names:
            loss_attr: LossWithParameters = pl_module.__getattr__(name)
            for key, value in loss_attr.params.items():
                params[f"loss_params/{key}"] = value

        pl_module.log_dict(params, add_dataloader_idx=False, sync_dist=True, on_step=True)
