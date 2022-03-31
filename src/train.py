import traceback
from typing import List, Optional

import hydra
import pytorch_lightning.loggers
from omegaconf import DictConfig
from pytorch_lightning import (Callback, LightningDataModule, LightningModule, seed_everything, Trainer)
from pytorch_lightning.loggers import LightningLoggerBase

from src.utils import utils
from src.utils.telegram_bot import Bot

log = utils.get_logger(__name__)


def train(config: DictConfig) -> Optional[float]:
    """Contains training pipeline.
    Instantiates all PyTorch Lightning objects from config.

    Args:
        config (DictConfig): Configuration composed by Hydra.

    Returns:
        Optional[float]: Metric score for hyperparameter optimization.
    """

    # Set seed for random number generators in pytorch, numpy and python.random
    if "seed" in config:
        seed_everything(config.seed, workers=True)

    # Init Lightning datamodule
    log.info(f"Instantiating datamodule <{config.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(config.datamodule)

    # Init Lightning datamodule
    log.info(f"Instantiating telegram bot <{config.telegram_bot._target_}>")
    bot: Bot = hydra.utils.instantiate(config.telegram_bot)

    # Init Lightning model
    log.info(f"Instantiating model <{config.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(config.model)

    if hasattr(model, "bot_getter"):
        model.bot_getter = lambda: bot

    # Init Lightning callbacks
    callbacks: List[Callback] = []
    if "callbacks" in config:
        for _, cb_conf in config["callbacks"].items():
            if "_target_" in cb_conf:
                log.info(f"Instantiating callback <{cb_conf._target_}>")
                callbacks.append(hydra.utils.instantiate(cb_conf))

    # Init Lightning loggers
    logger: List[LightningLoggerBase] = []
    if "logging" in config:
        for _, lg_conf in config["logging"].items():
            if "_target_" in lg_conf:
                log.info(f"Instantiating logging <{lg_conf._target_}>")
                logger.append(hydra.utils.instantiate(lg_conf))

    # Init Lightning trainer
    log.info(f"Instantiating trainer <{config.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        config.trainer, callbacks=callbacks, logger=logger, _convert_="partial"
    )

    # Send some parameters from config to all lightning loggers
    log.info("Logging hyperparameters!")
    utils.log_hyperparameters(
        config=config,
        model=model,
        datamodule=datamodule,
        trainer=trainer,
        callbacks=callbacks,
        logger=logger,
    )

    # Train the model
    log.info("Starting training!")
    experiment_url = None
    for one_log in logger:
        if isinstance(one_log, pytorch_lightning.loggers.WandbLogger):
            experiment_url = one_log.experiment.get_url()
            bot.send_message(
                f"😱😱😱 Started new experiment.\n"
                f"Notes: {one_log.experiment.notes}\n"
                f"Link: {experiment_url}"
            )
    try:
        trainer.fit(model=model, datamodule=datamodule)
    except Exception:
        bot.send_message(
            f"😭😭😭 Something wrong with experiment.\n"
            f"Link: {experiment_url}"
            f"<code>Error: {traceback.format_exc()}</code>"
        )
        log.exception(traceback.format_exc())
    finally:
        bot.send_message(
            f"🏳️🏳️🏳️ Finish experiment.\n"
            f"Link: {experiment_url}"
        )

    # Make sure everything closed properly
    log.info("Finalizing!")
    utils.finish(
        config=config,
        model=model,
        datamodule=datamodule,
        trainer=trainer,
        callbacks=callbacks,
        logger=logger,
    )

    # Print path to best checkpoint
    log.info(f"Best checkpoint path:\n{trainer.checkpoint_callback.best_model_path}")

    # Return metric score for hyperparameter optimization
    optimized_metric = config.get("optimized_metric")
    if optimized_metric:
        return trainer.callback_metrics[optimized_metric]
