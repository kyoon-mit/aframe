import logging
import sys
from typing import Union

import lightning.pytorch as pl
import torch
from architectures import Architecture

from train.metrics import TimeSlideAUROC

Tensor = torch.Tensor


class AframeBase(pl.LightningModule):
    """Shared infrastructure for all Aframe models.

    Owns the architecture wrapper, logging utilities, and the
    ``training_step`` dispatch loop. Detection-specific concerns
    (AUROC metric, timeslide validation, OneCycleLR) live in
    ``ClassificationAframe``; regression concerns live in
    ``RegressionAframe``.
    """

    def __init__(
        self,
        arch: Architecture,
        verbose: bool = False,
    ) -> None:
        super().__init__()
        self.model = arch
        self.verbose = verbose
        self._logger = self.init_logging(verbose)
        # NOTE: save_hyperparameters is NOT called here — subclasses own it
        # so they can control which callables/objects are excluded.

    def init_logging(self, verbose):
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        logging.basicConfig(
            format=log_format,
            level=logging.DEBUG if verbose else logging.INFO,
            stream=sys.stdout,
        )
        world_size, rank = self.get_world_size_and_rank()
        logger_name = self.__class__.__name__
        if world_size > 1:
            logger_name += f":{rank}"
        return logging.getLogger(logger_name)

    def get_world_size_and_rank(self) -> tuple[int, int]:
        if not torch.distributed.is_initialized():
            return 1, 0
        return (
            torch.distributed.get_world_size(),
            torch.distributed.get_rank(),
        )

    def forward(self, X: Tensor) -> Tensor:
        raise NotImplementedError

    def train_step(self, batch: Tensor) -> Union[Tensor, dict[str, Tensor]]:
        raise NotImplementedError

    def score(self, X: Tensor) -> Tensor:
        raise NotImplementedError

    def compute_loss_fn(self, **losses):
        raise NotImplementedError

    def on_validation_epoch_start(self):
        self.validating = True

    def on_validation_epoch_end(self):
        self.validating = False

    def on_validation_score(self, *tensors):
        for cb in self.trainer.callbacks:
            if hasattr(cb, "on_validation_score"):
                cb.on_validation_score(self.trainer, self, *tensors)

    def training_step(self, batch: tuple[Tensor, Tensor]) -> Tensor:
        loss = self.train_step(batch)

        if isinstance(loss, dict):
            for name, value in loss.items():
                self.log(
                    name,
                    value.mean(),
                    on_step=True,
                    on_epoch=True,
                    prog_bar=False,
                    logger=True,
                )
            loss = self.compute_loss_fn(**loss)

        loss = loss.mean()
        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return loss


class ClassificationAframe(AframeBase):
    """AframeBase + timeslide-AUROC validation and OneCycleLR optimizer.

    This is the base for all detection/classification models (``SupervisedAframe``
    and its variants).  Regression models inherit from ``RegressionAframe`` instead.
    """

    def __init__(
        self,
        arch: Architecture,
        metric: TimeSlideAUROC,
        learning_rate: float,
        pct_lr_ramp: float,
        weight_decay: float = 0.0,
        verbose: bool = False,
    ) -> None:
        super().__init__(arch, verbose)
        self.metric = metric
        self.save_hyperparameters(ignore=["arch", "metric"])

    def validation_step(self, batch, _) -> None:
        shift, X_bg, X_inj = batch

        y_bg = self.score(X_bg)

        num_views, batch, *shape = X_inj.shape
        X_inj = X_inj.view(num_views * batch, *shape)
        y_fg = self.score(X_inj)
        y_fg = y_fg.view(num_views, batch).mean(0)

        self.metric.update(shift, y_bg, y_fg)
        self.log(
            "valid_auroc",
            self.metric,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )

    def configure_optimizers(self):
        if not torch.distributed.is_initialized():
            world_size = 1
        else:
            world_size = torch.distributed.get_world_size()

        lr = self.hparams.learning_rate * world_size
        self._logger.info(f"Scaled lr by {world_size} to {lr}")
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr, weight_decay=self.hparams.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            pct_start=self.hparams.pct_lr_ramp,
            max_lr=lr,
            total_steps=self.trainer.estimated_stepping_batches,
        )
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}
