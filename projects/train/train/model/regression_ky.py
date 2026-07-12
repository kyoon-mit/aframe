import math
from typing import Callable, Optional

import torch

from train.model.regression import GaussianNLLRegressionAframe


class WarmupCosineAnnealingWarmRestarts(torch.optim.lr_scheduler.LRScheduler):
    """Linear warmup followed by CosineAnnealingWarmRestarts (epoch-based)."""

    def __init__(
        self,
        optimizer,
        warmup_epochs,
        T_0,
        T_mult=2,
        eta_min=1e-8,
        warmup_start_factor=0.01,
        last_epoch=-1,
    ):
        self.warmup_epochs = warmup_epochs
        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.warmup_start_factor = warmup_start_factor
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        e = self.last_epoch
        if e < self.warmup_epochs:
            alpha = self.warmup_start_factor + (
                1.0 - self.warmup_start_factor
            ) * e / max(1, self.warmup_epochs)
            return [base_lr * alpha for base_lr in self.base_lrs]
        t = e - self.warmup_epochs
        T_cur, T_i = self._cosine_position(t)
        return [
            self.eta_min
            + (base_lr - self.eta_min)
            * (1 + math.cos(math.pi * T_cur / T_i))
            / 2
            for base_lr in self.base_lrs
        ]

    def _cosine_position(self, t):
        T_i = self.T_0
        while t >= T_i:
            t -= T_i
            T_i *= self.T_mult
        return t, T_i


class GaussianNLLRegressionAframeCustomLR(GaussianNLLRegressionAframe):
    """
    GaussianNLLRegressionAframe with a two-group optimizer and a configurable
    epoch-based learning-rate schedule.

    Parameters whose names appear in SSM_PARAM_NAMES (the S4D state-space
    kernel parameters) are placed in their own optimizer group with learning
    rate ``ssm_lr`` and zero weight decay; all other parameters use
    ``learning_rate`` and ``weight_decay``. The schedule is built by the
    ``lr_scheduler`` factory (default: warmup followed by cosine annealing
    with warm restarts) and stepped per epoch.

    Args:
        ssm_lr:
            Learning rate for the S4D kernel parameters.
        lr_scheduler:
            Callable mapping an optimizer to a learning-rate scheduler. If
            ``None``, WarmupCosineAnnealingWarmRestarts is used with
            ``warmup_epochs=8, T_0=10, T_mult=2, eta_min=1e-7``.
        lr_scheduler_interval:
            "epoch" or "step"; how often the scheduler is stepped.
    """

    SSM_PARAM_NAMES = ("log_dt", "log_A_real", "A_imag")

    def __init__(
        self,
        *args,
        ssm_lr: float = 1e-4,
        lr_scheduler: Optional[
            Callable[[torch.optim.Optimizer], object]
        ] = None,
        lr_scheduler_interval: str = "epoch",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters("ssm_lr", "lr_scheduler_interval")
        self._lr_scheduler_factory = lr_scheduler

    def configure_optimizers(self):
        world_size = (
            torch.distributed.get_world_size()
            if torch.distributed.is_initialized()
            else 1
        )
        lr = self.hparams.learning_rate * world_size
        self._logger.info(f"Scaled lr by {world_size} to {lr}")

        ssm_params, other_params = [], []
        for name, p in self.model.named_parameters():
            leaf = name.rsplit(".", 1)[-1]
            if leaf in self.SSM_PARAM_NAMES:
                ssm_params.append(p)
            else:
                other_params.append(p)

        optimizer = torch.optim.AdamW(
            [
                {
                    "params": other_params,
                    "lr": lr,
                    "weight_decay": self.hparams.weight_decay,
                },
                {
                    "params": ssm_params,
                    "lr": self.hparams.ssm_lr,
                    "weight_decay": 0.0,
                },
            ]
        )

        if self._lr_scheduler_factory is not None:
            scheduler = self._lr_scheduler_factory(optimizer)
        else:
            scheduler = WarmupCosineAnnealingWarmRestarts(
                optimizer,
                warmup_epochs=8,
                T_0=10,
                T_mult=2,
                eta_min=1e-7,
            )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": self.hparams.lr_scheduler_interval,
            },
        }
