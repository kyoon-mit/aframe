"""Training of the next-sample noise model on whitened noise windows."""

import math
from typing import Callable, Optional

import lightning.pytorch as pl
import torch

from train.model.regression_ky import (
    WarmupCosineAnnealingWarmRestarts,
    clamp_ssm_params,
)

LOG_2PI = math.log(2 * math.pi)


def mixture_log_prob(x, mean, log_sigma, logits):
    """log sum_k w_k N(x; mean, sigma_k^2) per sample; x, mean (B, L),
    log_sigma and logits (B, K, L)."""
    z = (x - mean).unsqueeze(1) * torch.exp(-log_sigma)
    log_w = torch.log_softmax(logits, dim=1)
    terms = log_w - log_sigma - 0.5 * z.square() - 0.5 * LOG_2PI
    return torch.logsumexp(terms, dim=1)


class NextSampleTask(pl.LightningModule):
    """Minus the mean log-likelihood per sample of windows (B, 1, L).

    AdamW with the S4D kernel parameters in their own group (``ssm_lr``,
    no weight decay), kernel parameters clamped after each step, and an
    epoch-based schedule, as ``GaussianNLLRegressionAframeCustomLR``.
    Validation also logs the per-sample log-likelihood of unit white
    Gaussian noise on the same windows.
    """

    SSM_PARAM_NAMES = ("log_dt", "log_A_real", "A_imag")

    def __init__(
        self,
        arch: torch.nn.Module,
        learning_rate: float = 1e-4,
        ssm_lr: float = 1e-3,
        weight_decay: float = 0.01,
        lr_scheduler: Optional[Callable[[torch.optim.Optimizer], object]] = None,
        log_dt_min: float = -11.5,
        log_dt_max: float = 2.3,
        log_a_max: float = 4.6,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["arch", "lr_scheduler"])
        self.model = arch
        self._lr_scheduler_factory = lr_scheduler

    def nll(self, windows):
        x = windows[:, 0]
        mean, log_sigma, logits = self.model(windows)
        return -mixture_log_prob(x, mean, log_sigma, logits).mean()

    def training_step(self, windows, batch_idx):
        loss = self.nll(windows)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=len(windows))
        return loss

    def validation_step(self, windows, batch_idx):
        loss = self.nll(windows)
        unit = (0.5 * windows.square() + 0.5 * LOG_2PI).mean()
        n = len(windows)
        self.log("val/loss", loss, on_epoch=True, prog_bar=True, batch_size=n)
        self.log("val/nll_unit_gaussian", unit, on_epoch=True, batch_size=n)
        self.log("val/gain_over_unit_gaussian", unit - loss, on_epoch=True, batch_size=n)

    def on_train_batch_end(self, outputs, batch, batch_idx):
        clamp_ssm_params(
            self,
            log_dt_bounds=(self.hparams.log_dt_min, self.hparams.log_dt_max),
            log_a_max=self.hparams.log_a_max,
        )

    def configure_optimizers(self):
        ssm, other = [], []
        for name, p in self.model.named_parameters():
            (ssm if name.rsplit(".", 1)[-1] in self.SSM_PARAM_NAMES else other).append(p)
        optimizer = torch.optim.AdamW(
            [
                {"params": other, "lr": self.hparams.learning_rate, "weight_decay": self.hparams.weight_decay},
                {"params": ssm, "lr": self.hparams.ssm_lr, "weight_decay": 0.0},
            ]
        )
        if self._lr_scheduler_factory is not None:
            scheduler = self._lr_scheduler_factory(optimizer)
        else:
            scheduler = WarmupCosineAnnealingWarmRestarts(
                optimizer, warmup_epochs=8, T_0=10, T_mult=2, eta_min=1e-7
            )
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}
