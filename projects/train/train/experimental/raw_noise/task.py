"""Training of a next-sample noise model on raw strain."""

import torch

from train.experimental.next_sample.task import (
    LOG_2PI,
    NextSampleTask,
    mixture_log_prob,
)


class RawNextSampleTask(NextSampleTask):
    """Minus the mean log-likelihood per sample of ``RawWindows`` batches.

    The first ``burn_in`` samples of each window are left out of the loss.
    Validation also logs the same for the strain divided by ``scale``
    (``loss_raw``), both over the last ``last`` samples only, and unit white
    Gaussian noise on the same samples. ``kind`` and ``scale`` are read by
    the entry point to build the windows.
    """

    def __init__(
        self,
        *args,
        kind: str = "raw",
        scale: float = 3e-19,
        burn_in: int = 2048,
        last: int = 8192,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters(
            {"kind": kind, "scale": scale, "burn_in": burn_in, "last": last}
        )

    def nll(self, windows):
        mean, log_sigma, logits = self.model(windows)
        nll = -mixture_log_prob(windows[:, 0], mean, log_sigma, logits)
        return nll[:, self.hparams.burn_in :]

    def training_step(self, batch, batch_idx):
        windows, _ = batch
        loss = self.nll(windows).mean()
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=len(windows))
        return loss

    def validation_step(self, batch, batch_idx):
        windows, offset = batch
        nll = self.nll(windows)
        loss, last = nll.mean(), nll[:, -self.hparams.last :].mean()
        x = windows[:, 0, self.hparams.burn_in :]
        unit = (0.5 * x.square() + 0.5 * LOG_2PI).mean()
        n = len(windows)
        self.log("val/loss", loss, on_epoch=True, prog_bar=True, batch_size=n)
        self.log("val/loss_last", last, on_epoch=True, batch_size=n)
        self.log("val/loss_raw", loss + offset.mean(), on_epoch=True, prog_bar=True, batch_size=n)
        self.log("val/loss_raw_last", last + offset.mean(), on_epoch=True, batch_size=n)
        self.log("val/nll_unit_gaussian", unit, on_epoch=True, batch_size=n)
        self.log("val/gain_over_unit_gaussian", unit - loss, on_epoch=True, batch_size=n)
