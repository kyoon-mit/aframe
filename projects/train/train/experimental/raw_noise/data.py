"""Raw noise windows from an aframe training dataset."""

import math

import lightning.pytorch as pl
import torch

from train.data import BaseAframeDataset
from train.experimental.raw_noise.predict import in_context_errors, prediction_errors


class RawWindows(pl.LightningDataModule):
    """Batches ``(windows, offset)`` of noise with one detector per
    example, from the dataset's strain loader, sign flip and time reversal.

    ``windows`` (batch * detectors, 1, samples) is, for ``kind`` "raw", the
    strain divided by ``scale``; for "prediction_errors", the errors of the
    linear prediction of each strain sample from the second before it, with
    the spectrum of the window's first ``psd_length`` seconds. The first
    second is dropped in both. For "in_context", the errors of a linear
    predictor from the previous ``IN_CONTEXT_ORDER`` samples, fit by least
    squares on the window's own strain over the ``IN_CONTEXT_FIT`` samples
    before the last ``IN_CONTEXT_LAST``, divided by the fit's error spread;
    only those fit and last samples are returned. ``offset`` (batch *
    detectors,) is what to add to minus the log-likelihood per sample of
    ``windows`` to get that of the strain divided by ``scale``: 0 for "raw".
    """

    IN_CONTEXT_ORDER, IN_CONTEXT_FIT, IN_CONTEXT_LAST = 1024, 19 * 2048, 4 * 2048

    def __init__(self, dataset: BaseAframeDataset, kind: str, scale: float):
        super().__init__()
        if kind not in ("raw", "prediction_errors", "in_context"):
            raise ValueError(f"unknown kind {kind}")
        self.dataset = dataset
        self.kind = kind
        self.scale = scale

    def prepare_data(self):
        self.dataset.prepare_data()

    def setup(self, stage: str):
        self.dataset.trainer = self.trainer
        self.dataset.setup(stage)

    def train_dataloader(self):
        return self.dataset.strain_dataloader()

    def val_dataloader(self):
        dataset = self.dataset
        batches = dataset.hparams.val_batches or max(
            1, dataset.batches_per_epoch // 10
        )
        return dataset.strain_dataloader(dataset.valid_fnames, batches)

    @torch.no_grad()
    def on_after_batch_transfer(self, batch, dataloader_idx):
        [X] = batch
        dataset = self.dataset
        X = dataset.reverser(dataset.inverter(X))
        second = int(dataset.hparams.sample_rate)
        if self.kind == "in_context":
            x = (X[..., second:] / self.scale).reshape(-1, X.size(-1) - second)
            windows, std = in_context_errors(
                x, self.IN_CONTEXT_ORDER, self.IN_CONTEXT_FIT, self.IN_CONTEXT_LAST
            )
            return windows.float()[:, None], torch.log(std).float()
        if self.kind == "raw":
            windows = (X[..., second:] / self.scale).float()
            offset = torch.zeros(X.shape[:2], device=X.device)
        else:
            _, psds = dataset.psd_estimator(X)
            windows, first = prediction_errors(
                X, psds, dataset.hparams.sample_rate, second
            )
            offset = -(torch.log(first) + math.log(self.scale)).float()
        return windows.reshape(-1, 1, windows.shape[-1]), offset.reshape(-1)
