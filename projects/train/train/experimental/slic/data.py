"""Whitened noise-only windows from an aframe training dataset."""

import lightning.pytorch as pl
import torch

from train.data import BaseAframeDataset


@torch.no_grad()
def noise_windows(dataset: BaseAframeDataset, X: torch.Tensor):
    """Raw background (batch, detectors, samples) to whitened windows with the
    dataset's own PSD estimate, sign flip, time reversal and whitening, in the
    order of its ``inject``, with no signal added."""
    X, psds = dataset.psd_estimator(X)
    X = dataset.inverter(X)
    X = dataset.reverser(X)
    return dataset.whitener(X, psds)


class NoiseWindows(pl.LightningDataModule):
    """Batches of whitened noise windows with one detector per example,
    shape (batch * detectors, 1, samples).

    ``dataset`` is set up as configured. Training draws from its training
    background and validation from its validation background, using its
    strain loader only.
    """

    def __init__(self, dataset: BaseAframeDataset):
        super().__init__()
        self.dataset = dataset

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

    def on_after_batch_transfer(self, batch, dataloader_idx):
        [X] = batch
        windows = noise_windows(self.dataset, X)
        return windows.reshape(-1, 1, windows.shape[-1])
