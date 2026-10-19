"""Batches of whitened noise windows for the generator and the judges."""

import lightning.pytorch as pl
import torch

from train.experimental.noise_model.sources import Whitening, build_source


class ZipDatasets(torch.utils.data.IterableDataset):
    """One CPU batch from each dataset per step."""

    def __init__(self, datasets):
        self.datasets = datasets

    def __len__(self):
        return min(len(dataset) for dataset in self.datasets)

    def __iter__(self):
        return zip(*self.datasets)


class NoiseData(pl.LightningDataModule):
    """Whitened noise windows from one or two sources.

    With only ``first``, a batch is ``batch_size`` windows from it, for training
    the generator. With ``second`` too, a batch is ``batch_size`` windows from
    each, with labels 1 for ``first`` and 0 for ``second``, for training a judge.
    ``first`` and ``second`` are config entries for ``sources.build_source``.
    Training draws from the sources' fit part; validation and testing from
    their test part. Windows are whitened on the GPU after the batch arrives.
    """

    def __init__(
        self,
        ifos: list[str],
        sample_rate: float,
        kernel_length: float,
        psd_length: float,
        fduration: float,
        highpass: float,
        first: dict,
        second: dict | None = None,
        lowpass: float | None = None,
        fftlength: float | None = None,
        batch_size: int = 64,
        batches_per_epoch: int = 500,
        val_batches: int = 50,
        test_windows: int = 20000,
        num_workers: int = 4,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.whitening = Whitening(sample_rate, kernel_length, psd_length, fduration, highpass, lowpass, fftlength)
        self.sources = [build_source(spec, ifos) for spec in (first, second) if spec is not None]

    def dataloader(self, part: str, batches: int):
        datasets = [source.dataset(part, self.whitening, self.hparams.batch_size, batches) for source in self.sources]
        # batch_size=None: the datasets already yield whole batches
        return torch.utils.data.DataLoader(
            ZipDatasets(datasets), batch_size=None, num_workers=self.hparams.num_workers
        )

    def train_dataloader(self):
        return self.dataloader("fit", self.hparams.batches_per_epoch)

    def val_dataloader(self):
        return self.dataloader("test", self.hparams.val_batches)

    def test_dataloader(self):
        return self.dataloader("test", self.hparams.test_windows // self.hparams.batch_size)

    def on_after_batch_transfer(self, batch, dataloader_idx):
        self.whitening.to(batch[0].device)
        windows = [source.to_windows(raw, self.whitening) for source, raw in zip(self.sources, batch)]
        if len(windows) == 1:
            return windows[0]
        first, second = windows
        return torch.cat(windows), torch.cat([first.new_ones(len(first)), second.new_zeros(len(second))])

    def windows(self, which: int, part: str, batch_size: int, batches: int, device: str):
        """Whitened windows from source ``which`` (0 first, 1 second), outside training."""
        source = self.sources[which]
        dataset = source.dataset(part, self.whitening, batch_size, batches)
        self.whitening.to(device)
        for raw in torch.utils.data.DataLoader(dataset, batch_size=None, num_workers=self.hparams.num_workers):
            yield source.to_windows(raw.to(device), self.whitening)
