"""Dataset that validates a denoiser on paired (noisy, clean) batches.

The standard validation path builds timeslides and returns
``(shift, X_bg, X_fg, params)``, which carries no clean target: it exists to
score a detection statistic, not a reconstruction. A denoiser needs the clean
waveform to compare against, so this subclass runs validation through the
same injection pipeline as training, over the held-out background files.

Use with ``waveform_prob=1.0`` so every validation row carries signal.
"""

from typing import Optional, Sequence

import torch
from ml4gw.dataloading import Hdf5TimeSeriesDataset

from train.data.supervised.time_domain import (
    DenoisingTimeDomainSupervisedAframeDataset,
)


class DenoiserOnlyAframeDataset(DenoisingTimeDomainSupervisedAframeDataset):
    """Validation batches are injected the same way training batches are.

    Both loaders emit ``(X, X_clean, y, params)``.

    Args:
        val_batches: validation batches per epoch. Defaults to a tenth of
            the training batches.
        persistent_workers: keep strain dataloader workers alive between
            epochs.
    """

    def __init__(
        self,
        *args,
        val_batches: Optional[int] = None,
        persistent_workers: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters(
            {
                "val_batches": val_batches,
                "persistent_workers": persistent_workers,
            }
        )

    def strain_dataloader(
        self,
        fnames: Optional[Sequence[str]] = None,
        batches_per_epoch: Optional[int] = None,
    ) -> torch.utils.data.DataLoader:
        """The base strain loader, with persistent workers.

        Workers are forked from a parent holding a CUDA context, and a CUDA
        context does not survive fork, so re-forking them every epoch
        eventually loses the race and one dies with "CUDA error:
        initialization error". Keeping them alive forks once per fit.
        """
        dataset = Hdf5TimeSeriesDataset(
            fnames if fnames is not None else self.train_fnames,
            channels=self.hparams.ifos,
            kernel_size=int(self.hparams.sample_rate * self.sample_length),
            batch_size=self.hparams.batch_size,
            batches_per_epoch=batches_per_epoch or self.batches_per_epoch,
            coincident=False,
            num_files_per_batch=self.hparams.num_files_per_batch,
        )
        num_workers = self.num_workers
        return torch.utils.data.DataLoader(
            dataset,
            num_workers=num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=(
                self.hparams.persistent_workers and num_workers > 0
            ),
        )

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        """Injected batches drawn from the held-out validation background."""
        return self.train_dataloader(
            fnames=self.valid_fnames,
            batches_per_epoch=self.hparams.val_batches
            or max(1, self.batches_per_epoch // 10),
        )

    def on_after_batch_transfer(self, batch, dataloader_idx):
        """Route validation through the training augmentation path.

        The base implementation branches on trainer state and sends
        validation to the timeslide builder; here validation is just another
        injected batch, so temporarily present as training while the
        injection runs.
        """
        if self.trainer.validating or self.trainer.sanity_checking:
            if self.waveforms_from_disk:
                [X], (waveforms, params) = batch
                waveforms = self.slice_waveforms(waveforms)
            else:
                [X] = batch
                waveforms, params = self.waveform_sampler.sample(X)
            return self.inject(X=X, waveforms=waveforms, params=params)
        return super().on_after_batch_transfer(batch, dataloader_idx)

    def on_before_batch_transfer(self, batch, dataloader_idx):
        """Slice disk-loaded waveforms for validation batches too."""
        if (
            self.trainer.validating or self.trainer.sanity_checking
        ) and self.waveforms_from_disk:
            return batch
        return super().on_before_batch_transfer(batch, dataloader_idx)
