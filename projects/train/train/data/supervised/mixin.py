import h5py
import torch
from ml4gw.dataloading import Hdf5TimeSeriesDataset
from ml4gw.utils.slicing import sample_kernels

import lightning.pytorch as pl

from train.data.base import BaseAframeDataset, ZippedDataset
from train.utils.params import compute_target_params_tensor
from train.data.waveforms import ChunkedWaveformDataset, Hdf5WaveformLoader


class RegressionSupervisedMixin:
    """Mixin that adds GW parameter regression to any supervised domain class.

    Combine with a domain class by placing the mixin first in the MRO:

        class TimeDomainRegressionDataset(
            RegressionSupervisedMixin, TimeDomainSupervisedAframeDataset
        ):
            pass

    The mixin replaces:
    - ``setup``: skips timeslide construction; builds fixed-seed val prior params.
    - ``on_before_batch_transfer`` / ``on_after_batch_transfer``: handles param
      extraction and injects all samples (no classification mask).
    - ``val_dataloader``: simple random-injection validation, no timeslides.
    - ``train_dataloader``: adds ``Hdf5WaveformLoader`` with params for the disk path.

    Output batch format: ``(X, y_params, empty_z)`` — matches
    ``RegressionTimeDomainDataset`` and the ``_RegressionBase`` model.

    Domain-specific post-injection processing (whitening, Q-transform, etc.) is
    delegated to ``self.apply_transforms(X, psds)`` which must be defined on the
    domain class.
    """

    def __init__(
        self,
        *args,
        target_parameters: tuple[str, ...] = ("chirp_mass",),
        n_val_waveforms: int = 4096,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.target_parameters = target_parameters
        self.n_val_waveforms = n_val_waveforms

    # ------------------------------------------------------------------ #
    # Setup — skip timeslide construction                                  #
    # ------------------------------------------------------------------ #

    def setup(self, stage: str) -> None:
        world_size, rank = self.get_world_size_and_rank()
        self._logger = self.get_logger(world_size, rank)
        self.train_fnames, self.valid_fnames = self.train_val_split()

        with h5py.File(self.train_fnames[0], "r") as f:
            sample_rate = 1 / f[self.hparams.ifos[0]].attrs["dx"]
            if not sample_rate == self.hparams.sample_rate:
                raise ValueError(
                    f"Specified sample rate is {self.hparams.sample_rate} "
                    f"but background data is sampled at {sample_rate}"
                )

        self.build_transforms()
        self.transforms_to_device()

        # Fixed-seed validation prior params (consistent across epochs / ranks)
        rng_state = torch.get_rng_state()
        torch.manual_seed(42 + rank)
        n = self.n_val_waveforms // world_size
        self._val_prior_params = self.waveform_sampler.training_prior(
            n, device="cpu"
        )
        self.val_params = compute_target_params_tensor(
            self._val_prior_params, self.target_parameters
        )
        torch.set_rng_state(rng_state)

    # ------------------------------------------------------------------ #
    # Batch transfer hooks                                                 #
    # ------------------------------------------------------------------ #

    def on_before_batch_transfer(self, batch, _):
        if self.trainer.training and self.waveforms_from_disk:
            X, (polarizations, params) = batch
            polarizations = self.slice_waveforms(polarizations)
            batch = X, (polarizations, params)
        return batch

    def on_after_batch_transfer(self, batch, _):
        if self.trainer.training:
            if self.waveforms_from_disk:
                [X], (waveforms, params) = batch
                X, params_out = self._inject_all_from_disk(X, waveforms, params)
            else:
                [X] = batch
                X, params_out = self._inject_from_generator(X)
        elif self.trainer.validating or self.trainer.sanity_checking:
            [X] = batch
            X, params_out = self._inject_fixed_val(X)
        else:
            return batch

        B = params_out.shape[0]
        empty_z = torch.empty(B, 0, device=params_out.device)
        return X, params_out, empty_z

    # ------------------------------------------------------------------ #
    # Injection helpers — always inject all B samples                     #
    # ------------------------------------------------------------------ #

    def _project_and_inject(self, X, waveforms, psds):
        """Project waveforms onto IFOs, inject into background, return X."""
        B = X.shape[0]
        dec, psi, phi = self.sample_extrinsic(X)
        snrs = (
            self.snr_sampler.sample((B,)).to(X.device)
            if self.snr_sampler is not None
            else None
        )
        responses = self.projector(
            dec, psi, phi, snrs, psds,
            cross=waveforms[:, 0], plus=waveforms[:, 1],
        )
        kernels = sample_kernels(responses, kernel_size=X.size(-1), coincident=True)
        return X + kernels

    @torch.no_grad()
    def _inject_all_from_disk(self, X, waveforms, params_raw):
        X, psds = self.psd_estimator(X)
        X = self.inverter(X)
        X = self.reverser(X)

        B = X.shape[0]
        idx = torch.randperm(waveforms.shape[0])[:B]
        waveforms = waveforms[idx].to(X.device).float()
        params = compute_target_params_tensor(
            {k: v[idx].to(X.device) for k, v in params_raw.items()},
            self.target_parameters,
        )

        X = self._project_and_inject(X, waveforms, psds)
        X = self.apply_transforms(X, psds)
        return X, params

    @torch.no_grad()
    def _inject_from_generator(self, X):
        X, psds = self.psd_estimator(X)
        X = self.inverter(X)
        X = self.reverser(X)

        B = X.shape[0]
        prior_params = self.waveform_sampler.training_prior(B, device=X.device)
        hc, hp = self.waveform_sampler(**prior_params)
        waveforms = torch.stack([hc, hp], dim=1).float()
        waveforms = self.slice_waveforms(waveforms)
        params = compute_target_params_tensor(prior_params, self.target_parameters)

        X = self._project_and_inject(X, waveforms, psds)
        X = self.apply_transforms(X, psds)
        return X, params.to(X.device if not isinstance(X, tuple) else X[0].device)

    @torch.no_grad()
    def _inject_fixed_val(self, X):
        X, psds = self.psd_estimator(X)

        B = X.shape[0]
        idx = torch.randperm(len(self.val_params))[:B]
        prior_params = {
            k: v[idx].to(X.device) for k, v in self._val_prior_params.items()
        }
        hc, hp = self.waveform_sampler(**prior_params)
        waveforms = torch.stack([hc, hp], dim=1).float()
        waveforms = self.slice_waveforms(waveforms)
        params = self.val_params[idx].to(X.device)

        X = self._project_and_inject(X, waveforms, psds)
        X = self.apply_transforms(X, psds)
        return X, params

    # ------------------------------------------------------------------ #
    # Dataloaders                                                          #
    # ------------------------------------------------------------------ #

    def train_dataloader(self):
        raw_kernel = int(self.hparams.sample_rate * self.sample_length)
        bg_dataset = Hdf5TimeSeriesDataset(
            self.train_fnames,
            channels=self.hparams.ifos,
            kernel_size=raw_kernel,
            batch_size=self.hparams.batch_size,
            batches_per_epoch=self.batches_per_epoch,
            coincident=False,
            num_files_per_batch=self.hparams.num_files_per_batch,
        )
        pin_memory = isinstance(
            self.trainer.accelerator, pl.accelerators.CUDAAccelerator
        )
        bg_loader = torch.utils.data.DataLoader(
            bg_dataset,
            num_workers=self.num_workers,
            pin_memory=pin_memory,
        )

        if not self.waveforms_from_disk:
            return bg_loader

        world_size, _ = self.get_world_size_and_rank()
        batches_per_epoch = self.hparams.batches_per_epoch // world_size
        batches_per_chunk = (
            int(batches_per_epoch // self.hparams.chunks_per_epoch) + 1
        )

        wf_loader = Hdf5WaveformLoader(
            fnames=self.waveform_sampler.training_waveform_files,
            channels=["cross", "plus"],
            path="waveforms",
            param_keys=["mass_1", "mass_2"],
            batch_size=self.hparams.chunk_size,
            batches_per_epoch=self.hparams.chunks_per_epoch or 1,
        )
        wf_dl = torch.utils.data.DataLoader(
            wf_loader,
            num_workers=2,
            pin_memory=pin_memory,
            persistent_workers=True,
        )
        wf_dataset = ChunkedWaveformDataset(
            wf_dl,
            batch_size=self.hparams.batch_size,
            batches_per_chunk=batches_per_chunk,
        )
        return ZippedDataset(bg_loader, wf_dataset)

    def val_dataloader(self):
        val_batches = max(1, self.batches_per_epoch // 10)
        val_dataset = Hdf5TimeSeriesDataset(
            self.valid_fnames,
            channels=self.hparams.ifos,
            kernel_size=int(self.hparams.sample_rate * self.sample_length),
            batch_size=self.hparams.batch_size,
            batches_per_epoch=val_batches,
            coincident=False,
            num_files_per_batch=max(1, len(self.valid_fnames) // 4),
        )
        pin_memory = isinstance(
            self.trainer.accelerator, pl.accelerators.CUDAAccelerator
        )
        return torch.utils.data.DataLoader(
            val_dataset,
            num_workers=self.num_workers,
            pin_memory=pin_memory,
        )
