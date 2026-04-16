"""Regression data module using aframe's on-the-fly injection pipeline.

Architecture:
    Training:   background HDF5 → PSD estimation → waveform projection/injection
                → whitening → (X_whitened, params, empty_z)
    Validation: same pipeline over held-out background + validation waveforms,
                paired with their chirp_mass / mass_ratio labels.

Batch format throughout: (X, y_params, z_empty)
    X          : (B, n_ifos, L) whitened strain, channels-first
    y_params   : (B, n_target_params) e.g. chirp_mass, mass_ratio
    z_empty    : (B, 0)  placeholder (no observed variables used)
"""

import math
import warnings
from pathlib import Path
from typing import Iterable

import h5py
import lightning.pytorch as pl
import numpy as np
import torch
from ml4gw.utils.slicing import sample_kernels

from train.data.base import BaseAframeDataset, ZippedDataset
from train.data.waveforms import WaveformLoader


def _compute_target_params(
    m1: np.ndarray,
    m2: np.ndarray,
    target_parameters: tuple[str, ...],
) -> np.ndarray:
    """Compute requested parameters from component masses.

    Supported names: 'chirp_mass', 'mass_ratio', 'mass_1', 'mass_2'.

    Returns:
        Float32 array of shape (N, len(target_parameters)).
    """
    available = {
        "chirp_mass": ((m1 * m2) ** 3 / (m1 + m2)) ** (1 / 5),
        "mass_ratio": m2 / m1,
        "mass_1": m1,
        "mass_2": m2,
    }
    cols = []
    for name in target_parameters:
        if name not in available:
            raise ValueError(
                f"Unknown target parameter {name!r}. "
                f"Choose from {list(available)}"
            )
        cols.append(available[name])
    return np.stack(cols, axis=-1).astype(np.float32)


class _WaveformParamLoader(torch.utils.data.IterableDataset):
    """Load waveform polarisations and intrinsic parameters from HDF5 files.

    Mirrors ``Hdf5WaveformLoader`` but simultaneously reads ``mass_1`` and
    ``mass_2`` from the ``parameters/`` group and converts them to the
    requested ``target_parameters`` (e.g. chirp_mass, mass_ratio).

    Each iteration yields a tuple ``(polarizations, params)`` where:
        polarizations : (batch_size, 2, L)  — [cross, plus]
        params        : (batch_size, n_params)

    Args:
        fnames:
            Paths to HDF5 files written by ``WaveformPolarizationSet.write()``.
        target_parameters:
            Names of parameters to return (see ``_compute_target_params``).
        batch_size:
            Number of waveforms per yielded batch.
        batches_per_epoch:
            Total number of batches per call to ``__iter__``.
        chunk_size:
            Number of waveforms to read from disk at once.
    """

    def __init__(
        self,
        fnames: Iterable[Path],
        target_parameters: tuple[str, ...],
        batch_size: int,
        batches_per_epoch: int,
        chunk_size: int = 1000,
    ) -> None:
        self.fnames = list(fnames)
        self.target_parameters = target_parameters
        self.batch_size = batch_size
        self.batches_per_epoch = batches_per_epoch
        self.chunk_size = chunk_size

        self.sizes: dict[Path, int] = {}
        self._pol_dsets: dict[Path, dict[str, h5py.Dataset]] = {}
        self._param_dsets: dict[Path, dict[str, h5py.Dataset]] = {}
        self._files: dict[Path, h5py.File] = {}

        for fname in self.fnames:
            f = h5py.File(fname, "r")
            self._files[fname] = f
            wf_grp = f["waveforms"]
            pm_grp = f["parameters"]

            self._pol_dsets[fname] = {ch: wf_grp[ch] for ch in ("cross", "plus")}
            self._param_dsets[fname] = {k: pm_grp[k] for k in ("mass_1", "mass_2")}

            n = len(self._pol_dsets[fname]["cross"])
            self.sizes[fname] = n

            if self._pol_dsets[fname]["cross"].chunks is None:
                warnings.warn(
                    f"{fname} lacks chunked storage — I/O may be slow.",
                    stacklevel=2,
                )

        self.waveform_size = self._pol_dsets[self.fnames[0]]["cross"].shape[1]
        self.probs = np.array(
            [self.sizes[f] / self.total for f in self.fnames]
        )

    @property
    def total(self) -> int:
        return sum(self.sizes.values())

    @property
    def chunks_per_batch(self) -> int:
        return math.ceil(self.batch_size / self.chunk_size)

    def __len__(self) -> int:
        return self.batches_per_epoch

    def __del__(self) -> None:
        for f in self._files.values():
            try:
                f.close()
            except Exception:
                pass

    def _load_chunk(self, fname: Path, start: int, size: int):
        end = min(start + size, self.sizes[fname])
        pol = {
            ch: self._pol_dsets[fname][ch][start:end]
            for ch in ("cross", "plus")
        }
        pm = {
            k: self._param_dsets[fname][k][start:end]
            for k in ("mass_1", "mass_2")
        }
        return pol, pm

    def sample_batch(self):
        batch_pol = np.zeros(
            (self.batch_size, 2, self.waveform_size), dtype=np.float32
        )
        m1_buf = np.zeros(self.batch_size, dtype=np.float64)
        m2_buf = np.zeros(self.batch_size, dtype=np.float64)

        for i in range(self.chunks_per_batch):
            fname = np.random.choice(self.fnames, p=self.probs)
            chunk_size = min(
                self.chunk_size, self.batch_size - i * self.chunk_size
            )
            max_start = self.sizes[fname] - chunk_size
            start = np.random.randint(0, max_start + 1)

            pol_chunk, pm_chunk = self._load_chunk(fname, start, chunk_size)

            bs = i * self.chunk_size
            be = bs + chunk_size
            batch_pol[bs:be, 0, :] = pol_chunk["cross"]
            batch_pol[bs:be, 1, :] = pol_chunk["plus"]
            m1_buf[bs:be] = pm_chunk["mass_1"]
            m2_buf[bs:be] = pm_chunk["mass_2"]

        params = _compute_target_params(m1_buf, m2_buf, self.target_parameters)
        return torch.tensor(batch_pol), torch.tensor(params)

    def __iter__(self):
        for _ in range(self.batches_per_epoch):
            yield self.sample_batch()


class _ChunkedWaveformParamDataset(torch.utils.data.IterableDataset):
    """Sample batches of (polarizations, params) tuples from a chunk iterator.

    Mirrors ``ChunkedWaveformDataset`` but handles the two-tensor output of
    ``_WaveformParamLoader``.

    Args:
        chunk_it:
            Iterator (typically a DataLoader wrapping ``_WaveformParamLoader``)
            that yields ``([pol_chunk], [param_chunk])`` with shapes
            ``(chunk_size, 2, L)`` and ``(chunk_size, n_params)``.
        batch_size:
            Number of waveforms per yielded batch.
        batches_per_chunk:
            Number of batches to sample from each loaded chunk.
    """

    def __init__(
        self,
        chunk_it: Iterable,
        batch_size: int,
        batches_per_chunk: int,
    ) -> None:
        self.chunk_it = chunk_it
        self.batch_size = batch_size
        self.batches_per_chunk = batches_per_chunk

    def __len__(self) -> int:
        return len(self.chunk_it) * self.batches_per_chunk

    def __iter__(self):
        it = iter(self.chunk_it)
        # DataLoader adds a leading batch-dim of 1; unpack it.
        [pol_chunk], [param_chunk] = next(it)
        num_waveforms = pol_chunk.shape[0]

        while True:
            for _ in range(self.batches_per_chunk):
                idx = torch.randperm(num_waveforms)[: self.batch_size]
                yield pol_chunk[idx], param_chunk[idx]

            try:
                [pol_chunk], [param_chunk] = next(it)
            except StopIteration:
                break
            num_waveforms = pol_chunk.shape[0]


class RegressionTimeDomainDataset(BaseAframeDataset):
    """``BaseAframeDataset`` subclass for GW parameter regression.

    Uses aframe's background loading, PSD estimation, on-the-fly waveform
    projection and whitening pipeline.  Each training batch is produced by
    injecting a fresh set of waveforms (drawn from ``waveform_sampler``) into
    background strain and whitening in the same way as the supervised pipeline.

    Training batches: ``(X_whitened, y_params, empty_z)``
        X_whitened : (B, n_ifos, L)
        y_params   : (B, n_target_params)
        empty_z    : (B, 0)

    The ``waveform_sampler`` **must** be a ``WaveformLoader`` (disk-based),
    because we need the paired parameter labels from the HDF5 file.

    Args:
        target_parameters:
            Tuple of parameter names to predict, e.g. ``('chirp_mass',
            'mass_ratio')``.  Supported names: ``chirp_mass``, ``mass_ratio``,
            ``mass_1``, ``mass_2``.
        *args / **kwargs:
            All remaining arguments are forwarded to ``BaseAframeDataset``.
    """

    def __init__(
        self,
        *args,
        target_parameters: tuple[str, ...] = ("chirp_mass", "mass_ratio"),
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if not isinstance(self.waveform_sampler, WaveformLoader):
            raise ValueError(
                "RegressionTimeDomainDataset requires a WaveformLoader "
                "waveform_sampler (disk-based waveforms with parameters)."
            )
        self.target_parameters = target_parameters
        # Always load from disk so on_before_batch_transfer runs.
        self.waveforms_from_disk = True

    # ------------------------------------------------------------------ #
    # Setup: load validation params alongside waveforms                   #
    # ------------------------------------------------------------------ #

    def setup(self, stage: str) -> None:
        super().setup(stage)

        # Load val_waveforms was done by parent; load matching params here.
        world_size, rank = self.get_world_size_and_rank()
        start, stop = self.waveform_sampler.get_slice_bounds(
            self.waveform_sampler.num_val_waveforms, world_size, rank
        )
        with h5py.File(self.waveform_sampler.val_waveform_file, "r") as f:
            m1 = f["parameters/mass_1"][start:stop]
            m2 = f["parameters/mass_2"][start:stop]
        params = _compute_target_params(m1, m2, self.target_parameters)
        self.val_params = torch.tensor(params, dtype=torch.float32)

    # ------------------------------------------------------------------ #
    # Batch transfer hooks                                                 #
    # ------------------------------------------------------------------ #

    def on_before_batch_transfer(self, batch, _):
        """Slice polarizations to kernel length before device transfer."""
        if self.trainer.training:
            X, (polarizations, params) = batch
            polarizations = self.slice_waveforms(polarizations)
            batch = X, (polarizations, params)
        return batch

    def on_after_batch_transfer(self, batch, _):
        if self.trainer.training:
            [X], (waveforms, params) = batch
            batch = self.inject(X, waveforms, params)
        elif self.trainer.validating or self.trainer.sanity_checking:
            [background, _, timeslide_idx], [signals, params] = batch
            batch = self._val_inject(background, signals, params, timeslide_idx)
        return batch

    # ------------------------------------------------------------------ #
    # Injection                                                            #
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def inject(
        self,
        X: torch.Tensor,
        waveforms: torch.Tensor,
        params: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Inject waveforms into background, whiten, and attach params.

        All batch elements receive an injection (waveform_prob effectively 1).

        Args:
            X         : (B, n_ifos, L+psd_len) raw background strain
            waveforms : (M, 2, L_wf) cross/plus polarizations from loader
            params    : (M, n_params) target parameters matching waveforms

        Returns:
            X_whitened : (B, n_ifos, L)
            y_params   : (B, n_params) — subsampled to match batch size
            empty_z    : (B, 0)
        """
        X, psds = self.psd_estimator(X)
        X = self.inverter(X)
        X = self.reverser(X)

        B = X.shape[0]
        idx = torch.randperm(waveforms.shape[0])[:B]
        waveforms = waveforms[idx].to(X.device).float()
        params = params[idx].to(X.device).float()

        dec, psi, phi = self.sample_extrinsic(X)
        snrs = (
            self.snr_sampler.sample((B,)).to(X.device)
            if self.snr_sampler is not None
            else None
        )
        responses = self.projector(
            dec, psi, phi, snrs, psds, cross=waveforms[:, 0], plus=waveforms[:, 1]
        )
        kernels = sample_kernels(
            responses, kernel_size=X.size(-1), coincident=True
        )
        X = X + kernels
        X = self.whitener(X, psds)

        empty_z = torch.empty(B, 0, device=X.device)
        return X, params, empty_z

    @torch.no_grad()
    def _val_inject(
        self,
        background: torch.Tensor,
        signals: torch.Tensor,
        params: torch.Tensor,
        timeslide_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build a whitened validation batch paired with parameters.

        Injects each validation signal into a background kernel (first view
        of ``build_val_batches``), whitens, and returns in regression format.
        """
        X_bg, X_fg, psds = super().build_val_batches(background, signals)
        # X_fg: (num_valid_views, N, n_ifos, L)
        # Use the first view; whiten it with the estimated PSDs.
        X_inj = self.whitener(X_fg[0], psds)  # (N, n_ifos, L)
        empty_z = torch.empty(len(X_inj), 0, device=X_inj.device)
        return X_inj, params, empty_z

    # ------------------------------------------------------------------ #
    # Dataloaders                                                          #
    # ------------------------------------------------------------------ #

    def train_dataloader(self) -> ZippedDataset:
        from ml4gw.dataloading import Hdf5TimeSeriesDataset

        bg_dataset = Hdf5TimeSeriesDataset(
            self.train_fnames,
            channels=self.hparams.ifos,
            kernel_size=int(self.hparams.sample_rate * self.sample_length),
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

        waveform_loader = _WaveformParamLoader(
            fnames=self.waveform_sampler.training_waveform_files,
            target_parameters=self.target_parameters,
            batch_size=self.hparams.chunk_size,
            batches_per_epoch=self.hparams.chunks_per_epoch or 1,
        )

        world_size, _ = self.get_world_size_and_rank()
        batches_per_epoch = self.hparams.batches_per_epoch // world_size
        batches_per_chunk = (
            int(batches_per_epoch // self.hparams.chunks_per_epoch) + 1
        )
        self._logger.info(
            f"Regression training on pool of {waveform_loader.total} "
            f"waveforms. Sampling {batches_per_chunk} batches per chunk "
            f"from {self.hparams.chunks_per_epoch} chunks of size "
            f"{self.hparams.chunk_size} each epoch."
        )

        wf_dl = torch.utils.data.DataLoader(
            waveform_loader,
            num_workers=2,
            pin_memory=pin_memory,
            persistent_workers=True,
        )
        wf_dataset = _ChunkedWaveformParamDataset(
            wf_dl,
            batch_size=self.hparams.batch_size,
            batches_per_chunk=batches_per_chunk,
        )
        return ZippedDataset(bg_loader, wf_dataset)

    def val_dataloader(self):
        """Validation dataloader returning regression-format batches."""
        background_dataset = pl.utilities.combined_loader.CombinedLoader(
            self.timeslides, mode="sequential"
        )
        iter(background_dataset)  # gives it a __len__ property

        num_waveforms = len(self.val_waveforms)
        signal_batch_size = (num_waveforms - 1) // self.valid_loader_length + 1
        # Include params alongside waveforms so on_after_batch_transfer can
        # pair them.
        signal_dataset = torch.utils.data.TensorDataset(
            self.val_waveforms, self.val_params
        )
        signal_loader = torch.utils.data.DataLoader(
            signal_dataset,
            batch_size=signal_batch_size,
            shuffle=False,
            pin_memory=False,
        )
        return ZippedDataset(
            background_dataset,
            signal_loader,
            minimum=min(self.valid_loader_length, len(signal_loader)),
        )
