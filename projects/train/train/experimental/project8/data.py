"""Project 8 CRES events as (noisy, clean) I/Q pairs with truth parameters.

Each split directory holds HDF5 files of events, every event storing a clean
I and Q trace and, separately, a noise trace per channel. The noisy input is
formed here as clean + noise. Batches follow the aframe denoiser convention,

    (X, X_clean, y, params)
    X, X_clean  (B, 2, cutoff)  channel 0 = I, channel 1 = Q
    y           (B, 1)          ones: every event carries a signal
    params      dict of (B,)    truth parameters

so the denoise-and-regress tasks train on it unchanged. Each channel of the
noisy input is divided by its own standard deviation and the clean target by
the same number, as in the PhyTS Project 8 dataloader.

With ``resample_noise`` the training noise is drawn fresh on the GPU for every
batch instead of taken from the stored traces: white Gaussian at the stored
traces' per-channel level, or cavity noise synthesised from the stored
traces' complex spectrum. Validation and test always use the stored traces.

Splits are loaded into memory once; the full training split is ~16 GB, ~8 GB
when its noise is resampled.
"""

from pathlib import Path
from typing import Optional, Sequence

import h5py
import lightning.pytorch as pl
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

PARAMS = (
    "energy_eV",
    "start_carrier_frequency_Hz",
    "avg_carrier_frequency_Hz",
    "slope_Hz",
    "pitch_angle_deg",
    "avg_axial_frequency_Hz",
    "radius_m",
    "radius_input_m",
    "radius_phase",
)
CHANNELS = ("output_ts_I", "output_ts_Q")


def load_split(
    directory: Path, noise: str, offset: int, cutoff: int, with_noise: bool = True
) -> tuple[torch.Tensor, Optional[torch.Tensor], dict]:
    """Clean signal, noise and parameters of every event in a split."""
    paths = sorted(directory.glob("*.hdf5"))
    if not paths:
        raise FileNotFoundError(f"no HDF5 files in {directory}")
    window = slice(offset, offset + cutoff)
    signal, noise_trace, params = [], [], {k: [] for k in PARAMS}
    for path in paths:
        with h5py.File(path, "r") as f:
            signal.append(
                np.stack([f[c][:, window] for c in CHANNELS], 1).astype(np.float32)
            )
            if with_noise:
                noise_trace.append(
                    np.stack(
                        [f[f"{c}_{noise}_noise"][:, window] for c in CHANNELS], 1
                    ).astype(np.float32)
                )
            for k in PARAMS:
                params[k].append(f[k][:].astype(np.float32))
    return (
        torch.from_numpy(np.concatenate(signal)),
        torch.from_numpy(np.concatenate(noise_trace)) if with_noise else None,
        {k: torch.from_numpy(np.concatenate(v)) for k, v in params.items()},
    )


def noise_statistics(
    directory: Path, noise: str, offset: int, cutoff: int, n_files: int = 2
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-channel std and complex power spectrum of the stored noise.

    Returns ``sigma`` (2,) and ``power`` (cutoff,), the mean of
    ``|fft(n_I + i n_Q)|**2`` per bin, from the first ``n_files`` files.
    """
    window = slice(offset, offset + cutoff)
    total_sq, total_power, count = np.zeros(2), np.zeros(cutoff), 0
    for path in sorted(directory.glob("*.hdf5"))[:n_files]:
        with h5py.File(path, "r") as f:
            nI = f[f"{CHANNELS[0]}_{noise}_noise"]
            nQ = f[f"{CHANNELS[1]}_{noise}_noise"]
            for a in range(0, nI.shape[0], 500):
                i, q = nI[a : a + 500, window], nQ[a : a + 500, window]
                total_sq += [(i**2).sum(), (q**2).sum()]
                total_power += (np.abs(np.fft.fft(i + 1j * q, axis=-1)) ** 2).sum(0)
                count += len(i)
    sigma = np.sqrt(total_sq / (count * cutoff))
    return torch.tensor(sigma, dtype=torch.float32), torch.tensor(
        total_power / count, dtype=torch.float32
    )


def draw_noise(
    shape, kind: str, sigma: torch.Tensor, power: torch.Tensor, device
) -> torch.Tensor:
    """Fresh noise of ``shape`` (B, 2, L): white Gaussian or synthetic cavity."""
    if kind == "gauss":
        return torch.randn(shape, device=device) * sigma.to(device).view(1, 2, 1)
    batch, _, length = shape
    white = torch.complex(
        torch.randn(batch, length, device=device),
        torch.randn(batch, length, device=device),
    ) / np.sqrt(2.0)
    z = torch.fft.ifft(white * power.to(device).sqrt(), dim=-1)
    return torch.stack([z.real, z.imag], dim=1)


class Project8Dataset(Dataset):
    def __init__(self, signal, noise, params, repeats: int = 1):
        self.signal, self.noise, self.params = signal, noise, params
        self.repeats = repeats

    def __len__(self):
        return len(self.signal) * self.repeats

    def __getitem__(self, i):
        i = i % len(self.signal)
        clean = self.signal[i]
        params = {k: v[i] for k, v in self.params.items()}
        if self.noise is None:
            # noise is added on the GPU after transfer
            return clean, params
        noisy = clean + self.noise[i]
        scale = noisy.std(dim=-1, keepdim=True).clamp_min(1e-12)
        return noisy / scale, clean / scale, torch.ones(1), params


class Project8DataModule(pl.LightningDataModule):
    """Project 8 simulation dataset for the aframe denoise-and-regress tasks.

    Args:
        data_dir: directory holding the ``train``, ``valid`` and ``test``
            split directories.
        noise: noise model added to the signal, ``cav`` or ``gauss``.
        offset: first sample of the window kept from each event.
        cutoff: number of samples kept from each event, at most 24576.
        batch_size: events per batch.
        num_workers: dataloader workers.
        resample_noise: draw fresh training noise every batch instead of
            using the stored traces.
        repeats: passes over the training signals per epoch, each with its
            own noise when ``resample_noise`` is set.
    """

    def __init__(
        self,
        data_dir: str,
        noise: str = "cav",
        offset: int = 0,
        cutoff: int = 24576,
        batch_size: int = 128,
        num_workers: int = 4,
        resample_noise: bool = False,
        repeats: int = 1,
    ):
        super().__init__()
        if noise not in ("cav", "gauss"):
            raise ValueError(f"noise must be 'cav' or 'gauss', got {noise!r}")
        if repeats < 1:
            raise ValueError(f"repeats must be >= 1, got {repeats}")
        self.save_hyperparameters()
        self.splits: dict[str, Project8Dataset] = {}
        self.sigma: Optional[torch.Tensor] = None
        self.power: Optional[torch.Tensor] = None

    @property
    def num_ifos(self) -> int:
        return len(CHANNELS)

    def setup(self, stage: Optional[str] = None):
        hp = self.hparams
        wanted: Sequence[str] = ("train", "valid") if stage == "fit" else ("test",)
        if stage is None:
            wanted = ("train", "valid", "test")
        for split in wanted:
            if split in self.splits:
                continue
            resample = hp.resample_noise and split == "train"
            directory = Path(hp.data_dir) / split
            signal, noise, params = load_split(
                directory, hp.noise, hp.offset, hp.cutoff, with_noise=not resample
            )
            self.splits[split] = Project8Dataset(
                signal, noise, params, repeats=hp.repeats if split == "train" else 1
            )
            if resample:
                self.sigma, self.power = noise_statistics(
                    directory, hp.noise, hp.offset, hp.cutoff
                )

    def on_after_batch_transfer(self, batch, dataloader_idx):
        # only resampled training batches arrive as (clean, params)
        if not (isinstance(batch, (list, tuple)) and len(batch) == 2):
            return batch
        clean, params = batch
        noise = draw_noise(
            clean.shape, self.hparams.noise, self.sigma, self.power, clean.device
        )
        noisy = clean + noise
        scale = noisy.std(dim=-1, keepdim=True).clamp_min(1e-12)
        ones = torch.ones(clean.shape[0], 1, device=clean.device)
        return noisy / scale, clean / scale, ones, params

    def _loader(self, split, shuffle):
        return DataLoader(
            self.splits[split],
            batch_size=self.hparams.batch_size,
            shuffle=shuffle,
            num_workers=self.hparams.num_workers,
            persistent_workers=self.hparams.num_workers > 0,
            pin_memory=True,
            drop_last=shuffle,
        )

    def train_dataloader(self):
        return self._loader("train", shuffle=True)

    def val_dataloader(self):
        return self._loader("valid", shuffle=False)

    def test_dataloader(self):
        return self._loader("test", shuffle=False)
