"""Sources of whitened two-detector noise windows.

Each source has a dataset, which draws batches on the CPU in data-loading
workers, and a ``to_windows`` step, which turns a batch into whitened windows of
shape (windows, detectors, samples) on the GPU. Real and Gaussian noise go
through the same whitening; saved windows were whitened before saving.
"""

import glob
from pathlib import Path

import h5py
import numpy as np
import torch
from ml4gw.dataloading import Hdf5TimeSeriesDataset
from ml4gw.transforms import SpectralDensity, Whiten


class Whitening(torch.nn.Module):
    """Raw stretches to whitened windows, the same steps as aframe's training data.

    A stretch holds psd_length seconds to measure the spectrum, followed by
    kernel_length + fduration seconds to whiten. The whitening filter spoils
    fduration / 2 seconds at each end, which are cut off, leaving kernel_length
    seconds.
    """

    def __init__(
        self,
        sample_rate: float,
        kernel_length: float,
        psd_length: float,
        fduration: float,
        highpass: float,
        lowpass: float | None = None,
        fftlength: float | None = None,
    ):
        super().__init__()
        to_whiten_length = kernel_length + fduration
        self.sample_rate = sample_rate
        self.window_size = int(kernel_length * sample_rate)
        self.to_whiten_size = int(to_whiten_length * sample_rate)
        self.stretch_size = int((psd_length + to_whiten_length) * sample_rate)
        # as aframe's training data sets its PsdEstimator: median average, and the
        # fast estimate (inexact in the two lowest bins) only when there is a highpass
        self.spectral_density = SpectralDensity(
            sample_rate, fftlength or to_whiten_length, average="median", fast=highpass is not None
        )
        self.whiten = Whiten(fduration, sample_rate, highpass, lowpass)

    def psd(self, stretches: torch.Tensor) -> torch.Tensor:
        return self.spectral_density(stretches[..., : -self.to_whiten_size])

    def forward(self, stretches: torch.Tensor) -> torch.Tensor:
        return self.whiten(stretches[..., -self.to_whiten_size :], self.psd(stretches))


def background_files(directory: str, span: list[int] | None = None) -> list[str]:
    """Background files of aframe's layout, ``{directory}/background/*-{gps start}-{duration}.hdf5``.

    With ``span = [start, end]``, keeps the files whose GPS start time is in
    [start, end).
    """
    files = sorted(glob.glob(f"{directory}/background/*.hdf5"))
    if span is not None:
        files = [f for f in files if span[0] <= int(Path(f).stem.split("-")[-2]) < span[1]]
    if not files:
        raise ValueError(f"no background files in {directory} within GPS span {span}")
    return files


class TestStretches(torch.utils.data.IterableDataset):
    """Stretches for testing in which every window is used once.

    In each file the windows follow each other every ``window_size`` samples,
    in a fixed shuffled order across files. Each detector after the first is
    cut ``shift`` windows later than the first detector (wrapping around within
    the file), so the detectors are never cut at the same moment.
    """

    def __init__(
        self, files: list[str], ifos: list[str], stretch_size: int, window_size: int,
        batch_size: int, batches: int, shift: int = 1,
    ):
        self.files, self.ifos, self.shift = files, ifos, shift
        self.stretch_size, self.window_size, self.batch_size = stretch_size, window_size, batch_size
        self.places = []  # (file, window number, windows in the file)
        for name in files:
            with h5py.File(name, "r") as f:
                windows_in_file = (len(f[ifos[0]]) - stretch_size) // window_size + 1
            if windows_in_file > shift * (len(ifos) - 1):
                self.places += [(name, k, windows_in_file) for k in range(windows_in_file)]
        np.random.default_rng(0).shuffle(self.places)
        self.batches = min(batches, len(self.places) // batch_size)

    def __len__(self):
        return self.batches

    def __iter__(self):
        worker = torch.utils.data.get_worker_info()
        first, step = (0, 1) if worker is None else (worker.id, worker.num_workers)
        files = {name: h5py.File(name, "r") for name in self.files}
        for batch in range(first, self.batches, step):
            stretches = np.empty((self.batch_size, len(self.ifos), self.stretch_size), dtype=np.float32)
            places = self.places[batch * self.batch_size : (batch + 1) * self.batch_size]
            for row, (name, k, windows_in_file) in enumerate(places):
                for i, ifo in enumerate(self.ifos):
                    start = (k + i * self.shift) % windows_in_file * self.window_size
                    stretches[row, i] = files[name][ifo][start : start + self.stretch_size]
            yield torch.from_numpy(stretches)


class RealNoise:
    """Real detector noise. ``fit`` windows come from ``fit_dir`` and ``test``
    windows from ``test_dir``, each optionally limited to a GPS span. Fit windows
    are cut as in aframe's background: each detector at its own random file and
    time, so the detectors are almost never cut at the same moment (nothing
    forbids it). Test windows come from ``TestStretches``."""

    def __init__(
        self,
        ifos: list[str],
        fit_dir: str,
        test_dir: str,
        fit_span: list[int] | None = None,
        test_span: list[int] | None = None,
        num_files_per_batch: int | None = None,
    ):
        self.ifos = ifos
        self.files = {"fit": background_files(fit_dir, fit_span), "test": background_files(test_dir, test_span)}
        self.num_files_per_batch = num_files_per_batch

    def dataset(self, part: str, whitening: Whitening, batch_size: int, batches: int):
        if part == "test":
            return TestStretches(
                self.files["test"], self.ifos, whitening.stretch_size, whitening.window_size, batch_size, batches
            )
        # the sampler needs every file longer than one stretch
        files = []
        for name in self.files["fit"]:
            with h5py.File(name, "r") as f:
                if len(f[self.ifos[0]]) > whitening.stretch_size:
                    files.append(name)
        return Hdf5TimeSeriesDataset(
            files,
            channels=self.ifos,
            kernel_size=whitening.stretch_size,
            batch_size=batch_size,
            batches_per_epoch=batches,
            coincident=False,
            num_files_per_batch=self.num_files_per_batch,
        )

    def to_windows(self, stretches: torch.Tensor, whitening: Whitening) -> torch.Tensor:
        return whitening(stretches)


class GaussianNoise(RealNoise):
    """Gaussian noise with the spectrum of a real stretch, then whitened like real
    noise: the real spectrum of each moment and each detector, and nothing else
    (no glitches, no change within a stretch)."""

    def to_windows(self, stretches: torch.Tensor, whitening: Whitening) -> torch.Tensor:
        size = stretches.shape[-1]
        psd = whitening.psd(stretches)
        psd = torch.nn.functional.interpolate(psd, size=size // 2 + 1, mode="linear", align_corners=True)
        # each frequency bin gets a random complex amplitude whose squared size
        # averages sample_rate * size * psd / 2, which gives the noise the
        # one-sided spectrum psd; the zero and highest bins are left empty (the
        # highpass removes the zero bin, the whitening filter the highest)
        scale = (psd * whitening.sample_rate * size / 4).sqrt()
        amplitudes = torch.complex(scale * torch.randn_like(scale), scale * torch.randn_like(scale))
        amplitudes[..., 0] = 0
        amplitudes[..., -1] = 0
        return whitening(torch.fft.irfft(amplitudes, n=size).float())


class SavedBatches(torch.utils.data.IterableDataset):
    """Batches of saved windows from rows [start, stop) of an HDF5 file: random
    rows, or consecutive rows with each row used once when ``in_order``."""

    def __init__(self, path: str, start: int, stop: int, batch_size: int, batches: int, in_order: bool):
        if stop - start < batch_size:
            raise ValueError(f"{path}: {stop - start} windows in rows [{start}, {stop}), fewer than one batch")
        self.path, self.start, self.stop, self.batch_size, self.in_order = path, start, stop, batch_size, in_order
        self.batches = min(batches, (stop - start) // batch_size) if in_order else batches

    def __len__(self):
        return self.batches

    def __iter__(self):
        worker = torch.utils.data.get_worker_info()
        first, step = (0, 1) if worker is None else (worker.id, worker.num_workers)
        with h5py.File(self.path, "r") as f:
            windows = f["windows"]
            for batch in range(first, self.batches, step):
                if self.in_order:
                    rows = slice(self.start + batch * self.batch_size, self.start + (batch + 1) * self.batch_size)
                else:
                    # h5py reads rows only in increasing order
                    rows = self.start + np.sort(np.random.choice(self.stop - self.start, self.batch_size, replace=False))
                yield torch.from_numpy(windows[rows])


class SavedWindows:
    """Whitened windows written by ``save_windows.py``. The last ``test_fraction``
    of the file is kept for testing, read in order, each window once."""

    def __init__(self, path: str, test_fraction: float = 0.2):
        with h5py.File(path, "r") as f:
            count = len(f["windows"])
        cut = int(count * (1 - test_fraction))
        self.path = path
        self.rows = {"fit": (0, cut), "test": (cut, count)}

    def dataset(self, part: str, whitening: Whitening, batch_size: int, batches: int):
        return SavedBatches(self.path, *self.rows[part], batch_size, batches, in_order=part == "test")

    def to_windows(self, windows: torch.Tensor, whitening: Whitening) -> torch.Tensor:
        return windows


def build_source(spec: dict, ifos: list[str]):
    """A source from its config entry: ``kind`` is real, gaussian or saved; the
    other keys are the arguments of that source."""
    classes = {"real": RealNoise, "gaussian": GaussianNoise, "saved": SavedWindows}
    spec = dict(spec)
    source_class = classes[spec.pop("kind")]
    if source_class is SavedWindows:
        return source_class(**spec)
    return source_class(ifos, **spec)
