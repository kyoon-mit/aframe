import logging
import random
import time
from pathlib import Path
from typing import List
from zlib import adler32

import numpy as np
import torch
from gwpy.timeseries import TimeSeriesDict


def convert_to_detector_frame(samples: dict[str, np.ndarray]):
    """Converts mass parameters from source to detector frame"""
    for key in ["mass_1", "mass_2", "chirp_mass", "total_mass"]:
        if key in samples:
            samples[key] = samples[key] * (1 + samples["redshift"])
    return samples


def seed_worker(start: float, stop: float, shifts: List[float], seed: int):
    fingerprint = str((start, stop) + tuple(shifts))
    worker_hash = adler32(fingerprint.encode())
    logging.info(
        "Seeding data generation with seed {}, "
        "augmented by worker seed {}".format(seed, worker_hash)
    )
    np.random.seed(seed + worker_hash)
    random.seed(seed + worker_hash)


def calc_segment_injection_times(
    start: float,
    stop: float,
    spacing: float,
    buffer: float,
    waveform_duration: float,
):
    """
    Calculate the times at which to inject signals into a segment

    Args:
        start:
            The start time of the segment
        stop:
            The stop time of the segment
        spacing:
            The amount of time, in seconds, to leave between the end
            of one signal and the start of the next
        buffer:
            The amount of time, in seconds, on either side of the
            segment within which injection times will not be
            generated
        waveform_duration:
            The duration of the waveform in seconds

    Returns: np.ndarray of injection times
    """

    buffer += waveform_duration // 2
    spacing += waveform_duration
    injection_times = np.arange(start + buffer, stop - buffer, spacing)
    return injection_times


def load_psds(
    background: Path,
    ifos: List[str],
    df: float,
    sample_rate: float = None,
) -> torch.Tensor:
    """Estimate per-ifo PSDs from background strain files.

    ``background`` may be a single HDF5 file, a directory of
    ``background-*.hdf5`` files, or a list of files. With several files the
    result is the duration-weighted average of each file's median PSD, so
    the estimate represents the whole observing stretch rather than one
    segment.

    ``sample_rate`` is the rate of the waveforms the PSD will be compared
    against. When the background is sampled faster than that, its PSD
    extends beyond the waveform band and must be truncated to the
    waveform's Nyquist frequency. Without the truncation, downstream SNR
    integrals (ml4gw ``compute_ifo_snr``) interpolate the full-band PSD
    onto the waveform frequency grid, which squeezes the frequency axis
    and yields wrong SNRs.
    """
    if isinstance(background, (str, Path)):
        background = Path(background)
        if background.is_dir():
            fnames = sorted(background.glob("background-*.hdf5"))
        else:
            fnames = [background]
    else:
        fnames = [Path(f) for f in background]
    if not fnames:
        raise FileNotFoundError(f"No background files in {background}")

    weighted_sum, total_duration = None, 0.0
    for fname in fnames:
        strain = TimeSeriesDict.read(fname, path=ifos)
        duration = float(strain[ifos[0]].duration.value)
        # segments shorter than the FFT length can't produce a PSD
        # (Welch window would exceed the data); skip them
        if duration < 1 / df:
            logging.info(
                f"Skipping {fname}: {duration:.0f}s is shorter "
                f"than the {1 / df:.0f}s PSD fft length"
            )
            continue
        psd_stack = np.stack(
            [
                strain[ifo].psd(1 / df, window="hann", method="median").value
                for ifo in ifos
            ]
        )
        if weighted_sum is None:
            weighted_sum = psd_stack * duration
        else:
            weighted_sum += psd_stack * duration
        total_duration += duration

    if weighted_sum is None:
        raise ValueError(
            f"No background file in {background} is at least "
            f"{1 / df:.0f}s long; can't estimate a PSD"
        )
    psds = torch.tensor(weighted_sum / total_duration, dtype=torch.float64)
    if sample_rate is not None:
        num_bins = int(sample_rate / 2 / df) + 1
        psds = psds[:, :num_bins]
    return psds


def io_with_blocking(f, fname, timeout=10):
    """
    Function that assists with multiple processes writing to the same file
    """
    start_time = time.time()
    while True:
        try:
            return f(fname)
        except BlockingIOError:
            if (time.time() - start_time) > timeout:
                raise
