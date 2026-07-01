from concurrent.futures import ProcessPoolExecutor
from typing import Callable, Optional

from jsonargparse import ArgumentParser

from data.waveforms.utils import convert_to_detector_frame
from ledger.injections import BilbyParameterSet, WaveformPolarizationSet


def _sample_detector_frame_parameters(num_signals, prior):
    """Sample intrinsic/extrinsic parameters and return them in the detector
    frame as a ``BilbyParameterSet``."""
    prior, detector_frame_prior = prior()
    samples = prior.sample(num_signals)
    if not detector_frame_prior:
        samples = convert_to_detector_frame(samples)
    return BilbyParameterSet(**samples)


def _make_executor(pool, chunksize, work_size):
    """Build a ``ProcessPoolExecutor`` (or None) and resolve the ``ex.map``
    chunk size. ``work_size`` is the number of waveforms handed to a single
    ``map`` call (all of them for the in-memory path, one batch for the
    streaming path)."""
    if not pool:
        return None, 1
    if chunksize is None:
        # a few chunks per worker keeps them busy without paying the
        # pickling overhead of one task per waveform
        chunksize = max(1, work_size // (pool * 8))
    return ProcessPoolExecutor(max_workers=pool), chunksize


def training_waveforms(
    num_signals: int,
    sample_rate: int,
    waveform_duration: float,
    prior: Callable,
    minimum_frequency: float,
    reference_frequency: float,
    waveform_approximant: str,
    right_pad: float,
    pool: Optional[int] = None,
    chunksize: Optional[int] = None,
):
    """
    Generates random training waveforms polarizations from a
    distribution over waveform parameters, returning them in memory.

    For large banks prefer the streaming CLI path (``main``), which writes
    directly to disk without holding every waveform in memory.

    Args:
        num_signals:
            The number of signals to generate
        sample_rate:
            Sample rate of timeseries data, specified in Hz
        waveform_duration:
            Duration of waveform in seconds
        prior:
            A function that returns a Bilby PriorDict when called
        minimum_frequency:
            Minimum frequency of the gravitational wave. The part
            of the gravitational wave at lower frequencies will
            not be generated. Specified in Hz.
        reference_frequency:
            Frequency of the gravitational wave at the state of
            the merger that other quantities are defined with
            reference to
        waveform_approximant:
            Name of the waveform approximant to use.
        right_pad:
            Location of the defining point of the signal within
            the generated waveform relative to the right edge
            of the waveform (in seconds).
        pool:
            If set, the number of worker processes to use for
            generating waveforms in parallel. If None (default),
            waveforms are generated serially in the current process.
        chunksize:
            Number of waveforms dispatched to each worker at a time
            when ``pool`` is set. Defaults to a value that spreads
            the work into a handful of chunks per worker, which
            amortizes the per-task pickling/IPC overhead. Ignored
            when ``pool`` is None.

    Returns:
        An IntrinsicParameterSet generated from the sampled parameters
    """
    params = _sample_detector_frame_parameters(num_signals, prior)
    ex, chunksize = _make_executor(pool, chunksize, num_signals)
    try:
        waveforms = WaveformPolarizationSet.from_parameters(
            params,
            minimum_frequency,
            reference_frequency,
            sample_rate,
            waveform_duration,
            waveform_approximant,
            right_pad,
            ex=ex,
            chunksize=chunksize,
        )
    finally:
        if ex is not None:
            ex.shutdown()
    return waveforms


parser = ArgumentParser()
parser.add_function_arguments(training_waveforms)
parser.add_argument("--output_file", "-o", type=str)
parser.add_argument(
    "--write_batch_size",
    type=int,
    default=2048,
    help="Number of waveforms generated and flushed to the HDF5 file per "
    "iteration. Caps peak memory at roughly one batch of waveforms.",
)


def main(args):
    args = args.training_waveforms.as_dict()
    output_file = args.pop("output_file")
    write_batch_size = args.pop("write_batch_size")
    pool = args.pop("pool")
    chunksize = args.pop("chunksize")

    num_signals = args["num_signals"]
    sample_rate = args["sample_rate"]
    waveform_duration = args["waveform_duration"]

    params = _sample_detector_frame_parameters(num_signals, args["prior"])
    ex, chunksize = _make_executor(pool, chunksize, write_batch_size)
    chunks = (min(64, num_signals), int(sample_rate * waveform_duration))
    try:
        WaveformPolarizationSet.from_parameters_to_file(
            params,
            minimum_frequency=args["minimum_frequency"],
            reference_frequency=args["reference_frequency"],
            sample_rate=sample_rate,
            waveform_duration=waveform_duration,
            waveform_approximant=args["waveform_approximant"],
            right_pad=args["right_pad"],
            output_file=output_file,
            write_batch_size=write_batch_size,
            ex=ex,
            chunksize=chunksize,
            chunks=chunks,
        )
    finally:
        if ex is not None:
            ex.shutdown()
