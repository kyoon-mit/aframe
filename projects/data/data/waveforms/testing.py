import logging
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np
from jsonargparse import ArgumentParser

import data.waveforms.utils as utils
from data.waveforms.rejection import rejection_sample
from ledger.injections import InterferometerResponseSet, waveform_class_factory


def testing_waveforms(
    start: float,
    end: float,
    ifos: List[str],
    shifts: List[float],
    spacing: float,
    buffer: float,
    prior: Callable,
    minimum_frequency: float,
    reference_frequency: float,
    sample_rate: float,
    waveform_duration: float,
    waveform_approximant: str,
    right_pad: float,
    highpass: float,
    lowpass: float,
    snr_threshold: float,
    psd_file: Path,
    max_num_samples: int,
    output_dir: Path,
    jitter: float = 0.1,
    seed: Optional[int] = None,
    pool: Optional[int] = None,
    chunksize: Optional[int] = None,
    executor=None,
    psd=None,
    save_background: bool = False,
):
    """
    Generates testing waveforms via rejection sampling
    for a single segment.

    Args:
        start:
            GPS time of the beginning of the testing segment
        end:
            GPS time of the end of the testing segment
        ifos:
            List of interferometers to query data from. Expected to be given
            by prefix; e.g. "H1" for Hanford. Should be the same length as
            `shifts`
        shifts:
            The length of time in seconds by which each interferometer's
            timeseries will be shifted
        spacing:
            The amount of time, in seconds, to leave between the end
            of one signal and the start of the next
        buffer:
            The amount of time, in seconds, on either side of the
            segment within which injection times will not be
            generated
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
        sample_rate:
            Sample rate of timeseries data, specified in Hz
        waveform_duration:
            Duration of waveform in seconds
        waveform_approximant:
            Name of the waveform approximant to use.
        right_pad:
            Location of the defining point of the signal within
            the generated waveform relative to the right edge
            of the waveform (in seconds).
        highpass:
            The frequency to use for a highpass filter, specified
            in Hz
        lowpass:
            The frequency to use for a lowpass filter, specified
            in Hz
        snr_threshold:
            Minimum SNR of generated waveforms. Sampled parameters
            that result in an SNR below this threshold will be rejected,
            but saved for later use
        psd_file:
            Background file from which to calculate PSDs used for
            estimating waveforms SNR
        max_num_samples:
            Maximum number of samples to generate at once in the rejection
            sampling process.
        output_dir:
            Directory to which the waveform file and rejected parameter
            file will be written
        jitter:
            Scale of random jitter to add to injection times
        seed:
            Random seed to use for reproducibility

    Returns:
        The name of the waveform file and the name of the file containing the
        rejected parameters
    """

    if len(ifos) != len(shifts):
        raise ValueError(
            "Number of ifos must match number of shifts"
            f"got {len(ifos)} ifos and {len(shifts)} shifts"
        )

    # seed process based on start, end and shift
    if seed is not None:
        utils.seed_worker(start, end, shifts, seed)

    # calculate the injection times, determining
    # the number of samples we'll need to generate
    injection_times = utils.calc_segment_injection_times(
        start,
        end - max(shifts),  # TODO: should account for uneven last batch too
        spacing,
        buffer,
        waveform_duration,
    )
    num_signals = len(injection_times)

    # add random jitter to injection times
    jitter = np.random.uniform(-jitter, jitter, size=num_signals)
    injection_times += jitter

    # calculate psd that will be used for snr calculation, unless the
    # caller already provides one (e.g. an average over many segments)
    if psd is None:
        df = 1 / waveform_duration
        logging.info(f"Using background file {psd_file} for psd calculation")
        psd = utils.load_psds(psd_file, ifos, df=df, sample_rate=sample_rate)

    # perform the rejection sampling
    parameters, rejected_params = rejection_sample(
        num_signals=num_signals,
        prior=prior,
        ifos=ifos,
        minimum_frequency=minimum_frequency,
        reference_frequency=reference_frequency,
        sample_rate=sample_rate,
        waveform_duration=waveform_duration,
        waveform_approximant=waveform_approximant,
        right_pad=right_pad,
        highpass=highpass,
        lowpass=lowpass,
        snr_threshold=snr_threshold,
        psd=psd,
        max_num_samples=max_num_samples,
        pool=pool,
        chunksize=chunksize,
        executor=executor,
    )

    # create the ResponseSet dataclass based on the passed ifos
    ResponseSet = waveform_class_factory(
        ifos,
        InterferometerResponseSet,
        cls_name="ResponseSet",
    )

    # now, set the injection times and shifts,
    # and create the ResponseSet object
    parameters["injection_time"] = injection_times
    parameters["shift"] = np.array([shifts for _ in range(num_signals)])

    output_dir.mkdir(parents=True, exist_ok=True)
    response_set = ResponseSet(**parameters)
    waveform_fname = output_dir / "waveforms.hdf5"
    utils.io_with_blocking(response_set.write, waveform_fname)

    rejected_fname = output_dir / "rejected-parameters.hdf5"
    utils.io_with_blocking(rejected_params.write, rejected_fname)

    if save_background:
        _save_background_and_injected(
            waveform_fname,
            response_set,
            psd_file,
            ifos,
            shifts,
            sample_rate,
            waveform_duration,
            right_pad,
        )

    # TODO: compute probability of all parameters against
    # source and all target priors here then save them somehow
    return waveform_fname, rejected_fname


def _save_background_and_injected(
    fname,
    response_set,
    strain_file,
    ifos,
    shifts,
    sample_rate,
    waveform_duration,
    right_pad,
):
    """Append per-event background-only and background+injection windows.

    Reproduces what the inference pipeline sees for this time-slide: each
    ifo's strain is shifted by its entry in ``shifts`` (shifted index p
    reads raw index p + shift, matching the offline pipeline), ALL of the
    slide's signals are injected into that shifted strain at once (so
    windows include any bleed-over from neighboring injections), and a
    ``waveform_duration``-long window is cut around every injection with
    the coalescence ``right_pad`` seconds from the right edge.

    Writes two groups to ``fname``: ``background/<ifo>`` (clean shifted
    noise) and ``injected/<ifo>`` (same noise plus signals), each of shape
    ``(num_signals, sample_rate * waveform_duration)``.

    Assumes ``strain_file`` spans the injection times, i.e.
    testing_waveforms was called with ``start``/``end`` inside a single
    background segment.
    """
    import h5py
    from scipy.signal import resample_poly

    window_size = int(sample_rate * waveform_duration)
    # samples from the window start to the coalescence
    left = int((waveform_duration - right_pad) * sample_rate)

    # read + resample each ifo's strain to the signal sample rate
    raws = []
    with h5py.File(strain_file, "r") as f:
        segment_start = float(f[ifos[0]].attrs["x0"])
        for ifo in ifos:
            raw = f[ifo][:]
            raw_rate = int(round(1 / f[ifo].attrs["dx"]))
            if raw_rate != int(sample_rate):
                raw = resample_poly(raw, int(sample_rate), raw_rate)
            raws.append(raw.astype(np.float64))

    # apply the time-slide: after shifting, index p of every ifo refers
    # to the same analysis time, with the shifted ifos reading noise from
    # `shift` seconds later in their raw timeseries
    shift_samples = [int(s * sample_rate) for s in shifts]
    max_shift = max(shift_samples)
    num_valid = min(len(raw) for raw in raws) - max_shift
    background = np.stack(
        [
            raw[shift : shift + num_valid]
            for raw, shift in zip(raws, shift_samples, strict=False)
        ]
    )

    # inject every accepted signal into the full shifted strain in one
    # pass so neighboring injections bleed into each other's windows,
    # exactly as they would in the continuous inference strain
    injected = response_set.inject(background.copy(), segment_start)
    injected = injected[:, :num_valid]

    num_signals = len(response_set)
    bg_windows = np.empty((num_signals, len(ifos), window_size))
    inj_windows = np.empty((num_signals, len(ifos), window_size))
    for k, t in enumerate(response_set.injection_time):
        start = int(round((t - segment_start) * sample_rate)) - left
        if start < 0 or start + window_size > num_valid:
            raise ValueError(
                f"injection {k} at t={t:.1f} falls outside the usable "
                f"span of {strain_file}; save_background needs the "
                "segment to cover start..end minus the maximum shift."
            )
        bg_windows[k] = background[:, start : start + window_size]
        inj_windows[k] = injected[:, start : start + window_size]

    with h5py.File(fname, "a") as f:
        bg_group = f.create_group("background")
        inj_group = f.create_group("injected")
        for j, ifo in enumerate(ifos):
            bg_group.create_dataset(ifo.lower(), data=bg_windows[:, j])
            inj_group.create_dataset(ifo.lower(), data=inj_windows[:, j])


parser = ArgumentParser()
parser.add_function_arguments(testing_waveforms)


def main(args):
    args = args.testing_waveforms.as_dict()
    testing_waveforms(**args)
