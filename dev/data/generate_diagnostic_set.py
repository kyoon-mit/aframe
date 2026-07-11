"""Generate the diagnostic set: what the inference pipeline sees, on disk.

Signals are drawn from the astrophysical end-of-O3 rates-and-populations
BNS prior (source frame; the pipeline scales masses by 1+z into the
detector frame) and generated with PyCBC's IMRPhenomPv2 at the raw
background sample rate. Each draw's network SNR is measured against one
duration-weighted average of the median PSDs of every background segment,
and draws below the SNR threshold are rejected and redrawn. The
background is real strain: for a time slide of k seconds, L1 is read k
seconds later than H1 so the detectors see non-coincident noise, exactly
as the offline inference pipeline slides its background. All accepted
signals of a slide are injected into the full shifted strain in one pass
(so neighboring signals bleed into each other's windows, as they would in
continuous strain) and only then are the per-event windows cut, with the
coalescence sitting right_pad seconds from the right edge. Injection
happens at the background's native rate before any downsampling, matching
the inference order of operations.

Functionally, this file is only an orchestrator: for every (background
segment, L1 shift) pair it calls data.waveforms.testing.testing_waveforms
with save_background=True — the same routine the sensitive-volume
pipeline's DeployTestingWaveforms task deploys — and finally merges the
per-call files into one diagnostic.hdf5 holding background/<ifo> (noise
alone), injected/<ifo> (noise plus signals), and parameters. It differs
from the sensitive-volume pipeline in what it keeps: SV stores only the
signal waveforms (injected into strain later, on the fly) plus rejected
draws for volume normalization, while this set stores the ready-made
noise and noise+signal windows so models can be scored on them directly,
and drops the signal-only group.

Usage:
    python scripts/generate_diagnostic_set.py \
        [--config dev/configs/waveform_configs.json] [--pool 32]
All physics and bookkeeping arguments live in the json config under the
"diagnostic" key: background_dir, output_dir, ifos, prior,
waveform_approximant, target_events, l1_shifts, output_sample_rate (the
rate signals are generated at and windows are stored at; keep it equal
to the raw background rate so no resampling happens),
waveform_duration, right_pad, spacing, buffer, minimum_frequency,
reference_frequency, highpass, lowpass, snr_threshold, max_num_samples,
seed. --pool (worker processes) is the only command line override.
"""

import argparse
import glob
import importlib
import json
import logging
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import h5py

from data.waveforms.testing import testing_waveforms
from data.waveforms.utils import load_psds

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s"
)
log = logging.getLogger(__name__)

DEFAULT_CONFIG = (
    Path(__file__).resolve().parent.parent
    / "dev"
    / "configs"
    / "waveform_configs.json"
)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--pool", type=int, default=None)
    return parser.parse_args()


def load_prior_function(dotted_path):
    module_path, _, function_name = dotted_path.rpartition(".")
    return getattr(importlib.import_module(module_path), function_name)


def gps_segments(background_dir):
    """(start, end, file) GPS span of every background segment."""
    segments = []
    for fname in sorted(glob.glob(f"{background_dir}/background-*.hdf5")):
        with h5py.File(fname) as f:
            start = float(f["H1"].attrs["x0"])
            duration = len(f["H1"]) * float(f["H1"].attrs["dx"])
        segments.append((start, start + duration, fname))
    return segments


def generate_one_slide(
    config, segment, l1_shift, part_dir, reference_psd, executor
):
    """Run testing_waveforms for one (segment, L1 shift); return its file."""
    segment_start, segment_end, segment_file = segment
    waveform_file, _ = testing_waveforms(
        start=segment_start,
        end=segment_end,
        ifos=config["ifos"],
        shifts=[0.0, float(l1_shift)],
        spacing=config["spacing"],
        buffer=config["buffer"],
        prior=load_prior_function(config["prior"]),
        minimum_frequency=config["minimum_frequency"],
        reference_frequency=config["reference_frequency"],
        sample_rate=config["output_sample_rate"],
        waveform_duration=config["waveform_duration"],
        waveform_approximant=config["waveform_approximant"],
        right_pad=config["right_pad"],
        highpass=config["highpass"],
        lowpass=config["lowpass"],
        snr_threshold=config["snr_threshold"],
        psd_file=segment_file,
        max_num_samples=config["max_num_samples"],
        output_dir=part_dir,
        seed=config["seed"],
        executor=executor,
        psd=reference_psd,
        save_background=True,
    )
    return Path(waveform_file)


def merge_parts(part_files, merged_file, skip_groups=("waveforms",)):
    """Concatenate the parts' datasets into one file, one part at a time."""
    part_lengths = []
    for part in part_files:
        with h5py.File(part) as f:
            part_lengths.append(int(f.attrs["length"]))
    total_events = sum(part_lengths)

    with h5py.File(part_files[0]) as first:
        attributes = dict(first.attrs)
        dataset_shapes = {
            (group, name): first[group][name].shape
            for group in first
            if isinstance(first[group], h5py.Group)
            and group not in skip_groups
            for name in first[group]
        }

    with h5py.File(merged_file, "w") as merged:
        merged.attrs.update(attributes)
        merged.attrs["length"] = total_events
        merged.attrs["num_injections"] = total_events
        datasets = {
            key: merged.require_group(key[0]).create_dataset(
                key[1], shape=(total_events,) + shape[1:], dtype="f8"
            )
            for key, shape in dataset_shapes.items()
        }
        row = 0
        for part, length in zip(part_files, part_lengths, strict=True):
            with h5py.File(part) as f:
                for (group, name), dataset in datasets.items():
                    dataset[row : row + length] = f[group][name][:]
            row += length
    return total_events


def main():
    arguments = parse_arguments()
    with open(arguments.config) as config_file:
        sections = json.load(config_file)
    # fixed waveform specs live in "shared"; "diagnostic" adds what is
    # specific to this set and overrides shared keys where both exist
    config = {**sections["shared"], **sections["diagnostic"]}

    segments = gps_segments(config["background_dir"])
    log.info(f"{len(segments)} segments in {config['background_dir']}")

    log.info("averaging the median PSD of every segment (duration-weighted)")
    reference_psd = load_psds(
        config["background_dir"],
        config["ifos"],
        df=1 / config["waveform_duration"],
        sample_rate=config["output_sample_rate"],
    )

    output_dir = Path(config["output_dir"])
    scratch_dir = output_dir / "_parts"
    if scratch_dir.exists():
        shutil.rmtree(scratch_dir)
    scratch_dir.mkdir(parents=True)

    pool = arguments.pool
    executor = ProcessPoolExecutor(max_workers=pool) if pool else None
    part_files, total_events = [], 0
    try:
        for l1_shift in config["l1_shifts"]:
            log.info(f"time slide: L1 shifted by {l1_shift}s")
            for index, segment in enumerate(segments):
                if total_events >= config["target_events"]:
                    break
                part_dir = (
                    scratch_dir / f"shift{l1_shift:03.0f}_seg{index:02d}"
                )
                part_dir.mkdir()
                part_file = generate_one_slide(
                    config,
                    segment,
                    l1_shift,
                    part_dir,
                    reference_psd,
                    executor,
                )
                with h5py.File(part_file) as f:
                    total_events += int(f.attrs["length"])
                part_files.append(part_file)
                log.info(
                    f"L1+{l1_shift}s segment {index}: "
                    f"{total_events}/{config['target_events']} events"
                )
    finally:
        if executor is not None:
            executor.shutdown()

    if total_events < config["target_events"]:
        log.info("l1_shifts exhausted before target_events; merging anyway")

    merged_file = output_dir / "diagnostic.hdf5"
    log.info(f"merging {len(part_files)} part files -> {merged_file}")
    total_events = merge_parts(part_files, merged_file)
    shutil.rmtree(scratch_dir)
    log.info(f"DONE: {total_events} events in {merged_file}")


if __name__ == "__main__":
    main()
