"""How many time shifts does the diagnostic set need?

Answers the question with aframe's own sizing function,
utils.data.get_num_shifts_from_num_signals (the one the sensitive-volume
pipeline uses to deploy testing waveforms), fed with the real segment
spans of the background directory and the settings from the waveform
config. For comparison it also prints the exact per-slide enumeration
(the injection-time arithmetic testing_waveforms actually performs), so
any difference between aframe's continuum approximation and the exact
count is visible.

Usage:
    python compute_num_shifts.py [--config ../configs/waveform_configs.json]
"""

import argparse
import glob
import json
import math
from pathlib import Path

import h5py

from utils.data import get_num_shifts_from_num_signals

DEFAULT_CONFIG = (
    Path(__file__).resolve().parent.parent
    / "configs"
    / "waveform_configs.json"
)


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    return parser.parse_args()


def segment_spans(background_dir):
    """(start, stop) GPS span of every background file."""
    spans = []
    for fname in sorted(glob.glob(f"{background_dir}/background-*.hdf5")):
        with h5py.File(fname) as f:
            start = float(f["H1"].attrs["x0"])
            duration = len(f["H1"]) * float(f["H1"].attrs["dx"])
        spans.append((start, start + duration))
    return spans


def events_per_slide(durations, shift, waveform_duration, spacing, buffer):
    """Exact count testing_waveforms produces for one L1 shift."""
    guard = buffer + waveform_duration // 2
    step = spacing + waveform_duration
    return sum(
        max(0, math.ceil((duration - shift - 2 * guard) / step))
        for duration in durations
    )


def main():
    arguments = parse_arguments()
    with open(arguments.config) as config_file:
        sections = json.load(config_file)
    config = {**sections["shared"], **sections["diagnostic"]}

    spans = segment_spans(config["background_dir"])
    durations = [stop - start for start, stop in spans]
    print(f"{len(spans)} segments, {sum(durations):,.0f} s total livetime")

    num_shifts = get_num_shifts_from_num_signals(
        spans,
        config["target_events"],
        config["waveform_duration"],
        config["spacing"],
        shift=1,
        buffer=config["buffer"],
    )
    print(f"aframe get_num_shifts_from_num_signals: {num_shifts} shifts")

    total = 0
    for shift in range(1, num_shifts + 3):
        count = events_per_slide(
            durations,
            shift,
            config["waveform_duration"],
            config["spacing"],
            config["buffer"],
        )
        total += count
        marker = (
            "  <- target reached"
            if total >= config["target_events"]
            and total - count < config["target_events"]
            else ""
        )
        print(
            f"  shift {shift:2d}s: {count} events, cumulative {total}{marker}"
        )


if __name__ == "__main__":
    main()
