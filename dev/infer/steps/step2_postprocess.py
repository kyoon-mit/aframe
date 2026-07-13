"""Step 2: turn the cached raw scores into events, with a chosen filter.

Reads every ``branch_*/timeseries.hdf5`` written by step 1, applies the chosen
integration filter (boxcar or gaussian), clusters, recovers injections, and
rewrites ``branch_*/background.hdf5`` and ``branch_*/foreground.hdf5``.

Runs on CPU in seconds, so you can sweep filters without touching Triton.

Note: the boxcar is causal (past data only). A gaussian is symmetric, so it
also uses samples slightly ahead of each point.

Example:
    uv run python step2_postprocess.py \\
        --results /path/to/results \\
        --integration gaussian --gaussian-std 4
"""

import argparse
import glob
import os

import h5py
import numpy as np
from scipy.signal.windows import gaussian

from infer.postprocess import Postprocessor
from ledger.events import RecoveredInjectionSet
from ledger.injections import (
    InterferometerResponseSet,
    waveform_class_factory,
)


class GaussianPostprocessor(Postprocessor):
    """Same as Postprocessor, but integrates with a gaussian kernel."""

    def __init__(self, *args, gaussian_std=4.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.gaussian_std = gaussian_std

    def integrate(self, scores):
        window_size = self.integration_window_size
        window = gaussian(window_size, self.gaussian_std)
        window /= window.sum()
        integrated = np.convolve(scores, window, mode="full")
        return integrated[: -window_size + 1]


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--results", required=True, help="dir holding branch_* subdirs"
    )
    parser.add_argument(
        "--integration",
        choices=["boxcar", "gaussian"],
        default="boxcar",
        help="filter used to integrate the raw scores",
    )
    parser.add_argument(
        "--gaussian-std",
        type=float,
        default=4.0,
        help="gaussian width, in samples (only for --integration gaussian)",
    )
    parser.add_argument(
        "--integration-width",
        type=float,
        default=None,
        help="override the integration window length, in seconds",
    )
    parser.add_argument(
        "--cluster-width",
        type=float,
        default=None,
        help="override the cluster window length, in seconds",
    )
    return parser.parse_args()


def build_postprocessor(arguments, attributes):
    """Make the postprocessor for one branch from its saved settings."""
    integration_width = arguments.integration_width
    if integration_width is None:
        integration_width = float(attributes["integration_window_length"])
    cluster_width = arguments.cluster_width
    if cluster_width is None:
        cluster_width = float(attributes["cluster_window_length"])

    settings = {
        "t0": float(attributes["t0"]),
        "shifts": list(attributes["shifts"]),
        "psd_length": float(attributes["psd_length"]),
        "fduration": float(attributes["fduration"]),
        "inference_sampling_rate": float(
            attributes["inference_sampling_rate"]
        ),
        "integration_window_length": integration_width,
        "cluster_window_length": cluster_width,
    }
    if arguments.integration == "gaussian":
        return GaussianPostprocessor(
            gaussian_std=arguments.gaussian_std, **settings
        )
    return Postprocessor(**settings)


def recover_injections(foreground_events, attributes):
    """Match foreground events back to the injections for this branch."""
    ifos = [
        name.decode() if isinstance(name, bytes) else str(name)
        for name in attributes["ifos"]
    ]
    response_set = waveform_class_factory(
        ifos, InterferometerResponseSet, "ResponseSet"
    )
    t0 = float(attributes["t0"])
    injection_set = response_set.read(
        str(attributes["injection_set_fname"]),
        start=t0,
        end=t0 + float(attributes["duration"]),
        shifts=list(attributes["shifts"]),
    )
    if len(injection_set) == 0:
        return RecoveredInjectionSet()
    return RecoveredInjectionSet.recover(foreground_events, injection_set)


def main():
    arguments = parse_arguments()
    branch_files = sorted(
        glob.glob(f"{arguments.results}/branch_*/timeseries.hdf5")
    )
    if not branch_files:
        raise SystemExit(f"no timeseries.hdf5 under {arguments.results}")

    for branch_path in branch_files:
        branch_dir = os.path.dirname(branch_path)
        with h5py.File(branch_path, "r") as branch_file:
            attributes = dict(branch_file.attrs)
            background_scores = branch_file["background_ts"][:]
            foreground_scores = (
                branch_file["foreground_ts"][:]
                if "foreground_ts" in branch_file
                else None
            )

        postprocessor = build_postprocessor(arguments, attributes)
        background = postprocessor(background_scores)
        if foreground_scores is None:
            foreground = RecoveredInjectionSet()
        else:
            foreground_events = postprocessor(foreground_scores)
            foreground = recover_injections(foreground_events, attributes)

        background.write(os.path.join(branch_dir, "background.hdf5"))
        foreground.write(os.path.join(branch_dir, "foreground.hdf5"))
        print(
            f"{os.path.basename(branch_dir)}: "
            f"{len(background)} background, {len(foreground)} recovered",
            flush=True,
        )
    print(
        f"done: {len(branch_files)} branches with "
        f"{arguments.integration} integration",
        flush=True,
    )


if __name__ == "__main__":
    main()
