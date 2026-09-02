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
import pandas as pd
from scipy.signal.windows import gaussian

from infer.postprocess import Postprocessor
from ledger.events import RecoveredInjectionSet
from ledger.injections import (
    InterferometerResponseSet,
    waveform_class_factory,
)

# metric names -> how to fold the raw [mass, sigma] channels into one score.
# The score is the detection statistic: larger = more signal-like. All are
# computed offline from the cached 2-channel model output, so no re-serve is
# ever needed to try a new one.
METRICS = (
    "sigma",
    "score",
    "mass_over_sigma",
    "mass_minus_ksigma",
    "mass2_over_sigma",
    "inv_sigma",
    "neg_log_sigma",
    "mass_over_sigma2",
    "mass2_over_sigma2",
    "log_mass_over_sigma",
)


def apply_metric(scores, metric, ksigma):
    """Collapse the raw model output into a 1-D detection statistic.

    A 2-D cache is [mass, sigma] per window; a 1-D cache is the served scalar
    directly -- the regression's -sigma or the classifier's logit -- and is
    passed straight through (``sigma``/``score``, higher = more signal-like).
    """
    scores = np.asarray(scores)
    if scores.ndim == 1:
        if metric not in ("sigma", "score"):
            raise SystemExit(
                f"metric {metric!r} needs the 2-channel [mass, sigma] cache; "
                "this timeseries has only a single served channel"
            )
        return scores  # served scalar used directly
    mass = scores[:, 0]
    sigma = scores[:, 1]
    safe_sigma = np.clip(sigma, 1e-8, None)
    if metric == "sigma":
        return -sigma
    if metric == "mass_over_sigma":
        return mass / safe_sigma
    if metric == "mass_minus_ksigma":
        return mass - ksigma * sigma
    if metric == "mass2_over_sigma":
        return mass**2 / safe_sigma
    # pure-confidence variants: signal -> small sigma -> large statistic
    if metric == "inv_sigma":
        return 1.0 / safe_sigma
    if metric == "neg_log_sigma":
        return -np.log(safe_sigma)
    # inverse-variance weightings (penalize uncertainty harder than 1/sigma)
    if metric == "mass_over_sigma2":
        return mass / safe_sigma**2
    if metric == "mass2_over_sigma2":
        return mass**2 / safe_sigma**2
    # log to compress the mass dynamic range before the sigma weighting
    if metric == "log_mass_over_sigma":
        return np.log(np.clip(mass, 1e-8, None)) / safe_sigma
    raise SystemExit(f"unknown metric {metric!r}")


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


class ZScorePostprocessor(Postprocessor):
    """Normalize -sigma by a running local baseline, then boxcar-integrate.

    Background -sigma wanders slowly; a signal is a sharp ~1 s peak. The
    z-score (score minus running median, over running MAD) measures peak
    height against the LOCAL noise level, so slow wander stops mattering.
    """

    def __init__(self, *args, baseline_seconds=30.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.baseline_seconds = baseline_seconds

    def integrate(self, scores):
        window = int(self.baseline_seconds * self.inference_sampling_rate)
        series = pd.Series(scores)
        baseline = series.rolling(window, center=True, min_periods=1).median()
        deviation = (series - baseline).abs()
        mad = deviation.rolling(window, center=True, min_periods=1).median()
        zscore = (series - baseline) / (1.4826 * mad + 1e-12)
        return super().integrate(zscore.to_numpy())


class TemplatePostprocessor(Postprocessor):
    """Matched-filter the scores with the measured signal dip template."""

    def __init__(self, *args, template=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.template = template

    def integrate(self, scores):
        kernel = self.template[::-1]  # correlation via convolution
        filtered = np.convolve(scores, kernel, mode="full")
        return filtered[: -len(kernel) + 1]


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--results", required=True, help="dir holding branch_* subdirs"
    )
    parser.add_argument(
        "--metric",
        choices=METRICS,
        default="sigma",
        help="how to fold raw [mass, sigma] into one detection statistic",
    )
    parser.add_argument(
        "--ksigma",
        type=float,
        default=2.0,
        help="k in mass - k*sigma (only for --metric mass_minus_ksigma)",
    )
    parser.add_argument(
        "--integration",
        choices=["boxcar", "gaussian", "zscore", "template"],
        default="boxcar",
        help="filter/statistic used on the raw scores",
    )
    parser.add_argument(
        "--gaussian-std",
        type=float,
        default=4.0,
        help="gaussian width, in samples (only for --integration gaussian)",
    )
    parser.add_argument(
        "--zscore-baseline",
        type=float,
        default=30.0,
        help="running-baseline window in seconds (--integration zscore)",
    )
    parser.add_argument(
        "--template-file",
        default=None,
        help="npy dip template at 16 Hz (--integration template)",
    )
    parser.add_argument(
        "--tag",
        default=None,
        help="output tag override (default derived from the options)",
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
    if arguments.integration == "zscore":
        return ZScorePostprocessor(
            baseline_seconds=arguments.zscore_baseline, **settings
        )
    if arguments.integration == "template":
        template = np.load(arguments.template_file)
        return TemplatePostprocessor(template=template, **settings)
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


def integration_tag(arguments):
    """Short label appended to output filenames, e.g. boxcar / gaussian4."""
    if arguments.tag:
        return arguments.tag
    if arguments.integration == "gaussian":
        return f"gaussian{arguments.gaussian_std:g}"
    if arguments.integration == "zscore":
        return f"zscore{arguments.zscore_baseline:g}"
    if arguments.integration == "template":
        return "template"
    return "boxcar"


def main():
    arguments = parse_arguments()
    tag = integration_tag(arguments)
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

        # fold the raw [mass, sigma] channels into the chosen 1-D statistic
        background_scores = apply_metric(
            background_scores, arguments.metric, arguments.ksigma
        )
        if foreground_scores is not None:
            foreground_scores = apply_metric(
                foreground_scores, arguments.metric, arguments.ksigma
            )

        postprocessor = build_postprocessor(arguments, attributes)
        background = postprocessor(background_scores)
        if foreground_scores is None:
            foreground = RecoveredInjectionSet()
        else:
            foreground_events = postprocessor(foreground_scores)
            foreground = recover_injections(foreground_events, attributes)

        background.write(os.path.join(branch_dir, f"background_{tag}.hdf5"))
        foreground.write(os.path.join(branch_dir, f"foreground_{tag}.hdf5"))
        print(
            f"{os.path.basename(branch_dir)}: "
            f"{len(background)} background, {len(foreground)} recovered",
            flush=True,
        )
    print(
        f"done: {len(branch_files)} branches -> "
        f"background_{tag}.hdf5 / foreground_{tag}.hdf5",
        flush=True,
    )


if __name__ == "__main__":
    main()
