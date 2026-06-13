"""Compare two detection statistics for the regression model with ROC curves.

Runs a trained chirp-mass regression model over two kinds of strain windows,
drawn straight from a checkpoint -- windows holding an injected signal, and
windows of pure background -- and scores each with two candidate detection
statistics:

    -sigma        negative predicted uncertainty (the current statistic). A
                  confident, low-uncertainty prediction looks signal-like.
    |mu| / sigma  the standardized predicted mean. The mean is normalized so its
                  no-signal value is 0, so this is large only when the model
                  commits to a definite chirp mass.

Each event is scanned over a few window placements around the coalescence and
the loudest one is kept (mimicking the pipeline's clustering). Both statistics
are scored on the SAME events, so the comparison is fair, then drawn as ROC
curves (signal recovery vs background false-alarm rate).

This is a statistic-quality ROC for picking between the two statistics -- not
the full pipeline's sensitive-volume / FAR curve.

Usage:
    python roc_metrics.py --config <regression_infer_config.yaml> --output roc.png
                          [--n-inj 300 --n-bg 1200 --scan 6 --device cpu]
"""
import argparse
import glob

import h5py
import matplotlib
import numpy as np
import torch
import torch.nn as nn
import torchaudio
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (must follow matplotlib.use)

from ledger.injections import InterferometerResponseSet, waveform_class_factory
from ml4gw.transforms import Whiten
from train.model.regression import LitS4DGaussianNLL
from utils.preprocessing import PsdEstimator

# Read this many seconds of strain before / after the coalescence so the whole
# analysis window (PSD estimate + filter padding + kernel) is available even for
# the earliest placement in the scan.
READ_BEFORE = 35.0
READ_AFTER = 5.0


class WindowScorer:
    """A trained regression model plus the preprocessing it expects.

    Carves analysis windows out of a strain segment and turns them into the two
    candidate detection statistics.
    """

    def __init__(self, config: dict, device: torch.device):
        self.device = device
        self.ifos = config["ifos"]
        self.raw_sample_rate = int(config["raw_sample_rate"])
        self.sample_rate = int(config["sample_rate"])

        kernel_length = config["kernel_length"]
        fduration = config["fduration"]
        psd_length = config["psd_length"]

        # Window geometry, in seconds. One analysis window holds the PSD-estimation
        # stretch, the filter-settling padding, and the model kernel. `right_lead`
        # is how far the window START sits before the window's nominal right edge.
        self.window_length = psd_length + fduration + kernel_length
        self.window_samples = int(self.window_length * self.raw_sample_rate)
        self.right_lead = psd_length + fduration / 2 + kernel_length

        self.model = (
            LitS4DGaussianNLL
            .load_from_checkpoint(config["checkpoint"], strict=False)
            .eval()
            .to(device)
        )
        self.n_vars = self.model.n_vars
        self.softplus = nn.Softplus()
        self.resample = torchaudio.transforms.Resample(
            self.raw_sample_rate, self.sample_rate
        ).to(device)
        self.estimate_psd = PsdEstimator(
            kernel_length + fduration,
            self.sample_rate,
            config["fftlength"],
            average="median",
            fast=True,
        ).to(device)
        self.whiten = Whiten(
            fduration, self.sample_rate, config["highpass"], None
        ).to(device)

    def statistics(self, windows: np.ndarray, batch_size: int = 64):
        """Score raw windows of shape (N, n_ifo, window_samples).

        Returns ``(neg_sigma, abs_mu_over_sigma)``, one value per window.
        """
        neg_sigma, abs_mu_over_sigma = [], []
        for i in range(0, len(windows), batch_size):
            x = torch.from_numpy(windows[i : i + batch_size]).float().to(self.device)
            x = self.resample(x)
            x, psd = self.estimate_psd(x)
            x = self.whiten(x, psd)
            x = self.model._prepare_input(x)
            out = self.model(x)
            mu = out[:, 0]
            sigma = torch.sqrt(self.softplus(out[:, self.n_vars :])[:, 0])
            neg_sigma.append((-sigma).cpu().numpy())
            abs_mu_over_sigma.append((mu.abs() / sigma).cpu().numpy())
        return np.concatenate(neg_sigma), np.concatenate(abs_mu_over_sigma)

    def scan_event(
        self, seg_file, seg_start, coalescence, scan_edges, injection_path=None
    ):
        """Score the loudest window placement around one coalescence time.

        Reads strain around ``coalescence``; if ``injection_path`` is given, adds
        the signals from it that land in that stretch; slides the window across
        ``scan_edges`` (kernel-right-edge offsets relative to the coalescence);
        and returns the max of each statistic over those placements.
        """
        read_start = coalescence - READ_BEFORE
        start_idx = int((read_start - seg_start) * self.raw_sample_rate)
        stop_idx = int((coalescence + READ_AFTER - seg_start) * self.raw_sample_rate)
        with h5py.File(seg_file) as f:
            strain = np.stack(
                [f[ifo][start_idx:stop_idx] for ifo in self.ifos]
            ).astype(np.float32)

        if injection_path is not None:
            response_set = waveform_class_factory(
                self.ifos, InterferometerResponseSet, "ResponseSet"
            )
            injections = response_set.read(
                injection_path,
                start=read_start,
                end=coalescence + READ_AFTER,
                shifts=[0.0] * len(self.ifos),
            )
            strain = injections.inject(strain.copy(), read_start)

        window_starts = [
            int(round(
                (coalescence + edge - self.right_lead - read_start)
                * self.raw_sample_rate
            ))
            for edge in scan_edges
        ]
        windows = np.stack(
            [strain[:, s : s + self.window_samples] for s in window_starts]
        )
        neg_sigma, abs_mu_over_sigma = self.statistics(windows)
        return neg_sigma.max(), abs_mu_over_sigma.max()


def roc_curve(signal_scores, background_scores):
    """ROC curve and AUC for statistics where higher means more signal-like."""
    s = np.sort(signal_scores)
    b = np.sort(background_scores)
    thresholds = np.unique(np.concatenate([s, b]))[::-1]
    tpr = 1 - np.searchsorted(s, thresholds, "left") / len(s)
    fpr = 1 - np.searchsorted(b, thresholds, "left") / len(b)
    fpr = np.concatenate([[0], fpr, [1]])
    tpr = np.concatenate([[0], tpr, [1]])
    return fpr, tpr, float(np.trapz(tpr, fpr))


def load_segments(background_dir, ifo, raw_sample_rate):
    """Return ``[(start, end, file), ...]`` GPS spans for each background file."""
    segments = []
    for fname in sorted(glob.glob(f"{background_dir}/*.hdf5")):
        with h5py.File(fname) as f:
            start = float(f[ifo].attrs["x0"])
            n = len(f[ifo])
        segments.append((start, start + n / raw_sample_rate, fname))
    return segments


def sample_foreground(
    scorer, segments, injection_path, scan_edges, n_inj, snr_min, rng
):
    """Score ``n_inj`` injected signals.

    Returns ``(neg_sigma, abs_mu_over_sigma, snr)`` arrays.
    """
    with h5py.File(injection_path) as f:
        times = f["parameters/injection_time"][:]
        snrs = f["parameters/snr"][:]

    # Keep injections above the SNR floor that sit comfortably inside a segment
    # (margins match the read window: 40 s before, 6 s after the coalescence).
    def contained(t):
        return any(start <= t - 40 and t + 6 <= end for start, end, _ in segments)

    usable = [
        i for i in range(len(times)) if snrs[i] >= snr_min and contained(times[i])
    ]
    chosen = rng.choice(usable, size=min(n_inj, len(usable)), replace=False)

    neg_sigma, abs_mu_over_sigma, snr = [], [], []
    for i in chosen:
        t = float(times[i])
        seg = next(s for s in segments if s[0] <= t - 40 and t + 6 <= s[1])
        u, d = scorer.scan_event(seg[2], seg[0], t, scan_edges, injection_path)
        neg_sigma.append(u)
        abs_mu_over_sigma.append(d)
        snr.append(snrs[i])
    return np.array(neg_sigma), np.array(abs_mu_over_sigma), np.array(snr)


def sample_background(scorer, segments, injection_times, scan_edges, n_bg, rng):
    """Score ``n_bg`` signal-free noise windows.

    Returns ``(neg_sigma, abs_mu_over_sigma)`` arrays.
    """
    neg_sigma, abs_mu_over_sigma = [], []
    attempts = 0
    while len(neg_sigma) < n_bg and attempts < n_bg * 5:
        attempts += 1
        start, end, fname = segments[rng.integers(len(segments))]
        if end - start < 80:
            continue
        t = float(rng.uniform(start + 40, end - 6))
        if np.min(np.abs(injection_times - t)) < 30:  # stay clear of real signals
            continue
        u, d = scorer.scan_event(fname, start, t, scan_edges)
        neg_sigma.append(u)
        abs_mu_over_sigma.append(d)
    return np.array(neg_sigma), np.array(abs_mu_over_sigma)


def plot_roc(foreground, background, fg_snr, output, snr_min):
    """Draw both statistics' ROC curves and print their AUCs."""
    fg_neg_sigma, fg_dev = foreground
    bg_neg_sigma, bg_dev = background

    fig, ax = plt.subplots(figsize=(6.5, 6))
    curves = [
        ((fg_neg_sigma, bg_neg_sigma), "uncertainty  (−σ)", "#1f77b4"),
        ((fg_dev, bg_dev), "standardized dev.  (|μ|/σ)", "#d62728"),
    ]
    for (signal, background_scores), label, color in curves:
        fpr, tpr, auc = roc_curve(signal, background_scores)
        ax.plot(fpr, tpr, color=color, lw=2, label=f"{label}   AUC={auc:.3f}")

    # Also report AUC restricted to confidently detectable injections (SNR >= 15).
    loud = fg_snr >= 15
    if loud.sum() > 5:
        _, _, auc_neg_sigma = roc_curve(fg_neg_sigma[loud], bg_neg_sigma)
        _, _, auc_dev = roc_curve(fg_dev[loud], bg_dev)
        print(
            f"AUC (SNR>=15, n={loud.sum()}): "
            f"-sigma={auc_neg_sigma:.3f}  |mu|/sigma={auc_dev:.3f}"
        )

    ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5, label="random")
    ax.set_xlabel("False positive rate (noise)", fontsize=11)
    ax.set_ylabel("True positive rate (injections recovered)", fontsize=11)
    cut = f", SNR≥{snr_min:.0f}" if snr_min > 0 else ""
    ax.set_title(
        f"Detection-metric ROC  (n_inj={len(fg_neg_sigma)}{cut}, "
        f"n_bg={len(bg_neg_sigma)})",
        fontsize=11,
    )
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output, dpi=140, bbox_inches="tight")
    print(f"saved {output}")
    print(
        "  all-SNR AUC: "
        f"-sigma={roc_curve(fg_neg_sigma, bg_neg_sigma)[2]:.3f}  "
        f"|mu|/sigma={roc_curve(fg_dev, bg_dev)[2]:.3f}"
    )


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--config", required=True, help="regression_infer YAML")
    ap.add_argument("--output", required=True, help="output PNG path")
    ap.add_argument("--n-inj", type=int, default=300, help="injections to score")
    ap.add_argument("--n-bg", type=int, default=1200, help="background windows to score")
    ap.add_argument("--scan", type=int, default=6, help="window placements per event")
    ap.add_argument(
        "--snr-min", type=float, default=0.0,
        help="only score injections with SNR >= this (detectable subset)",
    )
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    config = yaml.safe_load(open(args.config))
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(0)
    torch.set_grad_enabled(False)

    scorer = WindowScorer(config, device)
    segments = load_segments(
        config["background_dir"], scorer.ifos[0], scorer.raw_sample_rate
    )
    # Kernel-right-edge offsets relative to coalescence, e.g. -6 s (pre-merger) .. +1 s.
    scan_edges = np.linspace(-6, 1, args.scan)
    injection_path = config["injection_set_fname"]

    fg_neg_sigma, fg_dev, fg_snr = sample_foreground(
        scorer, segments, injection_path, scan_edges, args.n_inj, args.snr_min, rng
    )
    with h5py.File(injection_path) as f:
        injection_times = f["parameters/injection_time"][:]
    bg_neg_sigma, bg_dev = sample_background(
        scorer, segments, injection_times, scan_edges, args.n_bg, rng
    )

    plot_roc(
        (fg_neg_sigma, fg_dev),
        (bg_neg_sigma, bg_dev),
        fg_snr,
        args.output,
        args.snr_min,
    )


if __name__ == "__main__":
    main()
