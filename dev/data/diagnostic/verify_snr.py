"""Recompute SNRs from stored waveforms and compare to the SNRs on file.

Works on either a WaveformSet (val: has waveforms/h1, waveforms/l1) or an
InterferometerResponseSet with background/injected groups (diagnostic:
signal = injected - background).

Writes <dest>/snrs.out with one line per event (stored, recomputed,
percent diff), <dest>/snr_summary.txt with the fraction of events within
each percent-difference threshold, and <dest>/snr_hist3d.png.

Options: --input hdf5, --background strain file/dir (PSD), --dest output
dir, --highpass/--lowpass Hz, --batch_size events per SNR batch,
--config plot json, --xy-bin-size histogram bin width, --recompute
true|false (false: reuse existing snrs.out, only remake the plot).

Example:
    python verify_snr.py \\
        --input /path/to/file.hdf5 \\
        --background /path/to/background_dir \\
        --dest /path/to/output_dir \\
        --highpass 20 --lowpass 1024 [--recompute false]
"""

import argparse
import json
import os
from pathlib import Path

import h5py
import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

from data.waveforms.utils import load_psds
from ml4gw.gw import compute_network_snr

DEFAULT_CONFIG = (
    Path(__file__).resolve().parent.parent / "configs" / "plot_configs.json"
)

THRESHOLDS = [
    0.001,
    0.002,
    0.005,
    0.01,
    0.02,
    0.05,
    0.1,
    0.2,
    0.5,
    1,
    2,
    5,
    10,
    20,
    50,
    100,
]


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--background", required=True)
    parser.add_argument("--dest", required=True)
    parser.add_argument("--highpass", type=float, default=20.0)
    parser.add_argument("--lowpass", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=2000)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--xy-bin-size", type=float, default=0.1)
    parser.add_argument(
        "--recompute",
        type=parse_bool,
        default=True,
        help="false: reuse the existing <dest>/snrs.out and "
        "snr_summary.txt and only remake the plot",
    )
    return parser.parse_args()


def parse_bool(value):
    """Interpret a true/false command line string as a bool."""
    if value.lower() in ("true", "1", "yes"):
        return True
    if value.lower() in ("false", "0", "no"):
        return False
    raise argparse.ArgumentTypeError(f"expected true or false, got {value}")


def plot_snr_hist3d(
    stored_snr, recomputed_snr, snr_config, dest_dir, xy_bin_size
):
    """3D histogram of stored vs recomputed SNR.

    Axis ranges come from the plot config's snr entry; histogram bins are
    xy_bin_size wide, while the config's bin_size sets the minor-tick
    spacing on both SNR axes.
    """
    minimum, maximum = snr_config["min"], snr_config["max"]
    edges = np.arange(minimum, maximum + xy_bin_size, xy_bin_size)
    counts, _, _ = np.histogram2d(
        recomputed_snr, stored_snr, bins=(edges, edges)
    )

    x_index, y_index = np.nonzero(counts)
    figure = plt.figure(figsize=(8, 7))
    axis = figure.add_subplot(projection="3d")
    axis.bar3d(
        edges[x_index],
        edges[y_index],
        np.zeros(len(x_index)),
        xy_bin_size,
        xy_bin_size,
        counts[x_index, y_index],
        color=snr_config.get("color", "darkblue"),
    )
    axis.set_xlabel("Recomputed SNR", labelpad=2)
    axis.set_ylabel("Stored SNR", labelpad=2)
    axis.set_zlabel("Count", labelpad=10)
    axis.set_xlim(minimum, maximum)
    axis.set_ylim(minimum, maximum)
    axis.set_zlim(0, counts.max())
    axis.xaxis.set_major_locator(MultipleLocator(10))
    axis.yaxis.set_major_locator(MultipleLocator(10))
    axis.xaxis.set_minor_locator(MultipleLocator(snr_config["bin_size"]))
    axis.yaxis.set_minor_locator(MultipleLocator(snr_config["bin_size"]))
    axis.tick_params(axis="both", pad=0)
    axis.tick_params(axis="z", pad=5)
    axis.text2D(
        0.02,
        0.98,
        f"Total Count: {len(stored_snr):,}",
        transform=axis.transAxes,
        ha="left",
        va="top",
        bbox={"boxstyle": "square", "facecolor": "white", "edgecolor": "gray"},
    )

    output_path = os.path.join(
        dest_dir, f"snr_hist3d_xy_bin_size_{xy_bin_size!s}.png"
    )
    figure.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.4)
    plt.close(figure)
    print("wrote", output_path, flush=True)


def load_responses(hdf5_file, ifos, start, stop):
    """Return (batch, num_ifos, num_samples) responses in [start, stop)."""
    if "waveforms" in hdf5_file:
        group = hdf5_file["waveforms"]
        return np.stack(
            [group[ifo.lower()][start:stop] for ifo in ifos], axis=1
        )
    injected = np.stack(
        [hdf5_file["injected"][ifo.lower()][start:stop] for ifo in ifos],
        axis=1,
    )
    background = np.stack(
        [hdf5_file["background"][ifo.lower()][start:stop] for ifo in ifos],
        axis=1,
    )
    return injected - background


def main():
    """Recompute SNRs, write snrs.out and snr_summary.txt, and plot.

    With --recompute false, skip the SNR recomputation and read the
    stored/recomputed values back from an existing snrs.out (both output
    text files must already exist in --dest), then only remake the plot.
    """
    arguments = parse_arguments()
    os.makedirs(arguments.dest, exist_ok=True)

    snrs_path = os.path.join(arguments.dest, "snrs.out")
    summary_path = os.path.join(arguments.dest, "snr_summary.txt")

    if not arguments.recompute:
        for path in (snrs_path, summary_path):
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"--recompute false requires {path} to exist"
                )
        data = np.loadtxt(snrs_path, delimiter=",", skiprows=1)
        stored_snr, recomputed_snr = data[:, 1], data[:, 2]
        with open(arguments.config) as config_file:
            snr_config = json.load(config_file)["parameters"]["snr"]
        plot_snr_hist3d(
            stored_snr,
            recomputed_snr,
            snr_config,
            arguments.dest,
            arguments.xy_bin_size,
        )
        return

    with h5py.File(arguments.input, "r") as hdf5_file:
        sample_rate = float(hdf5_file.attrs["sample_rate"])
        duration = float(hdf5_file.attrs["duration"])
        ifos = [str(ifo) for ifo in hdf5_file.attrs["ifos"]]
        num_events = int(hdf5_file.attrs["length"])
        stored_snr = hdf5_file["parameters/snr"][:]

    psd = load_psds(
        arguments.background, ifos, df=1 / duration, sample_rate=sample_rate
    )

    recomputed_snr = np.empty(num_events)
    with h5py.File(arguments.input, "r") as hdf5_file:
        for start in range(0, num_events, arguments.batch_size):
            stop = min(start + arguments.batch_size, num_events)
            responses = load_responses(hdf5_file, ifos, start, stop)
            recomputed_snr[start:stop] = compute_network_snr(
                torch.tensor(responses),
                psd,
                sample_rate,
                highpass=arguments.highpass,
                lowpass=arguments.lowpass,
            ).numpy()

    percent_diff = np.abs(recomputed_snr - stored_snr) / stored_snr * 100

    with open(snrs_path, "w") as snrs_file:
        snrs_file.write("index,stored_snr,recomputed_snr,percent_diff\n")
        for i, (stored, recomputed, diff) in enumerate(
            zip(stored_snr, recomputed_snr, percent_diff, strict=True)
        ):
            snrs_file.write(f"{i},{stored:.6f},{recomputed:.6f},{diff:.6f}\n")
    print("wrote", snrs_path, flush=True)

    num_events = len(stored_snr)
    with open(summary_path, "w") as summary_file:
        summary_file.write(f"input: {arguments.input}\n")
        summary_file.write(f"num_events: {num_events}\n")
        summary_file.write(
            f"max percent diff: {percent_diff.max():.6f}\n"
            f"mean percent diff: {percent_diff.mean():.6f}\n\n"
        )
        summary_file.write("threshold(%)  count  fraction\n")
        for threshold in THRESHOLDS:
            count = int((percent_diff <= threshold).sum())
            fraction = count / num_events
            summary_file.write(
                f"{threshold:<12g}  {count:<6d}  {fraction:.6f}\n"
            )
    print("wrote", summary_path, flush=True)

    with open(arguments.config) as config_file:
        snr_config = json.load(config_file)["parameters"]["snr"]
    plot_snr_hist3d(
        stored_snr,
        recomputed_snr,
        snr_config,
        arguments.dest,
        arguments.xy_bin_size,
    )


if __name__ == "__main__":
    main()
