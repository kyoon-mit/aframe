"""Compare PSDs of original 4096 Hz and downsampled 2048 Hz background files.

For each background file present in both directories, computes a median Welch
PSD per detector at each sample rate and plots them overlaid (H1 and L1 side
by side) with a ratio panel below. If the downsampling is correct, the curves
should agree to within a few percent up to ~80% of the 2048 Hz Nyquist
(~820 Hz), with departures only near 1024 Hz where the anti-aliasing filter
rolls off.

Example:
    python compare_psd_downsample.py \\
        --dir-orig /path/to/O3a_H1_L1_4096Hz \\
        --dir-down /path/to/O3a_H1_L1_2048Hz \\
        --outdir   /path/to/plots
"""

import argparse
import glob
import os

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch

IFOS = ["H1", "L1"]


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dir-orig", required=True)
    parser.add_argument("--dir-down", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument(
        "--fftlength",
        type=float,
        default=2.0,
        help="segment length in seconds for the Welch PSD",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=3,
        help="number of common files to compare (0 = all)",
    )
    return parser.parse_args()


def median_psd(x, fs, fftlength):
    nperseg = int(fftlength * fs)
    freqs, psd = welch(
        x, fs=fs, nperseg=nperseg, noverlap=nperseg // 2, average="median"
    )
    return freqs, psd


def load(fname, ifo):
    with h5py.File(fname, "r") as f:
        dx = f[ifo].attrs["dx"]
        return f[ifo][:], 1.0 / dx


def compare_file(base, dir_orig, dir_down, fftlength, outdir):
    # one column: H1 on top, L1 below; each panel overlays both sample rates
    fig, axes = plt.subplots(len(IFOS), 1, figsize=(9, 8), sharex=True)
    for row, ifo in enumerate(IFOS):
        x_o, fs_o = load(os.path.join(dir_orig, base), ifo)
        x_d, fs_d = load(os.path.join(dir_down, base), ifo)

        f_o, p_o = median_psd(x_o, fs_o, fftlength)
        f_d, p_d = median_psd(x_d, fs_d, fftlength)

        ax = axes[row]
        ax.loglog(f_o, p_o, label=f"{fs_o:.0f} Hz (original)", lw=1)
        ax.loglog(
            f_d, p_d, label=f"{fs_d:.0f} Hz (downsampled)", lw=1, ls="--"
        )
        ax.set_title(f"{ifo}", fontsize=10)
        ax.set_ylabel("PSD [1/Hz]")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, which="both")

        # band-limited agreement stat (printed, not plotted)
        p_o_interp = np.interp(f_d, f_o, p_o)
        ratio = p_d / p_o_interp
        band = (f_d > 20) & (f_d < 0.8 * fs_d / 2)
        print(
            f"{base} {ifo}: median ratio in [20, {0.8 * fs_d / 2:.0f}] Hz "
            f"= {np.median(ratio[band]):.4f} "
            f"(max dev {np.abs(ratio[band] - 1).max():.3f})"
        )

    axes[-1].set_xlabel("frequency [Hz]")
    fig.suptitle(base, fontsize=10)
    fig.tight_layout()
    out = os.path.join(outdir, f"psd_compare_{base.replace('.hdf5', '')}.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  -> {out}")


def main():
    args = parse_arguments()
    os.makedirs(args.outdir, exist_ok=True)

    orig = {
        os.path.basename(p)
        for p in glob.glob(f"{args.dir_orig}/background-*.hdf5")
    }
    down = {
        os.path.basename(p)
        for p in glob.glob(f"{args.dir_down}/background-*.hdf5")
    }
    common = sorted(orig & down)
    if not common:
        raise SystemExit("no common background files between the two dirs")
    if args.max_files:
        common = common[: args.max_files]

    for base in common:
        compare_file(
            base, args.dir_orig, args.dir_down, args.fftlength, args.outdir
        )


if __name__ == "__main__":
    main()
