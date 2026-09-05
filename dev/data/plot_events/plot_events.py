"""Plot the reference events in the DenoiserEvolutionCallback style.

Same two-panel layout the callback produces, one row per event and
interferometer: whitened strain against time to merger on the left, the
magnitude of its real FFT on the right. There is no prediction here, since
these plots describe the inputs rather than any model's output.

    python plot_events.py
    python plot_events.py --events other.hdf5 --output other.png
"""

import argparse
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
PLOT_ROOT = HERE.parent
DEFAULT_EVENTS = PLOT_ROOT / "merger_4s_2048Hz" / "plot_events.hdf5"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", default=str(DEFAULT_EVENTS))
    parser.add_argument(
        "--output", default=None,
        help="PNG to write; defaults to plot_events.png beside the events",
    )
    parser.add_argument(
        "--show-input", action="store_true", default=True,
        help="overlay the noisy input behind the clean target",
    )
    parser.add_argument(
        "--plot-window", type=float, nargs=2, default=None,
        metavar=("BEGIN", "END"),
        help="restrict the time axis to this window, in seconds relative "
             "to the start of the kernel",
    )
    parser.add_argument(
        "--window-begin", type=float, default=0.0,
        help="time the kernel starts at, for --plot-window",
    )
    parser.add_argument("--dpi", type=int, default=110)
    return parser.parse_args()


def rfft_magnitude(series, sample_rate):
    """Same as DenoiserEvolutionCallback._fft, clamp included.

    The clamp keeps an all-zero target (the background row) on a log axis
    instead of dropping it entirely.
    """
    magnitude = np.abs(np.fft.rfft(series))
    freqs = np.fft.rfftfreq(len(series), 1.0 / sample_rate)
    return freqs[1:], np.maximum(magnitude[1:], 1e-30)


def main():
    args = parse_args()
    if args.output is None:
        args.output = str(Path(args.events).with_suffix(".png"))
    with h5py.File(args.events) as handle:
        noisy = handle["noisy"][:]
        clean = handle["clean"][:]
        names = [name.decode() for name in handle["names"][:]]
        sample_rate = float(handle.attrs["sample_rate"])
        ifos = [ifo.decode() for ifo in handle.attrs["ifos"]]
        glitch_file = handle.attrs.get("glitch_file", b"")
        glitch_offset = float(handle.attrs.get("glitch_offset_s", 0.0))
        merger_index = handle.attrs.get("merger_index")
        long_noisy = handle["long_noisy"][:] if "long_noisy" in handle else None
        long_clean = handle["long_clean"][:] if "long_clean" in handle else None
        long_merger = handle.attrs.get("long_merger_index")

    n_events, n_ifos, length = clean.shape
    low, high = 0, length
    if args.plot_window:
        begin, end = args.plot_window
        low = max(0, int((begin - args.window_begin) * sample_rate))
        high = min(length, int((end - args.window_begin) * sample_rate))
    window = slice(low, high)
    n_rows = n_events * n_ifos
    fig, axes = plt.subplots(
        n_rows, 2, figsize=(14, 4.4 * n_rows), squeeze=False
    )

    for event in range(n_events):
        for ifo in range(n_ifos):
            row = event * n_ifos + ifo
            target = clean[event, ifo]
            # The builder records where the coalescence sits, so the origin
            # is the same for every row including a background row whose
            # target is identically zero. argmax is only a fallback for
            # files written before the index was stored.
            if merger_index is not None:
                merger = int(merger_index)
            elif np.any(target):
                merger = int(np.argmax(np.abs(target)))
            else:
                merger = 0
            times = (np.arange(length) - merger) / sample_rate
            snr = float(np.linalg.norm(clean[event]))

            ax = axes[row][0]
            if args.show_input:
                ax.plot(
                    times[window], noisy[event, ifo][window], lw=0.5,
                    color="0.6", alpha=0.45, label="noisy input",
                )
            ax.plot(
                times[window], target[window], lw=0.9, color="k",
                label="target",
            )
            ax.set_ylabel(f"{names[event]} / {ifos[ifo]}")
            ax.set_title(f"SNR = {snr:.1f}", fontsize=8, loc="right")
            if row == 0:
                ax.legend(fontsize=7, ncol=2, loc="upper left")
            if row == n_rows - 1:
                ax.set_xlabel("time to merger [s]")

            axf = axes[row][1]
            if args.show_input:
                freqs, magnitude = rfft_magnitude(
                    noisy[event, ifo][window], sample_rate
                )
                axf.loglog(
                    freqs, magnitude, lw=0.6, color="0.6", alpha=0.45,
                    label="noisy input",
                )
            freqs, magnitude = rfft_magnitude(target[window], sample_rate)
            axf.loglog(freqs, magnitude, lw=0.9, color="k", label="target")
            axf.set_ylabel("|rfft|")
            if row == 0:
                axf.set_title("abs(rfft) magnitude", fontsize=9)
                axf.legend(fontsize=7, loc="upper right")
            if row == n_rows - 1:
                axf.set_xlabel("frequency [Hz]")

    title = "reference events"
    if glitch_file:
        name = glitch_file.decode() if isinstance(glitch_file, bytes) else (
            glitch_file
        )
        title += f"   (glitch: {name} @ {glitch_offset:.1f} s)"
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.995))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=args.dpi, bbox_inches="tight")
    print(f"wrote {output}")
    plt.close(fig)

    if long_noisy is None:
        return

    # The long segment gets its own figure: its time axis spans tens of
    # seconds, so sharing one with the four-second events would compress
    # them to nothing.
    n_ifos = long_clean.shape[0]
    fig, axes = plt.subplots(n_ifos, 2, figsize=(14, 4.4 * n_ifos),
                             squeeze=False)
    merger = int(long_merger) if long_merger is not None else 0
    length = long_clean.shape[-1]
    times = (np.arange(length) - merger) / sample_rate
    for ifo in range(n_ifos):
        ax = axes[ifo][0]
        if args.show_input:
            ax.plot(times, long_noisy[ifo], lw=0.4, color="0.6", alpha=0.45,
                    label="noisy input")
        ax.plot(times, long_clean[ifo], lw=0.7, color="k", label="target")
        ax.set_ylabel(f"long / {ifos[ifo]}")
        ax.margins(x=0)
        if ifo == 0:
            ax.legend(fontsize=7, ncol=2, loc="upper left")
        if ifo == n_ifos - 1:
            ax.set_xlabel("time to merger [s]")

        axf = axes[ifo][1]
        if args.show_input:
            freqs, magnitude = rfft_magnitude(long_noisy[ifo], sample_rate)
            axf.loglog(freqs, magnitude, lw=0.5, color="0.6", alpha=0.45,
                       label="noisy input")
        freqs, magnitude = rfft_magnitude(long_clean[ifo], sample_rate)
        axf.loglog(freqs, magnitude, lw=0.7, color="k", label="target")
        axf.set_ylabel("|rfft|")
        if ifo == 0:
            axf.set_title("abs(rfft) magnitude", fontsize=9)
            axf.legend(fontsize=7, loc="upper right")
        if ifo == n_ifos - 1:
            axf.set_xlabel("frequency [Hz]")

    snr = float(np.linalg.norm(long_clean))
    fig.suptitle(
        f"long segment   {length / sample_rate:.0f} s   SNR {snr:.1f}   "
        f"({times[0]:+.0f} to {times[-1]:+.0f} s)", fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    long_out = output.with_name(output.stem + "_long" + output.suffix)
    fig.savefig(long_out, dpi=args.dpi, bbox_inches="tight")
    print(f"wrote {long_out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
