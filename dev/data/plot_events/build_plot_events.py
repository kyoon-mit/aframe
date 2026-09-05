"""Build a fixed set of reference events for denoiser diagnostic plots.

Writes one HDF5 file holding a small, curated batch: injections at several
signal-to-noise ratios, one injection sitting on a glitch, and one
background-only stretch. Because the file is fixed on disk, every training
run and every loss variant plots the same events, so their evolution plots
are directly comparable.

The background is timeslid before use: each interferometer is offset by a
different amount, so any real coincident signal in the data is destroyed and
what remains is noise plus whatever incoherent glitches are present.
Waveforms are then injected from the validation set at exact target SNRs.

Everything is argparsed, so the same script produces files for other
segments, durations, sample rates, and pre-merger windows.

Examples
--------
Default five-event file at 2048 Hz, 4 s kernels::

    python build_plot_events.py

A pre-merger set, 8 s kernels ending 1 s before coalescence, 512 Hz::

    python build_plot_events.py --kernel-length 8 --sample-rate 512 \\
        --right-pad -1 --output premerger_events.hdf5
"""

import argparse
import glob
import logging
from pathlib import Path

import h5py
import numpy as np
import torch

LOGGER = logging.getLogger("build_plot_events")

HERE = Path(__file__).resolve().parent
PLOT_ROOT = HERE.parent
DEFAULT_BACKGROUND = "/home/kyoon/SSM-BNS/DATA/O3a_H1_L1_2048Hz/background"
DEFAULT_WAVEFORMS = (
    "/home/kyoon/SSM-BNS/DATA/aframe/val/end_o3_ratesandpops_bns_snr4.hdf5"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    data = parser.add_argument_group("data")
    data.add_argument("--background-dir", default=DEFAULT_BACKGROUND)
    data.add_argument("--waveform-file", default=DEFAULT_WAVEFORMS)
    data.add_argument("--ifos", nargs="+", default=["H1", "L1"])
    data.add_argument(
        "--waveform-group", default="waveforms",
        help="group inside the waveform file holding the per-ifo strain",
    )
    data.add_argument(
        "--output", default=None,
        help="HDF5 file to write; defaults to a directory named for the "
             "segment geometry, e.g. merger_4s_2048Hz/plot_events.hdf5",
    )

    geometry = parser.add_argument_group("segment geometry")
    geometry.add_argument(
        "--sample-rate", type=float, default=2048.0,
        help="output rate; the data is decimated to it with an anti-alias "
             "filter if the file rate is higher",
    )
    geometry.add_argument(
        "--native-sample-rate", type=float, default=2048.0,
        help="rate the background files are stored at",
    )
    geometry.add_argument("--kernel-length", type=float, default=4.0)
    geometry.add_argument(
        "--left-pad", type=float, default=3.5,
        help="minimum seconds between the merger and the kernel start; this "
             "is what sets the placement, matching build_val_batches",
    )
    geometry.add_argument(
        "--right-pad", type=float, default=0.0,
        help="seconds between coalescence and the end of the kernel; "
             "negative values put the merger outside the window, which is "
             "how a pre-merger segment is built",
    )
    geometry.add_argument(
        "--waveform-right-pad", type=float, default=2.0,
        help="seconds between the coalescence and the end of the stored "
             "waveform; a property of the waveform file, not of the kernel",
    )
    geometry.add_argument("--psd-length", type=float, default=20.0)
    geometry.add_argument("--fduration", type=float, default=1.0)
    geometry.add_argument("--fftlength", type=float, default=2.0)
    geometry.add_argument("--highpass", type=float, default=20.0)

    events = parser.add_argument_group("events")
    events.add_argument(
        "--snrs", type=float, nargs="+", default=[50.0, 20.0, 4.0],
        help="SNRs to build injections at, on quiet background",
    )
    events.add_argument(
        "--glitch-snr", type=float, default=4.0,
        help="SNR of the injection placed on the glitch",
    )
    events.add_argument(
        "--no-glitch", action="store_true",
        help="skip the glitch event",
    )
    events.add_argument(
        "--no-background", action="store_true",
        help="skip the background-only event",
    )
    events.add_argument(
        "--waveform-index", type=int, default=0,
        help="which validation waveform to inject; the same one is used at "
             "every SNR so the cases differ only in amplitude",
    )

    search = parser.add_argument_group("glitch search")
    search.add_argument(
        "--max-files", type=int, default=6,
        help="background files to scan for the glitch",
    )
    search.add_argument(
        "--search-stride", type=float, default=2.0,
        help="seconds between candidate kernels",
    )
    search.add_argument(
        "--timeslide", type=float, default=8.0,
        help="seconds to slide each interferometer past the previous one, "
             "destroying any coincident astrophysical signal",
    )
    long = parser.add_argument_group("long segment")
    long.add_argument(
        "--long-seconds", type=float, default=32.0,
        help="length of the extra long segment; 0 disables it",
    )
    long.add_argument(
        "--long-snr", type=float, default=8.0,
        help="SNR of the injection in the long segment",
    )
    long.add_argument(
        "--long-right-pad", type=float, default=2.0,
        help="seconds between the coalescence and the end of the long "
             "segment, so a 32 s segment runs -30 s to +2 s",
    )

    search.add_argument("--seed", type=int, default=0)
    search.add_argument("--device", default="cuda")
    return parser.parse_args()


def load_background(path, ifos, native_rate, target_rate, timeslide):
    """Read one background file, timeslide the interferometers, decimate.

    The slide is what makes this safe to inject into: a real signal appears
    in both detectors within a light-travel time of each other, so offsetting
    one by seconds guarantees that no coincident astrophysical signal
    survives. Incoherent glitches are unaffected, which is the point.
    """
    with h5py.File(path) as handle:
        channels = [torch.tensor(handle[ifo][:], dtype=torch.float64)
                    for ifo in ifos]

    shift = int(timeslide * native_rate)
    length = min(channel.shape[-1] for channel in channels) - shift * len(ifos)
    slid = [
        channel[index * shift : index * shift + length]
        for index, channel in enumerate(channels)
    ]
    strain = torch.stack(slid)

    if target_rate < native_rate:
        ratio = native_rate / target_rate
        if abs(ratio - round(ratio)) > 1e-9:
            raise ValueError(
                f"sample rate {target_rate} does not divide {native_rate}"
            )
        from scipy.signal import decimate

        strain = torch.tensor(
            decimate(strain.numpy(), int(round(ratio)), ftype="fir", axis=-1),
            dtype=torch.float64,
        )
    elif target_rate > native_rate:
        raise ValueError(
            f"cannot upsample from {native_rate} to {target_rate}"
        )
    return strain


def scan_for_glitch(strain, whiten, spectral, kernel, psd_size, stride,
                    device, pad):
    """Rank kernels by peak whitened amplitude.

    Whitened background has unit variance by construction, so a glitch is a
    short excursion far above it. The peak separates a glitch from a merely
    loud stretch better than the mean power does.
    """
    # Whiten crops fduration/2 from each end, so read kernel + pad samples
    # to get exactly kernel back out.
    results = []
    start = psd_size
    while start + kernel + pad <= strain.shape[-1]:
        background = strain[:, start - psd_size : start].to(device)
        segment = strain[:, start : start + kernel + pad].to(device)
        psd = spectral(background)
        whitened = whiten(segment.unsqueeze(0), psd).squeeze(0)
        results.append(
            (
                float(whitened.abs().max()),
                float(whitened.pow(2).mean()),
                start,
                whitened.cpu(),
                psd.cpu(),
            )
        )
        start += stride
    return results


def build_long_event(args, rate, spectral, whiten, device):
    """One long segment, so the denoiser can be seen over a whole inspiral.

    The four-second events show only the last moments before coalescence.
    An S4D is a sequence-to-sequence recurrence and preserves length, so it
    accepts any window; a segment covering the full stored waveform shows
    whether the model tracks the chirp far from the merger as well as near
    it. Returns (noisy, clean, merger_index).
    """
    kernel = int(args.long_seconds * rate)
    pad = int(args.fduration * rate)
    psd_size = int(args.psd_length * rate)

    files = sorted(glob.glob(f"{args.background_dir}/*.hdf5"))
    strain = load_background(
        files[0], args.ifos, args.native_sample_rate, rate, args.timeslide
    )
    need = psd_size + kernel + pad
    if strain.shape[-1] < need:
        raise SystemExit(
            f"background too short for a {args.long_seconds:g} s segment"
        )
    background = strain[:, :psd_size].to(device)
    segment = strain[:, psd_size : psd_size + kernel + pad].to(device)
    psd = spectral(background)
    whitened = whiten(segment.unsqueeze(0), psd).squeeze(0).cpu()

    with h5py.File(args.waveform_file) as handle:
        group = handle[args.waveform_group]
        keys = [ifo.lower() for ifo in args.ifos if ifo.lower() in group]
        if len(keys) != len(args.ifos):
            keys = list(group)[: len(args.ifos)]
        waveform = np.stack([group[key][args.waveform_index] for key in keys])
    waveform = torch.tensor(waveform, dtype=torch.float64)
    if args.native_sample_rate > rate:
        from scipy.signal import decimate

        waveform = torch.tensor(
            decimate(
                waveform.numpy(),
                int(round(args.native_sample_rate / rate)),
                ftype="fir",
                axis=-1,
            ),
            dtype=torch.float64,
        )

    # place the coalescence long_right_pad seconds from the segment end
    signal_index = waveform.shape[-1] - int(args.waveform_right_pad * rate)
    stop = signal_index + int(args.long_right_pad * rate) + pad // 2
    start = stop - (kernel + pad)
    left = -min(start, 0)
    right = max(stop - waveform.shape[-1], 0)
    if left or right:
        waveform = torch.nn.functional.pad(waveform, [left, right])
        start += left
        stop += left
    padded = waveform[:, start:stop]

    white = whiten(padded.unsqueeze(0).to(device), psd.to(device))
    white = white.squeeze(0).cpu()
    clean = white / white.reshape(-1).norm().clamp_min(1e-12) * args.long_snr
    merger_index = signal_index + left - start - pad // 2

    LOGGER.info(
        "  %-12s snr %7.2f   %.0f s segment, merger at %.2f s",
        "long", float(clean.reshape(-1).norm()), kernel / rate,
        merger_index / rate,
    )
    return (whitened.double() + clean).float(), clean.float(), merger_index


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    torch.manual_seed(args.seed)

    if args.output is None:
        # name the directory after the geometry, so sets built at other
        # lengths, rates or pre-merger offsets land beside each other
        kind = "merger" if args.right_pad >= 0 else "premerger"
        geometry = f"{kind}_{args.kernel_length:g}s_{args.sample_rate:g}Hz"
        if args.right_pad:
            geometry += f"_pad{args.right_pad:g}s"
        args.output = str(PLOT_ROOT / geometry / "plot_events.hdf5")

    from ml4gw.transforms import SpectralDensity, Whiten

    device = args.device
    rate = args.sample_rate
    kernel = int(args.kernel_length * rate)
    psd_size = int(args.psd_length * rate)
    stride = int(args.search_stride * rate)
    pad = int(args.fduration * rate)

    spectral = SpectralDensity(
        sample_rate=rate, fftlength=args.fftlength, average="median"
    ).to(device)
    whiten = Whiten(
        fduration=args.fduration, sample_rate=rate, highpass=args.highpass
    ).to(device)

    files = sorted(glob.glob(f"{args.background_dir}/*.hdf5"))[: args.max_files]
    if not files:
        raise SystemExit(f"no background files in {args.background_dir}")
    LOGGER.info("scanning %d background files for a glitch", len(files))

    loudest = None
    quietest = None
    for path in files:
        strain = load_background(
            path, args.ifos, args.native_sample_rate, rate, args.timeslide
        )
        LOGGER.info(
            "  %s: %.1f s after a %.1f s timeslide",
            Path(path).name, strain.shape[-1] / rate, args.timeslide,
        )
        for peak, power, start, whitened, psd in scan_for_glitch(
            strain, whiten, spectral, kernel, psd_size, stride, device, pad
        ):
            if loudest is None or peak > loudest[0]:
                loudest = (peak, power, start, whitened, path, psd)
            if peak < 5.0 and (quietest is None or power < quietest[1]):
                quietest = (peak, power, start, whitened, path, psd)

    if loudest is None or quietest is None:
        raise SystemExit("scan found no usable kernels")

    LOGGER.info(
        "glitch:  %s at %.1f s  peak %.1f  power %.2f",
        Path(loudest[4]).name, loudest[2] / rate, loudest[0], loudest[1],
    )
    LOGGER.info(
        "quiet:   %s at %.1f s  peak %.1f  power %.2f",
        Path(quietest[4]).name, quietest[2] / rate, quietest[0], quietest[1],
    )

    # Injection. The whitened waveform's norm is its optimal matched-filter
    # SNR, so scaling to a target norm sets the SNR exactly.
    with h5py.File(args.waveform_file) as handle:
        # aframe stores polarisations under waveforms/<ifo>, lowercased
        group = handle[args.waveform_group]
        keys = [ifo.lower() for ifo in args.ifos if ifo.lower() in group]
        if len(keys) != len(args.ifos):
            keys = list(group)[: len(args.ifos)]
        waveform = np.stack([group[key][args.waveform_index] for key in keys])
        snr_recorded = None
        if "parameters/snr" in handle:
            snr_recorded = float(handle["parameters/snr"][args.waveform_index])
    waveform = torch.tensor(waveform, dtype=torch.float64)
    LOGGER.info(
        "waveform %d from %s: shape %s%s",
        args.waveform_index, "/".join(keys), tuple(waveform.shape),
        f", recorded snr {snr_recorded:.1f}" if snr_recorded else "",
    )

    if args.native_sample_rate > rate:
        from scipy.signal import decimate

        waveform = torch.tensor(
            decimate(
                waveform.numpy(),
                int(round(args.native_sample_rate / rate)),
                ftype="fir",
                axis=-1,
            ),
            dtype=torch.float64,
        )

    # Slice exactly as build_val_batches does, so these events sit in the
    # kernel the way validation events do. The window start is set by
    # left_pad, not right_pad: the merger lands left_pad_size samples from
    # the kernel start, which leaves the post-merger tail inside the window.
    # Driving it from right_pad instead truncates the merger and ringdown.
    signal_index = waveform.shape[-1] - int(args.waveform_right_pad * rate)
    kernel_size = kernel + pad
    left_pad_size = int(args.left_pad * rate) + pad // 2
    start = signal_index - left_pad_size
    stop = start + kernel_size

    left = -min(start, 0)
    right = max(stop - waveform.shape[-1], 0)
    if left or right:
        waveform = torch.nn.functional.pad(waveform, [left, right])
        start += left
        stop += left
    padded = waveform[:, start:stop]
    # Where the coalescence sits in the whitened kernel, known here by
    # construction. Recording it means the plotter never has to re-derive
    # the origin from an argmax, which drifts between rows and is undefined
    # on a background row whose target is identically zero.
    merger_index = signal_index + left - start - pad // 2
    LOGGER.info(
        "merger %.3f s into the %.1f s kernel (right_pad %.2f s)",
        merger_index / rate, kernel / rate, args.right_pad,
    )

    def whitened_unit(psd):
        """Whiten the padded waveform with this PSD and normalise it.

        The waveform has to pass through the same whitening as the data it
        is injected into, or the norm is not the SNR the detector would see.
        """
        white = whiten(
            padded.unsqueeze(0).to(device), psd.to(device)
        ).squeeze(0).cpu()
        return white / white.reshape(-1).norm().clamp_min(1e-12)

    names, noisy_rows, clean_rows = [], [], []

    def add(name, background_whitened, snr, psd):
        clean = whitened_unit(psd) * snr
        noisy_rows.append((background_whitened.double() + clean).float())
        clean_rows.append(clean.float())
        names.append(name)
        LOGGER.info(
            "  %-12s snr %7.2f   background peak %6.2f",
            name, float(clean.reshape(-1).norm()),
            float(background_whitened.abs().max()),
        )

    LOGGER.info("building events")
    for snr in args.snrs:
        add(f"snr{snr:g}", quietest[3], snr, quietest[5])
    if not args.no_glitch:
        add("glitch", loudest[3], args.glitch_snr, loudest[5])
    if not args.no_background:
        noisy_rows.append(loudest[3].float())
        clean_rows.append(torch.zeros_like(loudest[3]).float())
        names.append("background")
        LOGGER.info(
            "  %-12s snr    0.00   background peak %6.2f",
            "background", float(loudest[3].abs().max()),
        )

    long_event = None
    if args.long_seconds:
        long_event = build_long_event(args, rate, spectral, whiten, device)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output, "w") as handle:
        handle["noisy"] = torch.stack(noisy_rows).numpy()
        handle["clean"] = torch.stack(clean_rows).numpy()
        handle["names"] = np.array(names, dtype="S32")
        handle.attrs["sample_rate"] = rate
        handle.attrs["kernel_length"] = args.kernel_length
        handle.attrs["right_pad"] = args.right_pad
        handle.attrs["ifos"] = np.array(args.ifos, dtype="S8")
        handle.attrs["timeslide"] = args.timeslide
        handle.attrs["glitch_file"] = str(Path(loudest[4]).name)
        handle.attrs["glitch_offset_s"] = loudest[2] / rate
        handle.attrs["glitch_peak"] = loudest[0]
        handle.attrs["quiet_file"] = str(Path(quietest[4]).name)
        handle.attrs["quiet_offset_s"] = quietest[2] / rate
        handle.attrs["waveform_file"] = args.waveform_file
        handle.attrs["waveform_index"] = args.waveform_index
        handle.attrs["merger_index"] = int(merger_index)
        if long_event is not None:
            noisy_long, clean_long, merger_long = long_event
            handle["long_noisy"] = noisy_long.numpy().astype(np.float32)
            handle["long_clean"] = clean_long.numpy().astype(np.float32)
            handle.attrs["long_merger_index"] = int(merger_long)
            handle.attrs["long_seconds"] = args.long_seconds
            handle.attrs["long_snr"] = args.long_snr
            handle.attrs["long_right_pad"] = args.long_right_pad
    LOGGER.info("wrote %s  (%d events)", output, len(names))


if __name__ == "__main__":
    main()
