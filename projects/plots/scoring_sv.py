"""Sensitive-volume reference for alternative integration ("scoring") methods.

For a given aframe inference run this script takes the raw network-output
``timeseries.hdf5`` and, for each proposed integration method, regenerates a
real ``background.hdf5`` / ``foreground.hdf5`` pair and runs the production
sensitive-volume plotter on it.  The result is a folder ``plots/sv/`` holding
one ``<method>.png`` per method, so the alternatives explored in
``scratch/scoring_experiments.ipynb`` can be compared in *absolute* sensitive
volume (Mpc^3 vs FAR) rather than the notebook's relative efficiency proxy.

Only the *integrate* step changes between methods; the offset, clustering and
injection-recovery are the exact repo semantics (a faithful copy of
``projects/infer/infer/postprocess.py`` and
``ledger.events.RecoveredInjectionSet.recover``).

Run inside the plots project environment, e.g.::

    uv run --no-sync --directory projects/plots python scoring_sv.py \
        --run-dir /fast/barmstrong/aframe_results/runs/RUN/results_aframe
"""

import argparse
import hashlib
import logging
import shutil
from pathlib import Path

import h5py
import numpy as np
from scipy.ndimage import gaussian_filter1d, maximum_filter1d, median_filter
from scipy.signal import lfilter
from tqdm import tqdm

from ledger.events import EventSet
from plots.legacy import main as legacy_main
from plots.legacy.main import main as calc_sensitive_volume
from priors.priors import end_o3_ratesandpops_bns

# ---------------------------------------------------------------------------
# Cache the GWTC-3 pipeline reference curves.
#
# ``calc_sensitive_volume`` re-runs the GstLAL / PyCBC / cWB / MBTA reference
# SV (``plots.legacy.main.gwtc3_pipeline_sv``) on every call, i.e. once per
# scoring method.  That result depends only on ``mass_combos`` and the FAR
# thresholds, both of which are identical across methods (the background
# livetime ``Tb`` does not change with the integration method).  So we compute
# it once and reuse it, by wrapping the function the production plotter calls.
# ---------------------------------------------------------------------------
_gwtc3_pipeline_sv = legacy_main.gwtc3_pipeline_sv
_GWTC3_CACHE = {}


def _cached_gwtc3_pipeline_sv(
    mass_combos,
    injection_file,
    detection_criterion,
    detection_thresholds,
    output_dir,
    **kwargs,
):
    thresholds = np.asarray(detection_thresholds)
    key = (
        detection_criterion,
        tuple(tuple(c) for c in mass_combos),
        thresholds.shape,
        hashlib.sha1(thresholds.tobytes()).hexdigest(),
    )
    cached = _GWTC3_CACHE.get(key)
    if cached is None:
        sv, err = _gwtc3_pipeline_sv(
            mass_combos=mass_combos,
            injection_file=injection_file,
            detection_criterion=detection_criterion,
            detection_thresholds=detection_thresholds,
            output_dir=output_dir,
            **kwargs,
        )
        _GWTC3_CACHE[key] = (sv, err, output_dir / "gwtc-3_pipeline_sv.h5")
        return sv, err

    sv, err, src_h5 = cached
    logging.info("Reusing cached GWTC-3 pipeline SV")
    dst_h5 = output_dir / "gwtc-3_pipeline_sv.h5"
    if src_h5.exists() and src_h5.resolve() != dst_h5.resolve():
        output_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_h5, dst_h5)
    return sv, err


legacy_main.gwtc3_pipeline_sv = _cached_gwtc3_pipeline_sv

# ---------------------------------------------------------------------------
# Run parameters (defaults read from the infer condor config for this run).
# ---------------------------------------------------------------------------
RATE = 16.0  # inference_sampling_rate [Hz]
PSD_LENGTH = 64.0  # seconds sliced off the front of every segment
FDURATION = 2.0  # whitening filter length [s]
CLUSTER_WINDOW = 8.0  # clustering window [s]
ZEROLAG_SHIFT = (0, 0)  # un-shifted analysis, excluded from the background

# (label, kind, integration_window_seconds) -- mirrors the notebook.
DEFAULT_METHODS = [
    ("mean_1.0s_default", "mean", 1.0),
    ("none_raw", "none", 0.0),
    ("mean_0.25s", "mean", 0.25),
    ("mean_0.5s", "mean", 0.5),
    ("mean_2.0s", "mean", 2.0),
    ("mean_2.5s", "mean", 2.5),
    ("mean_3.0s", "mean", 3.0),
    ("mean_3.5s", "mean", 3.5),
    ("mean_4.0s", "mean", 4.0),
    ("max_1.0s", "max", 1.0),
    ("max_2.0s", "max", 2.0),
    ("ema_tau0.5s", "ema", 0.5),
    ("ema_tau1.0s", "ema", 1.0),
    ("median_1.0s", "median", 1.0),
    ("gauss_1.0s", "gauss", 1.0),
    # short-window family from the held-out scorer sweep (tri:0.25 was the best
    # single default; the others win on a few specific models).
    ("mean_0.125s", "mean", 0.125),
    ("tri_0.1s", "tri", 0.1),
    ("tri_0.15s", "tri", 0.15),
    ("tri_0.2s", "tri", 0.2),
    ("tri_0.25s", "tri", 0.25),
    ("median_0.1875s", "median", 0.1875),
    ("max_0.25s", "max", 0.25),
]

MASS_COMBOS = [[1.4, 1.4], [1.5, 1.5], [2.0, 2.0], [2.3, 2.3]]
IFOS = ["H1", "L1"]


class Postprocessor:
    """Faithful copy of ``infer.postprocess.Postprocessor`` whose ``integrate``
    step is generalised to the methods explored in the notebook.  Offset,
    clustering and time bookkeeping are identical to the repo."""

    def __init__(
        self,
        t0,
        shifts,
        psd_length,
        fduration,
        inference_sampling_rate,
        integration_window_length,
        cluster_window_length,
        method="mean",
    ):
        self.inference_sampling_rate = inference_sampling_rate
        self.shifts = shifts
        self.method = method
        # offset t0 by the PSD burn-in, filter settle-in and integration lag
        self.t0 = t0 + psd_length - fduration / 2 - integration_window_length
        self.offset = int(psd_length * inference_sampling_rate)
        # repo boxcar size convention; also used as the window for the
        # other (max/median/...) filters
        self.integration_window_size = (
            int(inference_sampling_rate * integration_window_length) + 1
        )
        self.win_samples = max(
            int(inference_sampling_rate * integration_window_length), 1
        )
        self.cluster_window_size = int(
            inference_sampling_rate * cluster_window_length
        )

    def integrate(self, y):
        kind = self.method
        if kind == "none":
            return y
        if kind == "mean":
            window_size = self.integration_window_size
            window = np.ones((window_size,)) / window_size
            integrated = np.convolve(y, window, mode="full")
            return integrated[: -window_size + 1]
        if kind == "max":
            # causal: origin shifts the window to look only at past samples
            return maximum_filter1d(
                y,
                size=self.win_samples,
                origin=(self.win_samples - 1) // 2,
            )
        if kind == "tri":
            # centred triangular (Bartlett) taper of half-width win_samples;
            # down-weights the window edges relative to a flat boxcar.
            size = 2 * self.win_samples + 1
            kernel = np.bartlett(size)
            kernel /= kernel.sum()
            return np.convolve(y, kernel, mode="same")
        if kind == "median":
            return median_filter(y, size=self.win_samples, mode="nearest")
        if kind == "ema":
            alpha = 1.0 / max(self.win_samples, 1)
            return lfilter([alpha], [1.0, -(1.0 - alpha)], y)
        if kind == "gauss":
            return gaussian_filter1d(y, sigma=max(self.win_samples / 4.0, 0.5))
        raise ValueError(f"unknown integration method {kind!r}")

    def cluster(self, y):
        window_size = int(self.cluster_window_size // 2)
        i = np.argmax(y[:window_size])
        events, times = [], []
        while i < len(y):
            val = y[i]
            window = y[i + 1 : i + 1 + window_size]
            if (val < window).any():
                i += np.argmax(window) + 1
            else:
                events.append(val)
                times.append(self.t0 + i / self.inference_sampling_rate)
                i += window_size + 1
        Tb = len(y) / self.inference_sampling_rate
        events = np.array(events)
        times = np.array(times)
        shifts = np.ones((len(events), len(self.shifts))) * self.shifts
        return EventSet(events, times, shifts, Tb)

    def __call__(self, y):
        if y is None:
            return EventSet()
        y = y[self.offset :]
        y = self.integrate(y)
        return self.cluster(y)


def parse_key(key):
    """Keys look like ``'1241443783.0_[0 1]'`` -> (t0, shift tuple)."""
    t0_str, shift_str = key.split("_", 1)
    shift = tuple(int(x) for x in shift_str.strip("[]").split())
    return float(t0_str), shift


def load_timeseries(ts_path, max_segments=None):
    """Load every segment's raw bg/fg timeseries into memory."""
    segments = []
    with h5py.File(ts_path, "r") as f:
        keys = list(f["timeseries"].keys())
        if max_segments:
            keys = keys[:max_segments]
        for key in tqdm(keys, desc="loading timeseries"):
            t0, shift = parse_key(key)
            grp = f["timeseries"][key]
            segments.append(
                {
                    "t0": t0,
                    "shift": shift,
                    "bg": grp["background"][:],
                    "fg": grp["foreground"][:],
                }
            )
    return segments


def nearest_recover(ev_times, ev_stats, inj_times):
    """For each injection, the statistic/time of the event nearest in time
    (the repo's nearest-in-time recovery, done with searchsorted instead of a
    full N x M difference matrix to stay memory-cheap)."""
    if ev_times.size == 0:
        return (
            np.full(inj_times.shape, -np.inf),
            inj_times.copy(),
        )
    order = np.argsort(ev_times)
    et, es = ev_times[order], ev_stats[order]
    pos = np.searchsorted(et, inj_times)
    pos = np.clip(pos, 1, len(et) - 1)
    left, right = pos - 1, pos
    take_left = np.abs(inj_times - et[left]) <= np.abs(et[right] - inj_times)
    choice = np.where(take_left, left, right)
    return es[choice], et[choice]


def score_method(name, kind, win_seconds, segments, inj, args, out_root):
    """Regenerate background/foreground for one method and plot its SV."""
    method_dir = out_root / name

    # skip methods that have already been fully computed (unless --force).
    # the final plot is the last artifact written, so its presence means the
    # whole pipeline (postprocess + SV + GWTC-3 reference) ran to completion.
    final_plot = out_root / f"{name}.png"
    if not args.force and final_plot.exists():
        logging.info("skipping %s -- already done (%s)", name, final_plot)
        return

    method_dir.mkdir(parents=True, exist_ok=True)

    bg_stats, bg_times, bg_shifts = [], [], []
    Tb_total = 0.0
    # foreground events accumulated per timeslide shift
    fg_by_shift = {sh: ([], []) for sh in inj["by_shift"]}

    for seg in tqdm(segments, desc=f"{name} postprocess", leave=False):
        pp = Postprocessor(
            t0=seg["t0"],
            shifts=list(seg["shift"]),
            psd_length=args.psd_length,
            fduration=args.fduration,
            inference_sampling_rate=args.rate,
            integration_window_length=win_seconds,
            cluster_window_length=args.cluster_window,
            method=kind,
        )
        # background -> FAR (zero-lag excluded)
        if seg["shift"] != ZEROLAG_SHIFT and seg["bg"].size > pp.offset:
            ev = pp(seg["bg"])
            if len(ev):
                bg_stats.append(ev.detection_statistic)
                bg_times.append(ev.detection_time)
                bg_shifts.append(ev.shift)
            Tb_total += ev.Tb
        # foreground events for shifts that carry injections
        if seg["shift"] in fg_by_shift and seg["fg"].size > pp.offset:
            ev = pp(seg["fg"])
            if len(ev):
                fg_by_shift[seg["shift"]][0].append(ev.detection_time)
                fg_by_shift[seg["shift"]][1].append(ev.detection_statistic)

    # --- write background.hdf5 -------------------------------------------
    background = EventSet(
        detection_statistic=np.concatenate(bg_stats),
        detection_time=np.concatenate(bg_times),
        shift=np.concatenate(bg_shifts),
        Tb=Tb_total,
    )
    background.write(method_dir / "background.hdf5")

    # --- recover injections, in the global injection order ---------------
    recovered_stat = np.full(inj["n"], -np.inf)
    recovered_time = inj["injection_time"].copy()
    for sh, idx in inj["by_shift"].items():
        times_list, stats_list = fg_by_shift[sh]
        if not times_list:
            continue
        ev_t = np.concatenate(times_list)
        ev_s = np.concatenate(stats_list)
        stat, time = nearest_recover(ev_t, ev_s, inj["injection_time"][idx])
        recovered_stat[idx] = stat
        recovered_time[idx] = time

    # --- write foreground.hdf5: clone the shipped ledger, overwrite the
    #     two detection columns (all injection params stay valid) ----------
    fg_path = method_dir / "foreground.hdf5"
    shutil.copy2(args.foreground, fg_path)
    with h5py.File(fg_path, "r+") as f:
        p = f["parameters"]
        p["detection_statistic"][:] = recovered_stat
        p["detection_time"][:] = recovered_time
        f.attrs["Tb"] = Tb_total

    # --- run the production sensitive-volume plotter ---------------------
    calc_sensitive_volume(
        background=method_dir / "background.hdf5",
        foreground=fg_path,
        rejected_params=Path(args.rejected_params),
        ifos=IFOS,
        mass_combos=MASS_COMBOS,
        source_prior=end_o3_ratesandpops_bns,
        output_dir=method_dir,
        dt=args.dt,
        backend="matplotlib",
    )
    shutil.copy2(method_dir / "sensitive_volume.png", out_root / f"{name}.png")
    logging.info("finished method %s -> %s.png", name, name)


def _read_method_sv(method_dir, combo_keys):
    """Load a method's saved SV curve (``fars`` + per-combo ``sv``/``err``)."""
    path = method_dir / "sensitive_volume.h5"
    if not path.exists():
        return None
    with h5py.File(path, "r") as f:
        fars = f["fars"][:]
        sv = {k: f[k]["sv"][:] for k in combo_keys}
        err = {k: f[k]["err"][:] for k in combo_keys}
    return fars, sv, err


def plot_all_methods(out_root, methods, mass_combos):
    """Overlay every method's SV-vs-FAR on a single figure (no reference
    pipelines), highlighting the best method.

    "Best" is measured by sensitive volume at the left (most stringent, i.e.
    smallest FAR) edge of the plot, averaged across mass combos as a fraction
    of the best method in each combo so the heavier-mass (larger-volume)
    combos don't dominate the ranking.
    """
    import matplotlib.pyplot as plt

    from plots.legacy.matplotlib_tools import make_grid, plot_err_bands

    combo_keys = ["-".join(map(str, c)) for c in mass_combos]

    data = {}
    for name, _, _ in methods:
        res = _read_method_sv(out_root / name, combo_keys)
        if res is not None:
            data[name] = res
    if not data:
        logging.warning("no per-method SV data found; skipping combined plot")
        return

    # rank by left-edge SV, normalised per combo so all combos count equally
    left = {
        n: np.array([sv[k][0] for k in combo_keys])
        for n, (_, sv, _) in data.items()
    }
    best_per_combo = np.max(np.stack(list(left.values())), axis=0)
    best_per_combo[best_per_combo == 0] = 1.0
    ranking = sorted(
        data, key=lambda n: np.mean(left[n] / best_per_combo), reverse=True
    )
    top = ranking[0]
    logging.info("combined plot: top method by left-edge SV = %s", top)

    names = sorted(data)
    cmap = plt.get_cmap("tab20", max(len(names), 1))
    colors = {name: cmap(i) for i, name in enumerate(names)}

    fig, axes = make_grid(mass_combos)
    for i, ax in enumerate(axes):
        key = combo_keys[i]
        # faded non-top methods underneath
        for name in names:
            if name == top:
                continue
            fars, sv, _ = data[name]
            linewidth = 2.0 if "default" in name else 1.0
            alpha = 0.8 if "default" in name else 0.4
            ax.plot(
                fars,
                sv[key],
                linewidth=linewidth,
                color=colors[name],
                alpha=alpha,
                zorder=2,
                label=name if i == 0 else None,
            )
        # highlighted top method on top
        fars, sv, err = data[top]
        ax.plot(
            fars,
            sv[key],
            linewidth=2.5,
            color="k",
            zorder=5,
            label=f"{top} (top)" if i == 0 else None,
        )
        plot_err_bands(ax, fars, sv[key], err[key], color="k", alpha=0.2)

    # legend on the top-left panel, with the highlighted method first
    handles, labels = axes[0].get_legend_handles_labels()
    order = sorted(
        range(len(labels)), key=lambda j: labels[j] != f"{top} (top)"
    )
    axes[0].legend(
        [handles[j] for j in order],
        [labels[j] for j in order],
        loc="lower right",
        fontsize=7,
        ncol=2,
        handlelength=1.5,
        borderpad=0.4,
        labelspacing=0.3,
    )

    out = out_root / "all_methods.png"
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    logging.info("combined plot -> %s", out)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--run-dir",
        required=True,
        help="results_aframe directory containing timeseries.hdf5 / "
        "foreground.hdf5",
    )
    ap.add_argument(
        "--rejected-params",
        default="/fast/barmstrong/ligoss/data/bns/aframe_test/"
        "rejected-parameters.hdf5",
    )
    ap.add_argument(
        "--out-subdir",
        default="plots/sv",
        help="output folder (relative to run-dir) for the per-method plots",
    )
    ap.add_argument(
        "--max-segments",
        type=int,
        default=None,
        help="only use the first N segments (quick smoke test)",
    )
    ap.add_argument(
        "--methods",
        nargs="*",
        default=None,
        help="subset of method labels to run (default: all)",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="recompute methods even if their output plot already exists",
    )
    ap.add_argument("--rate", type=float, default=RATE)
    ap.add_argument("--psd-length", type=float, default=PSD_LENGTH)
    ap.add_argument("--fduration", type=float, default=FDURATION)
    ap.add_argument("--cluster-window", type=float, default=CLUSTER_WINDOW)
    ap.add_argument("--dt", type=float, default=1.0)
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    run_dir = Path(args.run_dir)
    args.foreground = run_dir / "foreground.hdf5"
    out_root = run_dir / args.out_subdir
    out_root.mkdir(parents=True, exist_ok=True)

    methods = DEFAULT_METHODS
    if args.methods:
        methods = [m for m in DEFAULT_METHODS if m[0] in args.methods]
        if not methods:
            raise SystemExit(f"no methods matched {args.methods}")

    # drop methods that are already done so we can skip loading the (large)
    # timeseries entirely if there is nothing left to compute.
    selected = methods  # full set requested -- used for the combined plot
    if not args.force:
        pending = [
            m for m in methods if not (out_root / f"{m[0]}.png").exists()
        ]
        skipped = [m[0] for m in methods if m not in pending]
        if skipped:
            logging.info(
                "skipping already-done methods: %s", ", ".join(skipped)
            )
    else:
        pending = methods

    if pending:
        logging.info(
            "Loading raw timeseries from %s", run_dir / "timeseries.hdf5"
        )
        segments = load_timeseries(
            run_dir / "timeseries.hdf5", args.max_segments
        )

        # injection truth from the shipped foreground ledger (one row per
        # injection, with all source parameters needed by the SV plotter)
        with h5py.File(args.foreground, "r") as f:
            injection_time = f["parameters"]["injection_time"][:]
            inj_shift = f["parameters"]["shift"][:].astype(int)
        by_shift = {}
        for sh in np.unique(inj_shift, axis=0):
            by_shift[tuple(sh)] = np.flatnonzero((inj_shift == sh).all(axis=1))
        inj = {
            "n": len(injection_time),
            "injection_time": injection_time,
            "by_shift": by_shift,
        }
        logging.info(
            "injections per shift: %s",
            {k: len(v) for k, v in by_shift.items()},
        )

        logging.info("Running %d methods -> %s", len(pending), out_root)
        for name, kind, win in pending:
            logging.info("=== method %s (%s, %.2fs) ===", name, kind, win)
            score_method(name, kind, win, segments, inj, args, out_root)
    else:
        logging.info(
            "all methods already computed; (re)building combined plot"
        )

    # combined overview of every method (no reference pipelines)
    plot_all_methods(out_root, selected, MASS_COMBOS)

    logging.info("All done. Per-method plots in %s", out_root)


if __name__ == "__main__":
    main()
