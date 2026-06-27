"""Evaluate the learned scorers on the held-out segments and put them on the
same sensitive-volume axes as the (re-baselined) integration methods and the
external reference."""

import logging
import pickle

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch

from .core import MASS_COMBOS, normalize, time_split
from .features import window_features
from .models import TinyCNN, dense_cnn_score
from .pipeline import (
    boxcar,
    candidate_producer,
    dense_producer,
    evaluate,
    make_method,
    test_injections,
)


def _safe(label):
    """Filesystem-safe directory name for a method label."""
    return label.replace(":", "_").replace("@", "_at_")


COMBO_KEYS = ["-".join(map(str, c)) for c in MASS_COMBOS]


def _cnn_producer(obj_dir, args, device):
    ckpt = torch.load(obj_dir / "cnn.pt", map_location=device)
    model = TinyCNN(channels=ckpt["channels"], kernel=ckpt["kernel"]).to(
        device
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    stats, L = ckpt["stats"], ckpt["L"]

    def score_fn(y):
        return dense_cnn_score(
            model, normalize(y, stats), L, args.stride, device
        )

    return dense_producer(score_fn, 0.0, args)


def _feat_producer(obj_dir, args):
    with open(obj_dir / "features.pkl", "rb") as fh:
        feat = pickle.load(fh)
    model, task, stats = feat["model"], feat["task"], feat["stats"]
    # classifier -> log-odds margin (unbounded, no tie pile-up at prob=1);
    # regressor -> predicted SNR.  Both are monotone-in-loudness statistics.
    fn = model.decision_function if task == "classifier" else model.predict

    def prob_fn(W):
        Wn = np.stack([normalize(w, stats) for w in W])
        return fn(window_features(Wn))

    base_fn, base_lag = boxcar(args.baseline_win, args.rate)
    return candidate_producer(base_fn, base_lag, prob_fn, args)


def _producers(model_dir, args, device):
    """Build the producer set: the integration methods (and an optional
    cluster-window sweep on one of them), plus -- unless ``--skip-learned`` --
    every trained objective's CNN and feature model."""
    producers = {}
    if not args.skip_learned:
        for obj_dir in sorted(
            p for p in model_dir.iterdir() if (p / "cnn.pt").exists()
        ):
            obj = obj_dir.name
            producers[f"cnn_{obj}"] = _cnn_producer(obj_dir, args, device)
            if (obj_dir / "features.pkl").exists():
                producers[f"feat_{obj}"] = _feat_producer(obj_dir, args)

    # integration methods at the default clustering window
    for spec in args.methods:
        fn, lag = make_method(spec, args.rate)
        producers[spec] = dense_producer(fn, lag, args)

    # clustering-window sweep: hold one integration method fixed, vary cw
    if args.cluster_windows:
        fn, lag = make_method(args.cluster_sweep_method, args.rate)
        for cw in args.cluster_windows:
            producers[f"{args.cluster_sweep_method}@cw{cw:g}"] = (
                dense_producer(fn, lag, args, cluster_window=cw)
            )
    return producers


def evaluate_model(run_dir, name, args, model_dir, out_root):
    out_dir = out_root / name
    out_dir.mkdir(parents=True, exist_ok=True)
    device = (
        "cuda"
        if (args.device in ("auto", "cuda") and torch.cuda.is_available())
        else "cpu"
    )

    ts_path = run_dir / "timeseries.hdf5"
    with h5py.File(ts_path, "r") as f:
        split = time_split(f["timeseries"], "background", args.train_frac)
    test_keys = split.test
    if args.max_segments:
        test_keys = set(sorted(test_keys)[: args.max_segments])
    logging.info(
        "[%s] evaluating on %d held-out segments (%s)",
        name,
        len(test_keys),
        device,
    )

    inj = test_injections(
        run_dir / "foreground.hdf5", ts_path, test_keys, args
    )
    logging.info("[%s] %d held-out injections", name, int(inj[0].sum()))

    producers = _producers(model_dir, args, device)
    sv_paths = {}
    for label, produce in producers.items():
        sv = evaluate(
            run_dir,
            test_keys,
            produce,
            inj,
            args,
            out_dir / _safe(label),
            args.rejected_params,
        )
        if sv is not None:
            sv_paths[label] = sv

    _report_detected(name, out_dir, list(sv_paths))
    if sv_paths:
        _comparison_plot(name, out_dir, sv_paths, args)
    else:
        logging.warning(
            "[%s] no SV produced (too little held-out livetime?)", name
        )
    return sv_paths


def _report_detected(name, out_dir, labels):
    """Log the actual objective: how many injections clear the FAR-floor
    threshold (the n-th loudest background event) for each method."""
    logging.info("[%s] injections above FAR-floor threshold:", name)
    rows = []
    for label in labels:
        try:
            with h5py.File(
                out_dir / _safe(label) / "background.hdf5", "r"
            ) as f:
                bg = f["parameters"]["detection_statistic"][:]
            with h5py.File(
                out_dir / _safe(label) / "foreground.hdf5", "r"
            ) as f:
                fg = f["parameters"]["detection_statistic"][:]
        except OSError:
            continue
        thr = np.sort(bg)[-15] if len(bg) >= 15 else bg.min()
        rows.append((int((fg > thr).sum()), label))
    for n_above, label in sorted(rows, reverse=True):
        logging.info("    %5d   %s", n_above, label)


def _read_sv(path):
    with h5py.File(path, "r") as f:
        fars = f["fars"][:]
        sv = {k: f[k]["sv"][:] for k in COMBO_KEYS}
    return fars, sv


def _comparison_plot(name, out_dir, sv_paths, args):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    axes = axes.flatten()
    labels = list(sv_paths)
    cmap = plt.get_cmap("tab20")

    def style(label):
        if label.startswith(("cnn_", "feat_")):
            return "--", 1.6  # learned scorers
        if "@cw" in label:
            return ":", 2.0  # cluster-window sweep
        return "-", 2.2  # integration methods

    for i, (ax, key) in enumerate(zip(axes, COMBO_KEYS, strict=False)):
        ax.set_xscale("log")
        ax.set_title(f"$m_1$-$m_2$ = {key}")
        if i % 2 == 0:
            ax.set_ylabel("Sensitive Volume [Gpc$^3$]")
        if i >= 2:
            ax.set_xlabel("False Alarm Rate [yr$^{-1}$]")
        for j, label in enumerate(labels):
            ls, lw = style(label)
            fars, sv = _read_sv(sv_paths[label])
            ax.plot(
                fars,
                sv[key],
                color=cmap(j % 20),
                linewidth=lw,
                linestyle=ls,
                label=label if i == 0 else None,
            )

    # reference omitted here: it is full-data and a much stronger network, so
    # it dwarfs the held-out curves; the comparison of interest is learned vs
    # boxcar
    fig.legend(
        loc="lower center",
        ncol=4,
        fontsize=8,
        frameon=False,
        bbox_to_anchor=(0.5, -0.03),
    )
    has_learned = any(lab.startswith(("cnn_", "feat_")) for lab in labels)
    what = (
        "learned scorers vs integration methods"
        if has_learned
        else "integration methods"
    )
    fig.suptitle(f"{name} — {what} (held-out)")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    out = out_dir / "comparison.png"
    fig.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)
    logging.info("[%s] wrote %s", name, out)
