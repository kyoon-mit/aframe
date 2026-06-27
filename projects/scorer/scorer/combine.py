"""Overlay one integration method's held-out sensitive-volume curve across
every model, to check whether a method (e.g. ``tri:0.25``) wins reliably."""

import logging

import h5py
import matplotlib.pyplot as plt

from .core import MASS_COMBOS

COMBO_KEYS = ["-".join(map(str, c)) for c in MASS_COMBOS]


def _safe(label):
    return label.replace(":", "_").replace("@", "_at_")


def _read_sv(path):
    with h5py.File(path, "r") as f:
        fars = f["fars"][:]
        sv = {k: f[k]["sv"][:] for k in COMBO_KEYS}
    return fars, sv


def combine_models(runs, out_root, method, out_path):
    """One 2x2 (mass-combo) grid overlaying ``method``'s SV curve per run."""
    present = []
    for name in runs:
        sv_path = out_root / name / _safe(method) / "sensitive_volume.h5"
        if sv_path.exists():
            present.append((name, sv_path))
        else:
            logging.warning("no %s SV for %s (skipping)", method, name)
    if not present:
        logging.warning("nothing to combine for method %s", method)
        return None

    cmap = plt.get_cmap("tab20")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    axes = axes.flatten()
    for i, (ax, key) in enumerate(zip(axes, COMBO_KEYS, strict=False)):
        ax.set_xscale("log")
        ax.set_title(f"$m_1$-$m_2$ = {key}")
        if i % 2 == 0:
            ax.set_ylabel("Sensitive Volume [Gpc$^3$]")
        if i >= 2:
            ax.set_xlabel("False Alarm Rate [yr$^{-1}$]")
        for j, (name, sv_path) in enumerate(present):
            fars, sv = _read_sv(sv_path)
            ax.plot(
                fars,
                sv[key],
                color=cmap(j % 20),
                linewidth=1.8,
                label=name if i == 0 else None,
            )

    fig.legend(
        loc="lower center",
        ncol=4,
        fontsize=8,
        frameon=False,
        bbox_to_anchor=(0.5, -0.03),
    )
    fig.suptitle(f"{method} sensitive volume across models (held-out)")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    logging.info("wrote %s (%d models)", out_path, len(present))
    return out_path
