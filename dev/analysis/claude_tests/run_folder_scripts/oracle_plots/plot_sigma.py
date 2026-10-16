"""Validation predicted sigma against actual error for the three signal-only runs."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
d = json.load(open(HERE / "val_sigma.json"))
RUNS = [("sig_only_reg_free_kernel", "free kernels", "#1baf7a"),
        ("sig_only_reg_s4d_256x256x4", "S4D 256 × 256 × 4", "#13386b"),
        ("sig_only_reg_s4d_64x32x4", "S4D 64 × 32 × 4", "#2a78d6")]
INK, MUTED, GRID = "#0F1E33", "#5b6574", "#e3e6ea"
plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "legend.frameon": False, "lines.linewidth": 1.6,
    "axes.titlecolor": INK, "axes.titlesize": 10,
})
fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.4))
for name, label, c in RUNS:
    eps = sorted(int(e) for e in d[name])
    sig = [d[name][str(e)][0] for e in eps]
    mae = [d[name][str(e)][1] for e in eps]
    a.plot(eps, sig, color=c, label=f"{label}: predicted σ")
    a.plot(eps, mae, color=c, ls="--", label=f"{label}: actual mean |error|")
    b.plot(eps, [m / s for m, s in zip(mae, sig)], color=c, marker="o", ms=2.5, label=label)
a.set_yscale("log"); a.set_xlabel("epoch"); a.set_ylabel("chirp mass (M☉)")
a.set_title("predicted σ (solid) and actual mean |error| (dashed)")
b.axhline(0.8, color=MUTED, ls=":", lw=1.2)
b.text(48, 0.3, "dotted: calibrated Gaussian, 0.80", color=MUTED, fontsize=8.5, ha="right")
b.set_ylim(0, 3); b.set_xlabel("epoch"); b.set_ylabel("mean |error| / mean σ")
b.set_title("above 0.80: σ too small (overconfident)")
h, l = b.get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=3, fontsize=9)
fig.suptitle("Chirp-mass uncertainty on the clean signal: validation", color=INK)
fig.tight_layout(rect=(0, 0.07, 1, 0.95))
fig.savefig(HERE / "val_sigma.png", dpi=110)
