import sys
from pathlib import Path

import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

ROOT = Path(sys.argv[1])
C = ["#13386b", "#2a78d6", "#1baf7a", "#8a96a8"]
INK, MUTED, GRID = "#0F1E33", "#5b6574", "#e3e6ea"
plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "legend.frameon": False, "lines.linewidth": 2,
})
pct = matplotlib.ticker.PercentFormatter(1.0, decimals=0)
BINS = np.array([4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 20, 25, 35, 50])


def load(pop):
    w = h5py.File(ROOT / pop / "windows.h5")
    m = h5py.File(ROOT / pop / "mf.h5")
    names = list(w.attrs["param_names"])
    i = names.index("chirp_mass")
    return dict(
        true=w["params/chirp_mass"][:], snr=w["params/snr"][:],
        ml=w["ml_mean"][:, i], ml_sigma=w["ml_sigma"][:, i],
        mf=m["mf_chirp_mass"][:], mf_snr=m["mf_snr"][:],
    )


pops = [p for p in ("snr4_powerlaw", "snr8_powerlaw", "snr4_uniform") if (ROOT / p / "mf.h5").exists()]
data = {p: load(p) for p in pops}

# pool all signal rows for accuracy against SNR
true = np.concatenate([data[p]["true"] for p in pops])
sig = ~np.isnan(true)
snr = np.concatenate([data[p]["snr"] for p in pops])[sig]
true = true[sig]
est = {k: np.concatenate([data[p][k] for p in pops])[sig] for k in ("mf", "ml")}
centers = np.sqrt(BINS[:-1] * BINS[1:])
fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharey=True)
for ax, tol in zip(axes.flat, (0.01, 0.02, 0.05, 0.10)):
    for (k, lab), c, ls in zip((("mf", "matched filter"), ("ml", "ML, epoch 630")), C, ("-", "--")):
        ok = np.abs(est[k] - true) / true < tol
        frac = [ok[(snr >= a) & (snr < b)].mean() for a, b in zip(BINS[:-1], BINS[1:])]
        ax.plot(centers, frac, color=c, ls=ls, marker="o", ms=4, label=lab)
    ax.set_xscale("log")
    ax.set_xticks([4, 6, 8, 12, 16, 25, 50], ["4", "6", "8", "12", "16", "25", "50"])
    ax.set_ylim(0, 1.02)
    ax.yaxis.set_major_formatter(pct)
    ax.set_title(f"chirp mass within {tol:.0%}", color=INK, fontsize=10)
    ax.set_xlabel("injected SNR")
    ax.set_ylabel("fraction of events")
h, l = axes[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=2, fontsize=9)
fig.suptitle("Same 4 s test windows: matched filter against the network", color=INK, fontsize=11)
fig.tight_layout(rect=(0, 0.05, 1, 1))
fig.savefig(ROOT / "mf_vs_ml_accuracy.png", dpi=110)

# ROC per population, and SNR 8-12 only
fig, axes = plt.subplots(2, 2, figsize=(10, 9), sharey=True)
axes = axes.flat
panels = [(p, None) for p in pops] + [(pops[0], (8, 12))]
for ax, (p, band) in zip(axes, panels):
    d = data[p]
    s = ~np.isnan(d["true"])
    keep = s if band is None else s & (d["snr"] >= band[0]) & (d["snr"] < band[1])
    rows = keep | ~s
    y = s[rows].astype(int)
    for (score, lab), c, ls in zip(((d["mf_snr"], "matched filter (network SNR)"), (-d["ml_sigma"], "ML (−σ)")), C, ("-", "--")):
        fpr, tpr, _ = roc_curve(y, score[rows])
        ax.plot(fpr, tpr, color=c, ls=ls, label=lab)
    ax.plot([0, 1], [0, 1], color=GRID, lw=1)
    title = p.replace("snr4_", "SNR 4+, ").replace("snr8_", "SNR 8+, ").replace("powerlaw", "power law")
    ax.set_title(title if band is None else "SNR 8 to 12 only (power law from 4)", color=INK, fontsize=10)
    ax.set_xlabel("false positive rate")
axes = fig.axes
for a in axes:
    a.set_ylabel("true positive rate")
axes[0].legend(loc="lower right", fontsize=8)
fig.tight_layout()
fig.savefig(ROOT / "mf_vs_ml_roc.png", dpi=110)

# predicted against true chirp mass at SNR 8 to 12, and score distributions
mf_snr_all = np.concatenate([data[p]["mf_snr"] for p in pops])[sig]
edges = np.geomspace(4, 50, 31)
mid = np.sqrt(edges[:-1] * edges[1:])
fig, axes = plt.subplots(1, 2, figsize=(12, 5.6))
ax = axes[0]
med, lo, hi = [], [], []
for a_, b_ in zip(edges[:-1], edges[1:]):
    v = mf_snr_all[(snr >= a_) & (snr < b_)]
    q = np.percentile(v, [16, 50, 84]) if len(v) > 4 else [np.nan] * 3
    lo.append(q[0]); med.append(q[1]); hi.append(q[2])
ax.fill_between(mid, lo, hi, color=C[0], alpha=0.18, lw=0, label="1σ band")
ax.plot(mid, med, color=C[0], label="median")
lim = (4, 50)
ax.plot(lim, lim, color=MUTED, ls=":", lw=1.2)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect("equal")
ticks = [4, 6, 8, 12, 16, 25, 50]
ax.set_xticks(ticks, [str(t) for t in ticks]); ax.set_yticks(ticks, [str(t) for t in ticks])
ax.minorticks_off()
ax.set_xlabel("injected SNR"); ax.set_ylabel("matched-filter network SNR")
ax.set_title(f"Recovered against injected SNR ({sig.sum()} events)", color=INK, fontsize=10)
ax.legend(loc="upper left", fontsize=8)

ax = axes[1]
d = data[pops[0]]
s = ~np.isnan(d["true"])
weak = s & (d["snr"] >= 8) & (d["snr"] < 12)
bins = np.linspace(4, 16, 61)
ax.hist(d["mf_snr"][~s], bins=bins, density=True, color=C[3], alpha=0.8, label=f"empty data, no signal ({(~s).sum()} windows)")
ax.hist(d["mf_snr"][weak], bins=bins, density=True, color=C[0], alpha=0.7, label=f"data with a weak signal, SNR 8 to 12 ({weak.sum()})")
ax.set_xlabel("matched-filter score (best network SNR over the bank)")
ax.set_ylabel("density")
ax.set_title("Matched-filter score: empty data against weak signals", color=INK, fontsize=10)
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(ROOT / "mf_snr.png", dpi=110)
print("saved", ROOT)
