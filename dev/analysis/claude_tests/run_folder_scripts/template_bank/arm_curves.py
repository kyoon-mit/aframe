"""Validation within 1/2% against epoch for the long-convolution arms and the den-reg run."""
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import wandb
api = wandb.Api(); E = "kyoon-mit-massachusetts-institute-of-technology"
runs = [("den-reg (denoiser + 12 S4D layers)", f"{E}/DENOISER-SCAN/keej8hie", "#8a96a8", ":"),
        ("S4D, d_state 64, 1 layer", f"{E}/CLAUDE-TESTS/s4d_energy_max_256ch_dstate64_softmax_train_snr8_prob0.96", "#2a78d6", "--"),
        ("S4D, d_state 512, 1 layer", f"{E}/CLAUDE-TESTS/s4d_energy_max_256ch_dstate512_softmax_train_snr8_prob0.96", "#13386b", "-"),
        ("free kernels, 1 layer", f"{E}/CLAUDE-TESTS/free_kernel_energy_max_256ch_softmax_train_snr8_prob0.96", "#1baf7a", "-"),
        ("S4D width: 1024 ch mixed to 256", f"{E}/CLAUDE-TESTS/s4d_energy_max_1024ch_dstate64_mixed_256_softmax_train_snr8_prob0.96", "#c0392b", "-."),
        ("S4D depth: den-reg regressor, no denoiser", f"{E}/CLAUDE-TESTS/s4d_regressor_6layer_dstate64_energy_max_softmax_train_snr8_prob0.96", "#7d3c98", "--"),
        ("aframe ResNet1D", f"{E}/CLAUDE-TESTS/aframe_resnet1d_3463_softmax_train_snr8_prob0.96", "#d68910", ":")]
INK, MUTED, GRID = "#0F1E33", "#5b6574", "#e3e6ea"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "lines.linewidth": 1.8, "axes.titlecolor": INK, "axes.titlesize": 10})
fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
for label, path, c, ls in runs:
    h = api.run(path).history(keys=["epoch", "val/within_1pct_chirp_mass", "val/within_2pct_chirp_mass"], pandas=True, samples=10000)
    for ax, key in zip(axes, ["val/within_1pct_chirp_mass", "val/within_2pct_chirp_mass"]):
        e, v = h["epoch"].to_numpy(), h[key].to_numpy()
        k = 9; sm = np.convolve(v, np.ones(k) / k, mode="valid"); es = e[k // 2: len(e) - k // 2]
        ax.plot(es, sm, color=c, ls=ls, label=label)
for ax, t in zip(axes, ["within 1%", "within 2%"]):
    ax.set_title(f"validation: {t} of the true chirp mass"); ax.set_xlabel("epoch"); ax.set_ylabel("fraction of events")
    ax.set_xlim(0, 400); ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=3, fontsize=8)
fig.suptitle("Validation (power law −3 from SNR 4), 9-epoch running means", color=INK, fontsize=11)
fig.tight_layout(rect=(0, 0.14, 1, 1)); fig.savefig(sys.argv[1], dpi=110); print("wrote", sys.argv[1])
