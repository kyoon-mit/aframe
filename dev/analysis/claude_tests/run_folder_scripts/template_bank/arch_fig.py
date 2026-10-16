"""Figures for the write-up: architecture diagram, tone check, one event."""

import json
import sys
from pathlib import Path

import h5py
import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from train.experimental.template_bank.arch import TemplateBankNet

OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)
C = ["#13386b", "#2a78d6", "#1baf7a", "#8a96a8"]
INK, MUTED, GRID, WASH = "#0F1E33", "#5b6574", "#e3e6ea", "#e8eef7"
plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "legend.frameon": False, "lines.linewidth": 1.6,
    "axes.titlecolor": INK, "axes.titlesize": 10,
})
TB = "/home/kyoon/SSM-BNS/aframe/dev/runs/claude_tests/template_bank"
BENCH = ("/home/kyoon/SSM-BNS/aframe/dev/runs/claude_tests/den_reg_sched_alpha_0.5_floor_1e-2_"
         "c_rho_5_lambda_amp_3_powerlaw_m2_snr_d128_n6_mc_q_rep4_smooth20_wp0.96_ddp/mf_bench_ep630")

# 1. block diagram
fig, ax = plt.subplots(figsize=(11, 4.6))
ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(-0.12, 1)
blocks = [
    (0.01, 0.62, "whitened strain\nH1, L1, 4 s\n(B, 2, 8192)", False),
    (0.21, 0.62, "FFT, keep\n20 to 512 Hz\n(1024 Hz rate)", False),
    (0.43, 0.62, "multiply by 703 fixed\ntemplate spectra,\ninverse FFT\n(B, 2, 703, lags)", False),
    (0.64, 0.62, "SNR² = |z|²\nH1 + best L1\nwithin 10 ms", False),
    (0.83, 0.62, "max over lags\n(merger in\n3.45 to 4.05 s)\n(B, 703)", False),
    (0.83, 0.12, "β · SNR²,\nlog-sum-exp over\n3 spins, + bias\n(B, 235)", True),
    (0.60, 0.12, "softmax:\nposterior over\nchirp mass", True),
    (0.37, 0.12, "peak + parabola:\nchirp mass\nestimate", False),
    (0.14, 0.12, "χ² test per detector\n(16 bands): glitch-\nreweighted SNR²", False),
]
w, h = 0.16, 0.30
for x, y, text, learned in blocks:
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01,rounding_size=0.02",
                                fc="#d6efe6" if learned else WASH, ec=C[2] if learned else C[0], lw=1.3))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8.5, color=INK)


def arrow(a, b):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=12, color=C[0]))


for i in range(4):
    arrow((blocks[i][0] + w + 0.003, 0.77), (blocks[i + 1][0] - 0.003, 0.77))
arrow((0.91, 0.62), (0.91, 0.42))
arrow((0.83 - 0.003, 0.27), (0.60 + w + 0.003, 0.27))
arrow((0.60 - 0.003, 0.27), (0.37 + w + 0.003, 0.27))
arrow((0.72, 0.62), (0.22, 0.42))
ax.add_patch(FancyBboxPatch((0.02, -0.02), 0.025, 0.05, boxstyle="round,pad=0.005", fc="#d6efe6", ec=C[2]))
ax.text(0.055, 0.005, "learned: β and one bias per chirp mass (236 numbers) and a detection affine map (2)", fontsize=8.5, color=MUTED, va="center")
ax.text(0.055, -0.08, "no denoiser, no mean pooling; templates are injection-file waveforms, whitened once with an average O3a PSD", fontsize=8.5, color=MUTED, va="center")
fig.savefig(OUT / "template_bank_architecture.png", dpi=110, bbox_inches="tight")
plt.close(fig)

# 2. tone check, site palette, shared y
res = json.load(open("/home/kyoon/SSM-BNS/aframe/dev/runs/claude_tests/tone_check/results.json"))
print("tone_check keys:", list(res)[:10])
plt.close("all")

names = {"A_best_fourier": "best N Fourier tones (undamped)",
         "B_fixed_grid": "fixed tone grid, 20 to 800 Hz",
         "C_matrix_pencil": "free stable damped tones (matrix pencil)"}
fig, axes = plt.subplots(2, 2, figsize=(10, 7.4), sharey=True)
mcs = ["1.0", "1.2", "1.5", "2.0"]
for ax, (key, title) in zip(axes.flat[:3], names.items()):
    for mc, c, ls in zip(mcs, C, ["-", "--", ":", "-."]):
        d = res["match"][key][mc]
        n = sorted(int(k) for k in d)
        ax.plot(n, [d[str(k)] for k in n], color=c, ls=ls, marker="o", ms=3, label=f"chirp mass {mc} M☉")
    ax.set_title(title)
best = {}
ax = axes.flat[3]
for mc, c, ls in zip(mcs, C, ["-", "--", ":", "-."]):
    n = sorted(int(k) for k in res["match"]["A_best_fourier"][mc])
    m = [max(res["match"][k][mc][str(i)] for k in names) for i in n]
    ax.plot(n, m, color=c, ls=ls, marker="o", ms=3, label=f"chirp mass {mc} M☉")
ax.set_title("best of the three")
for ax in axes.flat:
    ax.set_xscale("log", base=2); ax.set_ylim(0, 1.02)
    ax.set_xticks([8, 32, 128, 512, 2048], ["8", "32", "128", "512", "2048"]); ax.minorticks_off()
    ax.axvline(32, color=MUTED, lw=1, ls=":")
    ax.text(34, 0.04, "den-reg S4D:\n32 tones", fontsize=8, color=MUTED)
    ax.set_xlabel("tones per channel"); ax.set_ylabel("match to the chirp kernel")
axes[0, 0].legend(fontsize=8, loc="upper left")
fig.suptitle("How well one S4D channel can hold a whitened 4 s BNS chirp", color=INK, fontsize=11)
fig.tight_layout(); fig.savefig(OUT / "tone_check.png", dpi=110); plt.close(fig)
print("wrote figures")
