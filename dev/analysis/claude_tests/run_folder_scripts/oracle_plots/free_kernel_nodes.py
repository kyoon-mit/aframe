"""Node-and-line diagram of the free-kernel network as a one-hidden-layer net."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

INK, MUTED, NAVY, GREEN, LINE = "#0F1E33", "#5b6574", "#13386b", "#1baf7a", "#9fb0c8"
fig, ax = plt.subplots(figsize=(12, 7.4))
ax.set_xlim(0, 12); ax.set_ylim(-0.6, 7.6); ax.axis("off")


def column(x, ys, color, fill, r=0.17):
    for y in ys:
        ax.add_patch(plt.Circle((x, y), r, fc=fill, ec=color, lw=1.4, zorder=3))


def dots(x, y):
    ax.text(x, y, "⋮", ha="center", va="center", fontsize=16, color=MUTED, zorder=3)


XI, XH, XO = 1.6, 6.0, 10.2
h1 = [6.9, 6.45, 6.0, 5.55]; h1_last = 4.65
l1 = [3.85, 3.4, 2.95, 2.5]; l1_last = 1.6
inputs = h1 + [h1_last] + l1 + [l1_last]
hidden = [6.6, 6.0, 5.4, 4.8, 4.2, 3.6, 3.0, 2.4]; hidden_last = 1.2
hid = hidden + [hidden_last]
outs = [4.6, 3.4]

for yi in inputs:
    for yh in hid:
        ax.plot([XI, XH], [yi, yh], color=LINE, lw=0.45, zorder=1)
for yh in hid:
    for yo in outs:
        ax.plot([XH, XO], [yh, yo], color=NAVY, lw=0.8, zorder=1, alpha=0.7)

column(XI, inputs, NAVY, "#eef2f7")
dots(XI, 5.1); dots(XI, 2.05)
column(XH, hid, GREEN, "#e3f5ee", r=0.2)
dots(XH, 1.8)
column(XO, outs, NAVY, "#eef2f7", r=0.24)

ax.text(XI - 0.3, h1[0], "x_H1(1)", ha="right", va="center", fontsize=9.5, color=INK)
ax.text(XI - 0.3, h1_last, "x_H1(8192)", ha="right", va="center", fontsize=9.5, color=INK)
ax.text(XI - 0.3, l1[0], "x_L1(1)", ha="right", va="center", fontsize=9.5, color=INK)
ax.text(XI - 0.3, l1_last, "x_L1(8192)", ha="right", va="center", fontsize=9.5, color=INK)
ax.text(XH + 0.32, hidden[0], "h₁", ha="left", va="center", fontsize=11, color=INK)
ax.text(XH + 0.32, hidden[1], "h₂", ha="left", va="center", fontsize=11, color=INK)
ax.text(XH + 0.32, hidden_last, "h₂₅₆", ha="left", va="center", fontsize=11, color=INK)
ax.text(XO + 0.38, outs[0], "μ  (chirp-mass mean)", ha="left", va="center", fontsize=10.5, color=INK)
ax.text(XO + 0.38, outs[1], "σ²  (variance, softplus)", ha="left", va="center", fontsize=10.5, color=INK)

ax.text(XI, 7.45, "Input\n2 detectors × 8192 samples\n(4 s at 2048 Hz)", ha="center", va="bottom", fontsize=10.5, color=INK, weight="bold", linespacing=1.3)
ax.text(XH, 7.45, "Hidden layer\n256 units, one per\nlearned 4 s kernel", ha="center", va="bottom", fontsize=10.5, color=GREEN, weight="bold", linespacing=1.3)
ax.text(XO, 5.2, "Output\n2 units", ha="center", va="bottom", fontsize=10.5, color=INK, weight="bold", linespacing=1.3)

ax.text((XI + XH) / 2, 0.55,
        "weights: kernel c, 8192 complex time samples,\n"
        "the same at every time shift (a convolution)\n"
        "256 × 8192 × 2 numbers (2.06M inside 20 to 1024 Hz)",
        ha="center", va="top", fontsize=9.5, color=MUTED, linespacing=1.35)
ax.text((XH + XO) / 2, 0.55, "weights: 256 × 2 + 2 biases\n= 514 numbers",
        ha="center", va="top", fontsize=9.5, color=MUTED, linespacing=1.35)
ax.text(8.3, 7.12,
        "unit c:\n"
        "1. convolve each detector with kernel c\n"
        "2. energy 2|y|²\n"
        "3. H1 + largest L1 within ±10 ms\n"
        "4. maximum over the 8192 shifts\n"
        "5. log(1 + ·), standardize over units",
        ha="left", va="top", fontsize=9, color=INK, linespacing=1.4,
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=GREEN, lw=1))
fig.tight_layout()
fig.savefig(Path(__file__).with_name("free_kernel_nodes.png"), dpi=110)
