"""Block diagram of the free-kernel network (signal-only regressor at 2048 Hz)."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

INK, MUTED, NAVY, GREEN = "#0F1E33", "#5b6574", "#13386b", "#1baf7a"
FILL, LEARN = "#eef2f7", "#e3f5ee"
W, H = 2.55, 1.55
ROWS = [
    [("Input", "whitened strain\nH1 and L1, 4 s", "2 × 8192", False),
     ("Long convolution", "256 learned complex\nkernels, 4 s each,\nshared by both detectors", "2 × 256 × 8192", True),
     ("Energy", "2 |y|²\n(phase-blind)", "2 × 256 × 8192", False),
     ("Detector sum", "H1 + largest L1\nwithin ±10 ms", "256 × 8192", False)],
    [("Max over time", "peak energy\nof each kernel", "256", False),
     ("Log, standardize", "log(1 + e), then zero mean,\nunit spread across\nthe 256 kernels", "256", False),
     ("Linear head", "256 → 2", "2", True),
     ("Output", "chirp-mass mean μ\nvariance σ² (softplus)", "μ, σ²", False)],
]
X0, DX, Y = 0.3, 3.35, (3.2, 0.3)

fig, ax = plt.subplots(figsize=(13.6, 5.6))
ax.set_xlim(0, 13.6); ax.set_ylim(-0.75, 5.35); ax.axis("off")
centers = []
for r, row in enumerate(ROWS):
    for c, (title, body, shape, learned) in enumerate(row):
        x, y = X0 + c * DX, Y[r]
        ax.add_patch(FancyBboxPatch((x, y), W, H, boxstyle="round,pad=0.02,rounding_size=0.08",
                                    fc=LEARN if learned else FILL, ec=GREEN if learned else NAVY, lw=1.6))
        ax.text(x + W / 2, y + H - 0.22, title, ha="center", va="top", fontsize=11.5, color=INK, weight="bold")
        ax.text(x + W / 2, y + H - 0.55, body, ha="center", va="top", fontsize=9.2, color=INK, linespacing=1.25)
        ax.text(x + W / 2, y - 0.1, shape, ha="center", va="top", fontsize=9.2, color=MUTED, family="monospace")
        centers.append((x, y))
arrow = dict(arrowstyle="-|>", mutation_scale=14, color=NAVY, lw=1.5)
for r in range(2):
    for c in range(3):
        x, y = X0 + c * DX, Y[r]
        ax.add_patch(FancyArrowPatch((x + W + 0.04, y + H / 2), (x + DX - 0.04, y + H / 2), **arrow))
# row 1 end to row 2 start
x4, y4 = X0 + 3 * DX, Y[0]
ax.add_patch(FancyArrowPatch((x4 + W / 2, y4 - 0.42), (X0 + W / 2, Y[1] + H + 0.04),
                             connectionstyle="angle,angleA=180,angleB=90,rad=0.15", **arrow))
# a sketch of one learned kernel inside the convolution box
x, y = X0 + DX, Y[0]
t = np.linspace(0, 1, 600)
k = np.sin(2 * np.pi * 9 * (t + 1.6 * t**4)) * (0.25 + 0.75 * t**2)
ax.plot(x + 0.25 + 2.05 * t, y + 0.22 + 0.13 * k, color=GREEN, lw=0.8)
ax.text(13.3, 5.2, "green: learned parameters", ha="right", va="top", fontsize=9, color=GREEN)
ax.text(13.3, 4.95, "grey: tensor shape (per window)", ha="right", va="top", fontsize=9, color=MUTED)
fig.tight_layout()
fig.savefig(Path(__file__).with_name("free_kernel_diagram.png"), dpi=110)
