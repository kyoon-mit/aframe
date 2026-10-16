"""Architecture figures for the chirp-kernel network."""

import sys
from pathlib import Path

import h5py
import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from train.experimental.chirp_kernel.arch import ChirpKernelNet

ckpt, OUT = sys.argv[1], Path(sys.argv[2])
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
B = ("/home/kyoon/SSM-BNS/aframe/dev/runs/claude_tests/den_reg_sched_alpha_0.5_floor_1e-2_"
     "c_rho_5_lambda_amp_3_powerlaw_m2_snr_d128_n6_mc_q_rep4_smooth20_wp0.96_ddp/mf_bench_ep630")

net = ChirpKernelNet().eval()
state = torch.load(ckpt, map_location="cpu", weights_only=False)["state_dict"]
net.load_state_dict({k[len("model."):]: v for k, v in state.items() if k.startswith("model.")})

# 1. block diagram
fig, ax = plt.subplots(figsize=(11, 4.6))
ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(-0.12, 1)
blocks = [
    (0.01, 0.62, "whitened strain\nH1, L1\n(B, 2, 8192)", False),
    (0.21, 0.62, "convolve with\n512 chirp kernels\ncos and sin\n(B, 2, 512, 8192) ×2", True),
    (0.43, 0.62, "energy\nz_c² + z_s²\nsum over H1, L1\n(B, 512, 8192)", False),
    (0.64, 0.62, "max over time\n(B, 512)", False),
    (0.83, 0.62, "log, standardize\nacross kernels\n(B, 512)", False),
    (0.83, 0.12, "logits\ns·z + W z + b\n(B, 512)", True),
    (0.60, 0.12, "softmax\nprobability over\nkernel chirp masses", False),
    (0.37, 0.12, "chirp mass of the\nmost probable kernel", False),
]
w, h = 0.16, 0.30
for x, y, text, learned in blocks:
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01,rounding_size=0.02",
                                fc=WASH if not learned else "#d6efe6", ec=C[0] if not learned else C[2], lw=1.3))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8.5, color=INK)
def arrow(a, b):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=12, color=C[0]))
for i in range(4):
    x0 = blocks[i][0] + w; x1 = blocks[i + 1][0]
    arrow((x0 + 0.003, 0.77), (x1 - 0.003, 0.77))
arrow((0.91, 0.62), (0.91, 0.42))
arrow((0.83 - 0.003, 0.27), (0.60 + w + 0.003, 0.27))
arrow((0.60 - 0.003, 0.27), (0.37 + w + 0.003, 0.27))
ax.add_patch(FancyBboxPatch((0.02, -0.02), 0.025, 0.05, boxstyle="round,pad=0.005", fc="#d6efe6", ec=C[2]))
ax.text(0.055, 0.005, "learned: the 512 kernel chirp masses; the scale s and the correction W, b", fontsize=8.5, color=MUTED, va="center")
ax.text(0.055, -0.08, "no mean pooling anywhere: every step after the convolution is an energy or a maximum", fontsize=8.5, color=MUTED, va="center")
fig.savefig(OUT / "ck_architecture.png", dpi=110, bbox_inches="tight")

# 2. a few kernels (cosine quadrature), last 1 s before merger
kc, _ = net.kernels(8192)
kmc = net.kernels.log_mc.exp().detach().numpy()
fig, axes = plt.subplots(3, 1, figsize=(9, 5.4), sharex=True)
lags = np.arange(2048) / 2048
for ax, j, c in zip(axes, (60, 256, 450), C):
    ax.plot(-lags, kc[j, :2048].detach().numpy(), color=c, lw=0.7)
    ax.set_title(f"kernel {j}: chirp mass {kmc[j]:.3f} M☉", fontsize=9)
    ax.set_yticks([])
axes[-1].set_xlabel("time before merger (s)   (the kernel is read backwards from its merger at lag 0)")
fig.tight_layout()
fig.savefig(OUT / "ck_kernels.png", dpi=110)

# 3. energy map of one real event
w_ = h5py.File(f"{B}/snr8_powerlaw/windows.h5")
mc = w_["params/chirp_mass"][:]; snr = w_["params/snr"][:]
i = int(np.where((snr > 10) & (snr < 11) & (mc > 1.3) & (mc < 1.6))[0][0])
x = torch.tensor(w_["X"][i:i + 1])
with torch.no_grad():
    kc, ks = net.kernels(8192)
    xf = torch.fft.rfft(x, n=16384)[:, :, None]
    zc = torch.fft.irfft(xf * torch.fft.rfft(kc, n=16384), n=16384)[..., :8192]
    zs = torch.fft.irfft(xf * torch.fft.rfft(ks, n=16384), n=16384)[..., :8192]
    e = (zc ** 2 + zs ** 2).sum(1)[0].numpy()
    p = torch.softmax(net(x), -1)[0].numpy()
t = np.arange(8192) / 2048
fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), gridspec_kw={"width_ratios": [2.2, 1, 1]})
sl = slice(int(3.3 * 2048), 8192)
im = axes[0].pcolormesh(t[sl], kmc, e[:, sl], cmap="Blues", shading="auto", vmax=np.percentile(e, 99.99))
axes[0].axhline(mc[i], color=C[2], lw=1, ls="--")
axes[0].set_xlabel("time in window (s)"); axes[0].set_ylabel("kernel chirp mass (M☉)")
axes[0].set_title(f"energy e_j(t), one event at SNR {snr[i]:.1f} (true chirp mass dashed)")
peak = e.max(-1)
axes[1].plot(peak, kmc, color=C[0]); axes[1].axhline(mc[i], color=C[2], lw=1, ls="--")
axes[1].set_xlabel("max over time"); axes[1].set_title("each kernel's peak energy"); axes[1].set_yticklabels([])
axes[2].plot(p, kmc, color=C[1]); axes[2].axhline(mc[i], color=C[2], lw=1, ls="--")
axes[2].set_xlabel("probability"); axes[2].set_title("softmax over kernels"); axes[2].set_yticklabels([])
for a in axes[1:]:
    a.set_ylim(axes[0].get_ylim())
fig.tight_layout()
fig.savefig(OUT / "ck_energy_map.png", dpi=110)
print("event", i, "snr", snr[i], "true", mc[i], "estimate", kmc[p.argmax()])
