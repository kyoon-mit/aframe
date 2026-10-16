import h5py
import numpy as np
import torch

torch.set_num_threads(8)
from train.experimental.template_bank.arch import TemplateBankNet

BANK = "/home/kyoon/SSM-BNS/aframe/dev/runs/claude_tests/template_bank/bank_mc0.5pct_spin3_q0.8.npz"
W = "/home/kyoon/SSM-BNS/aframe/dev/runs/claude_tests/s4d_energy_diag/train_windows_snr8_powerlaw.h5"
net = TemplateBankNet(BANK)
bank = np.load(BANK)
f = h5py.File(W, "r")
mc = f["params/chirp_mass"][:400]
sig = np.where(np.isfinite(mc))[0][:24]
xc = torch.tensor(f["X_clean"][sorted(sig)])
mc = mc[sorted(sig)]
snr = f["params/snr"][:400][sorted(sig)]
# band-limited norm^2 of the clean signal (<=512 Hz), per detector
spec = torch.fft.rfft(xc.double(), dim=-1)
freqs = torch.fft.rfftfreq(8192, 1 / 2048)
spec[..., freqs > 512] = 0
xb = torch.fft.irfft(spec, n=8192)
bl = (xb**2).sum(-1)  # (B, 2)
new, plain, peak, feats = net.newsnr2(xc, features=True)
rho2 = net.snr2_series(xc)
tmax = rho2.amax(2).argmax(-1)  # (B, I) lag index of max over templates
jbest = peak.argmax(-1)
merge = np.abs(xc.numpy()).argmax(-1)
for b in range(len(sig)):
    lag = tmax[b, 0].item() - net.max_lag
    print(
        f"snr {snr[b]:6.2f} bl {bl[b].sum():8.2f} plain {plain[b]:8.2f} ratio {plain[b]/bl[b].sum():.4f} "
        f"rhoH2/blH {feats[b,0]/bl[b,0]:.3f} rhoL2/blL {feats[b,1]/bl[b,1]:.3f} "
        f"mc {mc[b]:.4f} tmpl {np.exp(bank['grid_log_mc'][bank['grid_index'][jbest[b]]]):.4f} "
        f"lag {lag:+4d} ({lag*2/2048:+.4f}s) |X| peak at {merge[b,0]} -> {(merge[b,0]-7680)/2048:+.4f}s "
        f"redchi {feats[b,2]:.2f} {feats[b,3]:.2f} new/plain {new[b]/plain[b]:.3f}"
    )
