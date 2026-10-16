import math

import numpy as np
import torch

torch.set_num_threads(8)
from train.experimental.template_bank.arch import TemplateBankNet

BANK = "/home/kyoon/SSM-BNS/aframe/dev/runs/claude_tests/template_bank/bank_mc0.5pct_spin3_q0.8.npz"
net = TemplateBankNet(BANK)
bank = np.load(BANK)
J = 300
# whitened template, as the filter sees it (H and L whitening), at full rate
tmpl = torch.tensor(bank["templates"][J], dtype=torch.float64)
spec = torch.fft.rfft(tmpl, n=net.n_full)
# rebuild the per-detector whitening used for the filters
f_ = net.f
filt = net.filters[:, J].to(torch.complex128)  # (I, F) = conj(W) / sigma * analytic / d
# recover W_i (whitened spectrum, unnormalized) direction from the filter: conj(filter)
w = filt.conj()  # proportional to W * analytic / (sigma d)
analytic = torch.full((f_,), 2.0, dtype=torch.float64); analytic[0] = analytic[-1] = 1.0
W = w / analytic * 2 * 1.0  # proportional to W / sigma (times d=2 cancels below by renorm)
full = torch.zeros(2, net.n_full // 2 + 1, dtype=torch.complex128)
full[:, :f_] = W
x = torch.fft.irfft(full, n=net.n_full)[:, :8192]  # band-limited whitened template per detector
x = x / x.norm(dim=-1, keepdim=True) * 12.0
for shift_h, shift_l in [(0, 0), (300, 314), (-400, -386), (300, 286), (100, 100)]:
    d = torch.stack([torch.roll(x[0], shift_h), torch.roll(x[1], shift_l)])
    phase_rot = torch.fft.irfft(torch.fft.rfft(d[1]) * complex(math.cos(1.0), math.sin(1.0)), n=8192)
    d = torch.stack([d[0], phase_rot])[None].float()
    new, plain, peak, feats = net.newsnr2(d, features=True)
    f = dict(zip(TemplateBankNet.FEATURES, feats[0].tolist()))
    print(
        f"shiftH {shift_h:+5d} shiftL {shift_l:+5d}: plain {plain.item():.2f} (expect ~288) new {new.item():.2f} "
        f"redchi {f['chi2_first']:.4f} {f['chi2_second']:.4f} lag_off {f['lag_offset']*net.coincidence:+.1f} samples "
        f"(expect {(shift_l-shift_h)/2:+.1f}) cos {f['cos_phase_offset']:.3f} sin {f['sin_phase_offset']:.3f} best j {peak.argmax().item()}"
    )

# parabola check
g = net.grid_log_mc
true = g[100] + 0.3 * (g[101] - g[100])
logits = -((g - true) ** 2) / (2 * 0.01**2)
lm, spread = net.estimate(logits[None])
print("parabola vertex", lm.item(), "true", true.item(), "spread", spread.item())
