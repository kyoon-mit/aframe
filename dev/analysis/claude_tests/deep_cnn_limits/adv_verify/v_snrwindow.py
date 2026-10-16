"""In-window SNR against the SNR aframe rescales to (the sliced waveform, 5 s
before merger to 1.5 s after; kernel 4 s + fduration 1 s, left_pad 3.5 s,
right_pad 0), leading-order SPA, O3 average PSD. Ratio sqrt(rho^2(window)/rho^2(5 s))
for merger 3.5, 3.75 and 4.0 s after the window start."""
import numpy as np
TSUN = 4.925490947641267e-06
p = np.load("/home/kyoon/SSM-BNS/aframe/dev/runs/claude_tests/template_bank/bank_mc0.5pct_spin0_q0.8.npz")["psd"]
FP = np.arange(p.shape[1]) * 0.5
SN = 2 / (1 / p[0] + 1 / p[1])
fT = lambda T, mc: (5 / (256 * T)) ** 0.375 * (mc * TSUN) ** -0.625 / np.pi
for mc in (0.85, 1.2, 1.8, 2.75):
    fhi = min(1 / (6 ** 1.5 * np.pi * TSUN * 2 ** 1.2 * mc), 1024)
    f = np.linspace(20, fhi, 400001)
    W = f ** (-7 / 3) / np.interp(f, FP, SN)
    tot = W[f >= fT(4.5, mc)].sum()
    print(f"Mc {mc}: in-window/4.5 s SNR ratio, merger at 3.5/3.75/4.0 s: "
          + " ".join(f"{np.sqrt(W[f >= fT(T, mc)].sum() / tot):.3f}" for T in (3.5, 3.75, 4.0)))
