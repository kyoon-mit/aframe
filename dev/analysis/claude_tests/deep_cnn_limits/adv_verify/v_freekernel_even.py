"""Direct Monte Carlo of the even part of each channel's peak energy (max over
time of |k_c * x|^2, the network's own causal convolution, one detector) for the
whitened BNS chirp and the same chirp played backwards (identical power
spectrum), common noise, A = 1, 2, 4. Under Theorem 2 the A^2 part is equal."""
import numpy as np
import torch
exec(open("v_freekernel.py").read().split("s = np.load(SIG)")[0])
s = np.load(SIG)
sf = resample(s[None])[0]
sb = resample(s[::-1].copy()[None])[0]
rng = np.random.default_rng(7)
f2 = np.fft.rfftfreq(8192, 1 / 2048)
acc = {A: {"f": [], "b": []} for A in (1.0, 2.0, 4.0)}
for i in range(50):
    n = rng.standard_normal((8, 8192))
    Nf = np.fft.rfft(n)
    Nf[:, f2 < 20] = 0
    x = resample(np.fft.irfft(Nf, n=8192))
    P0 = (conv(x).abs() ** 2).amax(-1)
    for A in acc:
        for key, sig in (("f", sf), ("b", sb)):
            Pp = (conv(x + A * sig).abs() ** 2).amax(-1)
            Pm = (conv(x - A * sig).abs() ** 2).amax(-1)
            acc[A][key].append(((Pp + Pm) / 2 - P0).numpy())
yf = (conv(sf[None])[0].abs() ** 2).mean(-1).numpy()
top = np.argsort(yf)[-20:]
for A in acc:
    ef = np.concatenate(acc[A]["f"]); eb = np.concatenate(acc[A]["b"])
    mf, mb = ef.mean(0), eb.mean(0)
    se = np.sqrt(ef.var(0) / len(ef) + eb.var(0) / len(eb))
    print(f"A={A}: even part summed over channels fwd {mf.sum():.3f} bwd {mb.sum():.3f} (ratio {mf.sum() / mb.sum():.2f});"
          f" top-20 channels median ratio {np.median(mf[top] / mb[top]):.2f}; channels with |fwd-bwd| > 3 se: {(np.abs(mf - mb) > 3 * se).sum()}/256")
