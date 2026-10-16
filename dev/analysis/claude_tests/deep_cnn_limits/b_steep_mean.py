"""A steep function of the full-window matched-filter energy, averaged over time,
against the maximum over time and the plain mean (whitened BNS, 4,000 draws,
same noise for every rho). Detection rate at a 1% false-alarm rate.
Statistics: max_t E(t); mean_t E(t); mean_t 1{E(t) > 25}; mean_t exp(E(t)/2).
Output: out/b_steep_mean.txt
"""
import numpy as np
from scipy.signal import hilbert

N = 8192
s = np.load("out/sig_whitened_bns.npy")
z = hilbert(s)
Zc = np.conj(np.fft.fft(z))
rng = np.random.default_rng(3)
rhos = [0.0, 2.0, 4.0, 6.0, 8.0]
names = ["max E", "mean E", "mean 1{E > 25}", "mean exp(E/2)"]
vals = {r: {k: [] for k in names} for r in rhos}
for _ in range(20):
    n = rng.standard_normal((200, N))
    for r in rhos:
        E = np.abs(np.fft.ifft(np.fft.fft(n + r * s, axis=1) * Zc, axis=1)) ** 2
        vals[r]["max E"].append(E.max(1))
        vals[r]["mean E"].append(E.mean(1))
        vals[r]["mean 1{E > 25}"].append((E > 25).mean(1))
        vals[r]["mean exp(E/2)"].append(np.exp(E / 2).mean(1))
lines = ["detection rate at 1% false alarms (4,000 noise draws), whitened BNS, full-window matched-filter energy E(t)"]
lines.append("rho   " + "  ".join(f"{k:>16s}" for k in names))
thr = {k: np.quantile(np.concatenate(vals[0.0][k]), 0.99) for k in names}
for r in rhos[1:]:
    row = []
    for k in names:
        v = np.concatenate(vals[r][k])
        # ties at the threshold (sparse statistics): count strictly above, plus half of the ties
        above = np.mean(v > thr[k]) + 0.5 * np.mean(v == thr[k])
        row.append(f"{above:16.3f}")
    lines.append(f"{r:4.1f}  " + "  ".join(row))
z0 = np.concatenate(vals[0.0]["mean 1{E > 25}"])
lines.append(f"noise only: fraction of draws with any E(t) > 25: {np.mean(z0 > 0):.4f}")
print("\n".join(lines))
open("out/b_steep_mean.txt", "w").write("\n".join(lines) + "\n")
