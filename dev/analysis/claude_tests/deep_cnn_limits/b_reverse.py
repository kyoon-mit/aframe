"""Theorem 2 check with a chirp played backwards: same power spectrum, so every
time-shift-invariant statistic has the same first- and second-order response.

1. The mean-pooled random ReLU CNN of b_cnn_mc.py ('avg ReLU valid', same seed and
   noise draws) on the whitened BNS reversed in time; compare with the forward
   row of out/b_cnn_mc.json.
2. The maximum over time of the full-window matched-filter energy, kernel matched
   to the forward chirp, on the forward and on the reversed chirp (4,000 draws).
Output: out/b_reverse.txt
"""
import os, sys, json
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np
sys.path.insert(0, ".")
from scipy.signal import hilbert

lines = []
def P(s=""):
    print(s, flush=True); lines.append(s)

N = 8192
s = np.load("out/sig_whitened_bns.npy")
srev = s[::-1].copy()
Cf = np.fft.ifft(np.abs(np.fft.fft(s)) ** 2).real
Cr = np.fft.ifft(np.abs(np.fft.fft(srev)) ** 2).real
P(f"max |C_forward(d) - C_reversed(d)| = {np.abs(Cf - Cr).max():.2e} (C(0) = 1)")

# 2. max-pooled full-window matched-filter energy, kernel = forward chirp
z = hilbert(s)
Zc = np.conj(np.fft.fft(z))
rng = np.random.default_rng(3)
rhos = [0.0, 2.0, 4.0, 8.0]
res = {k: {r: [] for r in rhos} for k in ("forward", "reversed")}
for _ in range(20):
    n = rng.standard_normal((200, N))
    for r in rhos:
        for k, sig in (("forward", s), ("reversed", srev)):
            E = np.abs(np.fft.ifft(np.fft.fft(n + r * sig, axis=1) * Zc, axis=1)) ** 2
            res[k][r].append(E.max(1))
m0 = np.concatenate(res["forward"][0.0]); sd = m0.std(ddof=1)
P("maximum over time of the full-window matched-filter energy (kernel matched to the forward chirp), 4,000 draws:")
for r in rhos[1:]:
    a = np.concatenate(res["forward"][r]); b = np.concatenate(res["reversed"][r])
    da = a - m0; db = b - m0
    P(f"  rho = {r:3.0f}: SNR forward {da.mean()/sd:.3f} +- {da.std(ddof=1)/np.sqrt(len(da))/sd:.3f}, "
      f"reversed {db.mean()/sd:.3f} +- {db.std(ddof=1)/np.sqrt(len(db))/sd:.3f}")

# 1. the random CNN
import torch
torch.set_num_threads(int(os.environ.get("NTHREADS", "6")))
from b_cnn_mc import LocalNet, run, DT, AMPS
net = LocalNet(seed=1, act="relu", pool="mean").to(DT)
r_rev, _ = run(net, srev, int(os.environ.get("DRAWS", "1200")), 50, rng_seed=7)
J = json.load(open("out/b_cnn_mc.json"))
fwd = J["avg ReLU valid | whitened BNS (IMRPhenomD)"]
P("mean-pooled random ReLU CNN (R = 49), same noise draws, in SNR units:")
P(f"  {'A':>5s} {'even, forward':>20s} {'even, reversed':>20s} {'odd, forward':>20s} {'odd, reversed':>20s}")
for a in AMPS:
    ef = fwd["even"][f"{a:.1f}"]; er = r_rev["even"][a]
    of = fwd["odd"][f"{a:.1f}"]; orr = r_rev["odd"][a]
    P(f"  {a:5.1f} {ef[0]:+.5f} +- {ef[1]:.5f} {er[0]:+.5f} +- {er[1]:.5f} {of[0]:+.5f} +- {of[1]:.5f} {orr[0]:+.5f} +- {orr[1]:.5f}")
open("out/b_reverse.txt", "w").write("\n".join(lines) + "\n")
