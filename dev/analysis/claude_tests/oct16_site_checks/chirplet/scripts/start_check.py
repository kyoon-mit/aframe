"""How good is a chirplet chain laid along a leading-order chirp, with no fit, as a
kernel for a real whitened template?

Chains: P Gaussian chirplets at equal steps of the linear-chirp count, width 0.4 of
the local spacing, amplitude tau^(-1/4) and phase of the leading-order chirp at
each centre (no solve), 20 Hz to 1024 Hz. Target: the four whitened kernels of the
resonator fits (detector-frame chirp masses 1.0, 1.2, 1.5, 2.0). Match: largest
|cross-correlation| over lags, unit norms, over a grid of chain chirp masses and
both signs of frequency. Control: the exact leading-order chirp itself.
"""
import os

for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[v] = "1"

import json
import sys
import time

import numpy as np

T0 = time.time()
TSUN = 1.3271244e20 / 299792458.0**3
FS = 2048.0
KER = np.load(sys.argv[1])
OUT = sys.argv[2]
L = KER["K_1.2"].size
LAG = np.arange(L)
TAU = (LAG + 0.5) / FS   # kernel lag l looks back tau = l / fs from the merger


def f_of_tau(tau, mc):
    return (5.0 / (256.0 * tau)) ** 0.375 * (TSUN * mc) ** -0.625 / np.pi


def tau_of_f(f, mc):
    return 5.0 / 256.0 * (TSUN * mc) ** (-5.0 / 3.0) * (np.pi * f) ** (-8.0 / 3.0)


def phase_of_tau(tau, mc):
    return -2.0 * (tau / (5.0 * TSUN * mc)) ** 0.625


def fdot(tau, mc):
    return 0.375 * f_of_tau(tau, mc) / tau


def fddot(tau, mc):
    return 0.515625 * f_of_tau(tau, mc) / tau**2


TW = L / FS


def chain_kernel(mc, P, beta=0.4, fmax=1024.0):
    """Chirp-like kernel in lag: conj of the template read backwards, as K = conj(h(t_c - l/fs))."""
    K0 = 4.8 * TW * (fddot(TW, mc) / 6.0) ** (1 / 3)
    p = 5 / 24
    n_end = K0 * (1 - (tau_of_f(fmax, mc) / TW) ** p)
    step = n_end / P
    t = -TAU
    h = np.zeros(L, complex)
    for k in range(P):
        tauc = TW * (1 - (k + 0.5) * step / K0) ** (1 / p)
        s = t + tauc
        sig = beta * step * (6.0 / fddot(tauc, mc)) ** (1 / 3)
        h += tauc ** -0.25 * np.exp(-0.5 * (s / sig) ** 2 + 1j * (phase_of_tau(tauc, mc)
                                    + 2 * np.pi * f_of_tau(tauc, mc) * s + np.pi * fdot(tauc, mc) * s**2))
    return np.conj(h)


def exact_kernel(mc, fmax=1024.0):
    fr = f_of_tau(TAU, mc)
    keep = (fr > 20.0) & (fr < fmax)
    return np.conj(keep * TAU ** -0.25 * np.exp(1j * phase_of_tau(TAU, mc)))


def best_corr(A, B):
    n = 2 * L
    c = np.fft.ifft(np.conj(np.fft.fft(A, n)) * np.fft.fft(B, n))
    return float(np.max(np.abs(c)) / (np.linalg.norm(A) * np.linalg.norm(B)))


rows = {}
for name in ("1.0", "1.2", "1.5", "2.0"):
    K = KER[f"K_{name}"]
    mc0 = float(name)
    grid = mc0 * np.exp(np.linspace(np.log(0.9), np.log(1.1), 401))
    out = {}
    for label, fn in (("exact", lambda m: exact_kernel(m)), ("chain12", lambda m: chain_kernel(m, 12)),
                      ("chain10", lambda m: chain_kernel(m, 10)), ("chain16", lambda m: chain_kernel(m, 16))):
        best = (0.0, None, None)
        for m in grid:
            k = fn(m)
            for sign, kk in (("+", k), ("-", np.conj(k))):
                v = best_corr(kk, K)
                if v > best[0]:
                    best = (v, float(m), sign)
        out[label] = {"match": best[0], "mc_best": best[1], "sign": best[2]}
    rows[name] = out
    print(f"kernel {name}: " + "  ".join(f"{k} {v['match']:.3f} at {v['mc_best']:.4f}{v['sign']}" for k, v in out.items()),
          flush=True)
json.dump(rows, open(OUT, "w"), indent=1)
print(f"done in {time.time() - T0:.1f} s")
