"""Fraction within 1% averaged over the validation SNR population
(density proportional to SNR^-3 on [4, 100]; derivations #late-snr-bands),
from mc_ml.py runs at fixed SNR for one chirp mass, by linear interpolation
in SNR (constant beyond the largest SNR run). Errors: binomial at each SNR,
propagated through the interpolation weights.

usage: popavg.py MC0 run1.json run2.json ...
"""
import json
import sys

import numpy as np

L1 = np.log(1.01)
mc0 = float(sys.argv[1])
runs = [json.load(open(p)) for p in sys.argv[2:]]
runs = sorted([r for r in runs if abs(r["mc0"] - mc0) < 1e-9], key=lambda r: r["snr"])
snr = np.array([r["snr"] for r in runs])
models = list(runs[0]["err"].keys())
rho = np.linspace(4, 100, 200001)
p = rho ** -3.0
p /= np.trapz(p, rho)
print(f"Mc {mc0}: SNR points {snr.tolist()}")
print("SNR    " + " ".join(f"{m:>11s}" for m in models))
for r in runs:
    print(f"{r['snr']:<6} " + " ".join(f"{np.mean(np.abs(np.array(r['err'][m])) < L1):11.3f}" for m in models))
res = {}
for m in models:
    fr = np.array([np.mean(np.abs(np.array(r["err"][m])) < L1) for r in runs])
    n = np.array([len(r["err"][m]) for r in runs])
    se = np.sqrt(fr * (1 - fr) / n)
    # interpolation weights: avg = sum_i c_i fr_i
    c = np.zeros(len(runs))
    for i in range(len(runs)):
        e = np.zeros(len(runs))
        e[i] = 1
        c[i] = np.trapz(np.interp(rho, snr, e) * p, rho)
    avg = float(c @ fr)
    res[m] = (avg, float(np.sqrt(((c * se) ** 2).sum())))
print("population (SNR^-3 on [4, 100]) fraction within 1%: "
      + ", ".join(f"{m} {a:.3f} +- {s:.3f}" for m, (a, s) in res.items()))
below = np.trapz(p[rho < snr.min()], rho[rho < snr.min()])
print(f"weight below the smallest SNR run: {below:.3f}; weight above the largest: "
      f"{np.trapz(p[rho > snr.max()], rho[rho > snr.max()]):.3f}")
