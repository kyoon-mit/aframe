"""Section 5 checks: what one decaying resonator can take from a chirp.

Prediction (stationary phase, amplitude and delay slowly varying):
  M1(a) = |<r,K>|^2 / (|r|^2 |K|^2) = e(f_n) * 2a exp(-2 a tau) / (1 - exp(-2 a T_w)),
with e(f) the kernel's fraction of energy per Hz near f_n, tau = tau(f_n).
For tau << T_w it peaks at a = 1/(2 tau) with M1 = e(f_n) / (e tau).
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

from common import FS, L0, RES, kernel_of, tau_of_f, toy_chirp

TW = L0 / FS
out = {}
rows = []
for mc in (1.2, 2.75):
    K = kernel_of(toy_chirp(mc))
    Kf = np.fft.fft(K)
    fk = np.fft.fftfreq(L0, 1 / FS)
    efrac = np.abs(Kf) ** 2 / L0 / (FS / L0)        # fraction of energy per Hz
    l = np.arange(L0)
    for fn in (100.0, 150.0, 200.0, 300.0, 450.0):
        tau = float(tau_of_f(fn, mc))
        if tau > TW:
            continue
        sel = np.abs(fk - fn) <= 2.0
        e_n = float(np.mean(efrac[sel]))
        agrid = np.exp(np.linspace(np.log(0.02 / tau), np.log(50 / tau), 400))
        M = []
        for a in agrid:
            r = np.exp((-a + 2j * np.pi * fn) * l / FS)
            M.append(np.abs(np.vdot(r, K)) ** 2 / np.vdot(r, r).real)
        M = np.array(M)
        pred = e_n * 2 * agrid * np.exp(-2 * agrid * tau) / (-np.expm1(-2 * agrid * TW))
        i = int(np.argmax(M))
        ip = int(np.argmax(pred))
        rows.append({"mc": mc, "f_n": fn, "tau_s": tau,
                     "a_best_times_tau": float(agrid[i] * tau),
                     "a_best_pred_times_tau": float(agrid[ip] * tau),
                     "M1_best": float(M[i]), "M1_best_pred": float(pred[ip]),
                     "M1_best_over_pred": float(M[i] / pred[ip]),
                     "captured_band_Hz": float(M[i] / e_n),
                     "captured_band_times_tau": float(M[i] / e_n * tau),
                     "one_over_e": float(1 / np.e),
                     "amp_kept_at_merger_exp_minus_a_tau": float(np.exp(-agrid[i] * tau)),
                     "curve_max_rel_dev_where_pred_gt_10pct_of_peak":
                         float(np.max(np.abs(M / pred - 1)[pred > 0.1 * pred.max()]))})
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rows[-1].items()}))
out["single_resonator"] = rows
json.dump(out, open(os.path.join(RES, "c4_decay.json"), "w"), indent=1)
print("saved")
