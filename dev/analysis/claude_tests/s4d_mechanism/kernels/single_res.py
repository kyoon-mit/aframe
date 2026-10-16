"""Best in-band match of one damped complex resonance to each whitened template, over a grid of
frequency and ring time, maximized over output time. The reference for "one spectrogram row".
usage: python single_res.py RATE   (1024 or 2048)

Match convention (in-band): max_t |k * x_ab| / (||k_b|| ||x_ab||), x_ab the analytic template kept
at 20 Hz..Nyquist, k_b the resonance's positive in-band part; the same convention as the channels'
in-band match in analyze.py.
"""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kcommon as kc  # noqa: E402

torch.set_num_threads(8)
rate = int(sys.argv[1])
fs = float(rate)
x, mc = kc.templates(rate=rate, ifo=0)
L = x.shape[-1]
fmax = 500.0 if rate == 1024 else 1000.0
f0 = np.geomspace(20.0, fmax, 100 if rate == 1024 else 110)
T = 0.002 * 2 ** (np.arange(37) / 3)                      # 2 ms to 8.1 s
l = np.arange(L)
xb2 = 2 * kc.band_norm2(x, fs, 20, fs / 2)                 # ||x_a,band||^2, band-limited analytic template
m1 = np.empty((len(mc), len(f0), len(T)))                 # (J, f0, T)
step = 10
for s in range(0, len(f0), step):
    FF, TT = np.meshgrid(f0[s:s + step], T, indexing="ij")
    k = np.exp(np.outer(-1 / (TT.ravel() * fs) + 2j * np.pi * FF.ravel() / fs, l))
    k /= np.linalg.norm(k, axis=-1, keepdims=True)
    E2, _ = kc.peak_energy_ratio(k, x, jchunk=4, cchunk=512, analytic=True, fband=(20, fs / 2), fs=fs, xnorm2=xb2)
    # in-band normalization: share of each resonance's energy at 20 Hz..Nyquist (positive side)
    Pk = np.abs(np.fft.fft(k, n=4 * L, axis=-1)) ** 2
    fk = np.fft.fftfreq(4 * L, 1 / fs)
    share = Pk[:, (fk >= 20) & (fk < fs / 2)].sum(1) / Pk.sum(1)
    m1[:, s:s + step] = np.sqrt(E2 / share[None]).reshape(len(mc), -1, len(T))
best_T_idx = m1.argmax(-1)
m1_best = m1.max(-1)                                       # (J, f0)
T_best = T[best_T_idx]
np.savez_compressed(os.path.join(kc.HERE, f"single_res_{rate}.npz"), f0=f0, T=T, chirp_mass=mc,
                    match=m1.astype(np.float32), match_best=m1_best.astype(np.float32), T_best=T_best)
# summary: the best ring time against the in-step time, inside the chirp's in-window band
out = {"rate": rate, "f0_range": [float(f0[0]), float(f0[-1])], "T_range_s": [float(T[0]), float(T[-1])]}
rows = []
for j in (0, int(np.argmin(np.abs(mc - 1.2))), len(mc) - 1):
    fstart = kc.f_of_tau(kc.MERGE / 2048, mc[j])
    for fq in (60, 100, 150, 200, 300, 400):
        if fq < fstart or fq > fmax:
            continue
        i = int(np.argmin(np.abs(np.log(f0 / fq))))
        rows.append(dict(chirp_mass=round(float(mc[j]), 3), f0=round(float(f0[i]), 1),
                         best_match=round(float(m1_best[j, i]), 4), best_ring_s=round(float(T_best[j, i]), 4),
                         in_step_time_s=round(float(kc.t_step(f0[i], mc[j])), 4),
                         time_before_merger_s=round(float(kc.tau_of_f(f0[i], mc[j])), 4)))
out["examples"] = rows
inwin = np.array([[f0[i] >= kc.f_of_tau(kc.MERGE / 2048, mc[j]) for i in range(len(f0))] for j in range(len(mc))])
ratio = T_best / kc.t_step(f0[None, :], mc[:, None])
out["best_ring_over_in_step_time_in_window_band_quartiles"] = [float(q) for q in np.percentile(ratio[inwin], [25, 50, 75])]
out["max_single_resonance_match_over_all"] = float(m1_best.max())
out["single_resonance_match_quartiles_in_window_band"] = [float(q) for q in np.percentile(m1_best[inwin], [25, 50, 75])]
with open(os.path.join(kc.HERE, f"single_res_{rate}.json"), "w") as fh:
    json.dump(out, fh, indent=1)
print(json.dumps(out, indent=1))
