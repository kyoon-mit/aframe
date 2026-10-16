"""Toy model: maximum and mean over 1,000 time steps of a sum of 64 channels, with
channel dropout p = 0.2 (each channel kept with probability 0.8 and scaled by 1.25,
the same for all time steps). Also the expected maximum of n Gaussian values
against sqrt(2 ln n). Writes toy_mc.json next to this script."""
import json
import os

import numpy as np
from math import comb

HERE = os.path.dirname(os.path.abspath(__file__))
C, T, P = 64, 1000, 0.2
N_PAT, BATCH = 100_000, 10_000
rng = np.random.default_rng(20261015)
out = {}


def dropout_stats(G, w, p, n_pat, rng):
    """Mean over dropout patterns of max_t and mean_t of sum_c xi_c w_c G[c, t]."""
    y0 = w @ G
    mx, mn, mx2, mn2 = 0.0, 0.0, 0.0, 0.0
    for _ in range(n_pat // BATCH):
        xi = (rng.random((BATCH, len(w))) >= p) / (1.0 - p)
        y = (xi * w) @ G
        m = y.max(axis=1)
        mx += m.sum()
        mx2 += (m ** 2).sum()
        a = y.mean(axis=1)
        mn += a.sum()
        mn2 += (a ** 2).sum()
    mx, mn, mx2, mn2 = mx / n_pat, mn / n_pat, mx2 / n_pat, mn2 / n_pat
    return {"max_off": float(y0.max()), "max_on_mean": float(mx),
            "max_on_sem": float(np.sqrt((mx2 - mx ** 2) / n_pat)),
            "mean_off": float(y0.mean()), "mean_on_mean": float(mn),
            "mean_on_sem": float(np.sqrt(max(mn2 - mn ** 2, 0.0) / n_pat))}


# 1. one fixed draw of 64 independent standard Gaussian channels, weights 1
G = rng.standard_normal((C, T))
w = np.ones(C)
r = dropout_stats(G, w, P, N_PAT, rng)
r["ratio"] = r["max_on_mean"] / r["max_off"]
out["independent_one_draw"] = r

# 2. 200 draws of the channels (10,000 patterns each): distribution of the ratio
ratios, mean_diffs, ons, offs, zs = [], [], [], [], []
for _ in range(200):
    G2 = rng.standard_normal((C, T))
    s = dropout_stats(G2, w, P, 10_000, rng)
    ratios.append(s["max_on_mean"] / s["max_off"])
    mean_diffs.append(s["mean_on_mean"] - s["mean_off"])
    ons.append(s["max_on_mean"]); offs.append(s["max_off"])
    zs.append((s["mean_on_mean"] - s["mean_off"]) / s["mean_on_sem"])
ratios = np.array(ratios)
out["independent_200_draws"] = {
    "ratio_mean": float(ratios.mean()), "ratio_sd": float(ratios.std(ddof=1)),
    "ratio_min": float(ratios.min()), "ratio_max": float(ratios.max()),
    "frac_ratio_above_1": float(np.mean(ratios > 1)),
    "ratio_of_averages": float(np.sum(ons) / np.sum(offs)),
    "max_off_avg": float(np.mean(offs)), "max_on_avg": float(np.mean(ons)),
    "mean_shift_abs_max": float(np.max(np.abs(mean_diffs))),
    "mean_shift_z_abs_max": float(np.max(np.abs(zs))), "mean_shift_z_sd": float(np.std(zs, ddof=1)),
    "mean_sem_typical": float(np.median(np.abs(np.array(mean_diffs) / np.array(zs))))}

# prediction: spread grows by 1/sqrt(1-p); with K kept channels, 1.25 sqrt(K) / 8
k = np.arange(C + 1)
pk = np.array([comb(C, int(i)) * (1 - P) ** i * P ** (C - i) for i in k])
out["prediction"] = {"inv_sqrt_1mp": float(1 / np.sqrt(1 - P)),
                     "E_sqrtK_scaled": float((pk * np.sqrt(k)).sum() / (1 - P) / np.sqrt(C))}

# 3. channels that share a common peaked part: g_c(t) = a s(t) + n_c(t); 50 draws each
t = np.arange(T)
s = np.exp(-0.5 * ((t - 700) / 20.0) ** 2)
shared = {}
for a in (0.0, 0.5, 1.0, 3.0):
    on, off, rr = [], [], []
    for _ in range(50):
        Gs = a * s[None, :] + rng.standard_normal((C, T))
        q = dropout_stats(Gs, w, P, 10_000, rng)
        on.append(q["max_on_mean"]); off.append(q["max_off"]); rr.append(q["max_on_mean"] / q["max_off"])
    shared[str(a)] = {"ratio_of_averages": float(np.sum(on) / np.sum(off)), "mean_ratio": float(np.mean(rr)),
                      "ratio_sd": float(np.std(rr, ddof=1)), "max_off_avg": float(np.mean(off))}
out["shared_part"] = shared

# 4. other dropout rates, independent channels, 50 draws each
out["rates"] = {}
for p in (0.1, 0.2, 0.3, 0.5):
    on, off = [], []
    for _ in range(50):
        Gp = rng.standard_normal((C, T))
        q = dropout_stats(Gp, w, p, 10_000, rng)
        on.append(q["max_on_mean"]); off.append(q["max_off"])
    kk = np.arange(C + 1)
    pk_ = np.array([comb(C, int(i)) * (1 - p) ** i * p ** (C - i) for i in kk])
    out["rates"][str(p)] = {"ratio_of_averages": float(np.sum(on) / np.sum(off)),
                            "inv_sqrt_1mp": float(1 / np.sqrt(1 - p)),
                            "E_sqrtK_scaled": float((pk_ * np.sqrt(kk)).sum() / (1 - p) / np.sqrt(C))}

# 5. expected maximum of n standard Gaussian values against sqrt(2 ln n)
gm = {}
for n, reps in ((10, 200_000), (100, 100_000), (1000, 20_000), (8192, 4_000), (100_000, 400)):
    m = np.empty(reps)
    for i in range(0, reps, 100):
        m[i:i + 100] = rng.standard_normal((min(100, reps - i), n)).max(axis=1)
    gm[str(n)] = {"E_max": float(m.mean()), "sem": float(m.std(ddof=1) / np.sqrt(reps)),
                  "sqrt_2lnn": float(np.sqrt(2 * np.log(n))),
                  "ratio": float(m.mean() / np.sqrt(2 * np.log(n)))}
out["gaussian_max"] = gm

json.dump(out, open(f"{HERE}/toy_mc.json", "w"), indent=1)
print(json.dumps(out, indent=1))
