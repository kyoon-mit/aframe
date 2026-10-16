"""Side checks quoted in kernels.md: (1) group delay of one resonance at its centre against its ring
time; (2) 2 pi * integral of tau(f) df against the number of cycles in the window; (3) share of in-band
resonances with ring time within a factor 2 of tau(f; 1.2), at the start and trained; (4) L1 against H1
matches, best match at fixed chirp masses, band-start rings shortened below 1 s."""
import json
import os
import sys

import numpy as np
from scipy.integrate import quad

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kcommon as kc  # noqa: E402

K = kc.HERE
out = {}
fs, L = 1024.0, 4096
rows = []
for T, f0 in ((0.05, 100.0), (0.3, 150.0), (1.0, 60.0)):
    z = np.exp(-1 / (T * fs) + 2j * np.pi * f0 / fs)
    fr, H, gd = kc.transfer((z ** np.arange(L))[None], fs, 16 * L)
    i = np.argmin(np.abs(fr - f0))
    rows.append({"ring_s": T, "f0_Hz": f0, "group_delay_at_f0_s": round(float(gd[0, i]), 5),
                 "infinite_filter_|z|/(1-|z|)/fs_s": round(float(abs(z) / (1 - abs(z)) / fs), 5)})
out["one_resonance_group_delay"] = rows
rows = []
for mc in (0.85, 1.2, 2.75):
    f0 = kc.f_of_tau(kc.MERGE / 2048, mc)
    fisco = 1 / (6 ** 1.5 * np.pi * kc.TSUN * (2 ** (6 / 5) * mc))
    I = quad(lambda f: kc.tau_of_f(f, mc), f0, fisco, limit=200)[0]
    N = quad(lambda f: (8 / 3) * kc.tau_of_f(f, mc), f0, fisco, limit=200)[0]
    rows.append({"chirp_mass": mc, "from_Hz": round(float(f0), 1), "to_isco_Hz": round(float(fisco), 0),
                 "cycles": round(N, 1), "2pi_int_tau_df": round(2 * np.pi * I, 1), "ratio": round(2 * np.pi * I / N, 4)})
out["resonance_count_vs_cycles"] = rows
out["3pi_over_4"], out["5pi_over_4"] = round(3 * np.pi / 4, 4), round(5 * np.pi / 4, 4)
sh = {}
for n, nyq in (("noisy_s4d_d64", 512), ("noisy_s4d_d512", 512), ("noisy_s4d_d512_band", 512), ("clean_s4d_layer1", 1024)):
    e = {}
    for tag, fn in (("start", f"{n}_init_modes.npz"), ("trained", f"{n}_modes.npz")):
        z = np.load(os.path.join(K, fn))
        fa = np.abs(z["f"])
        inb = (fa >= 20) & (fa < nyq)
        r = np.log10(z["ring"][inb] / kc.tau_of_f(fa[inb], 1.2))
        e[tag] = round(float(np.mean(np.abs(r) <= np.log10(2))), 4)
    sh[n] = e
out["share_inband_ring_within_factor2_of_tau_Mc1.2"] = sh
ex = {}
for n in ("noisy_s4d_d64", "noisy_s4d_d512", "noisy_s4d_d512_band", "noisy_free", "clean_s4d_layer1", "clean_free"):
    T = json.load(open(os.path.join(K, f"{n}_table.json")))
    ch, real = T["channels"], T["info"]["real_kernel"]
    M = np.load(os.path.join(K, f"{n}_match.npz"))
    mc = M["chirp_mass"]
    h1 = np.array([r["best_match_real" if real else "best_match_pos"] for r in ch])
    l1 = np.array([r["best_match_real_L1" if real else "best_match_pos_L1"] for r in ch])
    big = h1 > 0.05
    e = {"L1_over_H1_best_unit_norm_match_q1_median_q3": [round(float(v), 4) for v in np.percentile(l1[big] / h1[big], [25, 50, 75])]}
    for target in (0.85, 1.2, 2.0, 2.74):
        j = int(np.argmin(np.abs(mc - target)))
        e[f"best_inband_match_at_Mc{mc[j]:.3f}"] = round(float(np.nanmax(M["inband_match"][j].astype(float))), 4)
    dl = [r for r in ch if r["class"] == "delay-line piece"]
    if dl:
        e["delay_line_chirp_delay_span_in_band_s_q1_median_q3"] = [round(float(v), 3) for v in np.percentile([r["chirp_delay_span_in_band_s"] for r in dl], [25, 50, 75])]
        e["delay_line_inband_match_q1_median_q3"] = [round(float(v), 3) for v in np.percentile([r["best_inband_match"] for r in dl], [25, 50, 75])]
    ex[n] = e
out["per_network"] = ex
z = np.load(os.path.join(K, "noisy_s4d_d512_band_modes.npz"))
fa, ring, w = np.abs(z["f"]), z["ring"], z["weight"]
inb = (fa >= 20) & (fa < 512)
short = inb & (ring < 1.0)
T = json.load(open(os.path.join(K, "noisy_s4d_d512_band_table.json")))["channels"]
dlm = np.array([r["class"] == "delay-line piece" for r in T])
out["band_start_rings_below_1s"] = {"modes": int(short.sum()), "of_inband": int(inb.sum()),
                                    "freq_Hz_p5_p25_p50_p75_p95": [round(float(v), 1) for v in np.percentile(fa[short], [5, 25, 50, 75, 95])],
                                    "channels_with_any": int(short.any(1).sum()),
                                    "delay_line_channels_with_any": int((short.any(1) & dlm).sum()), "delay_line_channels": int(dlm.sum())}
print(json.dumps(out, indent=1))
with open(os.path.join(K, "side_checks.json"), "w") as fh:
    json.dump(out, fh, indent=1)

# (5) where the S4D-Lin delay-line pieces' ring times come from: dt, and A_real against its start value 0.5
src = {}
for n in ("noisy_s4d_d64", "noisy_s4d_d512"):
    T = json.load(open(os.path.join(K, f"{n}_table.json")))["channels"]
    z = np.load(os.path.join(K, f"{n}_modes.npz"))
    f, ring, w, a, dt = z["f"], z["ring"], z["weight"], z["a"], z["dt"]
    e = {}
    for tag, chs in (("delay_line_pieces", [r["channel"] for r in T if r["class"] == "delay-line piece"]),
                     ("other_channels", [r["channel"] for r in T if r["class"] != "delay-line piece"])):
        dla, r0, rt = [], [], []
        for c in chs:
            r = T[c]
            sel = (np.abs(f[c]) >= r["f10"]) & (np.abs(f[c]) <= r["f90"])
            if not sel.any():
                continue
            ww = w[c, sel] / w[c, sel].sum()
            v = np.log(a[c, sel] / 0.5)
            o = np.argsort(v)
            dla.append(v[o][np.searchsorted(np.cumsum(ww[o]), 0.5)])
            r0.append(2 / (dt[c] * 1024.0) / kc.tau_of_f(r["f50"], r["best_chirp_mass"]))
            rr = ring[c, sel] / kc.tau_of_f(np.abs(f[c, sel]), r["best_chirp_mass"])
            o = np.argsort(rr)
            rt.append(rr[o][np.searchsorted(np.cumsum(ww[o]), 0.5)])
        e[tag] = {"n": len(dla),
                  "dt_q1_median_q3": [round(float(v), 5) for v in np.percentile([dt[c] for c in chs], [25, 50, 75])],
                  "weighted_median_log_a_over_start_q1_median_q3": [round(float(v), 3) for v in np.percentile(dla, [25, 50, 75])],
                  "ring_with_start_A_real_over_tau_f50_q1_median_q3": [round(float(v), 3) for v in np.percentile(r0, [25, 50, 75])],
                  "trained_ring_over_tau_q1_median_q3": [round(float(v), 3) for v in np.percentile(rt, [25, 50, 75])]}
    src[n] = e
out["delay_line_ring_source"] = src
print(json.dumps(src, indent=1))
with open(os.path.join(K, "side_checks.json"), "w") as fh:
    json.dump(out, fh, indent=1)
