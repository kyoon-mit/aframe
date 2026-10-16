"""Per-channel classification and per-network summary from the analyze.py and single_res.py outputs.

Writes <NAME>_table.json (per channel, final) and summary.json.
Rule (stated in kernels.md): G = (in-band match / best one-resonance in-band match in the channel's
10-90% band at the channel's best chirp mass)^2; memory ratio = energy-weighted median of ring time /
tau(f) over the channel's in-band modes inside the chirp's in-window band (S4D), or lag90 / tau(f50)
(free kernels, and S4D channels without such modes).
  delay-line piece: G >= 2 and group-delay slope against the chirp's delay in [0.5, 2];
  spectrogram row: G < 2 and memory ratio < 1;
  long, not in step: G < 2 and memory ratio >= 1;
  unclear: G >= 2 with the slope outside [0.5, 2] or undefined.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kcommon as kc  # noqa: E402

NETS = ["noisy_s4d_d64", "noisy_s4d_d512", "noisy_s4d_d512_band", "noisy_free", "clean_s4d_layer1", "clean_free"]
G_CUT, SLOPE = 2.0, (0.5, 2.0)
CLASSES = ("spectrogram row", "delay-line piece", "long, not in step", "unclear")


def wmedian(v, w):
    o = np.argsort(v)
    c = np.cumsum(w[o]) / w.sum()
    return float(v[o][np.searchsorted(c, 0.5)])


def q(v, ps=(10, 25, 50, 75, 90)):
    v = np.asarray([x for x in v if x is not None and np.isfinite(x)])
    return {f"p{p}": round(float(np.percentile(v, p)), 4) for p in ps} | {"max": round(float(v.max()), 4), "n": int(len(v))} if len(v) else {}


summary = {"rule": {"G_cut": G_CUT, "slope_range": SLOPE, "memory_ratio_cut": 1.0}}
for n in NETS:
    d = json.load(open(os.path.join(kc.HERE, f"{n}_channels.json")))
    info, ch = d["info"], d["channels"]
    fs = info["fs"]
    sr = np.load(os.path.join(kc.HERE, f"single_res_{int(fs)}.npz"))
    f0, m1b, Tb = sr["f0"], sr["match_best"], sr["T_best"]
    mfile = os.path.join(kc.HERE, f"{n}_modes.npz")
    modes = np.load(mfile) if os.path.exists(mfile) else None
    real = info["real_kernel"]
    for c, row in enumerate(ch):
        j, mc = row["best_template"], row["best_chirp_mass"]
        idx = np.where((f0 >= row["f10"]) & (f0 <= row["f90"]))[0]
        if len(idx) == 0:
            idx = [int(np.argmin(np.abs(np.log(f0 / max(row["f50"], 20.0)))))]
        i1 = idx[int(np.argmax(m1b[j, idx]))]
        row["one_resonance_bound"] = float(m1b[j, i1])
        row["one_resonance_bound_f0"] = float(f0[i1])
        row["one_resonance_bound_ring_s"] = float(Tb[j, i1])
        m = row["best_inband_match"]
        row["gain_over_one_resonance"] = None if m is None else float((m / row["one_resonance_bound"]) ** 2)
        fstart = float(kc.f_of_tau(kc.MERGE / 2048, mc))
        row["chirp_window_start_freq_Hz"] = fstart
        if modes is not None:
            f, ring, w, inb = modes["f"][c], modes["ring"][c], modes["weight"][c], modes["inband"][c]
            if real:
                sel = inb
            else:
                sel = inb & ((f < 0) if row["inband_side"] == "negative" else (f > 0))
            sel_win = sel & (np.abs(f) >= fstart)          # frequencies the chirp reaches inside the window
            row["n_modes_side_inband"] = int(sel.sum())
            row["n_modes_side_in_chirp_window_band"] = int(sel_win.sum())
            if sel_win.any():
                fa = np.abs(f[sel_win])
                rt = ring[sel_win] / kc.tau_of_f(fa, mc)
                rs = ring[sel_win] / kc.t_step(fa, mc)
                ww = w[sel_win]
                row["ring_over_tau_wmedian"] = wmedian(rt, ww)
                row["ring_over_tstep_wmedian"] = wmedian(rs, ww)
                row["weight_share_ring_ge_tstep"] = float(ww[rs >= 1].sum() / ww.sum())
                row["weight_share_ring_ge_0.3tau"] = float(ww[rt >= 0.3].sum() / ww.sum())
                row["ring_wmedian_s"] = wmedian(ring[sel_win], ww)
                row["mode_weight_in_chirp_window_band"] = float(ww.sum())
            else:
                row["ring_over_tau_wmedian"] = row["ring_over_tstep_wmedian"] = None
        lag_ratio = row["lag90_s"] / float(kc.tau_of_f(max(row["f50"], fstart), mc))
        row["lag90_over_tau_f50"] = lag_ratio
        if row.get("ring_over_tau_wmedian") is not None:
            row["memory_ratio"], row["memory_ratio_from"] = row["ring_over_tau_wmedian"], "ring time"
        else:
            row["memory_ratio"], row["memory_ratio_from"] = lag_ratio, "lag90"
        g, s = row["gain_over_one_resonance"], row["gd_slope_vs_chirp"]
        if g is not None and g >= G_CUT:
            cls = "delay-line piece" if (s is not None and SLOPE[0] <= s <= SLOPE[1]) else "unclear"
        else:
            cls = "spectrogram row" if row["memory_ratio"] < 1 else "long, not in step"
        row["class"] = cls
    with open(os.path.join(kc.HERE, f"{n}_table.json"), "w") as fh:
        json.dump({"info": info, "rule": summary["rule"], "channels": ch}, fh, indent=1)

    S = {"info": {k: info[k] for k in ("checkpoint", "epoch", "best_model_score_within_1pct", "fs", "L", "channels")},
         "checks": info["checks"]}
    S["best_inband_match"] = q([r["best_inband_match"] for r in ch])
    S["best_match_unit_norm"] = q([r["best_match_real"] if real else r["best_match_pos"] for r in ch])
    if not real:
        S["best_match_network_energy"] = q([r["best_match_net"] for r in ch])
        S["energy_share_negative_freq"] = q([r["energy_share_negative_freq"] for r in ch])
        S["channels_inband_side_negative"] = int(sum(r["inband_side"] == "negative" for r in ch))
    else:
        S["best_inband_match_with_D"] = q([r["best_inband_match_with_D"] for r in ch])
        S["D2_over_kernel_norm2"] = q([r["D2_over_kernel_norm2"] for r in ch])
    S["energy_share_20Hz_to_nyquist"] = q([r["energy_share_20Hz_to_nyquist"] for r in ch])
    S["one_resonance_bound"] = q([r["one_resonance_bound"] for r in ch])
    S["gain_over_one_resonance"] = q([r["gain_over_one_resonance"] for r in ch])
    S["gd_slope_vs_chirp"] = q([r["gd_slope_vs_chirp"] for r in ch])
    S["lag90_s"] = q([r["lag90_s"] for r in ch])
    S["best_chirp_mass"] = q([r["best_chirp_mass"] for r in ch])
    cls = [r["class"] for r in ch]
    S["class_counts"] = {k: cls.count(k) for k in CLASSES}
    S["memory_ratio_from_lag90_channels"] = int(sum(r["memory_ratio_from"] == "lag90" for r in ch))
    for gc in (1.5, 3.0):
        alt = []
        for r in ch:
            g, s = r["gain_over_one_resonance"], r["gd_slope_vs_chirp"]
            if g is not None and g >= gc:
                alt.append("delay-line piece" if (s is not None and SLOPE[0] <= s <= SLOPE[1]) else "unclear")
            else:
                alt.append("spectrogram row" if r["memory_ratio"] < 1 else "long, not in step")
        S[f"class_counts_if_G_cut_{gc}"] = {k: alt.count(k) for k in CLASSES}
    for mcut in (0.5, 2.0):
        alt = [r["class"] if r["class"] in ("delay-line piece", "unclear") else
               ("spectrogram row" if r["memory_ratio"] < mcut else "long, not in step") for r in ch]
        S[f"class_counts_if_memory_cut_{mcut}"] = {k: alt.count(k) for k in CLASSES}
    by = {}
    for k in CLASSES:
        rows = [r for r in ch if r["class"] == k]
        if not rows:
            continue
        e = {"n": len(rows),
             "inband_match_median": round(float(np.median([r["best_inband_match"] for r in rows])), 4),
             "gain_median": round(float(np.median([r["gain_over_one_resonance"] for r in rows if r["gain_over_one_resonance"] is not None])), 3),
             "gd_slope_median": round(float(np.nanmedian([np.nan if r["gd_slope_vs_chirp"] is None else r["gd_slope_vs_chirp"] for r in rows])), 3),
             "lag90_median_s": round(float(np.median([r["lag90_s"] for r in rows])), 4),
             "band_f10_f90_median_Hz": [round(float(np.median([r["f10"] for r in rows])), 1), round(float(np.median([r["f90"] for r in rows])), 1)],
             "best_chirp_mass_median": round(float(np.median([r["best_chirp_mass"] for r in rows])), 3),
             "band_ratio_f90_over_f10_median": round(float(np.median([r["f90"] / r["f10"] for r in rows])), 2),
             "energy_share_20Hz_to_nyquist_median": round(float(np.median([r["energy_share_20Hz_to_nyquist"] for r in rows])), 4),
             "memory_ratio_median": round(float(np.median([r["memory_ratio"] for r in rows])), 4),
             "lag90_over_tau_f50_median": round(float(np.median([r["lag90_over_tau_f50"] for r in rows])), 4)}
        if modes is not None:
            vt = [r.get("ring_over_tau_wmedian") for r in rows if r.get("ring_over_tau_wmedian") is not None]
            vs = [r.get("ring_over_tstep_wmedian") for r in rows if r.get("ring_over_tstep_wmedian") is not None]
            e["ring_over_tau_wmedian_median"] = round(float(np.median(vt)), 4) if vt else None
            e["ring_over_tstep_wmedian_median"] = round(float(np.median(vs)), 4) if vs else None
            e["modes_in_chirp_window_band_median"] = float(np.median([r.get("n_modes_side_in_chirp_window_band", 0) for r in rows]))
        if "readout_weight_norm" in rows[0]:
            e["readout_weight_norm_median"] = round(float(np.median([r["readout_weight_norm"] for r in rows])), 4)
        by[k] = e
    S["by_class"] = by
    if "readout_weight_norm" in ch[0]:
        rw = np.array([r["readout_weight_norm"] for r in ch])
        top = np.argsort(rw)[::-1][: len(rw) // 4]
        S["class_counts_top_quarter_by_readout_weight"] = {k: int(sum(ch[i]["class"] == k for i in top)) for k in CLASSES}
    if modes is not None:
        f, ring, w, inb, al = modes["f"], modes["ring"], modes["weight"], modes["inband"], modes["alias"]
        S["modes"] = {"total": int(f.size), "per_channel": int(f.shape[1]),
                      "inband_20Hz_to_nyquist": int(inb.sum()),
                      "inband_positive_freq": int((inb & (f > 0)).sum()),
                      "aliased_phase_step_over_pi": int(al.sum()),
                      "inband_per_channel": q(inb.sum(1), (0, 25, 50, 75, 100)),
                      "inband_ring_time_s": q(ring[inb], (5, 25, 50, 75, 95)),
                      "inband_ring_time_s_weighted_by_mode_energy_median": wmedian(ring[inb], w[inb]),
                      "mode_weight_inband_per_channel": q((w * inb).sum(1), (10, 25, 50, 75, 90))}
        # ring time against tau(f) and T_step(f) at the ends of the chirp-mass range, all in-band modes
        fa = np.abs(f[inb])
        for mcr in (0.85, 1.2, 2.75):
            S["modes"][f"inband_ring_over_tau_Mc{mcr}"] = q(ring[inb] / kc.tau_of_f(fa, mcr), (5, 25, 50, 75, 95))
            S["modes"][f"inband_share_ring_ge_0.3tau_Mc{mcr}"] = round(float((ring[inb] >= 0.3 * kc.tau_of_f(fa, mcr)).mean()), 4)
            S["modes"][f"inband_share_ring_ge_tau_Mc{mcr}"] = round(float((ring[inb] >= kc.tau_of_f(fa, mcr)).mean()), 4)
            S["modes"][f"inband_weighted_share_ring_ge_0.3tau_Mc{mcr}"] = round(float((w[inb] * (ring[inb] >= 0.3 * kc.tau_of_f(fa, mcr))).sum() / w[inb].sum()), 4)
            S["modes"][f"inband_share_ring_ge_tstep_Mc{mcr}"] = round(float((ring[inb] >= kc.t_step(fa, mcr)).mean()), 4)
    summary[n] = S
    print(f"== {n}: classes {S['class_counts']}; in-band match {S['best_inband_match']}")
with open(os.path.join(kc.HERE, "summary.json"), "w") as fh:
    json.dump(summary, fh, indent=1)
