"""Section 2 checks: group delay of one resonator, a wave packet through it,
and the matched filter of a chirp as a frequency-dependent delay tau(f)."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

from common import (FS, L0, RES, f_of_tau, group_delay, kernel_of, phi_of_tau,
                    tau_of_f, toy_chirp)

out = {}

# ---- (a) one resonator 1/(a + i(w - w_n)): group delay a/(a^2 + d^2)
a = 4.0
d = np.linspace(-200, 200, 2_000_001) * a
H = 1.0 / (a + 1j * d)
ph = np.unwrap(np.angle(H))
tg_num = -np.gradient(ph, d)
tg_ana = a / (a * a + d * d)
out["a_group_delay_max_abs_err_over_peak"] = float(np.max(np.abs(tg_num - tg_ana)) * a)
out["a_peak_group_delay_times_a"] = float(tg_num[np.argmin(np.abs(d))] * a)
# area: total phase turned across the resonance
out["a_phase_turned_over_pi_on_pm200a"] = float((ph[0] - ph[-1]) / np.pi)
out["a_phase_turned_exact_pm200a_over_pi"] = float(2 * np.arctan(200.0) / np.pi)
print("(a)", out)

# ---- (b) Gaussian wave packet at the resonance through e^{(-a+iw_n)t}
fs_b = 20000.0
fn = 100.0
wn = 2 * np.pi * fn
res_b = []
for a_b in (10.0,):
    T = 1 / a_b
    for sig_aT in (0.5, 1.0, 3.0, 10.0):
        sig = sig_aT * T
        t0 = 8 * sig + 1.0
        dur = t0 + 8 * sig + 12 * T
        t = np.arange(0, dur, 1 / fs_b)
        for det_aT in (0.0, 1.0):          # carrier offset in units of a
            w = wn + det_aT * a_b
            u = np.exp(-0.5 * ((t - t0) / sig) ** 2) * np.exp(1j * w * t)
            r = np.exp((-a_b + 1j * wn) * t) / fs_b
            y = np.convolve(u, r)[:t.size]
            env_in, env_out = np.abs(u), np.abs(y)
            c_in = np.sum(t * env_in) / np.sum(env_in)
            c_out = np.sum(t * env_out) / np.sum(env_out)
            p_in, p_out = t[np.argmax(env_in)], t[np.argmax(env_out)]
            pred = a_b / (a_b ** 2 + (w - wn) ** 2)
            res_b.append({"a": a_b, "sigma_times_a": sig_aT, "offset_over_a": det_aT,
                          "centroid_delay_times_a": float((c_out - c_in) * a_b),
                          "peak_delay_times_a": float((p_out - p_in) * a_b),
                          "group_delay_formula_times_a": float(pred * a_b)})
out["b_packet"] = res_b
for r in res_b:
    print("(b)", r)

# ---- (c) chirp: phase derivative equals 2 pi f(tau); tau(f) inverts f(tau)
mc = 1.2
tau = np.linspace(0.01, 4.0, 400001)
dphi = -np.gradient(phi_of_tau(tau, mc), tau)
out["c_dphi_dt_vs_2pif_max_rel_err"] = float(np.max(np.abs(dphi / (2 * np.pi * f_of_tau(tau, mc)) - 1)[5:-5]))
ff = np.array([80.0, 100.0, 200.0, 400.0, 800.0])
out["c_tau_of_f_roundtrip_max_rel_err"] = float(np.max(np.abs(f_of_tau(tau_of_f(ff, mc), mc) / ff - 1)))
out["c_tau_ms_at"] = {f"{x:.0f}": float(1e3 * tau_of_f(x, mc)) for x in ff}
print("(c)", out["c_dphi_dt_vs_2pif_max_rel_err"], out["c_tau_of_f_roundtrip_max_rel_err"], out["c_tau_ms_at"])

# ---- (d) matched-filter group delay of the toy chirp kernel equals tau(f)
gd = {}
for mc in (0.85, 1.2, 1.8, 2.75):
    h = toy_chirp(mc)
    K = kernel_of(h)
    f, tg, P = group_delay(K, pad=4)
    rows = []
    for fx in (100.0, 150.0, 200.0, 300.0, 400.0, 600.0, 800.0):
        if fx <= f_of_tau(4.0, mc) * 1.05:
            continue
        i = np.argmin(np.abs(f - fx))
        # average over +-2 Hz to remove the small ripple of the finite window
        sel = np.abs(f - fx) <= 2.0
        tgm = float(np.sum(tg[sel] * P[sel]) / np.sum(P[sel]))
        pred = float(tau_of_f(fx, mc) - 1 / FS)
        rows.append({"f": fx, "group_delay_ms": 1e3 * tgm, "tau_minus_1_sample_ms": 1e3 * pred,
                     "diff_ms": 1e3 * (tgm - pred), "rel": (tgm - pred) / pred})
    gd[str(mc)] = rows
    for r in rows:
        print("(d)", mc, r)
out["d_matched_filter_group_delay"] = gd

# ---- (e) pieces of the chirp near 100, 200 and 400 Hz enter seconds apart
# and leave the matched filter at the same sample (the merger)
mc = 1.2
h = toy_chirp(mc)
K = np.conj(h[::-1])
Hf = np.fft.fft(h, 2 * L0)
fr = np.fft.fftfreq(2 * L0, 1 / FS)
pieces = []
for fc in (100.0, 200.0, 400.0):
    win = np.exp(-0.5 * ((fr - fc) / 3.0) ** 2)
    p = np.fft.ifft(Hf * win)
    y = np.convolve(p, K)            # output sample index i means input index i
    t_in = np.sum(np.arange(p.size) * np.abs(p) ** 2) / np.sum(np.abs(p) ** 2)
    t_out = int(np.argmax(np.abs(y)))
    pieces.append({"f_center": fc, "piece_centroid_s_before_merger": float((L0 - t_in) / FS),
                   "tau_of_f_s": float(tau_of_f(fc, mc)),
                   "output_peak_sample": t_out, "merger_sample": L0 - 1})
out["e_pieces"] = pieces
for r in pieces:
    print("(e)", r)

json.dump(out, open(os.path.join(RES, "c2_delay.json"), "w"), indent=1)
print("saved")
