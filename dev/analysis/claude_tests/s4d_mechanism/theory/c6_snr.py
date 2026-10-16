"""Section 4: weighting by SNR with the O3 average noise curve.

A. Coverage model (stationary phase, Newtonian): SNR^2 per Hz
   rho2(f) ~ f^(-7/3) / S(f) on [f(4 s), f_end]. A resonator at f covers
   1/(c tau(f)) Hz, so it carries rho2(f) / (c tau(f)); a Fourier tone covers
   1/T_w and carries rho2(f) T_w^-1. Greedy: take the cells with the most SNR^2
   per unit cost first; match(N) = sqrt(captured / total).
B. The same model checked against Fourier counts measured earlier on whitened
   training templates (533 to 1,184), and recomputed here.
C. Whitened training templates (analytic h+): measured group delay and energy,
   the model's counts, and least-squares fits of resonator banks (cap R <= 2).
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import h5py
import numpy as np

from common import (FS, L0, MASSES, RES, TSUN, WF_H5, f_of_tau, fit_tradeoff, group_delay,
                    logz, o3_psd, place, tau_of_f)

TW = L0 / FS
WIN, POST, TAPER, FLOW = 8192, 128, 512, 20.0
L = WIN - POST
out = {}
t0 = time.time()
fpsd, S = o3_psd()


def f_isco_equal(mc):
    return 1.0 / (6 ** 1.5 * np.pi * TSUN * 2 ** 1.2 * mc)


def coverage(f, rho2, cost):
    """Greedy curve: order cells by rho2/cost, return cumulative cost and match."""
    o = np.argsort(-(rho2 / cost))
    cc = np.cumsum(cost[o])
    mm = np.sqrt(np.cumsum(rho2[o]) / rho2.sum())
    return cc, mm


def n_for(cc, mm, m):
    i = np.searchsorted(mm, m)
    return float(cc[min(i, cc.size - 1)])


def m_at(cc, mm, n):
    i = np.searchsorted(cc, n) - 1
    return float(mm[i]) if i >= 0 else 0.0


# ------------------------------------------------------------------ A. SPA model
A = {}
CS = (1.0, 1.3, 1.5, np.e)
for mc in MASSES + (1.0, 1.5, 2.0):
    f1 = float(f_of_tau(TW, mc))
    fend = float(min(f_isco_equal(mc), 1024.0))
    df = 0.01
    f = np.arange(f1 + df / 2, fend, df)
    rho2 = f ** (-7.0 / 3.0) / np.interp(f, fpsd, S) * df
    tau = np.minimum(tau_of_f(f, mc), TW)
    row = {"f1": f1, "f_end": fend}
    ccF, mmF = coverage(f, rho2, np.full(f.size, TW * df))
    row["fourier_N_0.9"] = n_for(ccF, mmF, 0.9)
    row["fourier_match_at"] = {str(n): m_at(ccF, mmF, n) for n in (16, 32, 256)}
    for c in CS:
        cc, mm = coverage(f, rho2, c * tau * df)
        key = f"c{c:.3f}"
        row[f"res_N_0.9_{key}"] = n_for(cc, mm, 0.9)
        row[f"res_match_at_{key}"] = {str(n): m_at(cc, mm, n) for n in (16, 32, 256)}
    # band that the c-independent greedy order picks first for 0.9
    o = np.argsort(-(rho2 / tau))
    k = np.searchsorted(np.sqrt(np.cumsum(rho2[o]) / rho2.sum()), 0.9)
    sel = f[o[:k + 1]]
    row["res_band_for_0.9"] = [float(sel.min()), float(sel.max())]
    o = np.argsort(-rho2)
    k = np.searchsorted(np.sqrt(np.cumsum(rho2[o]) / rho2.sum()), 0.9)
    sel = f[o[:k + 1]]
    row["fourier_band_for_0.9"] = [float(sel.min()), float(sel.max())]
    row["peak_rho2_per_Hz_at"] = float(f[np.argmax(rho2)])
    row["peak_rho2_over_tau_at"] = float(f[np.argmax(rho2 / tau)])
    A[str(mc)] = row
    print("A", mc, json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}), flush=True)
out["A_spa_model"] = A
json.dump(out, open(os.path.join(RES, "c6_snr.json"), "w"), indent=1)

# ------------------------------------------------------------------ C. templates
with h5py.File(WF_H5, "r") as fh:
    par = {k: fh["parameters"][k][:] for k in ("mass_1", "mass_2", "a_1", "a_2", "theta_jn")}
m1, m2 = par["mass_1"], par["mass_2"]
mcs = (m1 * m2) ** 0.6 / (m1 + m2) ** 0.2
q = np.minimum(m1, m2) / np.maximum(m1, m2)


def pick(Mc):
    cand = np.flatnonzero((np.abs(np.log(mcs / Mc)) < 0.002) & (q >= 0.85) & (q <= 0.95))
    return int(cand[np.argmin(par["a_1"][cand] + par["a_2"][cand])]) if cand.size else None


def whitened(row, use="plus"):
    with h5py.File(WF_H5, "r") as fh:
        hp = fh["waveforms/plus"][row]
        hc = fh["waveforms/cross"][row]
    m = int(np.argmax(hp ** 2 + hc ** 2))
    seg = (hp if use == "plus" else hp - 1j * hc)[m - (L - 1): m + POST + 1]
    af = np.abs(np.fft.fftfreq(WIN, 1 / FS))
    keep = af >= FLOW
    H = np.fft.fft(seg)
    Hw = np.zeros_like(H, dtype=complex)
    Hw[keep] = H[keep] / np.sqrt(np.interp(af[keep], fpsd, S))
    if use == "plus":                              # analytic: positive frequencies only
        Hw[np.fft.fftfreq(WIN, 1 / FS) <= 0] = 0
        Hw *= 2
    hw = np.fft.ifft(Hw)
    taper = np.ones(WIN)
    taper[:TAPER] = 0.5 * (1.0 - np.cos(np.pi * np.arange(TAPER) / TAPER))
    return hw * taper


def fourier_count(K, m):
    p = np.sort(np.abs(np.fft.fft(K)) ** 2)[::-1]
    c = np.sqrt(np.cumsum(p) / p.sum())
    return int(np.searchsorted(c, m) + 1)


C = {}
rows_used = {"1.0": 35720, "1.2": 24496, "1.5": 107049, "2.0": 159822}
for Mc in (0.85, 1.2, 1.8, 2.75, 1.0, 1.5, 2.0):
    row = rows_used.get(str(Mc)) or pick(Mc)
    if row is None:
        print("no row for", Mc)
        continue
    mc_det = float(mcs[row])
    entry = {"row": row, "chirp_mass_det": mc_det, "theta_jn": float(par["theta_jn"][row])}
    # B. recompute the earlier Fourier count on h+ - i hx (same construction as tone_check)
    hwc = whitened(row, use="complex")
    entry["fourier_count_0.9_hplus_minus_i_hcross"] = fourier_count(hwc[:L], 0.9)
    # C. analytic h+ kernel
    hw = whitened(row, use="plus")
    K = np.conj(hw[L - 1::-1])
    K = K / np.linalg.norm(K)
    entry["fourier_count_0.9_analytic_hplus"] = fourier_count(K, 0.9)
    f, tg, P = group_delay(K, pad=1)
    pos = (f > FLOW) & (f < 1024)
    f, tg, P = f[pos], tg[pos], P[pos]
    o = np.argsort(f)
    f, tg, P = f[o], tg[o], P[o]
    # smooth over +-1 Hz (energy-weighted for the delay)
    w = int(round(1.0 / (FS / L)))
    ker = np.ones(2 * w + 1)
    Ps = np.convolve(P, ker, "same")
    tgs = np.convolve(tg * P, ker, "same") / np.maximum(Ps, 1e-300)
    tgs = np.clip(tgs, 1.0 / FS, TW)
    Ps = Ps / Ps.sum()
    for fx in (100.0, 200.0, 400.0):
        i = np.argmin(np.abs(f - fx))
        entry[f"group_delay_ms_{int(fx)}"] = float(1e3 * tgs[i])
        entry[f"newtonian_tau_ms_{int(fx)}"] = float(1e3 * tau_of_f(fx, mc_det))
    dfb = FS / L
    rho2 = P / P.sum()
    ccF, mmF = coverage(f, rho2, np.full(f.size, TW * dfb))
    entry["model_fourier_N_0.9"] = n_for(ccF, mmF, 0.9)
    for c in CS:
        cc, mm = coverage(f, rho2, c * tgs * dfb)
        entry[f"model_res_N_0.9_c{c:.3f}"] = n_for(cc, mm, 0.9)
        entry[f"model_res_match_at_c{c:.3f}"] = {str(n): m_at(cc, mm, n) for n in (16, 32, 64, 128, 256)}
    # fits along the greedy order: keep the best cells holding fraction F,
    # place c * tau_g(f) resonators per Hz there, decay alpha / tau_g
    order = np.argsort(-(rho2 / tgs))
    cum = np.cumsum(rho2[order])

    def layout(Fk, c, alpha):
        k = np.searchsorted(cum, Fk)
        sel = np.zeros(f.size, bool)
        sel[order[:k + 1]] = True
        dens = c * tgs * sel
        fr = place(f, dens)
        if fr.size == 0:
            return fr, np.zeros(0, complex)
        a = alpha / np.interp(fr, f, tgs)
        return fr, logz(a, fr)

    def best_for(target=None, n_budget=None):
        best = None
        for Fk in (0.82, 0.86, 0.9, 0.94, 0.97, 1.0):
            for alpha in (0.5, 1.0, 1.5):
                if target is not None:          # bisection on c for the target match
                    lo, hi = 0.3, 4.0
                    fr, lz = layout(Fk, hi, alpha)
                    if fit_tradeoff(lz, K, caps=(2.0,))["2"]["match"] < target:
                        continue
                    for _ in range(12):
                        mid = 0.5 * (lo + hi)
                        fr, lz = layout(Fk, mid, alpha)
                        if lz.size and fit_tradeoff(lz, K, caps=(2.0,))["2"]["match"] >= target:
                            hi = mid
                        else:
                            lo = mid
                    fr, lz = layout(Fk, hi, alpha)
                    r = {"F": Fk, "alpha": alpha, "c": hi, "N": int(lz.size),
                         "match": fit_tradeoff(lz, K, caps=(2.0,))["2"]["match"]}
                    if best is None or r["N"] < best["N"]:
                        best = r
        return best

    cumtau = np.cumsum(tgs[order] * dfb)

    def best_budget(n_budget):
        """Fixed number of resonators: for each density c, keep the best cells
        whose int tau_g df equals n_budget / c, then fit."""
        best = None
        for c in (1.0, 1.15, 1.3, 1.5, 2.0, 2.7):
            k = int(np.searchsorted(cumtau, n_budget / c))
            Fk = float(cum[min(k, cum.size - 1)])
            for alpha in (0.5, 1.0, 1.5):
                fr, lz = layout(Fk, c, alpha)
                if lz.size == 0:
                    continue
                m = fit_tradeoff(lz, K, caps=(2.0,))["2"]["match"]
                r = {"F": Fk, "alpha": alpha, "c": c, "N": int(lz.size), "match": m}
                if best is None or r["match"] > best["match"]:
                    best = r
        return best

    entry["fit_N_0.9_cap2"] = best_for(target=0.9)
    entry["fit_match_at_N_cap2"] = {str(n): best_budget(n) for n in (16, 32, 64, 128, 256)}
    C[str(Mc)] = entry
    print("C", Mc, json.dumps(entry, default=float), f"[{time.time() - t0:.0f}s]", flush=True)
    out["C_templates"] = C
    json.dump(out, open(os.path.join(RES, "c6_snr.json"), "w"), indent=1, default=float)
print("done", time.time() - t0)
