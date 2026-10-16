"""Section 3/4 numerical fits: banks of decaying resonators laid out with
density c * tau(f) per Hz and decay rate alpha / tau(f), weights by least
squares (optionally with a ridge that caps the weight inflation R), fitted
to the matched-filter kernel of the 7 October toy chirp (4 s, 20-900 Hz).

Part 1: whole band, grid over (c, alpha).
Part 2: fewest resonators for a match of 0.9, keeping only the band
        [F_lo, 900] Hz that holds a fraction F of the kernel's energy.
Part 3: sub-bands of the 1.2 Msun chirp: resonators needed per unit of
        int tau df, against Fourier tones per unit of int tau df.
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

from common import (FS, L0, MASSES, RES, f_of_tau, fit_tradeoff, int_tau_df, kernel_of,
                    logz, match_float32, place, tau_of_f, toy_chirp)

TW = L0 / FS
CAPS = (np.inf, 10.0, 2.0)
part = sys.argv[1] if len(sys.argv) > 1 else "all"
out_path = os.path.join(RES, f"c5_fits_{part}.json")
out = {}


def tau_eff(f, mc):
    return np.minimum(tau_of_f(f, mc), TW)


def layout(mc, fa, fb, c, alpha):
    fn = np.linspace(fa, fb, 100001)
    fr = place(fn, c * tau_eff(fn, mc))
    a = alpha / tau_eff(fr, mc)
    return fr, logz(a, fr)


def energy_above(K):
    """Fraction of the kernel's DFT energy at frequencies >= f, as a function."""
    P = np.abs(np.fft.fft(K)) ** 2
    f = np.fft.fftfreq(K.size, 1 / FS)
    o = np.argsort(f)
    f, P = f[o], P[o]
    tail = np.cumsum(P[::-1])[::-1] / P.sum()
    return f, tail


def fourier_count(K, m):
    p = np.sort(np.abs(np.fft.fft(K)) ** 2)[::-1]
    c = np.sqrt(np.cumsum(p) / p.sum())
    return int(np.searchsorted(c, m) + 1)


def min_c_for(mc, K, fa, fb, alpha, target=0.9, cap="2", cmin=0.3, cmax=4.0):
    """Bisection on c for the smallest layout reaching the target match."""
    def m_of(c):
        fr, lz = layout(mc, fa, fb, c, alpha)
        if lz.size == 0:
            return 0.0, 0
        r = fit_tradeoff(lz, K, caps=(np.inf if cap == "inf" else float(cap),))
        return r[cap]["match"], lz.size
    mhi, nhi = m_of(cmax)
    if mhi < target:
        return None
    lo, hi = cmin, cmax
    for _ in range(14):
        mid = 0.5 * (lo + hi)
        if m_of(mid)[0] >= target:
            hi = mid
        else:
            lo = mid
    m, n = m_of(hi)
    return {"c": hi, "N": n, "match": m}


t0 = time.time()
# ------------------------------------------------------------------ part 1
if part in ("1", "all"):
    P1 = {}
    for mc in MASSES:
        K = kernel_of(toy_chirp(mc))
        f1 = float(f_of_tau(TW, mc))
        I = float(int_tau_df(f1, 900.0, mc))
        rows = []
        for c in (0.8, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.75, 2.0, np.e):
            for alpha in (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, np.pi):
                fr, lz = layout(mc, f1, 900.0, c, alpha)
                r = fit_tradeoff(lz, K, caps=CAPS, want_coef=True)
                row = {"c": c, "alpha": alpha, "N": r["n"], "rank": r["rank"]}
                for k in ("inf", "10", "2"):
                    row[f"match_{k}"] = r[k]["match"]
                    row[f"R_{k}"] = r[k]["inflation"]
                row["match_inf_float32"] = match_float32(lz, r["coef"]["inf"], K)
                row["match_2_float32"] = match_float32(lz, r["coef"]["2"], K)
                rows.append(row)
            best = max((x for x in rows if x["c"] == c), key=lambda x: x["match_2"])
            print(f"P1 mc={mc} c={c:.2f} N={best['N']} best alpha={best['alpha']:.2f} "
                  f"match cap2={best['match_2']:.4f} cap10={best['match_10']:.4f} inf={best['match_inf']:.4f} "
                  f"f32(inf)={best['match_inf_float32']:.4f} [{time.time() - t0:.0f}s]", flush=True)
        P1[str(mc)] = {"f1": f1, "int_tau_df": I, "rows": rows}
    out["part1"] = P1
    json.dump(out, open(out_path, "w"), indent=1)

# ------------------------------------------------------------------ part 2
if part in ("2", "all"):
    P2 = {}
    for mc in MASSES:
        K = kernel_of(toy_chirp(mc))
        f, tail = energy_above(K)
        best = None
        rows = []
        for F in (0.82, 0.85, 0.88, 0.91, 0.95, 1.0):
            if F >= 1.0:
                flo = float(f_of_tau(TW, mc))
            else:
                flo = float(f[np.flatnonzero(tail >= F)[-1]])
            flo = max(flo, float(f_of_tau(TW, mc)))
            for alpha in (0.5, 0.75, 1.0, 1.5):
                for cap in ("2", "inf"):
                    r = min_c_for(mc, K, flo, 900.0, alpha, cap=cap)
                    if r is None:
                        continue
                    row = {"F": F, "f_lo": flo, "alpha": alpha, "cap": cap,
                           "int_tau_df_band": float(int_tau_df(flo, 900.0, mc)), **r}
                    rows.append(row)
                    print(f"P2 mc={mc} F={F} flo={flo:.1f} alpha={alpha} cap={cap}: N={r['N']} c={r['c']:.3f} "
                          f"match={r['match']:.4f} [{time.time() - t0:.0f}s]", flush=True)
        for cap in ("2", "inf"):
            cand = [x for x in rows if x["cap"] == cap]
            b = min(cand, key=lambda x: x["N"]) if cand else None
            P2.setdefault(str(mc), {})[f"best_cap_{cap}"] = b
        # phase-budget estimate: c = 1 on the cheapest band holding 0.81
        flo81 = float(f[np.flatnonzero(tail >= 0.81)[-1]])
        flo81 = max(flo81, float(f_of_tau(TW, mc)))
        P2[str(mc)]["phase_budget_N_0.9"] = float(int_tau_df(flo81, 900.0, mc))
        P2[str(mc)]["f_lo_0.81"] = flo81
        P2[str(mc)]["fourier_count_0.9"] = fourier_count(K, 0.9)
        P2[str(mc)]["rows"] = rows
        print("P2 summary", mc, {k: v for k, v in P2[str(mc)].items() if k != "rows"}, flush=True)
    out["part2"] = P2
    json.dump(out, open(out_path, "w"), indent=1)

# ------------------------------------------------------------------ part 3
if part in ("3", "all"):
    P3 = {}
    for mc, edges in ((1.2, (82, 100, 125, 160, 210, 280, 380, 520, 720, 900)),
                      (2.75, (50, 62, 80, 105, 140, 190, 260, 360, 520, 900))):
        K = kernel_of(toy_chirp(mc))
        Kf = np.fft.fft(K)
        fk = np.fft.fftfreq(L0, 1 / FS)
        rows = []
        for fa, fb in zip(edges[:-1], edges[1:]):
            # smooth band window: flat on [fa, fb], raised-cosine edges 10% of the width
            ew = 0.1 * (fb - fa)
            win = np.zeros(L0)
            inb = (fk >= fa) & (fk <= fb)
            win[inb] = 1.0
            lo = (fk > fa - ew) & (fk < fa)
            hi = (fk > fb) & (fk < fb + ew)
            win[lo] = 0.5 * (1 - np.cos(np.pi * (fk[lo] - (fa - ew)) / ew))
            win[hi] = 0.5 * (1 + np.cos(np.pi * (fk[hi] - fb) / ew))
            piece = np.fft.ifft(Kf * win)
            piece /= np.linalg.norm(piece)
            I = float(int_tau_df(fa, fb, mc))
            nF = fourier_count(piece, 0.9)
            res = {}
            for alpha in (0.5, 1.0, 1.5):
                r = min_c_for(mc, piece, fa - ew, fb + ew, alpha, cap="2", cmin=0.2, cmax=6.0)
                res[str(alpha)] = r
            okr = [v for v in res.values() if v is not None]
            bestr = min(okr, key=lambda x: x["N"]) if okr else None
            row = {"band": [fa, fb], "int_tau_df": I, "bandwidth_x_4s": 4.0 * (fb - fa),
                   "fourier_count_0.9": nF, "resonators_0.9_cap2": bestr, "by_alpha": res,
                   "resonators_per_int_tau": (bestr["N"] / I) if bestr else None,
                   "fourier_per_int_tau": nF / I}
            rows.append(row)
            print(f"P3 mc={mc} band {fa}-{fb}: int tau df={I:.2f} Fourier={nF} resonators={bestr['N'] if bestr else None} "
                  f"(per int tau: res {row['resonators_per_int_tau']}, Fourier {row['fourier_per_int_tau']:.2f}) "
                  f"[{time.time() - t0:.0f}s]", flush=True)
        P3[str(mc)] = rows
    out["part3"] = P3
    json.dump(out, open(out_path, "w"), indent=1)

print("done", time.time() - t0)
