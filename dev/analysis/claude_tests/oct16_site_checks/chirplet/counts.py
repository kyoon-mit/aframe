"""Chirplet in-step time and piece counts for leading-order chirps.

1. In-step time of a tangent linear chirp, T_lin = (6/fddot)^(1/3), against the
   exact leading-order phase (and the tone rule fdot^(-1/2) as a control).
2. Piece counts over the last 4 s: closed forms against quadrature with a
   finite-difference fddot.
"""
import os

for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[v] = "1"

import json
import time

import numpy as np

T0 = time.time()
GM_SUN = 1.3271244e20
C = 299792458.0
TSUN = GM_SUN / C**3
TWIN = 4.0
MASSES = (0.85, 1.2, 1.8, 2.75)


def f_tau(tau, mc):
    return (1 / np.pi) * (5 / (256 * tau)) ** 0.375 * (TSUN * mc) ** (-0.625)


def tau_f(f, mc):
    return 5 / 256 * (TSUN * mc) ** (-5 / 3) * (np.pi * f) ** (-8 / 3)


def phi_tau(tau, mc):
    return -2 * (tau / (5 * TSUN * mc)) ** 0.625


def fdot(f, mc):
    return 96 / 5 * np.pi ** (8 / 3) * (TSUN * mc) ** (5 / 3) * f ** (11 / 3)


def fddot(f, mc):
    return 11 / 3 * fdot(f, mc) ** 2 / f


def t_lin(f, mc):
    return (6 / fddot(f, mc)) ** (1 / 3)


def t_tone(f, mc):
    return fdot(f, mc) ** -0.5


def instep_interval(mc, f0, kind, npts=600001):
    """Length of the connected interval around t0 where the exact phase stays
    within pi/4 of the tangent tone (kind='tone') or linear chirp ('lin')."""
    tau0 = tau_f(f0, mc)
    T = t_lin(f0, mc) if kind == "lin" else t_tone(f0, mc)
    S = 2.0 * T
    s_hi = min(S, tau0 * (1 - 1e-12))
    s = np.linspace(-S, s_hi, npts)
    ph = phi_tau(tau0 - s, mc)
    model = phi_tau(tau0, mc) + 2 * np.pi * f0 * s
    if kind == "lin":
        model = model + np.pi * fdot(f0, mc) * s**2
    bad = np.abs(ph - model) > np.pi / 4
    i0 = int(np.argmin(np.abs(s)))
    right = np.flatnonzero(bad[i0:])
    left = np.flatnonzero(bad[: i0 + 1][::-1])
    sr = s[i0 + right[0]] if right.size else s_hi
    sl = s[i0 - left[0]] if left.size else -S
    hit_merger = not right.size
    return float(sr - sl), float(T), float(-sl), float(sr), hit_merger


def counts_closed(mc, fmax=None):
    fT = f_tau(TWIN, mc)
    ncyc = 1.6 * TWIN * fT
    n_tone = 16 / 5 * np.sqrt(3 / 8) * np.sqrt(TWIN * fT)
    n_lin = 24 / 5 * (11 / 128) ** (1 / 3) * (TWIN * fT) ** (1 / 3)
    if fmax is not None:
        n_tone *= 1 - (fT / fmax) ** (5 / 6)
        n_lin *= 1 - (fT / fmax) ** (5 / 9)
    return fT, ncyc, n_tone, n_lin


def counts_quad(mc, fmax=None, n=400001):
    """Quadrature on a log grid in tau, fdot and fddot by finite differences of f(tau)."""
    tau_lo = 1e-9 if fmax is None else tau_f(fmax, mc)
    u = np.linspace(np.log(tau_lo), np.log(TWIN), n)
    tau = np.exp(u)
    h = 1e-5 * tau
    f0 = f_tau(tau, mc)
    fp, fm = f_tau(tau - h, mc), f_tau(tau + h, mc)   # t = tc - tau: tau - h is later
    fd = (fp - fm) / (2 * h)
    fdd = (fp - 2 * f0 + fm) / h**2
    w = tau  # dtau = tau du
    n_tone = np.trapz(np.sqrt(fd) * w, u)
    n_lin = np.trapz((fdd / 6) ** (1 / 3) * w, u)
    ncyc = np.trapz(f0 * w, u)
    return float(ncyc), float(n_tone), float(n_lin)


out = {"instep": [], "counts": []}
print("In-step times (exact phase vs formula):")
for mc in (0.85, 1.2, 2.75):
    for f0 in (50.0, 100.0, 300.0):
        row = {"mc": mc, "f": f0, "fdot": fdot(f0, mc), "fddot": fddot(f0, mc),
               "tau": tau_f(f0, mc)}
        for kind in ("tone", "lin"):
            L, T, sl, sr, hm = instep_interval(mc, f0, kind)
            row[kind] = {"formula": T, "exact": L, "rel": L / T - 1, "left": sl, "right": sr, "hit_merger": hm}
        row["ratio_lin_tone"] = row["lin"]["formula"] / row["tone"]["formula"]
        out["instep"].append(row)
        print(f" Mc {mc:4.2f} f {f0:5.0f}: tau {row['tau']:.4f} s fdot {row['fdot']:.4g} fddot {row['fddot']:.4g} | "
              f"tone {row['tone']['formula']:.4f} exact {row['tone']['exact']:.4f} ({100*row['tone']['rel']:+.3f}%) | "
              f"lin {row['lin']['formula']:.4f} exact {row['lin']['exact']:.4f} ({100*row['lin']['rel']:+.2f}%; "
              f"left {row['lin']['left']:.4f} right {row['lin']['right']:.4f}{' merger' if row['lin']['hit_merger'] else ''}) "
              f"ratio {row['ratio_lin_tone']:.3f}")

print("\nCounts over the last 4 s (closed form | quadrature):")
for fmax in (None, 512.0, 900.0, 1024.0):
    for mc in MASSES:
        fT, ncyc, nt, nl = counts_closed(mc, fmax)
        qc, qt, ql = counts_quad(mc, fmax)
        res = 0.375 * ncyc if fmax is None else 0.6 * (tau_f(fT, mc) * fT - tau_f(fmax, mc) * fmax)
        r = {"fmax": fmax, "mc": mc, "fT": fT, "ncyc_to_merger": ncyc, "tone": nt, "lin": nl,
             "quad_cyc": qc, "quad_tone": qt, "quad_lin": ql, "int_tau_df": res,
             "tone_over_sqrtN": nt / np.sqrt(ncyc) if fmax is None else None,
             "lin_over_cbrtN": nl / ncyc ** (1 / 3) if fmax is None else None}
        out["counts"].append(r)
        print(f" fmax {str(fmax):>6} Mc {mc:4.2f}: fT {fT:7.3f} cycles {ncyc:7.2f} | tone {nt:6.2f} | {qt:6.2f}  "
              f"lin {nl:6.2f} | {ql:6.2f}  (quad cycles {qc:7.2f}; int tau df {res:7.2f})")
print("\n tone/sqrt(N) =", 2 * np.sqrt(15) / 5, " lin/N^(1/3) =", 24 / 5 * (55 / 1024) ** (1 / 3))
json.dump(out, open(os.path.join(os.path.dirname(__file__), "counts.json"), "w"), indent=1)
print(f"done in {time.time() - T0:.1f} s")
