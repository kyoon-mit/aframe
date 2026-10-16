"""Independent check of the chirplet in-step time and piece counts (leading order).

1. In-step time of the tangent linear chirp, (6/fddot)^(1/3), and of the
   tangent steady wave, fdot^(-1/2), against the exact leading-order phase.
2. Piece counts over the last 4 s: closed forms against scipy quad, to the
   merger and to f_max.
3. Kept fractions of one piece: linear (cubic error) and steady (quadratic).
"""
import os

for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[v] = "1"

import json
import sys
import time

import numpy as np
from scipy import integrate, optimize

T0 = time.time()
TSUN = 1.3271244e20 / 299792458.0**3  # G M_sun / c^3 in s (IAU 2015 nominal)
TWIN = 4.0
OUT = sys.argv[1]


def f_of_tau(tau, mc):
    return (5.0 / (256.0 * tau)) ** 0.375 * (TSUN * mc) ** -0.625 / np.pi


def tau_of_f(f, mc):
    return 5.0 / 256.0 * (TSUN * mc) ** (-5.0 / 3.0) * (np.pi * f) ** (-8.0 / 3.0)


def phase_of_tau(tau, mc):
    """Leading-order phase; d/dt = -d/dtau gives 2 pi f."""
    return -2.0 * (tau / (5.0 * TSUN * mc)) ** 0.625


def fdot(tau, mc):
    return 0.375 * f_of_tau(tau, mc) / tau


def fddot(tau, mc):
    return 0.375 * 1.375 * f_of_tau(tau, mc) / tau**2


res = {"constants": {"TSUN_s": TSUN}}

# 0. sanity: d(phase)/dt = 2 pi f and fdot, fddot by finite differences
chk = []
for mc in (0.85, 1.2, 2.75):
    for tau in (3.9, 1.0, 0.1):
        h = 1e-4 * tau
        dphi = -(phase_of_tau(tau + h, mc) - phase_of_tau(tau - h, mc)) / (2 * h)
        fd = -(f_of_tau(tau + h, mc) - f_of_tau(tau - h, mc)) / (2 * h)
        fdd = (f_of_tau(tau + h, mc) - 2 * f_of_tau(tau, mc) + f_of_tau(tau - h, mc)) / h**2
        chk.append([mc, tau, dphi / (2 * np.pi * f_of_tau(tau, mc)) - 1, fd / fdot(tau, mc) - 1,
                    fdd / fddot(tau, mc) - 1])
res["finite_difference_rel_errors_phase_fdot_fddot"] = chk
print("max rel err (phase rate, fdot, fddot):", np.max(np.abs(np.array(chk)[:, 2:]), axis=0))

# 1. in-step intervals against the exact phase
rows = []
for mc in (0.85, 1.2, 2.75):
    for f0 in (50.0, 100.0, 300.0):
        tau0 = tau_of_f(f0, mc)
        fd0, fdd0 = fdot(tau0, mc), fddot(tau0, mc)
        T_lin = (6.0 / fdd0) ** (1.0 / 3.0)
        T_tone = fd0 ** -0.5
        out = {"mc": mc, "f": f0, "tau": tau0, "fdot": fd0, "fddot": fdd0, "T_lin": T_lin, "T_tone": T_tone}
        for kind, T in (("lin", T_lin), ("tone", T_tone)):
            span = 3.0 * T
            s = np.linspace(-span, min(span, tau0 * (1 - 1e-9)), 2_000_001)
            exact = phase_of_tau(tau0 - s, mc)
            model = phase_of_tau(tau0, mc) + 2 * np.pi * f0 * s
            if kind == "lin":
                model = model + np.pi * fd0 * s**2
            bad = np.abs(exact - model) > np.pi / 4
            i0 = int(np.argmin(np.abs(s)))
            r = np.flatnonzero(bad[i0:])
            l = np.flatnonzero(bad[: i0 + 1][::-1])
            right = s[i0 + r[0]] if r.size else np.nan
            left = -s[i0 - l[0]] if l.size else np.nan
            out[kind + "_exact"] = float(right + left)
            out[kind + "_rel"] = float((right + left) / T - 1)
            out[kind + "_left_right"] = [float(left), float(right)]
        out["ratio"] = T_lin / T_tone
        out["ratio_formula"] = (18.0 / 11.0 * f0 * T_tone) ** (1.0 / 3.0)
        rows.append(out)
        print(f"Mc {mc:4.2f} f {f0:5.0f}: tau {tau0:8.4f} fdot {fd0:10.4g} fddot {fdd0:10.4g} | "
              f"T_lin {T_lin:.4f} exact {out['lin_exact']:.4f} ({100 * out['lin_rel']:+.3f}%) | "
              f"T_tone {T_tone:.4f} exact {out['tone_exact']:.4f} ({100 * out['tone_rel']:+.3f}%) | "
              f"ratio {out['ratio']:.3f} ({out['ratio_formula']:.3f})")
res["instep"] = rows

# 2. counts: closed form against quad, to the merger and to f_max
cnt = []
for mc in (0.85, 1.2, 1.8, 2.75):
    fT = f_of_tau(TWIN, mc)
    N = 1.6 * TWIN * fT
    lin_cf = 24.0 / 5.0 * TWIN * (fddot(TWIN, mc) / 6.0) ** (1.0 / 3.0)
    lin_cf2 = 0.6 * (27.5 * N) ** (1.0 / 3.0)
    tone_cf = 16.0 / 5.0 * TWIN * fdot(TWIN, mc) ** 0.5
    tone_cf2 = 0.4 * np.sqrt(15.0 * N)
    row = {"mc": mc, "fT": fT, "cycles": N, "int_tau_df": 0.375 * N}
    # substitute tau = u^(24/5) (lin) or u^(16/5) (tone): the integrands become smooth
    lin_q = integrate.quad(lambda u: (fddot(u ** 4.8, mc) / 6.0) ** (1 / 3) * 4.8 * u**3.8,
                           0.0, TWIN ** (5 / 24), epsabs=1e-13, epsrel=1e-13, limit=200)[0]
    tone_q = integrate.quad(lambda u: fdot(u ** 3.2, mc) ** 0.5 * 3.2 * u**2.2,
                            0.0, TWIN ** (5 / 16), epsabs=1e-13, epsrel=1e-13, limit=200)[0]
    cyc_q = integrate.quad(lambda u: f_of_tau(u ** 1.6, mc) * 1.6 * u**0.6, 0.0, TWIN ** 0.625,
                           epsabs=1e-12, epsrel=1e-13, limit=200)[0]
    row.update({"lin_closed": lin_cf, "lin_closed_N": lin_cf2, "lin_quad": lin_q,
                "tone_closed": tone_cf, "tone_closed_N": tone_cf2, "tone_quad": tone_q, "cycles_quad": cyc_q})
    for fmax in (1024.0, 900.0, 512.0):
        tlo = tau_of_f(fmax, mc)
        lq = integrate.quad(lambda t: (fddot(t, mc) / 6.0) ** (1 / 3), tlo, TWIN, epsrel=1e-12, limit=200)[0]
        tq = integrate.quad(lambda t: fdot(t, mc) ** 0.5, tlo, TWIN, epsrel=1e-12, limit=200)[0]
        cq = integrate.quad(lambda t: f_of_tau(t, mc), tlo, TWIN, epsrel=1e-12, limit=200)[0]
        res_q = 0.6 * (TWIN * fT - tlo * fmax)
        row[f"fmax{int(fmax)}"] = {
            "lin_closed": lin_cf * (1 - (fT / fmax) ** (5 / 9)), "lin_quad": lq,
            "tone_closed": tone_cf * (1 - (fT / fmax) ** (5 / 6)), "tone_quad": tq,
            "cycles": cq, "int_tau_df": res_q, "tau_at_fmax": tlo}
    cnt.append(row)
    a = row["fmax1024"]
    b = row["fmax512"]
    print(f"Mc {mc:4.2f}: fT {fT:.3f} N {N:.2f} (quad {cyc_q:.2f}) int tau df {0.375 * N:.2f} | "
          f"lin {lin_cf:.3f}/{lin_cf2:.3f}/quad {lin_q:.3f} | tone {tone_cf:.3f}/{tone_cf2:.3f}/quad {tone_q:.3f} | "
          f"to 1024: lin {a['lin_closed']:.3f}/{a['lin_quad']:.3f} tone {a['tone_closed']:.3f}/{a['tone_quad']:.3f} "
          f"int tau df {a['int_tau_df']:.2f} | to 512: lin {b['lin_closed']:.3f}/{b['lin_quad']:.3f} "
          f"tone {b['tone_closed']:.3f}/{b['tone_quad']:.3f} int tau df {b['int_tau_df']:.2f}")
res["counts"] = cnt
print("coefficients: lin", 0.6 * 27.5 ** (1 / 3), "=", 24 / 5 * (55 / 1024) ** (1 / 3),
      " tone", 0.4 * np.sqrt(15), "=", 2 * np.sqrt(15) / 5)

# 3. kept fractions and the largest total overlap of one piece
k_lin = integrate.quad(lambda w: np.cos(np.pi * w**3 / 4), 0, 1, epsabs=1e-14)[0]
k_tone = abs(integrate.quad(lambda v: np.cos(np.pi * v**2), -0.5, 0.5)[0]
             + 1j * integrate.quad(lambda v: np.sin(np.pi * v**2), -0.5, 0.5)[0])
# total overlap of a cut linear piece of total length W*T_lin, in units of T_lin:
# |int_{-W/2}^{W/2} exp(i pi/3 fddot s^3) ds| / T_lin = int_{-W}^{W} cos(pi w^3/4) dw / 2 with w = 2 s / T_lin
ov = lambda W: integrate.quad(lambda w: np.cos(np.pi * w**3 / 4), 0, W, epsabs=1e-14)[0]  # noqa: E731
Wb = optimize.minimize_scalar(lambda W: -ov(W), bounds=(0.5, 3.0), method="bounded", options={"xatol": 1e-10}).x
from scipy.special import gamma  # noqa: E402

inf_formula = gamma(4 / 3) * np.cos(np.pi / 6) * (np.pi / 4) ** (-1 / 3)
inf_quad = integrate.quad(lambda w: np.cos(np.pi * w**3 / 4), 0, np.inf, weight=None, limit=2000)[0] \
    if False else float(np.trapz(np.cos(np.pi * np.linspace(0, 60, 6_000_001) ** 3 / 4) *
                                 np.exp(-(np.linspace(0, 60, 6_000_001) / 50.0) ** 2), np.linspace(0, 60, 6_000_001)))
res["kept"] = {"lin_piece": k_lin, "tone_piece": k_tone, "lin_best_total_overlap_T_lin": ov(Wb),
               "lin_best_length_T_lin": Wb, "lin_best_length_formula": 2 ** (1 / 3),
               "lin_infinite_overlap_formula": inf_formula, "lin_infinite_overlap_damped_trapz": inf_quad}
print(f"kept: linear piece {k_lin:.4f}, steady piece {k_tone:.4f}; linear piece of any length: at most "
      f"{ov(Wb):.4f} T_lin, at length {Wb:.4f} T_lin (2^(1/3) = {2 ** (1 / 3):.4f}); infinitely long "
      f"{inf_formula:.4f} (damped trapz {inf_quad:.4f})")
json.dump(res, open(OUT, "w"), indent=1)
print(f"done in {time.time() - T0:.1f} s")
