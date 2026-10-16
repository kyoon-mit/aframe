"""Section 3 checks: Newtonian delay integrals, the phase-budget bound for a
sum of N damped resonators, the all-pass construction that reaches it, and
the counts for chirp masses 0.85, 1.2, 1.8, 2.75 Msun in a 4 s window."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from numpy.polynomial import polynomial as P
from scipy.integrate import quad

from common import (MASSES, RES, cycles, f_of_tau, int_tau_df, kernel_of, phi_of_tau,
                    tau_of_f, toy_chirp)

out = {}

# ---------------------------------------------------------------- A. integrals
A = {}
for mc in MASSES + (1.0, 1.4, 2.4):
    f1 = float(f_of_tau(4.0, mc))
    row = {"f1_Hz": f1}
    for name, f2 in (("to_900Hz", 900.0), ("to_merger", np.inf)):
        num_int = quad(lambda f: tau_of_f(f, mc), f1, f2, limit=200)[0]
        if np.isinf(f2):
            closed = 0.6 * tau_of_f(f1, mc) * f1
            cyc_closed = 1.6 * tau_of_f(f1, mc) * f1
            cyc_num = abs(phi_of_tau(4.0, mc)) / (2 * np.pi)
        else:
            closed = int_tau_df(f1, f2, mc)
            cyc_closed = cycles(f1, f2, mc)
            cyc_num = (abs(phi_of_tau(4.0, mc)) - abs(phi_of_tau(tau_of_f(f2, mc), mc))) / (2 * np.pi)
        row[name] = {"int_tau_df_quad": num_int, "int_tau_df_closed": float(closed),
                     "cycles_from_phase": float(cyc_num), "cycles_closed": float(cyc_closed),
                     "ratio_int_over_cycles": float(num_int / cyc_num)}
    A[str(mc)] = row
    print("A", mc, json.dumps(row))
out["A_integrals"] = A
out["A_two_pi_times_3_over_8"] = 2 * np.pi * 3 / 8
out["A_five_pi_over_4"] = 5 * np.pi / 4
print("2pi*3/8 =", 2 * np.pi * 3 / 8, " 5pi/4 =", 5 * np.pi / 4)


# ---------------------------------------------------------------- B. phase budget
def channel_poly(a, w, C, D):
    """Numerator and denominator (in s, ascending powers) of
    H(s) = D + sum_n [C_n/(s - p_n) + conj(C_n)/(s - conj p_n)], p_n = -a_n + i w_n."""
    poles = np.concatenate([-a + 1j * w, -a - 1j * w])
    res = np.concatenate([C, np.conj(C)])
    den = P.polyfromroots(poles)
    num = D * den
    for k in range(poles.size):
        num = P.polyadd(num, res[k] * P.polyfromroots(np.delete(poles, k)))
    return num, den, poles


def phase_on(num, den, om):
    s = 1j * om
    return np.unwrap(np.angle(P.polyval(s, num) / P.polyval(s, den)))


def omega_grid(w, a):
    base = [np.linspace(0, 1.5 * w.max() + 50 * a.max(), 3_000_001)]
    base.append(np.logspace(np.log10(1.5 * w.max() + 50 * a.max()), 8, 200_001))
    return np.unique(np.concatenate(base))


rng = np.random.default_rng(7)
Brows = []
worst_ratio = 0.0
for trial in range(60):
    N = int(rng.integers(2, 9))
    a = np.exp(rng.uniform(np.log(2.0), np.log(60.0), N))
    w = 2 * np.pi * rng.uniform(20.0, 400.0, N)
    C = rng.standard_normal(N) + 1j * rng.standard_normal(N)
    if trial % 3 == 0:
        C *= np.exp(1j * rng.uniform(0, 2 * np.pi, N)) * rng.uniform(0.1, 10, N)
    D = 0.0 if trial % 2 else float(rng.standard_normal() * 5)
    num, den, poles = channel_poly(a, w, C, D)
    num = np.real_if_close(num, tol=1e6)
    zeros = P.polyroots(num)
    z_rhp = int(np.sum(zeros.real > 0))
    z_lhp = int(np.sum(zeros.real < 0))
    om = omega_grid(w, a)
    th = phase_on(num, den, om)
    lag_total = float(th[0] - th[-1])                  # phase dropped over (0, 1e8)
    pred_total = 0.5 * np.pi * (poles.size + z_rhp - z_lhp)
    run_max = np.maximum.accumulate(th)
    max_drop = float(np.max(run_max - th))             # largest drop over any band
    bound = np.pi * (N + 0.5 * z_rhp)
    worst_ratio = max(worst_ratio, max_drop / (2 * np.pi * N))
    Brows.append({"N": N, "D": D, "zeros": int(zeros.size), "z_rhp": z_rhp, "z_lhp": z_lhp,
                  "lag_total_over_pi": lag_total / np.pi, "pred_total_over_pi": pred_total / np.pi,
                  "max_band_drop_over_2piN": max_drop / (2 * np.pi * N),
                  "max_band_drop_le_pi_N_plus_half_zrhp": bool(max_drop <= bound + 1e-6)})
out["B_random_channels"] = Brows
out["B_worst_max_drop_over_2piN"] = worst_ratio
out["B_all_within_bound"] = all(r["max_band_drop_le_pi_N_plus_half_zrhp"] for r in Brows)
out["B_total_formula_max_abs_err_over_pi"] = max(abs(r["lag_total_over_pi"] - r["pred_total_over_pi"]) for r in Brows)
print("B random channels: worst drop/(2 pi N) =", worst_ratio,
      " all within pi(N + Z_rhp/2):", out["B_all_within_bound"],
      " total-lag formula max err/pi:", out["B_total_formula_max_abs_err_over_pi"])

# all-pass with N sections, written as an S4D channel (D + parallel sum)
allpass = []
for N in (3, 6, 10):
    fc = np.linspace(100, 300, N)
    a = np.full(N, 2 * np.pi * (fc[1] - fc[0]) / 2)        # half-width = half the spacing
    w = 2 * np.pi * fc
    p = -a + 1j * w
    poles = np.concatenate([p, np.conj(p)])
    zer = -np.conj(poles)                                   # mirror images
    def Hap(s):
        return np.prod([(s - z) / (s - q) for z, q in zip(zer, poles)], axis=0)
    # residues of the all-pass at its poles; D = 1
    R = np.array([np.prod([(q - z) for z in zer]) / np.prod([(q - q2) for q2 in poles if q2 != q]) for q in poles])
    om = omega_grid(w, a)
    s = 1j * om
    Hpar = 1.0 + np.sum([R[k] / (s - poles[k]) for k in range(poles.size)], axis=0)
    err = float(np.max(np.abs(Hpar - Hap(s))))
    th = np.unwrap(np.angle(Hpar))
    allpass.append({"N": N, "parallel_vs_product_max_abs_err": err,
                    "max_abs_gain_minus_1": float(np.max(np.abs(np.abs(Hpar) - 1))),
                    "phase_drop_over_2piN": float((th[0] - th[-1]) / (2 * np.pi * N)),
                    "residues_conjugate_pairs": bool(np.allclose(R[N:], np.conj(R[:N])))})
    print("all-pass", allpass[-1])
out["B_allpass"] = allpass

# ---------------------------------------------------------------- C. counts
def fourier_count(K, m):
    p = np.sort(np.abs(np.fft.fft(K)) ** 2)[::-1]
    c = np.sqrt(np.cumsum(p) / p.sum())
    return int(np.searchsorted(c, m) + 1)


Crows = {}
for mc in (1.0, 1.4, 1.8, 2.4) + MASSES:
    K = kernel_of(toy_chirp(mc))
    f1 = float(f_of_tau(4.0, mc))
    I900 = float(int_tau_df(f1, 900.0, mc))
    Iinf = float(0.6 * tau_of_f(f1, mc) * f1)
    row = {"f1_Hz": f1, "cycles_to_merger": float(1.6 * tau_of_f(f1, mc) * f1),
           "cycles_to_900": float(cycles(f1, 900.0, mc)),
           "int_tau_df_to_900": I900, "int_tau_df_to_merger": Iinf,
           "N_c1_to_900": I900, "N_ce_to_900": float(np.e * I900), "N_c2pi_to_900": float(2 * np.pi * I900),
           "fourier_full_band_4s_x_bw": float(4.0 * (900.0 - f1)),
           "fourier_count_0.9_toy": fourier_count(K, 0.9),
           "fourier_count_0.99_toy": fourier_count(K, 0.99)}
    Crows[str(mc)] = row
    print("C", mc, json.dumps({k: (round(v, 2) if isinstance(v, float) else v) for k, v in row.items()}))
out["C_counts"] = Crows
json.dump(out, open(os.path.join(RES, "c3_count.json"), "w"), indent=1)
print("saved")
