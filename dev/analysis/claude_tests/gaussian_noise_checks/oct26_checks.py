"""Numerical checks of the derivations behind the 26 October update.

1. Subtract-then-filter: for Gaussian noise with covariance R and the
   prediction-error filter L of the Levinson-Durbin recursion (errors E_t),
   h^T R^-1 d = sum_t (L h)_t (L d)_t / E_t.
2. Predictable share: for an AR(2) series, the order-1 prediction error is
   r_0 (1 - rho_1^2); a long Levinson-Durbin predictor and the
   Kolmogorov-Szego formula exp(mean log S) both give the innovation
   variance.
3. Threshold from noise draws: the spread of the empirical 1 - alpha quantile
   of n unit Gaussian draws, against sqrt(alpha (1 - alpha) / n) / phi(z),
   and the spread it causes in the share found at separation mu.
4. Repeats: E[ceil(N / R)] for N ~ Binomial(128, 0.96), against simulation.

usage: oct26_checks.py
"""

import numpy as np
from scipy.linalg import toeplitz
from scipy.stats import binom, norm

rng = np.random.default_rng(20261026)


def levinson(r, order):
    """Prediction coefficients a (x_t ~ sum_j a_j x_{t-j}) of every order up
    to ``order`` and their mean squared errors E_0..E_order."""
    a, E, coeffs = np.zeros(0), [r[0]], [np.zeros(0)]
    for m in range(1, order + 1):
        k = (r[m] - a @ r[m - 1:0:-1]) / E[-1]
        a = np.concatenate([a - k * a[::-1], [k]])
        E.append(E[-1] * (1 - k * k))
        coeffs.append(a)
    return coeffs, np.array(E)


def ar2_autocov(phi1, phi2, n):
    """Autocovariance of x_t = phi1 x_{t-1} + phi2 x_{t-2} + e_t, Var e = 1."""
    r1_over_r0 = phi1 / (1 - phi2)
    r0 = 1 / (1 - phi1 * r1_over_r0 - phi2 * (phi1 * r1_over_r0 + phi2))
    r = [r0, r0 * r1_over_r0]
    for _ in range(2, n):
        r.append(phi1 * r[-1] + phi2 * r[-2])
    return np.array(r)


def check_subtract_then_filter():
    n = 256
    t = np.arange(n)
    r = 0.9 ** t * np.cos(0.3 * t) + 0.5 * (t == 0)
    R = toeplitz(r)
    coeffs, E = levinson(r, n - 1)
    L = np.eye(n)
    for i in range(1, n):
        L[i, i - len(coeffs[i]):i] = -coeffs[i][::-1]
    h = np.sin(0.002 * t ** 2) * np.exp(-((t - 180) / 60.0) ** 2)
    d = np.linalg.cholesky(R) @ rng.standard_normal(n) + h
    direct = h @ np.linalg.solve(R, d)
    filtered = np.sum((L @ h) * (L @ d) / E)
    print(f"1. h^T R^-1 d = {direct:.10f}; sum (Lh)(Ld)/E = {filtered:.10f};"
          f" difference {abs(direct - filtered):.1e}")


def check_predictable_share():
    phi1, phi2 = 1.2, -0.5
    r = ar2_autocov(phi1, phi2, 2049)
    _, E = levinson(r, 2048)
    rho1 = r[1] / r[0]
    f = (np.arange(1 << 16) + 0.5) / (1 << 16) - 0.5
    S = 1 / np.abs(1 - phi1 * np.exp(-2j * np.pi * f) - phi2 * np.exp(-4j * np.pi * f)) ** 2
    print(f"2. AR(2) phi = ({phi1}, {phi2}): r_0 = {r[0]:.6f}, mean S = {S.mean():.6f};"
          f" order-1 error / r_0 = {E[1] / r[0]:.6f}, 1 - rho_1^2 = {1 - rho1 ** 2:.6f};"
          f" order-2048 error {E[-1]:.6f}, exp(mean log S) = {np.exp(np.log(S).mean()):.6f},"
          f" innovation variance 1; predictable share {1 - E[-1] / r[0]:.6f}")


def check_threshold_spread():
    alpha, n, trials, mu = 1e-3, 19_581, 4000, 4.2
    z = norm.isf(alpha)
    thresholds = np.array([np.quantile(rng.standard_normal(n), 1 - alpha) for _ in range(trials)])
    found = norm.sf(thresholds - mu)
    predicted = np.sqrt(alpha * (1 - alpha) / n) / norm.pdf(z)
    print(f"3. alpha {alpha}, {n} draws: z = {z:.4f}; threshold mean {thresholds.mean():.4f},"
          f" spread {thresholds.std():.4f} (formula {predicted:.4f}); share found at"
          f" separation {mu}: {norm.sf(z - mu):.4f}, spread {found.std():.4f}"
          f" (formula {norm.pdf(z - mu) * predicted:.4f})")


def check_repeats():
    k = np.arange(129)
    p = binom.pmf(k, 128, 0.96)
    for R in (4, 10):
        exact = (p * np.ceil(k / R)).sum()
        sim = np.ceil(rng.binomial(128, 0.96, 200_000) / R).mean()
        print(f"4. R = {R}: E[ceil(N / R)] = {exact:.4f}, simulated {sim:.4f}")


if __name__ == "__main__":
    check_subtract_then_filter()
    check_predictable_share()
    check_threshold_spread()
    check_repeats()
