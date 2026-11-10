"""Numerical checks of the derivations behind the 10 November update.

1. Entropy of a coin and a die, in nats and bits.
2. Entropy of unit-variance Gaussian, Laplace and uniform densities:
   closed forms against the average of -log p over draws, and the gap to
   the Gaussian (negentropy).
3. Cross-entropy: unit Gaussian draws scored by a Gaussian model of the
   wrong width give more than the entropy.
4. A series with memory, x_t = a x_{t-1} + e_t: variance, Kolmogorov's
   formula on a frequency grid, spectral flatness, the prediction error
   from the previous p samples (covariance matrix), and the same from a
   least-squares fit to a simulated series.
5. Nats per sample from the predictable shares of the 26 October update.

usage: nov10_checks.py
"""

import numpy as np
from scipy.linalg import toeplitz

rng = np.random.default_rng(20261110)
LN2 = np.log(2)


def H(p):
    p = np.asarray(p, float)
    return -(p * np.log(p)).sum()


print("1. discrete entropy (nats, bits)")
for name, p in [
    ("fair coin", [0.5, 0.5]),
    ("coin, heads 0.9", [0.9, 0.1]),
    ("fair die", [1 / 6] * 6),
]:
    print(f"   {name:16s} {H(p):.4f} {H(p) / LN2:.4f}")

print("2. unit-variance densities: closed form, -mean log p, gap to Gaussian")
n = 4_000_000
gauss = 0.5 * np.log(2 * np.pi * np.e)
x = rng.standard_normal(n)
mc = np.mean(0.5 * np.log(2 * np.pi) + 0.5 * x**2)
print(f"   Gaussian {gauss:.4f} {mc:.4f} {0:.4f}")
b = 1 / np.sqrt(2)
x = rng.laplace(0, b, n)
mc = np.mean(np.log(2 * b) + np.abs(x) / b)
h = 1 + np.log(2 * b)
print(f"   Laplace  {h:.4f} {mc:.4f} {gauss - h:.4f}  var {x.var():.4f}")
w = np.sqrt(12)
x = rng.uniform(-w / 2, w / 2, n)
h = np.log(w)
print(f"   uniform  {h:.4f} {h:.4f} {gauss - h:.4f}  var {x.var():.4f}")

print("3. unit Gaussian draws scored by a Gaussian of width 1.5")
s = 1.5
x = rng.standard_normal(n)
mc = np.mean(np.log(s) + 0.5 * np.log(2 * np.pi) + 0.5 * x**2 / s**2)
cf = np.log(s) + 0.5 * np.log(2 * np.pi) + 0.5 / s**2
print(f"   closed form {cf:.4f}, draws {mc:.4f}, entropy {gauss:.4f}")

print("4. x_t = a x_{t-1} + e_t, unit innovation variance")
for a in (0.5, 0.9, 0.99):
    r0 = 1 / (1 - a**2)
    f = (np.arange(1 << 20) + 0.5) / (1 << 20) - 0.5
    S = 1 / np.abs(1 - a * np.exp(-2j * np.pi * f)) ** 2
    kolm = np.exp(np.mean(np.log(S)))
    flat = kolm / np.mean(S)
    p = 50
    R = toeplitz(r0 * a ** np.arange(p + 1))
    Ep = np.exp(np.linalg.slogdet(R)[1] - np.linalg.slogdet(R[:p, :p])[1])
    N = 2_000_000
    e = rng.standard_normal(N)
    xs = np.empty(N)
    xs[0] = e[0] * np.sqrt(r0)
    for t in range(1, N):
        xs[t] = a * xs[t - 1] + e[t]
    q = 5
    A = np.stack([xs[q - j - 1 : N - j - 1] for j in range(q)], 1)
    coef, *_ = np.linalg.lstsq(A, xs[q:], rcond=None)
    res = np.var(xs[q:] - A @ coef)
    print(
        f"   a={a}: variance {r0:.4f} (mean S {np.mean(S):.4f}), "
        f"Kolmogorov {kolm:.6f}, flatness {flat:.4f} (1-a^2 {1 - a**2:.4f}), "
        f"error from 50 past {Ep:.6f}"
    )
    print(
        f"      simulated: variance {xs.var():.4f}, fit error {res:.4f}, "
        f"I = {0.5 * np.log(xs.var() / res):.4f} nats "
        f"(-0.5 ln(1-a^2) = {-0.5 * np.log(1 - a**2):.4f}, "
        f"{-0.5 * np.log(1 - a**2) / LN2:.4f} bits)"
    )

print("5. nats per sample from a predictable share s: -0.5 ln(1 - s)")
for name, sh in [
    ("H1, infinite past", 0.2473),
    ("L1, infinite past", 0.1228),
    ("H1, 20 Hz and above", 0.1719),
    ("L1, 20 Hz and above", 0.0403),
    ("raw strain H1, previous sample", 0.99953**2),
    ("raw strain L1, previous sample", 0.99967**2),
]:
    print(f"   {name:32s} share {sh:.4f}  {-0.5 * np.log(1 - sh):.4f}")
