"""Numerical checks of the 23 October derivations
(oct23-gaussian-chain-rule, oct23-known-template).

usage: check_formulas.py
"""

import math
import sys

import numpy as np
from scipy.stats import multivariate_normal

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import test1_likelihood  # noqa: E402

test1_likelihood.dev = "cpu"

rng = np.random.default_rng(0)
n = 256

# a positive spectrum on 2n points with a line, its autocovariance lags 0..n-1
f = np.fft.rfftfreq(2 * n)
S = 0.05 + 1 / (1 + (f / 0.05) ** 4) + 30 * np.exp(-0.5 * ((f - 0.3) / 0.004) ** 2)
r = np.fft.irfft(S, n=2 * n)[:n]
R = r[np.abs(np.arange(n)[:, None] - np.arange(n)[None])]
print("smallest eigenvalue of R:", np.linalg.eigvalsh(R).min())

# 1. chain rule with order-t predictors = multivariate Gaussian log density
preds, full_e = test1_likelihood.levinson(r, (n - 1,))
x = rng.multivariate_normal(np.zeros(n), R, size=4)
L, E = (v.numpy() for v in preds[n - 1])
e = x @ L.T
chain = -(0.5 * e**2 / E + 0.5 * np.log(E) + 0.5 * math.log(2 * math.pi)).sum(1)
exact = multivariate_normal(np.zeros(n), R).logpdf(x)
print("1. order n-1 chain rule minus exact log density:", np.abs(chain - exact).max())
print("   sum log E_t minus log det R:", np.log(full_e).sum() - np.linalg.slogdet(R)[1])

# 2. zero-padded periodogram -> biased autocovariance
y = rng.standard_normal(n)
lag = np.fft.irfft(np.abs(np.fft.rfft(y, n=2 * n)) ** 2 / n, n=2 * n)[:n]
direct = np.array([(y[: n - k] * y[k:]).sum() / n for k in range(n)])
print("2. inverse FFT of padded periodogram minus (1/N) sum x_t x_t+k:", np.abs(lag - direct).max())

# 3. Gaussian likelihood ratio with a known template
h = 0.3 * np.sin(np.linspace(0, 40, n) ** 1.3) * np.linspace(0, 1, n)
Ri = np.linalg.inv(R)
q = h @ Ri @ h
noise = rng.multivariate_normal(np.zeros(n), R, size=200_000)
lr_noise = noise @ Ri @ h - 0.5 * q
lr_signal = (noise + h) @ Ri @ h - 0.5 * q
print(f"3. q {q:.4f}: noise mean {lr_noise.mean():.4f} (-q/2 {-q / 2:.4f}), "
      f"variance {lr_noise.var():.4f}; signal mean {lr_signal.mean():.4f}; "
      f"separation {(lr_signal.mean() - lr_noise.mean()) / lr_noise.std():.4f} "
      f"(sqrt q {math.sqrt(q):.4f})")
mf = noise @ h / np.linalg.norm(h)
print(f"   matched filter on this noise: spread {mf.std():.4f} "
      f"(sqrt(h R h)/|h| {math.sqrt(h @ R @ h) / np.linalg.norm(h):.4f})")

# 4, 5. constants
print("4. 0.5 ln 2pi + 0.5 =", 0.5 * math.log(2 * math.pi) + 0.5)
print("5. (141852 - 2 * 41841) s =", 141852 - 2 * 41841, "s =", (141852 - 2 * 41841) / 86400, "days;",
      "141852 - 41841 =", 141852 - 41841, "s =", (141852 - 41841) / 86400, "days")
print("   2048 samples per second: 128 samples =", 128 / 2048 * 1000, "ms; 38493 s =", 38493 / 3600, "h")
print("   q from noise mean -7.82:", 2 * 7.82, "sqrt", math.sqrt(2 * 7.82))
