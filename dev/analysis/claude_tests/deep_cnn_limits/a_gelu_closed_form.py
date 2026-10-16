"""Closed form of Var GELU(n), n ~ N(0, sigma^2), against quadrature.

E[n Phi(n)]       = sigma^2 / sqrt(2 pi (1 + sigma^2))
E[n^2 Phi(n)^2]   = sigma^2/(1+2 sigma^2) (1/4 + a/(2 pi))
                  + sigma^4/(1+2 sigma^2) (1/2 + a/pi + sqrt(1+2 sigma^2)/(pi (1+sigma^2)))
with a = arcsin(sigma^2 / (1 + sigma^2)).  r = (sigma/2) / std GELU(n).
"""
import numpy as np
from scipy import integrate, special

def quad_moments(sigma):
    f = lambda x, p: (x * special.ndtr(x)) ** p * np.exp(-0.5 * (x / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
    m1 = integrate.quad(lambda x: f(x, 1), -14 * sigma, 14 * sigma, epsabs=1e-15, epsrel=1e-12, limit=400)[0]
    m2 = integrate.quad(lambda x: f(x, 2), -14 * sigma, 14 * sigma, epsabs=1e-15, epsrel=1e-12, limit=400)[0]
    return m1, m2

def closed(sigma):
    s2 = sigma ** 2
    a = np.arcsin(s2 / (1 + s2))
    m1 = s2 / np.sqrt(2 * np.pi * (1 + s2))
    m2 = s2 / (1 + 2 * s2) * (0.25 + a / (2 * np.pi)) + s2 ** 2 / (1 + 2 * s2) * (0.5 + a / np.pi + np.sqrt(1 + 2 * s2) / (np.pi * (1 + s2)))
    return m1, m2

lines = []
for sigma in (0.25, 0.5, 1.0, 2.0, 4.0, 10.0):
    q1, q2 = quad_moments(sigma)
    c1, c2 = closed(sigma)
    rq = (sigma / 2) / np.sqrt(q2 - q1 ** 2)
    rc = (sigma / 2) / np.sqrt(c2 - c1 ** 2)
    lines.append(f"sigma {sigma:5.2f}: E GELU quad {q1:.12f} closed {c1:.12f}; E GELU^2 quad {q2:.12f} closed {c2:.12f}; r quad {rq:.6f} closed {rc:.6f}")
lines.append(f"ReLU limit r = {0.5/np.sqrt(0.5 - 1/(2*np.pi)):.6f}")
print("\n".join(lines))
open("out/a_gelu_closed_form.txt", "w").write("\n".join(lines) + "\n")
