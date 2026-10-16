"""Independent quadrature of r = corr(n, phi(n)) and r2 = corr(phi(n), n^2 - s^2),
plus a small Monte Carlo of the in-step sum with same-sign pieces (K = 600)."""
import numpy as np
from scipy import integrate, optimize, special

SQ2 = np.sqrt(2)


def gauss_moments(phi, s, pts=()):
    pdf = lambda x: np.exp(-x * x / (2 * s * s)) / (s * np.sqrt(2 * np.pi))
    E = lambda g: sum(integrate.quad(lambda x: g(x) * pdf(x), a, b, limit=400, points=None)[0]
                      for a, b in zip((-np.inf,) + tuple(pts), tuple(pts) + (np.inf,)))
    m1 = E(phi)
    m2 = E(lambda x: phi(x) ** 2)
    xn = E(lambda x: x * phi(x))
    xq = E(lambda x: (x * x - s * s) * phi(x))
    sd = np.sqrt(m2 - m1 ** 2)
    return xn / (s * sd), xq / (np.sqrt(2) * s * s * sd), xn / s ** 2, sd


Phi = lambda x: 0.5 * (1 + special.erf(x / SQ2))
fns = {
    "relu": (lambda x: max(x, 0.0), (0.0,)),
    "sign": (lambda x: np.sign(x), (0.0,)),
    "step2": (lambda x: 1.0 if x > 2 else 0.0, (2.0,)),
    "square": (lambda x: x * x, ()),
    "abs": (lambda x: abs(x), (0.0,)),
    "two-sided>2": (lambda x: 1.0 if abs(x) > 2 else 0.0, (-2.0, 2.0)),
    "gelu": (lambda x: x * Phi(x), ()),
}
for name, (f, pts) in fns.items():
    sig = (0.5, 1.0, 2.0) if name == "gelu" else (1.0,)
    for s in sig:
        pp = tuple(p * s for p in pts) if name in ("step2", "two-sided>2") else pts
        r, r2, slope, sd = gauss_moments(f, s, pp)
        print(f"{name:12s} sigma={s}: r={r:.6f} r2={r2:.6f} mean slope={slope:.6f} sd={sd:.6f}")

rg = lambda s: -gauss_moments(fns["gelu"][0], s)[0]
res = optimize.minimize_scalar(lambda s: -rg(s), bounds=(0.5, 5), method="bounded")
print(f"GELU r minimum {-(-res.fun):.6f}? at sigma {res.x:.4f}")
print(f"GELU r min value {gauss_moments(fns['gelu'][0], res.x)[0]:.6f}")

# Monte Carlo: K pieces, equal signal a = rho/sqrt(K) each, same sign, equal weights
rng = np.random.default_rng(3)
K, rho, T = 600, 8.0, 4000
a = rho / np.sqrt(K)
acts = {"linear": lambda z: z, "relu": lambda z: np.maximum(z, 0), "gelu": lambda z: z * Phi(z),
        "sign": np.sign, "square": lambda z: z * z, "abs": np.abs}
for name, g in acts.items():
    s0, s1 = np.empty(T), np.empty(T)
    for t in range(T):
        n = rng.standard_normal(K)
        s0[t], s1[t] = g(n).sum(), g(n + a).sum()
    print(f"MC K={K} rho=8 same-sign {name:7s}: separation {(s1.mean() - s0.mean()) / s0.std():.3f}")
