"""Side checks: linear response kept by a nonlinearity before adding; stride-32 average of a tone."""
import numpy as np
from scipy import integrate, stats

phi, Phi = stats.norm.pdf, stats.norm.cdf
g = {
    "linear": (lambda x: x, lambda x: 1.0),
    "ReLU": (lambda x: np.maximum(x, 0), lambda x: float(x > 0)),
    "GELU": (lambda x: x * Phi(x), lambda x: Phi(x) + x * phi(x)),
    "threshold at 0 (sign)": (lambda x: np.sign(x), None),
    "threshold at 1": (lambda x: float(x > 1), None),
    "square": (lambda x: x * x, lambda x: 2 * x),
    "abs": (lambda x: abs(x), lambda x: np.sign(x)),
}
E = lambda f: integrate.quad(lambda x: f(x) * phi(x), -12, 12, limit=400, points=[0, 1])[0]
print("efficacy = E[n g(n)] / sd g(n)  (Stein: E[n g(n)] = E[g'(n)]); coherent SNR kept = efficacy * rho")
eff = {}
for name, (f, _) in g.items():
    slope = E(lambda x: x * f(x))
    var = E(lambda x: f(x) ** 2) - E(f) ** 2
    eff[name] = slope / np.sqrt(var)
    print(f"  {name:22s} mean slope {slope:+.4f}  sd {np.sqrt(var):.4f}  efficacy {eff[name]:.4f}")

# Monte Carlo: N = 600 in-step pieces at total SNR 8, statistic = sum of g(z_k)
rng = np.random.default_rng(0)
N, rho, T = 600, 8.0, 4000
rk = rho / np.sqrt(N)
print(f"Monte Carlo, N={N}, rho={rho}, {T} trials each: separation (mean_sig - mean_noise) / sd_noise")
for name, (f, _) in g.items():
    fv = np.vectorize(f) if name in ("threshold at 1",) else f
    n0 = rng.standard_normal((T, N)); n1 = rng.standard_normal((T, N))
    s0 = fv(n0).sum(1); s1 = fv(n1 + rk).sum(1)
    sep = (s1.mean() - s0.mean()) / s0.std()
    pred = eff[name] * rho if name != "square" else rho**2 / np.sqrt(2 * N)
    print(f"  {name:22s} measured {sep:6.2f}   predicted {pred:6.2f}" + ("  (rho^2/sqrt(2N))" if name == "square" else ""))

# stride-32 average of a tone over 256 feature times (j = 32 samples at 2048 Hz)
t = 32 * np.arange(256) / 2048
f = np.arange(20, 512.0001, 0.01)
ph = np.linspace(0, 2 * np.pi, 13, endpoint=False)
m = np.abs(np.cos(2 * np.pi * f[:, None, None] * t[None, :, None] + ph[None, None, :]).mean(1)).max(1)
near = np.min(np.abs(f[:, None] - 64 * np.arange(1, 9)[None]), 1)
print("stride-32 mean of cos over 256 feature times (4 s):")
print(f"  at multiples of 64 Hz: max |mean| {m[near < 1e-6].max():.4f}, min {m[near < 1e-6].min():.4f}")
for d in (0.25, 1.0, 5.0):
    print(f"  more than {d} Hz from a multiple of 64 Hz: max |mean| {m[near > d].max():.4f}")
print(f"  unstrided bound 1/(pi*20*4) = {1 / (np.pi * 20 * 4):.4f}")
