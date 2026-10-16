"""Diagnostic: in-step and equal-weight mean shifts against theory, using the exact noise spread."""
import numpy as np
from scipy import special, integrate
Phi = special.ndtr
K, FS, F0, RHO, T, CH = 8192, 2048.0, 101.3, 8.0, 8000, 250
t = np.arange(K) / FS
def E(f, s):
    p = lambda x: np.exp(-0.5 * (x / s) ** 2) / (s * np.sqrt(2 * np.pi))
    return sum(integrate.quad(lambda x: f(x) * p(x), a, b, epsabs=1e-14, epsrel=1e-13, limit=400)[0]
               for a, b in ((-14 * s, 0), (0, 14 * s)))
for seed in (161016, 7):
    rng = np.random.default_rng(seed)
    for name, phi, s in (("GELU", lambda x: x * Phi(x), 1.0), ("square", lambda x: x * x, 1.0)):
        m0 = E(phi, s); sd = np.sqrt(E(lambda x: phi(x) ** 2, s) - m0 ** 2)
        g = E(lambda x: x * phi(x), s) / s ** 2
        h = E(lambda x: (x * x - s * s) * phi(x), s) / s ** 4
        din, deq, o0in, o0eq = [], [], [], []
        for _ in range(T // CH):
            th = rng.uniform(0, 2 * np.pi, (CH, 1))
            sig = np.cos(2 * np.pi * F0 * t[None, :] + th); sig *= RHO * s / np.linalg.norm(sig, axis=1, keepdims=True)
            c = sig / np.linalg.norm(sig, axis=1, keepdims=True)
            n = s * rng.standard_normal((CH, K)); f0 = phi(n); df = phi(n + sig) - f0
            din.append((df * c).sum(1)); deq.append(df.sum(1) / np.sqrt(K))
            o0in.append((f0 * c).sum(1)); o0eq.append(f0.sum(1) / np.sqrt(K))
        din, deq, o0in, o0eq = map(np.concatenate, (din, deq, o0in, o0eq))
        pin, peq = g * RHO * s, 0.5 * h * (RHO * s) ** 2 / np.sqrt(K)
        print(f"seed {seed} {name}: in-step shift {din.mean():.5f} +- {din.std()/np.sqrt(T):.5f} (theory {pin:.5f});"
              f" equal shift {deq.mean():.6f} +- {deq.std()/np.sqrt(T):.6f} (theory {peq:.6f});"
              f" sd o0 in {o0in.std(ddof=1):.5f}, eq {o0eq.std(ddof=1):.5f} (exact {sd:.5f}, se {sd/np.sqrt(2*T):.5f})")
