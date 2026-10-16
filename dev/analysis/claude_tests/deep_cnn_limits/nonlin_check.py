"""Check of derivations #semi-coherent: does a nonlinearity applied to the
pieces before adding turn the sum into a power sum?

For z = mu + n, n ~ N(0, 1) and small mu:
    E g(mu + n) = m0 + s1 mu + s2 mu^2 / 2 + ...,
    s1 = E g'(n) = E[n g(n)],  s2 = E g''(n) = E[(n^2 - 1) g(n)]   (Stein).
Three ways of adding N pieces of total SNR rho = 8 (N = 600):
  (i)   in step: z_k = rho_k + n_k, S = sum g(z_k)
  (ii)  oscillating signal, plain sum (time average): z_t = a cos(w t + p) + n_t
  (iii) oscillating signal, weights that follow the phase: S = sum cos(w t + p) g(z_t)
Deflection d = (E S|signal - E S|noise) / sd(S|noise).
Predictions: (i) rho s1/sqrt(v0) + rho^2 s2 / (2 sqrt(N v0)); (ii) rho^2 s2 / (2 sqrt(N v0));
(iii) rho s1 / sqrt(v0).
"""
import json

import numpy as np
from scipy import integrate, special

OUT = "/tmp/claude-1024/-home-kyoon-SSM-BNS/82effc75-bb98-44ad-af20-e41e39a7258d/scratchpad/cnnproof/nonlin_results.json"
rng = np.random.default_rng(7)
N, RHO, TRIALS, CH = 600, 8.0, 20000, 2000
W = 2 * np.pi * 0.1234567

G = {
    "identity": lambda x: x,
    "square": lambda x: x * x,
    "abs": np.abs,
    "relu": lambda x: np.maximum(x, 0),
    "gelu": lambda x: x * 0.5 * (1 + special.erf(x / np.sqrt(2))),
    "sign": np.sign,
    "step at 0": lambda x: np.asarray(x > 0, float),
    "step at 1": lambda x: np.asarray(x > 1, float),
    "two-sided threshold |x|>1": lambda x: np.asarray(np.abs(x) > 1, float),
}


def gauss_moments(g):
    phi = lambda x: np.exp(-x * x / 2) / np.sqrt(2 * np.pi)
    pts = [-1.0, 0.0, 1.0]
    E = lambda h: integrate.quad(lambda x: h(x) * phi(x), -12, 12, points=pts, limit=400)[0]
    m0 = E(g)
    v0 = E(lambda x: g(x) ** 2) - m0 ** 2
    s1 = E(lambda x: x * g(x))
    s2 = E(lambda x: (x * x - 1) * g(x))
    return m0, v0, s1, s2


def deflection(g, case):
    t = np.arange(N)
    out = {"noise": [], "signal": []}
    for kind in ("noise", "signal"):
        for _ in range(TRIALS // CH):
            n = rng.standard_normal((CH, N))
            if case == "in step":
                z = n + (RHO / np.sqrt(N) if kind == "signal" else 0.0)
                S = g(z).sum(1)
            else:
                p = rng.uniform(0, 2 * np.pi, (CH, 1))
                c = np.cos(W * t[None, :] + p)
                a = RHO * np.sqrt(2.0 / N)
                z = n + (a * c if kind == "signal" else 0.0)
                S = g(z).sum(1) if case == "plain sum" else (c * g(z)).sum(1)
            out[kind].append(S)
    sn, ss = np.concatenate(out["noise"]), np.concatenate(out["signal"])
    d = (ss.mean() - sn.mean()) / sn.std()
    se = np.sqrt(ss.var() + sn.var()) / np.sqrt(TRIALS) / sn.std()
    return d, se


def main():
    R = {}
    print(f"N = {N}, rho = {RHO}, {TRIALS} trials per arm")
    for name, g in G.items():
        m0, v0, s1, s2 = gauss_moments(g)
        pred = {"in step": RHO * s1 / np.sqrt(v0) + RHO ** 2 * s2 / (2 * np.sqrt(N * v0)),
                "plain sum": RHO ** 2 * s2 / (2 * np.sqrt(N * v0)),
                "phase weights": RHO * s1 / np.sqrt(v0)}
        row = dict(m0=m0, v0=v0, s1=s1, s2=s2, eff_linear=s1 ** 2 / v0 if v0 > 0 else None)
        for case in ("in step", "plain sum", "phase weights"):
            d, se = deflection(g, case)
            row[case] = dict(mc=d, se=se, pred=pred[case])
        R[name] = row
        print(f"{name:26s} s1={s1:+.4f} s2={s2:+.4f} var={v0:.4f} eff={s1 ** 2 / v0:.4f} | "
              + " | ".join(f"{c}: MC {row[c]['mc']:.3f}+-{row[c]['se']:.3f} pred {row[c]['pred']:.3f}"
                           for c in ("in step", "plain sum", "phase weights")))
    json.dump(R, open(OUT, "w"), indent=1, default=float)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
