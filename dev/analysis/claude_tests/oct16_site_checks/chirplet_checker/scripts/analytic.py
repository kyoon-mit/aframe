"""Adversarial re-derivation of the chirplet in-step time and piece counts.

Written separately from the drafter's code: root finding (brentq) for the
in-step interval, closed forms derived again by hand, quad in tau with
explicit splitting for the counts.
"""
import os
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[v] = "1"
import json, sys
import numpy as np
from scipy import integrate, optimize

G = 6.67430e-11
MSUN = 1.98840987e30  # kg; GM_sun = 1.32712440e20 (IAU nominal) used below instead
GMSUN = 1.3271244e20
C = 299792458.0
TS = GMSUN / C**3
T = 4.0


def f_tau(tau, mc):
    # tau = 5/256 (TS mc)^(-5/3) (pi f)^(-8/3)
    return (5.0 / 256.0 / tau) ** (3.0 / 8.0) * (TS * mc) ** (-5.0 / 8.0) / np.pi


def tau_f(f, mc):
    return 5.0 / 256.0 * (TS * mc) ** (-5.0 / 3.0) * (np.pi * f) ** (-8.0 / 3.0)


def Phi(t, mc, tc=0.0):
    # phase as a function of time t < tc: integral of 2 pi f dt
    tau = tc - t
    # int 2 pi f(tau) dt = - int 2 pi f dtau ; f = A tau^(-3/8) -> int = A tau^(5/8)/(5/8)
    A = (5.0 / 256.0) ** (3.0 / 8.0) * (TS * mc) ** (-5.0 / 8.0) / np.pi
    return -2.0 * np.pi * A * tau ** (5.0 / 8.0) / (5.0 / 8.0)


def derivs(tau, mc):
    f = f_tau(tau, mc)
    fd = 3.0 / 8.0 * f / tau
    fdd = 3.0 / 8.0 * 11.0 / 8.0 * f / tau**2
    return f, fd, fdd


out = {}
# in-step intervals by root finding on the exact phase difference
rows = []
for mc in (0.85, 1.2, 2.75):
    for f0 in (50.0, 100.0, 300.0):
        tau0 = tau_f(f0, mc)
        t0 = -tau0
        f, fd, fdd = derivs(tau0, mc)
        P0 = Phi(t0, mc)

        def dlin(s):
            return Phi(t0 + s, mc) - (P0 + 2 * np.pi * f * s + np.pi * fd * s**2)

        def dstd(s):
            return Phi(t0 + s, mc) - (P0 + 2 * np.pi * f * s)

        Tl = (6.0 / fdd) ** (1.0 / 3.0)
        Ts = fd ** -0.5
        res = {}
        for name, fun, Tg in (("lin", dlin, Tl), ("std", dstd, Ts)):
            # right edge: s in (0, tau0)
            g = lambda s: abs(fun(s)) - np.pi / 4  # noqa
            # bracket from 1e-6 Tg upward
            sR = None
            grid = np.linspace(1e-6 * Tg, min(3 * Tg, tau0 * (1 - 1e-12)), 200001)
            vals = np.array([g(s) for s in grid[::100]])
            idx = np.flatnonzero(vals > 0)
            if idx.size:
                j = idx[0]
                a, b = grid[::100][j - 1], grid[::100][j]
                sR = optimize.brentq(g, a, b, xtol=1e-15)
            grid2 = -np.linspace(1e-6 * Tg, 3 * Tg, 200001)
            vals2 = np.array([g(s) for s in grid2[::100]])
            idx2 = np.flatnonzero(vals2 > 0)
            j = idx2[0]
            a, b = grid2[::100][j - 1], grid2[::100][j]
            sL = optimize.brentq(g, b, a, xtol=1e-15)
            L = (sR if sR is not None else tau0) - sL
            res[name] = (L, L / Tg - 1, sL, sR)
        rows.append(dict(mc=mc, f=f0, tau=tau0, fdot=fd, fddot=fdd, Tlin=Tl, Tstep=Ts,
                         ratio=Tl / Ts, ratio_formula=(18 / 11 * f0 * Ts) ** (1 / 3),
                         lin_exact=res["lin"][0], lin_rel=res["lin"][1], lin_edges=res["lin"][2:],
                         std_exact=res["std"][0], std_rel=res["std"][1]))
        print(f"Mc {mc} f {f0}: fddot {fdd:.4g} Hz/s2  Tlin {Tl:.4g} s  Tstep {Ts:.4g}  ratio {Tl/Ts:.3f} "
              f"| exact lin {res['lin'][0]:.5g} ({100*res['lin'][1]:+.3f}%), edges {res['lin'][2]:.4g},{res['lin'][3]} "
              f"| exact std {res['std'][0]:.5g} ({100*res['std'][1]:+.3f}%)")
out["instep"] = rows

# counts
crow = []
for mc in (0.85, 1.2, 1.8, 2.75):
    fT = f_tau(T, mc)
    N = integrate.quad(lambda tau: f_tau(tau, mc), 0, T, limit=500, points=[1e-6, 1e-3])[0]
    Nform = 1.6 * T * fT
    # linear pieces: int (fddot/6)^(1/3) dtau, singular as tau^(-19/24) at 0: split
    lin = lambda tau: (derivs(tau, mc)[2] / 6.0) ** (1.0 / 3.0)  # noqa
    std = lambda tau: derivs(tau, mc)[1] ** 0.5  # noqa
    # use substitution tau = x^k with k chosen so integrand is finite
    def qsub(fun, lo, hi, k):
        return integrate.quad(lambda x: fun(x**k) * k * x ** (k - 1), lo ** (1 / k), hi ** (1 / k),
                              limit=500, epsabs=0, epsrel=1e-13)[0]
    L_m = qsub(lin, 0.0, T, 24.0 / 5.0)
    S_m = qsub(std, 0.0, T, 16.0 / 5.0)
    L_form = 0.6 * (55.0 / 2.0 * Nform) ** (1 / 3)
    S_form = 0.4 * np.sqrt(15 * Nform)
    r = dict(mc=mc, fT=fT, N=N, Nform=Nform, lin_merger=L_m, lin_form=L_form, std_merger=S_m, std_form=S_form,
             itdf_merger=integrate.quad(lambda f: tau_f(f, mc), fT, np.inf, limit=500)[0])
    for fmax in (1024.0, 900.0, 512.0):
        tl = tau_f(fmax, mc)
        r[f"lin_{int(fmax)}"] = integrate.quad(lin, tl, T, limit=500, epsrel=1e-13)[0]
        r[f"lin_{int(fmax)}_form"] = L_form * (1 - (fT / fmax) ** (5 / 9))
        r[f"std_{int(fmax)}"] = integrate.quad(std, tl, T, limit=500, epsrel=1e-13)[0]
        r[f"std_{int(fmax)}_form"] = S_form * (1 - (fT / fmax) ** (5 / 6))
        r[f"itdf_{int(fmax)}"] = integrate.quad(lambda f: tau_f(f, mc), fT, fmax, limit=500)[0]
        r[f"N_{int(fmax)}"] = integrate.quad(lambda tau: f_tau(tau, mc), tl, T, limit=500)[0]
    crow.append(r)
    print(f"Mc {mc}: fT {fT:.3f} N {N:.2f} ({Nform:.2f}); lin merger {L_m:.4f} ({L_form:.4f}) "
          f"1024 {r['lin_1024']:.4f} ({r['lin_1024_form']:.4f}) 900 {r['lin_900']:.4f} 512 {r['lin_512']:.4f} ({r['lin_512_form']:.4f}) | "
          f"std merger {S_m:.4f} ({S_form:.4f}) 1024 {r['std_1024']:.4f} 900 {r['std_900']:.4f} 512 {r['std_512']:.4f} | "
          f"itdf merger {r['itdf_merger']:.3f} 1024 {r['itdf_1024']:.3f} 900 {r['itdf_900']:.3f}")
out["counts"] = crow
print("coef lin", 0.6 * 27.5 ** (1 / 3), "coef std", 0.4 * 15 ** 0.5)
# kept fractions
kl = integrate.quad(lambda w: np.cos(np.pi * w**3 / 4), 0, 1, epsabs=1e-15)[0]
# direct: mean over symmetric interval of exp(i pi w^3/4)
ws = np.linspace(-1, 1, 2000001)
kl2 = abs(np.trapz(np.exp(1j * np.pi * ws**3 / 4), ws) / 2)
ks = abs(integrate.quad(lambda v: np.cos(np.pi * v * v), -.5, .5)[0] + 1j * integrate.quad(lambda v: np.sin(np.pi * v * v), -.5, .5)[0])
print("kept linear", kl, kl2, "kept steady", ks)
out["kept"] = dict(lin=kl, lin_trapz=kl2, std=ks)
json.dump(out, open(sys.argv[1], "w"), indent=1, default=float)
