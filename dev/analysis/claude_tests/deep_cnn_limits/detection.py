"""SNR for 50% detection at false-alarm probability 1e-4 per trial when N
segment powers |z_k|^2 are added (2N degrees of freedom), against N = 1.

Each |z_k|^2 is noncentral chi-square with 2 degrees of freedom and
noncentrality rho_k^2 (unit-variance real and imaginary noise). The plain
sum is noncentral chi-square with 2N degrees of freedom and noncentrality
sum rho_k^2 = rho^2, whatever the split. Checked by Monte Carlo with the SPA
split. Also the sum weighted by each segment's expected share of SNR^2
(not requested; its distribution by characteristic-function inversion,
checked by Monte Carlo).
"""
import json

import numpy as np
from scipy import integrate, optimize, stats

from common import Noise
from fisher_seg import segmented

PFA, PD = 1e-4, 0.5
OUT = "/tmp/claude-1024/-home-kyoon-SSM-BNS/82effc75-bb98-44ad-af20-e41e39a7258d/scratchpad/cnnproof/detection_results.json"
rng = np.random.default_rng(20261015)


def rho50_plain(N):
    eta = stats.chi2.isf(PFA, 2 * N)
    rho = optimize.brentq(lambda r: stats.ncx2.sf(eta, 2 * N, r * r) - PD, 0.1, 50)
    return eta, rho


def sf_weighted(s, w, lam):
    """P(sum_k w_k X_k > s), X_k ~ ncx2(2, lam_k), by Gil-Pelaez."""
    w, lam = np.asarray(w, float), np.asarray(lam, float)

    def phi(t):
        d = 1 - 2j * w * t
        return np.prod(np.exp(1j * lam * w * t / d) / d)

    def integrand(t):
        return np.imag(np.exp(-1j * t * s) * phi(t)) / t

    val, _ = integrate.quad(integrand, 0, np.inf, limit=2000, epsabs=1e-12, epsrel=1e-10)
    return 0.5 + val / np.pi


def rho50_weighted(w):
    w = np.asarray(w, float)
    lo, hi = 0.0, 50.0 * w.sum() + 50
    eta = optimize.brentq(lambda s: sf_weighted(s, w, np.zeros_like(w)) - PFA, 1e-6, hi, xtol=1e-10)
    rho = optimize.brentq(lambda r: sf_weighted(eta, w, r * r * w / w.sum()) - PD, 0.1, 30, xtol=1e-8)
    return eta, rho


def mc_power(lam, eta, w=None, n=2_000_000, chunk=200_000):
    """Fraction of sum_k w_k |mu_k + n_k|^2 above eta (mu_k^2 = lam_k)."""
    lam = np.asarray(lam, float)
    w = np.ones_like(lam) if w is None else np.asarray(w, float)
    hits = 0
    for _ in range(n // chunk):
        re = rng.standard_normal((chunk, lam.size)) + np.sqrt(lam)
        im = rng.standard_normal((chunk, lam.size))
        hits += np.count_nonzero(((re * re + im * im) * w).sum(1) > eta)
    return hits / n


def main():
    R = {}
    rows = {}
    _, r1 = rho50_plain(1)
    for N in (1, 2, 4, 9, 10):
        eta, rho = rho50_plain(N)
        rows[N] = dict(eta=eta, rho50=rho, ratio=rho / r1, volume=(rho / r1) ** 3)
        print(f"N {N:2d}: threshold {eta:.3f}, rho50 {rho:.4f}, ratio to N=1 {rho / r1:.4f}, volume x{(rho / r1) ** 3:.3f}")
    R["plain_equal"] = rows

    # checks of the closed forms: N = 1 threshold 2 ln(1/PFA)
    R["check_eta1"] = (rows[1]["eta"], 2 * np.log(1 / PFA))
    print("N=1 threshold vs 2 ln(1e4):", R["check_eta1"])

    # Gil-Pelaez implementation against chi2/ncx2 for equal weights
    w_eq = np.ones(4)
    R["check_gp"] = dict(sf_null=sf_weighted(rows[4]["eta"], w_eq, np.zeros(4)),
                         sf_sig=sf_weighted(rows[4]["eta"], w_eq, np.full(4, rows[4]["rho50"] ** 2 / 4)))
    print("Gil-Pelaez equal weights, N=4: null sf", R["check_gp"]["sf_null"], "(1e-4); signal sf",
          R["check_gp"]["sf_sig"], "(0.5)")

    noise = Noise("o3")
    uneq = {}
    for mc in (0.85, 1.2, 2.75):
        for Rv in (0.44, 1.0, 2.0):
            frac = np.array(segmented(mc, Rv, noise)["frac"])
            N = frac.size
            eta, rho = rho50_plain(N)
            pd_mc = mc_power(rho ** 2 * frac, eta)
            pfa_mc = mc_power(np.zeros(N), eta, n=10_000_000, chunk=500_000) if Rv == 0.44 and mc == 1.2 else None
            neff = 1 / np.sum(frac ** 2)
            etaw, rhow = rho50_weighted(frac)
            pd_w = mc_power(rhow ** 2 * frac, etaw, w=frac)
            pfa_w = mc_power(np.zeros(N), etaw, w=frac, n=10_000_000, chunk=500_000) if Rv == 0.44 else None
            uneq[f"{mc}_{Rv}"] = dict(N=N, frac=frac.tolist(), rho50_plain=rho, pd_mc_plain=pd_mc,
                                      pfa_mc_plain=pfa_mc, n_eff=neff, rho50_weighted=rhow,
                                      ratio_weighted=rhow / r1, pd_mc_weighted=pd_w, pfa_mc_weighted=pfa_w)
            print(f"Mc {mc} R {Rv}: N {N}, plain rho50 {rho:.4f} (MC detection prob at it {pd_mc:.4f}"
                  + (f", MC false-alarm prob {pfa_mc:.2e}" if pfa_mc is not None else "")
                  + f"); participation N_eff {neff:.2f}; weighted rho50 {rhow:.4f} (x{rhow / r1:.4f} of N=1; "
                  f"MC detection prob {pd_w:.4f}" + (f", MC false-alarm {pfa_w:.2e}" if pfa_w is not None else "") + ")")
    R["unequal"] = uneq
    json.dump(R, open(OUT, "w"), indent=1, default=float)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
