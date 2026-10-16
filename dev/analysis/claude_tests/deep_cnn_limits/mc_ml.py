"""Maximum-likelihood chirp-mass estimates in simulated Gaussian noise for
the coherent model and the segmented models (one free phase per segment,
shared t_c; or one free phase and one free time per segment), on the same
discrete frequency grid as a discrete Fisher matrix.

Data: whitened frequency bins d_j = A w_j exp(i Psi_j) + n_j, n_j complex with
unit-variance real and imaginary parts, w_j^2 = f_j^(-7/3) / S(f_j), bins
every 0.25 Hz in the band of the true chirp mass (O3 average PSD). Statistic
maximized over ln Mc and time (|t| <= 0.25 s):
    coherent   |Z(lnMc, t)|
    shared t   sum_k |Z_k(lnMc, t)|
    free time  sum_k max_t |Z_k(lnMc, t)|
Each is the maximum-likelihood statistic of its model once the free phases
and the amplitude are maximized out. Search: grid in ln Mc over the full
prior 0.85 to 2.75 Msun (or +-HALF around the truth), step 0.001 (coherent),
0.0015 (R2), 0.002 (R1), 0.003 (R0.44), each below a third of that model's
peak half-width; FFT over time (8192 points, 0.488 ms); then a bounded Brent
search in ln Mc of the profile within 1.5 grid steps (FFT with 32768 points,
0.122 ms, parabolic peak interpolation).

usage: mc_ml.py MC0 SNR NREAL SEED OUT [HALF]
"""
import json
import sys
import time

import numpy as np
from scipy import fft as sfft
from scipy import optimize

from common import TSUN, Noise, band, segment_edges_f

DF, M, MREF, TMAX = 0.25, 8192, 32768, 0.25
STEP = {"coherent": 0.001, "R2": 0.0015, "R1": 0.002, "R0.44": 0.003}
WORKERS = 4
SEGS = {"coherent": 4.0, "R2": 2.0, "R1": 1.0, "R0.44": 0.44}
MODELS = [("coherent", "coherent", False), ("R2", "R2", False), ("R1", "R1", False), ("R0.44", "R0.44", False),
          ("R1_time", "R1", True), ("R0.44_time", "R0.44", True)]


def discrete_fisher(f, q, lab, mc0, snr, per_time):
    """sigma(ln Mc) from the discrete Fisher matrix on the bins (q = w^2)."""
    a = -5 / 128 * (np.pi * TSUN * mc0 * f) ** (-5 / 3)
    b = 2 * np.pi * f
    sc = snr ** 2 / q.sum()
    N = lab.max() + 1
    if not per_time:
        F = np.zeros((2 + N, 2 + N))
        F[0, 0], F[0, 1], F[1, 1] = (q * a * a).sum(), (q * a * b).sum(), (q * b * b).sum()
        for k in range(N):
            m = lab == k
            F[0, 2 + k], F[1, 2 + k], F[2 + k, 2 + k] = -(q * a)[m].sum(), -(q * b)[m].sum(), q[m].sum()
    else:
        # closed form: sum over segments of rho_k^2 times the weighted variance
        # of a left after a fit of c0 + c1 f inside the segment. A segment with
        # one or two bins carries no information (its phase and time fit it
        # exactly) and would make the block matrix singular.
        info = 0.0
        for k in range(N):
            m = lab == k
            if m.sum() < 3:
                continue
            qk, ak, bk = q[m], a[m], b[m]
            mk = qk.sum()
            ac, bc = ak - (qk * ak).sum() / mk, bk - (qk * bk).sum() / mk
            vaa, vab, vbb = (qk * ac * ac).sum(), (qk * ac * bc).sum(), (qk * bc * bc).sum()
            info += sc * (vaa - vab * vab / vbb)
        return float(1 / np.sqrt(info))
    F = np.triu(F) + np.triu(F, 1).T
    F *= sc
    d = 1 / np.sqrt(np.diag(F))
    C = d[:, None] * np.linalg.inv(d[:, None] * F * d[None, :]) * d[None, :]
    return float(np.sqrt(C[0, 0]))


def tindex(Mx):
    nt = int(round(TMAX * Mx * DF))
    return np.concatenate([np.arange(Mx - nt, Mx), np.arange(0, nt + 1)])   # times -TMAX..TMAX in order


def parabolic_max(y):
    """Max of each row of y by a parabola through the largest sample and its
    neighbours (rows: independent curves). Returns (value, fractional index)."""
    i = np.clip(y.argmax(-1), 1, y.shape[-1] - 2)
    r = np.arange(y.shape[0])
    y0, y1, y2 = y[r, i - 1], y[r, i], y[r, i + 1]
    den = y0 - 2 * y1 + y2
    dx = np.where(den < 0, 0.5 * (y0 - y2) / np.where(den < 0, den, -1), 0.0)
    return y1 - 0.25 * (y0 - y2) * dx, i + dx


def main():
    mc0, snr, nreal, seed, out = float(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
    half = float(sys.argv[6]) if len(sys.argv) > 6 else None
    rng = np.random.default_rng(seed)
    flo, fhi = band(mc0)
    j = np.arange(int(np.ceil(flo / DF)), int(np.floor(fhi / DF)) + 1)
    f = j * DF
    w = np.sqrt(f ** (-7 / 3) / Noise("o3")(f))
    q = w * w
    norm = np.sqrt(q.sum())
    A = snr / norm
    lnmc0 = np.log(mc0)
    lo, hi = (np.log(0.85), np.log(2.75)) if half is None else (lnmc0 - half, lnmc0 + half)
    grids = {s: np.arange(lo, hi + 1e-9, STEP[s]) for s in SEGS}
    labs = {}
    for name, R in SEGS.items():
        fe = segment_edges_f(mc0, R)
        labs[name] = np.clip(np.searchsorted(fe, f, side="right") - 1, 0, len(fe) - 2)
    masks = {name: [labs[name] == k for k in range(labs[name].max() + 1)] for name in labs}
    fish = {mname: discrete_fisher(f, q, labs[sname], mc0, snr, pt) for mname, sname, pt in MODELS}
    tid, tidr = tindex(M), tindex(MREF)

    # segments are contiguous runs of bins: (first FFT index, last + 1, first bin, last + 1)
    slices = {s: [(int(j[m][0]), int(j[m][-1]) + 1, int(np.flatnonzero(m)[0]), int(np.flatnonzero(m)[-1]) + 1)
                  for m in masks[s]] for s in masks}
    for s in masks:
        for (ja, jb, ia, ib), m in zip(slices[s], masks[s]):
            assert jb - ja == ib - ia == m.sum()
    bufs = {}

    def seg_abs(X, sname, Mx, ti):
        """|Z_k(t)| for every segment: list of (c, nt) arrays. Reuses one
        buffer per shape (fresh allocations page-fault on this host)."""
        key = (X.shape[0], Mx, X.dtype)
        if key not in bufs:
            bufs[key] = np.zeros((X.shape[0], Mx), X.dtype)
        b = bufs[key]
        outl = []
        for ja, jb, ia, ib in slices[sname]:
            b.fill(0)
            b[:, ja:jb] = X[:, ia:ib]
            y = sfft.fft(b, axis=1, workers=WORKERS, overwrite_x=True)
            outl.append(np.abs(y[:, ti]))
        return outl

    def xmat(d, g):
        return d[None, :] * w[None, :] * np.exp(-1j * 3 / 128 * (np.pi * TSUN * np.exp(g)[:, None] * f[None, :]) ** (-5 / 3)) / norm

    def grid_search(d):
        best = {mname: (-np.inf, 0.0) for mname, _, _ in MODELS}
        CH = 64
        for sname in SEGS:
            grid = grids[sname]
            for s in range(0, grid.size, CH):
                g = grid[s:s + CH]
                Zs = seg_abs(xmat(d, g).astype(np.complex64), sname, M, tid)
                cand = {sname: sum(Zs).max(1)}
                if sname in ("R1", "R0.44"):
                    cand[sname + "_time"] = sum(z.max(1) for z in Zs)
                for mname, v in cand.items():
                    i = int(np.argmax(v))
                    if v[i] > best[mname][0]:
                        best[mname] = (float(v[i]), float(g[i]))
        return best

    def profile(d, lnmc, sname, per_time):
        X = xmat(d, np.array([lnmc]))
        Zs = seg_abs(X, sname, MREF, tidr)
        if per_time:
            return float(sum(parabolic_max(z)[0][0] for z in Zs))
        return float(parabolic_max(sum(Zs))[0][0])

    res = {mname: [] for mname, _, _ in MODELS}
    vals = {mname: [] for mname, _, _ in MODELS}
    t0 = time.time()

    def dump():
        json.dump(dict(mc0=mc0, snr=snr, seed=seed, grid=[float(lo), float(hi), STEP],
                       fisher=fish, err=res, val=vals, n_bins=int(f.size), f_lo=float(f[0]), f_hi=float(f[-1])),
                  open(out, "w"))

    for it in range(nreal):
        n = rng.standard_normal(f.size) + 1j * rng.standard_normal(f.size)
        d = A * w * np.exp(1j * 3 / 128 * (np.pi * TSUN * mc0 * f) ** (-5 / 3)) + n
        best = grid_search(d)
        for mname, sname, pt in MODELS:
            g0, hs = best[mname][1], 1.5 * STEP[sname]
            r = optimize.minimize_scalar(lambda x: -profile(d, x, sname, pt), bounds=(g0 - hs, g0 + hs),
                                         method="bounded", options=dict(xatol=1e-7))
            res[mname].append(float(r.x - lnmc0))
            vals[mname].append(float(-r.fun))
        if (it + 1) % 5 == 0 or it + 1 == nreal:
            msg = " ".join(f"{k}:{np.mean(np.abs(v) < np.log(1.01)):.3f}" for k, v in res.items())
            print(f"[{it + 1}/{nreal}] {time.time() - t0:.0f}s within 1%: {msg}", flush=True)
            dump()
    dump()


if __name__ == "__main__":
    main()
