"""Population Monte Carlo with the searches of mc_ml.py: each realization
draws a chirp mass uniform on [0.85, 2.75] Msun (the training waveform file
is uniform there) and an SNR from p(rho) ~ rho^-3 on [4, 100] (the
validation population, derivations #late-snr-bands), then runs the same
grid-plus-refine maximum-likelihood search for all six models.

usage: mc_pop.py NREAL SEED OUT
       mc_pop.py --regress   (compare with mc_ml.py output mc_test.json)
"""
import json
import sys
import time

import numpy as np
from scipy import fft as sfft
from scipy import optimize

from common import TSUN, Noise, band, segment_edges_f
from mc_ml import DF, M, MODELS, MREF, SEGS, STEP, WORKERS, parabolic_max, tindex


class Searcher:
    def __init__(self, mc0, noise):
        self.mc0 = mc0
        flo, fhi = band(mc0)
        j = np.arange(int(np.ceil(flo / DF)), int(np.floor(fhi / DF)) + 1)
        f = j * DF
        w = np.sqrt(f ** (-7 / 3) / noise(f))
        self.j, self.f, self.w = j, f, w
        self.norm = np.sqrt((w * w).sum())
        self.lnmc0 = np.log(mc0)
        lo, hi = np.log(0.85), np.log(2.75)
        self.grids = {s: np.arange(lo, hi + 1e-9, STEP[s]) for s in SEGS}
        labs = {}
        for name, R in SEGS.items():
            fe = segment_edges_f(mc0, R)
            labs[name] = np.clip(np.searchsorted(fe, f, side="right") - 1, 0, len(fe) - 2)
        # a segment narrower than one bin can hold no bin; it then carries nothing
        masks = {name: [labs[name] == k for k in range(labs[name].max() + 1) if (labs[name] == k).any()] for name in labs}
        self.slices = {s: [(int(j[m][0]), int(j[m][-1]) + 1, int(np.flatnonzero(m)[0]), int(np.flatnonzero(m)[-1]) + 1)
                           for m in masks[s]] for s in masks}
        self.tid, self.tidr = tindex(M), tindex(MREF)
        self.bufs = {}

    def data(self, snr, rng):
        f, w = self.f, self.w
        n = rng.standard_normal(f.size) + 1j * rng.standard_normal(f.size)
        A = snr / self.norm
        return A * w * np.exp(1j * 3 / 128 * (np.pi * TSUN * self.mc0 * f) ** (-5 / 3)) + n

    def seg_abs(self, X, sname, Mx, ti):
        key = (X.shape[0], Mx, X.dtype)
        if key not in self.bufs:
            self.bufs[key] = np.zeros((X.shape[0], Mx), X.dtype)
        b = self.bufs[key]
        outl = []
        for ja, jb, ia, ib in self.slices[sname]:
            b.fill(0)
            b[:, ja:jb] = X[:, ia:ib]
            y = sfft.fft(b, axis=1, workers=WORKERS, overwrite_x=True)
            outl.append(np.abs(y[:, ti]))
        return outl

    def xmat(self, d, g):
        f, w = self.f, self.w
        return d[None, :] * w[None, :] * np.exp(-1j * 3 / 128 * (np.pi * TSUN * np.exp(g)[:, None] * f[None, :]) ** (-5 / 3)) / self.norm

    def grid_search(self, d):
        best = {mname: (-np.inf, 0.0) for mname, _, _ in MODELS}
        CH = 64
        for sname in SEGS:
            grid = self.grids[sname]
            for s in range(0, grid.size, CH):
                g = grid[s:s + CH]
                Zs = self.seg_abs(self.xmat(d, g).astype(np.complex64), sname, M, self.tid)
                cand = {sname: sum(Zs).max(1)}
                if sname in ("R1", "R0.44"):
                    cand[sname + "_time"] = sum(z.max(1) for z in Zs)
                for mname, v in cand.items():
                    i = int(np.argmax(v))
                    if v[i] > best[mname][0]:
                        best[mname] = (float(v[i]), float(g[i]))
        return best

    def profile(self, d, lnmc, sname, per_time):
        Zs = self.seg_abs(self.xmat(d, np.array([lnmc])), sname, MREF, self.tidr)
        if per_time:
            return float(sum(parabolic_max(z)[0][0] for z in Zs))
        return float(parabolic_max(sum(Zs))[0][0])

    def estimate(self, d):
        best = self.grid_search(d)
        out = {}
        for mname, sname, pt in MODELS:
            g0, hs = best[mname][1], 1.5 * STEP[sname]
            r = optimize.minimize_scalar(lambda x: -self.profile(d, x, sname, pt), bounds=(g0 - hs, g0 + hs),
                                         method="bounded", options=dict(xatol=1e-7))
            out[mname] = float(r.x - self.lnmc0)
        return out


def draw_snr(rng, a=4.0, b=100.0):
    u = rng.uniform()
    return (a ** -2 - u * (a ** -2 - b ** -2)) ** -0.5


def main():
    noise = Noise("o3")
    if sys.argv[1] == "--regress":
        ref = json.load(open("/tmp/claude-1024/-home-kyoon-SSM-BNS/82effc75-bb98-44ad-af20-e41e39a7258d/scratchpad/cnnproof/mc_test.json"))
        s = Searcher(1.2, noise)
        rng = np.random.default_rng(1)
        got = [s.estimate(s.data(8.0, rng)) for _ in range(2)]
        diff = max(abs(got[i][k] - ref["err"][k][i]) for i in range(2) for k in ref["err"])
        print("max |difference| from mc_ml.py on the same seed:", diff)
        return
    nreal, seed, out = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
    rng = np.random.default_rng(seed)
    rec = dict(mc0=[], snr=[], err={m: [] for m, _, _ in MODELS})
    t0 = time.time()
    for it in range(nreal):
        mc0 = rng.uniform(0.85, 2.75)
        snr = draw_snr(rng)
        s = Searcher(mc0, noise)
        e = s.estimate(s.data(snr, rng))
        rec["mc0"].append(mc0)
        rec["snr"].append(snr)
        for k, v in e.items():
            rec["err"][k].append(v)
        if (it + 1) % 10 == 0 or it + 1 == nreal:
            msg = " ".join(f"{k}:{np.mean(np.abs(v) < np.log(1.01)):.3f}" for k, v in rec["err"].items())
            print(f"[{it + 1}/{nreal}] {time.time() - t0:.0f}s within 1%: {msg}", flush=True)
            json.dump(rec, open(out, "w"))


if __name__ == "__main__":
    main()
