"""Chirp mass from coherent and segmented matched filters on simulated
whitened BNS chirps in white Gaussian noise.

Signal (leading order, whitened by the stationary-phase rule), tau = t_c - t:
    f(tau)   = (256 tau / 5)^(-3/8) (T_sun Mc)^(-5/8) / pi
    Phi(tau) = -2 (tau / (5 T_sun Mc))^(5/8)
    s(t)     = f^(2/3) S(f)^(-1/2) cos(Phi + phi0)    while f < f_hi,
f_hi = min(f_ISCO, 1024 Hz), f_ISCO = 1 / (6^1.5 pi T_sun M), M = 2^1.2 Mc;
S is the O3 average PSD (harmonic mean of H1 and L1, as in fisher_seg.py).
Data: 4 s at 2048 Hz (8192 samples), s scaled to sum s^2 = SNR^2 over the
window, plus white Gaussian noise of unit variance per sample.

Templates: Mc_j = 0.85 * 1.005^j (236 values, 0.85 to 2.744 Msun); complex
h_j = f^(2/3) S^(-1/2) exp(i Phi) on the 4 s ending at t_c (frame sample u
at tau = (8192 - u) / 2048 s), scaled to sum |h_j|^2 = 2. Segments of
length R are counted back from t_c: frame sample u is in segment
floor((8191 - u) / (R * 2048)).

For template j, segment k and lag m (template t_c at (8192 + m) / 2048 s),
Z_jk(m) = sum_u x[u + m] conj(h_jk[u]), with x = 0 outside the window:
    a    max over m in M of |Z_j(m)|^2
    b_R  max over m in M of sum_k |Z_jk(m)|^2
    c_R  sum_k sum over every m of |Z_jk(m)|^2, divided by 8192
M: m = -1024..0 (t_c from 3.5 to 4.0 s, the injection prior). c_R is the
segment energy averaged over all time shifts at which segment k touches the
window (Parseval: a weighted sum of the data's power spectrum); its noise
mean is exactly sum |h_j|^2 = 2 for every template.
csame_R (diagnostic): sum_k mean over m in C_k of |Z_jk(m)|^2, C_k the 8192
lags that put the centre of segment k on a sample of the window ('same'
padding); its noise mean depends on the template.
Estimate: Mc of the template with the largest statistic.

usage:
  sim.py run LO HI N SEED OUT.npz [STEP] [RS]
        N injections, SNR density ~ SNR^-3 on [LO, HI] (LO = HI: fixed SNR),
        Mc log-uniform on [0.85, 2.75], t_c uniform on [3.5, 4.0] s, phase
        uniform; noise of injection i from default_rng([SEED, i]).
        STEP: template ratio (default 1.005); RS: comma list of R or 'none'.
  sim.py noisefree N SEED OUT.npz      draws as in run, SNR scale 10, no noise
  sim.py checks OUT.json
  sim.py report OUT_PREFIX RUN.npz ...   tables (json, txt)
  sim.py figure OUT.png RUN.npz ...      fraction within 1% against SNR band
  sim.py post OUT.json                   band-edge test, grid spacing, widths, calibration (run from this folder)
"""
import json
import resource
import sys
import time

import numpy as np
import torch

FS = 2048.0
L = 8192
N = 2 * L
TSUN = 4.925490947e-6
PSD_FILE = "/home/kyoon/SSM-BNS/aframe/dev/runs/claude_tests/template_bank/bank_mc0.5pct_spin0_q0.8.npz"
MC_LO, MC_HI, STEP = 0.85, 2.75, 1.005
TC_LO, TC_HI = 3.5, 4.0
M_LO, M_HI = int(round(TC_LO * FS)) - L, int(round(TC_HI * FS)) - L     # -1024, 0
RS = (0.44, 1.0, 2.0, 4.0)
TOLS = (0.01, 0.02, 0.05)
GPU_BYTES = 3.0e9

_PSD = {}


def psd(f):
    """O3 average PSD, harmonic mean of H1 and L1, linear between 0.5 Hz knots."""
    if not _PSD:
        p = np.load(PSD_FILE)["psd"]
        _PSD["f"] = np.arange(p.shape[1]) * 0.5
        _PSD["s"] = 2 / (1 / p[0] + 1 / p[1])
    return np.interp(f, _PSD["f"], _PSD["s"])


def mc_grid(step=STEP):
    n = int(np.floor(np.log(MC_HI / MC_LO) / np.log(step))) + 1
    return MC_LO * step ** np.arange(n)


def f_hi(mc):
    return min(1 / (6 ** 1.5 * np.pi * TSUN * 2 ** 1.2 * mc), FS / 2)


def f_of_tau(tau, mc):
    return (256 * tau / 5) ** (-3 / 8) * (TSUN * mc) ** (-5 / 8) / np.pi


def chirp(tau, mc):
    """Whitened amplitude and phase at times tau (s) before coalescence."""
    t = np.where(tau > 0, tau, 1.0)
    f = f_of_tau(t, mc)
    on = (tau > 0) & (f < f_hi(mc))
    amp = np.where(on, f ** (2 / 3) / np.sqrt(psd(f)), 0.0)
    return amp, -2 * (t / (5 * TSUN * mc)) ** (5 / 8)


def segments(R):
    """(start, end) frame samples of each segment; k = 0 ends at t_c."""
    k = np.floor((L - 1 - np.arange(L)) / (R * FS)).astype(int)
    out = []
    for q in range(k.max() + 1):
        i = np.flatnonzero(k == q)
        out.append((int(i[0]), int(i[-1]) + 1))
    return out


def template_bank(grid):
    tau = (L - np.arange(L)) / FS
    H = np.empty((len(grid), L), np.complex128)
    for j, mc in enumerate(grid):
        a, p = chirp(tau, mc)
        H[j] = a * np.exp(1j * p) * np.sqrt(2 / np.sum(a * a))
    return H


def injection(mc, tc, ph, snr):
    a, p = chirp(tc - np.arange(L) / FS, mc)
    s = a * np.cos(p + ph)
    return s * snr / np.sqrt(np.sum(s * s))


def draw(n, lo, hi, seed):
    rng = np.random.default_rng(seed)
    u = rng.random(n)
    snr = (lo ** -2 - u * (lo ** -2 - hi ** -2)) ** -0.5
    mc = np.exp(rng.uniform(np.log(MC_LO), np.log(MC_HI), n))
    tc = rng.uniform(TC_LO, TC_HI, n)
    ph = rng.uniform(0, 2 * np.pi, n)
    return snr, mc, tc, ph


def noise(seed, i):
    return np.random.default_rng([seed, i]).standard_normal(L)


def gpu():
    dev = torch.device("cuda", 0)
    torch.cuda.set_per_process_memory_fraction(GPU_BYTES / torch.cuda.get_device_properties(dev).total_memory, dev)
    return dev


class Engine:
    """All statistics of the module docstring for a batch of windows."""

    def __init__(self, grid, rs=RS, dtype=torch.complex64, chunk_rows=1100, dev=None):
        self.dev = dev or gpu()
        self.grid, self.rs, self.dtype, self.chunk_rows = grid, tuple(rs), dtype, chunk_rows
        self.rdtype = torch.float64 if dtype == torch.complex128 else torch.float32
        self.rows = [("full", 0, L)] + [(R, a, b) for R in self.rs for a, b in segments(R)]
        self.nrow = len(self.rows)
        self.rows_of = {}
        for R in self.rs:
            r = [i for i, row in enumerate(self.rows) if row[0] == R]
            self.rows_of[R] = slice(r[0], r[-1] + 1)
        self.centre = [(a + b - 1) // 2 for _, a, b in self.rows]
        self.prior = torch.cat([torch.arange(N + M_LO, N), torch.arange(0, M_HI + 1)]).to(self.dev)
        Hn = template_bank(grid)
        self.Hc = torch.empty((len(grid), self.nrow, N), dtype=dtype, device=self.dev)
        for j0 in range(0, len(grid), 64):
            h = torch.as_tensor(Hn[j0:j0 + 64], device=self.dev).to(dtype)
            for r, (_, a, b) in enumerate(self.rows):
                hp = torch.zeros((h.shape[0], N), dtype=dtype, device=self.dev)
                hp[:, a:b] = h[:, a:b]
                self.Hc[j0:j0 + 64, r] = torch.fft.fft(hp).conj()

    def spectra(self, x):
        xp = torch.zeros((x.shape[0], N), dtype=self.rdtype, device=self.dev)
        xp[:, :L] = torch.as_tensor(x, device=self.dev).to(self.rdtype)
        return torch.fft.fft(xp.to(self.dtype))

    def raw(self, x, j):
        """Z_jr(m) for every row r of template j, lag m at index m mod N."""
        return torch.fft.ifft(self.spectra(x)[:, None, :] * self.Hc[j][None])

    def __call__(self, x):
        X = self.spectra(x)
        B, nT = x.shape[0], len(self.grid)
        out = {"a": np.empty((B, nT)), "a_lag": np.empty((B, nT), np.int64)}
        for R in self.rs:
            for key in ("b", "c", "csame"):
                out[f"{key}_{R}"] = np.empty((B, nT))
        J = max(1, self.chunk_rows // (B * self.nrow))
        for j0 in range(0, nT, J):
            j1 = min(nT, j0 + J)
            Y = torch.fft.ifft(X[:, None, None, :] * self.Hc[None, j0:j1])
            E = Y.real ** 2 + Y.imag ** 2                                   # (B, J, rows, N)
            del Y
            Ep = E.index_select(-1, self.prior)                              # lags M_LO..M_HI
            v, i = Ep[:, :, 0].max(-1)
            out["a"][:, j0:j1] = v.double().cpu().numpy()
            out["a_lag"][:, j0:j1] = i.cpu().numpy() + M_LO
            for R in self.rs:
                rr = self.rows_of[R]
                out[f"b_{R}"][:, j0:j1] = Ep[:, :, rr].sum(2).max(-1).values.double().cpu().numpy()
                c = 0
                for r in range(rr.start, rr.stop):
                    cr = self.centre[r]
                    c = c + E[:, :, r, N - cr:].sum(-1, dtype=torch.float64) + E[:, :, r, :L - cr].sum(-1, dtype=torch.float64)
                out[f"csame_{R}"][:, j0:j1] = (c / L).cpu().numpy()
                out[f"c_{R}"][:, j0:j1] = (E[:, :, rr].sum((2, 3), dtype=torch.float64) / L).cpu().numpy()
            del E, Ep
        return out


def methods(rs=RS):
    return ["a"] + [f"{k}_{R}" for k in ("b", "c", "csame") for R in rs]


def simulate(lo, hi, n, seed, out_file, step=STEP, rs=RS, with_noise=True, batch=4):
    t0 = time.time()
    grid = mc_grid(step)
    eng = Engine(grid, rs)
    snr, mc, tc, ph = draw(n, lo, hi, seed)
    names = methods(rs)
    est = {k: np.empty(n, np.int64) for k in names}
    top = {k: np.empty(n) for k in names}
    stat = {k: np.empty((n, len(grid)), np.float32) for k in names}
    lag = np.empty(n, np.int64)
    for i0 in range(0, n, batch):
        idx = np.arange(i0, min(n, i0 + batch))
        x = np.stack([injection(mc[i], tc[i], ph[i], snr[i]) + (noise(seed, i) if with_noise else 0.0) for i in idx])
        o = eng(x)
        for k in names:
            j = np.argmax(o[k], 1)
            est[k][idx] = j
            top[k][idx] = o[k][np.arange(len(idx)), j]
            stat[k][idx] = o[k]
        lag[idx] = o["a_lag"][np.arange(len(idx)), est["a"][idx]]
        if (i0 // batch) % 100 == 0:
            msg = " ".join(f"{k}:{np.mean(np.abs(grid[est[k][:idx[-1] + 1]] / mc[:idx[-1] + 1] - 1) < 0.01):.3f}"
                           for k in names if not k.startswith("csame"))
            print(f"[{idx[-1] + 1}/{n}] {time.time() - t0:.0f}s within 1%: {msg}", flush=True)
    np.savez(out_file, grid=grid, snr=snr, mc=mc, tc=tc, ph=ph, lo=lo, hi=hi, seed=seed, step=step,
             rs=np.array(rs, float), with_noise=with_noise, a_lag=lag,
             **{f"est_{k}": v for k, v in est.items()}, **{f"top_{k}": v for k, v in top.items()},
             **{f"stat_{k}": v for k, v in stat.items()})
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20
    print(f"DONE {out_file} in {time.time() - t0:.0f}s; peak GPU {torch.cuda.max_memory_allocated() / 2 ** 30:.2f} GiB, "
          f"peak RSS {rss:.2f} GiB", flush=True)


# ------------------------------------------------------------------ checks
def reference(x, H, segs, prior_lags):
    """Independent float64 route with scipy.signal.correlate for one window:
    b (max over prior lags of sum_k |Z_k|^2), c (sum over every lag, / L) and
    csame (sum_k mean over C_k of |Z_k|^2)."""
    from scipy.signal import correlate

    def at(z, q):
        ok = (q >= 0) & (q < len(z))
        return np.where(ok, z[np.clip(q, 0, len(z) - 1)], 0)

    b, c, cs = np.zeros(len(H)), np.zeros(len(H)), np.zeros(len(H))
    for j, h in enumerate(H):
        P = np.zeros(len(prior_lags))
        for a_, b_ in segs:
            # z[q] = sum_v x[v + q - (b_ - a_ - 1)] conj(h[a_ + v]), so lag m sits at q = m + b_ - 1
            z = correlate(x, h[a_:b_], mode="full")
            P += np.abs(at(z, prior_lags + b_ - 1)) ** 2
            c[j] += np.sum(np.abs(z) ** 2) / L
            cr = (a_ + b_ - 1) // 2
            cs[j] += np.sum(np.abs(at(z, np.arange(-cr, L - cr) + b_ - 1)) ** 2) / L
        b[j] = P.max()
    return b, c, cs


def check(out_file):
    t0 = time.time()
    res = {}
    grid = mc_grid()
    H = template_bank(grid)
    # 1. template quadratures
    hc, hs = H.real, H.imag
    nc, ns, cs = (hc * hc).sum(1), (hs * hs).sum(1), (hc * hs).sum(1)
    tot = (np.abs(H) ** 2).sum(1)
    res["quadrature"] = dict(norm_cos=[nc.min(), nc.max()], norm_sin=[ns.min(), ns.max()],
                             max_abs_overlap=float(np.abs(cs).max()), sum_abs2=[tot.min(), tot.max()])
    # band facts against fisher.md: f 4 s before t_c and f_hi
    res["band"] = {str(m): dict(f_4s=float(f_of_tau(4.0, m)), f_hi=float(f_hi(m))) for m in (0.85, 1.2, 2.75)}
    # first and last nonzero frame sample of the lightest and heaviest template
    res["template_support"] = {str(round(grid[j], 4)): [int(np.flatnonzero(np.abs(H[j]) > 0)[0]),
                                                        int(np.flatnonzero(np.abs(H[j]) > 0)[-1])] for j in (0, len(grid) - 1)}
    res["segments"] = {str(R): [(a, b) for a, b in segments(R)] for R in RS}
    eng = Engine(grid)
    rng = np.random.default_rng(7)

    # 2. noise-free injection exactly on a template, integer t_c
    on = []
    for _ in range(40):
        j0 = int(rng.integers(len(grid)))
        m0 = int(rng.integers(M_LO, M_HI + 1))
        ph = rng.uniform(0, 2 * np.pi)
        tc = (L + m0) / FS
        x = injection(grid[j0], tc, ph, 10.0)[None]
        o = eng(x)
        frac = np.sum(np.abs(H[j0, max(0, -m0):]) ** 2) / 2          # template energy inside the window
        on.append(dict(j0=j0, m0=m0, est={k: int(np.argmax(o[k][0])) for k in methods()},
                       lag=int(o["a_lag"][0, j0]), ratio=float(o["a"][0, j0] / (100 * frac))))
    res["on_grid"] = dict(
        n=len(on),
        all_methods_pick_j0={k: int(sum(r["est"][k] == r["j0"] for r in on)) for k in methods()},
        lag_equals_m0=int(sum(r["lag"] == r["m0"] for r in on)),
        a_over_snr2_times_inside_fraction=[min(r["ratio"] for r in on), max(r["ratio"] for r in on)])

    # 3. segments add up to the full template (linearity), all R
    x = injection(1.3, 3.8, 0.4, 10.0) + noise(99, 0)
    lin = {}
    for j in (0, 117, len(grid) - 1):
        Z = eng.raw(x[None], j)[0]                                       # (rows, N)
        for R in RS:
            rr = eng.rows_of[R]
            d = (Z[rr].sum(0) - Z[0]).abs().max().item() / Z[0].abs().max().item()
            lin[f"{j}_{R}"] = d
    res["linearity_max_rel"] = max(lin.values())

    # 4. independent scipy route against the engine (float32 GPU), 10 windows at SNR 10
    ref = []
    prior_lags = np.arange(M_LO, M_HI + 1)
    snr, mc, tc, ph = draw(10, 10.0, 10.0, 11)
    for i in range(10):
        x = injection(mc[i], tc[i], ph[i], snr[i]) + noise(11, i)
        o = eng(x[None])
        row = {}
        for R in (0.44, 4.0):
            b, c, cs = reference(x, H, segments(R), prior_lags)
            for key, v in ((f"b_{R}", b), (f"c_{R}", c), (f"csame_{R}", cs)):
                row[key] = (float(np.max(np.abs(o[key][0] / v - 1))), int(np.argmax(v) == np.argmax(o[key][0])))
            if R == 4.0:
                row["a"] = (float(np.max(np.abs(o["a"][0] / b - 1))), int(np.argmax(b) == np.argmax(o["a"][0])))
        ref.append(row)
    res["reference"] = {k: dict(max_rel=max(r[k][0] for r in ref), same_argmax=sum(r[k][1] for r in ref), n=len(ref))
                        for k in ref[0]}

    # 5. float64 engine against float32 engine, 100 windows at SNR 8 to 10
    eng64 = Engine(grid, dtype=torch.complex128, chunk_rows=600, dev=eng.dev)
    snr, mc, tc, ph = draw(100, 8.0, 10.0, 12)
    same = {k: 0 for k in methods()}
    rel = {k: 0.0 for k in methods()}
    for i in range(100):
        x = (injection(mc[i], tc[i], ph[i], snr[i]) + noise(12, i))[None]
        o32, o64 = eng(x), eng64(x)
        for k in methods():
            same[k] += int(np.argmax(o32[k][0]) == np.argmax(o64[k][0]))
            rel[k] = max(rel[k], float(np.max(np.abs(o32[k][0] / o64[k][0] - 1))))
    res["float32_vs_float64"] = dict(n=100, same_argmax=same, max_rel=rel)
    del eng64
    torch.cuda.empty_cache()

    # 6. noise only: |Z_j(0)|^2 has mean 2 (full overlap at m = 0); noise means of c_R (2 exactly) and csame_R
    a0, cvals, svals = [], {R: [] for R in RS}, {R: [] for R in RS}
    for i in range(200):
        x = noise(13, i)[None]
        Z = torch.fft.ifft(eng.spectra(x)[:, None, :] * eng.Hc[:, 0])[0, :, 0]
        a0.append((Z.real ** 2 + Z.imag ** 2).double().cpu().numpy())
        o = eng(x)
        for R in RS:
            cvals[R].append(o[f"c_{R}"][0])
            svals[R].append(o[f"csame_{R}"][0])
    a0 = np.array(a0)
    res["noise_lag0"] = dict(mean=float(a0.mean()), se=float(a0.std() / np.sqrt(a0.size)), var=float(a0.var()))
    # csame_R noise mean: sum over samples of |h_j[u]|^2 times the share of C_k lags that put u in the window
    for R in RS:
        e = np.zeros(len(grid))
        for a_, b_ in segments(R):
            cr = (a_ + b_ - 1) // 2
            u = np.arange(a_, b_)
            # u + m in [0, L - 1] and m in [-cr, L - 1 - cr]
            cnt = np.minimum(L - 1 - u, L - 1 - cr) - np.maximum(-u, -cr) + 1
            e += (np.abs(H[:, a_:b_]) ** 2 * np.clip(cnt, 0, None)).sum(1) / L
        cv, sv = np.array(cvals[R]), np.array(svals[R])
        res[f"noise_c_{R}"] = dict(mean_over_2=[float(cv.mean(0).min() / 2), float(cv.mean(0).max() / 2)],
                                   se_of_mean=float(cv.std(0).mean() / np.sqrt(len(cv))),
                                   sd_over_noise=float(cv.std(0).mean()))
        res[f"noise_csame_{R}"] = dict(expect_range=[float(e.min()), float(e.max())],
                                       expect_lightest_heaviest=[float(e[0]), float(e[-1])],
                                       mean_over_expect=[float((sv.mean(0) / e).min()), float((sv.mean(0) / e).max())],
                                       sd_over_noise=float(sv.std(0).mean()))
    res["seconds"] = time.time() - t0
    json.dump(res, open(out_file, "w"), indent=1, default=float)
    print(json.dumps(res, indent=1, default=float))


# ------------------------------------------------------------------ report
def wilson(k, n, z=1.0):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def summarize(files):
    rows = []
    for fn in files:
        z = np.load(fn)
        grid, mc = z["grid"], z["mc"]
        rs = tuple(float(r) for r in z["rs"])
        names = [k for k in methods(rs) if f"est_{k}" in z]
        hit = {k: {t: np.abs(grid[z[f"est_{k}"]] / mc - 1) < t for t in TOLS} for k in names}
        n = len(mc)
        row = dict(file=fn, lo=float(z["lo"]), hi=float(z["hi"]), n=n, step=float(z["step"]),
                   with_noise=bool(z["with_noise"]), snr_mean=float(z["snr"].mean()), res={})
        for k in names:
            row["res"][k] = {}
            for t in TOLS:
                h = hit[k][t]
                p = h.mean()
                d = hit["a"][t].astype(float) - h
                row["res"][k][str(t)] = dict(p=float(p), se=float(np.sqrt(p * (1 - p) / n)),
                                             wilson68=[float(v) for v in wilson(h.sum(), n)],
                                             minus_a=float(-d.mean()), minus_a_se=float(d.std(ddof=1) / np.sqrt(n)))
            err = grid[z[f"est_{k}"]] / mc - 1
            row["res"][k]["median_err"] = float(np.median(err))
            row["res"][k]["frac_beyond_10pct"] = float(np.mean(np.abs(err) > 0.10))
        rows.append(row)
    return rows


LABEL = {"a": "(a) coherent 4 s, max over time",
         **{f"b_{R}": f"(b) segments {R:g} s, power sum, max over time" for R in RS},
         **{f"c_{R}": f"(c) segments {R:g} s, energies averaged over time" for R in RS},
         **{f"csame_{R}": f"(c, 'same' lags, diagnostic) segments {R:g} s" for R in RS}}


def report(prefix, files):
    rows = summarize(files)
    json.dump(rows, open(prefix + ".json", "w"), indent=1)
    lines = []
    for row in rows:
        lines.append(f"\n{row['file']}: SNR {row['lo']:g} to {row['hi']:g}, n {row['n']}, step {row['step']}, "
                     f"noise {row['with_noise']}, mean SNR {row['snr_mean']:.2f}")
        lines.append("method | within 1% | within 2% | within 5% | 1%: minus (a), paired | median error | beyond 10%")
        for k, v in row["res"].items():
            cells = [f"{100 * v[str(t)]['p']:.1f} ± {100 * v[str(t)]['se']:.1f}" for t in TOLS]
            lines.append(f"{LABEL[k]} | " + " | ".join(cells) +
                         f" | {100 * v['0.01']['minus_a']:+.1f} ± {100 * v['0.01']['minus_a_se']:.1f}"
                         f" | {100 * v['median_err']:+.2f}% | {100 * v['frac_beyond_10pct']:.1f}%")
    txt = "\n".join(lines)
    open(prefix + ".txt", "w").write(txt)
    print(txt)


def figure(png, files):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = [r for r in summarize(files) if r["with_noise"] and r["lo"] != r["hi"]]
    rows.sort(key=lambda r: r["lo"])
    x = np.arange(len(rows))
    colors = {4.0: "#2a78d6", 2.0: "#eb6834", 1.0: "#1baf7a", 0.44: "#eda100"}
    lines = [("a", 4.0, "-", "o", "(a) coherent 4 s, max over time")]
    lines += [(f"b_{R}", R, "-", "o", f"(b) {R:g} s segments, power sum, max over time") for R in (2.0, 1.0, 0.44)]
    lines += [(f"c_{R}", R, "--", "s", f"(c) {R:g} s segments, energies averaged over time") for R in (4.0, 2.0, 1.0, 0.44)]
    surface, ink, muted, grid = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), dpi=150, gridspec_kw=dict(width_ratios=[1, 1]))
    fig.patch.set_facecolor(surface)
    for ax, zoom in zip(axes, (False, True)):
        ax.set_facecolor(surface)
        for k, R, ls, mk, lab in lines:
            if zoom and k.startswith("c"):
                continue
            p = np.array([r["res"][k]["0.01"]["p"] for r in rows])
            se = np.array([r["res"][k]["0.01"]["se"] for r in rows])
            ax.errorbar(x, 100 * p, yerr=100 * se, color=colors[R], ls=ls, lw=2, marker=mk, ms=6,
                        mec=surface, mew=1.5, capsize=3, label=None if zoom else lab)
        ax.set_xticks(x, [f"{r['lo']:g} to {r['hi']:g}" for r in rows])
        ax.set_xlabel("SNR band (in the 4 s window)", color=ink)
        ax.grid(axis="y", color=grid, lw=1)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color("#b9b8b2")
        ax.tick_params(colors=muted)
    axes[0].set_ylim(0, 102)
    axes[0].set_ylabel("events with chirp mass within 1% (%)", color=ink)
    axes[0].set_title("all methods", fontsize=10, color=ink)
    axes[1].set_ylim(88, 100.5)
    axes[1].set_title("same data, 88 to 100%: (a) and (b) only", fontsize=10, color=ink)
    fig.suptitle("Chirp mass from coherent and segmented matched filters, simulated whitened chirps in white noise\n"
                 f"{rows[0]['n']} injections per band (SNR density ∝ SNR$^{{-3}}$ inside each band); "
                 "error bars: binomial standard error", fontsize=10, color=ink)
    fig.legend(loc="lower center", ncol=2, fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.17, 1, 1))
    fig.savefig(png, facecolor=surface, bbox_inches="tight")
    print("wrote", png)

# ------------------------------------------------------------------ post-analysis
def edge_mechanism(fn):
    """Estimates against Mc_edge = Mc (t_c / 4)^(3/5): the chirp mass whose 4 s
    template band starts at the frequency where the in-window signal starts."""
    z = np.load(fn)
    grid, mc, tc = z["grid"], z["mc"], z["tc"]
    edge = mc * (tc / 4) ** 0.6
    split = 1 / (6 ** 1.5 * np.pi * TSUN * 2 ** 1.2 * FS / 2)            # f_ISCO = 1024 Hz
    out = {"mc_isco_at_1024Hz": float(split)}
    for k in methods(tuple(float(r) for r in z["rs"])):
        est = grid[z[f"est_{k}"]]
        et, ee = est / mc - 1, est / edge - 1
        out[k] = {}
        for name, sel in (("all", np.ones(len(mc), bool)), ("light", mc <= split), ("heavy", mc > split)):
            out[k][name] = dict(n=int(sel.sum()), within1_true=float(np.mean(np.abs(et[sel]) < 0.01)),
                                within1_edge=float(np.mean(np.abs(ee[sel]) < 0.01)),
                                median_err_true=float(np.median(et[sel])),
                                median_abs_err_edge=float(np.median(np.abs(ee[sel]))))
    # the same fractions predicted from t_c alone: |(t_c / 4)^(3/5) - 1| < tol
    out["edge_only_prediction"] = {str(t): float(np.mean(np.abs((tc / 4) ** 0.6 - 1) < t)) for t in TOLS}
    return out


def grid_loss(fn):
    """Noise-free runs: best-template |Z|^2 of (a) over SNR^2 times the share of
    the true chirp's 4 s-frame energy inside the window (the optimum)."""
    z = np.load(fn)
    mc, tc, snr, top = z["mc"], z["tc"], z["snr"], z["top_a"]
    tau = (L - np.arange(L)) / FS
    r = np.empty(len(mc))
    for i in range(len(mc)):
        a, _ = chirp(tau, mc[i])
        r[i] = top[i] / (snr[i] ** 2 * np.sum(a[tc[i] - tau >= 0] ** 2) / np.sum(a ** 2))
    return dict(median=float(np.median(r)), p5=float(np.percentile(r, 5)), min=float(r.min()), max=float(r.max()))


def compare(fa, fb, keys=("a", "b_0.44")):
    """Within-1% fractions on identical injections, and the paired difference fb - fa."""
    a, b = np.load(fa), np.load(fb)
    for q in ("mc", "tc", "ph", "snr"):
        assert np.array_equal(a[q], b[q]), q
    out = {}
    for k in keys:
        ha = np.abs(a["grid"][a[f"est_{k}"]] / a["mc"] - 1) < 0.01
        hb = np.abs(b["grid"][b[f"est_{k}"]] / b["mc"] - 1) < 0.01
        d = hb.astype(float) - ha
        out[k] = dict(first=float(ha.mean()), second=float(hb.mean()), diff=float(d.mean()),
                      diff_se=float(d.std(ddof=1) / np.sqrt(len(d))))
    return out


def core_width(fn):
    """Half the 16 to 84% range of ln(estimate / true Mc), per method (b, a)."""
    z = np.load(fn)
    out = {}
    for k in methods(tuple(float(r) for r in z["rs"])):
        if k.startswith("c"):
            continue
        e = np.log(z["grid"][z[f"est_{k}"]] / z["mc"])
        lo, hi = np.percentile(e, [16, 84])
        out[k] = float((hi - lo) / 2)
    out["grid_quantization_sd"] = float(np.log(float(z["step"])) / np.sqrt(12))
    return out


def calibrated(fn, keys=("c_0.44", "c_1.0", "c_2.0", "c_4.0")):
    """Constant rescaling of an estimate that maximizes within 1% on the
    even-numbered injections, scored on the odd-numbered ones."""
    z = np.load(fn)
    ks = np.exp(np.linspace(-0.02, 0.15, 3401))
    out = {}
    for k in keys:
        r = z["grid"][z[f"est_{k}"]] / z["mc"]
        tr, te = r[0::2], r[1::2]
        kb = ks[int(np.argmax([(np.abs(kk * tr - 1) < 0.01).mean() for kk in ks]))]
        out[k] = dict(factor=float(kb), held_out_within1=float((np.abs(kb * te - 1) < 0.01).mean()),
                      argmax_within1_same_half=float((np.abs(te - 1) < 0.01).mean()))
    return out


def edge_cap(tol=0.01):
    """Largest share within tol of any constant multiple of (t_c / 4)^(3/5),
    t_c uniform on [3.5, 4] s: x has density (40/3) x^(2/3) on [0.875^0.6, 1],
    increasing, so the best window is [(1 - tol) / (1 + tol), 1]."""
    x1 = (1 - tol) / (1 + tol)
    return float(8 * (1 - x1 ** (5 / 3)))


def deflection(fn_noisefree, checks_json):
    """Signal part of c_R at the best template (noise-free run, SNR 10) over
    the noise standard deviation of c_R from the noise-only check."""
    z = np.load(fn_noisefree)
    chk = json.load(open(checks_json))
    out = {}
    for R in RS:
        top = z[f"stat_c_{R}"].astype(float).max(1)
        sd = chk[f"noise_c_{R}"]["sd_over_noise"]
        out[str(R)] = dict(signal_median=float(np.median(top)), noise_sd=float(sd), ratio=float(np.median(top) / sd))
    return out


def post(out_file):
    d = "sim_out/"
    res = dict(edge_noisefree=edge_mechanism(d + "noisefree.npz"),
               edge_snr25=edge_mechanism(d + "snr25.npz"),
               grid_loss_noisefree=grid_loss(d + "noisefree.npz"),
               spacing_8_10=compare(d + "band_8_10.npz", d + "fine_8_10.npz"),
               spacing_10_12=compare(d + "band_10_12.npz", d + "fine_10_12.npz"),
               core_width={f: core_width(d + f + ".npz") for f in
                           ("fine_8_10", "fine_10_12", "band_8_10", "band_10_12", "band_12_16", "band_16_25")})
    # R = 4 s segment path against the coherent statistic on every main-run window
    eq = {}
    for f in ("band_8_10", "band_10_12", "band_12_16", "band_16_25", "snr25", "noisefree"):
        z = np.load(d + f + ".npz")
        eq[f] = dict(same_estimate=int(np.sum(z["est_b_4.0"] == z["est_a"])), n=int(len(z["mc"])),
                     max_abs_stat_diff=float(np.max(np.abs(z["stat_b_4.0"] - z["stat_a"]))))
    res["b4_equals_a"] = eq
    res["calibrated_c"] = {f: calibrated(d + f + ".npz") for f in
                           ("noisefree", "band_8_10", "band_10_12", "band_12_16", "band_16_25", "snr25")}
    res["edge_cap_within1"] = edge_cap()
    res["c_deflection_snr10"] = deflection(d + "noisefree.npz", d + "checks.json")
    json.dump(res, open(out_file, "w"), indent=1)
    print(json.dumps(res, indent=1))


def main():
    cmd = sys.argv[1]
    if cmd == "run":
        lo, hi, n, seed, out = float(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), sys.argv[6]
        step = float(sys.argv[7]) if len(sys.argv) > 7 else STEP
        rs = RS if len(sys.argv) <= 8 else (() if sys.argv[8] == "none" else tuple(float(r) for r in sys.argv[8].split(",")))
        simulate(lo, hi, n, seed, out, step, rs)
    elif cmd == "noisefree":
        simulate(10.0, 10.0, int(sys.argv[2]), int(sys.argv[3]), sys.argv[4], with_noise=False)
    elif cmd == "checks":
        check(sys.argv[2])
    elif cmd == "report":
        report(sys.argv[2], sys.argv[3:])
    elif cmd == "figure":
        figure(sys.argv[2], sys.argv[3:])
    elif cmd == "post":
        post(sys.argv[2])
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
