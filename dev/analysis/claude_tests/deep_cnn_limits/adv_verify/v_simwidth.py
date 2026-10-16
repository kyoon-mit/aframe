"""Event-wise prediction of the sim's core widths (half the 16-84% range of
ln(Mc_est/Mc)) on the 0.1% grid runs, from Fisher-type information with each
event's own in-window band, plus the two quantizations the sim has:
  - Mc grid (step 1.001): uniform offset;
  - whole-sample lags (0.488 ms): t on a grid feeds into ln Mc through the
    objective's Hessian slope H_Mt / H_MM.
For (b) two versions: the Fisher bound (one free phase per piece, shared t_c)
and the sandwich variance of the sim's actual statistic sum_k |Z_k|^2 (pieces
weighted by their energy share), which is not the maximum-likelihood statistic.
"""
import numpy as np

TSUN = 4.925490947e-6
FS = 2048.0
PSD_FILE = "/home/kyoon/SSM-BNS/aframe/dev/runs/claude_tests/template_bank/bank_mc0.5pct_spin0_q0.8.npz"
D = "/tmp/claude-1024/-home-kyoon-SSM-BNS/82effc75-bb98-44ad-af20-e41e39a7258d/scratchpad/cnnproof/sim_out/"
p = np.load(PSD_FILE)["psd"]
FP = np.arange(p.shape[1]) * 0.5
SN = 2 / (1 / p[0] + 1 / p[1])


def info(mc, tc, R=0.44, n=20001):
    """Return per-event 2x2 matrices for (lnMc, t_c) at unit SNR:
    coherent Fisher, segmented Fisher (free phase per piece), and the
    power-sum Hessian H and noise V (sum_k P_k^2 C_k, sum_k P_k^3 C_k)."""
    flo = (5 / (256 * tc)) ** 0.375 * (mc * TSUN) ** -0.625 / np.pi
    fhi = min(1 / (6 ** 1.5 * np.pi * TSUN * 2 ** 1.2 * mc), FS / 2)
    f = np.geomspace(flo, fhi, n)
    q = np.gradient(f)
    w = q * f ** (-7 / 3) / np.interp(f, FP, SN)
    w /= w.sum()
    a = -(5 / 128) * (np.pi * mc * TSUN * f) ** (-5 / 3)
    b = 2 * np.pi * f
    V = np.stack([a, b])
    mu = V @ w
    coh = (V - mu[:, None]) * w @ (V - mu[:, None]).T
    tau = 5 / 256 * (mc * TSUN) ** (-5 / 3) * (np.pi * f) ** (-8 / 3)
    seg = np.floor(tau / R).astype(int)
    G = np.zeros((2, 2)); H = np.zeros((2, 2)); Vn = np.zeros((2, 2))
    for k in np.unique(seg):
        m = seg == k
        Pk = w[m].sum()
        if Pk <= 0:
            continue
        muk = V[:, m] @ w[m] / Pk
        Ck = (V[:, m] - muk[:, None]) * w[m] @ (V[:, m] - muk[:, None]).T / Pk
        G += Pk * Ck
        H += Pk ** 2 * Ck
        Vn += Pk ** 3 * Ck
    return coh, G, H, Vn


def main():
    rng = np.random.default_rng(0)
    for name in ("fine_8_10", "fine_10_12"):
        z = np.load(D + name + ".npz")
        mc, tc, snr, grid = z["mc"], z["tc"], z["snr"], z["grid"]
        err = {k: np.log(grid[z[f"est_{k}"]] / mc) for k in ("a", "b_0.44")}
        n = len(mc)
        sa, sb, sps, slope_a, slope_b, slope_ps = (np.empty(n) for _ in range(6))
        for i in range(n):
            coh, G, H, Vn = info(mc[i], tc[i])
            r2 = snr[i] ** 2
            Ca = np.linalg.inv(coh * r2)
            Cb = np.linalg.inv(G * r2)
            Hi = np.linalg.inv(H)
            Cps = Hi @ Vn @ Hi / r2
            sa[i], sb[i], sps[i] = np.sqrt(Ca[0, 0]), np.sqrt(Cb[0, 0]), np.sqrt(Cps[0, 0])
            slope_a[i] = coh[0, 1] / coh[0, 0]            # Hessian slope H_Mt/H_MM
            slope_b[i] = G[0, 1] / G[0, 0]
            slope_ps[i] = H[0, 1] / H[0, 0]
        dt = 1 / FS
        lstep = np.log(float(z["step"]))

        def predicted(sig, slope, reps=200):
            e = (sig[None] * rng.standard_normal((reps, n))
                 - slope[None] * rng.uniform(-dt / 2, dt / 2, (reps, n)))
            # nearest grid point in ln Mc with a uniform offset of the truth
            off = rng.uniform(-0.5, 0.5, (reps, n)) * lstep
            e = np.round((e - off) / lstep) * lstep + off
            lo, hi = np.percentile(e, [16, 84], axis=1)
            return float(np.mean((hi - lo) / 2))

        def hw(e):
            lo, hi = np.percentile(e, [16, 84])
            return float((hi - lo) / 2)

        print(f"\n{name}: n {n}, mean SNR {snr.mean():.2f}")
        print(f"  measured core half-width: a {100 * hw(err['a']):.4f}%, b_0.44 {100 * hw(err['b_0.44']):.4f}%,"
              f" ratio {hw(err['b_0.44']) / hw(err['a']):.2f}")
        print(f"  median event sigma: coherent {100 * np.median(sa):.4f}%, seg Fisher {100 * np.median(sb):.4f}%,"
              f" power-sum sandwich {100 * np.median(sps):.4f}%; median ratio seg/coh {np.median(sb / sa):.2f},"
              f" powersum/seg {np.median(sps / sb):.3f}")
        print(f"  predicted (Gaussian, no quantization): a {100 * predicted(sa, 0 * slope_a) if False else 100 * hw(sa * rng.standard_normal(n)):.4f}%")
        pa0 = predicted(sa, 0 * slope_a)
        pa = predicted(sa, slope_a)
        pb = predicted(sb, slope_b)
        pps = predicted(sps, slope_ps)
        print(f"  predicted a: Mc grid only {100 * pa0:.4f}%, Mc grid + lag grid {100 * pa:.4f}%")
        print(f"  predicted b: Fisher bound {100 * pb:.4f}%, power-sum sandwich {100 * pps:.4f}%")
        print(f"  predicted ratio b/a: Fisher {pb / pa:.2f}, power-sum {pps / pa:.2f}")
        # normalized errors, events within 1%
        for k, s in (("a", sa), ("b_0.44", sb)):
            m = np.abs(err[k]) < 0.01
            print(f"  {k}: robust sd of err/sigma (within 1% events) {1.4826 * np.median(np.abs(err[k][m] / s[m])):.2f}")


if __name__ == "__main__":
    main()
