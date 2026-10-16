"""Independent maximum-likelihood Monte Carlo (own code) for one cell of
fisher.md section 5(b): Mc = 1.2, network SNR 8, fraction within 1% for
coherent, one phase per 0.44 s segment with shared time, and phase and time
per segment. Frequency domain, 0.25 Hz bins over the true band, complex noise
with unit variance per part, template grid step 0.001 in ln Mc over the full
prior, time grid 0.488 ms over |t| <= 0.25 s, segment edges of the true signal.
Claims (500 realizations): 99.8 / 94.6 / 52.6 %.
usage: v_mlmc.py NREAL SEED [MC] [SNR]
"""
import sys
import numpy as np

TSUN = 4.925490947641267e-06
p = np.load("/home/kyoon/SSM-BNS/aframe/dev/runs/claude_tests/template_bank/bank_mc0.5pct_spin0_q0.8.npz")["psd"]
FP = np.arange(p.shape[1]) * 0.5
SN = 2 / (1 / p[0] + 1 / p[1])


def main():
    nreal, seed = int(sys.argv[1]), int(sys.argv[2])
    mc0 = float(sys.argv[3]) if len(sys.argv) > 3 else 1.2
    rho = float(sys.argv[4]) if len(sys.argv) > 4 else 8.0
    df, nfft = 0.25, 8192
    f4 = (5 / (256 * 4.0)) ** 0.375 * (mc0 * TSUN) ** -0.625 / np.pi
    fhi = min(1 / (6 ** 1.5 * np.pi * TSUN * 2 ** 1.2 * mc0), 1024.0)
    f = np.arange(np.ceil(f4 / df), np.floor(fhi / df) + 1) * df
    w = np.sqrt(f ** (-7 / 3) / np.interp(f, FP, SN))
    w /= np.linalg.norm(w)                                            # unit-norm template amplitude
    tau = 5 / 256 * (mc0 * TSUN) ** (-5 / 3) * (np.pi * f) ** (-8 / 3)
    seg = np.floor(tau / 0.44).astype(int)
    segs = np.unique(seg)
    lgrid = np.arange(np.log(0.85), np.log(2.75), 0.001)
    psi = lambda lm: (3 / 128) * (np.pi * np.exp(lm)[:, None] * TSUN * f[None]) ** (-5 / 3)
    H = w[None] * np.exp(1j * psi(lgrid))                             # (T, F) templates at t = 0
    j0 = np.arange(len(f))
    lag = np.r_[0:513, nfft - 512:nfft]                               # |t| <= 0.25 s
    rng = np.random.default_rng(seed)
    hit = np.zeros(3)
    for r in range(nreal):
        phi = rng.uniform(0, 2 * np.pi)
        sig = rho * w * np.exp(1j * (psi(np.array([np.log(mc0)]))[0] + phi))
        d = sig + (rng.standard_normal(len(f)) + 1j * rng.standard_normal(len(f)))
        best = np.full(3, -np.inf)
        arg = np.zeros(3, int)
        for t0 in range(0, len(lgrid), 64):
            Hc = H[t0:t0 + 64]
            prod = np.conj(Hc) * d[None]                              # (B, F)
            # Z(t) = sum_j prod_j exp(+2 pi i f_j t); shift-free part through j
            full = np.fft.ifft(np.pad(prod, ((0, 0), (0, nfft - len(f)))), axis=-1)[:, lag] * nfft
            per = np.zeros((prod.shape[0], len(segs), len(lag)), complex)
            for q, k in enumerate(segs):
                pk = np.where(seg[None] == k, prod, 0)
                per[:, q] = np.fft.ifft(np.pad(pk, ((0, 0), (0, nfft - len(f)))), axis=-1)[:, lag] * nfft
            stats = [np.abs(full).max(-1), np.abs(per).sum(1).max(-1), np.abs(per).max(-1).sum(-1)]
            for m in range(3):
                i = int(np.argmax(stats[m]))
                if stats[m][i] > best[m]:
                    best[m], arg[m] = stats[m][i], t0 + i
        hit += np.abs(lgrid[arg] - np.log(mc0)) < np.log(1.01)
        if r % 50 == 0:
            print(r, hit / (r + 1), flush=True)
    se = np.sqrt(hit / nreal * (1 - hit / nreal) / nreal)
    print(f"Mc {mc0}, SNR {rho}, {nreal} realizations: within 1% coherent {100 * hit[0] / nreal:.1f} +- {100 * se[0]:.1f},"
          f" shared time {100 * hit[1] / nreal:.1f} +- {100 * se[1]:.1f}, free time {100 * hit[2] / nreal:.1f} +- {100 * se[2]:.1f}")


if __name__ == "__main__":
    main()
