"""Independent numpy (float64, CPU) re-implementation of the sim's statistics
(a), (b, 0.44 s) and (c, 0.44 s and 4 s), applied to the saved injections
(parameters read from the sim's npz, noise regenerated with the same seed
rule), comparing the argmax template event by event.
(c) is computed by Parseval in the frequency domain, a different route from
the sim's sum over lags.
usage: v_simrepro.py RUN.npz SEED N_EVENTS [noisefree]
"""
import sys
import numpy as np

FS, L, N = 2048.0, 8192, 16384
TSUN = 4.925490947e-6
PSD_FILE = "/home/kyoon/SSM-BNS/aframe/dev/runs/claude_tests/template_bank/bank_mc0.5pct_spin0_q0.8.npz"
p = np.load(PSD_FILE)["psd"]
FP = np.arange(p.shape[1]) * 0.5
SN = 2 / (1 / p[0] + 1 / p[1])


def wf(tau, mc):
    """Whitened leading-order chirp: amplitude f^(2/3)/sqrt(S), phase; zero for tau <= 0 or f >= f_hi."""
    fhi = min(1 / (6 ** 1.5 * np.pi * TSUN * 2 ** 1.2 * mc), FS / 2)
    amp = np.zeros_like(tau)
    ph = np.zeros_like(tau)
    pos = tau > 0
    f = (5 / (256 * tau[pos])) ** 0.375 * (TSUN * mc) ** -0.625 / np.pi
    ok = f < fhi
    idx = np.flatnonzero(pos)[ok]
    amp[idx] = f[ok] ** (2 / 3) / np.sqrt(np.interp(f[ok], FP, SN))
    ph[pos] = -2 * (tau[pos] / (5 * TSUN * mc)) ** 0.625
    return amp, ph


def main():
    fn, seed, nev = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    noisefree = len(sys.argv) > 4
    z = np.load(fn)
    grid = 0.85 * 1.005 ** np.arange(236)
    assert np.allclose(grid, z["grid"])
    u = np.arange(L)
    tau_t = (L - u) / FS
    piece = np.floor((L - 1 - u) / (0.44 * FS)).astype(int)
    npc = piece.max() + 1
    Hfull = np.empty((236, N), complex)
    Hpc = np.empty((236, npc, N), complex)
    G044 = np.zeros((236, N))
    for j, mc in enumerate(grid):
        a, ph = wf(tau_t, mc)
        h = a * np.exp(1j * ph)
        h *= np.sqrt(2 / np.sum(np.abs(h) ** 2))
        hp = np.zeros(N, complex)
        hp[:L] = h
        Hfull[j] = np.conj(np.fft.fft(hp))
        for k in range(npc):
            hk = np.zeros(N, complex)
            hk[:L][piece == k] = h[piece == k]
            Hpc[j, k] = np.conj(np.fft.fft(hk))
        G044[j] = (np.abs(Hpc[j]) ** 2).sum(0)
    G4 = np.abs(Hfull) ** 2
    lags = np.arange(-1024, 1)
    agree = {"a": 0, "b_0.44": 0, "c_0.44": 0, "c_4.0": 0}
    within = {k: 0 for k in agree}
    for i in range(nev):
        tau = z["tc"][i] - u / FS
        a, ph = wf(tau, z["mc"][i])
        s = a * np.cos(ph + z["ph"][i])
        s *= z["snr"][i] / np.sqrt(np.sum(s * s))
        x = s if noisefree else s + np.random.default_rng([seed, i]).standard_normal(L)
        xp = np.zeros(N)
        xp[:L] = x
        X = np.fft.fft(xp)
        P = np.abs(X) ** 2
        est = {}
        Za = np.fft.ifft(X[None] * Hfull)[:, lags % N]
        est["a"] = int(np.argmax((np.abs(Za) ** 2).max(1)))
        b = np.empty(236)
        for j in range(236):
            Zk = np.fft.ifft(X[None] * Hpc[j])[:, lags % N]
            b[j] = (np.abs(Zk) ** 2).sum(0).max()
        est["b_0.44"] = int(np.argmax(b))
        est["c_0.44"] = int(np.argmax(G044 @ P / N / L))
        est["c_4.0"] = int(np.argmax(G4 @ P / N / L))
        for k in agree:
            agree[k] += int(est[k] == int(z[f"est_{k}"][i]))
            within[k] += int(abs(grid[est[k]] / z["mc"][i] - 1) < 0.01)
        if i % 20 == 0:
            print(i, {k: agree[k] for k in agree}, flush=True)
    print(f"{fn}: {nev} events; same argmax as the sim: {agree}; within 1% (this code): {within}")


if __name__ == "__main__":
    main()
