"""Independent numbers for Theorems 2 and 3 on the theorem's own test signal
(out/sig_whitened_bns.npy, unit norm), rho = 8, circular window, R = 49:
  T1, T2 (closed form), T3 and T4 by brute-force enumeration of orbits,
  the third-term bound (1/N) sum s_t s_t' (e^K - 1 - K), and the second-order
  bound over all invariant statistics rho^2/sqrt(2 N_eff).
Also the Newtonian 1% chirp-mass separation bound at second order."""
import itertools
import math
import sys
import numpy as np

sys.path.insert(0, "/tmp/claude-1024/-home-kyoon-SSM-BNS/82effc75-bb98-44ad-af20-e41e39a7258d/scratchpad/cnnproof")
D = "/tmp/claude-1024/-home-kyoon-SSM-BNS/82effc75-bb98-44ad-af20-e41e39a7258d/scratchpad/cnnproof/out/"
N, R, rho = 8192, 49, 8.0
s = rho * np.load(D + "sig_whitened_bns.npy")
C = np.array([np.dot(s, np.roll(s, -d)) for d in range(N)])        # circular autocorrelation, C[d] = sum_t s_t s_{t+d}
C2 = np.fft.irfft(np.abs(np.fft.rfft(s)) ** 2, n=N)
print("autocorr FFT vs direct max diff", np.abs(C - C2).max())
T1 = s.sum() ** 2 / N
lagsR = np.r_[0:R, N - R + 1:N]
T2 = np.sum(C[lagsR] ** 2) / (2 * N)
T2all = np.sum(C ** 2) / (2 * N)
neff = N * rho ** 4 / np.sum(C ** 2)
print(f"T1 {T1:.3e}; T2(R=49) {T2:.6f} (claim 1.697317) -> SNR2 {np.sqrt(T2):.4f} (claim 1.303);"
      f" all invariant: {np.sqrt(T2all):.4f} (claim 1.354), N_eff {neff:.0f} (claim 1117)")

rolls = np.stack([np.roll(s, -j) for j in range(R)])              # rolls[j][t] = s_{t+j}


def brute(k):
    """T_k = sum over orbits (alpha = e_0 + beta, |beta| = k - 1, support in [0, R-1]) of S^2 / (N alpha!)."""
    tot = 0.0
    if k == 3:
        for i in range(R):
            for j in range(i, R):
                S = np.dot(s * rolls[i], rolls[j])
                cnt = {}
                for q in (0, i, j):
                    cnt[q] = cnt.get(q, 0) + 1
                tot += S * S / (N * math.prod(math.factorial(c) for c in cnt.values()))
    elif k == 4:
        for i in range(R):
            for j in range(i, R):
                v = s * rolls[i] * rolls[j]
                S = rolls[j:] @ v                                     # q = j..R-1
                for qi, q in enumerate(range(j, R)):
                    cnt = {}
                    for p in (0, i, j, q):
                        cnt[p] = cnt.get(p, 0) + 1
                    tot += S[qi] ** 2 / (N * math.prod(math.factorial(c) for c in cnt.values()))
    return tot


T3, T4 = brute(3), brute(4)
print(f"T3 brute {T3:.4e} (claim 3.123e-5); T4 brute {T4:.4e} (claim 3.241e-2)")

# third-term bound with K(t,t') = sum_{j<R} s_{t+j} s_{t'+j}
term3, k3, k4 = 0.0, 0.0, 0.0
B = 512
W = rolls.T                                                         # (N, R): W[t, j] = s_{t+j}
for a in range(0, N, B):
    K = W[a:a + B] @ W.T                                            # (B, N)
    E = np.expm1(K) - K
    term3 += s[a:a + B] @ E @ s
    k3 += s[a:a + B] @ (K ** 2 / 2) @ s
    k4 += s[a:a + B] @ (K ** 3 / 6) @ s
term3 /= N
k3 /= N
k4 /= N
print(f"bound on T3: {k3:.4e} (claim 3.180e-5); on T4: {k4:.4e} (claim 3.488e-2); all orders >= 3: {term3:.4e}")
print(f"SNR_max(R=49, rho=8) <= sqrt(T1+T2+term3) = {np.sqrt(T1 + T2 + term3):.4f} (claim 1.316)")
print(f"largest energy in 49 samples: {np.max(np.convolve(s ** 2, np.ones(R))):.3f} (claim 1.36);"
      f" in 899: {np.max(np.convolve(s ** 2, np.ones(899))):.2f} (claim 17.0)")

# Newtonian toy: 1% chirp-mass separation at second order, same merger time
from thm_signals import newtonian
s1 = rho * newtonian(1.2, 3.75)[0]
s2 = rho * newtonian(1.2 * 1.01, 3.75)[0]
A1 = np.fft.irfft(np.abs(np.fft.rfft(s1)) ** 2, n=N)
A2 = np.fft.irfft(np.abs(np.fft.rfft(s2)) ** 2, n=N)
print(f"Newtonian 1% pair, same merger time: 2nd-order separation bound {np.sqrt(np.sum((A1 - A2) ** 2) / (2 * N)):.4f}"
      " (claim 0.179)")
# coherent match over time shift and phase
Z = np.fft.ifft(np.fft.fft(s1 / rho) * np.conj(np.fft.fft(s2 / rho)))
# analytic signal of s2 needed for the phase maximum: use both quadratures via Hilbert transform
from scipy.signal import hilbert
h2 = hilbert(s2 / rho)
h2 /= np.sqrt(np.sum(np.abs(h2) ** 2) / 2)
Zc = np.fft.ifft(np.fft.fft(s1 / rho) * np.conj(np.fft.fft(h2)))
m = np.max(np.abs(Zc))
print(f"coherent match (circular shifts, phase maximized) {m:.3f} (claim 0.412) -> separation {rho * np.sqrt(2 * (1 - m)):.2f} (claim 8.68)")
