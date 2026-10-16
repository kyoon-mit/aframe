"""Extra checks for (B).

1. The orbit terms T_k = (1/N) sum_alpha S_alpha^2 / alpha! (canonical multi-indices of
   degree k with support in [0, R-1] containing 0, S_alpha = sum_t prod_j s_{t+j}^alpha_j)
   by brute-force enumeration for k = 2, 3, 4 and R = 49, against
   (a) the closed form T_2 = sum_{|d|<R} C(d)^2 / (2N) and
   (b) the Gram-matrix bounds T_k <= s^T K^(k-1) s / (N (k-1)!), K(t,t') = sum_{j<R} s_{t+j} s_{t'+j}.
2. Monte Carlo of the best second-order statistic for R = 49,
   o_R(x) = sum_{|d|<R} C(d) sum_t x_t x_{t+d} (circular): its SNR must equal sqrt(T_2).
3. The stride-4 network of b_cnn_mc.py: first-order response to the 512 Hz tone (on the comb)
   for the four sub-stride positions of the tone, and their average, from a = E[grad g].
Output: out/b_checks_extra.txt
"""
import sys, itertools, os, math
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np
sys.path.insert(0, ".")
from thm_signals import N

lines = []
def P(s=""):
    print(s, flush=True); lines.append(s)

s = np.load("out/sig_whitened_bns.npy")
rho = 8.0
x = rho * s
R = 49
rolls = np.stack([np.roll(x, -j) for j in range(R)])        # rolls[j][t] = x_{t+j}

def T_brute(k):
    tot = 0.0
    for rest in itertools.combinations_with_replacement(range(R), k - 1):
        u = (0,) + rest
        prod = np.ones(N)
        for j in u:
            prod = prod * rolls[j]
        S = prod.sum()
        mult = np.bincount(np.array(u), minlength=R)
        af = np.prod([math.factorial(m) for m in mult if m > 1]) if any(mult > 1) else 1
        tot += S * S / af
    return tot / N

def gram_bound(k):
    # s^T K^(k-1) s / (N (k-1)!) with elementwise power, K(t, t+d) by sliding sums
    idx = np.arange(N)
    tot = 0.0
    for d0 in range(0, N, 256):
        ds = np.arange(d0, min(d0 + 256, N))
        p = x[None, :] * x[(idx[None, :] + ds[:, None]) % N]
        ext = np.concatenate([p, p[:, :R]], axis=1)
        cs = np.concatenate([np.zeros((len(ds), 1)), np.cumsum(ext, axis=1)], axis=1)
        K = cs[:, R:N + R] - cs[:, :N]
        tot += np.sum(p * K ** (k - 1))
    return tot / (N * math.factorial(k - 1))

C = np.fft.ifft(np.abs(np.fft.fft(x)) ** 2).real
d = np.arange(N); dd = np.minimum(d, N - d)
T2_closed = np.sum(C[dd < R] ** 2) / (2 * N)
P(f"(1) whitened BNS, rho = {rho}, R = {R}")
for k in (2, 3, 4):
    tb = T_brute(k)
    gb = gram_bound(k)
    extra = f"; closed form {T2_closed:.6f}" if k == 2 else ""
    P(f"    T_{k}: brute force {tb:.6e}{extra}; Gram bound {gb:.6e}")

# 2. MC of the best second-order local statistic
rng = np.random.default_rng(5)
m = np.where(dd < R, C / rho ** 2, 0.0)              # weights C_hat(d), |d| < R
mf = np.fft.fft(m).real                              # symbol of the circulant quadratic form
def oR(X):
    Xf = np.abs(np.fft.fft(X, axis=-1)) ** 2
    return (Xf * mf).sum(-1) / N                       # sum_{d} m(d) sum_t x_t x_{t+d}
vals0, vals1 = [], []
for _ in range(20):
    n = rng.standard_normal((500, N))
    vals0.append(oR(n)); vals1.append(oR(n + x))
v0 = np.concatenate(vals0); v1 = np.concatenate(vals1)
snr = (v1.mean() - v0.mean()) / v0.std(ddof=1)
P(f"(2) best second-order statistic for R = {R}: Monte Carlo SNR {snr:.4f} +- {snr/np.sqrt(2*len(v0)):.4f} (10,000 draws); sqrt(T_2) = {np.sqrt(T2_closed):.4f}")

# 3. stride-4 net: first-order response vs sub-stride position of a 512 Hz tone
try:
    import torch
    torch.set_num_threads(4)
    from b_cnn_mc import LocalNet, local_gradient, w_from_a, DT
    net = LocalNet(seed=1, act="relu", pool="mean", strides=(1, 2, 2)).to(DT)
    a, a_se = local_gradient(net)
    w = w_from_a(a, net.S, net.R)
    # sigma_o of this net from b_cnn_mc output if present
    import json
    sig = None
    if os.path.exists("out/b_cnn_mc.json"):
        J = json.load(open("out/b_cnn_mc.json"))
        for key, v in J.items():
            if key.startswith("avg ReLU stride 4"):
                sig = v["sigma_o"]; break
    t = np.arange(N) / 2048.0
    vals = []
    for shift in range(4):
        tone = np.cos(2 * np.pi * 512.0 * (t - shift / 2048.0) + 0.3)
        tone /= np.linalg.norm(tone)
        vals.append(w @ tone)
    vals = np.array(vals)
    P(f"(3) stride-4 net, 512 Hz tone shifted by 0..3 samples: w.s = {['%+.3e' % v for v in vals]}; mean {vals.mean():+.2e}"
      + (f"; in SNR per unit amplitude: {['%+.4f' % (v/sig) for v in vals]}, mean {vals.mean()/sig:+.5f}" if sig else ""))
    toneoff = np.cos(2 * np.pi * 500.0 * t + 0.3); toneoff /= np.linalg.norm(toneoff)
    P(f"    500 Hz tone (off the comb): w.s = {w @ toneoff:+.3e}" + (f" ({(w @ toneoff)/sig:+.5f} per unit amplitude)" if sig else ""))
except Exception as e:
    P(f"(3) skipped: {e!r}")

open("out/b_checks_extra.txt", "w").write("\n".join(lines) + "\n")
