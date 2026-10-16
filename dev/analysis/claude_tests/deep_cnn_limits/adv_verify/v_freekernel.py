"""How far is the trained free-kernel network from time-shift invariance?

1. Kernel energy over lag: median channel lag holding 50/90% of |k|^2 (claim 1.60/3.25 s).
2. Noise-only argmax time t* of each channel's energy (one detector, white noise
   above 20 Hz): its distribution over the 4 s (uniform under invariance).
3. Regular part of the second-order response, per channel, R_c(s) =
   E_n |y_{s,c}(t*_c(n))|^2 with y_s = k_c * s (the network's causal FFT
   convolution), for the theorem's whitened BNS chirp (merger 3.75 s) and the
   same chirp played backwards. Under invariance (Theorem 2) both equal the
   time mean of |y_s|^2. Reports the ratio forward/backward.
"""
import numpy as np
import torch

CK = ("/home/kyoon/SSM-BNS/aframe/dev/runs/claude_tests/free_kernel_energy_max_256ch_softmax_train_snr8_prob0.96/"
      "CLAUDE-TESTS/free_kernel_energy_max_256ch_softmax_train_snr8_prob0.96/checkpoints/ck_248.ckpt")
SIG = "/tmp/claude-1024/-home-kyoon-SSM-BNS/82effc75-bb98-44ad-af20-e41e39a7258d/scratchpad/cnnproof/out/sig_whitened_bns.npy"
L, DEC = 4096, 2
torch.set_num_threads(4)

sd = torch.load(CK, map_location="cpu", weights_only=False)["state_dict"]
spec = sd["model.spectrum"].double()
freqs = torch.fft.fftfreq(L, DEC / 2048.0)
band = (freqs >= 20) & (freqs < 512)
k = torch.fft.ifft(torch.view_as_complex(spec.contiguous()) * band, dim=-1)
k = k / k.abs().pow(2).sum(-1, keepdim=True).sqrt()
C = k.shape[0]
cum = (k.abs() ** 2).cumsum(-1).numpy()
lag50 = np.median(np.argmax(cum >= 0.5, 1)) / 1024
lag90 = np.median(np.argmax(cum >= 0.9, 1)) / 1024
print(f"kernel energy over lag, median channel: 50% within {lag50:.2f} s, 90% within {lag90:.2f} s (claim 1.60/3.25 s)")
K = torch.fft.fft(k, n=2 * L)


def resample(x):
    X = torch.fft.rfft(torch.as_tensor(x, dtype=torch.float64), n=L * DEC)[..., : L // 2 + 1]
    return torch.fft.irfft(X, n=L) / np.sqrt(DEC)


def conv(x):
    """(B, L) -> (B, C, L) complex causal convolution, first L outputs."""
    return torch.fft.ifft(torch.fft.fft(x, n=2 * L)[:, None] * K[None], n=2 * L)[..., :L]


s = np.load(SIG)
sf = resample(s[None])
sb = resample(s[::-1].copy()[None])
yf = conv(sf)[0].abs() ** 2          # (C, L)
yb = conv(sb)[0].abs() ** 2
rng = np.random.default_rng(1)
f2 = np.fft.rfftfreq(8192, 1 / 2048)
tstar = []
for i in range(25):
    n = rng.standard_normal((16, 8192))
    Nf = np.fft.rfft(n)
    Nf[:, f2 < 20] = 0
    x = resample(np.fft.irfft(Nf, n=8192))
    e = conv(x).abs() ** 2
    tstar.append(e.argmax(-1).numpy())  # (16, C)
tstar = np.concatenate(tstar)            # (400, C)
h = np.histogram(tstar.ravel() / 1024, bins=[0, 1, 2, 3, 3.5, 3.75, 4.0])[0] / tstar.size
print("noise argmax time t* distribution over bins [0,1,2,3,3.5,3.75,4] s:", np.round(h, 3),
      "(uniform: 0.25 0.25 0.25 0.125 0.0625 0.0625)")
Rf = np.array([yf[c, tstar[:, c]].mean().item() for c in range(C)])
Rb = np.array([yb[c, tstar[:, c]].mean().item() for c in range(C)])
uni_f = yf.mean(-1).numpy()
uni_b = yb.mean(-1).numpy()
w = uni_f / uni_f.sum()
print(f"time-mean |y|^2 forward/backward (invariant prediction would be ~1): median channel {np.median(uni_f / uni_b):.3f}")
print(f"regular 2nd-order response forward/backward, R_c(fwd)/R_c(bwd): median {np.median(Rf / Rb):.2f},"
      f" 10-90% {np.percentile(Rf / Rb, 10):.2f}-{np.percentile(Rf / Rb, 90):.2f};"
      f" sum over channels {Rf.sum() / Rb.sum():.2f}")
top = np.argsort(uni_f)[-20:]
print(f"20 channels with the largest forward response: ratio median {np.median(Rf[top] / Rb[top]):.2f}")
print(f"forward R_c relative to the time mean: median {np.median(Rf / uni_f):.2f}; backward {np.median(Rb / uni_b):.2f}")
