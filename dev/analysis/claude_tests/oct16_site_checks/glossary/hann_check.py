"""Check: first zero of the Fourier transform of a Hann window T long lies at f = 2/T."""
import numpy as np
fs = 2048.0
for T in (1.0, 0.5, 2.0):
    N = int(T * fs)
    n = np.arange(N)
    hann = np.sin(np.pi * n / N) ** 2          # periodic Hann, as torch.hann_window(N)
    rect = np.ones(N)
    pad = 64
    for name, w in (("hann", hann), ("rect", rect)):
        W = np.abs(np.fft.rfft(w, n=pad * N))
        f = np.fft.rfftfreq(pad * N, d=1 / fs)
        # first local minimum after the peak at f = 0
        i = 1
        while W[i + 1] < W[i]:
            i += 1
        print(f"T={T} s {name}: first zero at {f[i]:.4f} Hz = {f[i]*T:.4f}/T, |W| there / peak = {W[i]/W[0]:.2e}")
# torch agrees with the formula used above
import torch
w_t = torch.hann_window(2048, dtype=torch.float64).numpy()
print("max |torch.hann_window - sin^2(pi n/N)| =", np.abs(w_t - np.sin(np.pi*np.arange(2048)/2048)**2).max())
