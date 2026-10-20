"""Numerical checks for the 20 October derivations."""
import numpy as np

rng = np.random.default_rng(0)
# per-dimension denoising loss of a Gaussian component with variance P
for P in (1e-4, 0.01, 1.0, 4.0):
    for sigma in (0.001, 0.1259, 0.631, 3.162):
        x = rng.normal(0, np.sqrt(P), 2_000_000)
        z = rng.normal(0, 1, x.size)
        xt = x + sigma * z
        opt = -sigma * xt / (P + sigma**2)
        white = -sigma * xt / (1 + sigma**2)
        lo, lw = np.mean((opt + z) ** 2), np.mean((white + z) ** 2)
        fo, fw = P / (P + sigma**2), (1 + sigma**2 * P) / (1 + sigma**2) ** 2
        print(f"P {P:7.4f} sigma {sigma:6.4f}: optimal {lo:.5f} (formula {fo:.5f}), "
              f"white {lw:.5f} (formula {fw:.5f})")
# dimensions below 20 Hz in a real FFT of 8192 samples at 2048 Hz
df = 2048 / 8192
bins = int(20 / df)            # bins 0 .. 79
print("bin width", df, "bins below 20 Hz", bins, "real dimensions", 1 + 2 * (bins - 1))
# PE cost
walkers, points, steps = 32, 2, 10000 + 20000
per_window = walkers * points * steps
print("waveform evaluations per window", per_window)
for label, sec in (("timing run, 2 windows, 40 steps: 46 s", 46 / 40 / 2),
                   ("live run, 4 windows: 2.918 s per step", 2.918 / 4)):
    h = sec * steps / 3600
    print(f"{label}: {sec:.4f} s per window-step, {h:.2f} h per window, "
          f"{400 * h / 24:.1f} days for 400")
print("examples", 200 * 1564 * 64, "paper", 500 * 40000)
