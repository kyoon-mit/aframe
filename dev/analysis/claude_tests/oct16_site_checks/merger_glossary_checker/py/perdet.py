"""Per-detector extremes with the window's end at first sample + 4 s (8,192 samples)."""
import numpy as np
T_SUN = 4.925490947e-6
fs = 2048.0
def f_of_tau(tau, mc):
    return (5.0 / (256.0 * tau)) ** 0.375 * (mc * T_SUN) ** (-0.625) / np.pi
def tf(tau, mc):
    return 0.0 if tau <= 0 else tau * f_of_tau(tau, mc)
ms = np.arange(7126, 8236)
for mc in [0.85, 1.2, 1.8, 2.75]:
    N = np.array([1.6 * (tf(m / fs, mc) - tf((m - 8192) / fs, mc)) for m in ms])
    I = np.array([0.6 * (tf(m / fs, mc) - tf((m - 8192) / fs, mc)) for m in ms])
    f0 = np.array([f_of_tau(m / fs, mc) for m in ms])
    print(f"{mc}: start freq {f0.min():.3f}..{f0.max():.3f} Hz; cycles {N.min():.3f} (m={ms[N.argmin()]})..{N.max():.3f} (m={ms[N.argmax()]}); "
          f"int tau df {I.min():.3f}..{I.max():.3f} (max at m={ms[I.argmax()]}); at m=8235: N {N[-1]:.3f}, I {I[-1]:.3f}")
