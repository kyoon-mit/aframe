"""Does the abrupt window start explain why leftover counts exceed q times the chirp's count?"""
import numpy as np
T_SUN = 4.925490947e-6
def tm(mc): return T_SUN * mc
def f_of_tau(tau, mc): return (5 / tau) ** (3 / 8) * tm(mc) ** (-5 / 8) / (8 * np.pi)
def phase(tau, mc): return -2 * (tau / (5 * tm(mc))) ** (5 / 8)
def count(sig):
    p = np.sort(np.abs(np.fft.fft(sig)) ** 2)[::-1]
    c = np.cumsum(p) / p.sum()
    return int(np.searchsorted(c, 0.81) + 1)
w = np.sqrt((2.75 / 0.85) ** (1 / 9)) - 1
q = 1 - (1 + w) ** (5 / 8); q = abs(q)
fs, L = 2048.0, 8192
t = np.arange(L) / fs
for ramp in (0.0, 0.25, 0.5, 1.0):
    taper = np.ones(L)
    n = int(ramp * fs)
    if n:
        taper[:n] = 0.5 * (1 - np.cos(np.pi * np.arange(n) / n))   # smooth onset at the window start
    out = []
    for mc in (1.0, 1.4, 1.8, 2.4):
        tau = np.maximum(4.0 - t, 0.5 / fs); f = f_of_tau(tau, mc)
        A = np.where((f >= 20) & (f <= 900), tau ** -0.25, 0.0) * taper
        c0 = count(A * np.exp(1j * phase(tau, mc)))
        c1 = count(A * np.exp(1j * (phase(tau, mc) - phase(tau, mc / (1 + w)))))
        out.append(f"Mc {mc}: chirp {c0}, q*chirp {q*c0:.1f}, leftover {c1}")
    print(f"onset ramp {ramp} s: " + "; ".join(out))
