"""Independent check of the ten-mass-grid heterodyne numbers."""
import math
import numpy as np

GM_SUN_C3 = 1.32712440018e20 / (299792458.0 ** 3)   # s, IAU nominal GM_sun / c^3
print("GM_sun/c^3 =", GM_SUN_C3)

def tc(mc):
    return GM_SUN_C3 * mc

def freq(tau, mc):
    # leading-order GW frequency at time tau before merger
    return (1.0 / math.pi) * (5.0 / (256.0 * tau)) ** 0.375 * tc(mc) ** (-0.625)

def phi(tau, mc):
    # leading-order GW phase (up to a constant): -2 (tau / (5 Tc))^(5/8)
    return -2.0 * (tau / (5.0 * tc(mc))) ** 0.625

# grid
lo, hi, n = 0.85, 2.75, 10
r = (hi / lo) ** (1.0 / (n - 1))
ep_hi = math.sqrt(r) - 1
ep_lo = 1 / math.sqrt(r) - 1
print(f"r = {r:.6f}, sqrt r = {math.sqrt(r):.6f}, eps+ = {100*ep_hi:.4f}%, eps- = {100*ep_lo:.4f}%")
grid = lo * r ** np.arange(n)
true = np.exp(np.linspace(math.log(lo), math.log(hi), 1_000_001))
idx = np.argmin(np.abs(np.log(true)[:, None] - np.log(grid)[None, :]), axis=1)
eps = true / grid[idx] - 1
print(f"scan 1,000,001: eps {100*eps.min():.4f}% .. {100*eps.max():.4f}%")
q = lambda e: abs(1 - (1 + e) ** 0.625)
for e in (ep_hi, ep_lo, -ep_hi, 0.01, -0.01):
    print(f"eps {100*e:+.3f}%: q = {100*q(e):.4f}%, first-order {62.5*abs(e):.4f}%, q^-1/2 = {q(e)**-0.5:.4f}")

f4 = freq(4.0, 1.2)
print(f"f(4 s; 1.2) = {f4:.4f} Hz")
for e in (ep_hi, ep_lo):
    print(f"  leftover at entry {q(e)*f4:.4f} Hz, at 512 Hz {q(e)*512:.4f} Hz")

# Fourier tone counts: complex A exp(i psi), 4 s at 2048 Hz, merger at end, A ~ tau^-1/4 while 20<=f<=900
fs = 2048.0
L = 8192
k = np.arange(L)
tau = (L - k) / fs            # 4.0 ... 1/fs
def count(sig):
    P = np.abs(np.fft.fft(sig)) ** 2
    P = np.sort(P)[::-1]
    c = np.cumsum(P) / P.sum()
    kk = int(np.argmax(c >= 0.81)) + 1
    assert math.sqrt(c[kk - 1]) >= 0.9 and math.sqrt(c[kk - 2]) < 0.9
    return kk
out = {}
for mc in (1.0, 1.4, 1.8, 2.4):
    f = np.array([freq(t_, mc) for t_ in tau])
    A = np.where((f >= 20.0) & (f <= 900.0), tau ** -0.25, 0.0)
    ph = np.array([phi(t_, mc) for t_ in tau])
    row = [count(A * np.exp(1j * ph))]
    for e in (ep_hi, ep_lo, -ep_hi, 0.01, -0.01):
        ph2 = np.array([phi(t_, mc / (1 + e)) for t_ in tau])
        row.append(count(A * np.exp(1j * (ph - ph2))))
    out[mc] = row
    print(f"Mc {mc}: before {row[0]}, +6.740% {row[1]}, -6.315% {row[2]}, -6.740% {row[3]}, +1% {row[4]}, -1% {row[5]}")

# in-step factor on the exact phase at 1.2 Msun: interval around tau0 where |dphase - tangent| <= pi/4
def interval(tau0, mc, scale):
    s = np.linspace(-8.0, tau0 - 1e-7, 4_000_001)
    tt = tau0 - s
    ph = -2.0 * (tt / (5.0 * tc(mc))) ** 0.625 * scale
    ph0 = -2.0 * (tau0 / (5.0 * tc(mc))) ** 0.625 * scale
    w0 = 2 * math.pi * freq(tau0, mc) * scale
    dev = np.abs(ph - ph0 - w0 * s)
    i0 = np.argmin(np.abs(s))
    bad = np.flatnonzero(dev > math.pi / 4)
    a = bad[bad < i0].max() + 1
    b = bad[bad > i0].min() - 1 if np.any(bad > i0) else len(s) - 1
    return s[b] - s[a]
for tau0 in (0.5, 1.0, 2.0, 3.0, 4.0):
    Tc_ = interval(tau0, 1.2, 1.0)
    print(f"tau0 {tau0}: chirp {Tc_:.4f} s, ratios " + ", ".join(f"{interval(tau0, 1.2, q(e))/Tc_:.4f}" for e in (ep_hi, ep_lo)))
