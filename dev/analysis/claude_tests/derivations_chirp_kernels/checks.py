"""Numerical checks for the chirp-kernel page (toy setup of chirpfigs.py)."""
import math
import os

import h5py
import numpy as np
from scipy.special import erf

FS, T = 2048, 4.0
L = int(FS * T)
GM_SUN = 1.3271244e20      # IAU 2015 nominal (GM)_sun, m^3 s^-2
C_LIGHT = 299792458.0
TSUN = GM_SUN / C_LIGHT**3
print(f"GM_sun/c^3 = {TSUN:.9e} s; x1.2 = {1.2*TSUN*1e6:.4f} us")


def f_of_tau(tau, mc):
    return (1 / (8 * np.pi)) * (5 / tau) ** (3 / 8) * (TSUN * mc) ** (-5 / 8)


def phi_of_tau(tau, mc):
    return -2 * (tau / (5 * TSUN * mc)) ** (5 / 8)


# dPhi/dt = 2 pi f, numerically
tau = np.linspace(0.5, 4.0, 2001)
dphidt = -np.gradient(phi_of_tau(tau, 1.2), tau)
print("max rel diff dPhi/dt vs 2 pi f:", np.max(np.abs(dphidt[5:-5] / (2 * np.pi * f_of_tau(tau, 1.2)[5:-5]) - 1)))

for mc in (0.85, 1.2, 1.8, 2.75):
    P = abs(phi_of_tau(4.0, mc))
    print(f"Mc {mc}: f(4s) {f_of_tau(4.0, mc):.2f} Hz; cycles {P/2/np.pi:.1f}; |Phi| {P:.1f} rad;"
          f" 1% shift exact {P*(1-1.01**(-5/8)):.2f} rad, linear {0.625*0.01*P:.2f} rad")

# ISCO estimate for equal masses
for mc in (0.85, 1.2, 1.8, 2.75):
    M = mc / 0.25 ** 0.6
    f_isco = 1 / (6 ** 1.5 * np.pi * TSUN * M)
    print(f"Mc {mc}: equal-mass total {M:.3f}, f_ISCO {f_isco:.0f} Hz")


def chirp(mc, n=L, f_hi=900.0, quad=False):
    tau = (n - np.arange(n)) / FS
    f = f_of_tau(tau, mc)
    phi = phi_of_tau(tau, mc)
    amp = tau ** (-0.25) * (f < f_hi) * (f > 20)
    h = amp * (np.sin(phi) if quad else np.cos(phi))
    return h, f


def unit(x):
    return x / np.linalg.norm(x)


# quadrature orthogonality and equal size
for mc in (0.85, 1.2, 1.8, 2.75):
    hc, _ = chirp(mc)
    hs, _ = chirp(mc, quad=True)
    print(f"Mc {mc}: <hc,hs>/(|hc||hs|) = {hc@hs/np.linalg.norm(hc)/np.linalg.norm(hs):+.2e};"
          f" |hs|^2/|hc|^2 - 1 = {hs@hs/(hc@hc)-1:+.2e}")

# tone counts: real chirp (rfft), complex chirps (full fft)
def curve(power):
    p = np.sort(power)[::-1]
    return np.sqrt(np.cumsum(p) / p.sum())

def n_for(c, th):
    return int(np.searchsorted(c, th)) + 1

h, _ = chirp(1.2)
c_real = curve(np.abs(np.fft.rfft(unit(h))) ** 2)
print("real chirp 1.2, rfft: n90", n_for(c_real, 0.9), "n99", n_for(c_real, 0.99), "match at 32", c_real[31])
# same with DC/Nyquist weighted correctly (pairs): weights 1 for DC/Nyq, 2 otherwise
H = np.abs(np.fft.rfft(unit(h))) ** 2
w = np.full(H.size, 2.0); w[0] = 1; w[-1] = 1
c_real_w = curve(H * w)
print("  with pair weights: n90", n_for(c_real_w, 0.9), "n99", n_for(c_real_w, 0.99))


def cchirp(mc):
    tau = (L - np.arange(L)) / FS
    fr = f_of_tau(tau, mc)
    return tau ** -0.25 * (fr < 900) * (fr > 20) * np.exp(1j * phi_of_tau(tau, mc))

bns = {m: curve(np.abs(np.fft.fft(cchirp(m))) ** 2) for m in (1.0, 1.2, 1.4, 1.8, 2.4)}
for m, c in bns.items():
    print(f"complex chirp {m}: n90 {n_for(c,0.9)}, n99 {n_for(c,0.99)}, match at 32 {c[31]:.3f}")
n90 = [n_for(bns[m], 0.9) for m in (1.0, 1.4, 1.8, 2.4)]
print("figure set (1.0,1.4,1.8,2.4) n90:", n90, "median", np.median(n90))
print("band bins 80->900 Hz for 1.2:", (900 - f_of_tau(4.0, 1.2)) * T)

# The Project 8 part of #ck-fourier-tones read local Project 8 records; it is not included.

# noise statistics of e for one kernel
p10, p15 = math.exp(-5), math.exp(-7.5)
print(f"P(e>10) {p10:.5f}, P(e>15) {p15:.6f}")
for N in (2048, 8192):
    H_N = np.sum(1 / np.arange(1, N + 1))
    print(f"N {N}: 2H_N {2*H_N:.2f}, 2(lnN+gamma) {2*(np.log(N)+0.5772156649):.2f}")

kc = unit(chirp(1.2)[0]); ks = unit(chirp(1.2, quad=True)[0])
def energy(u, kc, ks):
    out = []
    for k in (kc, ks):
        y = np.fft.irfft(np.fft.rfft(u, 2 * L) * np.fft.rfft(k[::-1], 2 * L), 2 * L)[:L]
        out.append(y)
    return out[0] ** 2 + out[1] ** 2
# expected noise energy at each output time: sum of kernel energy overlapping the data
cum_c = np.cumsum((kc ** 2)[::-1]); cum_s = np.cumsum((ks ** 2)[::-1])
exp_e = cum_c + cum_s   # at output t, kernel's last t+1 samples overlap
print("expected noise energy at t=3.0 s:", exp_e[int(3.0*FS)], " at end:", exp_e[-1])
rng = np.random.default_rng(0)
mx, means = [], []
for _ in range(400):
    e = energy(rng.standard_normal(L), kc, ks)[int(3.0 * FS):]
    mx.append(e.max()); means.append(e.mean())
mx = np.array(mx)
print(f"last 1 s, 400 noise windows: mean e {np.mean(means):.3f}; max: median {np.median(mx):.2f}, 10-90% {np.percentile(mx,10):.2f}-{np.percentile(mx,90):.2f}, frac max>10 {np.mean(mx>10):.3f}, >15 {np.mean(mx>15):.3f}")

# GELU mean on a sinusoid
from scipy.stats import norm
th = np.linspace(0, 2 * np.pi, 200001)[:-1]
for a in (0.1, 1.0, 3.0):
    x = a * np.cos(th)
    g = x * norm.cdf(x)
    print(f"GELU mean on a cos, a={a}: {g.mean():.5f}; small-a approx {a*a/(2*np.sqrt(2*np.pi)):.5f}")
