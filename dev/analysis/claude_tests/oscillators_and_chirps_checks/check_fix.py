"""Checks for the derivations added in the fixer pass of the 15 October lecture (leading-order chirp)."""
import glob
from pathlib import Path

import numpy as np
from scipy import integrate, optimize
from scipy.special import erf

T_SUN = 4.925490947e-6


def tm(mc): return T_SUN * mc
def f_of_tau(tau, mc): return (5 / tau) ** (3 / 8) * tm(mc) ** (-5 / 8) / (8 * np.pi)
def phase_left(tau, mc): return 2 * (tau / (5 * tm(mc))) ** (5 / 8)      # phase left to merger, rad
def fdot(tau, mc): return 3 * f_of_tau(tau, mc) / (8 * tau)
def tau_of_f(f, mc): return 5 / 256 * tm(mc) ** (-5 / 3) * (np.pi * f) ** (-8 / 3)


print("A) in-step stretches: integral of fdot^(1/2) dt over the last 4 s")
for mc in (0.85, 1.2, 2.75):
    N = phase_left(4.0, mc) / (2 * np.pi)
    q = integrate.quad(lambda t: np.sqrt(fdot(t, mc)), 0, 4.0, limit=400)[0]
    closed = 16 / 5 * 4.0 * np.sqrt(fdot(4.0, mc))
    print(f"  Mc {mc}: N = {N:.1f} cycles; quad {q:.3f}; (16/5) T fdot(T)^1/2 = {closed:.3f}; "
          f"(2 sqrt(15)/5) sqrt(N) = {2 * np.sqrt(15) / 5 * np.sqrt(N):.3f}; N/(f_T T) = {N / (f_of_tau(4.0, mc) * 4.0):.4f}")
print(f"  2 sqrt(15)/5 = {2 * np.sqrt(15) / 5:.4f}")
print("  SNR per stretch for 35 stretches:", [round(r / np.sqrt(35), 2) for r in (50, 25, 12)],
      f"; (2*35)^(1/4) = {70 ** 0.25:.3f}")

print("B) the nonlinear function x Phi(x): average slope and SNR kept by an in-step sum after it")
phi = lambda u: np.exp(-u * u / 2) / np.sqrt(2 * np.pi)
Phi = lambda u: 0.5 * (1 + erf(u / np.sqrt(2)))
def moments(s):
    m1 = integrate.quad(lambda u: s * u * Phi(s * u) * phi(u), -12, 12)[0]
    m2 = integrate.quad(lambda u: (s * u * Phi(s * u)) ** 2 * phi(u), -12, 12)[0]
    sl = integrate.quad(lambda u: (Phi(s * u) + s * u * phi(s * u)) * phi(u), -12, 12)[0]
    return sl, sl * s / np.sqrt(m2 - m1 * m1)
for s in (0.01, 0.1, 1.0, 10.0, 1000.0):
    sl, kept = moments(s)
    print(f"  spread {s:g}: average slope {sl:.6f}; SNR kept {kept:.4f}")
res = optimize.minimize_scalar(lambda ls: moments(np.exp(ls))[1], bounds=(-2, 3), method="bounded")
print(f"  smallest SNR kept {res.fun:.4f} at spread {np.exp(res.x):.3f}; "
      f"large-spread limit 1/(2 sqrt(1/2 - 1/(2 pi))) = {1 / (2 * np.sqrt(0.5 - 1 / (2 * np.pi))):.4f}")
rng = np.random.default_rng(5)
n = rng.standard_normal(4_000_000)
g = lambda x: x * Phi(x)
for sig in (0.1, 0.01):
    print(f"  Monte Carlo, unit-spread noise, signal {sig}: (E[g(n+s)] - E[g(n)])/s = {(g(n + sig) - g(n)).mean() / sig:.4f}")

print("C) band start: 256 frequencies evenly over 20-512 Hz, lifetime 4 s")
df = (512 - 20) / 256
print(f"  spacing {df:.6f} Hz; repeat 1/spacing {1 / df:.4f} s; amplitude left after one repeat exp(-P/4) = {np.exp(-1 / df / 4):.4f}; "
      f"repeats in 4 s {4 * df:.3f}")
t = np.arange(0, 4, 1 / 1024)
rng = np.random.default_rng(2)
c = rng.standard_normal(256) + 1j * rng.standard_normal(256)
fr = 20 + df * (np.arange(256) + 0.37)
x = (c[None, :] * np.exp((-1 / 4 + 2j * np.pi * fr[None, :]) * t[:, None])).sum(1)
k = int(round(1 / df * 1024))
P = 1 / df
xs = (c[None, :] * np.exp((-1 / 4 + 2j * np.pi * fr[None, :]) * (t[:200, None] + P))).sum(1)
ratio = xs / x[:200]
print(f"  sum shifted by one repeat / sum: |ratio| {np.abs(ratio).min():.4f} to {np.abs(ratio).max():.4f}, "
      f"phase {np.angle(ratio).min():.4f} to {np.angle(ratio).max():.4f} rad (2 pi f0/spacing mod 2 pi = {np.angle(np.exp(2j*np.pi*fr[0]/df)):.4f})")

print("D) the largest of 100 independent standard normal numbers")
e100 = integrate.quad(lambda u: u * 100 * phi(u) * Phi(u) ** 99, -10, 10)[0]
mc100 = np.random.default_rng(7).standard_normal((200_000, 100)).max(1).mean()
print(f"  quadrature {e100:.4f}; Monte Carlo (200,000 sets) {mc100:.4f}")

print("E) noise reuse in the training background")
fn = sorted(Path(f) for f in glob.glob('/home/kyoon/SSM-BNS/DATA/O3a_H1_L1_2048Hz/background/background/*.hdf5'))
d = [int(f.stem.split('-')[-1]) for f in fn]
vd = 0
while vd < 15000:                                     # min_valid_duration in the run configurations
    fn.pop(-1); vd += d.pop(-1)
tr = sum(d)
print(f"  {len(d)} training files, {tr} s = {tr / 86400:.4f} days; validation {vd} s = {vd / 86400:.4f} days; "
      f"(train - valid) = {(tr - vd) / 86400:.4f} days (the log's figure)")
print(f"  windows per epoch 1564*32 = {1564 * 32}; uses per epoch {1564 * 32 * 4 / tr:.4f}; "
      f"after 142 epochs (0 to 141) {142 * 1564 * 32 * 4 / tr:.1f}; separate 4 s stretches per file {sum(x // 4 for x in d)}")

print("F) heterodyne in time: leftover of exp(i(Phi(M) - Phi(M/(1+eps)))) (true = assumed (1+eps))")
for eps in (0.01, -0.01, 0.0674, -0.0674):
    r = abs(1 - (1 + eps) ** (5 / 8))
    print(f"  eps {eps:+.4f}: frequency fraction {100 * r:.3f}%; in-step time x{1 / np.sqrt(r):.3f}; "
          f"at 1.2 Msun {80.14 * r:.2f} Hz (window start) to {512 * r:.2f} Hz (at 512 Hz)")
fs = 2048.0; L = 8192; tt = np.arange(L) / fs
def count(sig):
    p = np.sort(np.abs(np.fft.fft(sig)) ** 2)[::-1]
    return int(np.searchsorted(np.cumsum(p) / p.sum(), 0.81) + 1)
for mc in (1.0, 1.4, 1.8, 2.4):
    tau = np.maximum(4.0 - tt, 0.5 / fs); f = f_of_tau(tau, mc)
    A = np.where((f >= 20) & (f <= 900), tau ** -0.25, 0.0)
    h = A * np.exp(-1j * phase_left(tau, mc))
    row = [count(h)]
    for eps in (0.01, -0.01, 0.0674, -0.0674):
        left = A * np.exp(-1j * (phase_left(tau, mc) - phase_left(tau, mc / (1 + eps))))
        row.append(count(left))
    # frequency-domain heterodyne: unit-modulus factor in frequency keeps every |FFT|
    H = np.fft.fft(h); R = H * np.exp(-1j * np.angle(np.fft.fft(A * np.exp(-1j * phase_left(tau, mc / 1.0674)))))
    print(f"  Mc {mc}: chirp {row[0]}; in time +1% {row[1]}, -1% {row[2]}, +6.74% {row[3]}, -6.74% {row[4]}; "
          f"in frequency (any eps) {count(np.fft.ifft(R))}")

print("G) residual phase across the window: time form N|1-(1+e)^(5/8)| against frequency form (3/8)N|1-(1+e)^(5/3)|")
mcs = np.geomspace(0.85, 2.75, 401)
for e in (0.001, 0.01, 0.02, 0.05, 0.10):
    tf, ff = [], []
    for sgn in (1, -1):
        for T in (3.5, 4.0):
            N = phase_left(T, mcs) / (2 * np.pi)
            tf.append(N * abs(1 - (1 + sgn * e) ** (5 / 8)))
            ff.append(3 / 8 * N * abs(1 - (1 + sgn * e) ** (5 / 3)))
    tf = np.concatenate(tf); ff = np.concatenate(ff)
    print(f"  {100 * e:g}%: time form {tf.min():.2f} to {tf.max():.2f}; frequency form {ff.min():.2f} to {ff.max():.2f}; "
          f"largest relative difference {100 * np.max(np.abs(ff / tf - 1)):.2f}%")

print("H) heterodyne in frequency: where the residual sits in time (FFT of a time-domain chirp)")
fs = 4096.0; Tf = 16.0; M = int(Tf * fs); tt = np.arange(M) / fs; tc = 8.0
fk = np.fft.fftfreq(M, 1 / fs)
def chirp(mc, T=4.0, taper=0.1):
    tau = tc - tt
    w = (tau > 0) & (tau <= T)
    ta = np.where(w, tau, 1.0)
    A = np.where(w & (f_of_tau(ta, mc) <= 900), ta ** -0.25, 0.0)
    on = np.clip((T - ta) / taper, 0, 1); A = A * np.where(w, 0.5 - 0.5 * np.cos(np.pi * on), 0.0)
    return A * np.exp(-1j * phase_left(ta, mc))
def psi(f, mc):  # stationary-phase phase of the transform, numpy convention, merger at tc
    tau = tau_of_f(f, mc)
    return -2 * np.pi * f * tc + 2 * np.pi * f * tau - phase_left(tau, mc) - np.pi / 4
for mc in (1.2, 2.0):
    H = np.fft.fft(chirp(mc))
    band = (fk > 1.15 * f_of_tau(4.0, mc)) & (fk < 700)
    for eps in (0.0, 0.0674, -0.0674, 0.0046):
        mt = mc / (1 + eps)
        R = np.zeros_like(H); pos = fk > 0
        R[pos] = H[pos] * np.exp(-1j * psi(fk[pos], mt))
        ph = np.unwrap(np.angle(R[band]))
        gd = -np.gradient(ph, fk[band]) / (2 * np.pi)                 # group delay, s
        pred = tau_of_f(fk[band], mc) * ((1 + eps) ** (5 / 3) - 1)
        c = (1 + eps) ** (5 / 3) - 1
        r = np.fft.ifft(R); E = np.abs(r) ** 2
        lag = (tt + Tf / 2) % Tf - Tf / 2                               # time from the reference, -8 to 8 s
        lo, hi = min(0, 4 * c) - 0.01, max(0, 4 * c) + 0.01
        inside = E[(lag >= lo) & (lag <= hi)].sum() / E.sum()
        print(f"  Mc {mc}, eps {eps:+.4f}: |R| = |H| exactly: {np.allclose(np.abs(R[pos]), np.abs(H[pos]))}; "
              f"group delay - c tau(f) over {fk[band][0]:.0f}-{fk[band][-1]:.0f} Hz: median |diff| {1e3 * np.median(np.abs(gd - pred)):.2f} ms; "
              f"c = {c:+.4f}, 4 s x |c| = {4 * abs(c):.3f} s; energy within [{lo:.3f}, {hi:.3f}] s: {100 * inside:.2f}%")
    R0 = np.zeros_like(H); R0[fk > 0] = np.abs(H[fk > 0])
    r0 = np.abs(np.fft.ifft(R0)) ** 2; lag = (tt + Tf / 2) % Tf - Tf / 2
    for w in (0.005, 0.01):
        print(f"    right chirp mass, phases removed entirely: energy within {1e3 * w:.0f} ms of the reference {100 * r0[np.abs(lag) <= w].sum() / r0.sum():.1f}%")
for eps in (0.0674, -0.0674, 0.004638, -0.004638):
    print(f"  4 s x |1-(1+eps)^(5/3)| for eps {eps:+.6f}: {4 * abs(1 - (1 + eps) ** (5 / 3)):.4f} s")

print("I) whitening response cut to 1 s with a Hann window: first zero of the window's transform")
fsw = 2048; n = fsw
w = np.hanning(n)
F = np.abs(np.fft.rfft(w, 1 << 22)); ff = np.fft.rfftfreq(1 << 22, 1 / fsw)
i = np.argmax(F[1:] > F[:-1]) if False else None
j = np.where((F[1:-1] < F[:-2]) & (F[1:-1] < F[2:]))[0][0] + 1
print(f"  first minimum of |transform| at {ff[j]:.4f} Hz (2 / 1 s = 2 Hz); value {F[j] / F[0]:.2e} of the peak")

print("J) wider network: learned numbers from the saved model")
import torch
ck = torch.load('/home/kyoon/SSM-BNS/aframe/dev/runs/claude_tests/s4d_energy_max_1024ch_dstate64_mixed_256_softmax_train_snr8_prob0.96/'
                'CLAUDE-TESTS/s4d_energy_max_1024ch_dstate64_mixed_256_softmax_train_snr8_prob0.96/checkpoints/last.ckpt',
                map_location='cpu', weights_only=False)
sd = ck['state_dict']
tot = 0
for k_, v in sd.items():
    if v.dtype.is_floating_point and not k_.endswith(('grid_log_mc', 'band')):
        tot += v.numel(); print(f"  {k_}: {tuple(v.shape)} = {v.numel()}")
print(f"  total {tot}; epoch {ck.get('epoch')}")
print(f"  tau(100 Hz, 1.2 Msun) = {tau_of_f(100.0, 1.2):.4f} s; S4D-Lin, 32 per channel: f_n T_n = n <= 31, so at 100 Hz T <= {31 / 100:.2f} s")
