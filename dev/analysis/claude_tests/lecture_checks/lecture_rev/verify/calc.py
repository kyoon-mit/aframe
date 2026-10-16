"""Independent checks of a few lecture numbers (leading-order chirp, run data files)."""
import glob, os, re
import numpy as np

T_SUN = 4.925490947e-6
def tm(mc): return T_SUN * mc
def f_of_tau(tau, mc): return (5 / tau) ** (3 / 8) * tm(mc) ** (-5 / 8) / (8 * np.pi)
def phase(tau, mc): return -2 * (tau / (5 * tm(mc))) ** (5 / 8)

print("1) heterodyne leftover frequency / true chirp frequency")
r = (2.75 / 0.85) ** (1 / 9)
w = np.sqrt(r) - 1
print(f"   ten-mass grid ratio {r:.6f}; worst error sqrt(r)-1 = {100*w:.3f}%")
# article convention: true = assumed * (1 + eps); leftover/true = |1 - (1+eps)^(5/8)|
for lab, eps in (("eps=+6.74% (assumed below true, grid case)", w),
                 ("eps=-6.31% (assumed above true by factor sqrt(r), grid case)", 1 / (1 + w) - 1),
                 ("eps=-6.74% (assumed above true by factor 1/0.9326 = 1.0723, not a grid case)", -w)):
    q = abs(1 - (1 + eps) ** (5 / 8))
    print(f"   {lab}: leftover {100*q:.2f}% of true chirp; in-step factor {1/np.sqrt(q):.2f}; "
          f"1.2 Msun 80.14 Hz -> {80.14*q:.2f} Hz, 512 Hz -> {512*q:.2f} Hz")
# judge/verify.py and rebut/misc.py formula
for eps in (w, -w):
    q = abs(1 - (1 + eps) ** (-5 / 8))
    print(f"   formula |1-(1+eps)^(-5/8)| at eps={100*eps:+.2f}%: {100*q:.2f}%")

print("2) Fourier tones for match 0.9 of the leftover (same method as adversary/check3.py)")
def count(sig):
    p = np.sort(np.abs(np.fft.fft(sig)) ** 2)[::-1]
    c = np.cumsum(p) / p.sum()
    return int(np.searchsorted(c, 0.81) + 1)
fs = 2048.0; L = 8192; t = np.arange(L) / fs
for mc in (1.0, 1.4, 1.8, 2.4):
    tau = np.maximum(4.0 - t, 0.5 / fs); f = f_of_tau(tau, mc)
    A = np.where((f >= 20) & (f <= 900), tau ** -0.25, 0.0)
    out = []
    for eps in (w, 1 / (1 + w) - 1, -w):
        out.append(count(A * np.exp(1j * (phase(tau, mc) - phase(tau, mc / (1 + eps))))))
    print(f"   Mc {mc}: eps +6.74% {out[0]}, -6.31% {out[1]}, -6.74% {out[2]}")

print("3) per-cycle SNR at SNR 8 over 305-636 cycles")
for n in (305, 600, 636):
    print(f"   N={n}: 8/sqrt(N) = {8/np.sqrt(n):.3f}")

print("4) merger 3.5 s from window start: entry frequency and cycles")
for mc in (0.85, 1.2, 2.75):
    for T in (3.5, 4.0):
        print(f"   Mc {mc}, {T} s: f = {f_of_tau(T, mc):.1f} Hz, cycles {abs(phase(T, mc))/(2*np.pi):.1f}")

print("5) training background files (O3a, top level, minus the two validation files)")
d = "/home/kyoon/SSM-BNS/DATA/O3a_H1_L1_2048Hz"
val = {"background-1238412673-14008.hdf5", "background-1238384700-27833.hdf5"}
fl = sorted(os.path.basename(p) for p in glob.glob(f"{d}/background-*.hdf5"))
tr = [x for x in fl if x not in val]
dur = [int(re.match(r"background-\d+-(\d+)\.hdf5", x).group(1)) for x in tr]
print(f"   {len(tr)} files, total {sum(dur)} s = {sum(dur)/86400:.4f} days; "
      f"validation {sum(int(x.split('-')[2].split('.')[0]) for x in val)} s")
print(f"   total/4 = {sum(dur)/4:.2f}; sum over files of floor(duration/4) = {sum(x // 4 for x in dur)}")
print(f"   reuse per epoch {1564*32*4/sum(dur):.4f}; x141 = {141*1564*32*4/sum(dur):.1f}; x142 = {142*1564*32*4/sum(dur):.1f}")

print("6) band-init channel match at 1.0, 1.2, 2.0 Msun (as adversary/bandinit.py), more digits")
fs = 1024.0; L = 4096; s = np.arange(L) / fs
for mc in (1.0, 1.2, 2.0):
    tau = np.maximum(s, 0.5 / fs)
    a = tau ** (-0.25) * (f_of_tau(tau, mc) < 512)
    c = a * np.exp(-1j * phase(tau, mc)); c /= np.linalg.norm(c)
    band = 20 + (512 - 20) / 256 * (np.arange(256) + 0.5)
    B = np.exp((-1 / 4.0 + 2j * np.pi * band[None, :]) * s[:, None])
    coef, *_ = np.linalg.lstsq(B, c, rcond=None)
    print(f"   Mc {mc}: band-init match {np.linalg.norm(B @ coef):.5f}")

print("7) misc")
print(f"   600*0.33 = {600*0.33:.1f}; 600*8/sqrt(600) = {600*8/np.sqrt(600):.2f}")
print(f"   1/sqrt(0.042) = {1/np.sqrt(0.042):.3f}; 5/8*6.74 = {5/8*6.74:.3f}")
