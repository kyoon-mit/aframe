"""Independent check of the 'start with no fit': 12 chirplets along leading-order chirps
on the 256-mass log grid 0.85..2.75, matched to the four whitened kernels.

Chains are built as forward-time templates on the last L/fs seconds and turned into
kernels by reversing and conjugating (the matched filter), as the kernels were made.
Centres from a numerically integrated count; rates from finite differences.
"""
import os
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[v] = "1"
import json, sys
import numpy as np

SP = "/tmp/claude-1024/-home-kyoon-SSM-BNS/82effc75-bb98-44ad-af20-e41e39a7258d/scratchpad"
KER = np.load(SP + "/s4dmath/fits/kernels.npz")
TS = 1.3271244e20 / 299792458.0**3
FS = 2048.0
L = 8064
TW = L / FS
n = np.arange(L)
tau = (L - n) / FS - 0.5 / FS     # sample centres, tau from TW-0.5/fs down to 0.5/fs
t = -tau


def f_tau(x, mc):
    return (5.0 / 256.0 / x) ** 0.375 * (TS * mc) ** -0.625 / np.pi


def tau_f(f, mc):
    return 5.0 / 256.0 * (TS * mc) ** (-5.0 / 3.0) * (np.pi * f) ** (-8.0 / 3.0)


def phi(x, mc):
    A = (5.0 / 256.0) ** 0.375 * (TS * mc) ** -0.625 / np.pi
    return -2 * np.pi * A * x ** 0.625 / 0.625


def fd_fdd(x, mc):
    h = 1e-5 * x
    fp, f0, fm = f_tau(x - h, mc), f_tau(x, mc), f_tau(x + h, mc)
    return (fp - fm) / (2 * h), (fp - 2 * f0 + fm) / h**2


def chain_atoms(mc, P=12, beta=0.4, fmax=1024.0):
    x = np.geomspace(TW, tau_f(fmax, mc), 200001)
    _, fdd = fd_fdd(x, mc)
    rate = (fdd / 6.0) ** (1 / 3)
    c = np.concatenate([[0.0], np.cumsum(0.5 * (rate[1:] + rate[:-1]) * (x[:-1] - x[1:]))])
    step = c[-1] / P
    G = np.empty((L, P), complex)
    for k in range(P):
        xc = float(np.interp((k + 0.5) * step, c, x))
        fd, fdd_c = fd_fdd(xc, mc)
        sig = beta * step * (6.0 / fdd_c) ** (1 / 3)
        s = t + xc
        G[:, k] = xc ** -0.25 * np.exp(-0.5 * (s / sig) ** 2 + 1j * (phi(xc, mc) + 2 * np.pi * f_tau(xc, mc) * s + np.pi * fd * s * s))
    return G   # forward-time template atoms


def to_kernel(h):
    return np.conj(h[::-1])


def xcorr_best(k, K):
    m = 2 * L
    c = np.fft.ifft(np.conj(np.fft.fft(k, m)) * np.fft.fft(K, m))
    j = int(np.argmax(np.abs(c)))
    return float(np.abs(c[j]) / np.linalg.norm(k) / np.linalg.norm(K)), (j if j < L else j - m)


