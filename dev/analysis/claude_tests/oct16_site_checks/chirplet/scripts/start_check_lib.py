"""Chirplet atoms along a leading-order chirp, in kernel lag (conjugated template read backwards)."""
import numpy as np

TSUN = 1.3271244e20 / 299792458.0**3
FS = 2048.0
L = 8064
TAU = (np.arange(L) + 0.5) / FS
TW = L / FS


def f_of_tau(tau, mc):
    return (5.0 / (256.0 * tau)) ** 0.375 * (TSUN * mc) ** -0.625 / np.pi


def tau_of_f(f, mc):
    return 5.0 / 256.0 * (TSUN * mc) ** (-5.0 / 3.0) * (np.pi * f) ** (-8.0 / 3.0)


def phase_of_tau(tau, mc):
    return -2.0 * (tau / (5.0 * TSUN * mc)) ** 0.625


def fdot(tau, mc):
    return 0.375 * f_of_tau(tau, mc) / tau


def fddot(tau, mc):
    return 0.515625 * f_of_tau(tau, mc) / tau**2


def chain_atoms(mc, P, beta=0.4, fmax=1024.0):
    K0 = 4.8 * TW * (fddot(TW, mc) / 6.0) ** (1 / 3)
    p = 5 / 24
    n_end = K0 * (1 - (tau_of_f(fmax, mc) / TW) ** p)
    step = n_end / P
    t = -TAU
    G = np.empty((L, P), complex)
    for k in range(P):
        tauc = TW * (1 - (k + 0.5) * step / K0) ** (1 / p)
        s = t + tauc
        sig = beta * step * (6.0 / fddot(tauc, mc)) ** (1 / 3)
        G[:, k] = np.conj(tauc ** -0.25 * np.exp(-0.5 * (s / sig) ** 2 + 1j * (phase_of_tau(tauc, mc)
                          + 2 * np.pi * f_of_tau(tauc, mc) * s + np.pi * fdot(tauc, mc) * s**2)))
    return G
