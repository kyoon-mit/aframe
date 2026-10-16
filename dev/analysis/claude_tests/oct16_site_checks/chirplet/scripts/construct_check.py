"""Independent construction check: pieces and chirplets laid along leading-order chirps.

Chirp: tau^(-1/4) exp(i Phi(tau)), last 4 s at 2048 Hz (8192 samples), kept where
20 Hz < f < f_max. Match: |<a, h>| / (|a| |h|).

A. Cut pieces: one per unit of the in-step count, tangent linear chirp (or steady
   wave) at the piece's centre, the chirp's own amplitude.
B. Gaussian chirplets at P equal steps of the linear-chirp count, width beta times
   the local spacing, frequency and rate of the chirp at the centre.
   B1: complex amplitudes by least squares.  B2: no solve, amplitude and phase of
   the chirp at the centre.
C. Strongest Fourier tones for a match of 0.9.
"""
import os

for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[v] = "1"

import json
import sys
import time

import numpy as np

T0 = time.time()
TSUN = 1.3271244e20 / 299792458.0**3
FS, NS = 2048.0, 8192
TW = NS / FS
OUT = sys.argv[1]
MASSES = (0.85, 1.2, 1.8, 2.75)


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


# sample k at time t_k = -(NS - k)/FS before the merger, so tau_k = (NS - k)/FS > 0
TAU = (NS - np.arange(NS)) / FS
T = -TAU


def chirp(mc, fmax):
    fr = f_of_tau(TAU, mc)
    keep = (fr > 20.0) & (fr < fmax)
    return keep * TAU ** -0.25 * np.exp(1j * phase_of_tau(TAU, mc)), keep


def count(tau, mc, kind):
    """In-step pieces from the window start (tau = TW) down to tau, closed form."""
    if kind == "lin":
        K = 4.8 * TW * (fddot(TW, mc) / 6.0) ** (1 / 3)
        return K * (1 - (tau / TW) ** (5 / 24)), K, 5 / 24
    K = 3.2 * TW * fdot(TW, mc) ** 0.5
    return K * (1 - (tau / TW) ** (5 / 16)), K, 5 / 16


def tau_at_count(m, mc, kind):
    _, K, p = count(TW, mc, kind)
    return TW * (1 - m / K) ** (1 / p)


def match(a, b):
    return float(abs(np.vdot(a, b)) / (np.linalg.norm(a) * np.linalg.norm(b)))


def cut_pieces(mc, fmax, kind):
    h, keep = chirp(mc, fmax)
    n, _, _ = count(TAU, mc, kind)
    n_end = float(count(tau_of_f(fmax, mc), mc, kind)[0])
    idx = np.floor(n).astype(int)
    approx = np.zeros(NS, complex)
    for j in np.unique(idx[keep]):
        sel = keep & (idx == j)
        lo, hi = j, min(j + 1, n_end)
        tauc = tau_at_count(0.5 * (lo + hi), mc, kind)
        s = T[sel] + tauc
        ph = phase_of_tau(tauc, mc) + 2 * np.pi * f_of_tau(tauc, mc) * s
        if kind == "lin":
            ph = ph + np.pi * fdot(tauc, mc) * s**2
        approx[sel] = np.abs(h[sel]) * np.exp(1j * ph)
    return n_end, int(np.unique(idx[keep]).size), match(approx, h)


def gauss_atoms(mc, fmax, P, beta):
    n_end = float(count(tau_of_f(fmax, mc), mc, "lin")[0])
    n_beg = float(count(tau_of_f(20.0, mc), mc, "lin")[0]) if f_of_tau(TW, mc) < 20 else 0.0
    step = (n_end - n_beg) / P
    G = np.empty((NS, P), complex)
    amp0 = np.empty(P, complex)
    for k in range(P):
        tauc = tau_at_count(n_beg + (k + 0.5) * step, mc, "lin")
        s = T + tauc
        sig = beta * step * (6.0 / fddot(tauc, mc)) ** (1 / 3)
        G[:, k] = np.exp(-0.5 * (s / sig) ** 2 + 1j * (2 * np.pi * f_of_tau(tauc, mc) * s + np.pi * fdot(tauc, mc) * s**2))
        amp0[k] = tauc ** -0.25 * np.exp(1j * phase_of_tau(tauc, mc))
    return G, amp0


def gauss_match(mc, fmax, P, beta):
    h, _ = chirp(mc, fmax)
    G, amp0 = gauss_atoms(mc, fmax, P, beta)
    a, *_ = np.linalg.lstsq(G, h, rcond=None)
    m_ls = match(G @ a, h)
    m_direct = match(G @ amp0, h)
    col = np.linalg.norm(G, axis=0)
    cancel = float(np.sum((np.abs(a) * col) ** 2) / np.linalg.norm(G @ a) ** 2)
    return m_ls, m_direct, cancel


def fourier_count(mc, fmax, level=0.9):
    h, _ = chirp(mc, fmax)
    e = np.sort(np.abs(np.fft.fft(h)) ** 2)[::-1]
    return int(np.searchsorted(np.cumsum(e) / e.sum(), level**2) + 1)


res = {"cut": [], "gauss": [], "fourier": {}}
print("A. cut pieces")
for fmax in (512.0, 900.0, 1024.0):
    for mc in MASSES:
        r = {"mc": mc, "fmax": fmax}
        for kind in ("lin", "tone"):
            ne, npc, m = cut_pieces(mc, fmax, kind)
            r[kind] = {"count": ne, "pieces": npc, "match": m}
        res["cut"].append(r)
        print(f" fmax {fmax:5.0f} Mc {mc:4.2f}: linear {r['lin']['count']:6.2f} ({r['lin']['pieces']}) "
              f"{r['lin']['match']:.4f} | steady {r['tone']['count']:6.2f} ({r['tone']['pieces']}) {r['tone']['match']:.4f}")
sys.stdout.flush()

print("B. Gaussian chirplets along the chirp")
betas = (0.3, 0.4, 0.5, 0.6, 0.8, 1.0)
Ps = list(range(4, 25)) + [28, 32]
for fmax in (900.0, 1024.0):
    for mc in MASSES:
        best, direct = {}, {}
        for P in Ps:
            vals = {b: gauss_match(mc, fmax, P, b) for b in betas}
            bb = max(vals, key=lambda b: vals[b][0])
            bd = max(vals, key=lambda b: vals[b][1])
            best[P] = (vals[bb][0], bb, vals[bb][2])
            direct[P] = (vals[bd][1], bd)
            res["gauss"].append({"mc": mc, "fmax": fmax, "P": P,
                                 "ls": {str(b): vals[b][0] for b in betas},
                                 "direct": {str(b): vals[b][1] for b in betas},
                                 "cancel": {str(b): vals[b][2] for b in betas}})
        need = {lv: next((P for P in Ps if best[P][0] >= lv), None) for lv in (0.9, 0.95, 0.97, 0.99)}
        needd = {lv: next((P for P in Ps if direct[P][0] >= lv), None) for lv in (0.9, 0.95, 0.97, 0.99)}
        print(f" fmax {fmax:5.0f} Mc {mc:4.2f}: P for 0.9/0.95/0.97/0.99, least squares {need}, no solve {needd}")
        print("   " + " ".join(f"P{P}:{best[P][0]:.3f}(b{best[P][1]},c{best[P][2]:.2f})/{direct[P][0]:.3f}"
                              for P in (6, 8, 9, 10, 11, 12, 14, 16, 20)))
        sys.stdout.flush()

print("C. Fourier tones for 0.9")
for fmax in (900.0, 1024.0):
    res["fourier"][str(fmax)] = {str(mc): fourier_count(mc, fmax) for mc in MASSES}
    print(f" fmax {fmax}: {res['fourier'][str(fmax)]}")
json.dump(res, open(OUT, "w"), indent=1)
print(f"done in {time.time() - T0:.1f} s")
