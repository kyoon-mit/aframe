"""Chirplets laid along leading-order chirps: match against the number of pieces.

Chirps: tau^(-1/4) exp(i Phi), last 4 s at 2048 Hz, kept where 20 < f < fmax
(the leading-order chirps of the 16 October resonator fits use fmax = 900 Hz).

A. Rectangular pieces, one per unit of the in-step count n(t) (an eighth of a
   cycle), each the tangent tone or tangent linear chirp at its centre, with
   the chirp's own amplitude: match.
B. Gaussian chirplets along the chirp, no fit: P centres at equal steps of n,
   width sigma = beta * spacing, frequency and rate of the chirp at the centre;
   only the complex amplitudes solved (least squares): match against P.
C. Strongest Fourier tones for a match of 0.9 (control against the site's count).
"""
import os

for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[v] = "1"

import json
import sys
import time

import numpy as np

T0 = time.time()
GM_SUN = 1.3271244e20
C = 299792458.0
TSUN = GM_SUN / C**3
FS = 2048.0
N = 8192
TWIN = N / FS
MASSES = (0.85, 1.2, 1.8, 2.75)


def f_tau(tau, mc):
    return (1 / np.pi) * (5 / (256 * tau)) ** 0.375 * (TSUN * mc) ** (-0.625)


def tau_f(f, mc):
    return 5 / 256 * (TSUN * mc) ** (-5 / 3) * (np.pi * f) ** (-8 / 3)


def phi_tau(tau, mc):
    return -2 * (tau / (5 * TSUN * mc)) ** 0.625


def fdot_f(f, mc):
    return 96 / 5 * np.pi ** (8 / 3) * (TSUN * mc) ** (5 / 3) * f ** (11 / 3)


def chirp(mc, fmax, fmin=20.0):
    tau = (N - np.arange(N)) / FS
    fr = f_tau(tau, mc)
    return tau, tau ** -0.25 * (fr < fmax) * (fr > fmin) * np.exp(1j * phi_tau(tau, mc))


def count_fn(mc, kind):
    """n(tau): in-step pieces from the window start (tau = TWIN) down to tau."""
    fT = f_tau(TWIN, mc)
    if kind == "lin":
        K = 24 / 5 * (11 / 128) ** (1 / 3) * (TWIN * fT) ** (1 / 3)
        p = 5 / 24
    else:
        K = 16 / 5 * np.sqrt(3 / 8) * np.sqrt(TWIN * fT)
        p = 5 / 16
    n = lambda tau: K * (1 - (tau / TWIN) ** p)                     # noqa: E731
    inv = lambda m: TWIN * (1 - m / K) ** (1 / p)                   # noqa: E731
    return n, inv


def match(a, b):
    return float(np.abs(np.vdot(a, b)) / (np.linalg.norm(a) * np.linalg.norm(b)))


def rect_pieces(mc, fmax, kind):
    tau, h = chirp(mc, fmax)
    n, inv = count_fn(mc, kind)
    tau_end = tau_f(fmax, mc)
    n_end = n(tau_end)
    edges = np.arange(0.0, np.ceil(n_end) + 1e-9, 1.0)
    edges[-1] = n_end
    approx = np.zeros(N, complex)
    t = -tau                                                        # time, merger at 0
    nn = n(np.maximum(tau, 1e-12))
    for k in range(edges.size - 1):
        tk0, tk1 = -inv(edges[k]), -inv(edges[k + 1])
        sel = (nn >= edges[k]) & (nn < edges[k + 1]) & (np.abs(h) > 0)
        if not sel.any():
            continue
        tc = 0.5 * (tk0 + tk1)
        tauc = -tc
        fc = f_tau(tauc, mc)
        s = t[sel] - tc
        ph = phi_tau(tauc, mc) + 2 * np.pi * fc * s
        if kind == "lin":
            ph = ph + np.pi * fdot_f(fc, mc) * s**2
        approx[sel] = np.abs(h[sel]) * np.exp(1j * ph)
    return n_end, edges.size - 1, match(approx, h)


def gauss_chain(mc, fmax, P, beta):
    tau, h = chirp(mc, fmax)
    n, inv = count_fn(mc, "lin")
    n_end = n(tau_f(fmax, mc))
    step = n_end / P
    t = -tau
    G = np.empty((N, P), complex)
    for k in range(P):
        m = (k + 0.5) * step
        tauc = inv(m)
        tc = -tauc
        fc = f_tau(tauc, mc)
        fd = fdot_f(fc, mc)
        # local spacing in time: step pieces of the local in-step time
        fdd = 11 / 3 * fd**2 / fc
        spacing = step * (6 / fdd) ** (1 / 3)
        sig = beta * spacing
        s = t - tc
        G[:, k] = np.exp(-0.5 * (s / sig) ** 2 + 1j * (phi_tau(tauc, mc) + 2 * np.pi * fc * s + np.pi * fd * s**2))
    a, *_ = np.linalg.lstsq(G, h, rcond=None)
    return match(G @ a, h)


def fourier_count(mc, fmax, level=0.9):
    _, h = chirp(mc, fmax)
    e = np.sort(np.abs(np.fft.fft(h)) ** 2)[::-1]
    c = np.cumsum(e) / e.sum()
    return int(np.searchsorted(c, level**2) + 1)


out = {"rect": [], "gauss": [], "fourier": []}
print("A. Rectangular pieces, one per in-step unit:")
for fmax in (512.0, 900.0, 1024.0):
    for mc in MASSES:
        row = {"mc": mc, "fmax": fmax}
        for kind in ("tone", "lin"):
            ne, npieces, m = rect_pieces(mc, fmax, kind)
            row[kind] = {"count": ne, "pieces": npieces, "match": m}
        out["rect"].append(row)
        print(f" fmax {fmax:6.0f} Mc {mc:4.2f}: tones {row['tone']['count']:6.2f} ({row['tone']['pieces']} pieces) match {row['tone']['match']:.4f} | "
              f"linear chirps {row['lin']['count']:6.2f} ({row['lin']['pieces']} pieces) match {row['lin']['match']:.4f}")
sys.stdout.flush()

print("\nB. Gaussian chirplets along the chirp, amplitudes by least squares (fmax 900):")
betas = (0.3, 0.4, 0.5, 0.6, 0.8, 1.0)
Ps = list(range(2, 41)) + [48, 56, 64]
for mc in MASSES:
    best = {}
    for P in Ps:
        ms = {b: gauss_chain(mc, 900.0, P, b) for b in betas}
        bb = max(ms, key=ms.get)
        best[P] = (ms[bb], bb, ms[0.5])
        out["gauss"].append({"mc": mc, "P": P, "by_beta": ms})
    need = {}
    for lev in (0.8, 0.9, 0.95, 0.97, 0.99):
        need[lev] = next((P for P in Ps if best[P][0] >= lev), None)
        need[f"{lev}_beta0.5"] = next((P for P in Ps if best[P][2] >= lev), None)
    print(f" Mc {mc:4.2f}: P for match 0.8/0.9/0.95/0.97/0.99 (best beta): "
          f"{need[0.8]}/{need[0.9]}/{need[0.95]}/{need[0.97]}/{need[0.99]}; beta 0.5: "
          f"{need['0.8_beta0.5']}/{need['0.9_beta0.5']}/{need['0.95_beta0.5']}/{need['0.97_beta0.5']}/{need['0.99_beta0.5']}")
    for P in (4, 6, 8, 10, 12, 16, 20, 24, 32):
        print(f"    P {P:3d}: best {best[P][0]:.4f} (beta {best[P][1]}), beta 0.5 {best[P][2]:.4f}")
    sys.stdout.flush()

print("\nC. Strongest Fourier tones for a match of 0.9 (fmax 900):")
for mc in MASSES:
    k = fourier_count(mc, 900.0)
    out["fourier"].append({"mc": mc, "count_0.9": k})
    print(f" Mc {mc:4.2f}: {k}")
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "chains.json"), "w"), indent=1)
print(f"done in {time.time() - T0:.1f} s")
