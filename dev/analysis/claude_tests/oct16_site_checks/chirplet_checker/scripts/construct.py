"""Adversarial rebuild of the constructions on leading-order chirps.

Separate from the drafter's code: the in-step count is integrated
numerically on the sample grid (cumulative trapezoid in time on a fine
grid), centres are found by interpolation, frequencies and rates come from
finite differences of the exact phase, and matches use a QR projection.
"""
import os
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[v] = "1"
import json, sys, time
import numpy as np

t0 = time.time()
TS = 1.3271244e20 / 299792458.0**3
FS, NS = 2048.0, 8192
MASSES = (0.85, 1.2, 1.8, 2.75)
tau = (NS - np.arange(NS)) / FS          # 4.0 ... 1/2048 s before merger
t = -tau


def f_tau(x, mc):
    return (5.0 / 256.0 / x) ** 0.375 * (TS * mc) ** -0.625 / np.pi


def tau_f(f, mc):
    return 5.0 / 256.0 * (TS * mc) ** (-5.0 / 3.0) * (np.pi * f) ** (-8.0 / 3.0)


def phi(x, mc):
    # phase vs tau: dPhi/dt = 2 pi f, t = -tau
    A = (5.0 / 256.0) ** 0.375 * (TS * mc) ** -0.625 / np.pi
    return -2 * np.pi * A * x ** 0.625 / 0.625


def fd_fdd(x, mc):
    # finite differences of the frequency in time (t = -tau)
    h = 1e-5 * x
    fp, f0, fm = f_tau(x - h, mc), f_tau(x, mc), f_tau(x + h, mc)  # t+h means tau-h
    return (fp - fm) / (2 * h), (fp - 2 * f0 + fm) / h**2


def chirp(mc, fmax):
    f = f_tau(tau, mc)
    keep = (f > 20) & (f < fmax)
    return np.where(keep, tau ** -0.25 * np.exp(1j * phi(tau, mc)), 0), keep


# numerical count on a fine grid in log tau
def count_grid(mc, fmax, kind):
    lo = tau_f(fmax, mc)
    x = np.geomspace(4.0, lo, 400001)        # decreasing tau = increasing time
    fd, fdd = fd_fdd(x, mc)
    rate = (fdd / 6.0) ** (1 / 3) if kind == "lin" else np.sqrt(fd)
    # d count = rate dt = rate * (-d tau)
    seg = 0.5 * (rate[1:] + rate[:-1]) * (x[:-1] - x[1:])
    c = np.concatenate([[0.0], np.cumsum(seg)])
    return x, c


def proj_match(G, h):
    Q, _ = np.linalg.qr(G)
    return float(np.linalg.norm(Q.conj().T @ h) / np.linalg.norm(h))


def match(a, b):
    return float(abs(np.vdot(a, b)) / np.linalg.norm(a) / np.linalg.norm(b))


def atoms(mc, fmax, P, beta):
    x, c = count_grid(mc, fmax, "lin")
    C = c[-1]
    step = C / P
    G = np.empty((NS, P), complex)
    a0 = np.empty(P, complex)
    for k in range(P):
        ck = (k + 0.5) * step
        xc = float(np.interp(ck, c, x))      # c increasing with index, x decreasing
        f0 = f_tau(xc, mc)
        fd, fdd = fd_fdd(xc, mc)
        tlin = (6.0 / fdd) ** (1 / 3)
        sig = beta * step * tlin
        s = t + xc
        G[:, k] = np.exp(-0.5 * (s / sig) ** 2 + 1j * (2 * np.pi * f0 * s + np.pi * fd * s * s))
        a0[k] = xc ** -0.25 * np.exp(1j * phi(xc, mc))
    return G, a0


def cut(mc, fmax, kind, where="time"):
    h, keep = chirp(mc, fmax)
    x, c = count_grid(mc, fmax, kind)
    C = c[-1]
    cs = np.interp(tau[::-1], x[::-1], c[::-1])[::-1]  # count at each sample
    idx = np.floor(cs).astype(int)
    out = np.zeros(NS, complex)
    for j in np.unique(idx[keep]):
        sel = keep & (idx == j)
        x_lo = np.interp(j, c, x)
        x_hi = np.interp(min(j + 1, C), c, x)
        xc = 0.5 * (x_lo + x_hi) if where == "time" else np.interp(0.5 * (j + min(j + 1, C)), c, x)
        f0 = f_tau(xc, mc)
        fd, _ = fd_fdd(xc, mc)
        s = t[sel] + xc
        ph = phi(xc, mc) + 2 * np.pi * f0 * s + (np.pi * fd * s * s if kind == "lin" else 0)
        out[sel] = np.abs(h[sel]) * np.exp(1j * ph)
    return C, len(np.unique(idx[keep])), match(out, h)


res = {"fourier": {}, "cut": [], "gauss": {}}
for fmax in (900.0, 1024.0):
    r = {}
    for mc in MASSES:
        h, _ = chirp(mc, fmax)
        e = np.sort(np.abs(np.fft.fft(h)) ** 2)[::-1]
        frac = np.cumsum(e) / e.sum()
        r[mc] = int(np.argmax(frac >= 0.81) + 1)
    res["fourier"][fmax] = r
    print("Fourier tones for 0.9, fmax", fmax, r)

for fmax in (512.0, 900.0, 1024.0):
    for mc in MASSES:
        Cl, nl, ml = cut(mc, fmax, "lin")
        Cl2, nl2, ml2 = cut(mc, fmax, "lin", where="count")
        Cs, ns, ms = cut(mc, fmax, "std")
        res["cut"].append(dict(mc=mc, fmax=fmax, lin=(Cl, nl, ml), lin_countmid=ml2, std=(Cs, ns, ms)))
        print(f"cut fmax {fmax} mc {mc}: lin count {Cl:.3f} pieces {nl} match {ml:.4f} (count-mid {ml2:.4f}) | std count {Cs:.3f} pieces {ns} match {ms:.4f}")
sys.stdout.flush()

betas = (0.3, 0.4, 0.5, 0.6, 0.8, 1.0)
Ps = list(range(4, 21))
for fmax in (900.0, 1024.0):
    for mc in MASSES:
        h, _ = chirp(mc, fmax)
        rows = {}
        for P in Ps:
            ls = {}
            dr = {}
            for b in betas:
                G, a0 = atoms(mc, fmax, P, b)
                ls[b] = proj_match(G, h)
                dr[b] = match(G @ a0, h)
            bl = max(ls, key=ls.get)
            bd = max(dr, key=dr.get)
            rows[P] = (ls[bl], bl, dr[bd], bd)
        need = {lv: next((P for P in Ps if rows[P][0] >= lv), None) for lv in (0.9, 0.95, 0.99)}
        needd = {lv: next((P for P in Ps if rows[P][2] >= lv), None) for lv in (0.9, 0.95, 0.99)}
        res["gauss"][f"{fmax}_{mc}"] = dict(rows={str(k): v for k, v in rows.items()}, need=need, need_direct=needd)
        print(f"gauss fmax {fmax} mc {mc}: need LS {need} direct {needd}; P12 {rows[12][0]:.4f} (b {rows[12][1]}), P10 {rows[10][0]:.4f} b{rows[10][1]}, P14 {rows[14][0]:.4f} b{rows[14][1]}, P16 {rows[16][0]:.4f} b{rows[16][1]}")
        sys.stdout.flush()
json.dump(res, open(sys.argv[1], "w"), indent=1, default=str)
print("done", time.time() - t0)
