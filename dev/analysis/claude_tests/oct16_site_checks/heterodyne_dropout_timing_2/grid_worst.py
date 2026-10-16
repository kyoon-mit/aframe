"""Worst case of the ten-mass heterodyne grid: leftover frequency, in-step factor, Fourier counts.

Independent of lecture_rev/verify/calc.py: the worst errors come from a scan over true chirp
masses, the frequency ratio from finite differences of the phase difference, the in-step
times from the exact leading-order phase, and the counts are re-derived with the
#ck-fourier-tones method (complex A e^{i Phi}, 4 s at 2048 Hz, 20 to 900 Hz, 81% of power).
"""
import numpy as np

T_SUN = 4.925490947e-6                       # G M_sun / c^3 in s


def tm(mc):
    return T_SUN * mc


def f_of_tau(tau, mc):
    return (5 / tau) ** (3 / 8) * tm(mc) ** (-5 / 8) / (8 * np.pi)


def phase(tau, mc):
    return -2 * (tau / (5 * tm(mc))) ** (5 / 8)


def fdot_of_tau(tau, mc):
    # df/dt = -df/dtau
    return (3 / 8) * f_of_tau(tau, mc) / tau


def q_of(eps):
    return abs(1 - (1 + eps) ** (5 / 8))


print("A) grids")
for name, lo, hi, n in (("ten-mass", 0.85, 2.75, 10), ("AFRAME", 1.0, 2.5, 100)):
    g = np.geomspace(lo, hi, n)
    r = g[1] / g[0]
    true = np.geomspace(lo, hi, 400_001)
    lg, lt = np.log(g), np.log(true)
    j = np.clip(np.searchsorted(lg, lt), 1, n - 1)
    k = np.where(np.abs(lt - lg[j - 1]) <= np.abs(lt - lg[j]), j - 1, j)
    eps = true / g[k] - 1                    # true = assumed * (1 + eps)
    q = np.abs(1 - (1 + eps) ** (5 / 8))
    print(f"  {name}: ratio {r:.6f}; sqrt(r)-1 = {100*(np.sqrt(r)-1):.4f}%, 1/sqrt(r)-1 = {100*(1/np.sqrt(r)-1):.4f}%")
    print(f"    scan: max eps {100*eps.max():+.4f}%, min eps {100*eps.min():+.4f}%, max leftover fraction {100*q.max():.4f}%")
    for e in (eps.max(), eps.min()):
        print(f"    eps {100*e:+.3f}%: q = {100*q_of(e):.3f}% (first order {62.5*abs(e):.3f}%), "
              f"in-step factor q^-1/2 = {q_of(e)**-0.5:.3f}")

w = np.sqrt((2.75 / 0.85) ** (1 / 9)) - 1
EPS = {"+6.74% (grid)": w, "-6.31% (grid)": 1 / (1 + w) - 1, "-6.74% (outside grid)": -w,
       "+1%": 0.01, "-1%": -0.01}

print("B) leftover frequency and sweep from finite differences of the phase difference")
tau = np.linspace(0.03, 4.0, 200_001)
for mc in (0.85, 1.2, 2.75):
    for lab in ("+6.74% (grid)", "-6.31% (grid)"):
        e = EPS[lab]
        dphi = phase(tau, mc) - phase(tau, mc / (1 + e))       # leftover phase
        f_left = np.abs(np.gradient(dphi, tau)) / (2 * np.pi)   # |d/dtau| = |d/dt|
        ratio = f_left / f_of_tau(tau, mc)
        sweep = np.abs(np.gradient(np.gradient(dphi, tau), tau)) / (2 * np.pi)
        rs = sweep[5:-5] / fdot_of_tau(tau, mc)[5:-5]
        print(f"  Mc {mc} {lab}: freq ratio {ratio[5:-5].min():.6f} to {ratio[5:-5].max():.6f}; "
              f"sweep ratio {rs.min():.6f} to {rs.max():.6f}; q = {q_of(e):.6f}")

print("C) leftover frequencies at 1.2 Msun between f(4 s) and 512 Hz")
f4 = f_of_tau(4.0, 1.2)
for lab in ("+6.74% (grid)", "-6.31% (grid)", "-6.74% (outside grid)"):
    q = q_of(EPS[lab])
    print(f"  {lab}: f(4 s) = {f4:.3f} Hz -> {q*f4:.3f} Hz; 512 Hz -> {q*512:.3f} Hz")
tau512 = 5 * (8 * np.pi * 512) ** (-8 / 3) * tm(1.2) ** (-5 / 3)
print(f"  the chirp reaches 512 Hz {1000*tau512:.2f} ms before the merger")

print("D) in-step interval on the exact leading-order phase (1.2 Msun): leftover / chirp")
def good_size_minus_1(x):
    return x.size - 1


def interval(tau0, mc, tol):
    # connected interval around tau0 where |Phi - tangent line| <= tol (radians), merger excluded
    s = np.linspace(-6.0, tau0 - 1e-6, 2_000_001)            # s > 0 toward the merger (t increases)
    tt = tau0 - s
    ok = tt > 0
    ph = phase(tt[ok], mc)
    w0 = 2 * np.pi * f_of_tau(tau0, mc)                       # dPhi/dt at tau0
    dev = np.abs(ph - phase(tau0, mc) - w0 * s[ok])
    i0 = np.argmin(np.abs(s[ok]))
    bad = np.flatnonzero(dev > tol)
    lo, hi = bad[bad < i0], bad[bad > i0]
    a = lo[-1] + 1 if lo.size else 0
    b = hi[0] - 1 if hi.size else good_size_minus_1(dev)
    hit_end = not hi.size
    return s[ok][b] - s[ok][a], hit_end
for tau0 in (4.0, 3.0, 2.0, 1.0, 0.5):
    Tc, _ = interval(tau0, 1.2, np.pi / 4)
    row = [f"tau0 {tau0} s: chirp {Tc:.4f} s (fdot^-1/2 = {fdot_of_tau(tau0, 1.2)**-0.5:.4f} s)"]
    for lab in ("+6.74% (grid)", "-6.31% (grid)"):
        q = q_of(EPS[lab])
        Tl, end = interval(tau0, 1.2, np.pi / 4 / q)        # leftover phase = q * chirp phase
        row.append(f"{lab}: {Tl:.4f} s, ratio {Tl/Tc:.3f} (q^-1/2 {q**-0.5:.3f}){' [reaches merger]' if end else ''}")
    print("  " + "; ".join(row))

print("E) Fourier tones for a match of 0.9 (81% of power), #ck-fourier-tones method")
def count(sig):
    p = np.sort(np.abs(np.fft.fft(sig)) ** 2)[::-1]
    c = np.cumsum(p) / p.sum()
    k = int(np.searchsorted(c, 0.81) + 1)
    assert np.sqrt(c[k - 1]) >= 0.9 > np.sqrt(c[k - 2])       # match with k tones >= 0.9 > with k-1
    return k
fs, L = 2048.0, 8192
t = np.arange(L) / fs
res = {}
for mc in (1.0, 1.4, 1.8, 2.4):
    tau = np.maximum(4.0 - t, 0.5 / fs)
    f = f_of_tau(tau, mc)
    A = np.where((f >= 20) & (f <= 900), tau ** -0.25, 0.0)
    row = {"chirp": count(A * np.exp(1j * phase(tau, mc)))}
    for lab, e in EPS.items():
        row[lab] = count(A * np.exp(1j * (phase(tau, mc) - phase(tau, mc / (1 + e)))))
    res[mc] = row
    print(f"  Mc {mc}: " + ", ".join(f"{k} {v}" for k, v in row.items()))
for lab in ["chirp"] + list(EPS):
    v = [res[m][lab] for m in res]
    print(f"  {lab}: {min(v)} to {max(v)}")
grid = [res[m][k] for m in res for k in ("+6.74% (grid)", "-6.31% (grid)")]
first = [res[m][k] for m in res for k in ("+6.74% (grid)", "-6.74% (outside grid)")]
one = [res[m][k] for m in res for k in ("+1%", "-1%")]
print(f"  grid worst, both signs: {min(grid)} to {max(grid)}; first version (+-6.74%): {min(first)} to {max(first)}; "
      f"1% both signs: {min(one)} to {max(one)}")

print("F) S4D-Lin channel of 32 oscillators lasting 1 s: f_n = n dt fs / 2 with dt fs = 2")
print("  frequencies", [n * 2 / 2 for n in range(32)][:3], "...", 31 * 2 / 2, "Hz")
