"""Checks for the ten-mass grid worst case: errors, leftover frequency ratio, in-step factor, Fourier counts."""
import numpy as np

T_SUN = 4.925490947e-6  # G M_sun / c^3 in s


def tm(mc):
    return T_SUN * mc


def f_of_tau(tau, mc):
    return (5 / tau) ** (3 / 8) * tm(mc) ** (-5 / 8) / (8 * np.pi)


def phase(tau, mc):
    return -2 * (tau / (5 * tm(mc))) ** (5 / 8)


def tau_of_f(f, mc):
    return 5 / 256 * (np.pi * f) ** (-8 / 3) * tm(mc) ** (-5 / 3)


# 1) grid errors and leftover fractions
lo, hi, n = 0.85, 2.75, 10
g = np.geomspace(lo, hi, n)
r = g[1] / g[0]
print(f"1) ratio r = {r:.6f} (closed form {(hi / lo) ** (1 / (n - 1)):.6f})")
up, dn = np.sqrt(r) - 1, 1 - 1 / np.sqrt(r)
print(f"   largest errors: true mass {100 * up:.4f}% above, {100 * dn:.4f}% below the nearest grid mass")
# brute force over true masses between the grid ends, nearest grid mass in log
mt = np.geomspace(lo, hi, 400001)
near = g[np.argmin(np.abs(np.log(mt)[:, None] - np.log(g)[None, :]), axis=1)]
rel = mt / near - 1
print(f"   brute force on 400,001 masses: max {100 * rel.max():.4f}%, min {100 * rel.min():.4f}%")
cases = {"+6.74%": up, "-6.31%": -dn}
q = {}
for k, eps in cases.items():
    q[k] = abs(1 - (1 + eps) ** (5 / 8))
    print(f"   eps {k}: q = |1-(1+eps)^(5/8)| = {q[k]:.6f}; first order 5/8|eps| = {5 / 8 * abs(eps):.6f}; "
          f"q^-1/2 = {q[k] ** -0.5:.4f}")
print(f"   r^(5/16)-1 = {r ** (5 / 16) - 1:.6f}, 1-r^(-5/16) = {1 - r ** (-5 / 16):.6f}")

# 2) leftover frequency and sweep are q times the chirp's, at every time (aligned at merger)
for mc in (0.85, 1.2, 2.75):
    tau = np.linspace(0.05, 4.0, 200001)
    for k, eps in cases.items():
        mt_ = mc / (1 + eps)  # assumed mass, true = assumed * (1 + eps)
        psi = phase(tau, mc) - phase(tau, mt_)
        d1, d1c = np.gradient(psi, tau), np.gradient(phase(tau, mc), tau)
        d2, d2c = np.gradient(d1, tau), np.gradient(d1c, tau)
        s = slice(10, -10)
        rf = np.abs(d1[s] / d1c[s]); rs = np.abs(d2[s] / d2c[s])
        print(f"2) Mc {mc} eps {k}: freq ratio {rf.min():.6f} to {rf.max():.6f}; sweep ratio {rs.min():.6f} to {rs.max():.6f}")

# 3) in-step time of the leftover against the chirp's, measured on the exact phases
def in_step_len(ph, t, t0):
    """Length of the interval around t0 where ph stays within pi/4 of its tangent line at t0."""
    i0 = np.argmin(np.abs(t - t0))
    slope = np.gradient(ph, t)[i0]
    dev = np.abs(ph - ph[i0] - slope * (t - t[i0]))
    ok = dev <= np.pi / 4
    a = i0
    while a > 0 and ok[a - 1]:
        a -= 1
    b = i0
    while b < len(t) - 1 and ok[b + 1]:
        b += 1
    return t[b] - t[a]

mc = 1.2
t = np.linspace(-12.0, -0.002, 2400001)  # time relative to merger, s
tau = -t
for f0 in (100.0, 200.0):
    t0 = -tau_of_f(f0, mc)
    Lc = in_step_len(phase(tau, mc), t, t0)
    for k, eps in cases.items():
        psi = phase(tau, mc) - phase(tau, mc / (1 + eps))
        Ll = in_step_len(psi, t, t0)
        print(f"3) Mc 1.2, chirp at {f0:.0f} Hz: chirp {Lc:.5f} s, leftover {Ll:.5f} s, ratio {Ll / Lc:.6f} "
              f"(q^-1/2 = {q[k] ** -0.5:.4f})")

# 4) Fourier waves for match 0.9 (81% of the power in the strongest complex coefficients)
def count(sig, frac=0.81):
    p = np.sort(np.abs(np.fft.fft(sig)) ** 2)[::-1]
    c = np.cumsum(p) / p.sum()
    return int(np.searchsorted(c, frac) + 1)

fs, L = 2048.0, 8192
tt = np.arange(L) / fs
eps_all = {"none": None, "+6.74%": up, "-6.31%": -dn, "+1%": 0.01, "-1%": -0.01}
rows = {}
for mc in (1.0, 1.4, 1.8, 2.4):
    tau = np.maximum(4.0 - tt, 0.5 / fs)
    f = f_of_tau(tau, mc)
    A = np.where((f >= 20) & (f <= 900), tau ** -0.25, 0.0)
    out = []
    for k, eps in eps_all.items():
        ph = phase(tau, mc) if eps is None else phase(tau, mc) - phase(tau, mc / (1 + eps))
        out.append(count(A * np.exp(1j * ph)))
    rows[mc] = out
    print(f"4) Mc {mc}: " + ", ".join(f"{k} {v}" for k, v in zip(eps_all, out)))
for j, k in enumerate(eps_all):
    v = [rows[m][j] for m in rows]
    print(f"   {k}: {min(v)} to {max(v)}")

# 5) leftover frequency band at 1.2 Msun while the chirp rises from f(4 s) to 512 Hz
f4 = f_of_tau(4.0, 1.2)
print(f"5) 1.2 Msun: f(4 s) = {f4:.3f} Hz; reaches 512 Hz {tau_of_f(512.0, 1.2) * 1e3:.2f} ms before the merger")
for k in cases:
    print(f"   eps {k}: leftover {q[k] * f4:.3f} Hz to {q[k] * 512:.3f} Hz")

# 6) S4D-Lin: tone n lasts n cycles, so 32 tones (n = 0..31) lasting 1 s sit at 0..31 Hz
n_ = np.arange(32)
dt_fs = 2.0  # T = 2 / (dt * fs) = 1 s
fn = n_ * dt_fs / 2
Tn = 2 / dt_fs
print(f"6) S4D-Lin, 32 tones, T = {Tn} s: frequencies {fn.min():.0f} to {fn.max():.0f} Hz")
