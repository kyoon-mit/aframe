"""New numbers for the 15 Oct lecture-style update (leading-order chirp)."""
import numpy as np
T_SUN = 4.925490947e-6  # G M_sun / c^3 in s

def dfdt(f, mc):  # Newtonian frequency sweep rate (Maggiore vol 1 ch 4)
    return 96 / 5 * np.pi ** (8 / 3) * (T_SUN * mc) ** (5 / 3) * f ** (11 / 3)

def tau_of_f(f, mc):  # time before merger at frequency f
    return 5 / 256 * (T_SUN * mc) ** (-5 / 3) * (np.pi * f) ** (-8 / 3)

def f_of_tau(tau, mc):
    return (5 / 256 / tau) ** (3 / 8) * (T_SUN * mc) ** (-5 / 8) / np.pi

def cycles(f1, mc):  # cycles from f1 to merger
    return (T_SUN * mc) ** (-5 / 3) * f1 ** (-5 / 3) / (32 * np.pi ** (8 / 3))

print("1) time a chirp stays in step with one fixed frequency, (df/dt)^(-1/2)")
for mc in (0.85, 1.2, 2.75):
    row = [f"{f:g} Hz: {dfdt(f, mc):.3g} Hz/s, {dfdt(f, mc) ** -0.5:.3g} s" for f in (50, 100, 300)]
    print(f"  Mc {mc}: " + "; ".join(row))
for mc in (0.85, 1.2, 2.75):
    f4 = f_of_tau(4.0, mc)
    print(f"  Mc {mc}: f 4 s before merger {f4:.1f} Hz, cycles in last 4 s {cycles(f4, mc):.0f}")

print("2) phase drift of k cycles across a sum: |mean of exp(2 pi i k x)|")
x = np.linspace(0, 1, 200001)
for k in (0.25, 0.5, 1.0):
    print(f"  k={k}: {abs(np.exp(2j * np.pi * k * x).mean()):.4f}")

print("3) leftover cycles after heterodyning at the nearest point of a log grid")
def leftover(lo, hi, n, T):
    grid = np.geomspace(lo, hi, n)
    m = np.geomspace(0.85, 2.75, 4001)
    m = m[(m >= lo) & (m <= hi)]
    ref = grid[np.abs(np.log(m[:, None] / grid[None, :])).argmin(1)]
    N = cycles(f_of_tau(T, m), m)  # true signal's cycles in the window
    res = N * np.abs(1 - (m / ref) ** (5 / 8))  # Phi ~ Mc^(-5/8) at fixed time
    err = np.abs(m / ref - 1)
    return err.max(), res.max(), res.min(), np.percentile(res, 50)
for lo, hi, n, lab in ((0.85, 2.75, 10, "10 masses over 0.85-2.75"), (1.0, 2.5, 10, "10 masses over 1.0-2.5"),
                       (1.0, 2.5, 100, "100 masses over 1.0-2.5 (AFRAME BNS)")):
    for T in (3.5, 4.0):
        e, mx, mn, md = leftover(lo, hi, n, T)
        print(f"  {lab}, window {T} s: worst error {100 * e:.2f}%, leftover cycles worst {mx:.1f}, median {md:.1f}")
for d in (0.001, 0.01, 0.02, 0.05, 0.1):
    lo_c = min(cycles(f_of_tau(T, mc), mc) * abs(1 - (1 + d) ** (5 / 8)) for mc in (0.85, 2.75) for T in (3.5, 4.0))
    hi_c = max(cycles(f_of_tau(T, mc), mc) * abs(1 - (1 - d) ** (5 / 8)) for mc in (0.85, 2.75) for T in (3.5, 4.0))
    print(f"  error {100 * d:g}%: leftover {lo_c:.2f} to {hi_c:.2f} cycles ({2 * np.pi * lo_c:.1f} to {2 * np.pi * hi_c:.1f} rad)")

print("4) AFRAME-style grid (100 log-spaced masses over 1.0-2.5) inside +-e of an estimate")
g = np.geomspace(1.0, 2.5, 100)
for e in (0.05, 0.10):
    counts = [((g >= m * (1 - e)) & (g <= m * (1 + e))).sum() for m in np.geomspace(1.15, 2.2, 200)]
    print(f"  +-{100 * e:g}%: {min(counts)} to {max(counts)} grid masses (formula {np.log((1 + e) / (1 - e)) / np.log(2.5) * 99:.1f})")

print("5) first-order complex equation -> second-order real oscillator, numeric check")
a, w = 3.0, 2 * np.pi * 40.0
dt = 1e-6; t = np.arange(0, 0.2, dt)
u = np.sin(2 * np.pi * 25 * t) * np.exp(-((t - 0.05) / 0.02) ** 2)
x = np.zeros(len(t), complex)
for i in range(1, len(t)):  # x' = (-a + i w) x + u, exact step for piecewise-constant u
    A = -a + 1j * w
    x[i] = x[i - 1] * np.exp(A * dt) + u[i - 1] * (np.exp(A * dt) - 1) / A
p = x.real
pdd = np.gradient(np.gradient(p, dt), dt); pd = np.gradient(p, dt); ud = np.gradient(u, dt)
lhs = pdd + 2 * a * pd + (a * a + w * w) * p
rhs = ud + a * u
s = slice(100, -100)
print(f"  max |lhs - rhs| / max |rhs| = {np.abs(lhs[s] - rhs[s]).max() / np.abs(rhs[s]).max():.2e}")
