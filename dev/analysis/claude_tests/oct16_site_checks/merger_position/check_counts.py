"""Leading-order frequency, cycles and integral of tau df for the stretch of chirp a window holds.

Two methods: closed forms (f from tau, N = (8/5) tau1 f1, int tau df = (3/5) tau1 f1) and
numerical quadrature of f dt and tau df with scipy, independently of the closed forms.
"""
import numpy as np
from scipy.integrate import quad

T_SUN = 4.925490947e-6          # G M_sun / c^3 in s (IAU 2015 nominal, as in #ck-chirp)
T_SUN_ALT = 6.67430e-11 * 1.98847e30 / 299792458.0**3
FS = 2048
MASSES = [0.85, 1.2, 1.8, 2.75]


def f_of_tau(tau, mc, tsun=T_SUN):
    T = mc * tsun
    return (5.0 / (256.0 * tau)) ** 0.375 * T ** (-0.625) / np.pi


def tau_of_f(f, mc, tsun=T_SUN):
    T = mc * tsun
    return 5.0 / 256.0 * T ** (-5.0 / 3.0) * (np.pi * f) ** (-8.0 / 3.0)


def closed(tau1, mc, tsun=T_SUN):
    f1 = f_of_tau(tau1, mc, tsun)
    return f1, 1.6 * tau1 * f1, 0.6 * tau1 * f1


def numeric(tau1, mc, tau2=0.0):
    """cycles = int_{tau2}^{tau1} f dtau ; int tau df over [f(tau1), f(tau2)]."""
    cyc = quad(lambda t: f_of_tau(t, mc), tau2, tau1, limit=500, points=[1e-6, 1e-3])[0]
    f1 = f_of_tau(tau1, mc)
    f2 = np.inf if tau2 == 0 else f_of_tau(tau2, mc)
    itd = quad(lambda f: tau_of_f(f, mc), f1, f2, limit=500)[0]
    return cyc, itd


print(f"G Msun/c^3 = {T_SUN:.9e} s (alt from G, Msun: {T_SUN_ALT:.9e} s)")
cases = {
    "4 s (merger at the window's last edge)": 4.0,
    "3.5 s": 3.5,
    "7169/2048 s (earliest Earth-centre merger)": 7169 / FS,
    "7126/2048 s (earliest in one detector, shift 43 samples)": 7126 / FS,
}
for label, tau1 in cases.items():
    print(f"\npre-merger stretch {label} = {tau1:.6f} s")
    for mc in MASSES:
        f1, n, itd = closed(tau1, mc)
        n2, itd2 = numeric(tau1, mc)
        f1a, na, itda = closed(tau1, mc, T_SUN_ALT)
        print(f"  Mc {mc:4.2f}: f1 {f1:8.3f} Hz, cycles {n:8.2f} (quad {n2:8.2f}), int tau df {itd:7.2f} (quad {itd2:7.2f}), "
              f"ratio {itd / n:.6f}; alt constants f1 {f1a:.3f} cycles {na:.2f} int {itda:.2f}")

r_cnt = (3.5 / 4.0) ** 0.625
r_f = (4.0 / 3.5) ** 0.375
print(f"\n(3.5/4)^(5/8) = {r_cnt:.6f}; (4/3.5)^(3/8) = {r_f:.6f}; (7169/8192)^(5/8) = {(7169 / 8192) ** 0.625:.6f}; (7126/8192)^(5/8) = {(7126 / 8192) ** 0.625:.6f}")

# merger after the window end in one detector: the window holds tau from 4 + d down to d
d = 43 / FS
print(f"\nmerger {d * 1e3:.3f} ms after the window's last edge (one detector): window holds tau from {4 + d:.6f} s to {d:.6f} s")
for mc in MASSES:
    cyc, itd = numeric(4.0 + d, mc, d)
    f_lo, f_hi = f_of_tau(4 + d, mc), f_of_tau(d, mc)
    print(f"  Mc {mc:4.2f}: f {f_lo:.3f} to {f_hi:.1f} Hz, cycles {cyc:.2f}, int tau df {itd:.2f} (below the 4 s value: {itd < closed(4.0, mc)[2]})")

# start frequency over all merger positions, and per-cycle SNR at SNR 8
lo = min(closed(4.0, mc)[0] for mc in MASSES)
hi = max(closed(3.5, mc)[0] for mc in MASSES)
print(f"\nstart frequency over 0.85 to 2.75 Msun and merger 3.5 to 4 s in: {lo:.3f} to {hi:.3f} Hz")
nmin = closed(3.5, 2.75)[1]
nmax = closed(4.0, 0.85)[1]
print(f"cycles over the same: {nmin:.2f} to {nmax:.2f}; 8/sqrt(N): {8 / np.sqrt(nmax):.4f} to {8 / np.sqrt(nmin):.4f}")
# chirp mass at which the 3.5 s stretch starts at 50 Hz
mc50_4 = 1.2 * (closed(4.0, 1.2)[0] / 50.0) ** 1.6
mc50_35 = 1.2 * (closed(3.5, 1.2)[0] / 50.0) ** 1.6
print(f"chirp mass whose stretch starts at 50 Hz: {mc50_4:.4f} Msun (4 s), {mc50_35:.4f} Msun (3.5 s)")
# pre-merger stretch at which a 2.75 Msun chirp starts at 50 Hz
print(f"2.75 Msun reaches 50 Hz at tau = {tau_of_f(50.0, 2.75):.4f} s before the merger")
# 1.2 Msun hold time at 80 Hz versus the window
print(f"1.2 Msun: tau(80 Hz) = {tau_of_f(80.0, 1.2):.4f} s; f(4 s) = {f_of_tau(4.0, 1.2):.3f} Hz; f(3.5 s) = {f_of_tau(3.5, 1.2):.3f} Hz")
