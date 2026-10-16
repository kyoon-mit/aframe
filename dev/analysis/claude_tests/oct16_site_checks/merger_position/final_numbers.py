"""Numbers quoted in the proposed edits, with the largest closed-form vs quadrature difference."""
import numpy as np
from scipy.integrate import quad

TS = 4.925490947e-6
TS_ALT = 6.67430e-11 * 1.98847e30 / 299792458.0**3
M = [0.85, 1.2, 1.8, 2.75]


def f_of_tau(tau, mc, ts=TS):
    return (5.0 / (256.0 * tau)) ** 0.375 * (mc * ts) ** (-0.625) / np.pi


def tau_of_f(f, mc, ts=TS):
    return 5.0 / 256.0 * (mc * ts) ** (-5.0 / 3.0) * (np.pi * f) ** (-8.0 / 3.0)


worst_n = worst_i = 0.0
for tau1 in (4.0, 3.5):
    print(f"tau1 = {tau1} s")
    for mc in M:
        f1 = f_of_tau(tau1, mc)
        n, i = 1.6 * tau1 * f1, 0.6 * tau1 * f1
        nq = quad(lambda t: f_of_tau(t, mc), 0.0, tau1, limit=500, epsabs=1e-12, epsrel=1e-13)[0]
        iq = quad(lambda f: tau_of_f(f, mc), f1, np.inf, limit=500, epsabs=1e-12, epsrel=1e-13)[0]
        worst_n = max(worst_n, abs(n - nq) / n)
        worst_i = max(worst_i, abs(i - iq) / i)
        f1a = f_of_tau(tau1, mc, TS_ALT)
        print(f"  {mc:4.2f}: f1 {f1:.4f} Hz (alt {f1a:.4f}), N {n:.3f}, int {i:.3f}, ceil(int) {int(np.ceil(i))}, "
              f"alt N {1.6 * tau1 * f1a:.3f} alt int {0.6 * tau1 * f1a:.3f}")
print(f"largest relative difference closed vs quadrature: cycles {worst_n:.1e}, int tau df {worst_i:.1e}")
print(f"(4/3.5)^(3/8) = {(4 / 3.5) ** 0.375:.5f}, (3.5/4)^(5/8) = {(3.5 / 4) ** 0.625:.5f}")
print(f"8/sqrt(N): {8 / np.sqrt(1.6 * 3.5 * f_of_tau(3.5, 2.75)):.4f} (2.75, 3.5 s), {8 / np.sqrt(1.6 * 4 * f_of_tau(4.0, 0.85)):.4f} (0.85, 4 s)")
