"""Newtonian GW frequency f(tau) and check d ln(fdot)/dt = 11/(8 tau)."""
import numpy as np
G, c, Msun = 6.67430e-11, 299792458.0, 1.98847e30
def f_gw(tau, mc):
    tm = G * mc * Msun / c**3
    return (1 / np.pi) * (5 / (256 * tau)) ** 0.375 * tm ** (-0.625)
mc = 1.2
for tau in [100.0, 10.0, 4.0, 1.0, 0.1]:
    h = tau * 1e-4
    # t increases as tau decreases: fdot = -df/dtau
    fdot = lambda T: -(f_gw(T + h, mc) - f_gw(T - h, mc)) / (2 * h)
    dln = -(np.log(fdot(tau + h)) - np.log(fdot(tau - h))) / (2 * h)
    print(f"tau={tau:6.1f} s  f={f_gw(tau, mc):8.2f} Hz  fdot={fdot(tau):10.3f} Hz/s  dln(fdot)/dt={dln:.5f}  11/(8tau)={11/(8*tau):.5f}")
