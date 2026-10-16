"""Leading-order f(tau), cycles and int tau df: closed form against quadrature (independent check)."""
import numpy as np
from scipy.integrate import quad

T_SUN = 4.925490947e-6   # G M_sun / c^3 in s (IAU nominal)
MCS = [0.85, 1.2, 1.8, 2.75]

def f_of_tau(tau, mc):
    tm = mc * T_SUN
    return (5.0 / (256.0 * tau)) ** 0.375 * tm ** (-0.625) / np.pi

def tau_of_f(f, mc):
    tm = mc * T_SUN
    return 5.0 / 256.0 * tm ** (-5.0 / 3.0) * (np.pi * f) ** (-8.0 / 3.0)

def n_closed(f1, mc):
    tm = mc * T_SUN
    return tm ** (-5.0 / 3.0) * f1 ** (-5.0 / 3.0) / (32.0 * np.pi ** (8.0 / 3.0))

def n_quad(tau1, mc, tau2=0.0):
    # cycles between tau1 and tau2 before merger: int f dtau
    v, _ = quad(lambda t: f_of_tau(t, mc), tau2, tau1, limit=500, epsabs=0, epsrel=1e-13)
    return v

def itdf_quad(f1, mc, f2=np.inf):
    v, _ = quad(lambda f: tau_of_f(f, mc), f1, f2, limit=500, epsabs=0, epsrel=1e-13)
    return v

fs = 2048.0
taus = {'4 s': 4.0, '3.5 s': 3.5, '7169/fs': 7169 / fs, '7126/fs (det min)': 7126 / fs}
maxrel = 0.0
for name, tau in taus.items():
    print(f"tau = {name} = {tau:.6f} s")
    for mc in MCS:
        f1 = f_of_tau(tau, mc)
        nc = n_closed(f1, mc)
        nq = n_quad(tau, mc)
        ic = 0.6 * tau * f1
        iq = itdf_quad(f1, mc)
        maxrel = max(maxrel, abs(nc - nq) / nc, abs(ic - iq) / ic, abs(nc - 1.6 * tau * f1) / nc)
        print(f"   {mc:4.2f}: f1 {f1:9.4f} Hz  N {nc:8.3f} (quad {nq:8.3f})  int tau df {ic:8.3f} (quad {iq:8.3f})  8/sqrt(N) {8/np.sqrt(nc):.4f}")
print(f"largest relative difference closed form vs quadrature: {maxrel:.2e}")
print(f"(4/3.5)^(3/8) = {(4/3.5)**0.375:.6f}, (3.5/4)^(5/8) = {(3.5/4)**0.625:.6f}")

# per-detector: merger at window sample m (first sample 0, last 8191); window covers samples 0..8191
print("per-detector merger positions m (window samples), chirp in window from tau_s = m/fs to tau_e = max(m-8191,0)/fs:")
for m in [7126, 7169, 8191, 8192, 8193, 8235]:
    ts = m / fs
    te = max(m - 8191, 0) / fs
    row = []
    for mc in MCS:
        fsx = f_of_tau(ts, mc)
        i_band = 0.6 * (ts * fsx - (te * f_of_tau(te, mc) if te > 0 else 0.0))
        n_band = 1.6 * (ts * fsx - (te * f_of_tau(te, mc) if te > 0 else 0.0))
        row.append(f"{mc}: f_s {fsx:.3f} N {n_band:.2f} I {i_band:.2f}")
    print(f"   m={m} ({ts:.6f} s, tau_e {te*1e3:.3f} ms): " + "; ".join(row))
# the drafter's case: window end taken at 4.0 s, merger 21.0 ms later (tau_e = 0.020996 s)
ts, te = 8235 / fs, 43 / fs
for mc in [0.85]:
    i_band = 0.6 * (ts * f_of_tau(ts, mc) - te * f_of_tau(te, mc))
    print(f"   drafter case 0.85: tau_s {ts:.6f}, tau_e {te*1e3:.3f} ms -> I {i_band:.3f}")
# maximum over all per-detector positions of I for 0.85
best = max((0.6 * ((m/fs) * f_of_tau(m/fs, 0.85) - ((m-8191)/fs * f_of_tau((m-8191)/fs, 0.85) if m > 8191 else 0)), m) for m in range(7126, 8236))
print(f"   max over m of I at 0.85: {best[0]:.3f} at m = {best[1]}")
bestN = min((1.6 * ((m/fs) * f_of_tau(m/fs, 2.75) - ((m-8191)/fs * f_of_tau((m-8191)/fs, 2.75) if m > 8191 else 0)), m) for m in range(7126, 8236))
print(f"   min over m of N at 2.75: {bestN[0]:.3f} at m = {bestN[1]}")
bestNmax = max((1.6 * ((m/fs) * f_of_tau(m/fs, 0.85) - ((m-8191)/fs * f_of_tau((m-8191)/fs, 0.85) if m > 8191 else 0)), m) for m in range(7126, 8236))
print(f"   max over m of N at 0.85: {bestNmax[0]:.3f} at m = {bestNmax[1]}")
# chirp mass whose 4 s window starts at 50 Hz, and at 3.5 s
from scipy.optimize import brentq
for tau in [4.0, 3.5, 7169/fs]:
    mc50 = brentq(lambda m: f_of_tau(tau, m) - 50.0, 0.5, 5)
    print(f"   chirp mass starting at 50 Hz with tau = {tau:.6f} s: {mc50:.4f}")
