"""Independent checks of the theory's numbers (own code, numpy/scipy only)."""
import json, math
import numpy as np
from scipy import integrate
TS = 4.925490947e-6
out = {}
def tau(f, mc): return 5 / 256 * (TS * mc) ** (-5 / 3) * (np.pi * f) ** (-8 / 3)
def f_of_tau(t, mc): return (256 * t / 5) ** (-3 / 8) * (TS * mc) ** (-5 / 8) / np.pi
# 1. int tau df / cycles, 4 s window to merger and to 900 Hz
tab = {}
for mc in (0.85, 1.2, 1.8, 2.75):
    f0 = f_of_tau(4.0, mc)
    itd_inf = integrate.quad(lambda f: tau(f, mc), f0, np.inf, limit=500)[0]
    itd_900 = integrate.quad(lambda f: tau(f, mc), f0, 900, limit=500)[0]
    itd_512 = integrate.quad(lambda f: tau(f, mc), f0, 512, limit=500)[0]
    cyc = integrate.quad(lambda t: f_of_tau(t, mc), 0, 4.0, limit=500)[0]          # cycles = int f dt
    tab[mc] = dict(f_4s=f0, cycles_to_merger=cyc, int_tau_df_to_merger=itd_inf, int_tau_df_to_900=itd_900,
                   int_tau_df_to_512=itd_512, ratio=itd_inf / cyc, two_pi_int=2 * np.pi * itd_inf)
out["table"] = tab
out["2pi*3/8"] = 2 * np.pi * 3 / 8; out["3pi/4"] = 3 * np.pi / 4; out["5pi/4"] = 5 * np.pi / 4
# 2. phase bound: random channels, complex one-sided (N poles + D) and real (N conjugate pairs + D)
rng = np.random.default_rng(11)
w = np.linspace(0, 2 * np.pi * 600, 400001)
worst = []
for trial in range(200):
    N = int(rng.integers(2, 9))
    a = rng.uniform(2, 60, N); f = rng.uniform(20, 400, N)
    C = rng.normal(size=N) + 1j * rng.normal(size=N)
    D = rng.normal() if trial % 2 else 0.0
    real = trial % 4 >= 2
    s = 1j * w
    H = D + sum(C[n] / (s - (-a[n] + 2j * np.pi * f[n])) for n in range(N))
    if real:
        H = H + sum(np.conj(C[n]) / (s - (-a[n] - 2j * np.pi * f[n])) for n in range(N))
    th = np.unwrap(np.angle(H))
    # largest phase decrease over any band [w1, w2], w1 < w2: max over j of (max_{i<=j} th_i) - th_j
    dd = float(np.max(np.maximum.accumulate(th) - th))
    worst.append(dd / (2 * np.pi * N))
out["phase_bound_random"] = dict(trials=200, max_fraction_of_2piN=float(max(worst)), median=float(np.median(worst)))
# all-pass with N sections: phase turn equals 2 pi N (continuous time, positive freqs, real filter)
N = 6; a = np.full(N, 5.0); f = np.linspace(50, 300, N)
p = np.concatenate([-a + 2j * np.pi * f, -a - 2j * np.pi * f])
wfull = np.linspace(0, 2 * np.pi * 1e5, 2000001)
Hs = np.prod([(1j * wfull + np.conj(pp) * -1 + 0j) / (1j * wfull - pp) for pp in p], axis=0)  # zeros at -conj(p)
th = np.unwrap(np.angle(Hs))
out["allpass_turn_over_2piN"] = float((th[0] - th[-1]) / (2 * np.pi * N)); out["allpass_absH_dev"] = float(np.abs(np.abs(Hs) - 1).max())
# 3. GELU odd part
from scipy.special import erf
y = np.linspace(-8, 8, 10001)
gelu = lambda v: 0.5 * v * (1 + erf(v / np.sqrt(2)))
out["gelu_odd_minus_half_y_max"] = float(np.abs((gelu(y) - gelu(-y)) / 2 - y / 2).max())
# 4. lone resonator, two readouts: amplitude at the merger vs maximum over time (white noise, Newtonian chirp)
fs = 2048.0; L = 8192; t = np.arange(L) / fs; mc = 1.2
tm = 4.0 - 1 / fs                             # merger at the last sample
tt = tm - t; tt = np.clip(tt, 1e-4, None)
fr = f_of_tau(tt, mc)
phase = -2 * np.pi * integrate.cumulative_trapezoid(fr[::-1], t, initial=0)[::-1]
h = fr ** (2 / 3) * np.exp(1j * (2 * np.pi * np.cumsum(fr) / fs)) * (fr < 900)
H = np.fft.fft(h, 2 * L)
res4 = {}
for f0 in (100.0, 200.0):
    ta = tau(f0, mc); fdot = 96 / 5 * np.pi ** (8 / 3) * (TS * mc) ** (5 / 3) * f0 ** (11 / 3); tstep = fdot ** -0.5
    best = {"merger": (-1, None), "max": (-1, None)}
    for T in np.geomspace(0.003, 10, 120):
        for df in np.linspace(-3 / T if T > 0.05 else -30, 3 / T if T > 0.05 else 30, 25):
            k = np.exp((-1 / T + 2j * np.pi * (f0 + df)) * t)
            y = np.fft.ifft(H * np.fft.fft(k, 2 * L))[:L]
            m2 = np.abs(y) ** 2 / (np.sum(np.abs(k) ** 2) * np.sum(np.abs(h) ** 2))
            for key, v in (("merger", m2[-1]), ("max", m2.max())):
                if v > best[key][0]:
                    best[key] = (float(v), float(T))
    res4[f0] = dict(tau=ta, two_tau=2 * ta, t_step=tstep, best_ring_merger_readout=best["merger"][1],
                    best_ring_max_readout=best["max"][1], match_merger=best["merger"][0] ** 0.5, match_max=best["max"][0] ** 0.5,
                    ratio_merger_ring_over_tau=best["merger"][1] / ta, ratio_max_ring_over_tstep=best["max"][1] / tstep)
out["lone_resonator"] = res4
json.dump(out, open("v_theory.json", "w"), indent=1, default=float)
print(json.dumps(out, indent=1, default=float))
