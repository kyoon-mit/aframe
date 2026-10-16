import numpy as np
from scipy.stats import chi2, binom
from scipy.optimize import brentq
print("== populations, density ~ x^-3")
def F(x, a, b):  # CDF of x^-3 on [a,b]
    return (a**-2 - x**-2) / (a**-2 - b**-2)
for a, b in [(4, 50), (4, 100), (8, 50), (8, 100)]:
    med = brentq(lambda x: F(x, a, b) - 0.5, a, b)
    out = f"[{a},{b}] median {med:.4f}"
    if a == 4:
        out += f" P(<8) {F(8,a,b):.5f} P(8-12) {F(12,a,b)-F(8,a,b):.5f} P(>=12) {1-F(12,a,b):.5f}"
    else:
        out += f" P(8-12) {F(12,a,b):.5f}"
    print(out)
# Monte Carlo check of the CDF with ml4gw-style inverse transform
rng = np.random.default_rng(0)
u = rng.random(2_000_000)
for a, b in [(4, 50), (4, 100)]:
    x = (a**-2 - u * (a**-2 - b**-2)) ** -0.5
    print(f"MC [{a},{b}]: P(<8) {np.mean(x<8):.4f} P(8-12) {np.mean((x>=8)&(x<12)):.4f} median {np.median(x):.4f}")
print("== uniform in volume, density ~ x^-4 above threshold t")
print("P(rho<1.5 t) =", 1 - 1.5**-3, " volume ratio (8/12)^3 =", (8/12)**3, " P(<8 | >=4) =", 1 - 2**-3)
x = 8 * rng.random(2_000_000) ** (-1/3)   # inverse CDF of 3 t^3/x^4
print("MC uniform-in-volume P(8-12) =", np.mean(x < 12))
# also from uniform in volume directly: r = R u^(1/3), rho = 8 R / r
r = rng.random(2_000_000) ** (1/3); rho = 8 / r
print("MC from positions: P(rho<12 | rho>=8) =", np.mean(rho < 12))
print("== validation counts (chirp-kernel run): 16 batches x 32, waveform_prob 0.96, [4,100]")
n = 512; p = 0.96 * (F(12,4,100) - F(8,4,100))
print(f"p per window {p:.5f} mean {n*p:.2f} sd {np.sqrt(n*p*(1-p)):.3f}; signal windows mean {n*0.96:.1f} sd {np.sqrt(n*0.96*0.04):.2f}")
sim = binom.rvs(n, p, size=200000, random_state=1); print("MC mean %.2f sd %.3f, 5-95%% %s" % (sim.mean(), sim.std(), np.percentile(sim,[5,95])))
for q in [0.9, 0.8, 0.7]:
    print(f"binomial sd of a fraction {q} on 68 events: {100*np.sqrt(q*(1-q)/68):.2f} points")
print("== kernel spacing")
s = np.log(2.75/0.85)/511; print("log step", s, "ratio-1 = %.4f%%" % (100*(np.exp(s)-1)))
print("== TSUN")
GM = 1.3271244e20; c = 299792458.0
print("GM/c^3 = %.7e s" % (GM/c**3))
print("== noise floor")
def med_max(N, k=4):
    return brentq(lambda x: N*np.log1p(-chi2.sf(x, k)) - np.log(0.5), 0.01, 400)
Nf = 613*1201*41
for N in [1, 613*1201*41]:
    print(N, "median max rho = %.3f" % np.sqrt(med_max(N)))
for rr in [5.967, 6.269]:
    print("rho", rr, "N_eff = %.3g" % (np.log(0.5)/np.log1p(-chi2.sf(rr*rr, 4))))
Ne = np.log(0.5)/np.log1p(-chi2.sf(5.967**2, 4))
print("10x N_eff -> rho %.3f ; 100x -> %.3f" % (np.sqrt(med_max(10*Ne)), np.sqrt(med_max(100*Ne))))
print("coincidence window 20 samples at 2048 Hz = %.3f ms" % (20/2048*1e3))
print("== time average of a tone: 1/(pi f T) at f=20 Hz, T=4 s:", 1/(np.pi*20*4))
t = np.arange(8192)/2048
worst = max(abs(np.mean(np.cos(2*np.pi*f*t+ph))) for f in np.linspace(20, 30, 201) for ph in np.linspace(0, np.pi, 13))
print("numerical worst |mean cos| over f in 20-30 Hz, phase:", worst)
print("== Newtonian scalings check")
TS = GM/c**3
def fgw(tau, M): T = TS*M; return (1/(8*np.pi))*(5/tau)**(3/8)*T**(-5/8)
def fdot(f, M): T = TS*M; return (96/5)*np.pi**(8/3)*T**(5/3)*f**(11/3)
print("f at tau=1 s: M=1.0 %.2f Hz, M=2.0 %.2f Hz, ratio %.4f vs 2^(-5/8) %.4f" % (fgw(1,1.0), fgw(1,2.0), fgw(1,2.0)/fgw(1,1.0), 2**(-5/8)))
print("df/dt at 100 Hz: ratio M2/M1 %.4f vs 2^(5/3) %.4f" % (fdot(100,2.0)/fdot(100,1.0), 2**(5/3)))
# check fdot consistency with f(tau): df/dtau numerically
tau = 1.0; h = 1e-6; M = 1.2
num = -(fgw(tau+h, M)-fgw(tau-h, M))/(2*h)
print("df/dt numeric %.4f vs formula %.4f" % (num, fdot(fgw(tau, M), M)))
# phase: dPhi/dt = 2 pi f with Phi = -2 (tau/(5T))^(5/8)
T = TS*M; Phi = lambda tau: -2*(tau/(5*T))**(5/8)
print("2pi f vs -dPhi/dtau: %.4f %.4f" % (2*np.pi*fgw(tau, M), (Phi(tau+h)-Phi(tau-h))/(2*h)))
