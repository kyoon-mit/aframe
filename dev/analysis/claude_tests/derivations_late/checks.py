import numpy as np
rng = np.random.default_rng(1)
print("== power-law fractions, density ~ rho^-3")
def F(lo, hi, a, b, k=3):  # fraction of density rho^-k on [a,b] lying in [lo,hi]
    I = lambda x, y: (x**(1-k) - y**(1-k))
    return I(lo, hi) / I(a, b)
print("pl4 [4,50]: <8 %.4f, 8-16 %.4f, >16 %.4f, 25-50 %.4f" % (F(4,8,4,50), F(8,16,4,50), F(16,50,4,50), F(25,50,4,50)))
print("pl4 [4,100] (validation): <8 %.4f" % F(4,8,4,100))
print("pl8 [8,50]: 8-16 %.4f, 25-50 %.4f" % (F(8,16,8,50), F(25,50,8,50)))
print("uniform [4,50]: 25-50 %.4f, >16 %.4f" % ((50-25)/46, (50-16)/46))
print("median pl4 [4,50]:", (0.5*(4**-2 + 50**-2))**-0.5)
print("volume law rho^-4 above 8: 8-16 %.4f (untruncated 1-(1/2)^3 = %.4f); truncated at 50: %.4f" % (1-(8/16)**3, 1-0.5**3, F(8,16,8,50,k=4)))
# Monte Carlo by inverse CDF for pl4
u = rng.random(2_000_000); a, b = 4, 50
x = (a**-2 - u*(a**-2 - b**-2))**-0.5
print("MC pl4: <8 %.4f, 8-16 %.4f" % ((x<8).mean(), ((x>=8)&(x<16)).mean()))
print("== MSE shrinkage")
n = 4096; S = 1.0
for N in (0.25, 1.0, 4.0):
    gs = []
    for _ in range(2000):
        s = rng.standard_normal(n); s *= np.sqrt(S*n)/np.linalg.norm(s)
        e = rng.standard_normal(n)*np.sqrt(N)
        sh = s + e
        gs.append(np.dot(sh, s)/np.dot(sh, sh))
    print("S/N", S/N, "MC optimal gain %.4f, S/(S+N) %.4f" % (np.mean(gs), S/(S+N)))
# fixed shape with overlap rho: least-squares size = rho*||s||
s = rng.standard_normal(n); u_ = rng.standard_normal(n); u_ -= np.dot(u_, s)/np.dot(s, s)*s; u_ /= np.linalg.norm(u_)
sh = s/np.linalg.norm(s); rho = 0.19; d = rho*sh + np.sqrt(1-rho**2)*u_
gs = np.linspace(0, 2, 20001)*np.linalg.norm(s)
err = [np.sum((g*d - s)**2) for g in gs]
print("fixed shape rho=0.19: best size / true size %.4f" % (gs[int(np.argmin(err))]/np.linalg.norm(s)))
print("== max log-likelihood ratio = rho^2/2")
L = 4096; t = np.arange(L)/1024.
h = np.sin(2*np.pi*(30*t + 20*t**2))*np.exp(-((t-2)/1.2)**2)
H = np.fft.fft(h); f = np.fft.fftfreq(L); H[f < 0] = 0; H[f > 0] *= 2; ha = np.fft.ifft(H)
hc, hs = ha.real, ha.imag
hc /= np.linalg.norm(hc); hs -= np.dot(hs, hc)*hc; hs /= np.linalg.norm(hs)
for trial in range(3):
    A0, ph0 = rng.uniform(5, 15), rng.uniform(0, 2*np.pi)
    x = A0*(np.cos(ph0)*hc + np.sin(ph0)*hs) + rng.standard_normal(L)
    rho2 = np.dot(x, hc)**2 + np.dot(x, hs)**2
    best = -np.inf
    for ph in np.linspace(0, 2*np.pi, 2001):
        tpl = np.cos(ph)*hc + np.sin(ph)*hs
        A = np.dot(x, tpl)          # amplitude maximizing the likelihood for this phase
        lnL = A*np.dot(x, tpl) - A**2/2
        best = max(best, lnL)
    print("trial %d: grid max ln Lambda %.4f, rho^2/2 %.4f" % (trial, best, rho2/2))
print("== binomial spreads")
for p, nn in ((0.20, 1966), (0.16, 1966), (0.416, 1966)):
    print(p, nn, "%.4f" % np.sqrt(p*(1-p)/nn))
print("diff 0.20 vs 0.16:", np.sqrt(0.2*0.8/1966 + 0.16*0.84/1966))
print("== volume factors from SNR fractions")
for r in (0.738, 0.773, 0.828, 0.894, 0.929):
    print(r, "%.3f" % r**-3)
print("== sigma / (mean|err|/0.798)")
for sig, mae in ((0.0029, 0.0023), (0.0155, 0.0166), (0.0229, 0.0363)):
    print(sig, mae, "%.4f" % (sig/(mae/np.sqrt(2/np.pi))))
print("sqrt(2/pi) = %.4f" % np.sqrt(2/np.pi))
print("== chirp-mass chance, uniform prior [0.85, 2.75]")
a, b = 0.85, 2.75
c = b*0.99
print("best fixed guess %.4f at %.4f; random draw approx 0.02*E[Mc]/(b-a) = %.4f" % (c*(1/0.99-1/1.01)/(b-a), c, 0.02*(a+b)/2/(b-a)))
print("== 125 kernels evenly in log over 0.85-2.74: spacing %.4f%%" % (100*(np.exp(np.log(2.74/0.85)/124)-1)))
print("== cycles at 0.85 and 2.7387 by leading order (see late.py)")
