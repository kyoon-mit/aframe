import os
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[v] = "1"
import numpy as np
from scipy.integrate import quad
GM_SUN = 1.3271244e20; C = 299792458.0; TSUN = GM_SUN / C**3
f_tau = lambda tau, mc: (1/np.pi)*(5/(256*tau))**0.375*(TSUN*mc)**(-0.625)
for mc in (0.85, 1.2, 1.8, 2.75):
    u = np.linspace(np.log(1e-40), np.log(4.0), 2000001); tau = np.exp(u); h = 1e-5*tau
    f0 = f_tau(tau, mc); fp, fm = f_tau(tau-h, mc), f_tau(tau+h, mc)
    fd = (fp-fm)/(2*h); fdd = (fp-2*f0+fm)/h**2
    nl = np.trapz((fdd/6)**(1/3)*tau, u); nt = np.trapz(np.sqrt(fd)*tau, u)
    fT = f_tau(4.0, mc); N = 1.6*4*fT
    print(f"Mc {mc}: quad lin {nl:.3f} closed {24/5*(55/1024)**(1/3)*N**(1/3):.3f} | quad tone {nt:.3f} closed {2*np.sqrt(15)/5*np.sqrt(N):.3f}")
k_lin = quad(lambda w: np.cos(np.pi*w**3/4), 0, 1)[0]
k_tone = abs(quad(lambda v: np.cos(np.pi*v**2), -0.5, 0.5)[0] + 1j*quad(lambda v: np.sin(np.pi*v**2), -0.5, 0.5)[0])
print(f"kept inside the in-step time: linear chirp {k_lin:.4f}, tone {k_tone:.4f}")
# Gaussian rule: sigma = T/4 -> |s| <= 2 sigma = T/2 is the 1/8-cycle bound
print("ratio T_lin/T_step at 1.2 Msun, 100 Hz from (18 f T_step / 11)^(1/3):",
      (18*100*0.2431/11)**(1/3))
