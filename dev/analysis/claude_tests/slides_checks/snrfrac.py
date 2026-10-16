import h5py, numpy as np
f = h5py.File('/home/kyoon/SSM-BNS/aframe/dev/runs/claude_tests/s4d_energy_diag/train_windows_snr8_powerlaw.h5', 'r')
psd = f['psds'][:2000].mean(axis=(0, 1))  # (2049,)
mc_all = f['params/chirp_mass'][:]
snr_all = f['params/snr'][:]
print('chirp mass in cache: min %.3f 1%% %.3f median %.3f 99%% %.3f max %.3f' % (mc_all.min(), *np.percentile(mc_all, [1, 50, 99]), mc_all.max()))
freqs = np.linspace(0, 1024, psd.size)
print('psd bins', psd.size, 'df', freqs[1] - freqs[0], 'psd@20,30,50,100,200,500,1000:',
      [float('%.3g' % np.sqrt(np.interp(x, freqs, psd))) for x in (20, 30, 50, 100, 200, 500, 1000)])
G_c3 = 4.925490948e-6
def f_of_tau(tau, mc):
    return (1 / np.pi) * (5 / 256 / tau) ** 0.375 * (mc * G_c3) ** (-0.625)
ff = np.linspace(20, 1000, 200001)
S = np.interp(ff, freqs, psd)
integrand = ff ** (-7 / 3) / S
cum = np.concatenate([[0], np.cumsum(0.5 * (integrand[1:] + integrand[:-1]) * np.diff(ff))])
def band(f1, f2=1000.0):
    f1 = max(f1, 20.0)
    return np.interp(f2, ff, cum) - np.interp(f1, ff, cum)
print('Mc | f(30s) | f(3.75s) | f(4s) | rho(3.75s)/rho(30s) | rho(4s)/rho(30s) | rho(3.75s)/rho(20Hz) | rho(4s)/rho(20Hz) | rho(30s)/rho(20Hz) | vol factor vs 20Hz (3.75s)')
for mc in (0.85, 1.0, 1.2, 1.5, 2.0, 2.5, 2.75):
    f30, f375, f4 = f_of_tau(30, mc), f_of_tau(3.75, mc), f_of_tau(4.0, mc)
    full = band(20); r30 = band(f30)
    a = np.sqrt(band(f375) / r30); b = np.sqrt(band(f4) / r30)
    c = np.sqrt(band(f375) / full); d = np.sqrt(band(f4) / full); e = np.sqrt(r30 / full)
    print('%.2f | %.1f | %.1f | %.1f | %.3f | %.3f | %.3f | %.3f | %.3f | %.2f' % (mc, f30, f375, f4, a, b, c, d, e, c ** -3))
