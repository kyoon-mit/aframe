import math, sys, numpy as np, torch, h5py
torch.set_num_threads(8)
sys.path.insert(0, '/home/kyoon/SSM-BNS/aframe/projects/train')
from train.experimental.template_bank.arch import TemplateBankNet
from ml4gw.spectral import truncate_inverse_power_spectrum
BANK = '/home/kyoon/SSM-BNS/aframe/dev/runs/claude_tests/template_bank/bank_mc0.5pct_spin3_q0.8.npz'
net = TemplateBankNet(BANK, highpass=20.0)
bank = np.load(BANK)
f = h5py.File('/home/kyoon/SSM-BNS/aframe/dev/runs/claude_tests/s4d_energy_diag/train_windows_snr8_powerlaw.h5', 'r')
snr_all = f['params/snr'][:]
sig = np.where(~np.isnan(snr_all))[0][::150][:48]
Xc = torch.tensor(f['X_clean'][sig]); psds = torch.tensor(f['psds'][sig]); snr = snr_all[sig]
p, n, N, F = net.n_bands, net.n, net.n_full, net.f
templates = torch.tensor(bank['templates'], dtype=torch.float64)
spectra = torch.fft.rfft(templates, n=N)[:, :F]
analytic = torch.full((F,), 2.0, dtype=torch.float64); analytic[0] = analytic[-1] = 1.0
parseval = torch.full((F,), 2.0, dtype=torch.float64); parseval[0] = 1.0

def lam_for(filt, band, spec):
    # filt, band: (F,), spec (F,) data spectrum; maximize |z| over lags
    prod = spec * filt
    zt = torch.fft.ifft(prod, n=n)[net.lags]
    t = int((zt.abs()).argmax()); lag = int(net.lags[t])
    k = torch.arange(F)
    c = prod * torch.exp(2j * math.pi * ((k * lag) % n).double() / n) / n
    zl = torch.stack([(c * (band == l)).sum() for l in range(p)])
    z = zl.sum()
    return (p * ((zl - z / p).abs() ** 2).sum() / z.abs() ** 2).item(), (z.abs() ** 2).item()

new, plain, peak, feats = net.newsnr2(Xc, features=True)
j = peak.argmax(-1)
rows = []
for b in range(len(sig)):
    spec = torch.fft.rfft(Xc[b].double(), n=N)[:, :F]
    psd = torch.nn.functional.interpolate(psds[b][None], size=(N // 2 + 1,), mode='linear')
    inv = torch.nan_to_num(truncate_inverse_power_spectrum(psd, 1.0, 2048.0, 20.0) ** -0.5, posinf=0.0)[0, :, :F]
    out = []
    for i in range(2):
        # average-PSD filter (as in the net)
        fa = net.filters[i, j[b]].to(torch.complex128); ba = net.band[i, j[b]]
        la, za = lam_for(fa, ba, spec[i])
        # own-PSD filter
        w = spectra[j[b]] * inv[i]
        sigma = ((w.abs() ** 2) * parseval).sum().div(N).sqrt()
        fo = w.conj() / sigma * analytic / net.decimate
        pw = fo.abs() ** 2
        bo = ((pw.cumsum(-1) / pw.sum()) * p).long().clamp(max=p - 1)
        lo, zo = lam_for(fo, bo, spec[i])
        out.append((la, lo, za, zo))
    rows.append(out)
r = np.array(rows)   # (B, I, 4)
print('lambda/|z|^2 avg-PSD filter: median', np.median(r[..., 0]).round(4), ' own-PSD filter: median', np.median(r[..., 1]).round(4))
print('|z|^2 own/avg median', np.median(r[..., 3] / r[..., 2]).round(4))
print('quantiles avg', np.percentile(r[..., 0], [10, 50, 90]).round(4), 'own', np.percentile(r[..., 1], [10, 50, 90]).round(4))
