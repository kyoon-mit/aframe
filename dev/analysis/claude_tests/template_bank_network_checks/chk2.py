import math, sys, numpy as np, torch, h5py
torch.set_num_threads(8)
sys.path.insert(0, '/home/kyoon/SSM-BNS/aframe/projects/train')
from train.experimental.template_bank.arch import TemplateBankNet
from ml4gw.spectral import truncate_inverse_power_spectrum
BANK = '/home/kyoon/SSM-BNS/aframe/dev/runs/claude_tests/template_bank/bank_mc0.5pct_spin3_q0.8.npz'
net = TemplateBankNet(BANK, highpass=20.0)
bank = np.load(BANK)
# full-rate inverse ASD exactly as in __init__
psd = torch.tensor(bank['psd'], dtype=torch.float64)[None]
psd = torch.nn.functional.interpolate(psd, size=(net.n_full // 2 + 1,), mode='linear')
inv_asd = truncate_inverse_power_spectrum(psd, 1.0, 2048.0, 20.0) ** -0.5
inv_asd = torch.nan_to_num(inv_asd, posinf=0.0)[0]   # (I, 4609)
q = torch.fft.irfft(inv_asd, n=net.n_full)            # (I, 9216) zero-phase FIR, support +-1024
wf = h5py.File('/home/kyoon/SSM-BNS/DATA/aframe/train/end_o3_ratesandpops_bns_uniform_chirp.hdf5', 'r')

def whiten_long(h):
    """h: (L,) float64 -> (I, L) whitened by q (linear conv, zero-phase)."""
    L = len(h)
    nfft = L + net.n_full
    qq = torch.zeros(2, nfft, dtype=torch.float64)
    half = net.n_full // 2
    qq[:, :half] = q[:, :half]
    qq[:, -half:] = q[:, -half:]
    H = torch.fft.rfft(torch.tensor(h), n=nfft)
    return torch.fft.irfft(H[None] * torch.fft.rfft(qq, n=nfft), n=nfft)[:, :L]

def internals(self, X):
    spectrum = torch.fft.rfft(X.float(), n=self.n_full)[..., : self.f]
    prod = spectrum[:, :, None] * self.filters[None]
    z = torch.fft.ifft(prod, n=self.n)[..., self.lags]
    rho2 = z.real**2 + z.imag**2
    return prod, z, rho2

def band_stats(prod_bj, lag, band_bj, p):
    k = torch.arange(net.f)
    lagk = (k * lag) % net.n
    c = prod_bj.to(torch.complex128) * torch.exp(2j * math.pi * lagk.double() / net.n) / net.n
    zl = torch.stack([(c * (band_bj == l)).sum() for l in range(p)])
    z = zl.sum()
    lam = p * ((zl - z / p).abs() ** 2).sum()
    return z, zl, lam

p = net.n_bands
rng = np.random.default_rng(0)
for j in [10, 150, 351, 520, 690]:
    row = int(bank['rows'][j])
    hp = wf['waveforms/plus'][row]; hc = wf['waveforms/cross'][row]
    merger = int(np.argmax(hp**2 + hc**2))
    for mix in ['plus', 'rot']:
        if mix == 'plus':
            h = hp
        else:
            a = 1.3
            h = np.cos(a) * hp + np.sin(a) * hc
        w = whiten_long(h)    # (I, L)
        out = []
        for md in [7168, 7300, 7500, 7680, 7900, 8100, 8191]:
            s0 = merger - md
            x = w[:, s0:s0 + 8192].float()[None]   # (1, 2, 8192)
            prod, z, rho2 = internals(net, x)
            # true lag index for this template: data merger at md -> lag (md - 7680)/2 at 1024 Hz
            r = rho2[0, 0, j]
            t = int(r.argmax())
            lag = int(net.lags[t])
            zz, zl, lam = band_stats(prod[0, 0, j], lag, net.band[0, j], p)
            out.append(f'md={md}: lagsamp={(lag if lag < net.n//2 else lag-net.n)} best_j={int(rho2[0,0].amax(-1).argmax())} lam/|z|^2={(lam/zz.abs()**2).item():.4f}')
        print(f'j={j} mc={bank["chirp_mass"][j]:.3f} {mix}:', ' | '.join(out))
