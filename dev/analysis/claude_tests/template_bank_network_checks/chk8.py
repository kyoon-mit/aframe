import math, sys, numpy as np, torch, h5py
torch.set_num_threads(8)
sys.path.insert(0, '/home/kyoon/SSM-BNS/aframe/projects/train')
from train.experimental.template_bank.arch import TemplateBankNet
from ml4gw.spectral import truncate_inverse_power_spectrum
BANK = '/home/kyoon/SSM-BNS/aframe/dev/runs/claude_tests/template_bank/bank_mc0.5pct_spin3_q0.8.npz'
net = TemplateBankNet(BANK, highpass=20.0)
bank = np.load(BANK)
psd = torch.tensor(bank['psd'], dtype=torch.float64)[None]
psd = torch.nn.functional.interpolate(psd, size=(net.n_full // 2 + 1,), mode='linear')
inv_asd = torch.nan_to_num(truncate_inverse_power_spectrum(psd, 1.0, 2048.0, 20.0) ** -0.5, posinf=0.0)[0]
q = torch.fft.irfft(inv_asd, n=net.n_full)
wf = h5py.File('/home/kyoon/SSM-BNS/DATA/aframe/train/end_o3_ratesandpops_bns_uniform_chirp.hdf5', 'r')
def whiten_long(h):
    L = len(h); nfft = L + net.n_full; half = net.n_full // 2
    qq = torch.zeros(2, nfft, dtype=torch.float64); qq[:, :half] = q[:, :half]; qq[:, -half:] = q[:, -half:]
    return torch.fft.irfft(torch.fft.rfft(torch.tensor(h), n=nfft)[None] * torch.fft.rfft(qq, n=nfft), n=nfft)[:, :L]
rng = np.random.default_rng(1)
rows = np.sort(rng.choice(200000, 40, replace=False))
P = wf['parameters']
m1, m2 = P['mass_1'][:], P['mass_2'][:]
mc = (m1*m2)**0.6/(m1+m2)**0.2
p = net.n_bands
lams, ffs = [], []
for r in rows:
    if not (0.86 < mc[r] < 2.7):
        continue
    hp = wf['waveforms/plus'][r]; hc = wf['waveforms/cross'][r]
    merger = int(np.argmax(hp**2 + hc**2))
    a = rng.uniform(0, 2*np.pi)
    w = whiten_long(np.cos(a)*hp + np.sin(a)*hc)
    x = w[:, merger-7680: merger-7680+8192].float()[None]
    x = x / x[0, 0].norm() * 20   # scale H to |x|=20
    new, plain, peak, feats = net.newsnr2(x, features=True)
    lam = (feats[0, 2] * (2*p-2) / feats[0, 0]).item()
    # self-match |z|^2 upper bound for H: band-limited norm^2 of x * 2
    X = torch.fft.rfft(x[0, 0].double(), n=net.n_full)[:net.f]
    par = torch.full((net.f,), 2.0, dtype=torch.float64); par[0] = 1
    selfnorm2 = 2 * ((X.abs()**2) * par).sum() / net.n_full
    ffs.append((feats[0, 0].item() / selfnorm2.item()) ** 0.5)
    lams.append(lam)
lams = np.array(lams); ffs = np.array(ffs)
print('n', len(lams), 'lambda/|z|^2 quantiles 10/50/90', np.percentile(lams, [10, 50, 90]).round(4), 'fitting factor quantiles', np.percentile(ffs, [10, 50, 90]).round(4))
