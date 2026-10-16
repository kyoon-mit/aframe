import math, sys, numpy as np, torch, h5py
torch.set_num_threads(8)
sys.path.insert(0, '/home/kyoon/SSM-BNS/aframe/projects/train')
from train.experimental.template_bank.arch import TemplateBankNet
BANK = '/home/kyoon/SSM-BNS/aframe/dev/runs/claude_tests/template_bank/bank_mc0.5pct_spin3_q0.8.npz'
net = TemplateBankNet(BANK, highpass=20.0)
f = h5py.File('/home/kyoon/SSM-BNS/aframe/dev/runs/claude_tests/s4d_energy_diag/train_windows_snr8_powerlaw.h5', 'r')
snr_all = f['params/snr'][:]
noise = np.where(np.isnan(snr_all))[0]
print('noise rows', len(noise))
idx = noise[:256]
X = torch.tensor(f['X'][idx])
outs = [net.newsnr2(X[c:c+16], features=True) for c in range(0, len(X), 16)]
new = torch.cat([o[0] for o in outs]); plain = torch.cat([o[1] for o in outs]); feats = torch.cat([o[3] for o in outs])
print('real noise: plain mean', plain.mean().item(), 'median', plain.median().item(), ' reduced mean H,L', feats[:, 2:4].mean(0).tolist(), 'median', feats[:, 2:4].median(0).values.tolist())
print('new/plain median', (new/plain).median().item(), ' frac new/plain<0.8', ((new/plain) < 0.8).float().mean().item())
# in-band variance at fixed template & lag 0 per window
spectrum = torch.fft.rfft(X.float(), n=net.n_full)[..., :net.f]
for j in (10, 351, 690):
    z = torch.fft.ifft(spectrum * net.filters[None, :, j], n=net.n)[..., net.lags]
    print(f'j={j} E|z|^2 over lags (H,L):', (z.abs()**2).mean((0, 2)).tolist())
fr = torch.fft.rfftfreq(8192, 1/2048)
P = (torch.fft.rfft(X.float()).abs()**2).mean((0)) * 2 / 8192 / 2048 * 2048
for lo, hi in ((0, 15), (15, 25), (25, 60), (60, 512), (512, 1024)):
    m = (fr >= lo) & (fr < hi)
    print(f'{lo}-{hi} Hz mean two-sided PSD*fs (unit=1 for unit white):', (P[:, m].mean(-1) / 2).tolist())
