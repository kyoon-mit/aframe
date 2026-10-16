import math, sys, numpy as np, torch, h5py
torch.set_num_threads(8)
sys.path.insert(0, '/home/kyoon/SSM-BNS/aframe/projects/train')
from train.experimental.template_bank.arch import TemplateBankNet
BANK = '/home/kyoon/SSM-BNS/aframe/dev/runs/claude_tests/template_bank/bank_mc0.5pct_spin3_q0.8.npz'
net = TemplateBankNet(BANK, highpass=20.0)
f = h5py.File('/home/kyoon/SSM-BNS/aframe/dev/runs/claude_tests/s4d_energy_diag/train_windows_snr8_powerlaw.h5', 'r')
snr_all = f['params/snr'][:]
sig = np.where(~np.isnan(snr_all))[0][::150][:320]
Xc = torch.tensor(f['X_clean'][sig]); snr = snr_all[sig]
p = net.n_bands
outs = [net.newsnr2(Xc[c:c+16], features=True) for c in range(0, len(Xc), 16)]
plain = torch.cat([o[1] for o in outs]).numpy(); feats = torch.cat([o[3] for o in outs]).numpy()
lam = feats[:, 2:4] * (2 * p - 2) / feats[:, 0:2]
print('clean signals', len(sig), 'plain/snr^2 median', np.median(plain / snr**2).round(4))
print('lambda/|z|^2 quantiles 10/50/90 (H,L):', np.percentile(lam, [10, 50, 90], axis=0).round(4).tolist())
pk = np.argmax((Xc.numpy()**2).sum(1), -1)
for lo, hi in ((7168, 7400), (7400, 7900), (7900, 8100), (8100, 8192)):
    m = (pk >= lo) & (pk < hi)
    if m.any():
        print(f'merger sample {lo}-{hi}: n={m.sum()} lambda/|z|^2 median {np.median(lam[m], 0).round(4).tolist()}')
