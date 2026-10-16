import math, sys, numpy as np, torch, h5py
torch.set_num_threads(8)
sys.path.insert(0, '/home/kyoon/SSM-BNS/aframe/projects/train')
from train.experimental.template_bank.arch import TemplateBankNet
BANK = '/home/kyoon/SSM-BNS/aframe/dev/runs/claude_tests/template_bank/bank_mc0.5pct_spin3_q0.8.npz'
net = TemplateBankNet(BANK, highpass=20.0)
f = h5py.File('/home/kyoon/SSM-BNS/aframe/dev/runs/claude_tests/s4d_energy_diag/train_windows_snr8_powerlaw.h5', 'r')
idx = np.arange(0, 50400, 197)[:256]
Xc = torch.tensor(f['X_clean'][idx]); X = torch.tensor(f['X'][idx]); snr = f['params/snr'][idx]
print('X std', X.std().item(), 'Xc std', Xc.std().item())
p = net.n_bands
res = {}
for name, data in (('clean', Xc), ('noisy', X)):
    outs = [net.newsnr2(data[c:c+16], features=True) for c in range(0, len(data), 16)]
    new = torch.cat([o[0] for o in outs]); plain = torch.cat([o[1] for o in outs]); feats = torch.cat([o[3] for o in outs])
    res[name] = (new, plain, feats)
new, plain, feats = res['clean']
lamfrac = (feats[:, 2:4] * (2 * p - 2) / feats[:, 0:2]).numpy()
print('clean: plain/snr^2 median', np.median(plain.numpy() / snr**2))
print('clean: lambda/|z|^2 per det quantiles 10/50/90:', np.percentile(lamfrac, [10, 50, 90], axis=0).round(4).tolist())
newn, plainn, featsn = res['noisy']
r = (newn / plainn).numpy()
for lo, hi in ((8, 12), (12, 20), (20, 40), (40, 200)):
    m = (snr >= lo) & (snr < hi)
    if m.any():
        print(f'noisy snr {lo}-{hi}: n={m.sum()} new/plain median {np.median(r[m]):.3f}  reduced median H,L {np.median(featsn[m, 2].numpy()):.2f},{np.median(featsn[m, 3].numpy()):.2f}')
