import math, sys, numpy as np, torch
torch.set_num_threads(8)
sys.path.insert(0, '/home/kyoon/SSM-BNS/aframe/projects/train')
from train.experimental.template_bank.arch import TemplateBankNet
BANK = '/home/kyoon/SSM-BNS/aframe/dev/runs/claude_tests/template_bank/bank_mc0.5pct_spin3_q0.8.npz'
net = TemplateBankNet(BANK, highpass=20.0)
torch.manual_seed(1)
reds, rho2s, plains, news, lagoff = [], [], [], [], []
for c in range(16):
    X = torch.randn(16, 2, 8192)
    new, plain, peak, feats = net.newsnr2(X, features=True)
    reds.append(feats[:, 2:4]); rho2s.append(feats[:, 0:2]); plains.append(plain); news.append(new); lagoff.append(feats[:, 4])
reds = torch.cat(reds); rho2s = torch.cat(rho2s); plains = torch.cat(plains); news = torch.cat(news)
print('noise windows', len(reds))
print('reduced chi2 mean H, L:', reds.mean(0).tolist(), ' median', reds.median(0).values.tolist())
print('rho2 per det mean', rho2s.mean(0).tolist(), 'plain mean', plains.mean().item(), 'new/plain median', (news/plains).median().item())
# unbiased: fixed template, fixed lag 0: per-band variance and E|z|^2
X = torch.randn(64, 2, 8192)
spectrum = torch.fft.rfft(X.float(), n=net.n_full)[..., : net.f]
j = torch.tensor([10, 351, 690])
prod = spectrum[:, :, None] * net.filters[None, :, j]           # (B, I, 3, F)
band = net.band[:, j]                                            # (I, 3, F)
for lagsamp in [0, -256, 255]:
    lag = lagsamp % net.n
    k = torch.arange(net.f)
    ph = torch.exp(2j * math.pi * ((k * lag) % net.n).double() / net.n)
    c = prod.to(torch.complex128) * ph / net.n
    zl = torch.stack([(c * (band == l)).sum(-1) for l in range(net.n_bands)], -1)  # (B, I, 3, p)
    z = zl.sum(-1)
    p = net.n_bands
    red = p * ((zl - z[..., None] / p).abs() ** 2).sum(-1) / (2 * p - 2)
    print(f'lag {lagsamp}: E|z|^2', (z.abs()**2).mean((0, 1)).tolist(), ' E red', red.mean((0, 1)).tolist(),
          ' band var min/max', (zl.abs()**2).mean((0, 1)).min().item()*p/2, (zl.abs()**2).mean((0, 1)).max().item()*p/2)
