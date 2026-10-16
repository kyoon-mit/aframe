import math, sys, numpy as np, torch
torch.set_num_threads(8)
sys.path.insert(0, '/home/kyoon/SSM-BNS/aframe/projects/train')
from train.experimental.template_bank.arch import TemplateBankNet
BANK = '/home/kyoon/SSM-BNS/aframe/dev/runs/claude_tests/template_bank/bank_mc0.5pct_spin3_q0.8.npz'
net = TemplateBankNet(BANK, highpass=20.0)
print('n', net.n, 'n_full', net.n_full, 'f', net.f, 'max_lag', net.max_lag, 'coinc', net.coincidence, 'T', len(net.lags))

def internals(self, X):
    spectrum = torch.fft.rfft(X.float(), n=self.n_full)[..., : self.f]
    prod = spectrum[:, :, None] * self.filters[None]
    zfull = torch.fft.ifft(prod, n=self.n)
    z = zfull[..., self.lags]
    rho2 = z.real**2 + z.imag**2
    B, I, J, T = rho2.shape
    other, where = torch.nn.functional.max_pool1d(rho2[:, 1].reshape(-1, 1, T), 2 * self.coincidence + 1, stride=1,
                                                  padding=self.coincidence, return_indices=True)
    netw = rho2[:, 0] + other.reshape(B, J, T)
    best = netw.reshape(B, -1).argmax(-1)
    rows = torch.arange(B)
    j, t = best // T, best % T
    t_other = where.reshape(B, J, T)[rows, j, t]
    lag = torch.stack([self.lags[t], self.lags[t_other]], 1)
    k = torch.arange(self.f)
    phase = torch.exp(2j * math.pi * k * lag[..., None] / self.n)
    contrib = prod[rows, :, j] * phase / self.n
    band = self.band[:, j].transpose(0, 1)
    zb = torch.zeros(B, I, self.n_bands, dtype=torch.complex128)
    zb_re = torch.zeros(B, I, self.n_bands).scatter_add_(-1, band, contrib.real)
    zb_im = torch.zeros(B, I, self.n_bands).scatter_add_(-1, band, contrib.imag)
    zb = torch.complex(zb_re, zb_im)
    return dict(prod=prod, z=z, rho2=rho2, other=other.reshape(B, J, T), where=where.reshape(B, J, T), j=j, t=t,
                t_other=t_other, lag=lag, zb=zb, band=band, netw=netw)

torch.manual_seed(0)
B = 8
X = torch.randn(B, 2, 8192)
d = internals(net, X)
rows = torch.arange(B)
# 1. max_pool indices semantic
g = torch.gather(d['rho2'][:, 1], -1, d['where'])
print('maxpool idx check max abs diff:', (g - d['other']).abs().max().item())
w = d['where']
T = d['rho2'].shape[-1]
print('where range', w.min().item(), w.max().item(), 'offset range', (w - torch.arange(T)).min().item(), (w - torch.arange(T)).max().item())
# 2. total vs z at lag
tot = d['zb'].sum(-1)
zH = d['z'][rows, 0, d['j'], d['t']]
zL = d['z'][rows, 1, d['j'], d['t_other']]
print('rel diff H:', ((tot[:, 0] - zH).abs() / zH.abs()).max().item(), ' L:', ((tot[:, 1] - zL).abs() / zL.abs()).max().item())
# 3. exact per band in float64 with integer-reduced phase
prod64 = d['prod'][rows, :, d['j']].to(torch.complex128)  # (B, I, F)
k = torch.arange(net.f)
lagk = (k * d['lag'][..., None]) % net.n
ph64 = torch.exp(2j * math.pi * lagk.double() / net.n)
c64 = prod64 * ph64 / net.n
zb64 = torch.zeros(B, 2, net.n_bands, dtype=torch.complex128)
for l in range(net.n_bands):
    zb64[..., l] = (c64 * (d['band'] == l)).sum(-1)
print('band sums: max |zb - exact| / max|exact|:', ((d['zb'] - zb64).abs().max() / zb64.abs().max()).item())
# masked-ifft independent per-band check for one row
b = 0
for i in range(2):
    full = torch.zeros(net.f, dtype=torch.complex128)
    for l in [0, 7, 15]:
        m = (d['band'][b, i] == l)
        zz = torch.fft.ifft(prod64[b, i] * m, n=net.n)[d['lag'][b, i]]
        print(f'det{i} band{l}: masked ifft {zz.item():.5f}  code {d["zb"][b, i, l].item():.5f}')
# 4. replicate == module output
new, plain, peak, feats = net.newsnr2(X, features=True)
zb = d['zb']; p = net.n_bands
total = zb.sum(-1, keepdim=True)
chi2 = p * ((zb - total / p).abs() ** 2).sum(-1)
red = chi2 / (2 * p - 2)
print('replica reduced vs module:', (red - feats[:, 2:4]).abs().max().item())
print('rho2_det sum vs plain rel:', ((feats[:, 0] + feats[:, 1] - plain).abs() / plain).max().item())
# chi2 with exact sums
tot64 = zb64.sum(-1, keepdim=True)
red64 = p * ((zb64 - tot64 / p).abs() ** 2).sum(-1) / (2 * p - 2)
print('reduced chi2 code vs exact max abs diff:', (red - red64).abs().max().item())
print('lags used:', d['lag'].tolist())
