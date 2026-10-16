import numpy as np
import torch

torch.set_num_threads(8)
from train.experimental.template_bank.arch import TemplateBankNet

BANK = "/home/kyoon/SSM-BNS/aframe/dev/runs/claude_tests/template_bank/bank_mc0.5pct_spin3_q0.8.npz"
net = TemplateBankNet(BANK)
print("n", net.n, "n_full", net.n_full, "f", net.f, "max_lag", net.max_lag, "coinc", net.coincidence)
torch.manual_seed(0)
acc = None
chis = []
nb = 0
for it in range(6):
    x = torch.randn(8, 2, 8192)
    r = net.snr2_series(x)  # (B, I, J, T)
    m = r.mean((0, 2)).double()  # (I, T)
    acc = m if acc is None else acc + m
    nb += 1
    new, plain, peak, feats = net.newsnr2(x, features=True)
    chis.append(feats[:, 2:4])
acc /= nb
T = acc.shape[-1]
lagv = torch.arange(T) - net.max_lag
for idx in [0, 1, 50, 150, 250, 300, 307, 357, 457, 557, 600, 613, 614]:
    print(f"lag {lagv[idx].item():5d} ({lagv[idx].item()*2/2048:+.3f}s): E|z|^2 H {acc[0, idx]:.4f} L {acc[1, idx]:.4f}")
chis = torch.cat(chis)
print("reduced chi2 at loudest template in noise: mean", chis.mean(0).tolist(), "median", chis.median(0).values.tolist())
