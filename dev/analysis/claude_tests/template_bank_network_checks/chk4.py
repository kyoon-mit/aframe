import math, sys, numpy as np, torch
torch.set_num_threads(8)
sys.path.insert(0, '/home/kyoon/SSM-BNS/aframe/projects/train')
from train.experimental.template_bank.arch import TemplateBankNet
BANK = '/home/kyoon/SSM-BNS/aframe/dev/runs/claude_tests/template_bank/bank_mc0.5pct_spin3_q0.8.npz'
net = TemplateBankNet(BANK, highpass=20.0)
N, n, p = net.n_full, net.n, net.n_bands
F = net.filters.to(torch.complex128)   # (I, J, f)
for j in [0, 10, 150, 351, 520, 690, 702]:
    for i in range(2):
        res = []
        for lagsamp in [-256, -128, 0, 128, 255]:
            md = 2 * lagsamp
            # G_l(u) = (1/n) sum_{k in l} F_k e^{-2 pi i k u / N}
            G = []
            for l in range(p):
                Fl = torch.zeros(N, dtype=torch.complex128)
                m = net.band[i, j] == l
                Fl[: net.f] = F[i, j] * m
                G.append(torch.fft.fft(Fl) / n)
            G = torch.stack(G)    # (p, N)
            t = torch.arange(8192)
            idx = (t - md) % N
            var_l = (G[:, idx].abs() ** 2).sum(-1)          # per-band variance inside the window
            Gt = G.sum(0)
            var = (Gt[idx].abs() ** 2).sum()
            # covariance matrix of band sums inside the window
            C = G[:, idx] @ G[:, idx].conj().T
            # exact E[chi2] = p * E sum|z_l - z/p|^2 = p*(sum var_l - var/p) ... general form with covariances
            P = torch.eye(p, dtype=torch.complex128) - torch.full((p, p), 1.0 / p, dtype=torch.complex128)
            Echi2 = p * torch.trace(P @ C @ P.conj().T).real
            res.append(f'lag{lagsamp}: E|z|^2={var.item():.3f} Ered={Echi2.item()/(2*p-2):.3f} bandvar*p/2 min={var_l.min().item()*p/2:.3f} max={var_l.max().item()*p/2:.3f}')
        print(f'j={j} det{i}:', ' | '.join(res))
