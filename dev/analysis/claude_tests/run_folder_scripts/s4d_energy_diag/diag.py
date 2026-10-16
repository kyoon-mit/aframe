"""CPU diagnostic: does an energy step plus max pooling let S4D learn chirp mass?

Three single-layer models, identical except where marked:
  A  baseline: real S4D output, GELU, mean over time
  B  energy:   real and imaginary S4D outputs squared and added, max over time
  C  B with each state's memory forced to cover the whole window

Train on dumped aframe windows (signal rows only), predict log chirp mass
with MSE, score on the matched-filter benchmark windows.

    python diag.py MODEL TRAIN_H5 EVAL_DIR [--steps N]
"""

import argparse
import math
import time

import h5py
import numpy as np
import torch
import torch.nn as nn

p = argparse.ArgumentParser()
p.add_argument("model", choices=["A", "B", "C", "D", "Dfixed"])
p.add_argument("train_h5")
p.add_argument("eval_dir")
p.add_argument("--steps", type=int, default=3000)
p.add_argument("--batch", type=int, default=32)
p.add_argument("--d_model", type=int, default=32)
p.add_argument("--d_state", type=int, default=64)
p.add_argument("--lr", type=float, default=1e-3)
p.add_argument("--threads", type=int, default=32)
p.add_argument("--max_train", type=int, default=None)
p.add_argument("--name", default=None)
p.add_argument("--no_wandb", action="store_true")
p.add_argument("--n_kernels", type=int, default=32)
p.add_argument("--head", choices=["mlp", "softmax"], default="mlp")
args = p.parse_args()
torch.set_num_threads(args.threads)
torch.manual_seed(0)
L = 8192


class Kernel(nn.Module):
    """ml4gw S4DKernel math, returning the complex kernel."""

    def __init__(self, H, N, dt_min=1e-3, dt_max=1e-1, long_memory=False):
        super().__init__()
        self.log_dt = nn.Parameter(torch.rand(H) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min))
        self.C = nn.Parameter(torch.view_as_real(torch.randn(H, N // 2, dtype=torch.cfloat)))
        self.log_A_real = nn.Parameter(torch.log(0.5 * torch.ones(H, N // 2)))
        self.A_imag = nn.Parameter(math.pi * torch.arange(N // 2).repeat(H, 1).float())
        self.long_memory = long_memory

    def forward(self, L):
        dt = self.log_dt.exp()
        decay = self.log_A_real.exp()
        if self.long_memory:
            # per-sample decay dt * alpha capped at 1/L: memory >= window
            decay = torch.minimum(decay, 1.0 / (L * dt[:, None]))
        A = torch.complex(-decay, self.A_imag)
        dtA = A * dt[:, None]
        C = torch.view_as_complex(self.C) * (torch.exp(dtA) - 1.0) / A
        K = torch.exp(dtA[..., None] * torch.arange(L))
        return 2 * torch.einsum("hn,hnl->hl", C, K)  # complex (H, L)


TSUN = 4.925491e-6  # G * (1 solar mass) / c^3, seconds
FS = 2048.0


class ChirpKernel(nn.Module):
    """J Newtonian chirps, merger at lag 0, two quadratures, unit norm."""

    def __init__(self, J=32, mc_min=0.85, mc_max=2.75, learn=True):
        super().__init__()
        log_mc = torch.linspace(math.log(mc_min), math.log(mc_max), J)
        self.log_mc = nn.Parameter(log_mc, requires_grad=learn)

    def forward(self, L):
        tau = (torch.arange(L) + 1) / FS
        T = TSUN * self.log_mc.exp()[:, None]
        phase = -2 * (tau / (5 * T)) ** (5 / 8)
        f = (1 / (8 * math.pi)) * (5 / tau) ** (3 / 8) * T ** (-5 / 8)
        # smooth band 20 to 800 Hz, so the chirp mass gradient is well behaved
        band = torch.sigmoid((f - 20) / 2) * torch.sigmoid((800 - f) / 20)
        amp = tau ** -0.25 * band
        kc, ks = amp * torch.cos(phase), amp * torch.sin(phase)
        norm = kc.pow(2).sum(-1, keepdim=True).sqrt().clamp_min(1e-12)
        return kc / norm, ks / norm  # each (J, L)


class ChirpModel(nn.Module):
    def __init__(self, learn, n_kernels=32, head="mlp"):
        super().__init__()
        self.kernel = ChirpKernel(J=n_kernels, learn=learn)
        J = self.kernel.log_mc.shape[0]
        self.head_kind = head
        if head == "mlp":
            self.head = nn.Sequential(nn.Linear(J, 64), nn.GELU(), nn.Linear(64, 1))
        else:
            # logits over the kernels' chirp masses: scaled energies plus a learned correction
            self.scale = nn.Parameter(torch.tensor(1.0))
            self.correction = nn.Linear(J, J)
            nn.init.zeros_(self.correction.weight)
            nn.init.zeros_(self.correction.bias)

    def forward(self, x):
        kc, ks = self.kernel(L)
        u_f = torch.fft.rfft(x, n=2 * L)[:, :, None]          # (B, det, 1, F)
        zc = torch.fft.irfft(u_f * torch.fft.rfft(kc, n=2 * L), n=2 * L)[..., :L]
        zs = torch.fft.irfft(u_f * torch.fft.rfft(ks, n=2 * L), n=2 * L)[..., :L]
        energy = (zc ** 2 + zs ** 2).sum(1)                    # (B, J, L)
        feat = torch.log1p(energy.amax(-1))
        if self.head_kind == "mlp":
            return self.head(feat).squeeze(-1)
        z = (feat - feat.mean(-1, keepdim=True)) / feat.std(-1, keepdim=True).clamp_min(1e-6)
        return self.scale * z * 5 + self.correction(z)        # logits (B, J)


class Model(nn.Module):
    def __init__(self, kind, d_model, d_state):
        super().__init__()
        self.kind = kind
        self.inp = nn.Conv1d(2, d_model, 1)
        self.kernel = Kernel(d_model, d_state, long_memory=(kind == "C"))
        self.D = nn.Parameter(torch.randn(d_model))
        self.head = nn.Sequential(nn.Linear(d_model, 64), nn.GELU(), nn.Linear(64, 1))

    def forward(self, x):
        u = self.inp(x)                                       # (B, H, L)
        k = self.kernel(L)                                    # (H, L) complex
        u_f = torch.fft.rfft(u, n=2 * L)
        yr = torch.fft.irfft(u_f * torch.fft.rfft(k.real, n=2 * L), n=2 * L)[..., :L]
        if self.kind == "A":
            y = nn.functional.gelu(yr + u * self.D[:, None])
            feat = y.mean(-1)
        else:
            yi = torch.fft.irfft(u_f * torch.fft.rfft(k.imag, n=2 * L), n=2 * L)[..., :L]
            feat = torch.log1p((yr ** 2 + yi ** 2).amax(-1))
        return self.head(feat).squeeze(-1)


def load(path, limit=None):
    with h5py.File(path) as f:
        mc = f["params/chirp_mass"][:]
        idx = np.where(~np.isnan(mc))[0]
        if limit:
            idx = idx[:limit]
        X = f["X"][np.sort(idx)] if len(idx) < len(mc) else f["X"][:]
        snr = f["params/snr"][:][idx]
        mc = mc[idx]
    return torch.tensor(X), torch.tensor(np.log(mc)), snr, mc


run = None
if not args.no_wandb:
    import wandb
    run = wandb.init(project="CLAUDE-TESTS", name=args.name or f"cpu_diag_{args.model}", config=vars(args))
t0 = time.time()
Xtr, ytr, _, _ = load(args.train_h5, args.max_train)
mu, sd = ytr.mean(), ytr.std()
print(f"train windows {len(Xtr)} loaded in {time.time() - t0:.0f}s", flush=True)
if args.model in ("D", "Dfixed"):
    model = ChirpModel(learn=args.model == "D", n_kernels=args.n_kernels, head=args.head)
else:
    model = Model(args.model, args.d_model, args.d_state)
opt = torch.optim.Adam(model.parameters(), lr=args.lr)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps)
t0 = time.time()
for step in range(1, args.steps + 1):
    i = torch.randint(len(Xtr), (args.batch,))
    if getattr(model, "head_kind", "mlp") == "softmax":
        log_k = model.kernel.log_mc
        spacing = (log_k[1:] - log_k[:-1]).abs().mean().detach()
        target = torch.softmax(-((ytr[i][:, None] - log_k[None].detach()) ** 2) / (2 * (0.5 * spacing) ** 2), -1)
        loss = -(target * torch.log_softmax(model(Xtr[i]), -1)).sum(-1).mean()
    else:
        loss = ((model(Xtr[i]) - (ytr[i] - mu) / sd) ** 2).mean()
    opt.zero_grad()
    loss.backward()
    opt.step()
    sched.step()
    if step % 100 == 0:
        print(f"step {step} loss {loss.item():.4f} {time.time() - t0:.0f}s", flush=True)
        if run:
            run.log({"train/loss": loss.item()}, step=step)

model.eval()
bands = [(4, 6), (6, 8), (8, 10), (10, 12), (12, 16), (16, 25), (25, 50)]
for pop in ("snr4_powerlaw", "snr8_powerlaw"):
    X, _, snr, mc = load(f"{args.eval_dir}/{pop}/windows.h5")
    with torch.no_grad():
        pred = torch.cat([model(X[a:a + 256]) for a in range(0, len(X), 256)])
    if getattr(model, "head_kind", "mlp") == "softmax":
        p_k = torch.softmax(pred, -1)
        est = np.exp((p_k * model.kernel.log_mc[None]).sum(-1).detach().numpy())
    else:
        est = np.exp((pred * sd + mu).numpy())
    err = np.abs(est - mc) / mc
    print(f"== {args.model} on {pop}")
    for lo, hi in bands:
        m = (snr >= lo) & (snr < hi)
        if m.sum():
            if run:
                run.summary.update({f"{pop}/snr{lo}-{hi}/within_{t}pct": float((err[m] < t / 100).mean()) for t in (1, 2, 5, 10)})
            print(f"  SNR {lo}-{hi} n={m.sum()} within 1/2/5/10%: "
                  + " / ".join(f"{(err[m] < t).mean():.0%}" for t in (0.01, 0.02, 0.05, 0.10)))
if args.model in ("D", "Dfixed"):
    mc_k = model.kernel.log_mc.exp().detach()
    print("kernel chirp masses:", " ".join(f"{m:.3f}" for m in mc_k[:64]))
torch.save(model.state_dict(), __file__.replace("diag.py", f"model_{args.model}.pt"))

if run:
    run.finish()
