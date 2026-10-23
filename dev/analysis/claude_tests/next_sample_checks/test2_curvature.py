"""Network log p(n + e h) - log p(n) for template scales e, against the
unit-Gaussian value e h.n - e^2 |h|^2 / 2; and for a white perturbation."""
import sys, numpy as np, torch, yaml
import lightning.pytorch as pl
from train.experimental.next_sample.arch import CausalS4Noise
from train.experimental.next_sample.task import mixture_log_prob
from train.experimental.slic.dump_windows import build_data_module
N=8192; dev="cuda"
config, ckpt = sys.argv[1:3]
torch.set_num_threads(16)
pl.seed_everything(yaml.safe_load(open(config))["seed_everything"])
dm = build_data_module(config, "test"); dm.inverter.prob = 0.0; dm.reverser.prob = 0.0
net = CausalS4Noise(3, 64, 32, 4, 0.0, 1e-4, 10.0)
sd = torch.load(ckpt, map_location="cpu", weights_only=False)["state_dict"]
net.load_state_dict({k.removeprefix("model."): v for k, v in sd.items()}); net = net.to(dev).eval()
def lp(x):
    flat = x.reshape(-1, 1, N).float()
    with torch.no_grad():
        m, s, w = net(flat)
        return mixture_log_prob(flat[:, 0], m, s, w).double().sum(-1).reshape(len(x), 2)
H, Nn = [], []
for batch in dm.test_dataloader():
    batch = dm.on_before_batch_transfer(batch, 0)
    X, Xc, y, p = dm.on_after_batch_transfer(batch, 0)
    H.append(Xc); Nn.append(X - Xc)
    if sum(len(h) for h in H) >= 512: break
h = torch.cat(H).to(dev).double(); n = torch.cat(Nn).to(dev).double()
keep = (n.abs().amax(-1) <= 8).all(1); h, n = h[keep], n[keep]
g = torch.Generator(device=dev).manual_seed(0)
white = torch.randn(h.shape, generator=g, device=dev, dtype=torch.float64)
white *= h.square().sum(-1, keepdim=True).sqrt() / white.square().sum(-1, keepdim=True).sqrt()
base = lp(n)
rho2 = h.square().sum(-1)
eps = [-2, -1, -0.5, -0.25, 0.25, 0.5, 1, 2]
for name, v in [("template", h), ("white, same norm", white)]:
    lin = (v * n).sum(-1)
    R = torch.stack([lp(n + e * v) - base for e in eps], -1)  # (B, 2, E)
    E = torch.tensor(eps, device=dev, dtype=torch.float64)
    # least squares fit R = a e + b e^2 per window and detector
    A = torch.stack([E, E**2], 1)
    coef = torch.linalg.lstsq(A, R.reshape(-1, len(eps)).T).solution.T.reshape(*R.shape[:2], 2)
    a, b = coef[..., 0], coef[..., 1]
    print(f"{name}: {keep.sum().item()} clean windows")
    for i, ifo in enumerate(["H1", "L1"]):
        ratio = (-2 * b[:, i] / rho2[:, i]).cpu().numpy()
        za = ((a[:, i] - lin[:, i]) / rho2[:, i].sqrt()).cpu().numpy()
        print(f"  {ifo}: curvature / Gaussian's: median {np.median(ratio):.3f} q10 {np.quantile(ratio,.1):.3f} q90 {np.quantile(ratio,.9):.3f};"
              f" (slope - h.n)/|h|: std {za.std():.2f}; slope/|h| std {(a[:, i]/rho2[:, i].sqrt()).std().item():.2f} (Gaussian 1)")
