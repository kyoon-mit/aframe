import sys, math, torch, numpy as np, lightning.pytorch as pl
from types import SimpleNamespace
from train.experimental.slic.dump_windows import build_data_module
import train.experimental.raw_noise.regression_data as rd
from train.experimental.next_sample.task import mixture_log_prob

pl.seed_everything(20260910)
dm = build_data_module(sys.argv[1], "val")
dev = torch.device("cuda")
dm.trainer.lightning_module = SimpleNamespace(device=dev)
dm.transforms_to_device()
calls = []
pe = rd.prediction_errors
def rec(*a, **k):
    out = pe(*a, **k); calls.append(out[0]); return out
rd.prediction_errors = rec
def mv(b):
    if torch.is_tensor(b): return b.to(dev)
    if isinstance(b, dict): return {k: mv(v) for k, v in b.items()}
    return [mv(v) for v in b]
K = slice(-10240 + 1024, -1024)
R = {k: [] for k in "mean_rms err_rms spread peak nll gauss_fit gauss_unit eaten snr quiet_nll quiet_fit".split()}
for i, batch in enumerate(dm.val_dataloader()):
    if i == 8: break
    calls.clear()
    X, Xc, y, params = dm.on_after_batch_transfer(mv(dm.on_before_batch_transfer(batch, 0)), 0)
    ctx = int(dm.hparams.context * 2048)
    err, cerr = calls[0][..., -ctx:], calls[1][..., -ctx:]
    noise = err - cerr
    m1, s1 = dm.remove_noise(err)
    m0, s0 = dm.remove_noise(noise)
    n, m, s = noise[..., K], m0[..., K], s0[..., K]
    R["mean_rms"].append(m.square().mean(-1).sqrt().flatten().cpu())
    R["err_rms"].append(n.square().mean(-1).sqrt().flatten().cpu())
    R["spread"].append(s.flatten()[::37].cpu())
    R["peak"].append(torch.stack([n.abs().amax(-1).flatten(), s.amax(-1).flatten(), s.median(-1).values.flatten()], 1).cpu())
    flat = noise.reshape(-1, 1, ctx)
    nll = torch.cat([-mixture_log_prob(c[:, 0], *dm.noise_model(c)) for c in flat.split(64)])[:, K].mean(-1)
    v = n.reshape(-1, n.size(-1)).square().mean(-1)
    R["nll"].append(nll.cpu()); R["gauss_fit"].append((0.5 * torch.log(2 * math.pi * v) + 0.5).cpu())
    R["gauss_unit"].append((0.5 * math.log(2 * math.pi) + 0.5 * v).cpu())
    # share of the clean signal that the mean takes away
    c = cerr[..., K]; w = 1 / s1[..., K].square()
    d = (m1 - m0)[..., K]
    has = ~torch.isnan(params["snr"])
    R["eaten"].append(((d * c * w).sum((1, 2)) / (c * c * w).sum((1, 2)))[has].cpu())
    R["snr"].append(params["snr"][has].cpu())
R = {k: torch.cat(v).double().numpy() for k, v in R.items() if v}
q = lambda x, p: np.percentile(x, p)
print(f"windows x detectors: {len(R['nll'])}")
print(f"error rms per window: median {q(R['err_rms'],50):.3f}, 1%-99% {q(R['err_rms'],1):.3f}-{q(R['err_rms'],99):.3f}")
print(f"mean rms / error rms: median {q(R['mean_rms']/R['err_rms'],50):.4f}, 99% {q(R['mean_rms']/R['err_rms'],99):.4f}; power share {np.mean(R['mean_rms']**2)/np.mean(R['err_rms']**2):.4f}")
print("spread percentiles 1/10/50/90/99/99.9/max:", " ".join(f"{q(R['spread'],p):.3f}" for p in (1,10,50,90,99,99.9,100)))
pk = R["peak"]; g = pk[:, 0] > 8
print(f"windows with a sample above 8: {g.sum()} of {len(g)}; their max spread median {np.median(pk[g,1]) if g.any() else float('nan'):.2f}, others {np.median(pk[~g,1]):.3f}; median spread {np.median(pk[g,2]) if g.any() else float('nan'):.3f} vs {np.median(pk[~g,2]):.3f}")
for name, k in (("all", np.ones(len(g), bool)), ("no sample above 8", ~g), ("a sample above 8", g)):
    if k.any(): print(f"nats per sample, {name} (n={k.sum()}): network {R['nll'][k].mean():.4f}, Gaussian with the window's own variance {R['gauss_fit'][k].mean():.4f}, unit Gaussian {R['gauss_unit'][k].mean():.4f}")
e, snr = R["eaten"], R["snr"]
for lo, hi in ((4, 8), (8, 12), (12, 100)):
    k = (snr >= lo) & (snr < hi)
    print(f"share of clean signal removed by the mean, SNR {lo}-{hi} (n={k.sum()}): median {np.median(e[k]):.4f}, 5%-95% {q(e[k],5):.4f} to {q(e[k],95):.4f}")
