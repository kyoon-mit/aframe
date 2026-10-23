"""Test 2 diagnosis: per-detector network log-likelihood ratio on noise windows, window NLL, 505-515 Hz power, and the ratio by 1/16 s block. usage: test2_spread.py <config> <ckpt> <out.npz>"""
import sys, numpy as np, torch, yaml
import lightning.pytorch as pl
from train.experimental.next_sample.arch import CausalS4Noise
from train.experimental.next_sample.task import mixture_log_prob
from train.experimental.slic.dump_windows import build_data_module
N=8192; dev="cuda"
config, ckpt, out = sys.argv[1:4]
torch.set_num_threads(16)
pl.seed_everything(yaml.safe_load(open(config))["seed_everything"])
dm = build_data_module(config, "test"); dm.inverter.prob = 0.0; dm.reverser.prob = 0.0
net = CausalS4Noise(3, 64, 32, 4, 0.0, 1e-4, 10.0)
sd = torch.load(ckpt, map_location="cpu", weights_only=False)["state_dict"]
net.load_state_dict({k.removeprefix("model."): v for k, v in sd.items()}); net = net.to(dev).eval()
f = np.fft.rfftfreq(N, 1/2048); band = torch.from_numpy((f >= 505) & (f <= 515)).to(dev)
def lp(x):
    flat = x.reshape(-1, 1, N).float()
    with torch.no_grad():
        m, s, w = net(flat)
        return mixture_log_prob(flat[:, 0], m, s, w).double().reshape(len(x), 2, N)
rows = {k: [] for k in ["lr", "nll", "band", "hband", "hnorm", "lr_t"]}
tot = 0
for batch in dm.test_dataloader():
    batch = dm.on_before_batch_transfer(batch, 0)
    X, Xc, y, p = dm.on_after_batch_transfer(batch, 0)
    h = Xc.to(dev).double(); n = X.to(dev).double() - h
    a, b = lp(n), lp(n - h)
    rows["lr"].append((b - a).sum(-1).cpu()); rows["nll"].append((-a.mean(-1)).cpu())
    P = torch.fft.rfft(n).abs().square() / N; Ph = torch.fft.rfft(h).abs().square() / N
    rows["band"].append(P[..., band].sum(-1).cpu()); rows["hband"].append(Ph[..., band].sum(-1).cpu())
    rows["hnorm"].append(h.square().sum(-1).sqrt().cpu())
    rows["lr_t"].append((b - a).reshape(len(X), 2, 64, 128).sum(-1).cpu())
    tot += len(X)
    if tot >= 4000: break
np.savez(out, **{k: torch.cat(v).numpy() for k, v in rows.items()})
print("->", out)
