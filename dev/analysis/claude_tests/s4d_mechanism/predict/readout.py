"""Read-outs trained from scratch on fixed channel features.

Features: z = standardized log1p(peak energy) per window over the 256 channels, as
LongConvEnergyNet.forward computes them. Target, loss and estimate as LongConvClassifier: soft
Gaussian labels over the 235-point log chirp-mass grid (width 0.5 grid step), cross-entropy,
parabola-refined peak (``grid_estimate``).

Training windows: data/train (SNR >= 8, training background); early stopping on data/es (SNR >= 4,
validation background) by within 1%, as the runs' checkpoints were chosen; test: data/testA (the
harness's 20,000 validation-like windows, SNR >= 4) overall, and testA + testB (SNR >= 8) per band.

usage: readout.py select TAG...      small grid of read-outs per extractor, best by early-stop set
       readout.py curve TAG...       best config on 1/4, 1/2, all of the training windows (and train2)
       readout.py raw TAG...         best config on unstandardized log1p features
"""
import copy
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from train.experimental.long_conv.arch import LongConvEnergyNet, grid_estimate

HERE = Path(__file__).resolve().parent
D = HERE / "data"
R = HERE / "results"
R.mkdir(exist_ok=True)
torch.set_num_threads(8)
DEV = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if DEV.type == "cuda":
    torch.cuda.set_per_process_memory_fraction(min(1.0, 1.5 * 2**30 / torch.cuda.get_device_properties(DEV).total_memory), DEV)
GRID = torch.arange(math.log(0.85), math.log(2.75) + 1e-9, 0.005).float()
WIDTH = 0.5 * 0.005
assert torch.equal(GRID, LongConvEnergyNet(channels=4, kernel="free").grid_log_mc)
BANDS = [(4, 8), (8, 10), (10, 12), (12, 16), (16, 25), (25, 100.01)]
CONFIGS = {  # name: (hidden layers, width, dropout)
    "linear": (0, 0, 0.0),
    "mlp1x1024_d0.1": (1, 1024, 0.1),
    "mlp2x1024_d0.1": (2, 1024, 0.1),
    "mlp2x1024_d0.3": (2, 1024, 0.3),
    "mlp3x1024_d0.2": (3, 1024, 0.2),
}


def znet(pe):
    """LongConvEnergyNet.forward's features: log1p, standardized over channels per window."""
    z = torch.log1p(torch.as_tensor(pe, dtype=torch.float32))
    return (z - z.mean(-1, keepdim=True)) / z.std(-1, keepdim=True).clamp_min(1e-6)


def load(which, tag):
    pe = np.load(D / which / f"pe__{tag}.npy")
    lab = np.load(D / which / "labels.npz")
    return pe, lab["chirp_mass"].astype(np.float64), lab["snr"].astype(np.float64)


def features(which, tag, kind="z", stats=None):
    pe, mc, snr = load(which, tag)
    if kind == "z":
        x = znet(pe)
    else:                                   # unstandardized log1p, affine-normalized by training stats
        x = torch.log1p(torch.as_tensor(pe, dtype=torch.float32))
        if stats is None:
            stats = (x.mean(0, keepdim=True), x.std(0, keepdim=True).clamp_min(1e-6))
        x = (x - stats[0]) / stats[1]
    return x, mc, snr, stats


def soft_target(mc):
    lt = torch.log(torch.as_tensor(mc, dtype=torch.float32, device=DEV))
    g = GRID.to(DEV)
    return torch.softmax(-((lt[:, None] - g[None]) ** 2) / (2 * WIDTH**2), -1)


def build(cfg, d_in=256):
    depth, width, p = cfg
    layers, d = [], d_in
    for _ in range(depth):
        layers += [nn.Linear(d, width), nn.GELU(), nn.Dropout(p)]
        d = width
    layers.append(nn.Linear(d, len(GRID)))
    return nn.Sequential(*layers).to(DEV)


@torch.no_grad()
def predict(model, x, bs=8192):
    model.eval()
    out = []
    for i in range(0, len(x), bs):
        out.append(grid_estimate(model(x[i:i + bs].to(DEV)), GRID.to(DEV))[0].exp().double().cpu())
    return torch.cat(out).numpy()


@torch.no_grad()
def ce(model, x, t, bs=8192):
    model.eval()
    s = 0.0
    for i in range(0, len(x), bs):
        s += float(-(t[i:i + bs] * torch.log_softmax(model(x[i:i + bs].to(DEV)), -1)).sum())
    return s / len(x)


def within(est, mc, tol=0.01):
    return np.abs(est - mc) / mc < tol


def train(cfg, xtr, mctr, xes, mces, seed=0, lr=1e-3, wd=1e-4, bs=512, max_epochs=400, patience=40):
    torch.manual_seed(seed)
    model = build(cfg, xtr.shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=10)
    xtr_d, ttr = xtr.to(DEV), soft_target(mctr)
    tes = soft_target(mces)
    n = len(xtr_d)
    best, hist, t0 = (-1.0, None, -1), [], time.time()
    for ep in range(max_epochs):
        model.train()
        perm = torch.randperm(n, device=DEV)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            loss = -(ttr[idx] * torch.log_softmax(model(xtr_d[idx]), -1)).sum(-1).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        acc = float(within(predict(model, xes), mces).mean())
        sched.step(acc)
        hist.append((ep, acc))
        if acc > best[0]:
            best = (acc, copy.deepcopy(model.state_dict()), ep)
        elif ep - best[2] >= patience:
            break
    model.load_state_dict(best[1])
    info = {"best_epoch": best[2], "epochs_run": ep + 1, "es_within1": best[0], "seconds": round(time.time() - t0, 1),
            "es_ce": ce(model, xes.to(DEV), tes), "train_ce": ce(model, xtr_d, ttr),
            "params": sum(p.numel() for p in model.parameters()), "lr_end": opt.param_groups[0]["lr"]}
    return model, info


def score(est, mc, snr):
    """Within 1% overall (and 2/5/10) and per SNR band, with counts and binomial standard errors."""
    out = {"n": int(len(mc))}
    for tol in (1, 2, 5, 10):
        out[f"within{tol}"] = float(within(est, mc, tol / 100).mean())
    w = within(est, mc)
    out["se1"] = float(np.sqrt(out["within1"] * (1 - out["within1"]) / len(mc)))
    out["bands"] = {}
    for lo, hi in BANDS:
        m = (snr >= lo) & (snr < hi)
        p = float(w[m].mean()) if m.any() else float("nan")
        out["bands"][f"{lo}-{hi:g}"] = {"n": int(m.sum()), "within1": p, "se": float(np.sqrt(p * (1 - p) / max(1, m.sum())))}
    return out


def evaluate(model, tag, kind="z", stats=None):
    """Test A overall and test A + test B per band; per-window correctness saved for paired comparisons."""
    xa, mca, snra, _ = features("testA", tag, kind, stats)
    xb, mcb, snrb, _ = features("testB", tag, kind, stats)
    ea, eb = predict(model, xa), predict(model, xb)
    res = {"testA": score(ea, mca, snra),
           "testAB_bands": score(np.concatenate([ea, eb]), np.concatenate([mca, mcb]), np.concatenate([snra, snrb]))["bands"]}
    return res, ea, eb


def own_head(tag):
    """(i): the network's own estimates on test A and test A + B (from features.py)."""
    base = tag
    ea = np.load(D / "testA" / f"est__{base}.npy")
    eb = np.load(D / "testB" / f"est__{base}.npy")
    _, mca, snra = load("testA", tag)
    _, mcb, snrb = load("testB", tag)
    return {"testA": score(ea, mca, snra),
            "testAB_bands": score(np.concatenate([ea, eb]), np.concatenate([mca, mcb]), np.concatenate([snra, snrb]))["bands"]}, ea, eb


def cmd_select(tags):
    for tag in tags:
        out = {"tag": tag, "configs": {}}
        xtr, mctr, _, _ = features("train", tag)
        xes, mces, _, _ = features("es", tag)
        if tag in ("d64", "d512", "band", "free", "d64_exact", "d512_exact"):
            out["own_head"], ea, eb = own_head(tag)
            np.save(R / f"est_testA__{tag}__own.npy", ea)
            np.save(R / f"est_testB__{tag}__own.npy", eb)
        for name, cfg in CONFIGS.items():
            model, info = train(cfg, xtr, mctr, xes, mces)
            res, ea, eb = evaluate(model, tag)
            out["configs"][name] = {"train": info, **res}
            np.save(R / f"est_testA__{tag}__{name}.npy", ea)
            np.save(R / f"est_testB__{tag}__{name}.npy", eb)
            print(f"{tag} {name}: es {info['es_within1']:.4f} (ep {info['best_epoch']}/{info['epochs_run']}, {info['seconds']} s) "
                  f"testA {res['testA']['within1']:.4f} bands " +
                  " ".join(f"{k}:{v['within1']:.3f}" for k, v in res["testAB_bands"].items()), flush=True)
        mlps = [k for k in CONFIGS if k != "linear"]
        out["best_mlp"] = max(mlps, key=lambda k: out["configs"][k]["train"]["es_within1"])
        json.dump(out, open(R / f"select__{tag}.json", "w"), indent=1)


def cmd_curve(tags):
    for tag in tags:
        sel = json.load(open(R / f"select__{tag}.json"))
        name = sel["best_mlp"]
        cfg = CONFIGS[name]
        xtr, mctr, _, _ = features("train", tag)
        xes, mces, _, _ = features("es", tag)
        out = {"tag": tag, "config": name, "points": {}}
        sizes = [len(xtr) // 4, len(xtr) // 2]
        for n in sizes:
            model, info = train(cfg, xtr[:n], mctr[:n], xes, mces)
            res, _, _ = evaluate(model, tag)
            out["points"][str(n)] = {"train": info, **res}
            print(f"{tag} {name} n={n}: testA {res['testA']['within1']:.4f}", flush=True)
        out["points"][str(len(xtr))] = sel["configs"][name]
        if (D / "train2" / f"pe__{tag}.npy").exists():
            x2, mc2, _, _ = features("train2", tag)
            xx, mm = torch.cat([xtr, x2]), np.concatenate([mctr, mc2])
            model, info = train(cfg, xx, mm, xes, mces)
            res, ea, eb = evaluate(model, tag)
            out["points"][str(len(xx))] = {"train": info, **res}
            np.save(R / f"est_testA__{tag}__{name}__n{len(xx)}.npy", ea)
            np.save(R / f"est_testB__{tag}__{name}__n{len(xx)}.npy", eb)
            print(f"{tag} {name} n={len(xx)}: testA {res['testA']['within1']:.4f}", flush=True)
        json.dump(out, open(R / f"curve__{tag}.json", "w"), indent=1)


def cmd_raw(tags):
    for tag in tags:
        sel = json.load(open(R / f"select__{tag}.json"))
        name = sel["best_mlp"]
        xtr, mctr, _, stats = features("train", tag, "raw")
        xes, mces, _, _ = features("es", tag, "raw", stats)
        model, info = train(CONFIGS[name], xtr, mctr, xes, mces)
        res, ea, eb = evaluate(model, tag, "raw", stats)
        np.save(R / f"est_testA__{tag}__{name}__raw.npy", ea)
        np.save(R / f"est_testB__{tag}__{name}__raw.npy", eb)
        json.dump({"tag": tag, "config": name, "train": info, **res}, open(R / f"raw__{tag}.json", "w"), indent=1)
        print(f"{tag} {name} raw: es {info['es_within1']:.4f} testA {res['testA']['within1']:.4f}", flush=True)


if __name__ == "__main__":
    {"select": cmd_select, "curve": cmd_curve, "raw": cmd_raw}[sys.argv[1]](sys.argv[2:])
