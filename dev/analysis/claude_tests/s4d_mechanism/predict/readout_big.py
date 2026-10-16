"""The chosen read-outs (best MLP from ``readout.py select`` and the linear head) retrained on more
training windows: the training chunks train, train2, ... (64,000 windows each, SNR >= 8, training
background). Same loss, optimizer, schedule and early stopping as readout.py; soft targets are built
per batch so 512,000 windows fit in memory.

usage: readout_big.py N TAG...    N = number of training windows (a multiple of 64,000)
"""
import copy
import json
import sys
import time

import numpy as np
import torch

import readout as R

CHUNKS = ["train"] + [f"train{i}" for i in range(2, 17)]


def load_train(tag, n):
    xs, ms = [], []
    for c in CHUNKS:
        if sum(len(x) for x in xs) >= n:
            break
        x, mc, _, _ = R.features(c, tag)
        xs.append(x)
        ms.append(mc)
    x, mc = torch.cat(xs)[:n], np.concatenate(ms)[:n]
    if len(x) < n:
        raise RuntimeError(f"only {len(x)} training windows for {tag}")
    return x, mc


def train(cfg, xtr, mctr, xes, mces, seed=0, lr=1e-3, wd=1e-4, bs=512, max_epochs=400, patience=40):
    torch.manual_seed(seed)
    model = R.build(cfg, xtr.shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=10)
    xtr_d = xtr.to(R.DEV)
    lt = torch.log(torch.as_tensor(mctr, dtype=torch.float32, device=R.DEV))
    g = R.GRID.to(R.DEV)
    n = len(xtr_d)
    best, t0 = (-1.0, None, -1), time.time()
    for ep in range(max_epochs):
        model.train()
        perm = torch.randperm(n, device=R.DEV)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            t = torch.softmax(-((lt[idx, None] - g[None]) ** 2) / (2 * R.WIDTH**2), -1)
            loss = -(t * torch.log_softmax(model(xtr_d[idx]), -1)).sum(-1).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        acc = float(R.within(R.predict(model, xes), mces).mean())
        sched.step(acc)
        if acc > best[0]:
            best = (acc, copy.deepcopy(model.state_dict()), ep)
        elif ep - best[2] >= patience:
            break
    model.load_state_dict(best[1])
    info = {"best_epoch": best[2], "epochs_run": ep + 1, "es_within1": best[0], "seconds": round(time.time() - t0, 1),
            "es_ce": R.ce(model, xes.to(R.DEV), R.soft_target(mces)), "params": sum(p.numel() for p in model.parameters()),
            "lr_end": opt.param_groups[0]["lr"], "n_train": n}
    return model, info


def main(n, tags):
    for tag in tags:
        sel = json.load(open(R.R / f"select__{tag}.json"))
        xtr, mctr = load_train(tag, n)
        xes, mces, _, _ = R.features("es", tag)
        out = {"tag": tag, "n_train": n, "configs": {}}
        for name in ("linear", sel["best_mlp"]):
            model, info = train(R.CONFIGS[name], xtr, mctr, xes, mces)
            res, ea, eb = R.evaluate(model, tag)
            out["configs"][name] = {"train": info, **res}
            np.save(R.R / f"est_testA__{tag}__{name}__n{n}.npy", ea)
            np.save(R.R / f"est_testB__{tag}__{name}__n{n}.npy", eb)
            print(f"{tag} {name} n={n}: es {info['es_within1']:.4f} (ep {info['best_epoch']}/{info['epochs_run']}, {info['seconds']} s) "
                  f"testA {res['testA']['within1']:.4f} bands " +
                  " ".join(f"{k}:{v['within1']:.3f}" for k, v in res["testAB_bands"].items()), flush=True)
        json.dump(out, open(R.R / f"big__{tag}__n{n}.json", "w"), indent=1)


if __name__ == "__main__":
    main(int(sys.argv[1]), sys.argv[2:])
