"""Scores of a linear predictor fit on each window's own past, in the unit of
val/loss_raw_last (minus log-likelihood per sample of strain / scale, last 4 s
of each window), paired with the spectrum Gaussian on the same windows.

For every window and detector, x_t is fit on x_{t-1}..x_{t-p} by least
squares over the ``span`` seconds just before the last 4 s (strain only, no
spectrum), with a small ridge; the error variance is the fit's. The last 4 s
are then scored with that predictor and variance.

usage: in_context_ar.py <raw.yaml> <val batches> <p:span> [...]
"""

import math
import sys
from types import SimpleNamespace

import lightning.pytorch as pl
import torch
import yaml

from train.experimental.raw_noise.predict import prediction_errors
from train.experimental.slic.dump_windows import build_data_module

LAST, SCALE, RIDGE = 8192, 3e-19, 1e-9
LOG2PI = math.log(2 * math.pi)


def lags(x, p):
    return x.unfold(-1, p + 1, 1).flip(-1)


def main():
    config, nval = sys.argv[1], int(sys.argv[2])
    variants = [tuple(int(v) for v in a.split(":")) for a in sys.argv[3:]]
    dev = torch.device("cuda")
    with open(config) as f:
        pl.seed_everything(yaml.safe_load(f)["seed_everything"])
    dm = build_data_module(config, "val")
    dm.trainer.lightning_module = SimpleNamespace(device=dev)
    dm.transforms_to_device()
    rate = int(dm.hparams.sample_rate)
    rows = {"spectrum": []} | {v: [] for v in variants}
    for [X] in dm.strain_dataloader(dm.valid_fnames, nval):
        X = dm.reverser(dm.inverter(X.to(dev).double()))
        _, psds = dm.psd_estimator(X.float())
        Xd = X[..., rate:]
        e, h0 = prediction_errors(Xd, psds, rate, rate)
        off = -(torch.log(h0.double()) + math.log(SCALE))
        rows["spectrum"].append((0.5 * e.double()[..., -LAST:] ** 2 + 0.5 * LOG2PI).mean(-1).add(off).cpu())
        x = Xd / SCALE
        for p, span in variants:
            out = torch.empty(x.shape[:2], dtype=torch.float64)
            for b in range(x.size(0)):
                for c in range(x.size(1)):
                    row = x[b, c]
                    fit = lags(row[-(LAST + span * rate + p) : -LAST], p)
                    R = fit.T @ fit
                    A, y = R[1:, 1:], R[1:, 0]
                    A = A + RIDGE * A.diagonal().mean() * torch.eye(p, dtype=A.dtype, device=dev)
                    a = torch.linalg.solve(A, y)
                    var = (R[0, 0] - a @ y) / len(fit)
                    test = lags(row[-(LAST + p) :], p)
                    err = test[:, 0] - test[:, 1:] @ a
                    out[b, c] = (0.5 * err**2 / var + 0.5 * torch.log(var) + 0.5 * LOG2PI).mean().item()
            rows[(p, span)].append(out)
        print(f"{sum(len(r) for r in rows['spectrum'])} windows", flush=True)
    spec = torch.cat(rows["spectrum"])
    print(f"nats per sample of strain / {SCALE}, last 4 s; spectrum Gaussian {spec.mean():.4f}")
    for v in variants:
        r = torch.cat(rows[v])
        d = r - spec
        print(f"  AR({v[0]}) fit on the past {v[1]} s: {r.mean():.4f}; minus spectrum "
              f"H1 {d[:, 0].mean():+.4f} +- {d[:, 0].std() / len(d) ** 0.5:.4f}, "
              f"L1 {d[:, 1].mean():+.4f} +- {d[:, 1].std() / len(d) ** 0.5:.4f}, "
              f"better in {(d < 0).double().mean():.0%} of windows")


if __name__ == "__main__":
    main()
