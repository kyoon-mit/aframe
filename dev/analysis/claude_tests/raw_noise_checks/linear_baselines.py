"""Scores of linear predictors of raw strain, in the unit of val/loss_raw_last
(minus log-likelihood per sample of strain / scale, last 4 s of each window).

  spectrum:        Gaussian from the window's own 20 s spectrum (the reference)
  spectrum_cal:    same, error variance measured on the window's earlier samples
  global AR(q):    one least-squares predictor from the past q samples, fit on
                   training windows, the same for every window; error variance
                   global, or measured on the window's earlier samples

usage: linear_baselines.py <raw.yaml> <train batches> <val batches>
"""

import math
import sys
from types import SimpleNamespace

import lightning.pytorch as pl
import torch
import yaml

from train.experimental.raw_noise.predict import prediction_errors
from train.experimental.slic.dump_windows import build_data_module

P, ORDERS, LAST, SCALE = 2048, (32, 128, 512, 2048), 8192, 3e-19
LOG2PI = math.log(2 * math.pi)


def windows(dm, fnames, batches, dev):
    for [X] in dm.strain_dataloader(fnames, batches):
        X = dm.reverser(dm.inverter(X.to(dev).double()))
        yield X


def lags(x, p):
    """(rows, p + 1) matrix: column 0 is x_t, column k is x_{t-k}."""
    return x.unfold(-1, p + 1, 1).flip(-1).reshape(-1, p + 1)


def main():
    config, ntrain, nval = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    dev = torch.device("cuda")
    with open(config) as f:
        pl.seed_everything(yaml.safe_load(f)["seed_everything"])
    dm = build_data_module(config, "val")
    dm.trainer.lightning_module = SimpleNamespace(device=dev)
    dm.transforms_to_device()
    rate = int(dm.hparams.sample_rate)

    R = torch.zeros(P + 1, P + 1, dtype=torch.float64, device=dev)
    n = 0
    for X in windows(dm, dm.train_fnames, ntrain, dev):
        x = (X[..., rate:] / SCALE).reshape(-1, X.size(-1) - rate)
        for row in x:
            A = lags(row, P)
            R += A.T @ A
            n += len(A)
    fits = {}
    for q in ORDERS:
        a = torch.linalg.solve(R[1 : q + 1, 1 : q + 1], R[1 : q + 1, 0])
        var = (R[0, 0] - a @ R[1 : q + 1, 0]) / n
        fits[q] = (a, var)
        print(f"AR({q}) fit on {n} samples: error std {var.sqrt():.5f} in strain / {SCALE}", flush=True)

    sums = {}

    def add(name, nll):
        s = sums.setdefault(name, [0.0, 0])
        s[0] += nll.sum().item()
        s[1] += nll.numel()

    for X in windows(dm, dm.valid_fnames, nval, dev):
        _, psds = dm.psd_estimator(X.float())
        Xd = X[..., rate:]
        e, h0 = prediction_errors(Xd, psds, rate, rate)
        e = e.double()
        off = -(torch.log(h0.double()) + math.log(SCALE))[..., None]
        tail, past = e[..., -LAST:], e[..., :-LAST]
        add("spectrum", 0.5 * tail**2 + 0.5 * LOG2PI + off)
        v = past.square().mean(-1, keepdim=True)
        add("spectrum_cal", 0.5 * tail**2 / v + 0.5 * torch.log(v) + 0.5 * LOG2PI + off)
        x = (Xd / SCALE).reshape(-1, Xd.size(-1))
        for q, (a, var) in fits.items():
            A = lags(x, q).reshape(len(x), -1, q + 1)
            err = A[..., 0] - A[..., 1:] @ a
            tail, past = err[:, -LAST:], err[:, :-LAST]
            add(f"global AR({q}), global variance", 0.5 * tail**2 / var + 0.5 * torch.log(var) + 0.5 * LOG2PI)
            v = past.square().mean(-1, keepdim=True)
            add(f"global AR({q}), window variance", 0.5 * tail**2 / v + 0.5 * torch.log(v) + 0.5 * LOG2PI)
    print("nats per sample of strain / 3e-19, last 4 s (lower is better):")
    for name, (s, c) in sums.items():
        print(f"  {name:36s} {s / c:8.4f}   ({c} samples)")


if __name__ == "__main__":
    main()
