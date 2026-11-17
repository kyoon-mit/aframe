"""Least-squares linear predictor of raw strain from its past, for
``arch.LinearS4Noise``.

    python -m train.experimental.raw_noise.fit_taps <raw.yaml> <order> <train batches> <out.npz>

Fits x_t on x_{t-1}..x_{t-order} over the config's training windows (strain
divided by the task's ``scale``, after the loader's sign flip and time
reversal, first second dropped), both detectors pooled, in float64. Saves
``taps`` (order,), ``std`` (spread of the errors over the fit) and ``count``.
"""

import sys
from types import SimpleNamespace

import lightning.pytorch as pl
import numpy as np
import torch
import yaml

from train.experimental.slic.dump_windows import build_data_module


def main():
    config, order, batches, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    with open(config) as f:
        cfg = yaml.safe_load(f)
    pl.seed_everything(cfg["seed_everything"])
    scale = cfg["model"]["init_args"]["scale"]
    dev = torch.device("cuda")
    dm = build_data_module(config, "val")
    dm.trainer.lightning_module = SimpleNamespace(device=dev)
    dm.transforms_to_device()
    rate = int(dm.hparams.sample_rate)
    R = torch.zeros(order + 1, order + 1, dtype=torch.float64, device=dev)
    n = 0
    for [X] in dm.strain_dataloader(dm.train_fnames, batches):
        X = dm.reverser(dm.inverter(X.to(dev).double()))[..., rate:] / scale
        for row in X.reshape(-1, X.size(-1)):
            A = row.unfold(-1, order + 1, 1).flip(-1)
            R += A.T @ A
            n += len(A)
    taps = torch.linalg.solve(R[1:, 1:], R[1:, 0])
    var = (R[0, 0] - taps @ R[1:, 0]) / n
    np.savez(out, taps=taps.cpu().numpy(), std=var.sqrt().item(), count=n)
    print(f"order {order}, {n} samples, error std {var.sqrt():.6f}")


if __name__ == "__main__":
    main()
