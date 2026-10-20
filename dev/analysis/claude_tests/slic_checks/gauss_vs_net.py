"""Score network vs white Gaussian noise model, denoising loss on the
validation noise windows of score.yaml, at fixed diffusion times.

Pass 1: per-detector variance of whitened training windows.
Pass 2: on validation windows, for each t, the same z for every model:
  network   out = net(x + sigma z, t)
  gauss_fit out = -sigma (x + sigma z) / (v_ifo + sigma^2)
  gauss_1   out = -sigma (x + sigma z) / (1 + sigma^2)
loss = sum over samples of (out + z)^2, per window.

usage: gauss_vs_net.py <score.yaml> <ckpt> <out.npz> [fit batches]
"""

import sys

import lightning.pytorch as pl
import numpy as np
import torch
from lightning.pytorch.cli import LightningCLI

import train.cli  # noqa: F401
from train.data import BaseAframeDataset
from train.experimental.slic.cli import ScoreCLI
from train.experimental.slic.data import NoiseWindows
from train.experimental.slic.sde import VESDE
from train.experimental.slic.task import load_score_net

T_VALUES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]


class TrainWindows(NoiseWindows):
    """Training-file windows served through the validation loop."""

    def __init__(self, dataset, batches):
        super().__init__(dataset)
        self.batches = batches

    def val_dataloader(self):
        return self.dataset.strain_dataloader(None, self.batches)


class Variance(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.register_buffer("power", torch.zeros(2, dtype=torch.float64))
        self.n = 0

    def validation_step(self, windows, batch_idx):
        x = windows.double().reshape(-1, 2, windows.shape[-1])
        self.power += x.square().mean(-1).sum(0)
        self.n += len(x)


class Compare(pl.LightningModule):
    def __init__(self, net, variance):
        super().__init__()
        self.net = net
        self.variance = variance
        self.sde = VESDE()
        self.rows = []

    @torch.no_grad()
    def validation_step(self, windows, batch_idx):
        x = windows.float()
        v = self.variance.to(x).repeat(len(x) // 2)[:, None, None]
        out = {}
        for t in T_VALUES:
            tt = torch.full((len(x),), t, dtype=x.dtype)
            sigma = float(self.sde.sigma(torch.tensor(t, dtype=torch.float64)))
            z = torch.randn_like(x)
            xt = x + sigma * z
            models = {
                "network": self.net(xt, tt),
                "gauss_fit": -sigma * xt / (v + sigma**2),
                "gauss_1": -sigma * xt / (1 + sigma**2),
            }
            for name, o in models.items():
                out[(name, t)] = (o + z).double().square().sum((1, 2))
        self.rows.append(out)


def main():
    config, ckpt, path = sys.argv[1:4]
    fit_batches = int(sys.argv[4]) if len(sys.argv) > 4 else 32
    torch.set_num_threads(16)
    cli = ScoreCLI(
        pl.LightningModule,
        BaseAframeDataset,
        subclass_mode_model=True,
        subclass_mode_data=True,
        parser_kwargs={"parser_mode": "omegaconf"},
        save_config_callback=None,
        seed_everything_default=101588,
        run=False,
        args=[
            "--config", config,
            "--trainer.accelerator=cpu",
            "--trainer.logger=false",
            "--trainer.enable_progress_bar=false",
        ],
    )
    dataset = cli.datamodule
    trainer = cli.trainer

    var_task = Variance()
    trainer.validate(var_task, datamodule=TrainWindows(dataset, fit_batches))
    variance = (var_task.power / var_task.n).float()
    print(f"training windows {var_task.n}: variance H1 {variance[0]:.4f} "
          f"L1 {variance[1]:.4f}", flush=True)

    net = load_score_net(ckpt, map_location="cpu")
    task = Compare(net, variance)
    trainer.validate(task, datamodule=NoiseWindows(dataset))

    keys = task.rows[0].keys()
    loss = {k: torch.cat([r[k] for r in task.rows]).numpy() for k in keys}
    n = len(loss[("network", 0.0)])
    print(f"validation windows {n} (H1 and L1 each {n // 2})")
    print("t     sigma      network     gauss_fit   gauss_1   "
          "net-fit (se)")
    save = {"variance": variance.numpy(), "t": np.array(T_VALUES)}
    for t in T_VALUES:
        sigma = 1e-3 * 1e7**t
        a, b, c = (loss[(m, t)] for m in ("network", "gauss_fit", "gauss_1"))
        d = a - b
        print(f"{t:.1f}  {sigma:9.4g}  {a.mean():10.2f}  {b.mean():10.2f}  "
              f"{c.mean():10.2f}  {d.mean():+9.2f} ({d.std() / np.sqrt(n):.2f})")
        for m, arr in (("network", a), ("gauss_fit", b), ("gauss_1", c)):
            save[f"{m}_t{t:.1f}"] = arr
    np.savez(path, **save)
    print("->", path)


if __name__ == "__main__":
    main()
