"""Score network vs white and colored Gaussian noise models, denoising loss on the
validation noise windows of score.yaml, at fixed diffusion times.

Pass 1: per-detector variance of whitened training windows.
Pass 2: on validation windows, for each t, the same z for every model:
  network   out = net(x + sigma z, t)
  gauss_fit out = -sigma (x + sigma z) / (v_ifo + sigma^2)
  gauss_1   out = -sigma (x + sigma z) / (1 + sigma^2)
  gauss_psd out = -sigma irfft(rfft(x + sigma z) / (P_ifo(f) + sigma^2)),
            P_ifo the mean |rfft|^2 (orthonormal) of training windows
  gauss_med same with P_ifo the median |rfft|^2 / ln 2
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
        self.register_buffer("spectrum", torch.zeros(2, 4097, dtype=torch.float64))
        self.n = 0
        self.spectra = []

    def validation_step(self, windows, batch_idx):
        x = windows.double().reshape(-1, 2, windows.shape[-1])
        self.power += x.square().mean(-1).sum(0)
        fx = torch.fft.rfft(x, norm="ortho")
        self.spectrum += fx.abs().square().sum(0)
        self.spectra.append(fx.abs().square().cpu())
        self.n += len(x)


class Compare(pl.LightningModule):
    def __init__(self, net, variance, spectrum, median):
        super().__init__()
        self.median = median
        self.net = net
        self.variance = variance
        self.spectrum = spectrum
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
            p = self.spectrum.to(torch.float64).repeat(len(x) // 2, 1)[:, None]
            fx = torch.fft.rfft(xt.double(), norm="ortho")
            colored = torch.fft.irfft(fx / (p + sigma**2), n=xt.shape[-1], norm="ortho")
            pm = self.median.to(torch.float64).repeat(len(x) // 2, 1)[:, None]
            colored_m = torch.fft.irfft(fx / (pm + sigma**2), n=xt.shape[-1], norm="ortho")
            models = {
                "gauss_psd": (-sigma * colored).float(),
                "gauss_med": (-sigma * colored_m).float(),
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
    spectrum = var_task.spectrum / var_task.n
    median = torch.cat(var_task.spectra).median(0).values / np.log(2)
    print(f"median spectrum 100-500 Hz H1 {median[0, 400:2000].mean():.4f} "
          f"L1 {median[1, 400:2000].mean():.4f}", flush=True)
    low = spectrum[:, :80].mean(1)
    mid = spectrum[:, 400:2000].mean(1)
    print(f"spectrum below 20 Hz H1 {low[0]:.4f} L1 {low[1]:.4f}; "
          f"100-500 Hz H1 {mid[0]:.4f} L1 {mid[1]:.4f}", flush=True)
    print(f"training windows {var_task.n}: variance H1 {variance[0]:.4f} "
          f"L1 {variance[1]:.4f}", flush=True)

    net = load_score_net(ckpt, map_location="cpu")
    task = Compare(net, variance, spectrum, median)
    trainer.validate(task, datamodule=NoiseWindows(dataset))

    keys = task.rows[0].keys()
    loss = {k: torch.cat([r[k] for r in task.rows]).numpy() for k in keys}
    n = len(loss[("network", 0.0)])
    print(f"validation windows {n} (H1 and L1 each {n // 2})")
    print("t     sigma      network   gauss_psd    gauss_1   "
          "net-psd median (se of mean)")
    print("(gauss_med and per-detector medians in the npz)")
    save = {"variance": variance.numpy(), "spectrum": spectrum.numpy(),
            "median_spectrum": median.numpy(),
            "t": np.array(T_VALUES)}
    for t in T_VALUES:
        sigma = 1e-3 * 1e7**t
        a, b, c = (loss[(m, t)] for m in ("network", "gauss_psd", "gauss_1"))
        d = a - b
        print(f"{t:.1f}  {sigma:9.4g}  {a.mean():10.2f}  {b.mean():10.2f}  "
              f"{c.mean():10.2f}  {np.median(d):+9.2f} ({d.std() / np.sqrt(n):.2f}) "
              f"net better {np.mean(d < 0):.3f}")
        for m, arr in (("network", a), ("gauss_psd", b), ("gauss_1", c),
                       ("gauss_med", loss[("gauss_med", t)])):
            save[f"{m}_t{t:.1f}"] = arr
    np.savez(path, **save)
    print("->", path)


if __name__ == "__main__":
    main()
