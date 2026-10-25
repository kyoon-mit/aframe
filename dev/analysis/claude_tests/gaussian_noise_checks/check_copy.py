"""Checks of GaussianNoiseDenoiserDataset on test batches of a config.

For raw background stretches: the PSD of the Gaussian copy against the PSD
of the real stretch (ratio per frequency, 20 Hz and above, median and
central 90% over stretches and frequencies). For injected batches: variance
of whitened noise (X - X_clean) per sample, its largest |value|, and the
whitened signal's norm over aframe's SNR label, for the copy and for real
noise (the same config with the plain DenoiserOnlyAframeDataset).

usage: check_copy.py <config.yaml> [batches]
"""

import sys

import lightning.pytorch as pl
import numpy as np
import torch
import yaml

from train.experimental.gaussian_noise.data import gaussian_copy
from train.experimental.slic.dump_windows import build_data_module


def stats(dm, batches):
    var, peak, ratio = [], [], []
    for i, batch in enumerate(dm.test_dataloader()):
        if i >= batches:
            break
        batch = dm.on_before_batch_transfer(batch, 0)
        X, X_clean, y, params = dm.on_after_batch_transfer(batch, 0)
        n = (X - X_clean).double()
        var.append(n.var(-1).flatten().cpu())
        peak.append(n.abs().amax(-1).flatten().cpu())
        keep = y[:, 0] > 0
        ratio.append((X_clean[keep].double().square().sum((1, 2)).sqrt() / params["snr"][keep]).cpu())
    return torch.cat(var), torch.cat(peak), torch.cat(ratio)


def main():
    config = sys.argv[1]
    batches = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    with open(config) as f:
        seed = yaml.safe_load(f)["seed_everything"]
    pl.seed_everything(seed)
    dm = build_data_module(config, "test")
    dm.inverter.prob = dm.reverser.prob = 0.0
    loader = dm.test_dataloader()
    batch = next(iter(loader))
    batch = dm.on_before_batch_transfer(batch, 0)
    [X], _ = batch if dm.waveforms_from_disk else (batch, None)
    _, psd_real = dm.psd_estimator(X)
    _, psd_copy = dm.psd_estimator(gaussian_copy(X, psd_real, dm.hparams.sample_rate))
    freqs = np.linspace(0, dm.hparams.sample_rate / 2, psd_real.shape[-1])
    r = (psd_copy / psd_real)[..., freqs >= 20].double().flatten().cpu().numpy()
    print(f"PSD copy / real, 20 Hz and above, {X.shape[0]} stretches: median {np.median(r):.4f}, "
          f"5% {np.quantile(r, 0.05):.4f}, 95% {np.quantile(r, 0.95):.4f}")
    for name, module in (("Gaussian copy", dm),):
        var, peak, ratio = stats(module, batches)
        print(f"{name}: whitened noise variance median {var.median():.4f} (5% {var.quantile(0.05):.4f}, "
              f"95% {var.quantile(0.95):.4f}); largest |value| median {peak.median():.3f}, max {peak.max():.3f}; "
              f"|X_clean| / SNR label median {ratio.median():.4f}")
    pl.seed_everything(seed)
    with open(config) as f:
        cfg = yaml.safe_load(f)
    cfg["data"]["class_path"] = "train.data.supervised.DenoiserOnlyAframeDataset"
    real_cfg = config + ".real.yaml"
    with open(real_cfg, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    dm_real = build_data_module(real_cfg, "test")
    dm_real.inverter.prob = dm_real.reverser.prob = 0.0
    var, peak, ratio = stats(dm_real, batches)
    print(f"real noise: whitened noise variance median {var.median():.4f} (5% {var.quantile(0.05):.4f}, "
          f"95% {var.quantile(0.95):.4f}); largest |value| median {peak.median():.3f}, max {peak.max():.3f}; "
          f"|X_clean| / SNR label median {ratio.median():.4f}")


if __name__ == "__main__":
    main()
