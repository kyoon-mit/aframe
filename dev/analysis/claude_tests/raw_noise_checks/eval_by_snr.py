"""Chirp-mass accuracy by SNR of den-reg checkpoints on whitened and on
noise-removed input, on the same windows.

Windows: the validation path of the noise-removed data config (validation
background and waveforms, SNR power law -3 from 4). Each window is given to
every network in the form its name asks for: "whitened" (the whitener's
output) or "noise_removed" (``NoiseRemovedAframeDataset``'s). Eval mode, no
dropout. Runs on GPU 0 of the visible ones.

usage: eval_by_snr.py <data config> <out.npz> <passes> <name=input:config:ckpt> [...]
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import lightning.pytorch as pl
import numpy as np
import torch
import yaml

from train.experimental.slic.dump_windows import build_data_module

sys.path.insert(0, str(Path(__file__).parents[1] / "gaussian_noise_checks"))
from eval_by_snr import BANDS, build_net, chirp_mass  # noqa: E402


def to_device(batch, device):
    if torch.is_tensor(batch):
        return batch.to(device)
    if isinstance(batch, dict):
        return {k: to_device(v, device) for k, v in batch.items()}
    return [to_device(b, device) for b in batch]


def main():
    data_config, out, passes = sys.argv[1], sys.argv[2], int(sys.argv[3])
    device = torch.device("cuda")
    nets = {}
    for arg in sys.argv[4:]:
        name, rest = arg.split("=", 1)
        kind, config, ckpt = rest.split(":", 2)
        if kind not in ("whitened", "noise_removed"):
            raise ValueError(f"unknown input {kind}")
        net, y_mean, y_std = build_net(config, ckpt)
        nets[name] = (kind, net.to(device), y_mean, y_std)
    with open(data_config) as f:
        pl.seed_everything(yaml.safe_load(f)["seed_everything"])
    dm = build_data_module(data_config, "val")
    dm.trainer.lightning_module = SimpleNamespace(device=device)
    dm.transforms_to_device()
    rows = {"snr": [], "true": []} | {name: [] for name in nets}
    for p in range(passes):
        for batch in dm.val_dataloader():
            batch = to_device(dm.on_before_batch_transfer(batch, 0), device)
            X, _, _, params = dm.on_after_batch_transfer(batch, 0)
            keep = ~torch.isnan(params["chirp_mass"])
            inputs = {"noise_removed": X[keep], "whitened": dm.last_whitened[keep]}
            rows["snr"].append(params["snr"][keep].cpu().numpy())
            rows["true"].append(params["chirp_mass"][keep].cpu().numpy())
            for name, (kind, net, y_mean, y_std) in nets.items():
                rows[name].append(chirp_mass(net, inputs[kind], y_mean, y_std, False))
        print(f"pass {p + 1}/{passes}, {sum(map(len, rows['snr']))} windows", flush=True)
    rows = {k: np.concatenate(v) for k, v in rows.items()}
    np.savez(out, **rows)

    print("share within 1% / 2% / 5% of the true chirp mass")
    for lo, hi in BANDS:
        k = (rows["snr"] >= lo) & (rows["snr"] < hi)
        line = f"  SNR {lo}-{hi}: n={k.sum():5d}"
        for name in nets:
            e = np.abs(rows[name][k] - rows["true"][k]) / rows["true"][k]
            line += f" | {name} " + " / ".join(f"{(e < t).mean():.3f}" for t in (0.01, 0.02, 0.05))
        print(line)


if __name__ == "__main__":
    main()
