"""Starting templates drawn from the training signal priors.

Runs the config's training pipeline with a signal in every window and keeps
the signal-only part of each window, whitened with that window's own
spectrum, cropped as the network sees it and scaled to unit length.

usage: python -m train.experimental.mf_like.templates <config> <count> <out.npz> [seed]
"""

import sys
from types import SimpleNamespace

import lightning.pytorch as pl
import numpy as np
import torch
from lightning.fabric.utilities.apply_func import move_data_to_device

from train.experimental.slic.dump_windows import build_data_module


def make_templates(config: str, count: int, seed: int = 0) -> np.ndarray:
    """(count, ifos * n) unit-length whitened signals."""
    pl.seed_everything(seed)
    dev = torch.device("cuda")
    dm = build_data_module(config, "train")
    dm.trainer.training = True
    dm.trainer.lightning_module = SimpleNamespace(device=dev)
    dm.hparams.waveform_prob = 1.0
    dm.transforms_to_device()

    psds = {}
    apply_transforms = dm.apply_transforms

    def record(X, p):
        psds["last"] = p
        return apply_transforms(X, p)

    dm.apply_transforms = record

    out, total = [], 0
    for batch in dm.train_dataloader():
        batch = dm.on_before_batch_transfer(batch, 0)
        batch = move_data_to_device(batch, dev)
        _, y, _ = dm.on_after_batch_transfer(batch, 0)
        if not (y[:, 0] > 0).all():
            raise ValueError("expected a signal in every window")
        clean = dm.whitener(dm._clean_signal, psds["last"]).flatten(1)
        out.append((clean / clean.norm(dim=1, keepdim=True)).cpu())
        total += len(clean)
        if total >= count:
            break
    return torch.cat(out)[:count].float().numpy()


if __name__ == "__main__":
    config, count, path = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    seed = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    templates = make_templates(config, count, seed)
    np.savez(path, templates=templates, config=config, seed=seed)
    print(f"{path}: {templates.shape}")
