"""Save each test batch (whitened windows, truth, PSDs) with the model's
predictions on it, so a matched filter can run on the same windows."""

import os

import h5py
import numpy as np
import torch
from lightning.pytorch.callbacks import Callback


class DumpTest(Callback):
    def __init__(self, out_file: str):
        self.out_file = out_file
        self.rows = {}

    def on_test_start(self, trainer, pl_module):
        dm = trainer.datamodule
        original = dm.apply_transforms

        def recording(X, psds):
            dm._last_psds = psds
            return original(X, psds)

        dm.apply_transforms = recording

    def _add(self, key, value, double=False):
        value = value.detach().cpu()
        value = value.double() if double else value.float()
        self.rows.setdefault(key, []).append(value.numpy())

    def on_test_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx=0):
        X, X_clean, _y, params = batch
        psds = trainer.datamodule._last_psds
        with torch.no_grad():
            _den, out = pl_module(X)
            mean, var = pl_module._split(out)
            mean = mean * pl_module.y_std + pl_module.y_mean
            sigma = var.sqrt() * pl_module.y_std
        self._add("X", X)
        self._add("X_clean", X_clean)
        self._add("psds", psds, double=True)
        self._add("ml_mean", mean)
        self._add("ml_sigma", sigma)
        for k, v in params.items():
            self._add(f"params/{k}", v)

    def on_test_end(self, trainer, pl_module):
        os.makedirs(os.path.dirname(self.out_file), exist_ok=True)
        with h5py.File(self.out_file, "w") as f:
            for k, v in self.rows.items():
                f[k] = np.concatenate(v)
            f.attrs["param_names"] = list(pl_module.param_names)
        print(f"DUMPED {self.out_file}")
