"""Training tasks: a judge (tell two sources apart) and the generator (diffusion)."""

import json
import math
from pathlib import Path

import lightning.pytorch as pl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Task(pl.LightningModule):
    """AdamW with the learning rate falling along a cosine to zero over training.
    Subclasses save ``learning_rate`` and ``weight_decay`` with their other
    settings; the networks are saved with them too, so ``load_from_checkpoint``
    can rebuild a task from its checkpoint alone."""

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), self.hparams.learning_rate, weight_decay=self.hparams.weight_decay
        )
        schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.trainer.estimated_stepping_batches)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": schedule, "interval": "step"}}


class JudgeTask(Task):
    """Trains a judge to tell windows of the data's first source (label 1) from
    its second source (label 0).

    The test reports the accuracy on held-out windows with its standard error
    (50% means the judge cannot tell the sources apart), and saves the
    ``keep_windows`` windows of each source that the judge was surest about:
    ``judge_test.json`` and ``judge_test_windows.npz`` in the run folder.
    """

    def __init__(self, judge: nn.Module, learning_rate: float, weight_decay: float = 0.0, keep_windows: int = 16):
        super().__init__()
        self.save_hyperparameters()
        self.judge = judge

    def scored(self, batch, stage: str):
        windows, labels = batch
        belief = self.judge(windows)
        loss = F.binary_cross_entropy_with_logits(belief, labels)
        accuracy = ((belief > 0) == (labels > 0.5)).float().mean()
        self.log(f"{stage}/loss", loss, on_step=stage == "train", on_epoch=True, prog_bar=True)
        self.log(f"{stage}/accuracy", accuracy, on_step=stage == "train", on_epoch=True, prog_bar=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self.scored(batch, "train")

    def validation_step(self, batch, batch_idx):
        self.scored(batch, "val")

    def on_test_epoch_start(self):
        self.right, self.total = 0, 0
        self.surest = {1: None, 0: None}

    def test_step(self, batch, batch_idx):
        windows, labels = batch
        belief = self.judge(windows)
        self.right += ((belief > 0) == (labels > 0.5)).sum().item()
        self.total += len(labels)
        for label in (1, 0):
            rows = labels == label
            # sureness: how strongly the judge points at the true source
            sureness = belief[rows] if label == 1 else -belief[rows]
            self.keep_surest(label, windows[rows], sureness)

    def keep_surest(self, label: int, windows: torch.Tensor, sureness: torch.Tensor):
        if self.surest[label] is not None:
            kept_windows, kept_sureness = self.surest[label]
            windows, sureness = torch.cat([kept_windows, windows]), torch.cat([kept_sureness, sureness])
        top = sureness.topk(min(self.hparams.keep_windows, len(sureness))).indices
        self.surest[label] = (windows[top], sureness[top])

    def on_test_epoch_end(self):
        accuracy = self.right / self.total
        standard_error = math.sqrt(accuracy * (1 - accuracy) / self.total)
        self.log("test/accuracy", accuracy)
        self.log("test/accuracy_standard_error", standard_error)
        folder = Path(self.trainer.log_dir or self.trainer.default_root_dir)
        folder.mkdir(parents=True, exist_ok=True)
        with open(folder / "judge_test.json", "w") as f:
            json.dump({"accuracy": accuracy, "standard_error": standard_error, "windows": self.total}, f, indent=1)
        np.savez(
            folder / "judge_test_windows.npz",
            first_windows=self.surest[1][0].cpu().numpy(), first_sureness=self.surest[1][1].cpu().numpy(),
            second_windows=self.surest[0][0].cpu().numpy(), second_sureness=self.surest[0][1].cpu().numpy(),
        )


class GeneratorTask(Task):
    """Trains the generator, a ``diffusion.Diffusion`` around a
    ``diffusion.StaticRemoverNet``, on real windows (data with only a first
    source)."""

    def __init__(self, diffusion: nn.Module, learning_rate: float, weight_decay: float = 0.0):
        super().__init__()
        self.save_hyperparameters()
        self.diffusion = diffusion

    def training_step(self, windows, batch_idx):
        loss = self.diffusion.loss(windows)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, windows, batch_idx):
        self.log("val/loss", self.diffusion.loss(windows), on_epoch=True, prog_bar=True)
