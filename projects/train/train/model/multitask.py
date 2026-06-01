from typing import List

import torch
import torch.nn.functional as F
from architectures import Architecture

from train.metrics import TimeSlideAUROC
from train.model.supervised import SupervisedAframe

Tensor = torch.Tensor


class SupervisedMultiTaskAframe(SupervisedAframe):
    """
    Multi-task model that jointly optimizes binary classification
    and injection parameter regression.

    The architecture's forward pass must return a tuple
    ``(logits, param_estimates)`` where ``logits`` has shape ``(N, 1)``
    and ``param_estimates`` has shape ``(N, len(param_names))``.

    Validation uses the standard AUROC metric on the classification head,
    so the existing validation infrastructure is fully reused.

    Args:
        arch:
            Architecture whose forward pass returns (logits, param_estimates).
        param_names:
            Ordered list of parameter names to regress on. Must match the
            output ordering of the architecture's regression head.
        regression_weight:
            Scalar multiplier applied to the regression loss before summing
            with the classification loss.
    """

    def __init__(
        self,
        arch: Architecture,
        metric: TimeSlideAUROC,
        param_names: List[str],
        *args,
        regression_weight: float = 1.0,
        **kwargs,
    ):
        super().__init__(arch, metric, *args, **kwargs)
        self.param_names = param_names
        self.regression_weight = regression_weight

    def score(self, X: Tensor) -> Tensor:
        logits, _ = self(X)
        return logits

    def train_step(self, batch):
        X, y, params = batch
        logits, param_estimates = self(X)

        clf_loss = F.binary_cross_entropy_with_logits(logits, y)

        targets = torch.stack([params[k] for k in self.param_names], dim=1)
        mask = ~torch.isnan(targets).any(dim=1)
        reg_loss = (
            F.mse_loss(param_estimates[mask], targets[mask])
            if mask.any()
            else torch.zeros(1, device=X.device)
        )

        return {"classification_loss": clf_loss, "regression_loss": reg_loss}

    def compute_loss_fn(self, classification_loss, regression_loss):
        return classification_loss + self.regression_weight * regression_loss


class MultiTaskAframeS4D(SupervisedMultiTaskAframe):
    """Multi-task S4D model with S4D-aware optimizer (warmup + cosine).

    The architecture must return ``(logits, param_estimates)`` as required
    by ``SupervisedMultiTaskAframe``.  SSM parameters tagged with ``._optim``
    get their own learning-rate group; all other parameters use ``base_lr``.
    """

    def __init__(
        self,
        arch: Architecture,
        metric: TimeSlideAUROC,
        param_names: List[str],
        base_lr: float = 1e-4,
        weight_decay: float = 0.0,
        warmup_steps: int = 1000,
        regression_weight: float = 1.0,
        normalize_input: bool = False,
    ):
        super().__init__(
            arch=arch,
            metric=metric,
            param_names=param_names,
            learning_rate=base_lr,
            pct_lr_ramp=0.0,
            weight_decay=weight_decay,
            regression_weight=regression_weight,
        )
        self.warmup_steps = warmup_steps
        self.normalize_input = normalize_input
        self.save_hyperparameters(ignore=["arch", "metric"])

    def _prepare_input(self, X: Tensor) -> Tensor:
        if self.normalize_input:
            return X / X.std(dim=-1, keepdim=True).clamp(min=1e-8)
        return X

    def score(self, X: Tensor) -> Tensor:
        logits, _ = self(self._prepare_input(X))
        return logits

    def train_step(self, batch):
        X, y, params = batch
        X = self._prepare_input(X)
        logits, param_estimates = self(X)

        clf_loss = F.binary_cross_entropy_with_logits(logits, y)

        targets = torch.stack([params[k] for k in self.param_names], dim=1)
        mask = ~torch.isnan(targets).any(dim=1)
        reg_loss = (
            F.mse_loss(param_estimates[mask], targets[mask])
            if mask.any()
            else torch.zeros(1, device=X.device)
        )

        return {"classification_loss": clf_loss, "regression_loss": reg_loss}

    def configure_optimizers(self):
        all_params = list(self.model.parameters())
        default_params = [p for p in all_params if not hasattr(p, "_optim")]
        optim_params = [p for p in all_params if hasattr(p, "_optim")]
        param_groups = [
            {
                "params": default_params,
                "lr": self.hparams.base_lr,
                "weight_decay": self.hparams.weight_decay,
            }
        ]
        unique_hps = [
            dict(s)
            for s in sorted(
                {frozenset(p._optim.items()) for p in optim_params}
            )
        ]
        for ohp in unique_hps:
            group = {
                "params": [p for p in optim_params if p._optim == ohp],
                "lr": ohp.get("lr", self.hparams.base_lr),
            }
            group.update(ohp)
            param_groups.append(group)
        optimizer = torch.optim.AdamW(param_groups)
        total_steps = self.trainer.estimated_stepping_batches
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1e-2,
            end_factor=1.0,
            total_iters=self.warmup_steps,
        )
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, total_steps - self.warmup_steps),
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup, cosine],
            milestones=[self.warmup_steps],
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }
