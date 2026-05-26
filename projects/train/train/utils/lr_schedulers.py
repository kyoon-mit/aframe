import math
import torch


class WarmupCosineAnnealingWarmRestarts(torch.optim.lr_scheduler.LRScheduler):
    """Linear warmup followed by CosineAnnealingWarmRestarts (epoch-based)."""

    def __init__(
        self,
        optimizer,
        warmup_epochs,
        T_0,
        T_mult=2,
        eta_min=1e-8,
        warmup_start_factor=0.01,
        last_epoch=-1,
    ):
        self.warmup_epochs = warmup_epochs
        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.warmup_start_factor = warmup_start_factor
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        e = self.last_epoch
        if e < self.warmup_epochs:
            alpha = self.warmup_start_factor + (
                1.0 - self.warmup_start_factor
            ) * e / max(1, self.warmup_epochs)
            return [base_lr * alpha for base_lr in self.base_lrs]
        t = e - self.warmup_epochs
        T_cur, T_i = self._cosine_position(t)
        return [
            self.eta_min
            + (base_lr - self.eta_min)
            * (1 + math.cos(math.pi * T_cur / T_i))
            / 2
            for base_lr in self.base_lrs
        ]

    def _cosine_position(self, t):
        T_i = self.T_0
        while t >= T_i:
            t -= T_i
            T_i *= self.T_mult
        return t, T_i
