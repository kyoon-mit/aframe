"""Detection training for the networks in ``arch``."""

from typing import Callable, Optional

import numpy as np
import torch

from train.model.supervised import SupervisedAframe

Tensor = torch.Tensor


class MFLikeDetector(SupervisedAframe):
    """Signal-or-noise training with logistic loss (Yan et al. 2022,
    Eq. 49). Batches of whitened strain (B, ifos, n) are flattened to
    (B, ifos * n), so the arch's ``d_input`` is ``ifos * n``; the arch
    returns one score per segment.

    Args:
        lr_scheduler: callable mapping the optimizer to a learning-rate
            scheduler; ``None`` keeps aframe's one-cycle schedule.
        lr_scheduler_interval: "epoch" or "step".
        lr_scheduler_monitor: metric a plateau scheduler steps on.
        lr_scheduler_frequency: scheduler steps every this many intervals.
        template_file: ``.npz`` with ``templates`` (N, d_input); its first
            ``k`` rows replace the weights of the arch's first linear layer
            (k outputs) at start.
    """

    def __init__(
        self,
        arch: torch.nn.Module,
        *args,
        lr_scheduler: Optional[
            Callable[[torch.optim.Optimizer], object]
        ] = None,
        lr_scheduler_interval: str = "epoch",
        lr_scheduler_monitor: Optional[str] = None,
        lr_scheduler_frequency: int = 1,
        template_file: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__(arch, *args, **kwargs)
        self._lr_scheduler_factory = lr_scheduler
        if template_file is not None:
            self._load_templates(template_file)

    def _load_templates(self, path: str) -> None:
        linear = next(
            m for m in self.model.modules() if isinstance(m, torch.nn.Linear)
        )
        k = linear.weight.shape[0]
        templates = np.load(path)["templates"]
        if templates.shape[0] < k or templates.shape[1] != linear.in_features:
            raise ValueError(
                f"{path}: templates {templates.shape}, layer needs "
                f"({k}, {linear.in_features})"
            )
        with torch.no_grad():
            linear.weight.copy_(torch.from_numpy(templates[:k]))

    def forward(self, X: Tensor) -> Tensor:
        return self.model(X.flatten(1))

    def train_step(self, batch: tuple[Tensor, Tensor, dict]) -> Tensor:
        X, y, _ = batch
        y_hat = self(X)
        return torch.nn.functional.binary_cross_entropy_with_logits(
            y_hat, y.view_as(y_hat)
        )

    def configure_optimizers(self):
        if self._lr_scheduler_factory is None:
            return super().configure_optimizers()
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )
        config = {
            "scheduler": self._lr_scheduler_factory(optimizer),
            "interval": self.hparams.lr_scheduler_interval,
            "frequency": self.hparams.lr_scheduler_frequency,
        }
        if self.hparams.lr_scheduler_monitor is not None:
            config["monitor"] = self.hparams.lr_scheduler_monitor
        return {"optimizer": optimizer, "lr_scheduler": config}
