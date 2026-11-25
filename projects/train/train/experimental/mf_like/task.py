"""Detection training for the networks in ``arch``."""

import torch

from train.model.supervised import SupervisedAframe

Tensor = torch.Tensor


class MFLikeDetector(SupervisedAframe):
    """Signal-or-noise training with logistic loss (Yan et al. 2022,
    Eq. 49). Batches of whitened strain (B, ifos, n) are flattened to
    (B, ifos * n), so the arch's ``d_input`` is ``ifos * n``; the arch
    returns one score per segment.
    """

    def __init__(self, arch: torch.nn.Module, *args, **kwargs) -> None:
        super().__init__(arch, *args, **kwargs)

    def forward(self, X: Tensor) -> Tensor:
        return self.model(X.flatten(1))

    def train_step(self, batch: tuple[Tensor, Tensor, dict]) -> Tensor:
        X, y, _ = batch
        y_hat = self(X)
        return torch.nn.functional.binary_cross_entropy_with_logits(
            y_hat, y.view_as(y_hat)
        )
