from typing import Optional

import torch

from train.model.denoiser_ky import Denoiser


class ConditionedDenoiser(Denoiser):
    """``Denoiser`` that passes injection parameters to a FiLM architecture.

    The parameters named in ``cond_params`` are read from the batch,
    mapped linearly from ``cond_bounds`` onto [-1, 1], and passed to the
    architecture as a conditioning vector. Training batches are passed
    without conditioning with probability ``cond_dropout``; validation is
    never conditioned. Rows with no injection carry NaN parameters and are
    conditioned on zeros.

    Args:
        cond_params: batch parameter names to condition on, in order.
            Their number must equal the architecture's ``cond_dim``.
        cond_bounds: (low, high) per name in ``cond_params``, the range
            mapped onto [-1, 1]. Values outside it are clamped.
        cond_dropout: probability that a training batch is passed with no
            conditioning.
    """

    def __init__(
        self,
        *args,
        cond_params: list[str],
        cond_bounds: list[list[float]],
        cond_dropout: float = 0.5,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if not 0.0 <= cond_dropout <= 1.0:
            raise ValueError(
                f"cond_dropout must be in [0, 1], got {cond_dropout}"
            )
        if len(cond_bounds) != len(cond_params):
            raise ValueError(
                f"cond_bounds has {len(cond_bounds)} entries for "
                f"{len(cond_params)} cond_params"
            )
        for name, (low, high) in zip(cond_params, cond_bounds, strict=True):
            if not high > low:
                raise ValueError(f"{name}: need high > low, got {low}, {high}")
        self.cond_params = list(cond_params)
        self.cond_dropout = cond_dropout
        self.register_buffer(
            "cond_low", torch.tensor([b[0] for b in cond_bounds]).float()
        )
        self.register_buffer(
            "cond_high", torch.tensor([b[1] for b in cond_bounds]).float()
        )
        self.save_hyperparameters(
            "cond_params", "cond_bounds", "cond_dropout"
        )
        self._cond: Optional[torch.Tensor] = None

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if self.hparams.normalize_input:
            X = X / X.std(dim=-1, keepdim=True).clamp(min=1e-8)
        return self.model(X, self._cond)

    def _build_cond(self, params: dict, device) -> Optional[torch.Tensor]:
        if not self.training:
            return None
        if torch.rand(()) < self.cond_dropout:
            return None
        cond = torch.stack(
            [params[name].to(device).float() for name in self.cond_params],
            dim=-1,
        )
        span = self.cond_high - self.cond_low
        cond = 2.0 * (cond - self.cond_low) / span - 1.0
        cond = cond.clamp(-1.0, 1.0)
        return torch.nan_to_num(cond, nan=0.0)

    def _shared_step(self, batch, stage: str) -> torch.Tensor:
        self._cond = self._build_cond(batch[3], batch[0].device)
        try:
            return super()._shared_step(batch, stage)
        finally:
            self._cond = None
