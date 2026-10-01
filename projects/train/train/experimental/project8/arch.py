"""Regressor-only architecture with the den_reg task's interface."""

from typing import Optional

import torch

from architectures import Architecture
from architectures.networks.s4d_variants import S4ModelPooled


class RegressionTimeDomainS4Pooled(Architecture):
    """The den_reg regressor alone, reading the noisy input directly.

    Same pooled S4D regressor as ``RegressionTimeDomainS4DenoiseRegressNorm``.
    ``forward`` returns ``(X, param_estimates)``: the input stands in for the
    denoised strain, so the denoise-and-regress tasks run it unchanged, with
    ``lambda_denoise: 0``.
    """

    def __init__(
        self,
        num_ifos: int,
        d_output: int = 2,
        regressor_d_model: int = 64,
        regressor_d_state: int = 64,
        regressor_n_layers: int = 4,
        regressor_dropout: float = 0.2,
        prenorm: bool = False,
        num_groups: Optional[int] = None,
        dt_min: float = 1e-3,
        dt_max: float = 0.1,
    ) -> None:
        super().__init__()
        self.regressor = S4ModelPooled(
            d_input=num_ifos,
            d_output=d_output,
            d_model=regressor_d_model,
            d_state=regressor_d_state,
            n_layers=regressor_n_layers,
            dropout=regressor_dropout,
            prenorm=prenorm,
            num_groups=num_groups,
            dt_min=dt_min,
            dt_max=dt_max,
        )

    def forward(self, X: torch.Tensor):
        return X, self.regressor(X)
