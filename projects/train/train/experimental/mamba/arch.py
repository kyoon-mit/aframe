"""Mamba-2 in place of S4D in the den-reg: a Mamba-2 denoiser feeding a
Mamba-2 regressor, block for block as ``RegressionTimeDomainS4DenoiseRegressNorm``
(encoder, residual blocks with GroupNorm or LayerNorm before or after, channel
dropout, linear decoder; the regressor averages over time)."""

from typing import Optional

import torch
import torch.nn as nn
from architectures.networks.lowband import LowBandSmoother
from architectures.networks.s4d_variants import S4ModelDenoiseRegress
from architectures.regression import RegressionArchitecture
from architectures.supervised import SupervisedArchitecture
from ml4gw.nn.norm import GroupNorm1D
from torch.utils.checkpoint import checkpoint

from train.experimental.mamba.ssd import Mamba2


class MambaBlocks(nn.Module):
    """Encoder and residual Mamba-2 blocks: (B, d_input, L) -> (B, L, d_model).
    With ``checkpointing`` each block is recomputed in the backward pass
    instead of keeping its activations; with ``compile`` each Mamba-2 layer
    goes through ``torch.compile``."""

    def __init__(self, d_input: int, d_output: int, d_model: int = 128, d_state: int = 64,
                 n_layers: int = 4, dropout: float = 0.2, prenorm: bool = False,
                 num_groups: Optional[int] = None, expand: int = 2, headdim: int = 64,
                 d_conv: int = 4, chunk: int = 64, dt_min: float = 1e-3, dt_max: float = 0.1,
                 checkpointing: bool = True, compile: bool = False):
        super().__init__()
        self.encoder = nn.Linear(d_input, d_model)
        self.layers = nn.ModuleList([
            Mamba2(d_model, d_state=d_state, expand=expand, headdim=headdim, d_conv=d_conv,
                   chunk=chunk, dt_min=dt_min, dt_max=dt_max)
            for _ in range(n_layers)
        ])
        self._groupnorm = num_groups is not None
        self.norms = nn.ModuleList([
            GroupNorm1D(num_channels=d_model, num_groups=num_groups) if self._groupnorm
            else nn.LayerNorm(d_model)
            for _ in range(n_layers)
        ])
        self.dropouts = nn.ModuleList([nn.Dropout1d(dropout) for _ in range(n_layers)])
        self.decoder = nn.Linear(d_model, d_output)
        self.prenorm = prenorm
        self.checkpointing = checkpointing
        if compile:
            for layer in self.layers:
                layer.forward = torch.compile(layer.forward)

    def _norm(self, norm: nn.Module, x: torch.Tensor) -> torch.Tensor:
        # x is (B, L, d_model); GroupNorm1D takes (B, d_model, L)
        if self._groupnorm:
            return norm(x.transpose(-1, -2)).transpose(-1, -2)
        return norm(x)

    def _block(self, x, layer, norm, dropout):
        if self.prenorm:
            z = layer(self._norm(norm, x))
            return x + dropout(z.transpose(-1, -2)).transpose(-1, -2)
        z = dropout(layer(x).transpose(-1, -2)).transpose(-1, -2)
        return self._norm(norm, z + x)

    def blocks(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x.transpose(-1, -2))
        for layer, norm, dropout in zip(self.layers, self.norms, self.dropouts, strict=True):
            if self.checkpointing and self.training:
                x = checkpoint(self._block, x, layer, norm, dropout, use_reentrant=False)
            else:
                x = self._block(x, layer, norm, dropout)
        return x


class MambaSeq2Seq(MambaBlocks):
    """(B, d_input, L) -> (B, d_output, L)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.blocks(x)).transpose(-1, -2)


class MambaPooled(MambaBlocks):
    """(B, d_input, L) -> (B, d_output), mean over time before the decoder."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.blocks(x).mean(dim=1))


class MambaDenoiseRegress(RegressionArchitecture, SupervisedArchitecture):
    """Mamba-2 denoiser feeding a Mamba-2 regressor, with the arguments of
    ``RegressionTimeDomainS4DenoiseRegressNorm`` plus the Mamba-2 ones
    (``expand``, ``headdim``, ``d_conv``, ``chunk``) and ``checkpointing``
    and ``compile``. ``forward`` returns
    ``(x_denoised, param_estimates)``."""

    def __init__(
        self,
        num_ifos: int,
        d_output: int = 2,
        denoiser_d_model: int = 128,
        denoiser_d_state: int = 64,
        denoiser_n_layers: int = 6,
        denoiser_dropout: float = 0.2,
        regressor_d_model: int = 128,
        regressor_d_state: int = 64,
        regressor_n_layers: int = 6,
        regressor_dropout: float = 0.2,
        prenorm: bool = True,
        num_groups: Optional[int] = 32,
        expand: int = 2,
        headdim: int = 64,
        d_conv: int = 4,
        chunk: int = 64,
        dt_min: float = 1e-3,
        dt_max: float = 0.1,
        checkpointing: bool = True,
        compile: bool = False,
        detach_denoiser: bool = False,
        smooth_below: Optional[float] = None,
        smooth_coeffs: int = 4,
        # linked from the CLI
        sample_rate: Optional[float] = None,
        kernel_length: Optional[float] = None,
    ) -> None:
        super().__init__()
        smoother = None
        if smooth_below is not None:
            if sample_rate is None or kernel_length is None:
                raise ValueError("smooth_below needs sample_rate and kernel_length")
            smoother = LowBandSmoother(
                int(kernel_length * sample_rate), sample_rate, smooth_below, smooth_coeffs)
        shared = dict(prenorm=prenorm, num_groups=num_groups, expand=expand, headdim=headdim,
                      d_conv=d_conv, chunk=chunk, dt_min=dt_min, dt_max=dt_max,
                      checkpointing=checkpointing, compile=compile)
        self.model = S4ModelDenoiseRegress(
            denoiser=MambaSeq2Seq,
            regressor=MambaPooled,
            denoiser_params=dict(d_input=num_ifos, d_output=num_ifos, d_model=denoiser_d_model,
                                 d_state=denoiser_d_state, n_layers=denoiser_n_layers,
                                 dropout=denoiser_dropout, **shared),
            regressor_params=dict(d_input=num_ifos, d_output=d_output, d_model=regressor_d_model,
                                  d_state=regressor_d_state, n_layers=regressor_n_layers,
                                  dropout=regressor_dropout, **shared),
            detach_denoiser=detach_denoiser,
            smoother=smoother,
        )

    def forward(self, X: torch.Tensor):
        return self.model(X)
