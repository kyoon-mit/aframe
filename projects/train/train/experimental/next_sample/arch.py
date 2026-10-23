"""Causal S4D that outputs a scale mixture of Gaussians per sample."""

import torch
from ml4gw.nn.ssm.s4d import S4D
from torch import nn


class CausalS4Noise(nn.Module):
    """For (B, 1, L) whitened noise, predicts each sample from the earlier
    ones: output t sees inputs 0..t-1 only. The network reads asinh of the
    earlier samples; its outputs describe the samples themselves.

    Pointwise encoder, ``n_layers`` pre-norm residual S4D blocks (S4D's
    convolution is causal; LayerNorm acts per sample), pointwise decoder.
    Returns the shared mean (B, L), log spreads (B, K, L) and mixture
    logits (B, K, L).
    """

    def __init__(
        self,
        num_components: int = 3,
        d_model: int = 64,
        d_state: int = 32,
        n_layers: int = 4,
        dropout: float = 0.0,
        dt_min: float = 1e-4,
        dt_max: float = 10.0,
    ):
        super().__init__()
        self.num_components = num_components
        self.encoder = nn.Conv1d(1, d_model, kernel_size=1)
        self.layers = nn.ModuleList(
            S4D(
                d_model,
                d_state=d_state,
                dropout=dropout,
                transposed=True,
                dt_min=dt_min,
                dt_max=dt_max,
            )
            for _ in range(n_layers)
        )
        self.norms = nn.ModuleList(nn.LayerNorm(d_model) for _ in range(n_layers))
        self.decoder = nn.Conv1d(d_model, 1 + 2 * num_components, kernel_size=1)

    def forward(self, x: torch.Tensor):
        x = torch.asinh(nn.functional.pad(x[..., :-1], (1, 0)))
        h = self.encoder(x)
        for layer, norm in zip(self.layers, self.norms, strict=True):
            z = norm(h.transpose(-1, -2)).transpose(-1, -2)
            h = h + layer(z)
        out = self.decoder(h)
        k = self.num_components
        return out[:, 0], out[:, 1 : 1 + k], out[:, 1 + k :]
