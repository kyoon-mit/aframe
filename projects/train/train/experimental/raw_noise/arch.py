"""Causal S4D noise model that reads raw strain."""

import torch
from torch import nn

from train.experimental.next_sample.arch import CausalS4Noise


class RawS4Noise(CausalS4Noise):
    """For (B, 1, L) raw strain divided by a constant, predicts each sample
    from the earlier ones. The network reads the earlier samples as they
    are and predicts the change from the previous sample, so the returned
    mean is the previous sample plus the network's first output.
    """

    def forward(self, x: torch.Tensor):
        previous = nn.functional.pad(x[..., :-1], (1, 0))
        h = self.encoder(previous)
        for layer, norm in zip(self.layers, self.norms, strict=True):
            z = norm(h.transpose(-1, -2)).transpose(-1, -2)
            h = h + layer(z)
        out = self.decoder(h)
        k = self.num_components
        return previous[:, 0] + out[:, 0], out[:, 1 : 1 + k], out[:, 1 + k :]
