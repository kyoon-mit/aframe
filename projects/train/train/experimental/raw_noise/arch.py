"""Causal S4D noise model that reads raw strain."""

import math

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


class LinearS4Noise(nn.Module):
    """For (B, 1, L) raw strain divided by a constant, predicts each sample
    from the earlier ones in two parts: a fixed causal linear filter over the
    previous ``len(taps)`` samples, run in float64, and a ``CausalS4Noise``
    that reads the filter's errors (divided by their training spread) and
    predicts what is left of them, with a spread per sample.

    ``taps_file`` is an ``.npz`` from ``fit_taps``: ``taps`` (N,), the weight
    of x_{t-1} first, and ``std``, the spread of the filter's errors on the
    training data. Outputs are in units of the input. The first N samples,
    where the filter has not seen its full past, are not meaningful.
    """

    def __init__(self, taps_file: str, **kwargs):
        super().__init__()
        import numpy as np

        data = np.load(taps_file)
        self.register_buffer("taps", torch.as_tensor(data["taps"], dtype=torch.float64))
        self.std = float(data["std"])
        self.residual = CausalS4Noise(**kwargs)
        self.num_components = self.residual.num_components

    def linear_prediction(self, x: torch.Tensor) -> torch.Tensor:
        """sum_k taps[k-1] x_{t-k}, k = 1..N, in float64; zero past before 0."""
        x = x.double()
        L, N = x.size(-1), self.taps.numel()
        n = L + N
        kernel = nn.functional.pad(self.taps, (1, 0))
        out = torch.fft.irfft(
            torch.fft.rfft(x, n=n) * torch.fft.rfft(kernel, n=n), n=n
        )
        return out[..., :L]

    def forward(self, x: torch.Tensor):
        linear = self.linear_prediction(x)
        errors = ((x.double() - linear) / self.std).float()
        mean, log_sigma, logits = self.residual(errors)
        mean = (linear[:, 0] + self.std * mean.double()).float()
        return mean, log_sigma + math.log(self.std), logits
