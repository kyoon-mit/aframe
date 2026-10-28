"""Two layers of learned 4 s kernels, each followed by energy, max over time.

Layer 1 is ``LongConvEnergyNet``'s with ``kernel="time"`` and ``decimate=1``:
each detector's whitened strain is convolved by FFT with ``channels``
unit-norm complex kernels spanning the whole window, and the energies 2|y|^2
are summed over the detectors (the second one at its largest value within
the light travel time). Layer 2 convolves these ``channels`` energy series
with ``channels`` unit-norm complex kernels, each spanning all of them, and
takes the energy 2|y|^2 again. Its maximum over time, as log energies
standardized across channels, is mapped linearly to logits over a log
chirp-mass grid.
"""

import math

import torch
import torch.nn as nn

from train.experimental.long_conv.arch import grid_estimate


def unit_norm(k: torch.Tensor, dims: tuple[int, ...]) -> torch.Tensor:
    return k / k.abs().pow(2).sum(dims, keepdim=True).sqrt().clamp_min(1e-12)


class TwoLayerLongConvEnergyNet(nn.Module):
    """Args:
        num_ifos: detectors in the input.
        channels: learned kernels per layer.
        sample_rate: of the input, in Hz.
        kernel_length: input and kernel length, in seconds.
        coincidence: light travel time between detectors, in seconds.
        mc_min, mc_max, grid_step: log chirp-mass grid of the logits.
    """

    def __init__(
        self,
        num_ifos: int = 2,
        channels: int = 256,
        sample_rate: float = 2048.0,
        kernel_length: float = 4.0,
        coincidence: float = 0.01,
        mc_min: float = 0.85,
        mc_max: float = 2.75,
        grid_step: float = 0.005,
    ):
        super().__init__()
        self.length = L = int(kernel_length * sample_rate)
        self.coincidence = int(coincidence * sample_rate)
        self.samples1 = nn.Parameter(torch.randn(channels, L, 2) / math.sqrt(L))
        self.samples2 = nn.Parameter(torch.randn(channels, channels, L, 2) / math.sqrt(channels * L))
        grid = torch.arange(math.log(mc_min), math.log(mc_max) + 1e-9, grid_step)
        self.register_buffer("grid_log_mc", grid.float(), persistent=False)
        self.out = nn.Linear(channels, len(grid))

    def layer1(self, X: torch.Tensor) -> torch.Tensor:
        """(B, I, L) whitened strain -> (B, channels, L) network energy."""
        L, n = self.length, 2 * self.length
        k = unit_norm(torch.view_as_complex(self.samples1), (-1,))             # (C, L)
        y = torch.fft.ifft(torch.fft.fft(X.float(), n=n)[:, :, None] * torch.fft.fft(k, n=n), n=n)
        energy = 2 * y[..., :L].abs().pow(2)                                    # (B, I, C, L)
        net = energy[:, 0]
        if energy.shape[1] > 1:
            B, I, C, _ = energy.shape
            other = nn.functional.max_pool1d(
                energy[:, 1:].reshape(-1, 1, L), 2 * self.coincidence + 1, stride=1, padding=self.coincidence
            ).reshape(B, I - 1, C, L)
            net = net + other.sum(1)
        return net

    def layer2(self, E: torch.Tensor) -> torch.Tensor:
        """(B, channels, L) energies -> (B, channels, L) energy of kernels spanning all channels."""
        L, n = self.length, 2 * self.length
        k = unit_norm(torch.view_as_complex(self.samples2), (-2, -1))          # (C2, C1, L)
        spec = torch.fft.fft(E, n=n).permute(2, 0, 1)                          # (n, B, C1)
        kspec = torch.fft.fft(k, n=n).permute(2, 1, 0)                         # (n, C1, C2)
        y = torch.fft.ifft((spec @ kspec).permute(1, 2, 0), n=n)               # (B, C2, n)
        return 2 * y[..., :L].abs().pow(2)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Logits over the chirp-mass grid, (B, G)."""
        z = torch.log1p(self.layer2(self.layer1(X)).amax(-1))
        z = (z - z.mean(-1, keepdim=True)) / z.std(-1, keepdim=True).clamp_min(1e-6)
        return self.out(z)

    def estimate(self, logits: torch.Tensor):
        """Log chirp mass at the logits' peak, refined by a parabola, and the softmax spread."""
        return grid_estimate(logits, self.grid_log_mc)
