"""Score network of SLIC: PyTorch port of NCSNpp1D from scoregen_jax
(models/model.py, models/layers.py, models/up_or_down_sampling_1d.py).

Layout is (batch, channels, samples). Defaults are the paper's
architectures/config_real_white_noise_both.json.
"""

import math
from functools import partial
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from train.experimental.slic.sde import VESDE

SQRT2 = 1.41421356


def variance_scaling_(weight: torch.Tensor, scale: float = 1.0) -> None:
    """JAX variance_scaling(scale, "fan_avg", "uniform"), in place."""
    scale = 1e-10 if scale == 0 else scale
    receptive = weight[0, 0].numel() if weight.dim() > 2 else 1
    fan_in = weight.shape[1] * receptive
    fan_out = weight.shape[0] * receptive
    limit = math.sqrt(3 * scale / ((fan_in + fan_out) / 2))
    with torch.no_grad():
        weight.uniform_(-limit, limit)


def conv(in_ch: int, out_ch: int, size: int, init_scale: float = 1.0):
    """Same-padded convolution, zero bias."""
    layer = nn.Conv1d(in_ch, out_ch, size, padding=size // 2)
    variance_scaling_(layer.weight, init_scale)
    nn.init.zeros_(layer.bias)
    return layer


def dense(in_features: int, out_features: int):
    layer = nn.Linear(in_features, out_features)
    variance_scaling_(layer.weight)
    nn.init.zeros_(layer.bias)
    return layer


def group_norm(channels: int):
    return nn.GroupNorm(min(channels // 4, 32), channels, eps=1e-6)


def fir_kernel(kernel: Sequence[int], gain: float) -> torch.Tensor:
    """Kernel normalized to sum 1, times gain, in float32."""
    k = np.asarray(kernel, dtype=np.float32)
    k /= np.sum(k)
    return torch.from_numpy(k * gain)


def upfirdn(
    x: torch.Tensor,
    kernel: torch.Tensor,
    up: int = 1,
    down: int = 1,
    pad0: int = 0,
    pad1: int = 0,
) -> torch.Tensor:
    """Insert up - 1 zeros after each sample, pad (crop if negative), filter
    each channel with ``kernel``, keep every ``down``-th sample."""
    B, C, H = x.shape
    if up > 1:
        x = F.pad(x[..., None], (0, up - 1)).reshape(B, C, H * up)
    x = F.pad(x, (max(pad0, 0), max(pad1, 0)))
    x = x[..., max(-pad0, 0) : x.shape[-1] - max(-pad1, 0)]
    weight = kernel.flip(0).to(x.dtype).repeat(C, 1, 1)
    return F.conv1d(x, weight, stride=down, groups=C)


def upsample(x: torch.Tensor, kernel: torch.Tensor, factor: int):
    """``kernel`` already scaled by ``factor``."""
    p = len(kernel) - factor
    return upfirdn(
        x, kernel, up=factor, pad0=(p + 1) // 2 + factor - 1, pad1=p // 2
    )


def downsample(x: torch.Tensor, kernel: torch.Tensor, factor: int):
    p = len(kernel) - factor
    return upfirdn(x, kernel, down=factor, pad0=(p + 1) // 2, pad1=p // 2)


class ResnetBlock(nn.Module):
    """ResnetBlockBigGANpp with FIR resampling and skip rescale."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int | None = None,
        temb_dim: int = 64,
        up: bool = False,
        down: bool = False,
        factor: int = 2,
        dropout: float = 0.1,
        kernel: Sequence[int] = (1, 3, 1),
        init_scale: float = 0.0,
    ):
        super().__init__()
        out_ch = out_ch or in_ch
        self.up, self.down, self.factor = up, down, factor
        self.norm_0 = group_norm(in_ch)
        self.conv_0 = conv(in_ch, out_ch, 3)
        self.dense_0 = dense(temb_dim, out_ch)
        self.norm_1 = group_norm(out_ch)
        self.dropout = nn.Dropout(dropout)
        self.conv_1 = conv(out_ch, out_ch, 3, init_scale)
        self.skip = None
        if in_ch != out_ch or up or down:
            self.skip = conv(in_ch, out_ch, 1)
        if up or down:
            gain = factor if up else 1
            self.register_buffer(
                "kernel", fir_kernel(kernel, gain), persistent=False
            )

    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.norm_0(x))
        if self.up:
            h = upsample(h, self.kernel, self.factor)
            x = upsample(x, self.kernel, self.factor)
        elif self.down:
            h = downsample(h, self.kernel, self.factor)
            x = downsample(x, self.kernel, self.factor)
        h = self.conv_0(h)
        h = h + self.dense_0(F.silu(temb))[:, :, None]
        h = F.silu(self.norm_1(h))
        h = self.dropout(h)
        h = self.conv_1(h)
        if self.skip is not None:
            x = self.skip(x)
        return (x + h) / SQRT2


class ScoreNet(nn.Module):
    """NCSNpp1D: U-Net of ResNet blocks with a Gaussian Fourier embedding of
    the diffusion time t.

    ``forward(x, t)`` returns sigma(t) times the score of the noised data;
    ``score(x, t)`` returns the score. Input length must be divisible by
    the product of ``res_mult``.
    """

    def __init__(
        self,
        nf: int = 16,
        ch_mult: Sequence[int] = (4, 4, 2, 2, 2, 1, 1, 1),
        res_mult: Sequence[int] = (2, 2, 2, 4, 4, 4, 4),
        num_res_blocks: int = 2,
        dropout: float = 0.1,
        fir_kernel: Sequence[int] = (1, 3, 1),
        init_scale: float = 0.01,
        fourier_scale: float = 16.0,
        sigma_min: float = 1e-3,
        sigma_max: float = 1e4,
        num_channels: int = 1,
    ):
        super().__init__()
        self.sde = VESDE(sigma_min, sigma_max)
        temb_dim = 4 * nf
        # Fourier weights are fixed, not trained
        self.register_buffer(
            "fourier_weight", torch.randn(nf // 2) * fourier_scale
        )
        self.temb_0 = dense(nf, temb_dim)
        self.temb_1 = dense(temb_dim, temb_dim)

        block = partial(
            ResnetBlock,
            temb_dim=temb_dim,
            dropout=dropout,
            kernel=fir_kernel,
            init_scale=init_scale,
        )
        self.stem = conv(num_channels, nf, 3)
        channels = [nf]
        c = nf
        self.down_blocks = nn.ModuleList()
        for i, mult in enumerate(ch_mult):
            for _ in range(num_res_blocks):
                self.down_blocks.append(block(c, nf * mult))
                c = nf * mult
                channels.append(c)
            if i != len(ch_mult) - 1:
                self.down_blocks.append(
                    block(c, down=True, factor=res_mult[i])
                )
                channels.append(c)

        self.middle_blocks = nn.ModuleList([block(c), block(c)])

        self.up_blocks = nn.ModuleList()
        for i in reversed(range(len(ch_mult))):
            for _ in range(num_res_blocks + 1):
                out_ch = nf * ch_mult[i]
                self.up_blocks.append(block(c + channels.pop(), out_ch))
                c = out_ch
            if i != 0:
                self.up_blocks.append(
                    block(c, up=True, factor=res_mult[i - 1])
                )

        self.head_norm = group_norm(c)
        self.head = conv(c, num_channels, 3, init_scale=1.0)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        angles = t[:, None] * self.fourier_weight[None, :] * 2 * math.pi
        temb = torch.cat([angles.sin(), angles.cos()], dim=1)
        temb = self.temb_1(F.silu(self.temb_0(temb)))

        sigma = self.sde.sigma(t)[:, None, None]
        x = x * (1 / torch.sqrt(sigma**2 + 1.0))

        hs = [self.stem(x)]
        for block in self.down_blocks:
            hs.append(block(hs[-1], temb))
        h = hs[-1]
        for block in self.middle_blocks:
            h = block(h, temb)
        for block in self.up_blocks:
            if block.up:
                h = block(h, temb)
            else:
                h = block(torch.cat([h, hs.pop()], dim=1), temb)

        h = self.head(F.silu(self.head_norm(h)))
        return h * sigma / torch.sqrt(sigma**2 + 1.0)

    def score(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self(x, t) / self.sde.sigma(t)[:, None, None]
