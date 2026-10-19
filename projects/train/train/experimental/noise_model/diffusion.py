"""The noise generator: a diffusion model built from S4D layers.

Training buries real whitened windows under added Gaussian static of a random
strength and teaches the network to recover the window. Generating starts from
pure static and removes it step by step. How the network's input and output
are scaled with the strength, which strengths training draws, and the steps of
generation follow Karras et al. 2022, "Elucidating the design space of
diffusion-based generative models" (arXiv:2206.00364). Their settings are for
data of spread 0.5; whitened noise has spread 1, so the defaults here are
their settings with every strength doubled (and the log-strength mean raised
by ln 2), which is the same problem at twice the scale.

"Static" means only the Gaussian noise added here, never the detector noise,
which is the data.
"""

import math

import torch
import torch.nn as nn

from train.experimental.noise_model.layers import BothWaysS4D


class Level(nn.Module):
    """S4D blocks at one time resolution, each told the static strength."""

    def __init__(self, width: int, blocks: int, d_state: int, features: int, dt_min: float, dt_max: float):
        super().__init__()
        self.blocks = nn.ModuleList([BothWaysS4D(width, d_state, dt_min, dt_max) for _ in range(blocks)])
        self.to_scale_shift = nn.ModuleList([nn.Linear(features, 2 * width) for _ in range(blocks)])

    def forward(self, h: torch.Tensor, strength_features: torch.Tensor) -> torch.Tensor:
        for block, to_scale_shift in zip(self.blocks, self.to_scale_shift):
            scale, shift = to_scale_shift(strength_features)[..., None].chunk(2, dim=1)
            h = block(h, scale, shift)
        return h


class StaticRemoverNet(nn.Module):
    """Takes a buried window and log(strength) / 4; returns the network's part of
    the guess of the clean window.

    The window is read at several time resolutions, each ``stride`` times
    coarser than the last, with ``widths[i]`` channels at resolution ``i``; on
    the way back up, each resolution adds what it saw on the way down. The
    window length must be divisible by ``stride ** (len(widths) - 1)``.
    """

    def __init__(
        self, num_ifos: int = 2, widths: tuple[int, ...] = (64, 128, 256), blocks: int = 2, stride: int = 4,
        d_state: int = 64, features: int = 128, dt_min: float = 0.001, dt_max: float = 0.1,
    ):
        super().__init__()
        # sine and cosine of the strength at frequencies spread evenly in log
        # over four decades, then a small network
        half = features // 2
        self.register_buffer("frequencies", torch.exp(-math.log(1e4) * torch.arange(half) / half), persistent=False)
        self.strength_net = nn.Sequential(nn.Linear(2 * half, features), nn.GELU(), nn.Linear(features, features))

        self.inp = nn.Conv1d(num_ifos, widths[0], 1)
        self.down_levels = nn.ModuleList([Level(w, blocks, d_state, features, dt_min, dt_max) for w in widths])
        self.downs = nn.ModuleList([nn.Conv1d(a, b, stride, stride=stride) for a, b in zip(widths[:-1], widths[1:])])
        self.ups = nn.ModuleList(
            [nn.ConvTranspose1d(b, a, stride, stride=stride) for a, b in zip(widths[:-1], widths[1:])]
        )
        self.up_levels = nn.ModuleList([Level(w, blocks, d_state, features, dt_min, dt_max) for w in widths[:-1]])
        self.out = nn.Conv1d(widths[0], num_ifos, 1)
        # zero output at the start: the guess begins as the scaled buried window
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, buried: torch.Tensor, scaled_log_strength: torch.Tensor) -> torch.Tensor:
        angles = scaled_log_strength[:, None] * self.frequencies
        strength_features = self.strength_net(torch.cat([angles.sin(), angles.cos()], dim=1))

        h = self.inp(buried)
        seen_on_the_way_down = []
        for i, level in enumerate(self.down_levels):
            h = level(h, strength_features)
            if i < len(self.downs):
                seen_on_the_way_down.append(h)
                h = self.downs[i](h)
        for i in reversed(range(len(self.ups))):
            h = self.ups[i](h) + seen_on_the_way_down[i]
            h = self.up_levels[i](h, strength_features)
        return self.out(h)


class Diffusion(nn.Module):
    """Training loss and generation around a ``StaticRemoverNet``."""

    def __init__(
        self, net: nn.Module, data_scale: float = 1.0, min_strength: float = 0.004, max_strength: float = 160.0,
        schedule_power: float = 7.0, train_log_strength_mean: float = -1.2 + math.log(2),
        train_log_strength_spread: float = 1.2,
    ):
        super().__init__()
        self.net = net
        self.data_scale = data_scale
        self.min_strength, self.max_strength, self.schedule_power = min_strength, max_strength, schedule_power
        self.train_log_strength_mean, self.train_log_strength_spread = train_log_strength_mean, train_log_strength_spread

    def clean_guess(self, buried: torch.Tensor, strength: torch.Tensor) -> torch.Tensor:
        """Guess of the clean window under static of the given strength (one per
        window). The network sees the buried window scaled to unit spread, and
        its output is mixed with the buried window: weak static is mostly passed
        through, strong static mostly replaced (Karras et al., section 5)."""
        s = strength[:, None, None]
        total = s**2 + self.data_scale**2
        keep = self.data_scale**2 / total
        correction = s * self.data_scale / total.sqrt()
        return keep * buried + correction * self.net(buried / total.sqrt(), strength.log() / 4)

    def loss(self, clean: torch.Tensor) -> torch.Tensor:
        log_strength = torch.randn(len(clean), device=clean.device) * self.train_log_strength_spread
        strength = (log_strength + self.train_log_strength_mean).exp()
        buried = clean + strength[:, None, None] * torch.randn_like(clean)
        # weights each strength so that all contribute alike (Karras et al., section 5)
        weight = (strength**2 + self.data_scale**2) / (strength * self.data_scale) ** 2
        error = (self.clean_guess(buried, strength) - clean).square().mean((1, 2))
        return (weight * error).mean()

    @torch.no_grad()
    def generate(self, count: int, num_ifos: int, size: int, steps: int = 18) -> torch.Tensor:
        """New windows: start from static at ``max_strength`` and lower it over
        ``steps`` strengths to zero, with a correction at each step (Heun's
        method; Karras et al., algorithm 1): 2 * steps - 1 network calls."""
        device = next(self.net.parameters()).device
        p = self.schedule_power
        high, low = self.max_strength ** (1 / p), self.min_strength ** (1 / p)
        fractions = torch.linspace(0, 1, steps, device=device)
        strengths = torch.cat([(high + fractions * (low - high)) ** p, torch.zeros(1, device=device)])

        x = strengths[0] * torch.randn(count, num_ifos, size, device=device)
        for strength, next_strength in zip(strengths[:-1], strengths[1:]):
            slope = (x - self.clean_guess(x, strength.expand(count))) / strength
            next_x = x + (next_strength - strength) * slope
            if next_strength > 0:
                next_slope = (next_x - self.clean_guess(next_x, next_strength.expand(count))) / next_strength
                next_x = x + (next_strength - strength) * (slope + next_slope) / 2
            x = next_x
        return x
