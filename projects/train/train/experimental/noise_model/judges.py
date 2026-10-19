"""Judges: classifiers that tell noise windows of one source from another's.

Each takes whitened windows of shape (windows, detectors, samples) and returns
one number per window, positive when it believes the window comes from the
first source.
"""

import math

import torch
import torch.nn as nn
from ml4gw.nn.norm import GroupNorm1DGetter
from ml4gw.nn.resnet import ResNet1D, ResNet2D

from train.experimental.noise_model.layers import BothWaysS4D
from train.experimental.template_bank.arch import TemplateBankNet


def small_network(inputs: int, hidden: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(inputs, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 1))


class TimeDomainJudge(nn.Module):
    """aframe's deep 1-D CNN (ResNet) on the whitened strain."""

    def __init__(
        self, num_ifos: int = 2, layers: tuple[int, ...] = (3, 4, 6, 3), kernel_size: int = 3, norm_groups: int = 16
    ):
        super().__init__()
        self.net = ResNet1D(
            num_ifos, layers, classes=1, kernel_size=kernel_size, norm_layer=GroupNorm1DGetter(groups=norm_groups)
        )

    def forward(self, windows: torch.Tensor) -> torch.Tensor:
        return self.net(windows)[:, 0]


class TimeFrequencyJudge(nn.Module):
    """A 2-D CNN (ResNet) on each detector's time-frequency image, the log power
    of short Fourier transforms."""

    def __init__(self, num_ifos: int = 2, fft_size: int = 256, hop: int = 64, layers: tuple[int, ...] = (2, 2, 2, 2)):
        super().__init__()
        self.fft_size, self.hop = fft_size, hop
        self.register_buffer("taper", torch.hann_window(fft_size), persistent=False)
        self.net = ResNet2D(num_ifos, layers, classes=1)

    def forward(self, windows: torch.Tensor) -> torch.Tensor:
        num_windows, num_ifos, size = windows.shape
        transforms = torch.stft(
            windows.reshape(-1, size), self.fft_size, self.hop, window=self.taper, return_complex=True
        )
        images = transforms.abs().square().log1p().reshape(num_windows, num_ifos, *transforms.shape[1:])
        return self.net(images)[:, 0]


class S4DJudge(nn.Module):
    """S4D blocks that read the whole window both ways, then the mean and the
    largest value of each channel over time. An S4D state's memory lasts about
    2 / dt samples, so ``dt_min`` sets the longest memory."""

    def __init__(
        self, num_ifos: int = 2, width: int = 64, blocks: int = 4, d_state: int = 64,
        dt_min: float = 0.001, dt_max: float = 0.1,
    ):
        super().__init__()
        self.inp = nn.Conv1d(num_ifos, width, 1)
        self.blocks = nn.ModuleList([BothWaysS4D(width, d_state, dt_min, dt_max) for _ in range(blocks)])
        self.out = nn.Linear(2 * width, 1)

    def forward(self, windows: torch.Tensor) -> torch.Tensor:
        x = self.inp(windows)
        for block in self.blocks:
            x = block(x)
        return self.out(torch.cat([x.mean(-1), x.amax(-1)], dim=1))[:, 0]


class BankJudge(nn.Module):
    """The template bank's output, each template's loudest network SNR^2 in the
    window, read by a small network. The bank's whitened templates are fixed;
    its own learned read-out is not used."""

    def __init__(
        self, bank_file: str, num_ifos: int = 2, sample_rate: float = 2048.0, kernel_length: float = 4.0,
        fduration: float = 1.0, highpass: float = 20.0, hidden: int = 64,
    ):
        super().__init__()
        # decimate=1: correlate at the full sample rate
        self.bank = TemplateBankNet(
            bank_file, num_ifos, sample_rate, kernel_length, decimate=1, fduration=fduration, highpass=highpass
        )
        self.bank.requires_grad_(False)
        self.net = small_network(self.bank.filters.shape[1], hidden)

    def forward(self, windows: torch.Tensor) -> torch.Tensor:
        return self.net(self.bank.peak_snr2(windows).log1p())[:, 0]


class StatisticsJudge(nn.Module):
    """Simple numbers for each detector, read by a small network: the share of
    samples whose size falls between successive ``value_edges`` (in units of
    whitened noise), the power in ``bands`` frequency bands spaced evenly in
    log frequency from ``highpass`` to the highest frequency, and the mean,
    spread, lopsidedness, tail weight and largest size of the values."""

    def __init__(
        self, num_ifos: int = 2, sample_rate: float = 2048.0, highpass: float = 20.0,
        value_edges: tuple[float, ...] = (0.5, 1, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10, 20, 50),
        bands: int = 16, hidden: int = 64,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.register_buffer("value_edges", torch.tensor(value_edges, dtype=torch.float32), persistent=False)
        self.register_buffer(
            "band_edges", torch.logspace(math.log10(highpass), math.log10(sample_rate / 2), bands + 1), persistent=False
        )
        # per detector: value shares, band powers, and the 5 numbers about the values
        features = num_ifos * (len(value_edges) + 1 + bands + 5)
        self.net = nn.Sequential(nn.LayerNorm(features), small_network(features, hidden))

    def forward(self, windows: torch.Tensor) -> torch.Tensor:
        num_windows, num_ifos, size = windows.shape

        value_bin = torch.bucketize(windows.abs(), self.value_edges)
        samples_per_value_bin = torch.zeros(num_windows, num_ifos, len(self.value_edges) + 1, device=windows.device)
        samples_per_value_bin.scatter_add_(-1, value_bin, torch.ones_like(windows))
        # adding one sample's share keeps the log of an empty bin finite
        value_shares = (samples_per_value_bin / size + 1 / size).log()

        power = torch.fft.rfft(windows).abs().square()
        frequencies = torch.fft.rfftfreq(size, 1 / self.sample_rate, device=windows.device)
        band = torch.bucketize(frequencies, self.band_edges) - 1  # -1 below the first edge
        inside = (band >= 0) & (band < len(self.band_edges) - 1)
        band_power = torch.zeros(num_windows, num_ifos, len(self.band_edges) - 1, device=windows.device)
        band_power.index_add_(-1, band[inside], power[..., inside])
        frequency_bins_per_band = torch.bincount(band[inside], minlength=len(self.band_edges) - 1).clamp(min=1)
        band_power = (band_power / frequency_bins_per_band).log()

        mean = windows.mean(-1, keepdim=True)
        spread = windows.std(-1, keepdim=True)
        standardized = (windows - mean) / spread
        about_the_values = torch.cat(
            [
                mean,
                spread.log(),
                standardized.pow(3).mean(-1, keepdim=True),
                standardized.pow(4).mean(-1, keepdim=True).log(),
                windows.abs().amax(-1, keepdim=True).log(),
            ],
            dim=-1,
        )
        return self.net(torch.cat([value_shares, band_power, about_the_values], dim=-1).flatten(1))[:, 0]
