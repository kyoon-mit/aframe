"""Long-convolution energy network: learned kernels, energy, max over time.

No templates and no denoiser. The whitened strain of each detector is
convolved with ``channels`` learned complex kernels spanning the whole window,
by FFT at ``sample_rate / decimate``. A kernel is either

- ``"s4d"``: an S4D kernel, a sum of ``d_state / 2`` damped complex tones
  (``ComplexS4DKernel``, positive frequencies at initialization, so the
  complex output gives a phase-blind energy), or
- ``"free"``: one free complex coefficient per positive frequency in the
  band, i.e. any analytic kernel of the window's length, or
- ``"time"``: one free complex value per time sample (a real kernel for each
  quadrature), stored in time with no band restriction.

Each kernel is scaled to unit norm, and the energy 2|y|^2 is the SNR^2 of
the kernel's own signal (white noise gives 2 on average). It is summed over the
detectors (the second one at its largest value within the light travel
time) and maximized over time. A linear map of the log energies,
standardized across channels, gives logits over a log chirp-mass grid.
"""

import math

import torch
import torch.nn as nn

from train.experimental.s4d_energy.arch import ComplexS4DKernel


class LongConvEnergyNet(nn.Module):
    """Args:
        num_ifos: detectors in the input.
        channels: number of learned kernels.
        kernel: ``"s4d"``, ``"free"`` or ``"time"``.
        d_state: S4D state size per channel (``d_state / 2`` tones).
        s4d_init: ``"lin"`` keeps S4D-Lin with timesteps in [dt_min, dt_max];
            ``"band"`` starts every channel's tones spread evenly over
            [highpass, sample_rate / decimate / 2] Hz (random offsets per
            channel) with an e-folding memory of ``memory`` seconds.
        memory: e-folding time of the tones under ``s4d_init="band"``.
        mix_from: if positive, build ``mix_from`` S4D channels and take each
            of the ``channels`` kernels as a learned complex linear
            combination of them (no nonlinearity in between).
        dt_min, dt_max: S4D timestep range, as in ``S4DKernel``.
        sample_rate: of the input, in Hz.
        kernel_length: input length, in seconds.
        decimate: convolve at ``sample_rate / decimate``.
        coincidence: light travel time between detectors, in seconds.
        highpass: lowest frequency a free kernel can hold, in Hz.
        mc_min, mc_max, grid_step: log chirp-mass grid of the logits.
    """

    def __init__(
        self,
        num_ifos: int = 2,
        channels: int = 256,
        kernel: str = "s4d",
        d_state: int = 64,
        s4d_init: str = "lin",
        memory: float = 4.0,
        mix_from: int = 0,
        dt_min: float = 1e-4,
        dt_max: float = 10.0,
        sample_rate: float = 2048.0,
        kernel_length: float = 4.0,
        decimate: int = 2,
        coincidence: float = 0.01,
        highpass: float = 20.0,
        mc_min: float = 0.85,
        mc_max: float = 2.75,
        grid_step: float = 0.005,
    ):
        super().__init__()
        if kernel not in ("s4d", "free", "time"):
            raise ValueError(f"kernel must be 's4d', 'free' or 'time', got {kernel!r}")
        self.kernel_type = kernel
        self.decimate = decimate
        self.length = int(kernel_length * sample_rate) // decimate
        self.f_keep = self.length // 2 + 1          # bins up to the new Nyquist
        self.coincidence = int(coincidence * sample_rate / decimate)
        self.mix_from = mix_from
        if kernel == "s4d":
            self.kernel = ComplexS4DKernel(mix_from or channels, N=d_state, dt_min=dt_min, dt_max=dt_max)
            if mix_from:
                mix = torch.randn(channels, mix_from, 2) / math.sqrt(2 * mix_from)
                self.mix = nn.Parameter(mix)
            if s4d_init == "band":
                self._band_init(sample_rate / decimate, highpass, memory)
            elif s4d_init != "lin":
                raise ValueError(f"s4d_init must be 'lin' or 'band', got {s4d_init!r}")
        elif kernel == "time":
            self.samples = nn.Parameter(torch.randn(channels, self.length, 2) / math.sqrt(self.length))
            self._register_load_state_dict_pre_hook(_rename_old_samples)
        else:
            freqs = torch.fft.fftfreq(self.length, decimate / sample_rate)
            band = (freqs >= highpass) & (freqs < sample_rate / decimate / 2)
            self.register_buffer("band", band, persistent=False)
            init = torch.randn(channels, self.length, 2) / math.sqrt(self.length)
            self.spectrum = nn.Parameter(init)
        grid = torch.arange(math.log(mc_min), math.log(mc_max) + 1e-9, grid_step)
        self.register_buffer("grid_log_mc", grid.float(), persistent=False)
        self.out = nn.Linear(channels, len(grid))

    @torch.no_grad()
    def _band_init(self, rate: float, f_low: float, memory: float, dt: float = 0.01):
        """Tones evenly over [f_low, rate / 2) with random per-channel offsets,
        each decaying by e over ``memory`` seconds; one timestep ``dt``."""
        H, n = self.kernel.A_imag.shape
        spacing = (rate / 2 - f_low) / n
        f = f_low + spacing * (torch.arange(n) + torch.rand(H, 1))       # (H, n) Hz
        self.kernel.log_dt.fill_(math.log(dt))
        self.kernel.A_imag.copy_(2 * math.pi * f / rate / dt)
        self.kernel.log_A_real.fill_(math.log(1 / (memory * rate) / dt))

    def kernels(self) -> torch.Tensor:
        """(channels, length) complex kernels with unit norm."""
        if self.kernel_type == "s4d":
            k = self.kernel(self.length)
            if self.mix_from:
                k = torch.view_as_complex(self.mix) @ k
        elif self.kernel_type == "time":
            k = torch.view_as_complex(self.samples)
        else:
            w = torch.view_as_complex(self.spectrum) * self.band
            k = torch.fft.ifft(w, dim=-1)
        return k / k.abs().pow(2).sum(-1, keepdim=True).sqrt().clamp_min(1e-12)

    def peak_energy(self, X: torch.Tensor) -> torch.Tensor:
        """(B, I, L) whitened strain -> (B, channels) peak network energy."""
        L = self.length
        # resample to sample_rate / decimate; 1 / sqrt(decimate) keeps white
        # noise at unit variance per sample
        spectrum = torch.fft.rfft(X.float(), n=L * self.decimate)[..., : self.f_keep]
        x = torch.fft.irfft(spectrum, n=L) / math.sqrt(self.decimate)        # (B, I, L)
        n = 2 * L
        k = self.kernels()                                                   # (C, L)
        y = torch.fft.ifft(torch.fft.fft(x, n=n)[:, :, None] * torch.fft.fft(k, n=n), n=n)
        # x2: a unit-norm analytic kernel gives |y|^2 = SNR^2 / 2 on its own signal
        energy = 2 * y[..., :L].abs().pow(2)                                 # (B, I, C, L)
        net = energy[:, 0]
        if energy.shape[1] > 1:
            B, I, C, L = energy.shape
            other = nn.functional.max_pool1d(
                energy[:, 1:].reshape(-1, 1, L), 2 * self.coincidence + 1, stride=1, padding=self.coincidence
            ).reshape(B, I - 1, C, L)
            net = net + other.sum(1)
        return net.amax(-1)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Logits over the chirp-mass grid, (B, G)."""
        z = torch.log1p(self.peak_energy(X))
        z = (z - z.mean(-1, keepdim=True)) / z.std(-1, keepdim=True).clamp_min(1e-6)
        return self.out(z)

    def estimate(self, logits: torch.Tensor):
        """Log chirp mass at the logits' peak, refined by a parabola through
        the peak and its neighbours, and the softmax spread."""
        return grid_estimate(logits, self.grid_log_mc)


def _rename_old_samples(state_dict, prefix, *args):
    """Load checkpoints that stored the time-sample kernels as ``taps``."""
    if prefix + "taps" in state_dict:
        state_dict[prefix + "samples"] = state_dict.pop(prefix + "taps")


def grid_estimate(logits: torch.Tensor, g: torch.Tensor):
    """Peak of logits over the log chirp-mass grid ``g``, refined by a
    parabola, and the softmax spread in log chirp mass."""
    j = logits.argmax(-1)
    jj = j.clamp(1, len(g) - 2)
    rows = torch.arange(len(j), device=j.device)
    y0, y1, y2 = logits[rows, jj - 1], logits[rows, jj], logits[rows, jj + 1]
    x0, x1, x2 = g[jj - 1], g[jj], g[jj + 1]
    d = (x0 - x1) * (x0 - x2) * (x1 - x2)
    a = (x2 * (y1 - y0) + x1 * (y0 - y2) + x0 * (y2 - y1)) / d
    b = (x2**2 * (y0 - y1) + x1**2 * (y2 - y0) + x0**2 * (y1 - y2)) / d
    vertex = -b / (2 * a)
    inner = (j > 0) & (j < len(g) - 1) & (a < 0) & (vertex > x0) & (vertex < x2)
    log_mc = torch.where(inner, vertex, g[j])
    probs = torch.softmax(logits, -1)
    mean = (probs * g).sum(-1, keepdim=True)
    spread = (probs * (g - mean) ** 2).sum(-1).sqrt()
    return log_mc, spread


class DeepS4EnergyNet(nn.Module):
    """The den-reg regressor alone, ending in energy and a maximum over time.

    ``EnergyMaxS4ModelPooled`` (encoder, ``n_layers`` S4D blocks, the last
    with a complex kernel whose energy is maximized over time, linear
    decoder) applied to the whitened strain of all detectors at the input
    rate, with one output per point of the log chirp-mass grid. No denoiser.
    Arguments as in ``RegressionTimeDomainS4DenoiseRegressNorm``'s regressor.
    """

    def __init__(
        self,
        num_ifos: int = 2,
        d_model: int = 128,
        d_state: int = 64,
        n_layers: int = 6,
        dropout: float = 0.2,
        prenorm: bool = True,
        num_groups: int = 32,
        dt_min: float = 1e-4,
        dt_max: float = 10.0,
        mc_min: float = 0.85,
        mc_max: float = 2.75,
        grid_step: float = 0.005,
    ):
        super().__init__()
        from train.experimental.s4d_energy.arch import EnergyMaxS4ModelPooled

        grid = torch.arange(math.log(mc_min), math.log(mc_max) + 1e-9, grid_step)
        self.register_buffer("grid_log_mc", grid.float(), persistent=False)
        self.net = EnergyMaxS4ModelPooled(
            d_input=num_ifos, d_output=len(grid), d_model=d_model, d_state=d_state,
            n_layers=n_layers, dropout=dropout, prenorm=prenorm, num_groups=num_groups,
            dt_min=dt_min, dt_max=dt_max,
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Logits over the chirp-mass grid, (B, G)."""
        return self.net(X)

    def estimate(self, logits: torch.Tensor):
        return grid_estimate(logits, self.grid_log_mc)


class ResNetGridNet(nn.Module):
    """aframe's time-domain ResNet (``ml4gw`` ``ResNet1D``: small kernels,
    strided residual stages, average pooling, linear head) with one output
    per point of the log chirp-mass grid. Defaults as aframe's BBH config:
    layers [3, 4, 6, 3], kernel size 3, GroupNorm with 16 groups."""

    def __init__(
        self,
        num_ifos: int = 2,
        layers: list = (3, 4, 6, 3),
        kernel_size: int = 3,
        norm_groups: int = 16,
        mc_min: float = 0.85,
        mc_max: float = 2.75,
        grid_step: float = 0.005,
    ):
        super().__init__()
        from ml4gw.nn.norm import GroupNorm1DGetter
        from ml4gw.nn.resnet.resnet_1d import ResNet1D

        grid = torch.arange(math.log(mc_min), math.log(mc_max) + 1e-9, grid_step)
        self.register_buffer("grid_log_mc", grid.float(), persistent=False)
        self.net = ResNet1D(
            num_ifos, layers=list(layers), classes=len(grid), kernel_size=kernel_size,
            norm_layer=GroupNorm1DGetter(groups=norm_groups),
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Logits over the chirp-mass grid, (B, G)."""
        return self.net(X)

    def estimate(self, logits: torch.Tensor):
        return grid_estimate(logits, self.grid_log_mc)
