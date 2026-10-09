"""Chirp-kernel network: physical chirp kernels, energy, maximum over time.

Each of J kernels is a Newtonian chirp with its own chirp mass, read
backwards from its merger at lag 0, in two quadratures. The whitened strain
of every detector is convolved with every kernel by FFT; the two
quadratures are squared and added and summed over detectors (the energy);
the energy is maximized over time. No mean pooling anywhere: averaging an
oscillating response cancels it. The output is one logit per kernel.
"""

import math

import torch
import torch.nn as nn

TSUN = 4.925491e-6  # G * (1 solar mass) / c^3, in seconds


class ChirpKernels(nn.Module):
    """J Newtonian chirps, merger at lag 0, two quadratures, unit norm."""

    def __init__(
        self,
        n_kernels: int,
        mc_min: float,
        mc_max: float,
        sample_rate: float,
        f_low: float,
        f_high: float,
        learn_chirp_mass: bool,
    ):
        super().__init__()
        log_mc = torch.linspace(math.log(mc_min), math.log(mc_max), n_kernels)
        self.log_mc = nn.Parameter(log_mc, requires_grad=learn_chirp_mass)
        self.sample_rate = sample_rate
        self.f_low = f_low
        self.f_high = f_high

    def forward(self, length: int):
        tau = (torch.arange(length, device=self.log_mc.device) + 1) / self.sample_rate
        T = TSUN * self.log_mc.exp()[:, None]
        phase = -2 * (tau / (5 * T)) ** (5 / 8)
        f = (1 / (8 * math.pi)) * (5 / tau) ** (3 / 8) * T ** (-5 / 8)
        # smooth band edges keep the chirp-mass gradient well behaved
        band = torch.sigmoid((f - self.f_low) / 2) * torch.sigmoid((self.f_high - f) / 20)
        amp = tau ** -0.25 * band
        kc, ks = amp * torch.cos(phase), amp * torch.sin(phase)
        norm = kc.pow(2).sum(-1, keepdim=True).sqrt().clamp_min(1e-12)
        return kc / norm, ks / norm  # each (J, length)


class ChirpKernelNet(nn.Module):
    """Chirp kernels, energy, max over time, logits over kernel chirp masses.

    Args:
        num_ifos: detectors in the input.
        n_kernels: number of chirp kernels J.
        mc_min, mc_max: chirp-mass range the kernels are spread over.
        sample_rate: of the input, in Hz.
        f_low, f_high: band of each kernel, in Hz.
        learn_chirp_mass: train the kernels' chirp masses.
    """

    def __init__(
        self,
        num_ifos: int = 2,
        n_kernels: int = 512,
        mc_min: float = 0.85,
        mc_max: float = 2.75,
        sample_rate: float = 2048.0,
        f_low: float = 20.0,
        f_high: float = 800.0,
        learn_chirp_mass: bool = True,
    ):
        super().__init__()
        self.kernels = ChirpKernels(
            n_kernels, mc_min, mc_max, sample_rate, f_low, f_high, learn_chirp_mass
        )
        # logits: scaled standardized log-energies plus a learned correction
        self.scale = nn.Parameter(torch.tensor(5.0))
        self.correction = nn.Linear(n_kernels, n_kernels)
        nn.init.zeros_(self.correction.weight)
        nn.init.zeros_(self.correction.bias)

    def energy_peaks(self, X: torch.Tensor) -> torch.Tensor:
        """(B, ifos, L) whitened strain -> (B, J) peak energy over time."""
        length = X.shape[-1]
        kc, ks = self.kernels(length)
        x_f = torch.fft.rfft(X, n=2 * length)[:, :, None]       # (B, I, 1, F)
        zc = torch.fft.irfft(x_f * torch.fft.rfft(kc, n=2 * length), n=2 * length)[..., :length]
        zs = torch.fft.irfft(x_f * torch.fft.rfft(ks, n=2 * length), n=2 * length)[..., :length]
        energy = (zc.pow(2) + zs.pow(2)).sum(1)                   # (B, J, L)
        return energy.amax(-1)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        feat = torch.log1p(self.energy_peaks(X))
        z = (feat - feat.mean(-1, keepdim=True)) / feat.std(-1, keepdim=True).clamp_min(1e-6)
        return self.scale * z + self.correction(z)                # (B, J) logits


class ChirpKernelS4DNet(nn.Module):
    """Frozen chirp kernels, per-detector energy maps, an S4D stack, max over time.

    The energy maps e_{j,D}(t) are reduced in time by a maximum over blocks
    of ``pool`` samples, projected to ``d_model`` channels, passed through
    ``n_layers`` residual S4D blocks, maximized over time and mapped to one
    logit per kernel. That correction is added to the single-layer bank's
    logits (``bank_scale`` times the standardized log peak energies, summed
    over detectors), and starts at zero, so the network begins as the bank.

    Args:
        num_ifos: detectors in the input.
        n_kernels: number of chirp kernels.
        mc_min, mc_max: chirp-mass range of the kernels.
        sample_rate: of the input, in Hz.
        f_low, f_high: band of each kernel, in Hz.
        pool: samples per time block for the energy-map maximum.
        d_model, d_state, n_layers, dropout: the S4D stack.
        bank_scale: fixed weight of the bank logits.
    """

    def __init__(
        self,
        num_ifos: int = 2,
        n_kernels: int = 512,
        mc_min: float = 0.85,
        mc_max: float = 2.75,
        sample_rate: float = 2048.0,
        f_low: float = 20.0,
        f_high: float = 800.0,
        pool: int = 8,
        d_model: int = 128,
        d_state: int = 64,
        n_layers: int = 4,
        dropout: float = 0.1,
        bank_scale: float = 5.0,
    ):
        super().__init__()
        from ml4gw.nn.ssm.s4d import S4D

        self.kernels = ChirpKernels(
            n_kernels, mc_min, mc_max, sample_rate, f_low, f_high, learn_chirp_mass=False
        )
        self.pool = pool
        self.bank_scale = bank_scale
        self.proj = nn.Conv1d(num_ifos * n_kernels, d_model, 1)
        self.layers = nn.ModuleList(
            [S4D(d_model, d_state=d_state, dropout=dropout, transposed=True) for _ in range(n_layers)]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
        self.out = nn.Linear(d_model, n_kernels)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    @torch.no_grad()
    def energy_maps(self, X: torch.Tensor) -> torch.Tensor:
        """(B, ifos, L) -> per-detector log energy (B, ifos, J, L)."""
        length = X.shape[-1]
        kc, ks = self.kernels(length)
        n = 2 * length
        x_f = torch.fft.rfft(X, n=n)[:, :, None]
        zc = torch.fft.irfft(x_f * torch.fft.rfft(kc, n=n), n=n)[..., :length]
        zs = torch.fft.irfft(x_f * torch.fft.rfft(ks, n=n), n=n)[..., :length]
        return torch.log1p(zc.pow(2) + zs.pow(2))

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        e = self.energy_maps(X)                                   # (B, I, J, L)
        # bank logits: summed-detector peak energy, standardized across kernels
        peak = torch.log1p(torch.expm1(e).sum(1).amax(-1))       # (B, J)
        z = (peak - peak.mean(-1, keepdim=True)) / peak.std(-1, keepdim=True).clamp_min(1e-6)
        bank = self.bank_scale * z

        B, I, J, L = e.shape
        h = nn.functional.max_pool1d(e.reshape(B, I * J, L), self.pool)  # (B, I*J, L/pool)
        h = self.proj(h)                                          # (B, d_model, L/pool)
        for layer, norm in zip(self.layers, self.norms):
            h = h + layer(h)
            h = norm(h.transpose(1, 2)).transpose(1, 2)
        h = h.amax(-1)                                            # max over time, (B, d_model)
        return bank + self.out(h)                                 # logits (B, J)
