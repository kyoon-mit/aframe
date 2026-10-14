"""Regressors for the signal-only test: mean and raw variance per parameter."""

import torch
import torch.nn as nn
from architectures import Architecture
from architectures.networks.s4d_variants import S4ModelPooled

from train.experimental.long_conv.arch import LongConvEnergyNet
from train.experimental.s4d_energy.arch import EnergyMaxS4ModelPooled


def _blocks(net, x, n=None):
    """Encoder and the first ``n`` S4D blocks of an ``S4ModelPooled``:
    (B, d_input, L) -> (B, d_model, L)."""
    x = net.encoder(x.transpose(-1, -2)).transpose(-1, -2)
    for layer, norm, dropout in list(zip(net.s4_layers, net.norms, net.dropouts))[:n]:
        if net.prenorm:
            x = x + dropout(layer(net._apply_norm(norm, x)))
        else:
            x = net._apply_norm(norm, dropout(layer(x)) + x)
    return x


class S4Regressor(Architecture):
    """The den-reg regressor (``S4ModelPooled``: encoder, S4D blocks, pool
    over time, linear readout) on its own. ``pool``: ``"mean"`` or ``"max"``."""

    def __init__(self, num_ifos: int = 2, num_params: int = 1, d_model: int = 128,
                 d_state: int = 64, n_layers: int = 4, dropout: float = 0.2,
                 prenorm: bool = True, num_groups: int = 32,
                 dt_min: float = 1e-4, dt_max: float = 10.0, pool: str = "mean"):
        super().__init__()
        if pool not in ("mean", "max"):
            raise ValueError(f"pool must be 'mean' or 'max', got {pool!r}")
        self.pool = pool
        self.net = S4ModelPooled(
            d_input=num_ifos, d_output=2 * num_params, d_model=d_model, d_state=d_state,
            n_layers=n_layers, dropout=dropout, prenorm=prenorm, num_groups=num_groups,
            dt_min=dt_min, dt_max=dt_max,
        )

    def forward(self, X):
        if self.pool == "mean":
            return self.net(X)
        return self.net.decoder(_blocks(self.net, X).amax(dim=-1))


class ComplexS4Regressor(Architecture):
    """S4D regressor whose last layer is a complex S4D kernel (no skip,
    activation or output mixing), read out by

    - ``"mean"``: real and imaginary parts as channels, mean over time;
    - ``"energy_max"``: log(1 + |y|^2), max over time (``EnergyMaxS4ModelPooled``).
    """

    def __init__(self, num_ifos: int = 2, num_params: int = 1, d_model: int = 128,
                 d_state: int = 64, n_layers: int = 4, dropout: float = 0.2,
                 prenorm: bool = True, num_groups: int = 32,
                 dt_min: float = 1e-4, dt_max: float = 10.0, readout: str = "mean"):
        super().__init__()
        if readout not in ("mean", "energy_max"):
            raise ValueError(f"readout must be 'mean' or 'energy_max', got {readout!r}")
        self.readout = readout
        self.net = EnergyMaxS4ModelPooled(
            d_input=num_ifos, d_output=2 * num_params, d_model=d_model, d_state=d_state,
            n_layers=n_layers, dropout=dropout, prenorm=prenorm, num_groups=num_groups,
            dt_min=dt_min, dt_max=dt_max,
        )
        if readout == "mean":
            self.net.decoder = nn.Linear(2 * d_model, 2 * num_params)

    def forward(self, X):
        if self.readout == "energy_max":
            return self.net(X)
        net = self.net
        z = _blocks(net, X, n=-1)
        z = net._apply_norm(net.norms[-1], z) if net.prenorm else z
        L = z.shape[-1]
        k = net.energy_kernel(L)
        z_f = torch.fft.rfft(z, n=2 * L)
        y_re = torch.fft.irfft(z_f * torch.fft.rfft(k.real, n=2 * L), n=2 * L)[..., :L]
        y_im = torch.fft.irfft(z_f * torch.fft.rfft(k.imag, n=2 * L), n=2 * L)[..., :L]
        return net.decoder(torch.cat([y_re, y_im], dim=1).mean(dim=-1))


class FreeKernelRegressor(Architecture):
    """``LongConvEnergyNet`` with free kernels at the input rate (energy, max
    over time, standardized log energies) and a linear readout to a mean and
    a raw variance per parameter. ``kernel``: ``"free"`` (stored as a
    spectrum) or ``"taps"`` (stored as time samples)."""

    def __init__(self, num_ifos: int = 2, num_params: int = 1, channels: int = 256,
                 sample_rate: float = 2048.0, kernel_length: float = 4.0, highpass: float = 20.0,
                 kernel: str = "free"):
        super().__init__()
        self.body = LongConvEnergyNet(
            num_ifos=num_ifos, channels=channels, kernel=kernel, sample_rate=sample_rate,
            kernel_length=kernel_length, decimate=1, highpass=highpass,
        )
        self.body.out = nn.Linear(channels, 2 * num_params)

    def forward(self, X):
        return self.body(X)
