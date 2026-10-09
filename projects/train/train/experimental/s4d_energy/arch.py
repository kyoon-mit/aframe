"""Joint denoiser and regressor whose regressor ends in energy and a maximum.

Identical to ``RegressionTimeDomainS4DenoiseRegressNorm`` except at the end
of the regressor: its last S4D layer keeps the complex kernel, outputs the
energy Re^2 + Im^2 of its response (phase-blind, never negative), and the
head takes the maximum over time instead of the mean.
"""

import torch
from architectures.networks.s4d_variants import S4ModelPooled
from architectures.regression import RegressionTimeDomainS4DenoiseRegressNorm
from ml4gw.nn.ssm.s4d import S4DKernel


class ComplexS4DKernel(S4DKernel):
    """``S4DKernel`` returning the complex kernel instead of its real part."""

    def forward(self, L: int) -> torch.Tensor:
        dt = torch.exp(self.log_dt)
        C = torch.view_as_complex(self.C)
        A = torch.complex(-torch.exp(self.log_A_real), self.A_imag)
        dtA = A * dt.unsqueeze(-1)
        K = dtA.unsqueeze(-1) * torch.arange(L, device=A.device)
        C = C * (torch.exp(dtA) - 1.0) / A
        return 2 * torch.einsum("hn, hnl -> hl", C, torch.exp(K))  # (H, L) complex


class EnergyMaxS4ModelPooled(S4ModelPooled):
    """``S4ModelPooled`` whose last layer outputs energy, pooled by a maximum."""

    def __init__(self, *args, d_model: int = 64, d_state: int = 64,
                 dt_min: float = 0.001, dt_max: float = 0.1, **kwargs):
        super().__init__(*args, d_model=d_model, d_state=d_state,
                         dt_min=dt_min, dt_max=dt_max, **kwargs)
        self.energy_kernel = ComplexS4DKernel(d_model, N=d_state, dt_min=dt_min, dt_max=dt_max)

    def _energy(self, x: torch.Tensor) -> torch.Tensor:
        L = x.shape[-1]
        k = self.energy_kernel(L)
        n = 2 * L
        x_f = torch.fft.rfft(x, n=n)
        y_re = torch.fft.irfft(x_f * torch.fft.rfft(k.real, n=n), n=n)[..., :L]
        y_im = torch.fft.irfft(x_f * torch.fft.rfft(k.imag, n=n), n=n)[..., :L]
        return y_re.pow(2) + y_im.pow(2)  # (B, d_model, L)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(-1, -2)
        x = self.encoder(x)
        x = x.transpose(-1, -2)  # (B, d_model, L)
        layers = list(zip(self.s4_layers, self.norms, self.dropouts, strict=True))
        for layer, norm, dropout in layers[:-1]:
            if self.prenorm:
                z = self._apply_norm(norm, x)
                z = dropout(layer(z))
                x = x + z
            else:
                z = dropout(layer(x))
                x = self._apply_norm(norm, z + x)
        _, norm, _ = layers[-1]
        z = self._apply_norm(norm, x) if self.prenorm else x
        energy = torch.log1p(self._energy(z))
        return self.decoder(energy.amax(dim=-1))  # max over time, (B, d_output)


class RegressionTimeDomainS4DenoiseRegressNormEnergy(RegressionTimeDomainS4DenoiseRegressNorm):
    """The joint model with ``EnergyMaxS4ModelPooled`` as its regressor."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        old = self.model.regressor
        params = dict(
            d_input=old.encoder.in_features,
            d_output=old.decoder.out_features,
            d_model=old.encoder.out_features,
            d_state=kwargs.get("regressor_d_state", 64),
            n_layers=len(old.s4_layers),
            dropout=kwargs.get("regressor_dropout", 0.2),
            prenorm=kwargs.get("prenorm", False),
            num_groups=kwargs.get("num_groups"),
            dt_min=kwargs.get("dt_min", 1e-3),
            dt_max=kwargs.get("dt_max", 0.1),
        )
        self.model.regressor = EnergyMaxS4ModelPooled(**params)
