"""Denoise in the heterodyned frame, score in the ordinary one.

The input is demodulated by the leading-order inspiral phase of its own
chirp mass before it reaches the network, which leaves a chirp sitting
near zero frequency as a slowly varying envelope while broadband noise
stays broadband. The network's output is remodulated by the same phase,
so every metric and plot compares ordinary whitened strain against the
ordinary clean target, exactly as the plain denoiser does.

This is a proof of concept: the chirp mass and the coalescence time are
the true ones from the injection, not estimates.
"""

from typing import Optional

import torch

from train.experimental.heterodyne.phase import dehierodyne, heterodyne
from train.model.denoiser_ky import Denoiser


class HeterodyneDenoiser(Denoiser):
    """A denoiser whose network sees the heterodyned signal.

    The network takes and returns twice as many channels as there are
    interferometers, one in-phase and one quadrature per detector, so an
    architecture built for this task needs ``num_ifos`` doubled.

    Args:
        sample_rate: of the model input, in Hz. Needed to turn rfft bins
            into frequencies.
        chirp_mass_sigma: relative standard deviation of a Gaussian error
            applied to the chirp mass before it is used, so the signal is
            demodulated by a wrong phase. 0 uses the true value.
    """

    def __init__(
        self,
        *args,
        sample_rate: float = 2048.0,
        chirp_mass_sigma: float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if chirp_mass_sigma < 0.0:
            raise ValueError(
                f"chirp_mass_sigma must be >= 0, got {chirp_mass_sigma}"
            )
        self.save_hyperparameters("sample_rate", "chirp_mass_sigma")
        self._conditioning: Optional[tuple] = None

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Heterodyne, denoise, de-heterodyne.

        Falls back to the plain forward pass when no conditioning has
        been set, which would leave the channel counts mismatched, so
        this raises instead of denoising something meaningless.
        """
        if self._conditioning is None:
            raise RuntimeError(
                "HeterodyneDenoiser needs the chirp mass and coalescence "
                "time of each row; call condition() before forward()"
            )
        chirp_mass, coalescence_time = self._conditioning
        if self.hparams.normalize_input:
            X = X / X.std(dim=-1, keepdim=True).clamp(min=1e-8)
        sample_rate = self.hparams.sample_rate
        z = heterodyne(X, chirp_mass, coalescence_time, sample_rate)
        z = self.model(z)
        return dehierodyne(z, chirp_mass, coalescence_time, sample_rate)

    def _jitter(self, chirp_mass: torch.Tensor) -> torch.Tensor:
        """Scale the chirp mass by a Gaussian factor centred on 1."""
        sigma = self.hparams.chirp_mass_sigma
        if not sigma:
            return chirp_mass
        factor = 1.0 + sigma * torch.randn_like(chirp_mass)
        return chirp_mass * factor.clamp_min(0.1)

    def condition(self, params: dict) -> None:
        self._conditioning = (
            self._jitter(params["chirp_mass"]),
            params["coalescence_time"],
        )

    def _shared_step(self, batch, stage: str) -> torch.Tensor:
        self.condition(batch[3])
        try:
            return super()._shared_step(batch, stage)
        finally:
            self._conditioning = None
