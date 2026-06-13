"""Two-stage cascade: a mass estimator feeds a heterodyned detector.

The first network ("mass model") sees the raw timeseries and predicts a
chirp mass. The raw timeseries is then heterodyned *per sample* with that
predicted chirp mass and fed to the second network ("detector model").

Unlike :class:`ml4gw.transforms.Heterodyne`, which bakes a fixed grid of
chirp masses into a precomputed phase buffer, the heterodyne here is
computed on the fly from a ``(B,)`` tensor of per-sample chirp masses, so it
is differentiable end to end: gradients from the detector can flow back into
the mass estimator (unless explicitly detached/frozen).
"""

import torch
import torch.nn as nn
from ml4gw.constants import MTSUN_SI

from architectures import Architecture


class PerSampleHeterodyne(nn.Module):
    r"""Heterodyne each sample with its own chirp mass.

    Applies the leading-order (0PN) inspiral phase factor

    .. math:: e^{\frac{3i}{128} (\pi \mathcal{M}_c f)^{-5/3}}

    in the frequency domain, where :math:`\mathcal{M}_c` is a per-sample
    chirp mass (solar masses) and :math:`f` the frequency grid. The DC bin
    is zeroed (matching :class:`ml4gw.transforms.Heterodyne`), which also
    sidesteps the ``f=0`` singularity of the phase.

    Args:
        sample_rate: Sampling rate (Hz) of the input timeseries.
        kernel_length: Duration (s) of the input timeseries segment.

    Shape:
        - Input: ``X`` ``(B, C, T)``, ``chirp_mass`` ``(B,)``
        - Output: ``(B, C, T)``
    """

    def __init__(self, sample_rate: float, kernel_length: float):
        super().__init__()
        self.sample_rate = sample_rate
        n = int(round(sample_rate * kernel_length))
        freq_grid = torch.fft.rfftfreq(n, d=1 / sample_rate)
        self.register_buffer("freq_grid", freq_grid)

    def forward(
        self, X: torch.Tensor, chirp_mass: torch.Tensor
    ) -> torch.Tensor:
        # (B, F): pi * Mc * f in geometrized units
        pi_m_f = (
            torch.pi
            * (chirp_mass[:, None] * MTSUN_SI)
            * self.freq_grid[None, :]
        )
        # clamp keeps the f=0 entry finite; we zero the DC phase below so the
        # placeholder value never reaches the output (and never poisons grads).
        safe = pi_m_f.clamp(min=torch.finfo(pi_m_f.dtype).tiny)
        phase = torch.exp((3j / 128) * safe ** (-5 / 3))  # (B, F) complex
        phase = phase.clone()
        phase[:, 0] = 0.0

        X_fft = torch.fft.rfft(X, dim=-1) / self.sample_rate  # (B, C, F)
        X_het = X_fft * phase[:, None, :]
        out = torch.fft.irfft(X_het, n=X.shape[-1], dim=-1) * self.sample_rate
        return out


class HeterodyneCascade(Architecture):
    """Mass estimator + heterodyne + detector.

    ``mass_model`` maps ``(B, C, T) -> (B, d_mass_out)``; its first output
    channel is treated as the (optionally normalized) chirp-mass estimate.
    The raw input is heterodyned per sample with the de-normalized chirp
    mass and passed to ``detector_model``, whose output is returned alongside
    the full mass-model output.

    The mass model's first output is assumed to be *normalized* as
    ``(chirp_mass - y_mean) / y_std`` (the convention used by
    ``RegressionAframe``); set ``y_mean``/``y_std`` to the values the mass
    model was trained with so a physical chirp mass is recovered for the
    heterodyne. Use ``y_mean=0, y_std=1`` if the model already outputs a
    physical chirp mass.

    The mass model and the detector can see different-length views of the
    same kernel. The full ``kernel_length``-second input is heterodyned and
    fed to the detector; the mass model sees only the last
    ``mass_kernel_length`` seconds (the merger sits at the end of the kernel
    under the aframe windowing convention, so the tail is the merger-rich
    segment the ``merger_1s`` estimator was trained on). Leave
    ``mass_kernel_length`` ``None`` to give the mass model the full input.

    Args:
        mass_model: Pre-built network: ``(B, C, T_mass) -> (B, d_mass_out)``.
        detector_model: Pre-built network consuming the heterodyned
            timeseries ``(B, C, T)``.
        sample_rate: Sampling rate (Hz) of the input.
        kernel_length: Duration (s) of the full input (detector + heterodyne).
        mass_kernel_length: Duration (s) of the trailing segment fed to the
            mass model. ``None`` (default) feeds it the full input. Must be
            ``<= kernel_length``.
        y_mean: Mean used to normalize the mass model's chirp-mass target.
        y_std: Std used to normalize the mass model's chirp-mass target.
        min_chirp_mass: Lower clamp on the de-normalized chirp mass before
            heterodyning, guarding against unphysical / zero values early in
            training.
        detach_chirp_mass: If ``True``, stop gradients from the detector
            from flowing into the mass model through the heterodyne phase.
            (The mass model can still be trained via its own output/loss.)
        freeze_mass_model: If ``True``, put the mass model in eval-only mode
            with ``requires_grad=False`` (use the pretrained estimator as-is).
        mass_weights: Optional path to a state dict for ``mass_model``.
        detector_weights: Optional path to a state dict for ``detector_model``.

    Returns:
        ``(detector_out, mass_out)`` where ``mass_out`` is the raw mass-model
        output ``(B, d_mass_out)`` (e.g. ``[mean, log-var]``).
    """

    def __init__(
        self,
        mass_model: nn.Module,
        detector_model: nn.Module,
        sample_rate: float,
        kernel_length: float,
        mass_kernel_length: float | None = None,
        y_mean: float = 0.0,
        y_std: float = 1.0,
        min_chirp_mass: float = 0.1,
        detach_chirp_mass: bool = False,
        freeze_mass_model: bool = False,
        mass_weights: str | None = None,
        detector_weights: str | None = None,
    ) -> None:
        super().__init__()
        self.mass_model = mass_model
        self.detector_model = detector_model
        self.heterodyne = PerSampleHeterodyne(sample_rate, kernel_length)

        if mass_kernel_length is not None:
            if mass_kernel_length > kernel_length:
                raise ValueError(
                    f"mass_kernel_length ({mass_kernel_length}s) must be "
                    f"<= kernel_length ({kernel_length}s)."
                )
            self.mass_kernel_size = int(
                round(sample_rate * mass_kernel_length)
            )
        else:
            self.mass_kernel_size = None

        self.y_mean = y_mean
        self.y_std = y_std
        self.min_chirp_mass = min_chirp_mass
        self.detach_chirp_mass = detach_chirp_mass
        self.freeze_mass_model = freeze_mass_model

        if mass_weights is not None:
            self._load(self.mass_model, mass_weights)
        if detector_weights is not None:
            self._load(self.detector_model, detector_weights)

        if freeze_mass_model:
            self.mass_model.eval()
            for p in self.mass_model.parameters():
                p.requires_grad_(False)

    @staticmethod
    def _load(module: nn.Module, weights: str) -> None:
        state_dict = torch.load(weights, map_location="cpu", weights_only=True)
        module.load_state_dict(state_dict)

    def train(self, mode: bool = True):
        # keep a frozen mass model in eval mode (dropout/BN off) regardless
        super().train(mode)
        if self.freeze_mass_model:
            self.mass_model.eval()
        return self

    def _mass_input(self, X: torch.Tensor) -> torch.Tensor:
        """Trailing segment of the kernel fed to the mass model."""
        if self.mass_kernel_size is not None:
            return X[..., -self.mass_kernel_size :]
        return X

    def estimate_chirp_mass(self, X: torch.Tensor) -> torch.Tensor:
        """De-normalized, clamped per-sample chirp mass ``(B,)``."""
        mass_out = self.mass_model(self._mass_input(X))
        cm_norm = mass_out[..., 0]
        chirp_mass = cm_norm * self.y_std + self.y_mean
        return chirp_mass.clamp(min=self.min_chirp_mass), mass_out

    def forward(self, X: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        chirp_mass, mass_out = self.estimate_chirp_mass(X)
        cm_for_het = (
            chirp_mass.detach() if self.detach_chirp_mass else chirp_mass
        )
        X_het = self.heterodyne(X, cm_for_het)
        det_out = self.detector_model(X_het)
        return det_out, mass_out
