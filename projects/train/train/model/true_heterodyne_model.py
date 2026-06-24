"""Detection model that heterodynes its input with the *true* chirp mass.

This mirrors :class:`~train.model.supervised.SupervisedAframe` (BCE-trained
detection model evaluated with ``TimeSlideAUROC``), but before the strain is
passed to the network it is heterodyned *per sample* with the true chirp mass
of the injected signal. Injected samples use the chirp mass derived from their
``mass_1``/``mass_2`` parameters; pure-background samples (which have no true
chirp mass) are heterodyned with a chirp mass drawn at random from the chirp
mass distribution of a reference waveform file.

The heterodyne itself reuses
:class:`architectures.networks.heterodyne_cascade.PerSampleHeterodyne`, so it
is the same leading-order inspiral phase factor used by the heterodyne cascade.
"""

import h5py
import torch
from architectures.networks.heterodyne_cascade import PerSampleHeterodyne
from architectures.supervised import SupervisedArchitecture

from train.model.supervised import SupervisedAframe

Tensor = torch.Tensor


class TrueHeterodyneAframe(SupervisedAframe):
    """Supervised detection model with true-chirp-mass heterodyning.

    Args:
        arch: Architecture to train on.
        sample_rate: Sampling rate (Hz) of the input timeseries.
        kernel_length: Duration (s) of the input timeseries fed to the model.
            Must match the time length of ``X`` so the heterodyne phase grid
            lines up with the FFT of the input.
        background_chirp_mass_file: Path to an HDF5 waveform file (same layout
            as the training/validation waveform files) whose ``parameters``
            group is used to build the pool of chirp masses sampled for
            background segments. A ``chirp_mass`` dataset is used directly if
            present, otherwise it is derived from ``mass_1``/``mass_2``.
        min_chirp_mass: Lower clamp on the chirp mass before heterodyning,
            guarding against unphysical / zero values.
        chirp_mass_perturbation: Fractional scale of the random perturbation
            applied to the chirp mass during both training and validation,
            simulating a noisy mass estimate. A value of ``0.05`` perturbs
            each true chirp mass by ~5%. With
            ``perturbation_dist="gaussian"`` it is the 1-sigma
            relative scatter (``Mc -> Mc * (1 + N(0, 0.05))``); with
            ``"uniform"`` it is the half-width of a relative uniform draw
            (``Mc -> Mc * U(1 - 0.05, 1 + 0.05)``). ``0.0`` disables it.
        perturbation_dist: Distribution used for the perturbation, either
            ``"gaussian"`` (default) or ``"uniform"``.
    """

    def __init__(
        self,
        arch: SupervisedArchitecture,
        sample_rate: float,
        kernel_length: float,
        background_chirp_mass_file: str,
        min_chirp_mass: float = 0.1,
        chirp_mass_perturbation: float = 0.0,
        perturbation_dist: str = "gaussian",
        *args,
        **kwargs,
    ) -> None:
        super().__init__(arch, *args, **kwargs)
        self.heterodyne = PerSampleHeterodyne(sample_rate, kernel_length)
        self.min_chirp_mass = min_chirp_mass
        self.chirp_mass_perturbation = chirp_mass_perturbation
        if perturbation_dist not in ("gaussian", "uniform"):
            raise ValueError(
                "perturbation_dist must be 'gaussian' or 'uniform', "
                f"got {perturbation_dist!r}"
            )
        self.perturbation_dist = perturbation_dist
        self._load_background_chirp_masses(background_chirp_mass_file)

    def _load_background_chirp_masses(self, fname: str) -> None:
        """Read the pool of chirp masses sampled for background segments."""
        with h5py.File(fname, "r") as f:
            params = f["parameters"]
            if "chirp_mass" in params:
                chirp_mass = params["chirp_mass"][:]
            else:
                m1 = params["mass_1"][:]
                m2 = params["mass_2"][:]
                chirp_mass = (m1 * m2) ** (3 / 5) / (m1 + m2) ** (1 / 5)
        self.register_buffer(
            "background_chirp_masses",
            torch.as_tensor(chirp_mass, dtype=torch.float32),
        )

    @staticmethod
    def m1_m2_to_chirp_mass(m1: Tensor, m2: Tensor) -> Tensor:
        return (m1 * m2) ** (3 / 5) / (m1 + m2) ** (1 / 5)

    def _sample_background_chirp_mass(
        self, n: int, device: torch.device
    ) -> Tensor:
        """Draw ``n`` chirp masses at random from the reference pool."""
        pool = self.background_chirp_masses
        idx = torch.randint(0, pool.numel(), (n,), device=pool.device)
        return pool[idx].to(device)

    def _resolve_chirp_mass(
        self, params: dict, device: torch.device
    ) -> Tensor:
        """Per-sample chirp mass, filling non-injected samples with draws.

        Injected samples carry a finite ``mass_1``/``mass_2``; samples that
        weren't injected have NaN params (see the supervised dataset) and are
        assigned a chirp mass sampled from the reference distribution.
        """
        chirp_mass = self.m1_m2_to_chirp_mass(
            params["mass_1"], params["mass_2"]
        )
        nan_mask = torch.isnan(chirp_mass)
        if nan_mask.any():
            draws = self._sample_background_chirp_mass(
                int(nan_mask.sum()), device
            )
            chirp_mass = chirp_mass.clone()
            chirp_mass[nan_mask] = draws
        return chirp_mass.clamp(min=self.min_chirp_mass)

    def _perturb_chirp_mass(self, chirp_mass: Tensor) -> Tensor:
        """Apply a random fractional perturbation to ``chirp_mass``.

        Simulates a noisy mass estimate. Disabled when
        ``chirp_mass_perturbation`` is 0; otherwise draws a per-sample
        relative offset from the configured distribution and clamps the
        result to ``min_chirp_mass``.
        """
        if self.chirp_mass_perturbation <= 0:
            return chirp_mass
        if self.perturbation_dist == "gaussian":
            # 1-sigma relative scatter
            frac = torch.randn_like(chirp_mass) * self.chirp_mass_perturbation
        else:
            # uniform half-width relative draw in [-eps, eps]
            frac = (
                torch.empty_like(chirp_mass).uniform_(-1.0, 1.0)
                * self.chirp_mass_perturbation
            )
        return (chirp_mass * (1 + frac)).clamp(min=self.min_chirp_mass)

    def forward(self, X: Tensor, chirp_mass: Tensor) -> Tensor:
        # heterodyning is a fixed (non-learnable) preprocessing step; run it
        # outside the autograd graph so the FFTs aren't recorded for backward.
        # match chirp_mass to X's dtype so a float64 chirp mass doesn't promote
        # the FFT phase (and thus the heterodyned output) to double.
        with torch.no_grad():
            X = self.heterodyne(X, chirp_mass.to(X.dtype)).to(X.dtype)
        return self.model(X)

    def score(self, X: Tensor, chirp_mass: Tensor) -> Tensor:
        return self(X, chirp_mass)

    def train_step(self, batch: tuple[Tensor, Tensor, dict]) -> Tensor:
        X, y, params = batch
        chirp_mass = self._resolve_chirp_mass(params, X.device)
        chirp_mass = self._perturb_chirp_mass(chirp_mass)
        y_hat = self(X, chirp_mass)
        return torch.nn.functional.binary_cross_entropy_with_logits(y_hat, y)

    def validation_step(self, batch, _) -> None:
        shift, X_bg, X_inj, params = batch

        # background has no true chirp mass: draw from the reference pool
        bg_chirp_mass = self._sample_background_chirp_mass(
            X_bg.shape[0], X_bg.device
        ).clamp(min=self.min_chirp_mass)
        bg_chirp_mass = self._perturb_chirp_mass(bg_chirp_mass)
        y_bg = self.score(X_bg, bg_chirp_mass)

        # injections use their true chirp mass, repeated across views
        num_views, batch_size, *shape = X_inj.shape
        X_inj = X_inj.view(num_views * batch_size, *shape)
        fg_chirp_mass = self.m1_m2_to_chirp_mass(
            params["mass_1"], params["mass_2"]
        ).clamp(min=self.min_chirp_mass)
        # X_inj is flattened view-major (view * batch_size + b), so tile the
        # per-sample chirp masses the same way
        fg_chirp_mass = fg_chirp_mass.repeat(num_views)
        # apply the same noisy-estimate perturbation used in training so the
        # validation metric reflects performance under the mass offset
        fg_chirp_mass = self._perturb_chirp_mass(fg_chirp_mass)

        y_fg = self.score(X_inj, fg_chirp_mass)
        y_fg = y_fg.view(num_views, batch_size).mean(0)

        self.metric.update(shift, y_bg, y_fg)

        metric_name = self.metric.__class__.__name__
        self.log(
            f"validation/{metric_name}",
            self.metric,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )
