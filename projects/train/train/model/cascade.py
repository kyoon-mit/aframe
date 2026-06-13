"""Lightning module for the two-stage heterodyne cascade.

Stage 1 (the "mass model") regresses the chirp mass from the raw strain.
Stage 2 (the "detector model") sees the strain heterodyned with that
predicted chirp mass and *also* regresses the chirp mass; its predicted
variance is used as the detection statistic (lower uncertainty -> higher
score), evaluated with ``TimeSlideAUROC``.

The architecture
(``architectures.networks.heterodyne_cascade.HeterodyneCascade``)
returns ``(detector_out, mass_out)``. This module:

  * trains the detector with GaussianNLL against the chirp-mass target,
  * optionally adds the mass model's own GaussianNLL as an auxiliary term
    (``lambda_aux_mass``) so it can be fine-tuned jointly, and
  * scores detections from the detector's predicted variance.

Set ``freeze_mass_model: true`` (on the arch) + ``lambda_aux_mass: 0`` to use
a frozen pretrained estimator; set ``freeze_mass_model: false`` +
``lambda_aux_mass > 0`` to fine-tune both jointly.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from train.model.regression_kevin import (
    RegressionAframeS4D,
    _log_gaussian_nll,
)


class CascadeAframeS4D(RegressionAframeS4D):
    """Heterodyne cascade trained for detection (AUROC) + chirp-mass NLL.

    Expects ``arch`` to be a ``HeterodyneCascade`` whose two sub-models each
    output ``[mean, pre-variance]`` for the chirp mass (``d_output=2``).

    Args:
        lambda_aux_mass: Weight on the mass model's own GaussianNLL loss.
            ``0`` disables it (detector-only training); ``>0``
            trains/fine-tunes
            the mass model jointly. Ignored gradient-wise if the arch has
            ``freeze_mass_model=True``, but the term is still logged.
    """

    def __init__(self, *args, lambda_aux_mass: float = 0.0, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.lambda_aux_mass = lambda_aux_mass
        self.save_hyperparameters(ignore=["arch", "lr_scheduler", "metric"])

    def forward(self, X: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns ``(detector_out, mass_out)``."""
        return self.model(X)

    def score(self, X: torch.Tensor) -> torch.Tensor:
        det_out, _ = self(self._prepare_input(X))
        _, var_pre = det_out.chunk(2, dim=-1)
        return -self.var_activation(var_pre).mean(dim=-1)

    def _nll_terms(self, out: torch.Tensor, y_norm: torch.Tensor):
        mean, var_pre = out.chunk(2, dim=-1)
        var = self.var_activation(var_pre)
        mean = mean.reshape(y_norm.shape)
        var = var.reshape(y_norm.shape)
        nll = self.criterion(mean, y_norm, var)
        return mean, var, nll

    def compute_loss(self, batch):
        X, labels, params = batch

        det_out, mass_out = self(self._prepare_input(X))

        chirp_mass = self.m1_m2_to_chirp_mass(
            params["mass_1"], params["mass_2"]
        )
        n_vars = det_out.shape[-1] // 2
        y_norm = self._normalize_target(chirp_mass).reshape(-1, n_vars)

        det_mean, det_var, det_nll = self._nll_terms(det_out, y_norm)
        indiv_mse = nn.MSELoss(reduction="none")(det_mean, y_norm).mean(dim=0)
        spread = F.softplus(
            y_norm.detach().var(dim=0) - det_mean.var(dim=0)
        ).mean()

        loss = det_nll + self.lambda_spread * spread

        if self.lambda_aux_mass > 0:
            _, _, aux_nll = self._nll_terms(mass_out, y_norm)
            loss = loss + self.lambda_aux_mass * aux_nll
            self.aux_nll = aux_nll.detach()
        else:
            self.aux_nll = None

        # mirror RegressionAframe.compute_loss return signature so the
        # inherited validation_step works unchanged (detector quantities)
        return loss, det_nll, spread, indiv_mse, det_var, det_mean

    def train_step(self, batch):
        loss, nll, spread, indiv_mse, var, _ = self.compute_loss(batch)
        _log_gaussian_nll(self, "train", nll, indiv_mse, var)
        self.log("train/spread_penalty", spread)
        if self.aux_nll is not None:
            self.log("train/aux_mass_nll", self.aux_nll, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        X, y_target, _ = batch
        det_out, _ = self(self._prepare_input(X))
        mean_norm = det_out[:, : self.n_vars]
        sigma_norm = torch.sqrt(self.var_activation(det_out[:, self.n_vars :]))
        mean_phys, sigma_phys = self._unnormalize_output(mean_norm, sigma_norm)
        return {
            "y_true": y_target.detach().cpu(),
            "y_pred": mean_phys.detach().cpu(),
            "y_sigma": sigma_phys.detach().cpu(),
        }
