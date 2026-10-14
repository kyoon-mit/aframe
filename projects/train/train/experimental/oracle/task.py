"""Regression on the whitened signal alone: no detector noise."""

import torch

from train.model.regression_ky import GaussianNLLRegressionAframeCustomLR


class SignalOnlyRegression(GaussianNLLRegressionAframeCustomLR):
    """``GaussianNLLRegressionAframeCustomLR`` fed ``X_clean`` instead of ``X``.

    ``X_clean`` is the injected waveform whitened with the window's own PSD
    and scaled to the sampled SNR, so amplitudes match the noisy runs; the
    noise is left out. Batches are ``(X, X_clean, y, params)`` from
    ``DenoiserOnlyAframeDataset``, for validation too.
    """

    def train_step(self, batch):
        _, X_clean, y, params = batch
        return super().train_step((X_clean, y, params))

    def validation_step(self, batch, batch_idx):
        _, X_clean, y, params = batch
        return super().validation_step((None, None, X_clean[None], params), batch_idx)

    def test_step(self, batch, _):
        """Outputs for ``PlotParamEstCallback``, on the clean signal."""
        _, X_clean, _, params = batch
        mean, var = self._split(self(X_clean))
        mean = mean * self.y_std + self.y_mean
        sigma = var.sqrt() * self.y_std
        targets = torch.stack([params[k] for k in self.param_names], dim=1)
        rows = ~torch.isnan(targets).any(dim=1)
        return {
            "y_true": targets[rows].cpu(),
            "y_pred": mean[rows].detach().cpu(),
            "y_sigma": sigma[rows].detach().cpu(),
            "snr": params["snr"][rows].cpu(),
        }
