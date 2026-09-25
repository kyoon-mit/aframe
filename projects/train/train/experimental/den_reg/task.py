import torch
import torch.nn.functional as F

from train.model.regression_ky import DenoisedGaussianNLLRegression


class DenoisedRegressionWithDenoiserMetrics(DenoisedGaussianNLLRegression):
    """Joint denoiser and regressor, validated the way each is alone.

    The parent validates on timeslide batches, which carry no clean
    target, so only the regression head is scored and the denoised output
    is discarded. Pair this with ``DenoiserOnlyAframeDataset`` instead,
    whose validation batches come from the injection path and look like
    training ones, and both halves can be measured: the denoiser against
    the clean target it was given, the regressor against the injection
    parameters.

    The denoiser metrics are then the ones the standalone ``Denoiser``
    task logs, computed the same way on the same kind of batch, so the
    two are directly comparable.
    """

    def _log_denoiser_metrics(self, denoised, target, stage="val"):
        loss = self.denoiser_loss(denoised, target)
        self.log(f"{stage}/den_loss", loss, on_epoch=True, sync_dist=True)
        for attribute, name in (
            ("last_time_term", "den_time_loss"),
            ("last_spectral_term", "den_freq_loss"),
            ("last_amp_term", "den_amp_loss"),
            ("last_rho_term", "den_rho_loss"),
        ):
            value = getattr(self.denoiser_loss, attribute, None)
            if value is not None:
                self.log(f"{stage}/{name}", value, on_epoch=True)

        residual = denoised - target
        self.log(
            f"{stage}/l1_per_event", residual.abs().sum(dim=(-2, -1)).mean()
        )
        self.log(
            f"{stage}/rss_per_event", residual.pow(2).sum(dim=(-2, -1)).mean()
        )
        pred_rms = denoised.pow(2).mean().sqrt()
        target_rms = target.pow(2).mean().sqrt()
        self.log(f"{stage}/pred_rms", pred_rms)
        self.log(f"{stage}/target_rms", target_rms)
        self.log(f"{stage}/diff_pred_target_rms", pred_rms - target_rms)

        # rho is the normalised inner product with the clean target, so a
        # shrunken copy scores near zero however small its loss; gain is
        # the amplitude realised. Rows with no signal are skipped.
        pred_flat = denoised.reshape(denoised.shape[0], -1)
        target_flat = target.reshape(target.shape[0], -1)
        target_norm = target_flat.pow(2).sum(-1).sqrt()
        carries_signal = target_norm > 0.5
        if not carries_signal.any():
            return
        pred_flat = pred_flat[carries_signal]
        target_flat = target_flat[carries_signal]
        pred_norm = pred_flat.pow(2).sum(-1).sqrt()
        target_norm = target_norm[carries_signal].clamp_min(1e-12)
        overlap = (pred_flat * target_flat).sum(-1)
        self.log(
            f"{stage}/rho",
            (overlap / (pred_norm * target_norm).clamp_min(1e-12)).mean(),
        )
        self.log(f"{stage}/gain", (pred_norm / target_norm).mean())

    def _log_regression_validation(self, out, params):
        """The parent's val/* regression set, on one view of the batch."""
        mask = ~torch.isnan(next(iter(params.values())))
        if not mask.any():
            return
        targets = torch.stack(
            [params[name][mask] for name in self.param_names], dim=1
        )
        mean, var = self._split(out[mask])
        y_norm = self._normalize(targets)
        nll = self.criterion(mean, y_norm, var)
        spread = self._spread_penalty(y_norm, mean)
        self.log("val/gaussnll", nll, on_epoch=True, sync_dist=True)
        self.log("val/spread_penalty", spread, on_epoch=True, sync_dist=True)
        self.log(
            "val/loss",
            nll + self.hparams.lambda_spread * spread,
            on_epoch=True,
            sync_dist=True,
        )

        mean_phys = mean * self.y_std + self.y_mean
        sigma_phys = torch.sqrt(var) * self.y_std
        rel_err = (mean_phys - targets).abs() / targets.abs().clamp(min=1e-8)
        for i, name in enumerate(self.param_names):
            self.log(
                f"val/mse_{name}",
                F.mse_loss(mean_phys[:, i], targets[:, i]),
                on_epoch=True,
                sync_dist=True,
            )
            self.log(
                f"val/mae_{name}",
                F.l1_loss(mean_phys[:, i], targets[:, i]),
                on_epoch=True,
                sync_dist=True,
            )
            self.log(
                f"val/sigma_{name}",
                sigma_phys[:, i].mean(),
                on_epoch=True,
                sync_dist=True,
            )
            for pct in (1, 2, 5, 10):
                self.log(
                    f"val/within_{pct}pct_{name}",
                    (rel_err[:, i] < pct / 100.0).float().mean(),
                    on_epoch=True,
                    sync_dist=True,
                )

    def validation_step(self, batch, _):
        """One injected batch, scored for both halves.

        ``(X, X_clean, y, params)``, as the training step receives, rather
        than the timeslide batch the parent expects.
        """
        X, X_clean, _, params = batch
        with torch.no_grad():
            denoised, out = self(X)
            self._log_denoiser_metrics(denoised, X_clean)
            self._log_regression_validation(out, params)

    def test_step(self, batch, _):
        """Raw regression outputs for ``PlotParamEstCallback``.

        The parent unpacks a three-element batch and calls the network
        for the regression output alone; here the batch carries the
        clean target as well and the network returns the denoised strain
        beside the prediction. Rows whose parameters are NaN carry no
        injection and supply the background predictions.
        """
        X, _X_clean, _y, params = batch
        _denoised, out = self(X)
        mean, var = self._split(out)
        mean_phys = mean * self.y_std + self.y_mean
        sigma_phys = torch.sqrt(var) * self.y_std

        targets = torch.stack([params[k] for k in self.param_names], dim=1)
        injected = ~torch.isnan(targets).any(dim=1)

        outputs = {
            "y_true": targets[injected].detach().cpu(),
            "y_pred": mean_phys[injected].detach().cpu(),
            "y_sigma": sigma_phys[injected].detach().cpu(),
        }
        if (~injected).any():
            outputs["y_pred_bg"] = mean_phys[~injected].detach().cpu()
            outputs["y_sigma_bg"] = sigma_phys[~injected].detach().cpu()
        if "snr" in params:
            outputs["snr"] = params["snr"][injected].detach().cpu()
        return outputs
