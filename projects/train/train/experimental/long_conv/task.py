"""Chirp mass as a classification over a log chirp-mass grid."""

import torch

from train.model.denoiser_ky import Denoiser


class LongConvClassifier(Denoiser):
    """Train ``LongConvEnergyNet`` with cross-entropy over its grid.

    The target is a Gaussian over the grid's log chirp masses, centred on the
    true value, with width ``label_width`` times the grid spacing. Noise-only
    rows carry no chirp mass and are skipped. The estimate is the logits'
    peak, refined by a parabola. Batches are ``(X, X_clean, y, params)``, as
    ``DenoiserOnlyAframeDataset`` gives. S4D parameters get ``ssm_lr`` and no
    weight decay, as in ``Denoiser``.
    """

    def __init__(self, *args, label_width: float = 0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters("label_width")

    def _shared_step(self, batch, stage: str) -> torch.Tensor:
        X, params = batch[0], batch[3]
        chirp_mass = params["chirp_mass"]
        signal = ~torch.isnan(chirp_mass)
        logits = self(X)
        if not signal.any():
            return logits.sum() * 0.0

        logits = logits[signal]
        grid = self.model.grid_log_mc
        width = self.hparams.label_width * (grid[1] - grid[0])
        log_true = chirp_mass[signal].log()
        target = torch.softmax(-((log_true[:, None] - grid[None]) ** 2) / (2 * width**2), -1)
        loss = -(target * torch.log_softmax(logits, -1)).sum(-1).mean()

        on_step = stage == "train"
        self.log(f"{stage}/loss_ce", loss, on_step=on_step, on_epoch=True, prog_bar=True)
        with torch.no_grad():
            log_mc, _ = self.model.estimate(logits)
            error = (log_mc.exp() - chirp_mass[signal]).abs() / chirp_mass[signal]
            snr = params["snr"][signal]
            for tol in (1, 2, 5, 10):
                self.log(f"{stage}/within_{tol}pct_chirp_mass", (error < tol / 100).float().mean(), on_epoch=True, on_step=False)
            band = (snr >= 8) & (snr < 12)
            if band.any():
                for tol in (1, 2):
                    self.log(f"{stage}/within_{tol}pct_chirp_mass_snr8-12", (error[band] < tol / 100).float().mean(), on_epoch=True, on_step=False)
        return loss

    param_names = ["chirp_mass"]

    def _predict(self, X: torch.Tensor):
        """Chirp mass at the logits' peak and the softmax spread, in M_sun."""
        log_mc, spread = self.model.estimate(self(X))
        estimate = log_mc.exp()
        return estimate, estimate * spread

    def test_step(self, batch, _):
        """Outputs for ``PlotParamEstCallback``; noise-only rows give the
        background predictions."""
        X, params = batch[0], batch[3]
        chirp_mass = params["chirp_mass"]
        signal = ~torch.isnan(chirp_mass)
        estimate, sigma = self._predict(X)
        outputs = {
            "y_true": chirp_mass[signal, None].detach().cpu(),
            "y_pred": estimate[signal, None].detach().cpu(),
            "y_sigma": sigma[signal, None].detach().cpu(),
            "snr": params["snr"][signal].detach().cpu(),
        }
        if (~signal).any():
            outputs["y_pred_bg"] = estimate[~signal, None].detach().cpu()
            outputs["y_sigma_bg"] = sigma[~signal, None].detach().cpu()
        return outputs
