"""Chirp mass as a posterior over a template bank's chirp-mass grid."""

from pathlib import Path

import numpy as np
import torch

from train.model.denoiser_ky import Denoiser


class TemplateBankClassifier(Denoiser):
    """Train ``TemplateBankNet``'s read-out with cross-entropy over its grid,
    plus a detection term.

    The chirp-mass target is a Gaussian over the grid's log chirp masses,
    centred on the true value, with width ``label_width`` times the grid
    spacing; only rows with a signal enter it. The detection logit is trained
    with binary cross-entropy on every row, weighted by ``detection_weight``.
    The estimate is the posterior peak, refined by a parabola. Batches are
    ``(X, X_clean, y, params)``, as ``DenoiserOnlyAframeDataset`` gives.
    """

    def __init__(self, *args, label_width: float = 0.5, detection_weight: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters("label_width", "detection_weight")
        self._test_scores = {k: [] for k in (
            "signal", "background", "snr", "newsnr2_signal", "newsnr2_background",
            "snr2_signal", "snr2_background",
        )}

    def _shared_step(self, batch, stage: str) -> torch.Tensor:
        X, params = batch[0], batch[3]
        chirp_mass = params["chirp_mass"]
        signal = ~torch.isnan(chirp_mass)
        logits, det = self(X)
        on_step = stage == "train"

        det_loss = torch.nn.functional.binary_cross_entropy_with_logits(det, signal.float())
        loss = self.hparams.detection_weight * det_loss
        self.log(f"{stage}/loss_detection", det_loss, on_step=on_step, on_epoch=True)
        if signal.any():
            grid = self.model.grid_log_mc
            width = self.hparams.label_width * (grid[1] - grid[0])
            log_true = chirp_mass[signal].log()
            target = torch.softmax(-((log_true[:, None] - grid[None]) ** 2) / (2 * width**2), -1)
            ce = -(target * torch.log_softmax(logits[signal], -1)).sum(-1).mean()
            loss = loss + ce
            self.log(f"{stage}/loss_ce", ce, on_step=on_step, on_epoch=True, prog_bar=True)
            with torch.no_grad():
                log_mc, _ = self.model.estimate(logits[signal])
                error = (log_mc.exp() - chirp_mass[signal]).abs() / chirp_mass[signal]
                snr = params["snr"][signal]
                for tol in (1, 2, 5, 10):
                    self.log(f"{stage}/within_{tol}pct_chirp_mass", (error < tol / 100).float().mean(), on_epoch=True, on_step=False)
                band = (snr >= 8) & (snr < 12)
                if band.any():
                    for tol in (1, 2):
                        self.log(f"{stage}/within_{tol}pct_chirp_mass_snr8-12", (error[band] < tol / 100).float().mean(), on_epoch=True, on_step=False)
        self.log(f"{stage}/beta", self.model.log_beta.exp(), on_epoch=True, on_step=False)
        return loss

    param_names = ["chirp_mass"]

    def _predict(self, X: torch.Tensor):
        """Chirp mass at the posterior peak, its spread, and the detection logit."""
        logits, det = self(X)
        log_mc, spread = self.model.estimate(logits)
        estimate = log_mc.exp()
        return estimate, estimate * spread, det

    def test_step(self, batch, _):
        """Outputs for ``PlotParamEstCallback``; noise-only rows give the
        background predictions. Detection logits, and the loudest
        template's network SNR^2 with and without the chi^2 reweighting,
        are kept for ROC curves."""
        X, params = batch[0], batch[3]
        chirp_mass = params["chirp_mass"]
        signal = ~torch.isnan(chirp_mass)
        new, plain, peak, feats = self.model.newsnr2(X, features=True)
        logits, det = self.model.head(peak)
        if self.model.detection_hidden > 0:
            det = self.model.detect(feats, new, plain)
        log_mc, spread = self.model.estimate(logits)
        estimate = log_mc.exp()
        sigma = estimate * spread
        for name, rows in (("signal", signal), ("background", ~signal)):
            self._test_scores[name].append(det[rows].detach().cpu())
            self._test_scores[f"newsnr2_{name}"].append(new[rows].detach().cpu())
            self._test_scores[f"snr2_{name}"].append(plain[rows].detach().cpu())
        self._test_scores["snr"].append(params["snr"][signal].detach().cpu())
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

    def on_test_epoch_start(self) -> None:
        for scores in self._test_scores.values():
            scores.clear()

    def on_test_epoch_end(self) -> None:
        out = Path(self.trainer.default_root_dir) / "detection_scores.npz"
        np.savez(out, **{k: torch.cat(v).numpy() for k, v in self._test_scores.items() if v})
