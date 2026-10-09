"""Chirp mass as a classification over the kernels' chirp masses."""

import torch

from train.model.denoiser_ky import Denoiser


class ChirpKernelClassifier(Denoiser):
    """Train ``ChirpKernelNet`` with cross-entropy over its kernel grid.

    The target is a Gaussian over the kernels' log chirp masses, centred
    on the true value, with width ``label_width`` times the mean kernel
    spacing. The estimate is the chirp mass of the most likely kernel.
    Noise-only rows carry no chirp mass and are skipped. Batches are
    ``(X, X_clean, y, params)``, as ``DenoiserOnlyAframeDataset`` gives.
    """

    SSM_PARAM_NAMES = Denoiser.SSM_PARAM_NAMES + ("log_mc",)

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
        log_true = chirp_mass[signal].log()
        log_k = self.model.kernels.log_mc
        spacing = (log_k[1:] - log_k[:-1]).abs().mean().detach()
        width = self.hparams.label_width * spacing
        target = torch.softmax(
            -((log_true[:, None] - log_k[None].detach()) ** 2) / (2 * width**2), -1
        )
        loss = -(target * torch.log_softmax(logits, -1)).sum(-1).mean()

        on_step = stage == "train"
        self.log(f"{stage}/loss_ce", loss, on_step=on_step, on_epoch=True, prog_bar=True)
        with torch.no_grad():
            estimate = log_k[logits.argmax(-1)].exp()
            error = (estimate - chirp_mass[signal]).abs() / chirp_mass[signal]
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
        """Chirp mass of the most likely kernel, and the softmax spread."""
        probs = torch.softmax(self(X), -1)
        kernel_mc = self.model.kernels.log_mc.exp()
        estimate = kernel_mc[probs.argmax(-1)]
        mean = (probs * kernel_mc).sum(-1, keepdim=True)
        spread = (probs * (kernel_mc - mean) ** 2).sum(-1).sqrt()
        return estimate, spread

    def test_step(self, batch, _):
        """Outputs for ``PlotParamEstCallback``; noise-only rows give the
        background predictions."""
        X, params = batch[0], batch[3]
        chirp_mass = params["chirp_mass"]
        signal = ~torch.isnan(chirp_mass)
        estimate, spread = self._predict(X)
        outputs = {
            "y_true": chirp_mass[signal, None].detach().cpu(),
            "y_pred": estimate[signal, None].detach().cpu(),
            "y_sigma": spread[signal, None].detach().cpu(),
            "snr": params["snr"][signal].detach().cpu(),
        }
        if (~signal).any():
            outputs["y_pred_bg"] = estimate[~signal, None].detach().cpu()
            outputs["y_sigma_bg"] = spread[~signal, None].detach().cpu()
        return outputs
