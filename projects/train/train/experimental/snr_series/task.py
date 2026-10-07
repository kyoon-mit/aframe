"""Denoiser trained on the matched-filter SNR series."""

import torch

from train.experimental.snr_series.loss import SNRSeriesLoss
from train.model.denoiser_ky import Denoiser


class SNRSeriesDenoiser(Denoiser):
    """``Denoiser`` whose loss sees the data as well as the target.

    Hands each batch's whitened strain to ``SNRSeriesLoss`` and logs the
    peak SNR of the denoiser's output, on background rows and on signal
    rows separately, as the detection metric.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not isinstance(self.denoiser_loss, SNRSeriesLoss):
            raise TypeError("SNRSeriesDenoiser needs an SNRSeriesLoss")
        if self.hparams.predict_residual:
            raise ValueError("SNRSeriesDenoiser needs the waveform output")

    def _shared_step(self, batch, stage: str) -> torch.Tensor:
        X, X_clean = batch[0], batch[1]
        self.denoiser_loss.data = X
        try:
            loss = super()._shared_step(batch, stage)
        finally:
            self.denoiser_loss.data = None

        # peak over shifts, max over detectors: one number per row
        peak = self.denoiser_loss.last_snr_pred.amax(dim=(-2, -1))
        carries_signal = X_clean.pow(2).sum(dim=(-2, -1)) > 0
        on_step = stage == "train"
        for mask, name in ((~carries_signal, "bkg"), (carries_signal, "sig")):
            if mask.any():
                self.log(
                    f"{stage}/snr_peak_{name}",
                    peak[mask].mean(),
                    on_step=on_step,
                    on_epoch=True,
                )
        return loss
