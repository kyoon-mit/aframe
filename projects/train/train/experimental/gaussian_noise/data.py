"""Aframe batches with the real background replaced by Gaussian noise of the
same spectrum.

Each raw background stretch is measured by the data module's own PSD
estimator (its first psd_length seconds), and a fresh Gaussian stretch of the
same length is drawn with that one-sided PSD. Injection, SNR scaling and
whitening then run unchanged on the Gaussian stretch, so the signals, their
SNRs and the whitened view are as in the real-noise runs, without lines that
change in time, drifts or glitches.
"""

import torch

from train.data.supervised.denoiser import DenoiserOnlyAframeDataset


def gaussian_copy(X: torch.Tensor, psd: torch.Tensor, sample_rate: float) -> torch.Tensor:
    """Gaussian noise of the shape of ``X`` (B, C, L) whose one-sided PSD is
    ``psd`` (B, C, F), given on a uniform grid from 0 Hz to Nyquist and
    interpolated linearly to the rfft grid of length L."""
    size, bins = X.shape[-1], psd.shape[-1]
    freqs = torch.fft.rfftfreq(size, 1 / sample_rate, device=X.device, dtype=torch.float64)
    pos = freqs / (sample_rate / 2) * (bins - 1)
    lo = pos.floor().long().clamp(max=bins - 2)
    frac = pos - lo
    psd = psd.double()
    S = psd[..., lo] * (1 - frac) + psd[..., lo + 1] * frac
    white = torch.randn(X.shape, device=X.device, dtype=torch.float64)
    noise = torch.fft.irfft(torch.fft.rfft(white) * (S * sample_rate / 2).sqrt(), n=size)
    return noise.to(X.dtype)


class GaussianNoiseDenoiserDataset(DenoiserOnlyAframeDataset):
    """``DenoiserOnlyAframeDataset`` on Gaussian copies of the background, in
    training, validation and test alike."""

    def inject(self, X, waveforms, params):
        _, psd = self.psd_estimator(X)
        X = gaussian_copy(X, psd, self.hparams.sample_rate)
        return super().inject(X=X, waveforms=waveforms, params=params)
