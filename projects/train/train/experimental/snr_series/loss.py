"""Matched-filter SNR time-series loss.

For each detector, the data ``d`` is matched-filtered with a template
``h`` over every time shift::

    z(t; h)   = 4 int d~(f) h~*(f) / S(f) e^{2 pi i f t} df
    sigma^2(h) = 4 int |h~(f)|^2 / S(f) df
    SNR(t; h) = |z(t; h)| / sigma(h)

The data and templates are already whitened, so ``S = 1`` and both
integrals become sums over samples: ``Re z`` is the cross-correlation
of ``d`` with ``h``, ``|z|`` its envelope (maximized over phase), and
``sigma = ||h||``. Unit-variance noise filtered with an unrelated
template gives ``Re z / sigma ~ N(0, 1)``.

The loss matches the denoiser's series to the true template's::

    L = mean_{b, c, t} [ SNR(t; h_den) - SNR(t; h_true) ]^2

On background ``h_true = 0`` and the target is 0 at every shift. The
denoiser's ``sigma`` is floored, ``sqrt(sigma^2 + eps^2)``, so a zero
output scores 0 instead of 0/0.
"""

from typing import Optional

import torch
import torch.nn as nn

from train.losses_denoiser_ky import TermStashMixin


def snr_series(
    data: torch.Tensor,
    template: torch.Tensor,
    eps: float = 0.0,
) -> torch.Tensor:
    """Matched-filter SNR of ``data`` against ``template`` at every shift.

    Args:
        data: whitened strain, (B, C, L).
        template: whitened template, (B, C, L).
        eps: floor on the template norm. 0 sets the SNR of an all-zero
            template to 0.

    Returns:
        (B, C, 2L - 1) SNR, one value per linear time shift.
    """
    length = data.shape[-1]
    n = 2 * length
    # zero padding to 2L makes the correlation linear, not circular
    product = torch.fft.rfft(data, n=n) * torch.fft.rfft(template, n=n).conj()

    # analytic signal: positive frequencies doubled, negative ones dropped,
    # so Re z is the correlation and Im z its quadrature
    spectrum = product.new_zeros(*product.shape[:-1], n)
    spectrum[..., : n // 2 + 1] = product
    spectrum[..., 1 : n // 2] *= 2
    z = torch.fft.ifft(spectrum, n=n)

    # reorder lags to -(L-1) .. L-1
    z = torch.cat([z[..., n - length + 1 :], z[..., :length]], dim=-1)

    sigma2 = template.pow(2).sum(dim=-1, keepdim=True)
    if eps:
        return z.abs() / (sigma2 + eps**2).sqrt()
    sigma = sigma2.sqrt()
    return torch.where(sigma > 0, z.abs() / sigma.clamp_min(1e-30), 0.0)


class SNRSeriesLoss(TermStashMixin, nn.Module):
    """MSE between the denoiser's and the true template's SNR series.

    The data is not an argument of ``forward``, which keeps the
    ``(pred, target)`` signature every denoiser loss has. The task sets
    ``data`` before each call.

    Args:
        eps: floor on the denoiser template's norm, in whitened units.
            Outputs with ``sigma`` well below it score near 0, so it sets
            how faint an output still counts as a detection.
    """

    def __init__(self, eps: float = 0.3):
        super().__init__()
        if eps <= 0:
            raise ValueError(f"eps must be > 0, got {eps}")
        self.eps = eps
        self.data: Optional[torch.Tensor] = None

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """pred, target: (B, C, L) whitened waveforms. Returns scalar."""
        if self.data is None:
            raise RuntimeError("SNRSeriesLoss needs .data set before forward")
        snr_pred = snr_series(self.data, pred, eps=self.eps)
        with torch.no_grad():
            snr_true = snr_series(self.data, target)
        loss = (snr_pred - snr_true).pow(2).mean()
        self.last_snr_pred = snr_pred.detach()
        self._stash(snr=loss)
        return loss
