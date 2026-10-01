import math

import torch
import torch.nn as nn


class LowBandSmoother(nn.Module):
    """Replace the spectral magnitude below a corner with a smooth fit.

    For every FFT bin ``0 < f < cutoff``, ``log|Y(f)|`` is replaced by its
    least-squares polynomial fit in ``log f`` and the phase is kept. Bins at
    and above the cutoff, and the DC bin, are left unchanged. Only the low
    bins are transformed, by direct DFT, so no full FFT is computed.

    Args:
        length: samples per window.
        sample_rate: sample rate in Hz.
        cutoff: corner frequency in Hz.
        num_coeffs: polynomial coefficients in the fit.
    """

    def __init__(
        self,
        length: int,
        sample_rate: float,
        cutoff: float,
        num_coeffs: int = 4,
        eps: float = 1e-6,
    ):
        super().__init__()
        bins = torch.arange(1, length // 2 + 1, dtype=torch.float64)
        bins = bins[bins * sample_rate / length < cutoff]
        if len(bins) < num_coeffs:
            raise ValueError(
                f"{len(bins)} bins below {cutoff} Hz, fewer than "
                f"num_coeffs={num_coeffs}"
            )
        self.eps = eps

        n = torch.arange(length, dtype=torch.float64)
        dft = torch.exp(-2j * math.pi * torch.outer(n, bins) / length)
        # a change at bin k of a real signal appears at k and -k, so its
        # inverse is twice the real part over the length
        self.register_buffer("dft", dft.to(torch.complex64), persistent=False)
        self.register_buffer(
            "idft",
            (dft.conj().T * (2.0 / length)).to(torch.complex64),
            persistent=False,
        )

        x = torch.log(bins * sample_rate / length)
        x = (x - x.mean()) / x.std()
        basis = torch.stack([x**p for p in range(num_coeffs)], dim=1)
        projection = basis @ torch.linalg.pinv(basis)
        self.register_buffer(
            "projection", projection.float(), persistent=False
        )

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        """y: (B, C, length) -> (B, C, length)."""
        low = y.to(self.dft.dtype) @ self.dft
        magnitude = low.abs() + self.eps
        fit = torch.exp(torch.log(magnitude) @ self.projection)
        change = low * (fit / magnitude) - low
        return y + (change @ self.idft).real
