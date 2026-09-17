from typing import Optional

import torch
from ml4gw.spectral import truncate_inverse_power_spectrum

from train.model.denoiser_ky import Denoiser


class HighpassDenoiser(Denoiser):
    """``Denoiser`` whose output is highpassed the way its input was.

    The data is whitened with a highpass folded into the whitening filter,
    but nothing stops the network emitting power below that corner. This
    applies the same highpass to the network's output, built each batch
    from that batch's own PSDs by the same ml4gw routine that built the
    data's filter, so the output is band limited exactly as the target is.
    The filter is linear, so the gradient is band limited too and the
    network receives no learning signal below the corner.

    Expects batches of ``(X, X_clean, y, params, psds)``, as
    ``PsdDenoiserDataset`` provides.

    Args:
        highpass: corner frequency in Hz. Must match the data's.
        fduration: length of the whitening filter's time-domain response
            in seconds. Must match the data's.
        sample_rate: of the model's input and output.
    """

    def __init__(
        self,
        *args,
        highpass: float = 20.0,
        fduration: float = 1.0,
        sample_rate: float = 2048.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.highpass = highpass
        self.fduration = fduration
        self.sample_rate = sample_rate
        self.save_hyperparameters("highpass", "fduration", "sample_rate")
        # the batch's PSDs, set by _shared_step for forward to read
        self._psds: Optional[torch.Tensor] = None

    def _response(self, psds: torch.Tensor, length: int) -> torch.Tensor:
        """Highpass response for these PSDs, on ``length`` samples.

        Whitening divides the spectrum by the truncated PSD's square root.
        The ratio of that with and without the highpass is the highpass
        alone, as ml4gw applies it.
        """
        num_freqs = length // 2 + 1
        psds = psds.double()
        if psds.shape[-1] != num_freqs:
            psds = torch.nn.functional.interpolate(
                psds, size=(num_freqs,), mode="linear"
            )
        with_hp = truncate_inverse_power_spectrum(
            psds.clone(), self.fduration, self.sample_rate, self.highpass
        )
        without = truncate_inverse_power_spectrum(
            psds.clone(), self.fduration, self.sample_rate
        )
        denominator = torch.nan_to_num(without**-0.5).clamp_min(1e-30)
        return (torch.nan_to_num(with_hp**-0.5) / denominator).float()

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        output = super().forward(X)
        if self._psds is None:
            return output
        response = self._response(self._psds, output.shape[-1])
        spectrum = torch.fft.rfft(output, dim=-1) * response
        return torch.fft.irfft(spectrum, n=output.shape[-1], dim=-1)

    def _shared_step(self, batch, stage: str) -> torch.Tensor:
        self._psds = batch[4]
        try:
            return super()._shared_step(batch, stage)
        finally:
            self._psds = None
