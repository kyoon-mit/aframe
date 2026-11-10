"""Linear prediction of raw strain from its own past, from a spectrum."""

import math

import torch


@torch.no_grad()
def prediction_error_filter(psd, num_samples, sample_rate, length):
    """Causal filter (B, C, ``length``) whose output on noise with the
    one-sided spectrum ``psd`` (B, C, F) is white with unit variance: the
    error of the best linear prediction of each sample from the earlier
    ones, divided by its spread. ``psd`` is interpolated to ``num_samples``
    points and the filter is the minimum-phase factor of its inverse, cut
    to ``length`` samples with the second half of a Hann window.
    """
    psd = psd.double()
    num_freqs = num_samples // 2 + 1
    if psd.size(-1) != num_freqs:
        shape = psd.shape[:-1]
        psd = torch.nn.functional.interpolate(
            psd.reshape(1, -1, psd.size(-1)), size=(num_freqs,), mode="linear"
        ).reshape(*shape, num_freqs)
    log_gain = 0.5 * (math.log(2 / sample_rate) - torch.log(psd))
    cepstrum = torch.fft.irfft(log_gain, n=num_samples)
    fold = torch.zeros(num_samples, dtype=psd.dtype, device=psd.device)
    fold[0] = 1
    fold[1 : (num_samples + 1) // 2] = 2
    if num_samples % 2 == 0:
        fold[num_samples // 2] = 1
    response = torch.fft.irfft(
        torch.exp(torch.fft.rfft(cepstrum * fold, n=num_samples)), n=num_samples
    )
    taper = torch.hann_window(2 * length, dtype=psd.dtype, device=psd.device)
    return response[..., :length] * taper[length:]


@torch.no_grad()
def prediction_errors(X, psd, sample_rate, length):
    """Raw strain (B, C, N) to its linear prediction errors (B, C, N -
    ``length``), in float32, and the first filter coefficient (B, C). The
    first ``length`` samples, where the filter has not seen its full past,
    are dropped.
    """
    N = X.size(-1)
    taps = prediction_error_filter(psd, N, sample_rate, length)
    spectrum = torch.fft.rfft(X.double(), n=N) * torch.fft.rfft(taps, n=N)
    errors = torch.fft.irfft(spectrum, n=N)[..., length:]
    return errors.float(), taps[..., 0]
