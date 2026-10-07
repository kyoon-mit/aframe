import math

import pytest
import torch

from train.experimental.snr_series.loss import SNRSeriesLoss, snr_series


def _chirp(length=4096, sample_rate=2048.0, f0=30.0, f1=300.0):
    t = torch.arange(length) / sample_rate
    duration = length / sample_rate
    phase = 2 * math.pi * (f0 * t + 0.5 * (f1 - f0) / duration * t**2)
    return torch.sin(phase) * torch.hann_window(length)


def _scaled(waveform, snr):
    return waveform * snr / waveform.norm()


def test_noise_free_peak_is_optimal_snr():
    h = _scaled(_chirp(), 8.0).view(1, 1, -1)
    snr = snr_series(h, h)
    assert snr.max().item() == pytest.approx(8.0, rel=1e-3)
    # the peak sits at zero shift
    assert snr.argmax().item() == h.shape[-1] - 1


def test_peak_is_phase_invariant():
    length = 4096
    h = _scaled(_chirp(length), 8.0)
    spectrum = torch.fft.rfft(h)
    shifted = torch.fft.irfft(spectrum * 1j, n=length)
    snr = snr_series(shifted.view(1, 1, -1), h.view(1, 1, -1))
    assert snr.max().item() == pytest.approx(8.0, rel=2e-2)


def test_unrelated_template_is_rayleigh():
    torch.manual_seed(0)
    noise = torch.randn(64, 2, 4096)
    h = _chirp().expand(64, 2, -1)
    snr = snr_series(noise, h)
    # central shifts only, where the overlap is complete
    centre = snr[..., 4096 - 1 - 200 : 4096 - 1 + 200]
    assert centre.mean().item() == pytest.approx(math.sqrt(math.pi / 2), rel=0.05)


def test_zero_template_scores_zero():
    noise = torch.randn(2, 2, 1024)
    assert snr_series(noise, torch.zeros_like(noise)).abs().max() == 0
    snr = snr_series(noise, torch.zeros_like(noise), eps=0.3)
    assert snr.abs().max() == 0


def test_loss_zero_for_true_template():
    torch.manual_seed(0)
    h = _scaled(_chirp(), 8.0).expand(4, 2, -1).clone()
    data = h + torch.randn_like(h)
    loss_fn = SNRSeriesLoss(eps=1e-4)
    loss_fn.data = data
    assert loss_fn(h, h).item() < 1e-6


def test_copying_background_is_penalized():
    torch.manual_seed(0)
    noise = torch.randn(4, 2, 4096)
    loss_fn = SNRSeriesLoss(eps=0.3)
    loss_fn.data = noise
    silent = loss_fn(torch.zeros_like(noise), torch.zeros_like(noise))
    copied = loss_fn(noise, torch.zeros_like(noise))
    assert silent.item() == 0
    assert copied.item() > 1.0


def test_needs_data():
    with pytest.raises(RuntimeError):
        SNRSeriesLoss()(torch.zeros(1, 1, 8), torch.zeros(1, 1, 8))
