import pytest
import torch

from architectures.networks.lowband import LowBandSmoother

LENGTH, SAMPLE_RATE, CUTOFF = 8192, 2048.0, 20.0
LOW = int(CUTOFF * LENGTH / SAMPLE_RATE)  # first bin at the cutoff


@pytest.fixture
def smoother():
    return LowBandSmoother(LENGTH, SAMPLE_RATE, CUTOFF, num_coeffs=4)


@pytest.fixture
def y():
    torch.manual_seed(0)
    return torch.randn(3, 2, LENGTH)


def test_bins_from_settings(smoother):
    assert smoother.dft.shape == (LENGTH, LOW - 1)


def test_other_bins_unchanged(smoother, y):
    before = torch.fft.rfft(y, dim=-1)
    after = torch.fft.rfft(smoother(y), dim=-1)
    for b in (slice(0, 1), slice(LOW, None)):
        assert torch.allclose(after[..., b], before[..., b], atol=1e-3)


def test_low_band_on_the_fit(smoother, y):
    """Smoothing is a projection: smoothing twice changes nothing."""
    once = smoother(y)
    assert torch.allclose(smoother(once), once, atol=1e-4)
    rough = torch.fft.rfft(y, dim=-1)[..., 1:LOW].abs().log().diff(dim=-1)
    smooth = torch.fft.rfft(once, dim=-1)[..., 1:LOW].abs().log().diff(dim=-1)
    assert smooth.abs().mean() < 0.1 * rough.abs().mean()


def test_phase_kept(smoother, y):
    before = torch.fft.rfft(y, dim=-1)[..., 1:LOW]
    after = torch.fft.rfft(smoother(y), dim=-1)[..., 1:LOW]
    assert torch.allclose(after.angle(), before.angle(), atol=1e-3)


def test_matches_full_fft(smoother, y):
    Y = torch.fft.rfft(y.double(), dim=-1)
    low = Y[..., 1:LOW]
    fit = torch.exp(low.abs().add(smoother.eps).log() @ smoother.projection.double())
    Y[..., 1:LOW] = low * fit / (low.abs() + smoother.eps)
    expected = torch.fft.irfft(Y, n=LENGTH, dim=-1).float()
    assert torch.allclose(smoother(y), expected, atol=1e-4)


def test_gradient_finite(smoother, y):
    y = y.clone().requires_grad_()
    smoother(y).pow(2).sum().backward()
    assert torch.isfinite(y.grad).all()


def test_too_few_bins():
    with pytest.raises(ValueError):
        LowBandSmoother(LENGTH, SAMPLE_RATE, 0.5, num_coeffs=4)
