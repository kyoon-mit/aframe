import math

import numpy as np
import pytest
import torch

from train.experimental.long_conv.arch import LongConvEnergyNet

LENGTH = 8192
SAMPLE_RATE = 2048.0


def _chirp(length=LENGTH, merge=7680):
    t = (merge - np.arange(length)) / SAMPLE_RATE
    tau = np.clip(t, 1 / SAMPLE_RATE, None)
    h = tau ** -0.25 * np.cos(-493 * tau ** (5 / 8))
    h[np.arange(length) > merge] = 0
    h[: 512] *= np.hanning(1024)[:512]
    return torch.tensor(h, dtype=torch.float32)


@pytest.mark.parametrize("kernel,d_state", [("s4d", 64), ("s4d", 512), ("free", 64), ("time", 64)])
def test_shapes_and_unit_kernels(kernel, d_state):
    net = LongConvEnergyNet(channels=8, kernel=kernel, d_state=d_state)
    logits = net(torch.randn(3, 2, LENGTH))
    assert logits.shape == (3, len(net.grid_log_mc))
    assert torch.isfinite(logits).all()
    k = net.kernels()
    assert k.shape == (8, LENGTH // 2)
    assert torch.allclose(k.abs().pow(2).sum(-1), torch.ones(8), atol=1e-4)


@pytest.mark.parametrize("kernel", ["s4d", "free", "time"])
def test_gradients_reach_the_kernels(kernel):
    net = LongConvEnergyNet(channels=4, kernel=kernel, d_state=64)
    loss = net(torch.randn(2, 2, LENGTH)).logsumexp(-1).sum()
    loss.backward()
    params = [net.kernel.C, net.kernel.A_imag] if kernel == "s4d" else [getattr(net, {"free": "spectrum", "time": "samples"}[kernel])]
    for p in params + [net.out.weight]:
        assert p.grad is not None and p.grad.abs().sum() > 0


def test_white_noise_gives_unit_energy_per_sample():
    torch.manual_seed(0)
    net = LongConvEnergyNet(channels=4, kernel="free", highpass=0.0)
    L = net.length
    x = torch.randn(64, 1, LENGTH)
    spectrum = torch.fft.rfft(x, n=L * net.decimate)[..., : net.f_keep]
    xd = torch.fft.irfft(spectrum, n=L) / math.sqrt(net.decimate)
    # the resampled noise keeps unit variance up to the band edge
    assert xd.var().item() == pytest.approx(1.0, rel=0.05)


def test_free_kernel_set_to_a_chirp_finds_it():
    net = LongConvEnergyNet(channels=2, kernel="free")
    L, dec = net.length, net.decimate
    h = _chirp()
    # the chirp at the network's rate, time-reversed so the causal output peaks at the merger
    hs = torch.fft.irfft(torch.fft.rfft(h, n=L * dec)[: net.f_keep], n=L) / math.sqrt(dec)
    reversed_h = torch.flip(hs, [0])
    spec = torch.fft.fft(reversed_h)
    with torch.no_grad():
        net.spectrum[0] = torch.view_as_real(spec * 2)      # analytic: band keeps positive frequencies
    snr = 12.0
    x = (h / h.norm() * snr).expand(1, 2, -1)
    e = net.peak_energy(x)
    # both detectors hold the same signal: about 2 * snr^2 for the matched kernel
    assert e[0, 0].item() == pytest.approx(2 * snr**2, rel=0.1)
    assert e[0, 0] > 5 * e[0, 1]


def test_mixed_s4d_kernels():
    net = LongConvEnergyNet(channels=8, kernel="s4d", d_state=64, mix_from=32)
    assert net.kernel.C.shape[0] == 32 and net.mix.shape == (8, 32, 2)
    k = net.kernels()
    assert k.shape == (8, LENGTH // 2)
    loss = net(torch.randn(2, 2, LENGTH)).logsumexp(-1).sum()
    loss.backward()
    assert net.mix.grad.abs().sum() > 0 and net.kernel.C.grad.abs().sum() > 0


def test_deep_s4_energy_net():
    from train.experimental.long_conv.arch import DeepS4EnergyNet

    net = DeepS4EnergyNet(d_model=16, d_state=8, n_layers=2, num_groups=4)
    logits = net(torch.randn(2, 2, LENGTH))
    assert logits.shape == (2, len(net.grid_log_mc))
    logits.logsumexp(-1).sum().backward()
    log_mc, spread = net.estimate(logits.detach())
    assert log_mc.shape == (2,) and torch.isfinite(spread).all()


def test_resnet_grid_net():
    from train.experimental.long_conv.arch import ResNetGridNet

    net = ResNetGridNet(layers=[1, 1], norm_groups=4)
    logits = net(torch.randn(2, 2, LENGTH))
    assert logits.shape == (2, len(net.grid_log_mc))
    logits.logsumexp(-1).sum().backward()


def test_band_init_puts_tones_in_band_with_long_memory():
    net = LongConvEnergyNet(channels=8, kernel="s4d", d_state=64, s4d_init="band", memory=4.0)
    k = net.kernel
    dt = k.log_dt.exp()[:, None]
    freq = (dt * k.A_imag) / (2 * math.pi) * 1024
    efold = 1 / (dt * k.log_A_real.exp()) / 1024
    assert ((freq >= 20) & (freq < 512)).all()
    assert torch.allclose(efold, torch.full_like(efold, 4.0), rtol=1e-4)
    assert torch.isfinite(net(torch.randn(2, 2, LENGTH))).all()


def test_time_kernels_load_old_checkpoint_names():
    net = LongConvEnergyNet(channels=4, kernel="time")
    sd = {("taps" if k == "samples" else k): v for k, v in net.state_dict().items()}
    fresh = LongConvEnergyNet(channels=4, kernel="time")
    fresh.load_state_dict(sd)
    assert torch.equal(fresh.samples, net.samples)
