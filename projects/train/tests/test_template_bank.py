import math

import numpy as np
import pytest
import torch

from train.experimental.template_bank.arch import TemplateBankNet

SAMPLE_RATE = 2048.0
LENGTH = 8192
MERGE = 7680


def _chirp(rate, length=LENGTH, merge=MERGE):
    """A chirp ending at ``merge``; ``rate`` sets how fast frequency grows."""
    t = (merge - np.arange(length)) / SAMPLE_RATE
    tau = np.clip(t, 1 / SAMPLE_RATE, None)
    phase = -493 * (tau * rate) ** (5 / 8)  # 30 Hz at 3.75 s, 655 Hz at 1 ms
    h = tau ** -0.25 * np.cos(phase)
    h[np.arange(length) > merge] = 0
    h[: int(0.25 * SAMPLE_RATE)] *= np.hanning(int(0.5 * SAMPLE_RATE))[: int(0.25 * SAMPLE_RATE)]
    return h.astype(np.float32)


@pytest.fixture(scope="module")
def bank_file(tmp_path_factory):
    rates = np.exp(np.linspace(0.0, 0.3, 7))
    grid_log_mc = np.log(np.linspace(1.0, 1.6, 7))
    path = tmp_path_factory.mktemp("bank") / "bank.npz"
    np.savez(
        path,
        templates=np.stack([_chirp(r) for r in rates]),
        grid_index=np.arange(7),
        spin=np.zeros(7),
        grid_log_mc=grid_log_mc,
        psd=np.ones((2, 2049)),
    )
    return str(path), rates


def _whitened_injection(net, template, snr, shift=0):
    """The template band-limited like the filters, scaled to the given
    optimal SNR, in both detectors."""
    x = torch.tensor(np.roll(template, shift), dtype=torch.float64)
    spectrum = torch.fft.rfft(x)
    freqs = torch.fft.rfftfreq(LENGTH, 1 / SAMPLE_RATE)
    spectrum[freqs > SAMPLE_RATE / net.decimate / 2] = 0
    x = torch.fft.irfft(spectrum, n=LENGTH)
    x = x / x.norm() * snr
    return x.float().expand(1, 2, -1).clone()


def test_noise_gives_unit_snr2_per_quadrature(bank_file):
    net = TemplateBankNet(bank_file[0], highpass=None)
    torch.manual_seed(0)
    noise = torch.randn(256, 2, LENGTH)
    # at zero lag every template lies inside the window
    rho2 = net.snr2_series(noise)[..., net.max_lag]
    assert rho2.mean().item() == pytest.approx(2.0, rel=0.05)


def test_injected_template_peaks_at_its_snr(bank_file):
    net = TemplateBankNet(bank_file[0], highpass=None)
    template = np.load(bank_file[0])["templates"][3]
    x = _whitened_injection(net, template, 10.0, shift=-200)
    peak = net.peak_snr2(x)
    # two detectors, same signal: network SNR^2 = 2 * 10^2
    assert peak[0, 3].item() == pytest.approx(200.0, rel=0.02)
    assert peak[0].argmax().item() == 3


def test_peak_is_phase_invariant(bank_file):
    net = TemplateBankNet(bank_file[0], highpass=None)
    template = np.load(bank_file[0])["templates"][2]
    x = _whitened_injection(net, template, 10.0)
    rotated = torch.fft.irfft(torch.fft.rfft(x) * 1j, n=LENGTH)
    assert net.peak_snr2(rotated)[0, 2].item() == pytest.approx(net.peak_snr2(x)[0, 2].item(), rel=0.02)


def test_estimate_lands_on_injected_grid_point(bank_file):
    net = TemplateBankNet(bank_file[0], highpass=None)
    template = np.load(bank_file[0])["templates"][4]
    torch.manual_seed(1)
    x = _whitened_injection(net, template, 30.0) + torch.randn(1, 2, LENGTH)
    logits, det = net(x)
    log_mc, spread = net.estimate(logits)
    grid = net.grid_log_mc
    assert logits.argmax(-1).item() == 4
    assert abs(log_mc.item() - grid[4].item()) < (grid[5] - grid[4]).item() / 2
    assert det.item() > net(torch.randn(1, 2, LENGTH))[1].item()


def test_missing_spin_is_ignored(tmp_path):
    grid_log_mc = np.log([1.0, 1.1, 1.2])
    path = tmp_path / "bank.npz"
    np.savez(
        path,
        templates=np.stack([_chirp(r) for r in (1.0, 1.05, 1.1, 1.15)]),
        grid_index=np.array([0, 1, 1, 2]),
        spin=np.array([0.0, 0.0, 0.1, 0.0]),
        grid_log_mc=grid_log_mc,
        psd=np.ones((2, 2049)),
    )
    net = TemplateBankNet(str(path), highpass=None)
    assert net.table.tolist() == [[0, -1], [1, 2], [3, -1]]
    logits, _ = net(torch.randn(2, 2, LENGTH))
    assert torch.isfinite(logits).all()


def test_chi2_keeps_signals_and_downweights_glitches(bank_file):
    net = TemplateBankNet(bank_file[0], highpass=None)
    template = np.load(bank_file[0])["templates"][3]
    torch.manual_seed(2)
    noise = torch.randn(64, 2, LENGTH)
    signal = noise + _whitened_injection(net, template, 15.0)
    new, plain, _ = net.newsnr2(signal)
    # a matching signal passes the chi^2 test: little reweighting
    assert (new / plain).median().item() > 0.9
    # a loud short burst in one detector, where the templates sweep 150 Hz
    t = (torch.arange(LENGTH) - 7570) / SAMPLE_RATE
    burst = 40 * torch.exp(-(t / 0.01) ** 2) * torch.cos(2 * math.pi * 150 * t)
    glitch = noise.clone()
    glitch[:, 0] += burst / burst.norm() * 200
    new_g, plain_g, _ = net.newsnr2(glitch)
    assert plain_g.median().item() > 100
    assert (new_g / plain_g).median().item() < 0.3


def test_newsnr_matches_peak_snr(bank_file):
    net = TemplateBankNet(bank_file[0], highpass=None)
    torch.manual_seed(3)
    x = torch.randn(8, 2, LENGTH)
    _, plain, peak = net.newsnr2(x)
    assert torch.allclose(peak, net.peak_snr2(x), rtol=1e-4)
    assert torch.allclose(plain, peak.amax(-1), rtol=1e-4)


def test_features_of_a_coincident_signal(bank_file):
    net = TemplateBankNet(bank_file[0], highpass=None)
    template = np.load(bank_file[0])["templates"][5]
    torch.manual_seed(4)
    x = torch.randn(4, 2, LENGTH) + _whitened_injection(net, template, 20.0)
    new, plain, peak, feats = net.newsnr2(x, features=True)
    assert feats.shape == (4, len(TemplateBankNet.FEATURES))
    f = dict(zip(TemplateBankNet.FEATURES, feats.T))
    # same signal in both detectors: no lag or phase offset, SNR^2 ~ 400 each
    assert f["lag_offset"].abs().max().item() <= 0.2
    assert f["cos_phase_offset"].min().item() > 0.95
    assert (f["rho2_first"] + f["rho2_second"]).sub(plain).abs().max().item() < 1e-2 * plain.max().item()
    assert torch.allclose(f["log_chirp_mass"], net.grid_log_mc[5].expand(4))


def test_detection_mlp_trains_on_features(bank_file):
    net = TemplateBankNet(bank_file[0], highpass=None, detection_hidden=8)
    template = np.load(bank_file[0])["templates"][2]
    torch.manual_seed(5)
    x = torch.randn(8, 2, LENGTH)
    x[:4] += _whitened_injection(net, template, 12.0)
    logits, det = net(x)
    assert logits.shape == (8, 7) and det.shape == (8,)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(det, (torch.arange(8) < 4).float())
    loss.backward()
    assert net.det_net[0].weight.grad.abs().sum() > 0
