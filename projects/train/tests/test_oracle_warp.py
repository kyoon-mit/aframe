import math
from types import SimpleNamespace

import torch

from train.experimental.oracle.data import WarpedDenoiserOnlyAframeDataset as D


class _Fake(SimpleNamespace):
    warp = D.warp
    peak_time = D.peak_time


def _ds(fs=2048.0, warp_rate=None):
    return _Fake(hparams=SimpleNamespace(sample_rate=fs), warp_rate=warp_rate, upsample=8)


def _chirp(t, t_c, mc=1.2):
    tsun = 4.925491e-6
    tau = (t_c - t).clamp(min=1e-4)
    phi = -2 * (tau / (5 * tsun * mc)) ** 0.625
    return tau ** -0.25 * torch.cos(phi) * (t < t_c)


def _freq_spread(x):
    """Ratio of the highest to the lowest mean instantaneous frequency over the
    middle 80% of a real signal, in eight equal segments."""
    import numpy as np
    L = x.shape[-1]
    F = torch.fft.fft(x)
    h = torch.zeros(L); h[0] = 1; h[1:L // 2] = 2; h[L // 2] = 1
    f = np.diff(np.unwrap(torch.angle(torch.fft.ifft(F * h)).numpy()))[L // 10: -L // 10]
    m = [seg.mean() for seg in np.array_split(f, 8)]
    return max(m) / min(m)


def test_merger_window_becomes_a_tone():
    fs = 2048.0
    t = torch.arange(8192) / fs
    t_c = 3.8
    x = _chirp(t, t_c)[None, None]
    ds = _ds(fs)
    w = ds.warp(x, torch.tensor([[t_c]]))
    assert w.shape == (1, 1, 8192)
    assert _freq_spread(x[0, 0, : int(3.7 * fs)]) > 1.5
    assert _freq_spread(w[0, 0]) < 1.05


def test_premerger_window_rate_and_flatness():
    fs = 2048.0
    t = torch.arange(8192) / fs
    t_c = 4.0 + 25.3
    x = _chirp(t, t_c)[None, None]
    ds = _ds(fs, warp_rate=256.0)
    w = ds.warp(x, torch.tensor([[t_c]]))
    assert w.shape == (1, 1, 1024)
    assert _freq_spread(w[0, 0]) < _freq_spread(x[0, 0])
    assert _freq_spread(w[0, 0]) < 1.01


def test_peak_time():
    fs = 2048.0
    t = torch.arange(8192) / fs
    x = _chirp(t, 3.6)[None, None]
    assert abs(_ds(fs).peak_time(x).item() - 3.6) < 0.01
