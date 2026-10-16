"""Where the merger falls in the training windows, using the pipeline's own functions.

Each stored waveform is replaced by an index ramp (value k at sample k), so after
slicing, the detector shift, the random segment choice and the whitening crop,
the value at a window sample is the stored-waveform sample it came from. The
coalescence sample of the stored file is 32 s * 2048 - 2 s * 2048 = 61440.
"""
import math
from types import SimpleNamespace

import torch

torch.set_num_threads(1)
torch.manual_seed(20261016)

from ml4gw import gw
from ml4gw.distributions import Cosine
from ml4gw.transforms import Whiten
from ml4gw.utils.slicing import sample_kernels, slice_kernels
from train.data.base import BaseAframeDataset
from train.experimental.oracle.data import WarpedDenoiserOnlyAframeDataset
from utils.preprocessing import PsdEstimator

FS = 2048.0
WAVE_LEN = 65536          # 32 s file waveforms
FILE_RIGHT_PAD = 2.0      # file attribute right_pad
MERGER = WAVE_LEN - int(FILE_RIGHT_PAD * FS)


def stub(merger_offset=None):
    s = SimpleNamespace()
    s.hparams = SimpleNamespace(sample_rate=FS, kernel_length=4.0, fduration=1.0,
                                left_pad=3.5, right_pad=0.0, psd_length=20.0)
    s.waveform_sampler = SimpleNamespace(right_pad=FILE_RIGHT_PAD)
    s.filter_size = BaseAframeDataset.filter_size.fget(s)
    s.left_pad_size = BaseAframeDataset.left_pad_size.fget(s)
    s.right_pad_size = BaseAframeDataset.right_pad_size.fget(s)
    if merger_offset is not None:
        s.merger_offset = merger_offset
        s.offset_size = int(round(merger_offset * FS))
    return s


print(f"coalescence sample in the stored waveform: {MERGER}")
s = stub()
print(f"filter_size {s.filter_size}, left_pad_size {s.left_pad_size}, right_pad_size {s.right_pad_size}")

# whitening keeps the centre of the 5 s segment: check the crop and that the filter does not move a peak
whiten = Whiten(1.0, FS, highpass=20.0)
x = torch.zeros(1, 1, 10240, dtype=torch.float64)
for p in (2000, 5000, 9000):
    x.zero_()
    x[..., p] = 1.0
    y = whiten(x, torch.ones(1, 1, 10240 // 2 + 1, dtype=torch.float64))
    print(f"whiten: input length 10240 -> {y.shape[-1]}; impulse at {p} -> peak at {int(y.abs().argmax())} (p - 1024 = {p - 1024})")

# the PSD estimator returns the last kernel_length + fduration seconds of each 25 s sample
est = PsdEstimator(5.0, FS, 2.0, fast=True)
ramp = torch.arange(int(25 * FS), dtype=torch.float64).repeat(3, 2, 1)
xk, _ = est(ramp)
print(f"PSD estimator: segment for whitening is samples {int(xk[0,0,0])} to {int(xk[0,0,-1])} of {ramp.shape[-1]}")

tensors, vertices = gw.get_ifo_geometry("H1", "L1")
C = 299792458.0
vmax = vertices.double().norm(dim=-1)
print(f"|vertex| / c: H1 {vmax[0] / C * 1e3:.4f} ms, L1 {vmax[1] / C * 1e3:.4f} ms; "
      f"in samples {vmax[0] / C * FS:.3f}, {vmax[1] / C * FS:.3f} (shift is truncated to an integer)")
dec_dist = Cosine()


def run(name, slicer, s, n_chunks=8, B=2048, shift=True):
    wave = torch.arange(WAVE_LEN, dtype=torch.float32).repeat(1, 2, 1)
    sliced = slicer(s, wave)  # deterministic
    tm_geo, tm_det, starts = [], [], []
    for _ in range(n_chunks):
        resp = sliced.repeat(B, 1, 1)
        if shift:
            dec = dec_dist.sample((B,))
            phi = torch.rand(B) * 2 * math.pi - math.pi
            theta = math.pi / 2 - dec
            det = gw.shift_responses(resp, theta, phi, vertices, FS)
        else:
            det = resp
        k_geo, idx = sample_kernels(resp, kernel_size=10240, coincident=True, return_idx=True)
        # same start index for the shifted responses, as in the pipeline (shift happens before sampling)
        k_det = slice_kernels(det, idx, 10240)
        w_geo = k_geo[..., 1024:-1024]
        w_det = k_det[..., 1024:-1024]
        assert torch.all(w_geo[..., -1] - w_geo[..., 0] == 8191)
        assert torch.all(w_det[..., -1] - w_det[..., 0] == 8191)
        tm_geo.append((MERGER - w_geo[..., 0]) / FS)
        tm_det.append((MERGER - w_det[..., 0]) / FS)
        starts.append(idx)
    tm_geo = torch.cat(tm_geo)
    tm_det = torch.cat(tm_det)
    starts = torch.cat(starts)
    print(f"[{name}] sliced length {sliced.shape[-1]}, start index drawn {int(starts.min())} to {int(starts.max())} "
          f"({starts.unique().numel()} distinct of {starts.numel()} draws)")
    print(f"[{name}] merger, Earth centre: {tm_geo.min():.6f} to {tm_geo.max():.6f} s after the window's first sample "
          f"(samples {int(tm_geo.min() * FS)} to {int(tm_geo.max() * FS)}); mean {tm_geo.mean():.4f} s")
    print(f"[{name}] merger, each detector: {tm_det.min():.6f} to {tm_det.max():.6f} s; "
          f"largest shift {((tm_det - tm_geo).abs().max() * FS):.0f} samples; H1 minus L1 at most {((tm_det[:, 0] - tm_det[:, 1]).abs().max() * FS):.0f} samples")
    return tm_geo, tm_det


g, d = run("signal-only and noisy runs (left pad 3.5 s, right pad 0)", BaseAframeDataset.slice_waveforms, stub())
print(f"   fraction of detector windows whose merger lies after the last sample: {(d * FS > 8191).float().mean():.4f}")
g2, d2 = run("warp, merger_offset 25.5 s", WarpedDenoiserOnlyAframeDataset.slice_waveforms, stub(25.5))
print(f"   window end before the merger: {(g2 - 4.0).min():.6f} to {(g2 - 4.0).max():.6f} s (Earth centre)")
g3, _ = run("warp, merger_offset 0", WarpedDenoiserOnlyAframeDataset.slice_waveforms, stub(0.0), n_chunks=2, shift=False)
