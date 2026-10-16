"""Trace the merger index through the training pipeline's own functions (CPU)."""
import time
from types import SimpleNamespace

import torch

t0 = time.time()
from lightning.pytorch.utilities.parsing import AttributeDict
from ml4gw.utils.slicing import sample_kernels
from ml4gw import gw
from ml4gw.constants import C
from train.data.base import BaseAframeDataset
from train.experimental.oracle.data import WarpedDenoiserOnlyAframeDataset

print(f"imports {time.time() - t0:.1f} s")

fs = 2048
hp = AttributeDict(sample_rate=2048.0, kernel_length=4.0, left_pad=3.5, right_pad=0.0, fduration=1.0)

# waveform file layout: 32 s, right_pad 2 s -> coalescence index 65536 - 4096
L = 32 * fs
coal = L - int(2.0 * fs)
w = torch.zeros(3, 2, L)
w[..., coal] = 1.0  # marker at the coalescence sample

ds = BaseAframeDataset.__new__(BaseAframeDataset)
ds._hparams = hp
ds.waveform_sampler = SimpleNamespace(right_pad=2.0)
sl = ds.slice_waveforms(w)
m_slice = int(sl[0, 0].argmax())
print("filter_size", ds.filter_size, "left_pad_size", ds.left_pad_size, "right_pad_size", ds.right_pad_size)
print("sliced length", sl.shape[-1], "marker index in slice", m_slice)

# kernel sampling exactly as inject(): kernel_size = X.size(-1) = (kernel_length + fduration) * fs
ksize = int((4.0 + 1.0) * fs)
torch.manual_seed(0)
big = sl[:1].repeat(200000, 1, 1)
kern, idx = sample_kernels(big, kernel_size=ksize, coincident=True, return_idx=True)
print("start index range", int(idx.min()), int(idx.max()), "distinct", idx.unique().numel())
pad = int(1.0 * fs / 2)  # ml4gw whiten crop per side
m_unwh = (kern[:, 0].argmax(-1))
m_wh = m_unwh - pad
print("whitened window length", ksize - 2 * pad)
print("merger index in whitened window: min", int(m_wh.min()), "max", int(m_wh.max()))
print("merger time after first sample (s): min %.6f max %.6f" % (m_wh.min() / fs, m_wh.max() / fs))
print("last sample time (s): %.6f" % ((ksize - 2 * pad - 1) / fs))

# the warp module's own geometric t_c formula
wd = WarpedDenoiserOnlyAframeDataset.__new__(WarpedDenoiserOnlyAframeDataset)
wd._hparams = hp
wd.merger_offset = 0.0
tc = (wd.right_slack_size() - idx.float() + wd.offset_size) / fs
print("warp module t_c (offset 0): min %.6f max %.6f" % (tc.min(), tc.max()))
assert torch.equal((wd.right_slack_size() - idx), m_wh)
wd.merger_offset = 25.5
tc2 = (wd.right_slack_size() - idx.float() + wd.offset_size) / fs
print("warp module t_c (offset 25.5): min %.6f max %.6f; window end before merger %.6f to %.6f s"
      % (tc2.min(), tc2.max(), tc2.min() - 4.0, tc2.max() - 4.0))

# light travel time from geocenter to each detector
tensors, vertices = gw.get_ifo_geometry("H1", "L1")
r = vertices.double().norm(dim=-1)
print("|r| H1, L1 (m):", [f"{x:.1f}" for x in r.tolist()])
print("max delay (ms):", [f"{1e3 * x / C:.3f}" for x in r.tolist()],
      "samples (trunc):", [int(x / C * fs) for x in r.tolist()])
# brute force over sky: largest integer shift shift_responses applies
th = torch.rand(400000, dtype=torch.float64) * torch.pi
ph = torch.rand(400000, dtype=torch.float64) * 2 * torch.pi
om = torch.stack([th.sin() * ph.cos(), th.sin() * ph.sin(), th.cos()], -1).view(-1, 1, 3)
dt = -(om * vertices.double()).sum(-1) * fs / C
print("shift samples over random sky: min", int(torch.trunc(dt).min()), "max", int(torch.trunc(dt).max()))
print(f"done {time.time() - t0:.1f} s")
