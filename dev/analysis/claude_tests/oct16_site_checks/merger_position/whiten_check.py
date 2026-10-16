"""Run ml4gw's Whiten (as built by the data module) on marker kernels and locate the marker."""
from types import SimpleNamespace

import torch
from lightning.pytorch.utilities.parsing import AttributeDict
from ml4gw.transforms import Whiten
from ml4gw.utils.slicing import sample_kernels
from train.data.base import BaseAframeDataset

torch.set_num_threads(1)
fs = 2048
hp = AttributeDict(sample_rate=2048.0, kernel_length=4.0, left_pad=3.5, right_pad=0.0, fduration=1.0)
ds = BaseAframeDataset.__new__(BaseAframeDataset)
ds._hparams = hp
ds.waveform_sampler = SimpleNamespace(right_pad=2.0)

L = 32 * fs
w = torch.zeros(1, 2, L)
w[..., L - 2 * fs] = 1.0
sl = ds.slice_waveforms(w).repeat(4000, 1, 1)
torch.manual_seed(1)
kern, idx = sample_kernels(sl, kernel_size=5 * fs, coincident=True, return_idx=True)
whiten = Whiten(1.0, 2048.0, 20.0, None)  # fduration, sample_rate, highpass, lowpass as in build_transforms
psd = torch.ones(2, 5 * fs // 2 + 1, dtype=torch.float64)
out = whiten(kern.double(), psd)
print("whitened shape", tuple(out.shape))
pos = out.abs().argmax(-1)[:, 0]
expect = 8192 - idx
inside = expect <= 8191
print("starts", int(idx.min()), int(idx.max()))
print("marker inside window: argmax == 8192 - s for all:", bool((pos[inside] == expect[inside]).all()),
      "n inside", int(inside.sum()), "n at s=0 (marker one past end)", int((~inside).sum()))
print("argmax range (inside):", int(pos[inside].min()), int(pos[inside].max()))
