"""Independent check of where the merger falls, with the pipeline's own functions on index ramps."""
import math
from types import SimpleNamespace
import torch
torch.set_num_threads(1)
torch.manual_seed(123)
from ml4gw import gw
from ml4gw.distributions import Cosine
from ml4gw.transforms import Whiten
from ml4gw.utils.slicing import sample_kernels, slice_kernels
from train.data.base import BaseAframeDataset
from train.experimental.oracle.data import WarpedDenoiserOnlyAframeDataset

FS = 2048.0
N_STORED = 65536
MERGER = N_STORED - int(2.0 * FS)

def stub(offset=None):
    s = SimpleNamespace()
    s.hparams = SimpleNamespace(sample_rate=FS, kernel_length=4.0, fduration=1.0, left_pad=3.5, right_pad=0.0)
    s.waveform_sampler = SimpleNamespace(right_pad=2.0)
    s.filter_size = BaseAframeDataset.filter_size.fget(s)
    s.left_pad_size = BaseAframeDataset.left_pad_size.fget(s)
    s.right_pad_size = BaseAframeDataset.right_pad_size.fget(s)
    if offset is not None:
        s.merger_offset = offset
        s.offset_size = WarpedDenoiserOnlyAframeDataset.offset_size.fget(s)
    return s

ramp = torch.arange(N_STORED, dtype=torch.float64)[None, None, :].repeat(1, 2, 1)
for name, fn, st in [("base", BaseAframeDataset.slice_waveforms, stub()),
                     ("warp 0", WarpedDenoiserOnlyAframeDataset.slice_waveforms, stub(0.0)),
                     ("warp 25.5", WarpedDenoiserOnlyAframeDataset.slice_waveforms, stub(25.5))]:
    cut = fn(st, ramp)
    first, last = int(cut[0, 0, 0]), int(cut[0, 0, -1])
    print(f"{name}: cut length {cut.shape[-1]}, stored samples {first}..{last}, merger at cut index {MERGER - first}")
    # every start index k: window = cut[k+1024 : k+9216]
    ks = torch.arange(0, cut.shape[-1] - 10240)
    pos = MERGER - (first + ks + 1024)          # merger index in the window (first sample = 0)
    print(f"   k range 0..{int(ks[-1])} ({ks.numel()} values); merger at window sample {int(pos.min())}..{int(pos.max())} "
          f"= {pos.min()/FS:.6f}..{pos.max()/FS:.6f} s after first sample; exact mean {pos.double().mean()/FS:.6f} s")
    print(f"   window end (first + 4 s) minus merger: {(8192-pos.max())/FS:+.6f}..{(8192-pos.min())/FS:+.6f} s; "
          f"last sample minus merger: {(8191-pos.max())/FS:+.6f}..{(8191-pos.min())/FS:+.6f} s")
    if name == "warp 25.5":
        print(f"   window first sample before merger: {pos.min()/FS:.6f}..{pos.max()/FS:.6f} s; window end (first + 4 s) before merger: {(pos.min()-8192)/FS:.6f}..{(pos.max()-8192)/FS:.6f} s")
        print(f"   t_c the code uses: right_slack {WarpedDenoiserOnlyAframeDataset.right_slack_size(st)} -> {(WarpedDenoiserOnlyAframeDataset.right_slack_size(st) - 1023 + st.offset_size)/FS:.6f}..{(WarpedDenoiserOnlyAframeDataset.right_slack_size(st) + st.offset_size)/FS:.6f} s")

# sample_kernels draws k from 0..1023 uniformly
cut = BaseAframeDataset.slice_waveforms(stub(), ramp)
_, idx = sample_kernels(cut.repeat(20000, 1, 1).float(), kernel_size=10240, coincident=True, return_idx=True)
print(f"sample_kernels: k in {int(idx.min())}..{int(idx.max())}, {idx.unique().numel()} distinct of {idx.numel()}")

# whitening: crop and zero-phase (impulse stays put, minus the 1024 dropped samples) with a non-flat PSD too
wh = Whiten(1.0, FS, highpass=20.0)
f = torch.fft.rfftfreq(10240, 1/FS)
for label, psd in [("flat", torch.ones(1, 1, f.numel(), dtype=torch.float64)),
                   ("steep", (1.0 + (40.0 / f.clamp(min=1.0)) ** 8 + (f / 300.0) ** 2)[None, None].double())]:
    for p in (3000, 9000):
        x = torch.zeros(1, 1, 10240, dtype=torch.float64); x[..., p] = 1.0
        y = wh(x, psd)
        print(f"whiten ({label} PSD): in 10240 -> out {y.shape[-1]}; impulse at {p} -> |peak| at {int(y.abs().argmax())} (expected {p-1024})")

# detector shift: the pipeline's shift_responses on a ramp, and its formula for many directions
tensors, vertices = gw.get_ifo_geometry("H1", "L1")
C = 299792458.0
r = vertices.double().norm(dim=-1)
print(f"|vertex|/c: H1 {r[0]/C*1e3:.4f} ms, L1 {r[1]/C*1e3:.4f} ms -> {r[0]/C*FS:.3f}, {r[1]/C*FS:.3f} samples")
B = 4096
dec = Cosine().sample((B,)); phi = torch.rand(B) * 2 * math.pi - math.pi
theta = math.pi / 2 - dec
sh = gw.shift_responses(cut.repeat(B, 1, 1).float(), theta, phi, vertices, FS)
dt_ramp = (cut[0, 0, 100] - sh[:, :, 100]).round()   # positive = later
omega = torch.column_stack([torch.sin(theta)*torch.cos(phi), torch.sin(theta)*torch.sin(phi), torch.cos(theta)]).view(-1, 1, 3)
dt_formula = torch.trunc(-(omega * vertices).sum(-1) * FS / C)
print(f"shift_responses ramp shift equals trunc formula: {bool(torch.all(dt_ramp == dt_formula))}; range {int(dt_formula.min())}..{int(dt_formula.max())}")

# fraction of detector windows with the merger after the last sample (window sample > 8191), and after the window end (> 8192)
M = 4_000_000
dec = Cosine().sample((M,)); phi = torch.rand(M) * 2 * math.pi - math.pi
theta = math.pi / 2 - dec
omega = torch.column_stack([torch.sin(theta)*torch.cos(phi), torch.sin(theta)*torch.sin(phi), torch.cos(theta)]).view(-1, 1, 3)
dt = torch.trunc(-(omega.double() * vertices.double()).sum(-1) * FS / C)
k = torch.randint(0, 1024, (M, 1)).double()
m = 8192 - k + dt
print(f"MC {M} windows x 2 detectors: merger after last sample {float((m > 8191).double().mean())*100:.3f}%, after window end (first+4 s) {float((m > 8192).double().mean())*100:.3f}%; "
      f"position {int(m.min())}..{int(m.max())} = {m.min()/FS:.6f}..{m.max()/FS:.6f} s")
# exact expectation for isotropic directions: x = (r/c) fs * u, u uniform(-1,1); s = trunc(x)
def p_s_ge(kk, R):
    # P(trunc(x) >= kk) for x uniform(-R, R)
    if kk <= 0:
        lo = kk - 1   # trunc(x) >= kk (kk <= 0)  <=>  x > kk - 1
        return min(1.0, max(0.0, (R - lo) / (2 * R)))
    return min(1.0, max(0.0, (R - kk) / (2 * R)))
for name, R in [("H1", float(r[0]/C*FS)), ("L1", float(r[1]/C*FS))]:
    after_last = sum(p_s_ge(kk, R) for kk in range(0, 1024)) / 1024          # s >= k   <=> 8192-k+s > 8191
    after_end = sum(p_s_ge(kk + 1, R) for kk in range(0, 1024)) / 1024       # s >= k+1 <=> 8192-k+s > 8192
    print(f"exact {name}: after last sample {after_last*100:.3f}%, after window end {after_end*100:.3f}%")
