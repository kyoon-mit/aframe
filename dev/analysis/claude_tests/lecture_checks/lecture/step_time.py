"""GPU time per training batch: data preparation (PSD, projection, whitening) against the model step.

Random tensors with the shapes of the noisy long-conv runs (batch 32, H1+L1, 2048 Hz,
psd_length 20 s, kernel 4 s, fduration 1 s, fftlength 2 s, highpass 20 Hz,
signal_repeats 4). No data module is built.
"""
import sys
import torch
sys.path.insert(0, "/home/kyoon/SSM-BNS/aframe/projects/train")
sys.path.insert(0, "/home/kyoon/SSM-BNS/aframe/libs/utils")
from ml4gw.transforms import Whiten
from utils.preprocessing import PsdEstimator
from train import augmentations as aug
from train.experimental.long_conv.arch import LongConvEnergyNet

dev = "cuda"
fs, B = 2048.0, 32
psd_est = PsdEstimator(5.0, fs, 2.0, fast=True, average="median").to(dev)
whiten = Whiten(1.0, fs, 20.0, None).to(dev)
proj = aug.WaveformProjector(["H1", "L1"], fs, 20.0, None).to(dev)


def timeit(fn, n=40, warm=8):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(n):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / n


X = torch.randn(B, 2, int(25 * fs), device=dev)
wf = torch.randn(8, 2, 11264, device=dev) * 1e-21


def prep():
    x, psds = psd_est(X)
    dec = torch.rand(8, device=dev) * 3 - 1.5
    psi = torch.rand(8, device=dev) * 3.14
    phi = torch.rand(8, device=dev) * 6.28
    r = proj(dec, psi, phi, cross=wf[:, 0], plus=wf[:, 1])
    r = r.repeat_interleave(4, dim=0)[:B]
    r = proj.rescaler(r, psds, torch.full((B,), 10.0, device=dev))
    xi = x + r[..., : x.shape[-1]]
    w = whiten(xi, psds)
    wc = whiten(r[..., : x.shape[-1]], psds)  # clean target, denoiser dataset
    return w, wc


with torch.no_grad():
    t_prep = timeit(prep)
print(f"data preparation per batch: {t_prep:.1f} ms")

for kernel, d_state in (("free", 64), ("s4d", 64), ("s4d", 512)):
    net = LongConvEnergyNet(num_ifos=2, channels=256, kernel=kernel, d_state=d_state, dt_min=1e-4, dt_max=10.0,
                            sample_rate=fs, kernel_length=4.0, decimate=2, coincidence=0.01, highpass=20.0,
                            mc_min=0.85, mc_max=2.75, grid_step=0.005).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3)
    xin = torch.randn(B, 2, 8192, device=dev)

    def step():
        opt.zero_grad(set_to_none=True)
        logits = net(xin)
        loss = torch.nn.functional.cross_entropy(logits, torch.randint(0, logits.shape[-1], (B,), device=dev))
        loss.backward()
        opt.step()

    t = timeit(step)
    print(f"model step ({kernel}, d_state {d_state}): {t:.1f} ms; preparation share {100 * t_prep / (t_prep + t):.0f}%")
print("GPU shared with running training jobs: absolute times are upper bounds; the ratio is the point.")
