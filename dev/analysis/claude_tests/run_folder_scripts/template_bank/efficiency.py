"""Parameters, FLOPs per window and GPU throughput of the compared models.

FLOPs: matmul/conv from torch's FlopCounterMode, plus FFTs counted by
wrapping torch.fft (complex length-n FFT 5 n log2 n, real 2.5 n log2 n).
"""

import math
import sys
import time

import torch
from torch.utils.flop_counter import FlopCounterMode

from architectures.regression import RegressionTimeDomainS4DenoiseRegressNorm
from train.experimental.chirp_kernel.arch import ChirpKernelNet
from train.experimental.template_bank.arch import TemplateBankNet

TB = "/home/kyoon/SSM-BNS/aframe/dev/runs/claude_tests/template_bank"
fft_flops = [0.0]
originals = {}


def wrap(name, cost):
    fn = getattr(torch.fft, name)
    originals[name] = fn

    def counted(x, *args, n=None, dim=-1, **kwargs):
        out = fn(x, *args, n=n, dim=dim, **kwargs)
        length = n if n is not None else (2 * (x.shape[dim] - 1) if name == "irfft" else x.shape[dim])
        batch = x.numel() / x.shape[dim]
        fft_flops[0] += cost * batch * length * math.log2(length)
        return out

    setattr(torch.fft, name, counted)


for name, cost in (("rfft", 2.5), ("irfft", 2.5), ("fft", 5.0), ("ifft", 5.0)):
    wrap(name, cost)

models = {
    "den-reg (good run)": RegressionTimeDomainS4DenoiseRegressNorm(
        num_ifos=2, d_output=4, denoiser_d_model=128, denoiser_d_state=64, denoiser_n_layers=6,
        denoiser_dropout=0.2, regressor_d_model=128, regressor_d_state=64, regressor_n_layers=6,
        regressor_dropout=0.2, prenorm=True, num_groups=32, dt_min=1e-4, dt_max=10.0,
        detach_denoiser=False, smooth_below=20.0, smooth_coeffs=4, sample_rate=2048.0, kernel_length=4.0,
    ),
    "chirp kernels 512": ChirpKernelNet(),
    "template bank 703": TemplateBankNet(f"{TB}/bank_mc0.5pct_spin3_q0.8.npz"),
    "template bank 235": TemplateBankNet(f"{TB}/bank_mc0.5pct_spin0_q0.8.npz"),
}
dev = torch.device("cuda")
B = int(sys.argv[1]) if len(sys.argv) > 1 else 32
x = torch.randn(B, 2, 8192, device=dev)
print(f"{'model':22s} {'trainable params':>16s} {'buffer numbers':>14s} {'GFLOP/window':>13s} {'ms/batch':>9s} {'windows/s':>10s}")
for name, m in models.items():
    m = m.to(dev).eval()
    trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
    buffers = sum(b.numel() * (2 if b.is_complex() else 1) for b in m.buffers())
    fft_flops[0] = 0.0
    with torch.no_grad(), FlopCounterMode(display=False) as fc:
        m(x[:1])
    gflop = (fc.get_total_flops() + fft_flops[0]) / 1e9
    with torch.no_grad():
        for _ in range(3):
            m(x)
        torch.cuda.synchronize()
        times = []
        for _ in range(10):
            t0 = time.perf_counter()
            m(x)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
    ms = 1000 * sorted(times)[len(times) // 2]
    print(f"{name:22s} {trainable:16,d} {buffers:14,d} {gflop:13.2f} {ms:9.1f} {B / ms * 1000:10.0f}")
    m.cpu()
    torch.cuda.empty_cache()
