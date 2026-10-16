"""Independent check of the cuFFT C2R claim: irfft(spectrum[..., :2049], n=4096) on GPU
adds -Im(X[2048])/4096 to every sample when >= 64 transforms are batched."""
import json, numpy as np, torch
torch.set_num_threads(4)
torch.cuda.set_per_process_memory_fraction(0.02, 0)
dev = torch.device("cuda:0")
rng = np.random.default_rng(123)
res = {}
for B in [1, 8, 15, 16, 31, 32, 33, 48, 64]:
    X = rng.standard_normal((B, 2, 8192)).astype(np.float32)
    xt = torch.from_numpy(X).to(dev)
    spec = torch.fft.rfft(xt, n=8192)[..., :2049]
    y = torch.fft.irfft(spec, n=4096).cpu().numpy().astype(np.float64)
    s = np.fft.rfft(X.astype(np.float64), n=8192)[..., :2049]
    ref = np.fft.irfft(s, n=4096)          # numpy discards Im of DC and Nyquist bins
    err = y - ref
    pred = -s[..., 2048].imag / 4096       # claimed offset
    dc_part = err.mean(-1)
    resid = err - dc_part[..., None]
    res[B] = dict(n_transforms=2 * B, max_abs_err=float(np.abs(err).max()),
                  rms_err=float(np.sqrt((err ** 2).mean())),
                  dc_vs_pred_max=float(np.abs(dc_part - pred).max()),
                  pred_rms=float(np.sqrt((pred ** 2).mean())),
                  resid_rms_after_dc=float(np.sqrt((resid ** 2).mean())))
    print(B, res[B])
# also: complex FFT path (fft/ifft with n=2L) used after decimation is exact?
json.dump(res, open("v_cufft.json", "w"), indent=1)
