"""Expected S4D-Lin init tone statistics (ml4gw S4DKernel) over dt ~ logU[dt_min, dt_max].
Mode n pole z = exp(dt*(-0.5 + i*pi*n)); freq f = fs/2 * |wrap(n*dt)| (wrap to [-1,1]);
e-fold = 2/(dt*fs) s. Loops over modes to keep memory small."""
import numpy as np

def stats(fs, n_modes, f_lo, f_hi, dt_min=1e-4, dt_max=10.0, ngrid=200001):
    dt = np.exp(np.linspace(np.log(dt_min), np.log(dt_max), ngrid))
    w = np.full(ngrid, 1.0); w[0] = w[-1] = 0.5; w /= w.sum()
    efold = (2.0 / (dt * fs)) > 1.0
    acc = dict(inband=0.0, inband_unaliased=0.0, both=0.0, aliased=0.0)
    for n in range(n_modes):
        x = n * dt
        f = fs / 2.0 * np.abs(((x + 1.0) % 2.0) - 1.0)
        inb = (f >= f_lo) & (f <= f_hi)
        acc["inband"] += (inb * w).sum()
        acc["inband_unaliased"] += ((inb & (x <= 1.0)) * w).sum()
        acc["both"] += ((inb & efold) * w).sum()
        acc["aliased"] += ((x > 1.0) * w).sum()
    out = {k: v / n_modes for k, v in acc.items()}
    out["efold_gt_1s"] = float((efold * w).sum())
    out["dt_cut_efold"] = 2.0 / fs
    return out

for name, fs, nm, lo, hi in [
    ("den-reg (fs 2048, d_state 64 -> 32 modes), band 20-1024", 2048, 32, 20, 1024),
    ("oracle 64x32x4 (fs 2048, d_state 32 -> 16 modes), band 20-1024", 2048, 16, 20, 1024),
    ("long_conv d64 (fs 1024, 32 modes), band 20-512", 1024, 32, 20, 512),
    ("long_conv d512 (fs 1024, 256 modes), band 20-512", 1024, 256, 20, 512),
]:
    s = stats(fs, nm, lo, hi)
    print(name, {k: round(float(v), 4) for k, v in s.items()}, flush=True)
