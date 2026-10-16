"""Template bank of 256 whitened chirps spanning the chirp-mass prior, as unit-norm complex 4 s
kernels at 1024 Hz in the LongConvEnergyNet convention (y[t] = sum_l k[l] x[t - l]).

Rows: one per chirp mass on 256 log-spaced points from 0.85 to 2.75 M_sun, chosen from the
training waveform file by the rule of train/experimental/template_bank/bank.py (spin target 0):
|log(mc / point)| < step / 4, |q - 0.8| < 0.1, |chi_eff| < 0.03, smallest |chi_eff| / 0.03 + in-plane
spin. h_plus is whitened with ml4gw.spectral.whiten (H1 average PSD of the existing bank file,
fduration 1 s, highpass 20 Hz) on an 8 s stretch, decimated to 1024 Hz as LongConvEnergyNet does,
turned into its analytic signal, cut to the 4 s ending ``r`` samples after the merger (peak of
h_plus^2 + h_cross^2), reversed and conjugated, unit norm.

``r`` is chosen from candidates by the mean fraction of SNR^2 a template keeps on its own noiseless
signal with the merger uniform over the data's range (whitened samples 7169-8192 at 2048 Hz).
"""
import json
import math
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
from ml4gw.spectral import whiten

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "harness"))
sys.path.insert(0, str(HERE.parent / "kernels"))
import harness  # noqa: E402

torch.set_num_threads(8)
OLD_BANK = "/home/kyoon/SSM-BNS/aframe/dev/runs/claude_tests/template_bank/bank_mc0.5pct_spin0_q0.8.npz"
FS = 2048
L = 4096                   # kernel length at 1024 Hz
N_T = 256
PRE, POST = 13312, 3072    # raw samples before / after the merger (whitened: 6 s before, 1 s after)
M_W = PRE - FS // 2        # merger index in the whitened 2048 Hz stretch (12288)
M_D = M_W // 2             # merger index in the decimated stretch (6144)
R_CANDIDATES = [0, 2, 4, 8, 16, 32, 64]


def select_rows():
    with h5py.File(harness.WAVEFORM_FILE, "r") as f:
        p = {k: f["parameters"][k][:] for k in ("mass_1", "mass_2", "a_1", "a_2", "tilt_1", "tilt_2")}
    m1, m2 = p["mass_1"], p["mass_2"]
    mc = (m1 * m2) ** 0.6 / (m1 + m2) ** 0.2
    q = np.minimum(m1, m2) / np.maximum(m1, m2)
    chi = (m1 * p["a_1"] * np.cos(p["tilt_1"]) + m2 * p["a_2"] * np.cos(p["tilt_2"])) / (m1 + m2)
    inplane = p["a_1"] * np.abs(np.sin(p["tilt_1"])) + p["a_2"] * np.abs(np.sin(p["tilt_2"]))
    points = np.exp(np.linspace(np.log(0.85), np.log(2.75), N_T))
    step = (np.log(2.75) - np.log(0.85)) / (N_T - 1)
    rows, widen = [], []
    for c in points:
        tol, k = step / 4, 0
        while True:
            cand = np.where((np.abs(np.log(mc / c)) < tol) & (np.abs(q - 0.8) < 0.1) & (np.abs(chi) < 0.03))[0]
            if len(cand):
                break
            tol, k = tol * 2, k + 1
        rows.append(int(cand[np.argmin(np.abs(chi[cand]) / 0.03 + inplane[cand])]))
        widen.append(k)
    rows = np.array(rows)
    return dict(rows=rows, points=points, chirp_mass=mc[rows], mass_ratio=q[rows], chi_eff=chi[rows],
                inplane=inplane[rows], widenings=np.array(widen), log_step=step)


def decimate(x, n_out):
    """LongConvEnergyNet's resampling: rfft, keep bins up to the new Nyquist, irfft, / sqrt 2."""
    spec = torch.fft.rfft(x, n=2 * n_out)[..., : n_out // 2 + 1]
    return torch.fft.irfft(spec, n=n_out) / math.sqrt(2)


def analytic(x):
    n = x.shape[-1]
    w = torch.zeros(n, dtype=torch.float64)
    w[0] = 1.0
    w[1:n // 2] = 2.0
    w[n // 2] = 1.0
    return torch.fft.ifft(torch.fft.fft(x) * w)


def build():
    sel = select_rows()
    rows = sel["rows"]
    order = np.argsort(rows)
    with h5py.File(harness.WAVEFORM_FILE, "r") as f:
        hp = np.empty((len(rows), f["waveforms/plus"].shape[-1]))
        hc = np.empty_like(hp)
        hp[order] = f["waveforms/plus"][np.sort(rows)]
        hc[order] = f["waveforms/cross"][np.sort(rows)]
    peak = np.argmax(hp ** 2 + hc ** 2, axis=1)
    if (peak - PRE).min() < 0 or (peak + POST).max() > hp.shape[1]:
        raise RuntimeError("stretch out of range")
    raw = np.stack([hp[j, m - PRE:m + POST] for j, m in enumerate(peak)])          # (J, 16384)
    psd = torch.tensor(np.load(OLD_BANK)["psd"][:1], dtype=torch.float64)          # (1, 2049) H1
    white = whiten(torch.tensor(raw)[:, None], psd, 1.0, FS, highpass=20.0)[:, 0].double()   # (J, 14336)
    dec = decimate(white, white.shape[-1] // 2)                                    # (J, 7168) at 1024 Hz
    ha = analytic(dec)
    return sel, peak, white, dec, ha


def kernels_for(ha, r):
    frame = ha[:, M_D + r - (L - 1): M_D + r + 1]                                   # ends r after merger
    k = frame.flip(-1).conj()
    return k / k.abs().pow(2).sum(-1, keepdim=True).sqrt()


def conv_energy(k, x):
    """LongConvEnergyNet's convolution and energy for one detector: (J, L) kernels, (J, L) inputs."""
    n = 2 * L
    y = torch.fft.ifft(torch.fft.fft(x, n=n) * torch.fft.fft(k, n=n), n=n)[..., :L]
    return 2 * y.abs().pow(2)


def self_capture(white, k, t_ms):
    """Mean over merger positions t_ms (2048 Hz window samples) of max_t E / ||x||^2 for each
    template's own noiseless window: (J,)."""
    fr = []
    for tm in t_ms:
        s0 = M_W - tm
        x = white[:, s0:s0 + 2 * L]                                                # (J, 8192), merger at tm
        xd = decimate(x, L)
        e = conv_energy(k, xd.to(k.dtype)).amax(-1) / xd.pow(2).sum(-1)
        fr.append(e)
    return torch.stack(fr).mean(0)


def main():
    sel, peak, white, dec, ha = build()
    t_ms = np.linspace(7169, 8191, 64).round().astype(int)
    capture = {}
    for r in R_CANDIDATES:
        capture[r] = self_capture(white, kernels_for(ha, r), t_ms)
    best_r = max(R_CANDIDATES, key=lambda r: float(capture[r].mean()))
    k = kernels_for(ha, best_r)
    # same number through LongConvEnergyNet.peak_energy (module code), both detectors fed the same window
    from train.experimental.long_conv.arch import LongConvEnergyNet

    arch = LongConvEnergyNet(num_ifos=2, channels=N_T, kernel="free").double()
    kk = k.clone()
    arch.kernels = lambda: kk
    j = np.arange(0, N_T, 32)
    tm = 7680
    x = white[j, M_W - tm:M_W - tm + 2 * L].float().double()
    pe = arch.peak_energy(torch.stack([x, x], 1))                                  # (len(j), 256)
    own = pe[torch.arange(len(j)), torch.as_tensor(j)] / 2 / decimate(x, L).pow(2).sum(-1)
    xd = decimate(x, L)
    mine = conv_energy(k[j], xd.to(k.dtype)).amax(-1) / xd.pow(2).sum(-1)
    checks = {
        "rows_unique": bool(len(np.unique(sel["rows"])) == N_T),
        "tolerance_widenings": int(sel["widenings"].sum()),
        "chirp_mass_rel_offset_from_point_max": float(np.max(np.abs(sel["chirp_mass"] / sel["points"] - 1))),
        "mass_ratio_range": [float(sel["mass_ratio"].min()), float(sel["mass_ratio"].max())],
        "chi_eff_abs_max": float(np.abs(sel["chi_eff"]).max()),
        "log_step": float(sel["log_step"]),
        "self_capture_mean_by_r": {str(r): float(capture[r].mean()) for r in R_CANDIDATES},
        "self_capture_min_by_r": {str(r): float(capture[r].min()) for r in R_CANDIDATES},
        "chosen_r_samples_1024Hz": int(best_r),
        "module_vs_direct_energy_ratio_maxreldiff": float(((own - mine).abs() / mine).max()),
        "module_self_energy_fraction_at_3.75s": [round(float(v), 5) for v in own],
        "kernel_energy_below_20Hz_or_negative_freq_max": float(
            (lambda K: (K.abs().pow(2) * ((torch.fft.fftfreq(L, 1 / 1024) < 20).double())).sum(-1).max()
             / K.abs().pow(2).sum(-1).min())(torch.fft.fft(k.conj().flip(-1)))),
        "post_merger_share_of_decimated_whitened_energy_median": float(
            np.median((dec[:, M_D + 1:].pow(2).sum(-1) / dec.pow(2).sum(-1)).numpy())),
    }
    np.savez(HERE / "data" / "bank256.npz", kernels=k.numpy().astype(np.complex64), rows=sel["rows"],
             points=sel["points"], chirp_mass=sel["chirp_mass"], mass_ratio=sel["mass_ratio"],
             chi_eff=sel["chi_eff"], inplane=sel["inplane"], r=best_r, peak_in_file=peak)
    json.dump(checks, open(HERE / "logs" / "bank_checks.json", "w"), indent=1)
    print(json.dumps(checks, indent=1))


if __name__ == "__main__":
    main()
