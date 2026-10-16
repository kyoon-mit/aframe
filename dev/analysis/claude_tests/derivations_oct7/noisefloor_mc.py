"""Matched-filter benchmark code (mf.py) run on Gaussian white noise and on real
background windows, to compare the noise floor. CPU only."""
import sys
import h5py, numpy as np, torch
from ml4gw.spectral import truncate_inverse_power_spectrum
torch.set_num_threads(16)
WAVE = "/home/kyoon/SSM-BNS/DATA/aframe/train/end_o3_ratesandpops_bns_uniform_chirp.hdf5"
R='/home/kyoon/SSM-BNS/aframe/dev/runs/claude_tests/den_reg_sched_alpha_0.5_floor_1e-2_c_rho_5_lambda_amp_3_powerlaw_m2_snr_d128_n6_mc_q_rep4_smooth20_wp0.96_ddp/mf_bench_ep630/snr4_powerlaw'
FS, L = 2048, 8192; N = 2 * L; MERGE_T = 7680; MAX_LAG = 600; COINC = 20
FDURATION, HIGHPASS = 1.0, 20.0; STEP, QS = 0.005, (0.5, 0.7, 0.9)
dev = torch.device("cpu")
NWIN = int(sys.argv[1])

def build_bank():
    f = h5py.File(WAVE); p = f["parameters"]
    m1, m2 = p["mass_1"][:], p["mass_2"][:]
    mc = (m1 * m2) ** 0.6 / (m1 + m2) ** 0.2
    q = np.minimum(m1, m2) / np.maximum(m1, m2)
    spin = p["a_1"][:] + p["a_2"][:]
    grid = np.exp(np.arange(np.log(mc.min()), np.log(mc.max()), STEP))
    chosen = []
    for g in grid:
        near = np.where(np.abs(np.log(mc / g)) < STEP / 4)[0]
        if not len(near): continue
        for qq in QS:
            cand = near[np.abs(q[near] - qq) < 0.05]
            if not len(cand): continue
            i = cand[np.argmin(spin[cand])]
            if i not in chosen: chosen.append(i)
    chosen = np.sort(np.array(chosen))
    hp = f["waveforms/plus"][chosen]; hc = f["waveforms/cross"][chosen]
    bank = np.zeros((len(chosen), L), np.float32)
    taper = np.ones(L, np.float32); ramp = int(0.25 * FS)
    taper[:ramp] = 0.5 * (1 - np.cos(np.pi * np.arange(ramp) / ramp))
    for j in range(len(chosen)):
        peak = int(np.argmax(hp[j] ** 2 + hc[j] ** 2))
        padded = np.pad(hp[j], (L, L)); start = peak + L - MERGE_T
        bank[j] = padded[start:start + L] * taper
    # spins of chosen templates
    a1, a2 = p["a_1"][:][chosen], p["a_2"][:][chosen]
    t1, t2 = p["tilt_1"][:][chosen], p["tilt_2"][:][chosen]
    ce = (m1[chosen]*a1*np.cos(t1) + m2[chosen]*a2*np.cos(t2))/(m1[chosen]+m2[chosen])
    return torch.tensor(bank), mc[chosen], q[chosen], ce, a1, a2

bank, bank_mc, bank_q, bank_ce, a1, a2 = build_bank()
print(f"bank {len(bank)} templates; mc {bank_mc.min():.4f}-{bank_mc.max():.4f}; |chi_eff| max {np.abs(bank_ce).max():.4f} median {np.median(np.abs(bank_ce)):.4f}; a1 max {a1.max():.4f} a2 max {a2.max():.4f}")
lm = np.log(np.sort(bank_mc)); u = np.unique(np.round(lm, 6)); d = np.diff(u)
print(f"distinct template chirp masses {len(u)}; log gaps: median {np.median(d):.5f} max {d.max():.5f}")
H = torch.fft.rfft(bank, n=N); F = H.shape[-1]
wp = torch.full((F,), 2.0); wp[0] = wp[-1] = 1.0
lag_idx = torch.cat([torch.arange(N - MAX_LAG, N), torch.arange(0, MAX_LAG + 1)])

def mf(X, psd):
    psd = torch.nn.functional.interpolate(psd, size=(F,), mode="linear")
    W = torch.nan_to_num(truncate_inverse_power_spectrum(psd, FDURATION, FS, HIGHPASS) ** -0.5).float()
    D = torch.fft.rfft(X, n=N)
    HW = H[None, None] * W[:, :, None]
    sigma = ((HW.abs() ** 2) * wp).sum(-1).div(N).sqrt()
    prod = D[:, :, None] * HW.conj()
    spec = torch.zeros(*prod.shape[:-1], N, dtype=prod.dtype)
    spec[..., :F] = prod; spec[..., 1:F - 1] *= 2
    z = torch.fft.ifft(spec, n=N)[..., lag_idx]
    rho = z.abs() / sigma[..., None].clamp_min(1e-30)
    rhoL = torch.nn.functional.max_pool1d(rho[:, 1].reshape(-1, 1, rho.shape[-1]) ** 2, 2 * COINC + 1, stride=1, padding=COINC).reshape(rho[:, 1].shape)
    net2 = rho[:, 0] ** 2 + rhoL
    return net2.amax(-1).amax(-1).sqrt(), (rho[:, 0]**2).amax(-1).amax(-1).sqrt()

w = h5py.File(R + '/windows.h5'); m = h5py.File(R + '/mf.h5')
mcw = w['params/chirp_mass'][:]; bk = np.sort(np.where(np.isnan(mcw))[0][:NWIN])
g = torch.Generator().manual_seed(0)
gauss, real, ref, gH = [], [], m['mf_snr'][:][bk], []
for i0 in range(0, len(bk), 2):
    ii = bk[i0:i0 + 2]
    psd = torch.tensor(w['psds'][ii]).double()
    Xr = torch.tensor(w['X'][ii])
    Xg = torch.randn(Xr.shape, generator=g)
    a, _ = mf(Xr, psd); b, bh = mf(Xg, psd)
    real += a.tolist(); gauss += b.tolist(); gH += bh.tolist()
real, gauss, gH = map(np.array, (real, gauss, gH))
print(f"real background ({len(real)} windows): median {np.median(real):.3f}; reproduces mf.h5: max abs diff {np.abs(real-ref).max():.2e}")
print(f"Gaussian white noise, unit variance: median {np.median(gauss):.3f} 16/84 {np.percentile(gauss,[16,84])} 99% {np.percentile(gauss,99):.3f} max {gauss.max():.3f}")
print(f"Gaussian, H1 alone (max over templates and lags): median {np.median(gH):.3f}")
np.save('/tmp/claude-1024/-home-kyoon-SSM-BNS/82effc75-bb98-44ad-af20-e41e39a7258d/scratchpad/oct7work/noisefloor_gauss.npy', np.stack([gauss, real]))
