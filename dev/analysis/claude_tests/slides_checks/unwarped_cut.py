"""Unwarped Fourier tone counts of the real whitened templates truncated at the warp cut (apples-to-apples check)."""
import json, os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "2"
import h5py, numpy as np
FS = 2048
GM_SUN = 1.3271244e20; C_LIGHT = 299792458.0; TSUN = GM_SUN / C_LIGHT**3
def tau_of_f(f, mc):
    return 5 * (8 * np.pi * f) ** (-8 / 3) * (TSUN * mc) ** (-5 / 3)
def n_for_x(x, th=0.9):
    p = np.sort(np.abs(np.fft.fft(x)) ** 2)[::-1]
    c = np.sqrt(np.cumsum(p) / p.sum())
    return int(np.searchsorted(c, th)) + 1
WF = "/home/kyoon/SSM-BNS/DATA/aframe/train/end_o3_ratesandpops_bns_uniform_chirp.hdf5"
PSDF = "/home/kyoon/SSM-BNS/aframe/dev/runs/claude_tests/s4d_energy_diag/train_windows_snr8_powerlaw.h5"
TC = json.load(open("/home/kyoon/SSM-BNS/aframe/dev/runs/claude_tests/tone_check/results.json"))
WARP = json.load(open("/tmp/claude-1024/-home-kyoon-SSM-BNS/82effc75-bb98-44ad-af20-e41e39a7258d/scratchpad/qa/warp/results.json"))
WIN, POST, TAPER = 8192, 128, 512
L = WIN - POST
with h5py.File(PSDF, "r") as f:
    S = f["psds"][:2000].mean(axis=(0, 1))
f_psd = np.arange(S.size) * 0.5
for mc in ("1.0", "1.2", "1.5", "2.0"):
    r = TC["templates"][mc]["row"]; mct = TC["templates"][mc]["chirp_mass"]
    with h5py.File(WF, "r") as f:
        plus = f["waveforms/plus"][r]; cross = f["waveforms/cross"][r]
    h = plus - 1j * cross
    m = int(np.argmax(plus**2 + cross**2))
    seg = h[m - (L - 1): m + POST + 1]
    af = np.abs(np.fft.fftfreq(WIN, 1 / FS)); keep = af >= 20.0
    H = np.fft.fft(seg); Hw = np.zeros_like(H); Hw[keep] = H[keep] / np.sqrt(np.interp(af[keep], f_psd, S))
    hw = np.fft.ifft(Hw)
    taper = np.ones(WIN); taper[:TAPER] = 0.5 * (1.0 - np.cos(np.pi * np.arange(TAPER) / TAPER))
    hw = hw * taper
    K = hw[:L]
    # find fitted dtc from the warp results
    dtc = None
    for k, v in WARP.items():
        pass
    out = [f"Mc {mc}: full n90 {n_for_x(K)}"]
    for name, d in (("peak", 0.0), ("fit", 0.0188)):
        tc_idx = (L - 1) + d * FS
        for fcut in (300.0, 900.0):
            cut = int(np.floor(tc_idx - tau_of_f(fcut, mct) * FS))
            if cut > L - 1:
                continue
            Kc = K.copy(); Kc[cut + 1:] = 0
            out.append(f"{name}|{int(fcut)}Hz: cut at sample {cut}, kept energy {np.sum(np.abs(Kc)**2)/np.sum(np.abs(K)**2):.3f}, unwarped n90 of truncated {n_for_x(Kc)}")
    print(" ; ".join(out))
