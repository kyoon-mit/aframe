"""Whitened kernels for a row's parameters at a scaled chirp mass.

Whitening, window and kernel construction copied from s4dmath/fits/common.py
(same constants as tone_check.py). Waveforms: pycbc get_td_waveform with
IMRPhenomPv2, f_lower 20 Hz, f_ref 50 Hz, 32 s, right_pad 2 s, aligned as in
ledger.injections._WaveformGenerator (this reproduces the stored rows).
"""

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

from pathlib import Path  # noqa: E402

import h5py  # noqa: E402
import numpy as np  # noqa: E402

WF_PATH = Path("/home/kyoon/SSM-BNS/DATA/aframe/train/end_o3_ratesandpops_bns_uniform_chirp.hdf5")
PSD_PATH = Path("/home/kyoon/SSM-BNS/aframe/dev/runs/claude_tests/s4d_energy_diag/train_windows_snr8_powerlaw.h5")
FS = 2048.0
DT = 1.0 / FS
WIN = 8192
POST = 128
L = WIN - POST
TAPER = 512
F_LOW = 20.0
N_PSD_ROWS = 2000
PSD_DF = 0.5
DUR, RIGHT_PAD, F_MIN, F_REF = 32.0, 2.0, 20.0, 50.0
MSUN = 1.988409870698051e30


def load_psd():
    with h5py.File(PSD_PATH, "r") as f:
        psds = f["psds"][:N_PSD_ROWS]
    S = psds.mean(axis=(0, 1))
    return np.arange(S.size) * PSD_DF, S


def row_params(row):
    with h5py.File(WF_PATH, "r") as f:
        par = {k: float(f["parameters"][k][row]) for k in f["parameters"]}
        plus = f["waveforms/plus"][row]
        cross = f["waveforms/cross"][row]
    return par, plus, cross


def whitened_window(plus, cross, f_psd, S):
    h = plus - 1j * cross
    m = int(np.argmax(plus**2 + cross**2))
    start, stop = m - (L - 1), m + POST + 1
    if start < 0 or stop > h.size:
        raise RuntimeError(f"window [{start}, {stop}) out of range")
    seg = h[start:stop]
    af = np.abs(np.fft.fftfreq(WIN, DT))
    keep = af >= F_LOW
    H = np.fft.fft(seg)
    Hw = np.zeros_like(H)
    Hw[keep] = H[keep] / np.sqrt(np.interp(af[keep], f_psd, S))
    hw = np.fft.ifft(Hw)
    taper = np.ones(WIN)
    taper[:TAPER] = 0.5 * (1.0 - np.cos(np.pi * np.arange(TAPER) / TAPER))
    return m, hw * taper


def kernel_from(hw):
    K = np.conj(hw[L - 1 :: -1])
    return K / np.linalg.norm(K)


def generate(par, scale=1.0):
    """plus, cross for the row's parameters with both masses times `scale`."""
    from lalsimulation import SimInspiralTransformPrecessingNewInitialConditions
    from pycbc.waveform import get_td_waveform

    m1, m2 = par["mass_1"] * scale, par["mass_2"] * scale
    inc, s1x, s1y, s1z, s2x, s2y, s2z = SimInspiralTransformPrecessingNewInitialConditions(
        par["theta_jn"], par["phi_jl"], par["tilt_1"], par["tilt_2"], par["phi_12"],
        par["a_1"], par["a_2"], m1 * MSUN, m2 * MSUN, F_REF, par["phase"])
    hp, hc = get_td_waveform(approximant="IMRPhenomPv2", f_lower=F_MIN, f_ref=F_REF, delta_t=DT,
                             mass1=m1, mass2=m2, spin1x=s1x, spin1y=s1y, spin1z=s1z,
                             spin2x=s2x, spin2y=s2y, spin2z=s2z, inclination=inc, distance=100.0,
                             coa_phase=par["phase"])
    t_final = hp.sample_times.data[-1]
    w = np.stack([hp.data, hc.data])
    n = int(FS * DUR)
    shift = int((t_final - RIGHT_PAD + 1 / FS) * FS)
    if w.shape[-1] < n:
        w = np.roll(np.pad(w, ((0, 0), (n - w.shape[-1], 0))), shift, axis=-1)
    else:
        w = np.roll(w, shift, axis=-1)[:, -n:]
    return w[0], w[1]


def chirp_mass(m1, m2):
    return (m1 * m2) ** 0.6 / (m1 + m2) ** 0.2
