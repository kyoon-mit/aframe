"""Probe: regenerate the 1.2 Msun training row with pycbc and compare to the stored polarizations."""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "4"
import sys
import h5py
import numpy as np
from lalsimulation import SimInspiralTransformPrecessingNewInitialConditions
from pycbc.waveform import get_td_waveform

WF = "/home/kyoon/SSM-BNS/DATA/aframe/train/end_o3_ratesandpops_bns_uniform_chirp.hdf5"
MSUN = 1.988409870698051e30
row = int(sys.argv[1]) if len(sys.argv) > 1 else 24496
apx_list = sys.argv[2].split(",") if len(sys.argv) > 2 else ["IMRPhenomPv2"]
with h5py.File(WF, "r") as f:
    par = {k: float(f["parameters"][k][row]) for k in f["parameters"]}
    plus = f["waveforms/plus"][row]
    cross = f["waveforms/cross"][row]
print(par)
fs, dur, right_pad, fmin, fref = 2048.0, 32.0, 2.0, 20.0, 50.0
inc, s1x, s1y, s1z, s2x, s2y, s2z = SimInspiralTransformPrecessingNewInitialConditions(
    par["theta_jn"], par["phi_jl"], par["tilt_1"], par["tilt_2"], par["phi_12"],
    par["a_1"], par["a_2"], par["mass_1"] * MSUN, par["mass_2"] * MSUN, fref, par["phase"])


def align(stacked, t_final):
    n = int(fs * dur)
    def shift(w):
        st = t_final - right_pad + 1 / fs
        return np.roll(w, int(st * fs), axis=-1)
    if stacked.shape[-1] < n:
        stacked = np.pad(stacked, ((0, 0), (n - stacked.shape[-1], 0)))
        stacked = shift(stacked)
    else:
        stacked = shift(stacked)[:, -n:]
    return stacked


for apx in apx_list:
    hp, hc = get_td_waveform(approximant=apx, f_lower=fmin, f_ref=fref, delta_t=1 / fs,
                             mass1=par["mass_1"], mass2=par["mass_2"], spin1x=s1x, spin1y=s1y, spin1z=s1z,
                             spin2x=s2x, spin2y=s2y, spin2z=s2z, inclination=inc, distance=100.0,
                             coa_phase=par["phase"])
    w = align(np.stack([hp.data, hc.data]), hp.sample_times.data[-1])
    for name, a, b in (("plus", w[0], plus), ("cross", w[1], cross)):
        scale = np.vdot(a, b).real / np.vdot(a, a).real
        rel = np.linalg.norm(b - scale * a) / np.linalg.norm(b)
        print(f"{apx} {name}: amp ratio stored/regen {scale:.6e}, rel residual {rel:.3e}, peak idx regen {np.argmax(np.abs(a))} stored {np.argmax(np.abs(b))}")
