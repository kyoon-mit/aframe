"""Direct model-response check (no sliding/clustering/recovery).

For the loudest injections: inject into real background, scan the kernel alignment
the way training placed the signal, run the EXACT inference preprocessing, and report
predicted chirp_mass + sigma at the most-confident alignment vs. signal-free background.

If predicted Mc tracks the truth and sigma drops well below background -> the model
works and the SV pipeline (clustering/recovery) is what loses it. If even a perfectly
aligned loud injection gives sigma ~ prior width -> the model/preprocessing is the limit.
"""
import glob
import numpy as np
import h5py
import torch
import torch.nn as nn
import torchaudio

from ml4gw.transforms import Whiten
from utils.preprocessing import PsdEstimator
from ledger.injections import InterferometerResponseSet, waveform_class_factory
from train.model.regression import LitS4DGaussianNLL

CKPT = "/n/holystore01/LABS/iaifi_lab/Lab/kyoon/aframe_linoss/projects/train/logs/BNS-PUBLICATION/chirp_mass_snr_4_50_merger_4s_d64_s64_l4_aframe_kyoon_dev_1/checkpoints/s4d_chirp_mass_mse_1959.ckpt"
BG_DIR = "/n/holystore01/LABS/iaifi_lab/Lab/kyoon/DATA/O3a_H1_L1_4096Hz"
INJ = "/n/holystore01/LABS/iaifi_lab/Lab/kyoon/aframe_linoss/runs/regression_sv/bns_injections/waveforms.hdf5"
IFOS = ["H1", "L1"]
RAW_SR, SR = 4096, 2048
KERNEL, FDUR, PSD_LEN, FFTLEN, HP = 4.0, 1.0, 20.0, 2.0, 20.0
W_SEC = PSD_LEN + FDUR + KERNEL                 # 25 s window fed to the model
RIGHT_EDGE_OFF = PSD_LEN + FDUR / 2 + KERNEL    # kernel right edge = window_start + this (24.5)
N_INJ = 8
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    torch.set_grad_enabled(False)
    model = LitS4DGaussianNLL.load_from_checkpoint(CKPT, strict=False).eval().to(DEV)
    nv = model.n_vars
    ymean = float(np.atleast_1d(model.y_mean.cpu().numpy())[0])
    ystd = float(np.atleast_1d(model.y_std.cpu().numpy())[0])
    sp = nn.Softplus()
    resamp = torchaudio.transforms.Resample(RAW_SR, SR).to(DEV)
    psd = PsdEstimator(KERNEL + FDUR, SR, FFTLEN, average="median", fast=True).to(DEV)
    wh = Whiten(FDUR, SR, HP, None).to(DEV)

    def score(win):                              # win (B, nifo, W_raw)
        x = torch.from_numpy(win).float().to(DEV)
        x = resamp(x)
        x, psds = psd(x)
        x = wh(x, psds)
        x = model._prepare_input(x)
        out = model(x)
        mean = out[:, 0].cpu().numpy() * ystd + ymean
        sigma = torch.sqrt(sp(out[:, nv:])[:, 0]).cpu().numpy() * ystd
        return mean, sigma

    segs = []
    for fn in sorted(glob.glob(f"{BG_DIR}/*.hdf5")):
        with h5py.File(fn) as f:
            ds = f[IFOS[0]]; t0 = float(ds.attrs["x0"]); n = len(ds)
        segs.append((t0, t0 + n / RAW_SR, fn))

    with h5py.File(INJ) as f:
        p = f["parameters"]
        snr = p["snr"][:]; itime = p["injection_time"][:]
        m1 = p["mass_1"][:]; m2 = p["mass_2"][:]
        z = p["redshift"][:] if "redshift" in p else np.zeros_like(snr)
        cm = p["chirp_mass"][:] if "chirp_mass" in p else (m1 * m2) ** 0.6 / (m1 + m2) ** 0.2
    mc_det = (m1 * m2) ** 0.6 / (m1 + m2) ** 0.2          # detector-frame from stored masses
    mc_src = mc_det / (1 + z)

    chosen = []
    for idx in np.argsort(snr)[::-1]:
        T = float(itime[idx])
        for (t0, t1, fn) in segs:
            if T - 30 >= t0 and T + 4 <= t1:
                chosen.append((idx, T, t0, fn)); break
        if len(chosen) >= N_INJ:
            break

    cls = waveform_class_factory(IFOS, InterferometerResponseSet, "RS")
    edges = np.arange(-0.5, 2.0 + 1e-9, 0.125)            # kernel-right-edge minus T
    Wn = int(round(W_SEC * RAW_SR))

    print(f"device={DEV}  y_mean={ymean:.3f} y_std={ystd:.3f}  (prior width ~= y_std)")
    print(f"{'#':>2} {'SNR':>5} {'Mc_src':>6} {'Mc_det':>6} | "
          f"{'INJ predMc':>10} {'INJ sigma':>9} {'@edge-T':>7} | {'BG predMc':>9} {'BG sig(med)':>11} {'BG sig(min)':>11}")
    for c, (idx, T, t0seg, fn) in enumerate(chosen):
        lo = T - 30.0
        ilo = int(round((lo - t0seg) * RAW_SR)); ihi = int(round((T + 4.0 - t0seg) * RAW_SR))
        with h5py.File(fn) as f:
            bg = np.stack([f[ifo][ilo:ihi] for ifo in IFOS]).astype(np.float32)
        sub = cls.read(INJ, start=lo, end=T + 4.0, shifts=[0.0, 0.0])
        injd = sub.inject(bg.copy(), lo)
        starts = [int(round((T + e - RIGHT_EDGE_OFF - lo) * RAW_SR)) for e in edges]
        wi = np.stack([injd[:, s:s + Wn] for s in starts])
        wb = np.stack([bg[:, s:s + Wn] for s in starts])
        mi, si = score(wi)
        mb, sb = score(wb)
        k = int(np.argmin(si))
        print(f"{c:>2} {snr[idx]:>5.1f} {mc_src[idx]:>6.3f} {mc_det[idx]:>6.3f} | "
              f"{mi[k]:>10.3f} {si[k]:>9.3f} {edges[k]:>7.2f} | "
              f"{np.median(mb):>9.3f} {np.median(sb):>11.3f} {sb.min():>11.3f}")


if __name__ == "__main__":
    main()
