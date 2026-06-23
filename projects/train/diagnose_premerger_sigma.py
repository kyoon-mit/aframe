"""Pre-merger sigma diagnostic.

Slides the trained chirp-mass regression model over a handful of HIGH-SNR
injections and records the predicted chirp_mass sigma as a function of time
relative to coalescence.  This answers the question the SV pipeline cannot:

    does the model actually produce a confident (low-sigma) prediction, and
    if so, at what time offset from the merger?

For each chosen injection it scans the kernel center over [T-10, T+2] s and
also scores the SAME background with no injection (baseline noise sigma).
A dip of injected sigma below baseline near t-T ~ -3.5 s confirms the model
works and that the SV pipeline is reading it at the wrong time.

Outputs a PNG + text summary; does NOT touch the SV pipeline.
"""

import glob
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
import torchaudio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ml4gw.transforms import Whiten
from utils.preprocessing import PsdEstimator
from ledger.injections import InterferometerResponseSet, waveform_class_factory
from train.model.regression import LitS4DGaussianNLL

# ── config (matches training chirp_mass_1s_d64_s64_l4.yaml) ───────────────── #
CKPT = "/n/holystore01/LABS/iaifi_lab/Lab/kyoon/aframe_linoss/projects/train/logs/aframe/chirp_mass_snr_4_50_59-60s_d64_s64_l4_id11/checkpoints/s4d_chirp_mass_mse_927.ckpt"
BG_DIR = "/n/holystore01/LABS/iaifi_lab/Lab/kyoon/DATA/O3a_H1_L1_4096Hz"
INJ = "/n/holystore01/LABS/iaifi_lab/Lab/kyoon/aframe_linoss/runs/regression_sv/bns_injections/waveforms.hdf5"
OUTDIR = Path("/n/holystore01/LABS/iaifi_lab/Lab/kyoon/aframe_linoss/runs/regression_sv/premerger_diag")

IFOS = ["H1", "L1"]
RAW_SR, SR = 4096, 256
KERNEL, FDUR, PSD_LEN, FFTLEN, HIGHPASS = 1.0, 1.0, 20.0, 2.0, 20.0
N_INJ = 6                       # number of loudest injections to scan
SCAN_LO, SCAN_HI, SCAN_DT = -10.0, 2.0, 0.0625   # kernel-center time rel. to T
WINSTART_OFF = PSD_LEN + FDUR / 2 + KERNEL / 2    # window_start = tc - this
W_SEC = PSD_LEN + FDUR + KERNEL
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    torch.set_grad_enabled(False)

    model = LitS4DGaussianNLL.load_from_checkpoint(CKPT, strict=False).eval().to(DEV)
    n_vars = model.n_vars
    y_std = float(np.atleast_1d(getattr(model, "y_std", [1.0]))[0])
    y_mean = float(np.atleast_1d(getattr(model, "y_mean", [0.0]))[0])
    softplus = nn.Softplus()
    print(f"model loaded on {DEV}; n_vars={n_vars} y_mean={y_mean} y_std={y_std}")

    resampler = torchaudio.transforms.Resample(RAW_SR, SR).to(DEV)
    psd_est = PsdEstimator(KERNEL + FDUR, SR, FFTLEN, average="median", fast=True).to(DEV)
    whitener = Whiten(FDUR, SR, HIGHPASS).to(DEV)

    def score(windows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """windows: (B, n_ifos, W_raw) -> (sigma_norm, mean_norm) arrays."""
        sig, mu = [], []
        for i in range(0, len(windows), 64):
            x = torch.from_numpy(windows[i : i + 64]).float().to(DEV)
            x = resampler(x)
            x, psds = psd_est(x)
            x = whitener(x, psds)
            x = model._prepare_input(x)
            out = model(x)
            var = softplus(out[:, n_vars:])
            sig.append(torch.sqrt(var[:, 0]).cpu().numpy())
            mu.append(out[:, 0].cpu().numpy())
        return np.concatenate(sig), np.concatenate(mu)

    # ── segment index ─────────────────────────────────────────────────────── #
    segs = []
    for fn in sorted(glob.glob(f"{BG_DIR}/*.hdf5")):
        with h5py.File(fn, "r") as f:
            ds = f[IFOS[0]]
            t0 = float(ds.attrs["x0"]); n = len(ds)
        segs.append((t0, t0 + n / RAW_SR, fn, n))

    # ── pick loudest injections that fit a segment with room for the scan ──── #
    cls = waveform_class_factory(IFOS, InterferometerResponseSet, "RS")
    full = cls.read(INJ)
    times = np.asarray(full.injection_time)
    rank_field = "snr" if hasattr(full, "snr") and full.snr is not None else "distance"
    rank = np.asarray(getattr(full, rank_field))
    cm = np.asarray(full.chirp_mass) if hasattr(full, "chirp_mass") else None
    order = np.argsort(rank)[:: 1 if rank_field == "distance" else -1]  # loud first
    print(f"ranking injections by {rank_field}")

    chosen = []
    for idx in order:
        T = float(times[idx])
        for (t0, t1, fn, n) in segs:
            if T - 70 >= t0 and T + 6 <= t1:
                chosen.append((idx, T, t0, fn))
                break
        if len(chosen) >= N_INJ:
            break
    print(f"chosen {len(chosen)} injections")

    tcs = np.arange(SCAN_LO, SCAN_HI + 1e-9, SCAN_DT)
    Wn = int(round(W_SEC * RAW_SR))

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(chosen)))
    summary = []

    for c, (idx, T, t0seg, fn) in enumerate(chosen):
        lo = T - 70.0
        i_lo = int(round((lo - t0seg) * RAW_SR))
        i_hi = int(round((T + 6.0 - t0seg) * RAW_SR))
        with h5py.File(fn, "r") as f:
            bg = np.stack([f[ifo][i_lo:i_hi] for ifo in IFOS]).astype(np.float32)
        injd = full.inject(bg.copy(), lo)

        starts = [int(round((T + tc - WINSTART_OFF - lo) * RAW_SR)) for tc in tcs]
        win_inj = np.stack([injd[:, s : s + Wn] for s in starts])
        win_bg = np.stack([bg[:, s : s + Wn] for s in starts])
        sig_inj, mu_inj = score(win_inj)
        sig_bg, _ = score(win_bg)

        kmin = int(np.argmin(sig_inj))
        dip_t, dip_sig = tcs[kmin], sig_inj[kmin]
        base = float(np.median(sig_bg))
        true_cm = float(cm[idx]) if cm is not None else float("nan")
        pred_cm = mu_inj[kmin] * y_std + y_mean
        snrv = float(rank[idx]) if rank_field == "snr" else float("nan")
        summary.append((c, snrv, true_cm, dip_t, dip_sig, base, dip_sig / base, pred_cm))

        ax.plot(tcs, sig_inj, color=colors[c], lw=1.5,
                label=f"inj{c} {rank_field}={rank[idx]:.1f} Mc={true_cm:.2f}")
        ax.plot(tcs, sig_bg, color=colors[c], lw=0.8, ls=":", alpha=0.6)

    ax.axvline(-3.5, color="r", ls="--", lw=1, label="trained window (~-3.5 s)")
    ax.set_xlabel("kernel-center time - coalescence  [s]")
    ax.set_ylabel("predicted chirp_mass sigma (normalized)")
    ax.set_title("Pre-merger sigma: injected (solid) vs background (dotted)")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(OUTDIR / "premerger_sigma.png", dpi=130, bbox_inches="tight")
    print(f"saved {OUTDIR/'premerger_sigma.png'}")

    print("\n=== SUMMARY (sigma normalized; dip/base < 1 means signal more confident than noise) ===")
    print(f"{'inj':>3} {'snr':>6} {'true_Mc':>8} {'dip_t[s]':>9} {'dip_sig':>8} {'base_sig':>8} {'dip/base':>9} {'pred_Mc':>8}")
    for (c, snrv, tcm, dt, ds_, bs, ratio, pcm) in summary:
        print(f"{c:>3} {snrv:>6.1f} {tcm:>8.3f} {dt:>9.2f} {ds_:>8.4f} {bs:>8.4f} {ratio:>9.3f} {pcm:>8.3f}")


if __name__ == "__main__":
    main()
