"""Raw per-window model-output dump (front-end for the SV pipeline).

Reads a pre-made sig/bkg file (``injected`` = background+signal, ``background``
= clean, both already at the model sample rate) and, for every kernel position,
writes the RAW model output ``(pred_chirp_mass, pred_sigma)`` -- no clustering
or integration. Those become cheap post-processing on the CSV.

The signal is injected in the file, NOT here (avoids the shift/alignment
pitfalls). Kernel positions are labelled RELATIVE TO THE MERGER, which sits at
sample ``(duration - right_pad) * sample_rate`` in every segment. A row's
``kernel_left_s .. kernel_right_s`` is the 4 s kernel span w.r.t. coalescence
(merger at 0): e.g. ``[-8, -4]`` = kernel ends 4 s before merger.

Preprocessing matches inference (PSD -> whiten -> normalize_input -> S4D).

Output: two tidy long CSVs (one row per strain x kernel position):
    <outdir>/raw_sig.csv   <outdir>/raw_bkg.csv
Pivot to the matrix sketch with:
    df.pivot(index="strain_id", columns="kernel_right_s",
             values=["pred_chirp_mass","pred_sigma"])

Run (GPU):
    uv run python save_outputs.py --outdir /path/out [--min-snr X]
"""

import argparse
import os

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from ml4gw.transforms import Whiten
from ml4gw.nn.ssm.s4d import S4Model
from utils.preprocessing import PsdEstimator


def load_net(ckpt_path, device):
    """Rebuild the bare S4Model + output-norm stats straight from the ckpt."""
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    hparams = checkpoint["hyper_parameters"]
    net = S4Model(
        d_input=hparams["d_input"],
        d_output=hparams["d_output"],
        d_model=hparams["d_model"],
        d_state=hparams["d_state"],
        n_layers=hparams["n_layers"],
        dropout=hparams["dropout"],
        dt_min=hparams.get("dt_min", 1e-3),
        dt_max=hparams.get("dt_max", 1.0),
    )
    state_dict = {
        key[len("model.") :]: value
        for key, value in checkpoint["state_dict"].items()
        if key.startswith("model.")
    }
    net.load_state_dict(state_dict, strict=True)
    net.eval().to(device)

    n_vars = hparams["d_output"] // 2
    y_mean = float(np.atleast_1d(checkpoint["state_dict"]["y_mean"])[0])
    y_std = float(np.atleast_1d(checkpoint["state_dict"]["y_std"])[0])
    normalize_input = bool(hparams.get("normalize_input", False))
    return net, n_vars, y_mean, y_std, normalize_input


# defaults: merger_4s id2 model + pre-made sig/bkg file
CKPT = "/n/holystore01/LABS/iaifi_lab/Lab/kyoon/MODEL/aframe/kyoon-dev/BNS-PUBLICATION/merger_4s/chirp_mass_snr_8_50_60-64s_d64_s64_l4_on_disk_id2/checkpoints/s4d_chirp_mass_mse_4023.ckpt"  # noqa: E501
DATA = "/n/holystore01/LABS/iaifi_lab/Lab/kyoon/DATA/aframe_data/test/waveforms_sig_bkg_50k.hdf5"  # noqa: E501
KERNEL, FDUR, PSD_LEN, FFTLEN, HIGHPASS = 4.0, 1.0, 20.0, 2.0, 20.0
WINDOW_SEC = PSD_LEN + FDUR + KERNEL  # 25 s context window fed to the model
# kernel right edge = window_start + this
RIGHT_EDGE_OFF = PSD_LEN + FDUR / 2 + KERNEL


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--ckpt", default=CKPT)
    parser.add_argument("--data", default=DATA, help="pre-made sig/bkg hdf5")
    parser.add_argument(
        "--max-strains",
        type=int,
        default=None,
        help="cap number of strains (default all)",
    )
    parser.add_argument("--min-snr", type=float, default=0.0)
    # kernel right edge (rel. merger) grid: default -> spans [-8,-4]..[-3,1]
    parser.add_argument("--edge-min", type=float, default=-4.0)
    parser.add_argument("--edge-max", type=float, default=1.0)
    parser.add_argument("--edge-step", type=float, default=0.25)
    parser.add_argument(
        "--row-batch",
        type=int,
        default=4,
        help="strains per model batch; model batch = row_batch*n_edges "
        "(keep <~6 so batch stays <128 -> fits 20GB MIG; S4D FFT ~ batch*seq)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    torch.set_grad_enabled(False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net, n_vars, y_mean, y_std, normalize_input = load_net(args.ckpt, device)
    softplus = nn.Softplus()

    with h5py.File(args.data, "r") as h5:
        sample_rate = float(h5.attrs["sample_rate"])
        duration = float(h5.attrs["duration"])
        right_pad = float(h5.attrs["right_pad"])
        n_strains = int(h5.attrs["length"])
        segment_len = int(round(duration * sample_rate))
        merger_idx = int(round((duration - right_pad) * sample_rate))

        params = h5["parameters"]
        snr = params["snr"][:]
        injection_time = params["injection_time"][:]
        mass_1 = params["mass_1"][:]
        mass_2 = params["mass_2"][:]
        redshift = (
            params["redshift"][:]
            if "redshift" in params
            else np.zeros_like(snr)
        )
        chirp_det = (mass_1 * mass_2) ** 0.6 / (mass_1 + mass_2) ** 0.2
        chirp_src = chirp_det / (1 + redshift)

        psd_estimator = PsdEstimator(
            KERNEL + FDUR, sample_rate, FFTLEN, average="median", fast=True
        ).to(device)
        whiten = Whiten(FDUR, sample_rate, HIGHPASS, None).to(device)

        edges = np.arange(args.edge_min, args.edge_max + 1e-9, args.edge_step)
        n_edges = len(edges)
        window_len = int(round(WINDOW_SEC * sample_rate))
        window_starts = [
            merger_idx + int(round((edge - RIGHT_EDGE_OFF) * sample_rate))
            for edge in edges
        ]
        assert all(
            0 <= start and start + window_len <= segment_len
            for start in window_starts
        ), "window falls outside segment"

        strain_idxs = np.arange(n_strains)
        if args.min_snr > 0:
            strain_idxs = strain_idxs[snr[strain_idxs] >= args.min_snr]
        if args.max_strains:
            strain_idxs = strain_idxs[: args.max_strains]
        print(
            f"device={device}  strains={len(strain_idxs)}  "
            f"merger_idx={merger_idx}  edges={n_edges}  "
            f"y_mean={y_mean:.3f} y_std={y_std:.3f}",
            flush=True,
        )

        def score(windows):
            """(B, 2, window_len) strain -> physical (mean, sigma) per row."""
            batch = torch.from_numpy(windows).float().to(device)
            batch, psds = psd_estimator(batch)
            batch = whiten(batch, psds)
            if normalize_input:
                batch = batch / batch.std(dim=-1, keepdim=True).clamp(min=1e-8)
            outputs = net(batch)
            mean = (outputs[:, 0] * y_std + y_mean).cpu().numpy()
            sigma = torch.sqrt(softplus(outputs[:, n_vars:])[:, 0]) * y_std
            return mean, sigma.cpu().numpy()

        injected_h1, injected_l1 = h5["injected/h1"], h5["injected/l1"]
        background_h1, background_l1 = h5["background/h1"], h5["background/l1"]
        sig_rows, bkg_rows = [], []

        for batch_start in range(0, len(strain_idxs), args.row_batch):
            batch_idxs = strain_idxs[
                batch_start : batch_start + args.row_batch
            ]
            sig_windows, bkg_windows = [], []
            for idx in batch_idxs:
                sig = np.stack([injected_h1[idx], injected_l1[idx]]).astype(
                    np.float32
                )
                bkg = np.stack(
                    [background_h1[idx], background_l1[idx]]
                ).astype(np.float32)
                sig_windows.extend(
                    sig[:, start : start + window_len]
                    for start in window_starts
                )
                bkg_windows.extend(
                    bkg[:, start : start + window_len]
                    for start in window_starts
                )
            sig_mean, sig_sigma = score(np.stack(sig_windows))
            bkg_mean, bkg_sigma = score(np.stack(bkg_windows))

            for row_in_batch, idx in enumerate(batch_idxs):
                truth = {
                    "strain_id": int(idx),
                    "injection_time": float(injection_time[idx]),
                    "snr": float(snr[idx]),
                    "mass_1": float(mass_1[idx]),
                    "mass_2": float(mass_2[idx]),
                    "redshift": float(redshift[idx]),
                    "true_chirp_mass_det": float(chirp_det[idx]),
                    "true_chirp_mass_src": float(chirp_src[idx]),
                }
                for edge_i, edge in enumerate(edges):
                    flat = row_in_batch * n_edges + edge_i
                    base_row = dict(
                        truth,
                        kernel_left_s=round(float(edge) - KERNEL, 4),
                        kernel_right_s=round(float(edge), 4),
                    )
                    sig_rows.append(
                        dict(
                            base_row,
                            pred_chirp_mass=float(sig_mean[flat]),
                            pred_sigma=float(sig_sigma[flat]),
                        )
                    )
                    bkg_rows.append(
                        dict(
                            base_row,
                            pred_chirp_mass=float(bkg_mean[flat]),
                            pred_sigma=float(bkg_sigma[flat]),
                        )
                    )

            done = batch_start + len(batch_idxs)
            if (batch_start // args.row_batch) % 20 == 0 or done == len(
                strain_idxs
            ):
                print(f"  {done}/{len(strain_idxs)} strains", flush=True)

    sig_path = os.path.join(args.outdir, "raw_sig.csv")
    bkg_path = os.path.join(args.outdir, "raw_bkg.csv")
    pd.DataFrame(sig_rows).to_csv(sig_path, index=False)
    pd.DataFrame(bkg_rows).to_csv(bkg_path, index=False)
    print(f"wrote {len(sig_rows)} rows -> {sig_path}", flush=True)
    print(f"wrote {len(bkg_rows)} rows -> {bkg_path}", flush=True)


if __name__ == "__main__":
    main()
