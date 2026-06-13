"""Generate premerger BNS waveforms for 57-58s, 58-59s, 59-60s windows.

For each window, generates:
  - training_waveforms.hdf5  (WaveformPolarizationSet, cross/plus)
  - val_waveforms.hdf5       (WaveformSet, projected onto H1+L1, SNR >= 4)

right_pad encodes pre-merger offset: coal at (64 - right_pad) seconds.
  59-60s: right_pad=4.0  (coal at 60s)
  58-59s: right_pad=5.0  (coal at 59s)
  57-58s: right_pad=6.0  (coal at 58s)
"""

import sys
from pathlib import Path

import h5py
import numpy as np
import scipy.signal
import torch

REPO = Path("/n/holystore01/LABS/iaifi_lab/Lab/kyoon/aframe_linoss")
# data project source on the path so we can use its utils
sys.path.insert(0, str(REPO / "projects/data"))

from ledger.injections import WaveformPolarizationSet, WaveformSet, waveform_class_factory
from priors.priors import end_o3_ratesandpops_bns
from data.waveforms.training import training_waveforms
from data.waveforms.rejection import rejection_sample

# ------------------------------------------------------------------ #
# Config                                                               #
# ------------------------------------------------------------------ #
DATA_DIR  = Path("/n/holystore01/LABS/iaifi_lab/Lab/kyoon/DATA")
OUT_DIR   = DATA_DIR / "aframe/waveforms_64s_256Hz"
BG_DIR    = DATA_DIR / "O3a_H1_L1_4096Hz"

SAMPLE_RATE  = 256
DURATION     = 64.0
F_MIN        = 20.0
F_REF        = 50.0
APPROXIMANT  = "IMRPhenomPv2"
IFOS         = ["H1", "L1"]
NUM_TRAIN    = 200_000
NUM_VAL      = 4_096
SNR_THRESH   = 4.0
LOWPASS      = 1024.0

# right_pad per window
WINDOWS = {
    "59-60s": 4.0,
    "58-59s": 5.0,
    "57-58s": 6.0,
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")


# ------------------------------------------------------------------ #
# PSD estimation from a background HDF5 file                          #
# ------------------------------------------------------------------ #

def estimate_psd(bg_file: Path, ifos: list, sample_rate: float, df: float) -> torch.Tensor:
    """Estimate PSD from an aframe background HDF5 file using Welch's method."""
    print(f"Estimating PSD from {bg_file.name} ...")
    nperseg = int(sample_rate / df)   # samples per segment = duration in samples
    psds = []
    with h5py.File(bg_file, "r") as f:
        for ifo in ifos:
            raw = f[ifo][:]
            raw_sr = int(round(1.0 / f[ifo].attrs["dx"]))
            # Downsample if background is at higher rate than needed
            if raw_sr != int(sample_rate):
                factor = raw_sr // int(sample_rate)
                raw = raw[::factor]
            freqs, psd = scipy.signal.welch(
                raw, fs=sample_rate, nperseg=nperseg, window="hann", average="median"
            )
            psds.append(psd)
    psd_tensor = torch.tensor(np.stack(psds), dtype=torch.float64)
    print(f"PSD shape: {psd_tensor.shape}, freq resolution: {freqs[1]:.4f} Hz")
    return psd_tensor


# Pick a background file long enough for a 64s PSD window
bg_files = sorted(BG_DIR.glob("background-*.hdf5"))
bg_files = [f for f in bg_files if int(f.stem.split("-")[-1]) >= int(DURATION * 2)]
assert bg_files, f"No background files long enough in {BG_DIR}"
bg_file = bg_files[0]

psd = estimate_psd(bg_file, IFOS, SAMPLE_RATE, df=1.0 / DURATION)


# ------------------------------------------------------------------ #
# Generate waveforms for each window                                   #
# ------------------------------------------------------------------ #
IfoWaveformSet = waveform_class_factory(IFOS, WaveformSet, "IfoWaveformSet")

for window_name, right_pad in WINDOWS.items():
    window_dir = OUT_DIR / window_name
    window_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*60}")
    print(f"Window {window_name}  right_pad={right_pad}s  coal@{DURATION - right_pad:.0f}s")

    # ---- Training waveforms ----------------------------------------
    train_file = window_dir / "training_waveforms.hdf5"
    if train_file.exists():
        print(f"  training_waveforms.hdf5 already exists, skipping")
    else:
        print(f"  Generating {NUM_TRAIN:,} training waveforms ...")
        waveforms = training_waveforms(
            num_signals=NUM_TRAIN,
            sample_rate=SAMPLE_RATE,
            waveform_duration=DURATION,
            prior=end_o3_ratesandpops_bns,
            minimum_frequency=F_MIN,
            reference_frequency=F_REF,
            waveform_approximant=APPROXIMANT,
            right_pad=right_pad,
        )
        chunks = (min(64, NUM_TRAIN), int(SAMPLE_RATE * DURATION))
        waveforms.write(str(train_file), chunks=chunks)
        print(f"  Saved → {train_file}")

    # ---- Validation waveforms (projected, SNR-filtered) -------------
    val_file = window_dir / "val_waveforms.hdf5"
    if val_file.exists():
        print(f"  val_waveforms.hdf5 already exists, skipping")
    else:
        print(f"  Generating {NUM_VAL:,} validation waveforms (SNR >= {SNR_THRESH}) ...")
        parameters, rejected = rejection_sample(
            num_signals=NUM_VAL,
            prior=end_o3_ratesandpops_bns,
            ifos=IFOS,
            minimum_frequency=F_MIN,
            reference_frequency=F_REF,
            sample_rate=SAMPLE_RATE,
            waveform_duration=DURATION,
            waveform_approximant=APPROXIMANT,
            right_pad=right_pad,
            highpass=F_MIN,
            lowpass=LOWPASS,
            snr_threshold=SNR_THRESH,
            psd=psd,
            max_num_samples=500_000,
        )
        waveform_set = IfoWaveformSet(**parameters)
        waveform_set.write(str(val_file))
        print(f"  Saved → {val_file}")

print("\nAll done.")
