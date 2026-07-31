"""Generate an InterferometerResponseSet for sensitive-volume evaluation.

Produces ``waveforms.hdf5`` (projected H1/L1 waveforms with injection_time and
shift) + ``rejected-parameters.hdf5`` in the output directory. This is the
"test"/injection-campaign set the SV pipeline consumes; it is NOT raw
polarizations (training) nor a projected SNR-filtered WaveformSet (validation).

To get far more injections than a single zero-lag timeline allows, a fresh
population is drawn into each of ``n_shifts`` time-slides (shift = [0, k] for
k = 1..n_shifts). Each slide is an independent noise realization, so the total
injection count is roughly ``n_shifts * span / (spacing + waveform_duration)``.
The per-slide files are merged with ``Ledger.aggregate`` (streamed, so memory
stays at ~one slide). The matching infer step injects each slide's injections
into its time-shifted background.

Usage
-----
    python scripts/generate_test_injection_set.py \
        --background_dir /n/.../DATA/O3b_H1_L1_4096Hz \
        --output_dir     /n/.../DATA/aframe_data/test \
        --n_shifts 36 --pool 32
"""

import argparse
import logging
import shutil
from pathlib import Path

import h5py

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s"
)
log = logging.getLogger(__name__)


def get_gps_range(background_dir: str) -> tuple[float, float]:
    files = sorted(Path(background_dir).glob("background-*.hdf5"))
    if not files:
        raise FileNotFoundError(f"No background-*.hdf5 in {background_dir}")
    starts, ends = [], []
    for f in files:
        with h5py.File(f, "r") as hf:
            x0 = float(hf["H1"].attrs["x0"])
            dx = float(hf["H1"].attrs["dx"])
            n = len(hf["H1"])
            starts.append(x0)
            ends.append(x0 + n * dx)
    return min(starts), max(ends)


def main(
    background_dir: str,
    output_dir: str,
    prior: str = "end_o3_ratesandpops_bns",
    n_shifts: int = 1,
    pool: int = None,
    spacing: float = 64.0,
    buffer: float = 64.0,
    snr_threshold: float = 4.0,
    sample_rate: float = 2048.0,
    waveform_duration: float = 32.0,
    right_pad: float = 2.0,
    seed: int = 42,
):
    import importlib

    from data.waveforms.testing import testing_waveforms
    from ledger.injections import (
        InjectionParameterSet,
        InterferometerResponseSet,
        waveform_class_factory,
    )

    module, _, name = prior.rpartition(".")
    prior_fn = getattr(
        importlib.import_module(module or "priors.priors"), name
    )

    start, end = get_gps_range(background_dir)
    log.info(
        f"{Path(background_dir).name} GPS range: {start:.0f} - {end:.0f}  "
        f"({(end - start) / 86400:.2f} d)"
    )
    psd_file = sorted(Path(background_dir).glob("background-*.hdf5"))[0]
    log.info(f"PSD file: {psd_file}")
    ifos = ["H1", "L1"]

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    tmp = out / "_per_shift"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir()

    wf_files, rej_files = [], []
    for k in range(1, n_shifts + 1):
        d = tmp / f"shift_{k:04d}"
        d.mkdir()
        log.info(f"--- slide {k}/{n_shifts}  shift=[0, {k}] ---")
        wf, rej = testing_waveforms(
            start=start,
            end=end,
            ifos=ifos,
            shifts=[0.0, float(k)],
            spacing=spacing,
            buffer=buffer,
            prior=prior_fn,
            minimum_frequency=20.0,
            reference_frequency=50.0,
            sample_rate=sample_rate,
            waveform_duration=waveform_duration,
            waveform_approximant="IMRPhenomPv2",
            right_pad=right_pad,
            highpass=20.0,
            lowpass=None,
            snr_threshold=snr_threshold,
            psd_file=psd_file,
            max_num_samples=4096,
            output_dir=d,
            seed=seed,
            pool=pool,
        )
        wf_files.append(Path(wf))
        rej_files.append(Path(rej))

    ResponseSet = waveform_class_factory(
        ifos, InterferometerResponseSet, "ResponseSet"
    )
    log.info(f"aggregating {len(wf_files)} slide files -> {out}")
    ResponseSet.aggregate(wf_files, out / "waveforms.hdf5", clean=True)
    InjectionParameterSet.aggregate(
        rej_files, out / "rejected-parameters.hdf5", clean=True
    )
    shutil.rmtree(tmp)

    with h5py.File(out / "waveforms.hdf5", "r") as f:
        n = f.attrs["length"]
    with h5py.File(out / "rejected-parameters.hdf5", "r") as f:
        nr = f.attrs["length"]
    log.info(
        f"DONE: {n} injections across {n_shifts} slides "
        f"({nr} rejected) -> {out}/waveforms.hdf5"
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--background_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--prior", default="end_o3_ratesandpops_bns")
    p.add_argument("--n_shifts", type=int, default=1)
    p.add_argument("--pool", type=int, default=None)
    p.add_argument("--spacing", type=float, default=64.0)
    p.add_argument("--buffer", type=float, default=64.0)
    p.add_argument("--snr_threshold", type=float, default=4.0)
    p.add_argument("--sample_rate", type=float, default=2048.0)
    p.add_argument("--waveform_duration", type=float, default=32.0)
    p.add_argument("--right_pad", type=float, default=2.0)
    p.add_argument("--seed", type=int, default=42)
    main(**vars(p.parse_args()))
