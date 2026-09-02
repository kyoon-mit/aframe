"""Step 0: pre-extract one branch's injection subset from the full
injection-set file, so step1 (on scarce gpu_test time) does a fast full-load
of a small cached file instead of a slow fancy-indexed HDF5 read against the
full (tens-of-GB) injection set every branch.

``ledger.injections.InjectionParameterSet.read`` filters by boolean mask then
does ``_load_with_idx`` -- non-contiguous per-row HDF5 reads that take ~10+
minutes per branch on network storage, even though the actual Triton
streaming (step1) only takes ~10-15s. That work doesn't need the GPU, so it
belongs on a CPU-only partition with far more cores and no gpu_test QOS cap.

Usage (one branch):
    uv run python step0_cache_injections.py \\
        --injection_set_fname /path/to/injections.hdf5 \\
        --background_fname /path/to/background-....hdf5 \\
        --ifos H1 L1 --shifts 0 1 \\
        --outdir /path/to/cache/branch_0
"""

import argparse
import os

import h5py

from ledger.injections import InterferometerResponseSet, waveform_class_factory


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--injection_set_fname", required=True)
    p.add_argument("--background_fname", required=True)
    p.add_argument("--ifos", nargs="+", required=True)
    p.add_argument("--shifts", nargs="+", type=float, required=True)
    p.add_argument("--outdir", required=True)
    args = p.parse_args()

    out_path = os.path.join(args.outdir, "injections.hdf5")
    if os.path.exists(out_path):
        print(f"cache exists, skip: {out_path}", flush=True)
        return

    with h5py.File(args.background_fname, "r") as f:
        dataset = f[args.ifos[0]]
        sample_rate = 1 / dataset.attrs["dx"]
        t0 = dataset.attrs["x0"]
        duration = len(dataset) / sample_rate

    cls = waveform_class_factory(
        args.ifos, InterferometerResponseSet, "ResponseSet"
    )
    injection_set = cls.read(
        args.injection_set_fname,
        start=t0,
        end=t0 + duration,
        shifts=args.shifts,
    )

    os.makedirs(args.outdir, exist_ok=True)
    injection_set.write(out_path)
    print(f"wrote {out_path} ({len(injection_set)} injections)", flush=True)


if __name__ == "__main__":
    main()
