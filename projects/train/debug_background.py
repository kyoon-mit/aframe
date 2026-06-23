"""CPU-only sanity check of the O3a background HDF5 files used for training.

For each file + ifo it reports: length, NaN/inf counts, exact-zero fraction,
the longest run of consecutive identical samples (gaps are often zero- or
constant-filled), and basic amplitude stats.  A near-zero / constant / gappy
segment yields a near-zero PSD bin, which makes the whitener blow up
(explains the huge max|X| seen in training).
"""
import glob
import sys

import h5py
import numpy as np


def longest_constant_run(x):
    # vectorized longest run of consecutive identical samples
    d = (np.diff(x) == 0).view(np.int8)
    if not d.any():
        return 1
    edges = np.flatnonzero(np.diff(np.concatenate(([0], d, [0]))))
    runs = edges[1::2] - edges[::2]
    return int(runs.max()) + 1


def main(bg_dir, ifos=("H1", "L1")):
    files = sorted(glob.glob(f"{bg_dir}/background/*.hdf5")) or sorted(glob.glob(f"{bg_dir}/*.hdf5"))
    print(f"found {len(files)} files in {bg_dir}")
    worst = []
    for f in files:
        with h5py.File(f, "r") as h:
            row = [f.split('/')[-1]]
            for ifo in ifos:
                if ifo not in h:
                    row.append(f"{ifo}:MISSING"); continue
                x = h[ifo][:]
                n = x.size
                nan = int(np.isnan(x).sum()); inf = int(np.isinf(x).sum())
                zero_frac = float((x == 0).mean())
                fin = x[np.isfinite(x)]
                amax = float(np.abs(fin).max()) if fin.size else float("nan")
                amin_nz = float(np.abs(fin[fin != 0]).min()) if np.any(fin != 0) else 0.0
                std = float(fin.std()) if fin.size else float("nan")
                lcr = longest_constant_run(x)
                flag = ""
                if nan or inf or zero_frac > 1e-4 or lcr > 256:
                    flag = "  <<< SUSPECT"
                    worst.append((f.split('/')[-1], ifo, nan, inf, zero_frac, lcr))
                row.append(f"{ifo}: n={n} nan={nan} inf={inf} zerofrac={zero_frac:.2e} "
                           f"const_run={lcr} std={std:.2e} max={amax:.2e}{flag}")
            print(" | ".join(row), flush=True)
    print("\n==== SUSPECT segments (nan/inf/zeros/long-constant) ====")
    for w in worst:
        print("  ", w)
    if not worst:
        print("  none — background is clean of zeros/gaps/nan/inf")


if __name__ == "__main__":
    bg = sys.argv[1] if len(sys.argv) > 1 else "/n/holystore01/LABS/iaifi_lab/Lab/kyoon/DATA/O3a_H1_L1_4096Hz"
    main(bg)
