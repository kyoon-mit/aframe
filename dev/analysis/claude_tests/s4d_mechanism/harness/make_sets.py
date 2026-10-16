"""Write the shared window sets: 4,000 clean (seed 1001) and 20,000 noisy (seed 2002).

Arrays are streamed to .npy batch by batch (never all in memory): float16 copies
for sharing and float32 copies for the float16 check. Labels go to .npz.
"""
import os
import resource
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import harness  # noqa: E402

OUT = Path(__file__).resolve().parent / "data"
SETS = {"clean": dict(n=4000, seed=1001, want=("X_clean",)),
        "noisy": dict(n=20000, seed=2002, want=("X", "X_clean"))}


class NpyWriter:
    def __init__(self, path, shape, dtype):
        self.f = open(path, "wb")
        self.dtype = np.dtype(dtype)
        np.lib.format.write_array_header_1_0(
            self.f, {"descr": np.lib.format.dtype_to_descr(self.dtype), "fortran_order": False, "shape": shape})
        self.rows = 0

    def write(self, a):
        self.f.write(np.ascontiguousarray(a, dtype=self.dtype).tobytes())
        self.rows += len(a)

    def close(self):
        self.f.close()


def main(which):
    OUT.mkdir(exist_ok=True)
    for name in which:
        cfg = SETS[name]
        n, seed, want = cfg["n"], cfg["seed"], cfg["want"]
        shape = (n, len(harness.IFOS), harness.WINDOW_SIZE)
        stem = {"X": f"{name}_X", "X_clean": f"{name}_Xclean"} if name == "noisy" else {"X_clean": "clean_X"}
        writers = {}
        for k in want:
            writers[(k, "f16")] = NpyWriter(OUT / f"{stem[k]}.npy", shape, np.float16)
            writers[(k, "f32")] = NpyWriter(OUT / f"{stem[k]}_f32.npy", shape, np.float32)
        t0 = time.time()
        labels = None
        for i, batch in harness.iter_windows(n, seed, batch_size=64, want=want):
            for k in want:
                writers[(k, "f16")].write(batch[k])
                writers[(k, "f32")].write(batch[k])
            labels = batch["labels"]
            if i % 2560 == 0:
                print(f"{name}: {i + len(batch[want[0]])}/{n} windows, {time.time() - t0:.0f} s, "
                      f"maxrss {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6:.2f} GB", flush=True)
        for w in writers.values():
            assert w.rows == n, (w.rows, n)
            w.close()
        labels = dict(labels)
        labels["seed"] = np.array(seed)
        np.savez(OUT / f"{name}_labels.npz", **labels)
        print(f"{name}: done in {time.time() - t0:.0f} s; maxrss {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6:.2f} GB")


if __name__ == "__main__":
    main(sys.argv[1:] or ["clean", "noisy"])
