"""Check gen.py against the harness's saved noisy set (seed 2002, n 20000): same draws, same windows.
Also time window generation on CPU, both background splits."""
import json
import os
import resource
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gen  # noqa: E402
from gen import harness  # noqa: E402

torch.set_num_threads(8)
out = {}
labels, paths = harness.load_set("noisy")
ref = harness.read_rows(Path(str(paths["X"]).replace(".npy", "_f32.npy")), 0, 128)
t0 = time.time()
got = []
for i, X, d in gen.iter_windows(20000, 2002, bg="valid", snr_min=4.0, rows=None, batch_size=64):
    got.append(X)
    if i >= 64:
        break
got = np.concatenate(got)[:128]
out["max_abs_diff_first128_vs_saved_f32"] = float(np.abs(got - ref).max())
for k in ("chirp_mass", "snr", "merger_sample", "waveform_row"):
    out[f"labels_equal_{k}"] = bool(np.array_equal(np.asarray(d[k]), np.asarray(labels[k])))
# speed, both splits
for bg in ("valid", "train"):
    t0 = time.time()
    n = 0
    for i, X, d in gen.iter_windows(640, 99, bg=bg, snr_min=8.0, batch_size=64):
        n += len(X)
    dt = time.time() - t0
    out[f"windows_per_s_{bg}"] = n / dt
    out[f"snr_min_max_{bg}"] = [float(d["snr"].min()), float(d["snr"].max())]
    out[f"n_bg_files_{bg}"] = len(d["bg_files"])
out["maxrss_GB"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
print(json.dumps(out, indent=1))
json.dump(out, open(HERE / "logs" / "check_gen.json", "w"), indent=1)
