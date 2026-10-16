"""float16 storage check: estimates from the float16 files vs the float32 files, all windows."""
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import harness  # noqa: E402
from evaluate import NETS, run  # noqa: E402

torch.set_float32_matmul_precision("high")
out = {}
for name, tag, which in NETS:
    labels, paths = harness.load_set(which)
    true = labels["chirp_mass"].astype(np.float64)
    f16 = paths["X"]
    f32 = Path(str(f16).replace(".npy", "_f32.npy"))
    net = harness.load_net(name, tag)
    e16, e32 = run(net, f16, len(true)), run(net, f32, len(true))
    rel = np.abs(e16 - e32) / e32
    s16, s32 = harness.score(e16, true), harness.score(e32, true)
    out[f"{name}:{tag}"] = {
        "n": len(true), "score_f16": s16, "score_f32": s32,
        "max_abs_score_change_pts": max(abs(100 * (s16[k] - s32[k])) for k in s16),
        "frac_estimate_change_gt_1e-3": float(np.mean(rel > 1e-3)), "frac_estimate_change_gt_1e-2": float(np.mean(rel > 1e-2)),
        "median_rel_change": float(np.median(rel)),
    }
    o = out[f"{name}:{tag}"]
    print(f"{name}:{tag} n={len(true)} max |score change| {o['max_abs_score_change_pts']:.3f} pts; estimates moved >0.1%: {100 * o['frac_estimate_change_gt_1e-3']:.2f}%, >1%: {100 * o['frac_estimate_change_gt_1e-2']:.2f}%; median rel change {o['median_rel_change']:.1e}", flush=True)
    del net
    torch.cuda.empty_cache()
# input-level rounding error
for which in ("clean", "noisy"):
    labels, paths = harness.load_set(which)
    a = harness.read_rows(paths["X"], 0, 1000)
    b = harness.read_rows(Path(str(paths["X"]).replace(".npy", "_f32.npy")), 0, 1000)
    err = a - b
    print(f"{which}: float16 rounding error rms {float(np.sqrt((err ** 2).mean())):.2e} vs signal rms {float(np.sqrt((b ** 2).mean())):.3f}; max |x| {float(np.abs(b).max()):.1f}; inf/nan in f16: {int((~np.isfinite(a)).sum())}")
json.dump(out, open(HERE / "f16_check.json", "w"), indent=1)
