"""Same networks and windows, matmul precision 'high' (TF32, as train/cli.py sets) vs 'highest'."""
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import harness  # noqa: E402
from evaluate import NETS, run  # noqa: E402

for name, tag, which in NETS:
    if sys.argv[1:] and f"{name}:{tag}" not in sys.argv[1:]:
        continue
    labels, paths = harness.load_set(which)
    true = labels["chirp_mass"]
    out = {}
    for prec in ("highest", "high"):
        torch.set_float32_matmul_precision(prec)
        net = harness.load_net(name, tag)     # kernel cache is built under this precision
        out[prec] = run(net, paths["X"], len(true))
        del net
        torch.cuda.empty_cache()
    s = {p: harness.score(e, true) for p, e in out.items()}
    rel = np.abs(out["high"] - out["highest"]) / out["highest"]
    print(f"{name}:{tag} n={len(true)} within_1pct highest {100 * s['highest']['within_1pct']:.2f}%  high(TF32) {100 * s['high']['within_1pct']:.2f}%  "
          f"| within_10pct {100 * s['highest']['within_10pct']:.2f} -> {100 * s['high']['within_10pct']:.2f} | median rel change of estimate {np.median(rel):.2e}, "
          f"frac changed >1% {np.mean(rel > 0.01):.3f}", flush=True)
    np.save(harness.DATA_DIR / f"est_{which}__{name}__{tag}__tf32.npy", out["high"])
