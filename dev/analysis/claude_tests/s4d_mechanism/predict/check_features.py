"""Checks of the stored test-A features and own-head estimates.

1. The harness's own network call (``harness.load_net(...)(X)``, batches of 32) on the first 2,048
   float32 windows gives the same estimates as est__<net>.npy (exact equality counted).
2. Own head recomputed from the stored peak energies (pe__<net>.npy) on the GPU equals est__<net>.npy.
3. Within-1% on all 20,000 windows against the harness's saved estimates (computed from the float16
   copy), with decision flips; exact-FFT variants against the harness's exact-FFT estimates.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import features as F  # noqa: E402
from gen import harness  # noqa: E402

D = HERE / "data" / "testA"
labels, paths = harness.load_set("noisy")
true = labels["chirp_mass"]
X = harness.read_rows(Path(str(paths["X"]).replace(".npy", "_f32.npy")), 0, 2048)
dev = torch.device("cuda:0")
out = {}
for tag, (name, ck) in F.NETS.items():
    net = harness.load_net(name, ck, device=dev)
    mine = np.load(D / f"est__{tag}.npy")
    theirs = net(X)
    r = {"harness_call_equal_first2048": int((theirs == mine[:2048]).sum()),
         "harness_call_max_rel_diff_first2048": float(np.max(np.abs(theirs - mine[:2048]) / mine[:2048]))}
    pe = torch.tensor(np.load(D / f"pe__{tag}.npy"))
    rec = np.concatenate([F.head(net.arch, pe[i:i + 32].to(dev), len(pe[i:i + 32])) for i in range(0, len(pe), 32)])
    r["recomputed_from_stored_pe_equal"] = int((rec == mine).sum())
    r["recomputed_from_stored_pe_max_rel_diff"] = float(np.max(np.abs(rec - mine) / mine))
    h16 = np.load(harness.DATA_DIR / f"est_noisy__{name}__{ck}.npy")
    w_mine = np.abs(mine - true) / true < 0.01
    w_h16 = np.abs(h16 - true) / true < 0.01
    r["within1_mine_f32"] = float(w_mine.mean())
    r["within1_harness_f16"] = float(w_h16.mean())
    r["within1_flips_vs_f16"] = int((w_mine != w_h16).sum())
    if tag in F.EXACT:
        mx = np.load(D / f"est__{tag}_exact.npy")
        hx = np.load(harness.DATA_DIR / f"est_noisy__{name}__{ck}__exactfft.npy")
        r["exact_within1_mine_f32"] = float((np.abs(mx - true) / true < 0.01).mean())
        r["exact_within1_harness_f16"] = float((np.abs(hx - true) / true < 0.01).mean())
        r["exact_within1_flips_vs_f16"] = int(((np.abs(mx - true) / true < 0.01) != (np.abs(hx - true) / true < 0.01)).sum())
    out[tag] = r
    print(tag, json.dumps(r), flush=True)
    del net
    torch.cuda.empty_cache()
json.dump(out, open(HERE / "logs" / "check_features_testA.json", "w"), indent=1)
