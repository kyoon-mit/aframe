"""Validation within 1% per epoch window for the deeper network and the band start (read-only)."""
import glob, json, sys
import numpy as np
from wandb.sdk.internal import datastore
from wandb.proto import wandb_internal_pb2 as pb
R = "/home/kyoon/SSM-BNS/aframe/dev/runs/claude_tests"
K = "val/within_1pct_chirp_mass"
for run in sys.argv[1:]:
    by = {}
    for f in sorted(glob.glob(f"{R}/{run}/wandb/run-*/*.wandb")):
        ds = datastore.DataStore(); ds.open_for_scan(f)
        while (d := ds.scan_data()) is not None:
            r = pb.Record(); r.ParseFromString(d)
            if r.WhichOneof("record_type") != "history":
                continue
            row = {(i.key or "/".join(i.nested_key)): i.value_json for i in r.history.item}
            if K in row:
                by[int(json.loads(row.get("epoch", "-1")))] = 100 * json.loads(row[K])
    e = np.array(sorted(by)); v = np.array([by[i] for i in e])
    print(run[:60], "epochs", e[0], "-", e[-1], "n", len(e))
    for lo in range(0, e[-1] + 1, 20):
        s = (e >= lo) & (e < lo + 20)
        if s.any():
            print(f"  {lo:3d}-{lo+19:3d}: mean {v[s].mean():5.2f}  (n={s.sum()})")
    s = e >= e[-1] - 9
    print(f"  last ten ({e[s][0]}-{e[-1]}): {v[s].mean():.2f}; last {v[-1]:.1f}; best {v.max():.1f} at {e[v.argmax()]}")
    # linear trend over the last 60 epochs
    s = e >= e[-1] - 59
    p = np.polyfit(e[s], v[s], 1)
    resid = v[s] - np.polyval(p, e[s])
    se = resid.std(ddof=2) / np.sqrt(((e[s] - e[s].mean()) ** 2).sum())
    print(f"  slope over last 60 epochs: {100*p[0]:.2f} points per 100 epochs (+- {100*se:.2f})")
