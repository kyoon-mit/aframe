"""Validation within 1% per epoch for the noisy arms, read from local .wandb files (read-only)."""
import glob, json, sys
from wandb.sdk.internal import datastore
from wandb.proto import wandb_internal_pb2 as pb

R = "/home/kyoon/SSM-BNS/aframe/dev/runs/claude_tests"
RUNS = sys.argv[1:]

def scan(f):
    out = []
    ds = datastore.DataStore(); ds.open_for_scan(f)
    while (d := ds.scan_data()) is not None:
        r = pb.Record(); r.ParseFromString(d)
        if r.WhichOneof("record_type") != "history":
            continue
        row = {(i.key or "/".join(i.nested_key)): i.value_json for i in r.history.item}
        keys = [k for k in row if k.startswith("val/") and "within_1" in k]
        if keys:
            out.append({k: json.loads(v) for k, v in row.items() if k.startswith("val/within") or k == "epoch"})
    return out

for run in RUNS:
    by = {}
    for f in sorted(glob.glob(f"{R}/{run}/wandb/run-*/*.wandb")):
        try:
            for r in scan(f):
                by[int(r.get("epoch", -1))] = r
        except Exception as e:
            print("  error", f, e)
    eps = sorted(by)
    if not eps:
        print(run, "no rows"); continue
    k1 = [k for k in by[eps[-1]] if "within_1" in k and "pct" in k]
    print(run, "keys", sorted(by[eps[-1]].keys()))
    vals = [(e, by[e].get(k1[0])) for e in eps] if k1 else []
    print("  epochs", eps[0], "to", eps[-1], "n", len(eps))
    print("  " + " ".join(f"{e}:{100*v:.1f}" for e, v in vals[-12:]))
    best = max(vals, key=lambda t: t[1])
    print(f"  best {100*best[1]:.1f} at epoch {best[0]}")
