"""Audit: per-run means of several validation keys over epoch ranges (read-only)."""
import glob, json, sys
from wandb.sdk.internal import datastore
from wandb.proto import wandb_internal_pb2 as pb
R = "/home/kyoon/SSM-BNS/aframe/dev/runs/claude_tests"
for run in sys.argv[1:]:
    by = {}
    keys = set()
    for f in sorted(glob.glob(f"{R}/{run}/wandb/run-*/*.wandb")):
        ds = datastore.DataStore(); ds.open_for_scan(f)
        while (d := ds.scan_data()) is not None:
            r = pb.Record(); r.ParseFromString(d)
            if r.WhichOneof("record_type") != "history":
                continue
            row = {(i.key or "/".join(i.nested_key)): i.value_json for i in r.history.item}
            vk = [k for k in row if k.startswith("val/within")]
            if vk:
                e = int(json.loads(row.get("epoch", "-1")))
                by.setdefault(e, {}).update({k: json.loads(row[k]) for k in vk})
                keys.update(vk)
    eps = sorted(by)
    print(run[:70], "epochs", eps[0], "-", eps[-1], "n", len(eps))
    for k in sorted(keys):
        def m(lo, hi):
            v = [100 * by[e][k] for e in eps if lo <= e <= hi and k in by[e] and isinstance(by[e][k], (int, float))]
            return sum(v) / len(v) if v else float('nan')
        print(f"   {k}: mean200-250 {m(200,250):.1f}  mean20-39 {m(20,39):.1f} mean40-59 {m(40,59):.1f} mean60-76 {m(60,76):.1f} last {100*by[eps[-1]].get(k, float('nan')):.1f}")
