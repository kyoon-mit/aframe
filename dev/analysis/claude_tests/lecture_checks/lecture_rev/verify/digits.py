"""Exact last/best/last-ten values of val within 1% for given runs."""
import glob, json, sys
from wandb.proto import wandb_internal_pb2 as pb
from wandb.sdk.internal import datastore
R = "/home/kyoon/SSM-BNS/aframe/dev/runs/claude_tests"
K = "val/within_1pct_chirp_mass"
for run in sys.argv[1:]:
    rows = {}
    for f in sorted(glob.glob(f"{R}/{run}/wandb/run-*/*.wandb")):
        ds = datastore.DataStore(); ds.open_for_scan(f)
        while True:
            try: d = ds.scan_data()
            except Exception: break
            if d is None: break
            r = pb.Record(); r.ParseFromString(d)
            if r.WhichOneof("record_type") != "history": continue
            row = {(i.key or "/".join(i.nested_key)): i.value_json for i in r.history.item}
            if K in row and "epoch" in row:
                rows[int(json.loads(row["epoch"]))] = json.loads(row[K])
    e = sorted(rows)
    print(run[:40], "last", e[-1], f"{100*rows[e[-1]]:.4f}", "mean last10", f"{100*sum(rows[x] for x in e[-10:])/10:.4f}",
          "best", f"{100*max(rows.values()):.4f}")
