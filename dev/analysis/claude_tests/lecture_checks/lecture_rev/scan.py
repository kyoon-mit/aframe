"""Best and last val within 1% per run from local wandb files."""
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
    if not rows: print(run, "none"); continue
    eb = max(rows, key=rows.get); el = max(rows)
    print(f"{run}: n={len(rows)} best={100*rows[eb]:.2f}% @ep{eb}; last={100*rows[el]:.2f}% @ep{el}")
