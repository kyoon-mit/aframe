"""Hours to reach 80% and 90% of the best val within-1% (from local wandb files)."""
import glob, json, sys
from wandb.sdk.internal import datastore
from wandb.proto import wandb_internal_pb2 as pb
R = "/home/kyoon/SSM-BNS/aframe/dev/runs/claude_tests"
K = "val/within_1pct_chirp_mass"
for run in sys.argv[1:]:
    rows = []
    t_off = 0.0
    for f in sorted(glob.glob(f"{R}/{run}/wandb/run-*/*.wandb")):
        ds = datastore.DataStore(); ds.open_for_scan(f)
        last_rt = 0.0
        while (d := ds.scan_data()) is not None:
            r = pb.Record(); r.ParseFromString(d)
            if r.WhichOneof("record_type") != "history":
                continue
            row = {(i.key or "/".join(i.nested_key)): i.value_json for i in r.history.item}
            rt = json.loads(row.get("_runtime", "0"))
            last_rt = max(last_rt, rt)
            if K in row:
                rows.append((t_off + rt, int(json.loads(row.get("epoch", "-1"))), json.loads(row[K])))
        t_off += last_rt
    if not rows:
        print(run, "no rows"); continue
    best = max(v for _, _, v in rows)
    out = [f"{run}: best {100*best:.1f}% (ep {max(rows, key=lambda x: x[2])[1]}), last ep {rows[-1][1]} at {rows[-1][0]/3600:.1f} h"]
    for frac in (0.8, 0.9):
        hit = next((r for r in rows if r[2] >= frac * best), None)
        out.append(f"{int(frac*100)}% of best at ep {hit[1]}, {hit[0]/3600:.1f} h")
    print("; ".join(out))
