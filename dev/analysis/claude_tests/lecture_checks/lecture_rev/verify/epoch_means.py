"""Per-epoch validation metrics from local wandb files: key list, means over epoch ranges, last/best."""
import glob, json, sys
from wandb.proto import wandb_internal_pb2 as pb
from wandb.sdk.internal import datastore
R = "/home/kyoon/SSM-BNS/aframe/dev/runs/claude_tests"
K1 = "val/within_1pct_chirp_mass"
RANGES = ((20, 39), (40, 59), (60, 76), (100, 122), (200, 250))
for run in sys.argv[1:]:
    rows, keys = {}, set()
    for f in sorted(glob.glob(f"{R}/{run}/wandb/run-*/*.wandb")):
        ds = datastore.DataStore(); ds.open_for_scan(f)
        while True:
            try:
                d = ds.scan_data()
            except Exception:
                break
            if d is None:
                break
            r = pb.Record(); r.ParseFromString(d)
            if r.WhichOneof("record_type") != "history":
                continue
            row = {(i.key or "/".join(i.nested_key)): i.value_json for i in r.history.item}
            if K1 in row and "epoch" in row:
                keys |= {k for k in row if k.startswith("val/")}
                rows[int(json.loads(row["epoch"]))] = {k: json.loads(v) for k, v in row.items() if k.startswith("val/")}
    print(f"== {run}: epochs {min(rows)}-{max(rows)}, n={len(rows)}")
    print("   val keys:", sorted(k for k in keys if "pct" in k))
    for tol in (1, 2, 5, 10):
        k = f"val/within_{tol}pct_chirp_mass"
        out = []
        for lo, hi in RANGES:
            v = [rows[e][k] for e in rows if lo <= e <= hi and k in rows[e]]
            out.append(f"{lo}-{hi}: {100*sum(v)/len(v):.2f} (n={len(v)})" if v else f"{lo}-{hi}: -")
        print(f"   within {tol}%: " + "; ".join(out))
    eps = sorted(rows)
    v1 = {e: rows[e][K1] for e in eps}
    eb = max(v1, key=v1.get)
    last10 = eps[-10:]
    print(f"   within 1%: last {100*v1[eps[-1]]:.2f} @ep{eps[-1]}; best {100*v1[eb]:.2f} @ep{eb}; mean last 10 {100*sum(v1[e] for e in last10)/10:.2f}")
