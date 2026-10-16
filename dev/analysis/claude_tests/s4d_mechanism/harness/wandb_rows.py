"""Read logged validation rows from local wandb files (read-only; same logic as warp_table.rows, no cache)."""
import glob
import json
import sys

from wandb.proto import wandb_internal_pb2 as pb
from wandb.sdk.internal import datastore

R = "/home/kyoon/SSM-BNS/aframe/dev/runs/claude_tests"
KEYS = [f"val/within_{p}pct_chirp_mass" for p in (1, 2, 5, 10)]


def scan(f, keys=KEYS):
    out = []
    ds = datastore.DataStore()
    ds.open_for_scan(f)
    while (d := ds.scan_data()) is not None:
        r = pb.Record()
        r.ParseFromString(d)
        if r.WhichOneof("record_type") != "history":
            continue
        row = {(i.key or "/".join(i.nested_key)): i.value_json for i in r.history.item}
        if keys[0] in row:
            out.append({k: json.loads(row[k]) for k in list(keys) + ["epoch", "_step"] if k in row})
    return out


def rows(run, keys=KEYS):
    """Val rows by epoch; a resumed run's later file overrides repeated epochs."""
    by_epoch = {}
    for f in sorted(glob.glob(f"{R}/{run}/wandb/run-*/*.wandb")):
        for r in scan(f, keys):
            by_epoch[int(r.get("epoch", -1))] = r
    return [by_epoch[e] for e in sorted(by_epoch)]


if __name__ == "__main__":
    run = sys.argv[1]
    eps = [int(e) for e in sys.argv[2:]]
    h = rows(run)
    print(run, "epochs logged:", len(h), "last", h[-1].get("epoch") if h else None)
    best = max(h, key=lambda r: r.get(KEYS[0], 0))
    print("best within_1pct:", best)
    for r in h:
        if not eps or int(r.get("epoch", -1)) in eps:
            print(r)
