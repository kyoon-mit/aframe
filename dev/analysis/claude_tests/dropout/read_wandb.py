"""Validation and training within-1% by epoch for the four runs, from local wandb files.

Validation rows come from warp_table.rows() (its definitions only; cache redirected
to a copy here). Training-mode within 1% and learning rate come from a scan of the
same files. Writes curves.json next to this script.
"""
import glob
import json
import os
import time

from wandb.proto import wandb_internal_pb2 as pb
from wandb.sdk.internal import datastore

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = "/home/kyoon/SSM-BNS/aframe/dev/runs/claude_tests/warp_table.py"
R = "/home/kyoon/SSM-BNS/aframe/dev/runs/claude_tests"
RUNS = {
    "max_p02": "sig_only_reg_s4d_max_64x32x4",
    "max_p0": "sig_only_reg_s4d_max_dropout0_64x32x4",
    "mean_p02": "sig_only_reg_s4d_64x32x4",
    "energy_max_p02": "sig_only_reg_s4d_complex_energy_max_64x32x4",
}
EXTRA = ["train/within_1pct_chirp_mass", "train/gaussnll", "val/gaussnll"]

src = open(SRC).read().split("\nmode = sys.argv")[0]
src = src.replace('CACHE = f"{R}/queue/warp_table_cache.json"',
                  f'CACHE = "{HERE}/warp_table_cache_copy.json"')
ns = {}
exec(compile(src, SRC, "exec"), ns)
rows = ns["rows"]


def scan_extra(f):
    """Rows with an epoch and any EXTRA key, from one .wandb file."""
    out = []
    ds = datastore.DataStore()
    ds.open_for_scan(f)
    while True:
        try:
            d = ds.scan_data()
        except Exception:
            break
        if d is None:
            break
        r = pb.Record()
        r.ParseFromString(d)
        if r.WhichOneof("record_type") != "history":
            continue
        row = {(i.key or "/".join(i.nested_key)): i.value_json for i in r.history.item}
        keys = [k for k in EXTRA + ["lr-AdamW", "lr-AdamW/pg1"] if k in row]
        if keys and "epoch" in row:
            out.append({k: json.loads(row[k]) for k in keys + ["epoch"]})
    return out


def extra(run):
    by_epoch = {}
    for f in sorted(glob.glob(f"{R}/{run}/wandb/run-*/*.wandb")):
        for r in scan_extra(f):
            e = int(r["epoch"])
            by_epoch.setdefault(e, {}).update(r)
    return by_epoch


out = {"read_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "runs": {}}
for tag, run in RUNS.items():
    val = rows(run)
    ex = extra(run)
    out["runs"][tag] = {
        "run": run,
        "epoch": [int(r["epoch"]) for r in val],
        "val_within_1": [r["val/within_1pct_chirp_mass"] for r in val],
        "val_within_2": [r.get("val/within_2pct_chirp_mass") for r in val],
        "train_within_1": {str(e): v["train/within_1pct_chirp_mass"] for e, v in ex.items()
                           if "train/within_1pct_chirp_mass" in v},
        "lr": {str(e): v.get("lr-AdamW") for e, v in ex.items() if "lr-AdamW" in v},
    }
json.dump(ns["cache"], open(ns["CACHE"], "w"))
json.dump(out, open(f"{HERE}/curves.json", "w"), indent=1)
print("read at", out["read_at"])
for tag, d in out["runs"].items():
    ep, v = d["epoch"], d["val_within_1"]
    print(f"{tag:15s} {d['run']}: epochs {ep[0]}..{ep[-1]} ({len(ep)} rows)")
    print("  val within 1% ep0-15:", " ".join(f"{100*x:.1f}" for x in v[:16]))
    tr = d["train_within_1"]
    print("  train within 1% ep0-15:", " ".join(f"{100*tr[str(e)]:.1f}" if str(e) in tr else "-" for e in range(16)))
