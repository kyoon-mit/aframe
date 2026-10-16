"""Validation within 1% (all SNR, and SNR 8-12) per epoch for the noisy arms; read-only."""
import glob, json, sys
from wandb.sdk.internal import datastore
from wandb.proto import wandb_internal_pb2 as pb
R = "/home/kyoon/SSM-BNS/aframe/dev/runs/claude_tests"
K, K812 = "val/within_1pct_chirp_mass", "val/within_1pct_chirp_mass_snr8-12"
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
                by[int(json.loads(row.get("epoch", "-1")))] = (100 * json.loads(row[K]), 100 * json.loads(row[K812]) if K812 in row else float("nan"))
    eps = sorted(by)
    a = [by[e][0] for e in eps]
    best = max(range(len(a)), key=a.__getitem__)
    m = lambda lo, hi: sum(by[e][0] for e in eps if lo <= e <= hi) / max(1, sum(1 for e in eps if lo <= e <= hi))
    print(f"{run[:60]}: epochs {eps[0]}-{eps[-1]}; last {a[-1]:.1f} (8-12: {by[eps[-1]][1]:.1f}); best {a[best]:.1f} at {eps[best]}; "
          f"mean 200-250 {m(200,250):.1f}; mean last 10 {sum(a[-10:])/len(a[-10:]):.1f}; mean 60-76 {m(60,76):.1f}; mean 100-122 {m(100,122):.1f}")
