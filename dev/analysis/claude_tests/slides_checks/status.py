import glob, json, os, subprocess, sys
from wandb.sdk.internal import datastore
from wandb.proto import wandb_internal_pb2 as pb
running = subprocess.run(["ps", "-eo", "args"], capture_output=True, text=True).stdout
def last_rows(path):
    ds = datastore.DataStore(); ds.open_for_scan(path)
    rows = []
    while True:
        d = ds.scan_data()
        if d is None: break
        r = pb.Record(); r.ParseFromString(d)
        if r.WhichOneof("record_type") == "history":
            row = {}
            for i in r.history.item:
                k = i.key or "/".join(i.nested_key)
                try: row[k] = json.loads(i.value_json)
                except Exception: pass
            if any(k.startswith("val") for k in row): rows.append(row)
    return rows
out = {}
for run in sorted(d for d in os.listdir(".") if os.path.isdir(d) and glob.glob(f"{d}/wandb/run-*/*.wandb")):
    rows = []
    for f in sorted(glob.glob(f"{run}/wandb/run-*/*.wandb")):
        try: rows += last_rows(f)
        except Exception as e: pass
    if not rows: continue
    keys = sorted({k for r in rows for k in r if "within" in k and k.startswith("val")})
    ep = [r.get("epoch") for r in rows if r.get("epoch") is not None]
    def series(k): return [(r.get("epoch"), r[k]) for r in rows if k in r]
    best = {k: max(v for _, v in series(k)) for k in keys}
    last = {k: series(k)[-1] for k in keys}
    out[run] = {"running": run in running, "last_epoch": ep[-1] if ep else None, "n": len(rows), "best": best, "last": last}
json.dump(out, open(sys.argv[1], "w"), indent=1)
for run, d in out.items():
    k1 = [k for k in d["best"] if "1pct" in k or "within_1" in k]
    print(("RUN " if d["running"] else "    ") + run[:72].ljust(72), "ep", d["last_epoch"], "|", "; ".join(f"{k.replace('val/','')}: last {d['last'][k][1]:.3f} best {d['best'][k]:.3f}" for k in k1[:3]))
