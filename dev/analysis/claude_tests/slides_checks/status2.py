import glob, json, subprocess, sys
from wandb.sdk.internal import datastore
from wandb.proto import wandb_internal_pb2 as pb
RUNS = sys.argv[2:]
running = subprocess.run(["ps", "-eo", "args"], capture_output=True, text=True).stdout
res = {}
for run in RUNS:
    files = sorted(glob.glob(f"{run}/wandb/run-*/*.wandb"))
    rows = []
    for f in files[-2:]:
        ds = datastore.DataStore(); ds.open_for_scan(f)
        while True:
            d = ds.scan_data()
            if d is None: break
            r = pb.Record(); r.ParseFromString(d)
            if r.WhichOneof("record_type") != "history": continue
            row = {}
            for i in r.history.item:
                k = i.key or "/".join(i.nested_key)
                if k == "epoch" or ("within" in k and k.startswith("val")):
                    try: row[k] = json.loads(i.value_json)
                    except Exception: pass
            if any("within" in k for k in row): rows.append(row)
    keys = sorted({k for r in rows for k in r if "within" in k})
    res[run] = {"running": run in running, "epoch": rows[-1].get("epoch") if rows else None,
                "last": {k: rows[-1].get(k) for k in keys} if rows else {},
                "best": {k: max(r[k] for r in rows if k in r) for k in keys}}
    k1 = [k for k in keys if "1pct" in k]
    print(("RUN " if res[run]["running"] else "    ") + run[:70].ljust(70), "ep", res[run]["epoch"], {k.replace("val/", ""): (round(res[run]["last"][k] or 0, 3), round(res[run]["best"][k], 3)) for k in k1})
    sys.stdout.flush()
json.dump(res, open(sys.argv[1], "w"), indent=1)
