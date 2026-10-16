"""Extract per-epoch val metrics containing 'within' from local .wandb files (read-only). Usage: wbval.py out.json file1.wandb [file2.wandb ...]"""
import sys, json
from wandb.sdk.internal import datastore
from wandb.proto import wandb_internal_pb2 as pb
out = {}
for path in sys.argv[2:]:
    ds = datastore.DataStore()
    ds.open_for_scan(path)
    while True:
        try:
            data = ds.scan_data()
        except Exception:
            break
        if data is None:
            break
        rec = pb.Record()
        try:
            rec.ParseFromString(data)
        except Exception:
            continue
        if rec.WhichOneof("record_type") != "history":
            continue
        d = {}
        for it in rec.history.item:
            key = it.key if it.key else "/".join(it.nested_key)
            if key == "epoch" or ("val" in key and "within" in key):
                try:
                    d[key] = json.loads(it.value_json)
                except Exception:
                    pass
        if len(d) > 1 and "epoch" in d:
            out.setdefault(str(d["epoch"]), {}).update({k: v for k, v in d.items() if k != "epoch"})
json.dump(out, open(sys.argv[1], "w"))
print(sys.argv[1], "epochs", len(out))
