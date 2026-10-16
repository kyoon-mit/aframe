"""Print per-epoch validation 'within' metrics from local .wandb files."""
import sys, json
from wandb.sdk.internal import datastore
from wandb.proto import wandb_internal_pb2

def rows(path):
    ds = datastore.DataStore()
    ds.open_for_scan(path)
    while True:
        data = ds.scan_data()
        if data is None:
            break
        rec = wandb_internal_pb2.Record()
        rec.ParseFromString(data)
        if rec.WhichOneof("record_type") == "history":
            d = {it.key or "/".join(it.nested_key): json.loads(it.value_json) for it in rec.history.item}
            yield d

for p in sys.argv[1:]:
    keys = None
    for d in rows(p):
        ks = [k for k in d if "within" in k.lower() and "val" in k.lower()]
        if ks:
            if keys is None:
                keys = sorted(ks)
                print("#", p.split("/")[-1], keys)
            print(d.get("epoch"), " ".join(f"{k.split('/')[-1]}={d[k]:.4f}" for k in sorted(ks) if isinstance(d[k], (int, float))))
