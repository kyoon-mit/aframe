"""Read history rows from local .wandb files (read-only). Usage: wbread.py <file.wandb> [key-substring ...]"""
import sys, json
from wandb.sdk.internal import datastore
from wandb.proto import wandb_internal_pb2 as pb

def rows(path):
    ds = datastore.DataStore()
    ds.open_for_scan(path)
    while True:
        try:
            data = ds.scan_data()
        except Exception as e:
            break
        if data is None:
            break
        rec = pb.Record()
        try:
            rec.ParseFromString(data)
        except Exception:
            continue
        if rec.WhichOneof("record_type") == "history":
            d = {}
            for it in rec.history.item:
                key = it.key if it.key else "/".join(it.nested_key)
                try:
                    d[key] = json.loads(it.value_json)
                except Exception:
                    d[key] = it.value_json
            yield d

if __name__ == "__main__":
    path = sys.argv[1]
    subs = sys.argv[2:]
    keys = set()
    out = []
    for d in rows(path):
        keys |= set(d)
        if subs and any(any(s in k for s in subs) for k in d):
            out.append({k: v for k, v in d.items() if k in ("epoch", "trainer/global_step") or any(s in k for s in subs)})
    if not subs:
        print(sorted(keys))
    else:
        for o in out:
            print(json.dumps(o))
