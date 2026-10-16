import sys, json
from wandb.sdk.internal import datastore
from wandb.proto import wandb_internal_pb2 as pb
path = sys.argv[1]; keys = sys.argv[2].split(',')
ds = datastore.DataStore(); ds.open_for_scan(path)
rows = []
while True:
    data = ds.scan_data()
    if data is None: break
    rec = pb.Record(); rec.ParseFromString(data)
    if rec.WhichOneof("record_type") == "history":
        d = {}
        for it in rec.history.item:
            k = it.key or "/".join(it.nested_key)
            d[k] = it.value_json
        rows.append(d)
out = []
for d in rows:
    if any(k in d for k in keys):
        out.append({k: (json.loads(d[k]) if k in d else None) for k in ['epoch','trainer/global_step'] + keys})
json.dump(out, open(sys.argv[3], 'w'))
print(len(rows), len(out))
