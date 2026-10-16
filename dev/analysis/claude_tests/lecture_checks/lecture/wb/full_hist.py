import glob, json, sys
from wandb.sdk.internal import datastore
from wandb.proto import wandb_internal_pb2 as pb
R = "/home/kyoon/SSM-BNS/aframe/dev/runs/claude_tests"
run = sys.argv[1]
for f in sorted(glob.glob(f"{R}/{run}/wandb/run-*/*.wandb")):
    ds = datastore.DataStore(); ds.open_for_scan(f)
    out = []
    while (d := ds.scan_data()) is not None:
        r = pb.Record(); r.ParseFromString(d)
        if r.WhichOneof("record_type") != "history":
            continue
        row = {(i.key or "/".join(i.nested_key)): i.value_json for i in r.history.item}
        if "val/within_1pct_chirp_mass" in row:
            out.append((int(json.loads(row.get("epoch", "-1"))), 100 * json.loads(row["val/within_1pct_chirp_mass"]),
                        100 * json.loads(row.get("val/within_1pct_chirp_mass_snr8-12", "nan")) if "val/within_1pct_chirp_mass_snr8-12" in row else float("nan")))
    print(f.split("/wandb/")[1][:40], len(out))
    print(" ".join(f"{e}:{v:.1f}" for e, v, _ in out))
