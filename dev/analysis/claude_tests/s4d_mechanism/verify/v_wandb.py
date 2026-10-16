"""Read logged val/within_1pct per epoch from local wandb files (read-only) and check the 'selected maximum' explanation."""
import glob, json, sys
import numpy as np
from wandb.proto import wandb_internal_pb2 as pb
from wandb.sdk.internal import datastore
R = "/home/kyoon/SSM-BNS/aframe/dev/runs/claude_tests"
K = "val/within_1pct_chirp_mass"
def rows(run):
    by = {}
    for f in sorted(glob.glob(f"{R}/{run}/wandb/run-*/*.wandb")):
        ds = datastore.DataStore(); ds.open_for_scan(f)
        while (d := ds.scan_data()) is not None:
            r = pb.Record(); r.ParseFromString(d)
            if r.WhichOneof("record_type") != "history": continue
            row = {(i.key or "/".join(i.nested_key)): i.value_json for i in r.history.item}
            if K in row and "epoch" in row:
                by[int(json.loads(row["epoch"]))] = 100 * float(json.loads(row[K]))
    return by
out = {}
for run, ep in [("s4d_energy_max_256ch_dstate64_softmax_train_snr8_prob0.96", 266),
                ("s4d_energy_max_256ch_dstate512_softmax_train_snr8_prob0.96", 288),
                ("s4d_energy_max_256ch_dstate512_bandinit_softmax_train_snr8_prob0.96", 122),
                ("free_kernel_energy_max_256ch_softmax_train_snr8_prob0.96", 248),
                ("sig_only_reg_s4d_max_dropout0_64x32x4", 69), ("sig_only_reg_s4d_max_dropout0_64x32x4", 81)]:
    by = rows(run)
    e = np.array(sorted(by)); v = np.array([by[k] for k in e])
    best_ep = int(e[np.argmax(v)])
    win = v[(e >= ep - 10) & (e <= ep + 10)]
    # residual sd around a centred 21-epoch running median (late half of training)
    late = e >= e.max() // 2
    med = np.array([np.median(v[(e >= k - 10) & (e <= k + 10)]) for k in e])
    sd = float(np.std((v - med)[late]))
    out[f"{run}:{ep}"] = dict(n_epochs=len(e), last_epoch=int(e.max()), value_at_ep=by.get(ep), best_epoch_now=best_ep, best_now=float(v.max()),
                              pm10_mean=float(win.mean()), resid_sd_late=sd, z_of_ep=(by.get(ep, np.nan) - float(win.mean())) / sd)
    print(run, ep, out[f"{run}:{ep}"])
json.dump(out, open("v_wandb.json", "w"), indent=1)
