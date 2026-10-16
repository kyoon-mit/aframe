"""Calibration table: logged validation vs harness (saved sets) vs replay (pipeline loaders)."""
import json
import sys
from pathlib import Path

import numpy as np

H = Path(__file__).resolve().parent
sys.path.insert(0, str(H))
import wandb_rows  # noqa: E402

cal = json.load(open(H / "calibration.json"))
reps = [json.load(open(p)) for p in sorted(H.glob("replay_seed*.json"))]
table = {}
for k, c in cal.items():
    name = k.split(":")[0]
    ep = c["epoch"]
    h = wandb_rows.rows(name)
    e = np.array([int(r["epoch"]) for r in h])
    m = (e >= ep - 10) & (e <= ep + 10) & (e != ep)
    win = [100 * float(np.mean([r[f"val/within_{p}pct_chirp_mass"] for r, mm in zip(h, m) if mm])) for p in (1, 2, 5, 10)]
    lg = [100 * c["logged_at_epoch"][f"within_{p}pct_chirp_mass"] for p in (1, 2, 5, 10)]
    h4 = [100 * c["harness_first4000_f16"][f"within_{p}pct"] for p in (1, 2, 5, 10)]
    ha = [100 * c["harness_all_f16"][f"within_{p}pct"] for p in (1, 2, 5, 10)]
    rr = [(r[k]["within_1pct_pooled"], r[k]["rows"]) for r in reps if k in r]
    rep = 100 * float(np.average([x for x, _ in rr], weights=[w for _, w in rr])) if rr else None
    nrep = int(sum(w for _, w in rr))
    se = 100 * float(np.sqrt(rep / 100 * (1 - rep / 100) / nrep)) if rr else None
    table[k] = dict(epoch=ep, set=c["set"], n=c["n"], logged=lg, logged_pm10_mean=win, harness_first4000=h4, harness_all=ha,
                    diff1_harness4000=h4[0] - lg[0], replay_within1=rep, replay_rows=nrep, replay_se=se,
                    replay_each=[100 * x for x, _ in rr])
    run = name.replace("_softmax_train_snr8_prob0.96", "")
    print(f"{run:44s} {k.split(':')[1]:6s} ep{ep:>3} | logged {lg[0]:6.2f} | harness 4000 {h4[0]:6.2f} ({h4[0] - lg[0]:+.2f}) | harness {c['n']} {ha[0]:6.2f} | "
          f"replay {rep if rep is None else round(rep, 2)} +- {se if se is None else round(se, 2)} (n={nrep}) | logged +-10 ep {win[0]:6.2f}")
    print(f"{'':52s} 2/5/10%: logged {np.round(lg[1:], 2).tolist()} | +-10 ep {np.round(win[1:], 2).tolist()} | harness {c['n']} {np.round(ha[1:], 2).tolist()}")
json.dump(table, open(H / "calibration_table.json", "w"), indent=1)
