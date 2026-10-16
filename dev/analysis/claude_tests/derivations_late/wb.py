import wandb, pandas as pd, numpy as np
api = wandb.Api(timeout=60); e = api.default_entity
runs = {"with": "keej8hie", "without": "den_reg_sched_alpha_0.5_floor_1e-2_train_snr8_mc_q_lowband_prob0.96"}
keys = ["val/within_1pct_chirp_mass", "val/within_2pct_chirp_mass", "val/rho", "val/gain"]
out = {}
for lab, rid in runs.items():
    r = api.run(f"{e}/DENOISER-SCAN/{rid}")
    h = r.history(samples=100000, keys=["epoch"] + keys).dropna(subset=keys[:1]).groupby("epoch").last()
    h.to_csv(f"hist_{lab}.csv")
    print(lab, r.id, r.name, len(h), h.index.min(), h.index.max())
    print({k: r.config.get(k) for k in list(r.config)[:0]})
