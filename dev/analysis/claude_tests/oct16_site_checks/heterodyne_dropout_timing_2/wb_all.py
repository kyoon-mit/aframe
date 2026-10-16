"""Dropout of every run in the DENOISER-SCAN wandb project (read-only)."""
import wandb
api = wandb.Api(timeout=60)
def find(d, keys, path=""):
    out = []
    if isinstance(d, dict):
        for k, v in d.items():
            p = f"{path}.{k}" if path else k
            if k in keys:
                out.append((k, v))
            out += find(v, keys, p)
    elif isinstance(d, list):
        for v in d:
            out += find(v, keys, path)
    return out
runs = api.runs("kyoon-mit-massachusetts-institute-of-technology/DENOISER-SCAN", per_page=200)
n = 0
for r in runs:
    n += 1
    d = find(r.config, {"dropout", "denoiser_dropout", "regressor_dropout"})
    rho = r.summary.get("val/rho")
    print(f"{r.name} | {r.created_at[:10]} | dropout {sorted(set(v for _, v in d))} | val/rho {rho if rho is None else round(rho, 4)} | epoch {r.summary.get('epoch')}")
print("runs:", n)
