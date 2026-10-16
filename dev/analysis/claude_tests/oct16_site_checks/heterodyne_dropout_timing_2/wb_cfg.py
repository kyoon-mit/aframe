"""Read dropout and key settings of the 23 September denoiser runs from wandb (read-only)."""
import wandb
api = wandb.Api(timeout=30)
P = "kyoon-mit-massachusetts-institute-of-technology/DENOISER-SCAN/"
ids = [
    "premerger_16s_sched_alpha_0.1_floor_1e-3_c_rho_5_lambda_amp_1_uniform_snr",
    "het_premerger_16s_wsched_alpha_0.5_beta_1_floor_1e-1_c_rho_5_lambda_amp_1_uniform_snr",
    "het_prenorm_wsched_alpha_0.5_beta_1_floor_1e-1_c_rho_5_lambda_amp_1_uniform_snr",
    "het_mcsig_1e-3_prenorm_wsched_alpha_0.5_beta_1_floor_1e-1_c_rho_5_lambda_amp_1_uniform_snr",
]
def find(d, key, path=""):
    out = []
    if isinstance(d, dict):
        for k, v in d.items():
            p = f"{path}.{k}" if path else k
            if k == key:
                out.append((p, v))
            out += find(v, key, p)
    elif isinstance(d, list):
        for i, v in enumerate(d):
            out += find(v, key, f"{path}[{i}]")
    return out
for i in ids:
    try:
        r = api.run(P + i)
        c = r.config
        print(i, "| state", r.state, "| dropout", find(c, "dropout"), "| prenorm", find(c, "prenorm"),
              "| alpha", find(c, "alpha")[:1], "| log_floor", find(c, "log_floor")[:1])
        s = r.summary
        ks = [k for k in s.keys() if "rho" in k.lower()][:8]
        print("   summary rho keys:", {k: s[k] for k in ks}, "| epoch", s.get("epoch"))
    except Exception as e:
        print(i, "ERROR", type(e).__name__, str(e)[:200])
