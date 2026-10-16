"""Score a chirp-kernel checkpoint on the matched-filter benchmark windows."""

import sys

import h5py
import numpy as np
import torch

from train.experimental.chirp_kernel.arch import ChirpKernelNet

torch.set_num_threads(60)
ckpt, out = sys.argv[1], sys.argv[2]
B = ("/home/kyoon/SSM-BNS/aframe/dev/runs/claude_tests/den_reg_sched_alpha_0.5_floor_1e-2_"
     "c_rho_5_lambda_amp_3_powerlaw_m2_snr_d128_n6_mc_q_rep4_smooth20_wp0.96_ddp/mf_bench_ep630")

trained = ChirpKernelNet().eval()
state = torch.load(ckpt, map_location="cpu", weights_only=False)["state_dict"]
trained.load_state_dict({k[len("model."):]: v for k, v in state.items() if k.startswith("model.")})
untrained = ChirpKernelNet().eval()

rows = {}
for pop in ("snr4_powerlaw", "snr8_powerlaw", "snr4_uniform"):
    w = h5py.File(f"{B}/{pop}/windows.h5")
    m = h5py.File(f"{B}/{pop}/mf.h5")
    mc = w["params/chirp_mass"][:]
    sig = np.where(~np.isnan(mc))[0]
    names = list(w.attrs["param_names"])
    est_t, est_u, probs = [], [], []
    with torch.no_grad():
        for a in range(0, len(sig), 32):
            x = torch.tensor(w["X"][sig[a:a + 32]])
            lt = trained(x)
            est_t.append(trained.kernels.log_mc[lt.argmax(-1)].exp())
            est_u.append(untrained.kernels.log_mc[untrained(x).argmax(-1)].exp())
            if a < 32 * 8:
                probs.append(torch.softmax(lt, -1))
    rows[pop] = dict(
        true=mc[sig], snr=w["params/snr"][:][sig],
        trained=torch.cat(est_t).numpy(), untrained=torch.cat(est_u).numpy(),
        mf=m["mf_chirp_mass"][:][sig], ml=w["ml_mean"][:, names.index("chirp_mass")][sig],
        probs=torch.cat(probs).numpy(),
    )
    print(pop, "done", flush=True)
np.savez(out, kernel_mc=trained.kernels.log_mc.exp().detach().numpy(),
         **{f"{p}__{k}": v for p, d in rows.items() for k, v in d.items()})

bands = [(4, 6), (6, 8), (8, 10), (10, 12), (12, 16), (16, 25), (25, 50)]
for p, d in rows.items():
    print("==", p)
    for lo, hi in bands:
        k = (d["snr"] >= lo) & (d["snr"] < hi)
        if not k.sum():
            continue
        line = f"  SNR {lo}-{hi} n={k.sum()}"
        for name in ("trained", "untrained", "mf", "ml"):
            e = np.abs(d[name][k] - d["true"][k]) / d["true"][k]
            line += f" | {name} " + "/".join(f"{(e < t).mean():.0%}" for t in (0.01, 0.02, 0.05, 0.10))
        print(line)
