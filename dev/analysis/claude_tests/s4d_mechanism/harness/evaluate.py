"""Run the seven networks on the saved sets, check float16 storage, compare with logged validation."""
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import harness  # noqa: E402
import wandb_rows  # noqa: E402

NETS = [  # name, checkpoint tag, set
    ("sig_only_reg_s4d_max_dropout0_64x32x4", "ck_69", "clean"),
    ("sig_only_reg_resnet", "ck_37", "clean"),
    ("sig_only_reg_free_kernel", "ck_46", "clean"),
    # newer bests of the two clean runs still training (copied 21:59)
    ("sig_only_reg_s4d_max_dropout0_64x32x4", "ck_81", "clean"),
    ("sig_only_reg_resnet", "ck_47", "clean"),
    ("s4d_energy_max_256ch_dstate64_softmax_train_snr8_prob0.96", "ck_266", "noisy"),
    ("s4d_energy_max_256ch_dstate512_softmax_train_snr8_prob0.96", "ck_288", "noisy"),
    ("s4d_energy_max_256ch_dstate512_bandinit_softmax_train_snr8_prob0.96", "ck_122", "noisy"),
    ("free_kernel_energy_max_256ch_softmax_train_snr8_prob0.96", "ck_248", "noisy"),
    # last epochs: not chosen by their validation value, so no selection bias
    ("s4d_energy_max_256ch_dstate64_softmax_train_snr8_prob0.96", "last", "noisy"),
    ("s4d_energy_max_256ch_dstate512_softmax_train_snr8_prob0.96", "last", "noisy"),
    ("free_kernel_energy_max_256ch_softmax_train_snr8_prob0.96", "last", "noisy"),
]
KEYS = [f"val/within_{p}pct_chirp_mass" for p in (1, 2, 5, 10)]
CHUNK = 1000


def run(net, path, n):
    out = []
    for i in range(0, n, CHUNK):
        out.append(net(harness.read_rows(path, i, i + CHUNK)))
    return np.concatenate(out)


def main(only=None):
    results = {}
    for name, tag, which in NETS:
        if only and f"{name}:{tag}" not in only:
            continue
        labels, paths = harness.load_set(which)
        true = labels["chirp_mass"]
        n = len(true)
        f16 = paths["X"]
        f32 = Path(str(f16).replace(".npy", "_f32.npy"))
        t0 = time.time()
        net = harness.load_net(name, tag)
        est16 = run(net, f16, n)
        est32 = run(net, f32, n)
        np.save(harness.DATA_DIR / f"est_{which}__{name}__{tag}.npy", est16)
        err16 = np.abs(est16 - true) / true
        err32 = np.abs(est32 - true) / true
        flips = {f"{t}pct": int(((err16 < t / 100) != (err32 < t / 100)).sum()) for t in (1, 2, 5, 10)}
        rel = float(np.max(np.abs(est16 - est32) / est32))
        epoch = int(net.epoch)
        hist = wandb_rows.rows(name)
        by_ep = {int(r["epoch"]): r for r in hist}
        logged = {k.split("/")[1]: by_ep[epoch][k] for k in KEYS}
        near = [by_ep[e][KEYS[0]] for e in range(epoch - 10, epoch + 11) if e != epoch and e in by_ep]
        res = {
            "checkpoint": net.ckpt, "epoch": net.epoch, "set": which, "n": n,
            "harness_first4000_f16": harness.score(est16[:4000], true[:4000]),
            "harness_first4000_f32": harness.score(est32[:4000], true[:4000]),
            "harness_all_f16": harness.score(est16, true),
            "logged_at_epoch": logged,
            "logged_within_1pct_epochs_pm10_mean": float(np.mean(near)), "logged_pm10_n": len(near),
            "logged_within_1pct_epochs_pm10_range": [float(min(near)), float(max(near))],
            "f16_vs_f32_max_rel_diff_estimate": rel, "f16_vs_f32_decision_flips": flips,
            "seconds": round(time.time() - t0, 1),
            "maxrss_GB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6,
            "peak_gpu_GB": torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0,
        }
        results[f"{name}:{tag}"] = res
        p, l1 = res["harness_first4000_f16"]["within_1pct"], logged["within_1pct_chirp_mass"]
        print(f"{name}: harness(4000) {100 * p:.2f}%  logged ep{epoch} {100 * l1:.2f}%  diff {100 * (p - l1):+.2f} pts  "
              f"| all {n}: {100 * res['harness_all_f16']['within_1pct']:.2f}%  | logged +-10 ep mean {100 * res['logged_within_1pct_epochs_pm10_mean']:.2f}%  "
              f"| f16 flips {flips} max rel {rel:.2e} | {res['seconds']} s rss {res['maxrss_GB']:.2f} GB gpu {res['peak_gpu_GB']:.2f} GB", flush=True)
        del net
        torch.cuda.empty_cache()
    out = HERE / "calibration.json"
    old = json.load(open(out)) if out.exists() else {}
    old.update(results)
    json.dump(old, open(out, "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1:])
