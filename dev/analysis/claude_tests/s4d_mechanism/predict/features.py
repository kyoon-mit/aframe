"""Channel features of the four noisy one-layer networks and the 256-template bank.

For every window: ``LongConvEnergyNet.peak_energy`` (decimation, convolution with the unit-norm
kernels, energy 2|y|^2, second detector's maximum within the coincidence window added, maximum over
time) of each extractor, run on the GPU in batches of 32 windows (a short last batch padded with
copies of its own rows), as training and validation ran. The networks' own estimates come from the
same tensors through ``out`` and ``estimate`` (= ``forward`` then ``estimate``).

Stored per set in data/<set>/: pe__<extractor>.npy (n, 256) float32 peak energies, est__<net>.npy
(n,) own-head chirp-mass estimates, labels.npz.

usage: features.py SET [N]   SET in testA, train, train2, es, testB, smoke
"""
import json
import os
import queue
import resource
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gen  # noqa: E402
from gen import harness  # noqa: E402

torch.set_num_threads(8)
NETS = {
    "d64": ("s4d_energy_max_256ch_dstate64_softmax_train_snr8_prob0.96", "ck_266"),
    "d512": ("s4d_energy_max_256ch_dstate512_softmax_train_snr8_prob0.96", "ck_288"),
    "band": ("s4d_energy_max_256ch_dstate512_bandinit_softmax_train_snr8_prob0.96", "ck_122"),
    "free": ("free_kernel_energy_max_256ch_softmax_train_snr8_prob0.96", "ck_248"),
}
B = harness.VALIDATION_BATCH   # 32
POOL_SEED = 31000
SETS = {  # n, seed, background, snr_min
    "train": (64000, 31001, "train", 8.0),
    "es": (10000, 31002, "valid", 4.0),
    "testB": (20000, 31003, "valid", 8.0),
    "smoke": (640, 31009, "valid", 4.0),
    "train2": (64000, 31004, "train", 8.0),
    **{f"train{i}": (64000, 31010 + i, "train", 8.0) for i in range(3, 9)},
    "es8": (10000, 31020, "valid", 8.0),
}
EXACT = ("d64", "d512")   # also run in two halves of 16 windows (32 transforms): no cuFFT DC offset
POOL_SIZES = {"es": 10000, "testB": 20000}       # train (and train2) take the rest


def row_pools():
    """Disjoint waveform-row pools: every row except testA's and the bank's, shuffled once."""
    labels, _ = harness.load_set("noisy")
    bank = np.load(HERE / "data" / "bank256.npz")
    n_wf = 200000
    used = np.zeros(n_wf, bool)
    used[labels["waveform_row"]] = True
    used[bank["rows"]] = True
    free = np.flatnonzero(~used)
    free = np.random.default_rng(POOL_SEED).permutation(free)
    a, b = POOL_SIZES["es"], POOL_SIZES["es"] + POOL_SIZES["testB"]
    pools = {"es": free[:a], "es8": free[:a], "testB": free[a:b], "smoke": free[b:],
             **{k: free[b:] for k in SETS if k.startswith("train")}}
    return pools, int(used.sum())


def extractors(device):
    out = {}
    for tag, (name, ck) in NETS.items():
        out[tag] = harness.load_net(name, ck, device=device).arch
    from train.experimental.long_conv.arch import LongConvEnergyNet

    bank = LongConvEnergyNet(num_ifos=2, channels=256, kernel="free").to(device).eval()
    K = torch.tensor(np.load(HERE / "data" / "bank256.npz")["kernels"]).to(device)
    bank.kernels = lambda: K
    out["bank"] = bank
    return out


@torch.no_grad()
def run_batch(ext, X, device):
    """X (m, 2, 8192) float32, m <= 32: peak energies per extractor and own-head estimates."""
    m = len(X)
    x = torch.from_numpy(X).to(device, torch.float32)
    if m < B:
        x = torch.cat([x] + [x] * (B // m))[:B]
    pe, est = {}, {}
    for tag, arch in ext.items():
        p = arch.peak_energy(x)
        pe[tag] = p[:m].float().cpu().numpy()
        if tag in NETS:
            est[tag] = head(arch, p, m)
        if tag in EXACT:
            p = torch.cat([arch.peak_energy(x[:B // 2]), arch.peak_energy(x[B // 2:])])
            pe[tag + "_exact"] = p[:m].float().cpu().numpy()
            est[tag + "_exact"] = head(arch, p, m)
    return pe, est


@torch.no_grad()
def head(arch, p, m):
    """The network's own read-out of peak energies p: log1p, standardize over channels, out, estimate."""
    z = torch.log1p(p)
    z = (z - z.mean(-1, keepdim=True)) / z.std(-1, keepdim=True).clamp_min(1e-6)
    log_mc, _ = arch.estimate(arch.out(z)[:m].float())
    return log_mc.exp().double().cpu().numpy()


def producer(which, n, q):
    try:
        if which == "testA":
            _, paths = harness.load_set("noisy")
            f32 = Path(str(paths["X"]).replace(".npy", "_f32.npy"))
            for i in range(0, n, 1024):
                q.put((i, harness.read_rows(f32, i, min(n, i + 1024))))
        else:
            pools, _ = row_pools()
            _, seed, bg, snr_min = SETS[which]
            for i, X, d in gen.iter_windows(n, seed, bg=bg, snr_min=snr_min, rows=pools[which], batch_size=64):
                if i == 0:
                    q.put(("labels", d))
                q.put((i, X))
    except Exception as e:  # noqa: BLE001
        q.put(("error", repr(e)))
    q.put(None)


def main(which, n=None):
    device = torch.device("cuda:0")
    out = HERE / "data" / which
    out.mkdir(parents=True, exist_ok=True)
    if which == "testA":
        labels, _ = harness.load_set("noisy")
        n = n or len(labels["chirp_mass"])
    else:
        n = n or SETS[which][0]
        labels = None
    ext = extractors(device)
    tags = list(ext) + [t + "_exact" for t in EXACT]
    pe = {t: np.zeros((n, 256), np.float32) for t in tags}
    est = {t: np.zeros(n, np.float64) for t in list(NETS) + [t + "_exact" for t in EXACT]}
    q = queue.Queue(maxsize=6)
    th = threading.Thread(target=producer, args=(which, n, q), daemon=True)
    th.start()
    t0, done = time.time(), 0
    while True:
        item = q.get()
        if item is None:
            break
        if item[0] == "error":
            raise RuntimeError(item[1])
        if item[0] == "labels":
            labels = item[1]
            continue
        i0, X = item
        for j in range(0, len(X), B):
            p, e = run_batch(ext, X[j:j + B], device)
            for t in pe:
                pe[t][i0 + j:i0 + j + len(p[t])] = p[t]
            for t in est:
                est[t][i0 + j:i0 + j + len(e[t])] = e[t]
        done = i0 + len(X)
        if (i0 // len(X)) % 50 == 0:
            print(f"{which}: {done}/{n} {time.time() - t0:.0f} s rss {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6:.2f} GB "
                  f"gpu {torch.cuda.max_memory_allocated() / 2**30:.2f} GB", flush=True)
    th.join()
    if done != n:
        raise RuntimeError(f"got {done} of {n}")
    for t, a in pe.items():
        np.save(out / f"pe__{t}.npy", a)
    for t, a in est.items():
        np.save(out / f"est__{t}.npy", a)
    keep = {k: np.asarray(v)[:n] if np.ndim(v) and len(v) >= n else np.asarray(v) for k, v in labels.items()}
    np.savez(out / "labels.npz", **keep)
    info = {"set": which, "n": n, "seconds": round(time.time() - t0, 1),
            "maxrss_GB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6,
            "peak_gpu_GB": torch.cuda.max_memory_allocated() / 2**30}
    if which != "testA":
        pools, n_excluded = row_pools()
        info.update(seed=SETS[which][1], background=SETS[which][2], snr_min=SETS[which][3],
                    row_pool_size=len(pools[which]), rows_excluded_testA_and_bank=n_excluded)
    json.dump(info, open(out / "info.json", "w"), indent=1)
    print(json.dumps(info), flush=True)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else None)
