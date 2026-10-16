"""Validation replayed with the pipeline's own loaders and modules, no data module setup.

Background: Hdf5TimeSeriesDataset(valid files, 25 s, batch 32, coincident=False,
num_files_per_batch=1). Waveforms: Hdf5WaveformLoader over the training waveform
file wrapped in a DataLoader and ChunkedWaveformDataset, as train_dataloader builds
them, but with chunks of CHUNK rows (pipeline: 2000) and BATCHES_PER_CHUNK batches
per chunk (pipeline: 63) to stay under 4 GB of host memory. Each batch goes through
DenoiserOnlyAframeDataset.on_after_batch_transfer (slice_waveforms, inject with the
real SignalInverter, SignalReverser, sky distributions and validation SnrSampler,
waveform_prob 0.96, whitening). Networks score each batch as the task does
(per-batch fraction, then the mean over batches).
"""
import json
import sys
import time
import resource
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import harness  # noqa: E402
from test_faithful import pipeline_object  # noqa: E402

from ml4gw.augmentations import SignalInverter, SignalReverser  # noqa: E402
from ml4gw.dataloading import Hdf5TimeSeriesDataset  # noqa: E402
from ml4gw.distributions import Cosine, PowerLaw  # noqa: E402
from train.augmentations import SnrSampler  # noqa: E402
from train.data.waveforms import ChunkedWaveformDataset, Hdf5WaveformLoader  # noqa: E402

CHUNK = 256
BATCHES_PER_CHUNK = 8
NETS = [
    ("s4d_energy_max_256ch_dstate64_softmax_train_snr8_prob0.96", "ck_266"),
    ("s4d_energy_max_256ch_dstate512_softmax_train_snr8_prob0.96", "ck_288"),
    ("s4d_energy_max_256ch_dstate512_bandinit_softmax_train_snr8_prob0.96", "ck_122"),
    ("free_kernel_energy_max_256ch_softmax_train_snr8_prob0.96", "ck_248"),
    ("sig_only_reg_s4d_max_dropout0_64x32x4", "ck_69"),
    ("sig_only_reg_resnet", "ck_37"),
    ("sig_only_reg_free_kernel", "ck_46"),
]


def cross_entropy(net, logits, chirp_mass):
    """``LongConvClassifier._shared_step`` loss: Gaussian target over the log grid."""
    grid = net.arch.grid_log_mc.cpu()
    width = 0.5 * (grid[1] - grid[0])
    log_true = torch.as_tensor(chirp_mass).float().log()
    target = torch.softmax(-((log_true[:, None] - grid[None]) ** 2) / (2 * width**2), -1)
    return float(-(target * torch.log_softmax(logits, -1)).sum(-1).mean())


def main(n_batches, seed, nets=None):
    global NETS
    if nets:
        NETS = [tuple(x.split(":")) for x in nets]
    torch.set_float32_matmul_precision("high")       # as train/cli.py main()
    dev = torch.device("cuda", 0)
    np.random.seed(seed)
    torch.manual_seed(seed)
    obj = pipeline_object()
    obj._hparams["waveform_prob"] = 0.96
    obj.inverter, obj.reverser = SignalInverter(0.5), SignalReverser(0.5)
    obj.dec = Cosine()
    obj.psi = torch.distributions.Uniform(0.0, 3.14159, validate_args=False)
    obj.phi = torch.distributions.Uniform(-3.14159, 3.14159, validate_args=False)
    snr = dict(constant_params={"index": -3, "maximum": 100.0}, end_params={"minimum": 4.0}, decay_steps=1, distribution=PowerLaw)
    obj.val_snr_sampler = SnrSampler(start_params={"minimum": 4.0}, **snr)
    obj.snr_sampler = SnrSampler(start_params={"minimum": 8.0}, **{**snr, "end_params": {"minimum": 8.0}})
    obj.trainer = type("T", (), dict(training=False, validating=True, sanity_checking=False))()
    obj.waveforms_from_disk = True
    obj.signal_repeats = 4
    obj.swapper = obj.muter = None
    obj.swap_prob = obj.mute_prob = 0
    from train.transforms import ChirpMass, MassRatio
    obj.param_transforms = [ChirpMass(), MassRatio()]
    for m in (obj.psd_estimator, obj.whitener, obj.projector):
        m.to(dev)

    strain = Hdf5TimeSeriesDataset(harness.valid_fnames(), channels=harness.IFOS, kernel_size=harness.SAMPLE_SIZE,
                                   batch_size=32, batches_per_epoch=n_batches, coincident=False, num_files_per_batch=1)
    wl = Hdf5WaveformLoader([Path(harness.WAVEFORM_FILE)], channels=["cross", "plus"], batch_size=CHUNK,
                            batches_per_epoch=n_batches // BATCHES_PER_CHUNK + 1, path="waveforms")
    wloader = torch.utils.data.DataLoader(wl, num_workers=0)
    wds = ChunkedWaveformDataset(wloader, batch_size=32, batches_per_chunk=BATCHES_PER_CHUNK)

    nets = [harness.load_net(n, t, device=dev) for n, t in NETS]
    per_batch = {f"{n}:{t}": [] for n, t in NETS}
    ce = {f"{n}:{t}": [] for n, t in NETS}
    pooled = {k: [0, 0] for k in per_batch}
    t0 = time.time()
    for b, (X, (waveforms, params)) in enumerate(zip(strain, wds)):
        if b >= n_batches:
            break
        batch = ([X.to(dev)], (waveforms.to(dev), {k: v.to(dev) for k, v in params.items()}))
        Xn, Xc, y, p = obj.on_after_batch_transfer(batch, 0)
        signal = ~torch.isnan(p["chirp_mass"])
        true = p["chirp_mass"][signal].cpu().numpy()
        for (n, t), net in zip(NETS, nets):
            inp = Xn if net.task == "long_conv" else Xc
            if net.task == "long_conv":
                logits = net.raw(inp[signal])
                log_mc, _ = net.arch.estimate(logits.to(dev))
                est = log_mc.exp().double().cpu().numpy()
                ce[f"{n}:{t}"].append(cross_entropy(net, logits, true))
            else:
                est = net(inp[signal])
            ok = np.abs(est - true) / true < 0.01
            per_batch[f"{n}:{t}"].append(float(ok.mean()))
            pooled[f"{n}:{t}"][0] += int(ok.sum())
            pooled[f"{n}:{t}"][1] += len(ok)
        if b % 64 == 0:
            print(f"batch {b}: {time.time() - t0:.0f} s, maxrss {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6:.2f} GB, "
                  f"gpu {torch.cuda.max_memory_allocated() / 2**30:.2f} GB", flush=True)
    out = {}
    for k in per_batch:
        v = np.array(per_batch[k])
        out[k] = {"batches": len(v), "rows": pooled[k][1], "within_1pct_batch_mean": float(v.mean()),
                  "within_1pct_pooled": pooled[k][0] / pooled[k][1],
                  "se_pooled": float(np.sqrt(v.mean() * (1 - v.mean()) / pooled[k][1])),
                  "loss_ce_batch_mean": float(np.mean(ce[k])) if ce[k] else None,
                  "loss_ce_se": float(np.std(ce[k]) / np.sqrt(len(ce[k]))) if ce[k] else None,
                  "within_1pct_64batch_blocks": [float(v[i:i + 64].mean()) for i in range(0, len(v) - 63, 64)]}
        blocks = np.array(out[k]["within_1pct_64batch_blocks"]) * 100
        extra = f", loss_ce {out[k]['loss_ce_batch_mean']:.4f} +- {out[k]['loss_ce_se']:.4f}" if ce[k] else ""
        print(f"{k}: replay within 1% {100 * out[k]['within_1pct_batch_mean']:.2f}% (batch mean), {100 * out[k]['within_1pct_pooled']:.2f}% pooled over {pooled[k][1]} rows"
              f"{extra}; 64-batch blocks sd {blocks.std(ddof=1):.2f} pts over {len(blocks)}")
    tag = "_".join(sorted({t for _, t in NETS})) if nets else "best"
    json.dump(out, open(HERE / f"replay_seed{seed}_{tag}.json", "w"), indent=1)


if __name__ == "__main__":
    main(int(sys.argv[1]), int(sys.argv[2]), sys.argv[3:])
