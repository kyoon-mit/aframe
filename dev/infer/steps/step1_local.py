"""Step 1 without Triton: raw [mass, sigma] timeseries by direct PyTorch inference.

Reproduces the streaming client offline. Each request is a 128-update
batch at the inference sampling rate, prefixed by the snapshot state
(zeros at the start of a segment, as on the server), whitened by
``BatchWhitener`` and scored by the model. Many requests are stacked per GPU
call. Writes ``<outdir>/branch_<i>/timeseries.hdf5`` in the format step 2
reads.

    python step1_local.py --config <run config.yaml> --ckpt <ckpt> \
        --background-dir <dir> --injection-set <hdf5> --shifts 1 2 ... \
        --outdir <dir> [--exclude-longest] [--requests-per-call 8]
"""

import argparse
import math
import re
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import yaml

from ledger.injections import InterferometerResponseSet, waveform_class_factory
from train.cli import AframeCLI
from utils.preprocessing import BatchWhitener

IFOS = ["H1", "L1"]


def load_model(config, ckpt, device):
    # a saved fit config carries run-only keys that building a model rejects
    cfg = yaml.safe_load(open(config))
    keep = ("seed_everything", "trainer", "model", "data")
    cfg = {k: v for k, v in cfg.items() if k in keep}
    cli = AframeCLI(
        run=False,
        args=[
            "--config", yaml.safe_dump(cfg),
            "--trainer.devices=1", "--trainer.strategy=auto",
            "--trainer.logger=false", "--trainer.callbacks=[]",
        ],
    )
    model = cli.model
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(state["state_dict"])
    return model.eval().to(device), cli.datamodule.hparams


def branches(background_dir, shifts, exclude_longest):
    files = sorted(Path(background_dir).glob("background-*.hdf5"))
    durs = [int(re.search(r"-(\d+)\.hdf5", f.name).group(1)) for f in files]
    if exclude_longest:
        files.pop(int(np.argmax(durs)))
    return [(f, s) for f in files for s in shifts]


@torch.no_grad()
def run_branch(model, whitener, fname, shift, args, hp, device):
    sr = hp.sample_rate
    isr, batch = args.inference_sampling_rate, args.batch_size
    stride = int(sr / isr)
    step = stride * batch
    state = int((hp.kernel_length + hp.fduration + args.psd_length - 1 / isr) * sr)

    with h5py.File(fname, "r") as f:
        size = len(f[IFOS[0]])
        t0 = float(f[IFOS[0]].attrs["x0"])
        shift_samples = [0, int(shift * sr)]
        usable = size - max(shift_samples)
        x = np.stack(
            [f[ifo][s : s + usable] for ifo, s in zip(IFOS, shift_samples)]
        ).astype(np.float32)

    n_req = math.ceil(usable / step)
    num_pad = (step - usable % step) % step
    x = np.pad(x, ((0, 0), (0, num_pad)))

    cls = waveform_class_factory(IFOS, InterferometerResponseSet, "ResponseSet")
    injections = cls.read(
        args.injection_set, start=t0, end=t0 + size / sr, shifts=[0, shift]
    )
    streams = {"background": x}
    if len(injections):
        streams["foreground"] = injections.inject(x.copy(), t0)

    y_mean = model.y_mean.to(device)[0]
    y_std = model.y_std.to(device)[0]
    out = {}
    for name, data in streams.items():
        s = torch.from_numpy(np.pad(data, ((0, 0), (state, 0)))).to(device)
        windows = s.unfold(-1, state + step, step).permute(1, 0, 2)  # (n_req, 2, L)
        scores = []
        for i in range(0, n_req, args.requests_per_call):
            chunk = windows[i : i + args.requests_per_call]
            g = len(chunk)
            if g == 2:
                # BatchWhitener returns one request's kernels for a batch of 2
                kernels = torch.cat([whitener(chunk[j : j + 1]) for j in range(2)])
            else:
                # it interleaves requests (kernel w of every request, then
                # w + 1); restore request-major time order
                kernels = whitener(chunk)
                kernels = kernels.view(batch, g, *kernels.shape[1:])
                kernels = kernels.transpose(0, 1).reshape(-1, *kernels.shape[2:])
            kernels = kernels.float()
            _, raw = model(kernels)
            mean, var = model._split(raw)
            mass = mean[:, 0] * y_std + y_mean
            sigma = var[:, 0].sqrt() * y_std
            scores.append(torch.stack([mass, sigma], 1).cpu())
        ts = torch.cat(scores).numpy()
        cut = num_pad // stride
        out[name] = ts[: len(ts) - cut] if cut else ts

    branch_dir = args.outdir / f"branch_{args.branch_index}"
    branch_dir.mkdir(parents=True, exist_ok=True)
    with h5py.File(branch_dir / "timeseries.hdf5", "w") as f:
        f.create_dataset("background_ts", data=out["background"])
        if "foreground" in out:
            f.create_dataset("foreground_ts", data=out["foreground"])
        f.attrs["t0"] = t0
        f.attrs["duration"] = size / sr
        f.attrs["sample_rate"] = float(sr)
        f.attrs["shifts"] = np.asarray([0.0, float(shift)])
        f.attrs["ifos"] = IFOS
        f.attrs["background_fname"] = str(fname)
        f.attrs["psd_length"] = args.psd_length
        f.attrs["fduration"] = float(hp.fduration)
        f.attrs["inference_sampling_rate"] = float(isr)
        f.attrs["integration_window_length"] = 1.0
        f.attrs["cluster_window_length"] = 0.5
        f.attrs["injection_set_fname"] = str(args.injection_set)
    return len(out["background"]), len(injections)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--background-dir", required=True)
    p.add_argument("--injection-set", required=True)
    p.add_argument("--shifts", type=float, nargs="+", required=True)
    p.add_argument("--outdir", type=Path, required=True)
    p.add_argument("--exclude-longest", action="store_true")
    p.add_argument("--psd-length", dest="psd_length", type=float, default=20.0)
    p.add_argument("--inference-sampling-rate", type=float, default=16.0)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--requests-per-call", type=int, default=8)
    p.add_argument("--only", type=int, nargs="*", help="branch indices to run")
    args = p.parse_args()

    device = "cuda"
    model, hp = load_model(args.config, args.ckpt, device)
    whitener = BatchWhitener(
        kernel_length=hp.kernel_length,
        sample_rate=hp.sample_rate,
        inference_sampling_rate=args.inference_sampling_rate,
        batch_size=args.batch_size,
        fduration=hp.fduration,
        fftlength=hp.fftlength,
        highpass=hp.highpass,
    ).to(device)

    todo = branches(args.background_dir, args.shifts, args.exclude_longest)
    print(f"{len(todo)} branches", flush=True)
    for i, (fname, shift) in enumerate(todo):
        if args.only and i not in args.only:
            continue
        if (args.outdir / f"branch_{i}" / "timeseries.hdf5").exists():
            continue
        args.branch_index = i
        start = time.time()
        n, n_inj = run_branch(model, whitener, fname, shift, args, hp, device)
        print(
            f"branch {i}: {fname.name} shift {shift:g}s, {n} windows, "
            f"{n_inj} injections, {time.time() - start:.0f}s",
            flush=True,
        )


if __name__ == "__main__":
    main()
