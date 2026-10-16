"""Detection-MLP scores for a finished ``step1_local_bank.py`` output.

Reads every branch's features (snr2 tree) and its newsnr2 and snr2 series,
applies the detection MLP of a TemplateBankNet checkpoint, and writes the
result as a new statistic tree ``<results>/<name>/branch_*/timeseries.hdf5``
with the same attributes, for step 2 with ``--metric score``.

    python step1_bank_mlp_scores.py --results <step1 outdir> --ckpt <ckpt> \
        --hidden 32 [--name mlp]
"""

import argparse
from pathlib import Path

import h5py
import torch

from train.experimental.template_bank.arch import TemplateBankNet


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--hidden", type=int, default=32)
    p.add_argument("--name", default="mlp")
    args = p.parse_args()

    width = len(TemplateBankNet.FEATURES) + 2
    net = torch.nn.Sequential(
        torch.nn.Linear(width, args.hidden), torch.nn.GELU(),
        torch.nn.Linear(args.hidden, args.hidden), torch.nn.GELU(),
        torch.nn.Linear(args.hidden, 1),
    )
    state = torch.load(args.ckpt, map_location="cpu", weights_only=False)["state_dict"]
    prefix = "model.det_net."
    net.load_state_dict({k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)})
    net.eval()

    branches = sorted((args.results / "snr2").glob("branch_*"))
    for branch in branches:
        out_dir = args.results / args.name / branch.name
        out_dir.mkdir(parents=True, exist_ok=True)
        with h5py.File(branch / "timeseries.hdf5", "r") as f, \
                h5py.File(args.results / "newsnr2" / branch.name / "timeseries.hdf5", "r") as g, \
                h5py.File(out_dir / "timeseries.hdf5", "w") as o:
            for stream in ("background", "foreground"):
                if f"{stream}_features" not in f:
                    continue
                feats = torch.tensor(f[f"{stream}_features"][:])
                plain = torch.tensor(f[f"{stream}_ts"][:])
                new = torch.tensor(g[f"{stream}_ts"][:])
                with torch.no_grad():
                    score = net(TemplateBankNet.detection_inputs(feats, new, plain))[:, 0]
                o.create_dataset(f"{stream}_ts", data=score.numpy())
            for key, value in f.attrs.items():
                o.attrs[key] = value
    print(f"{len(branches)} branches -> {args.results / args.name}")


if __name__ == "__main__":
    main()
