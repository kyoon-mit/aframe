"""Convert a BNSReg LitModelS4DGaussianNLLLoss checkpoint to an aframe LitS4DGaussianNLL checkpoint.

BNSReg uses torch.compile, so state_dict keys are prefixed with 'model._orig_mod.'
Aframe expects 'model.*'. This script strips that prefix and rewraps the weights.

Usage
-----
    uv run python convert_bnsreg_checkpoint.py \
        --input  /path/to/bnsreg.ckpt \
        --output /path/to/aframe_converted.ckpt
"""

import argparse
from pathlib import Path

import lightning
import torch

from train.model.regression import LitS4DGaussianNLL


def convert(input_path: str, output_path: str) -> None:
    src = torch.load(input_path, map_location="cpu", weights_only=False)

    hp = src["hyper_parameters"]
    cfg = hp["model_cfg"]

    model = LitS4DGaussianNLL(
        d_input=cfg["d_input"],
        d_output=cfg["d_output"],
        d_model=cfg["d_model"],
        d_state=cfg["d_state"],
        n_layers=cfg["n_layers"],
        dropout=cfg["dropout"],
        dt_min=cfg.get("dt_min", 1e-3),
        dt_max=cfg.get("dt_max", 1.0),
        lr=cfg.get("lr"),
        base_lr=hp.get("base_lr", 1e-4),
        weight_decay=hp.get("weight_decay", 0.0),
        beta_nll=hp.get("beta_nll", 0.5),
        lambda_spread=hp.get("lambda_spread", 0.0),
    )

    # BNSReg uses torch.compile → keys are 'model._orig_mod.*'
    # Aframe expects                             'model.*'
    src_sd = src["state_dict"]
    dst_sd = {}
    for k, v in src_sd.items():
        new_key = k.replace("model._orig_mod.", "model.", 1)
        dst_sd[new_key] = v

    missing, unexpected = model.load_state_dict(dst_sd, strict=False)
    if missing:
        print(f"Missing keys  ({len(missing)}): {missing}")
    if unexpected:
        print(f"Unexpected keys ({len(unexpected)}): {unexpected}")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "hyper_parameters": dict(model.hparams),
        "pytorch-lightning_version": lightning.__version__,
    }, output_path)
    print(f"Saved converted checkpoint → {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="BNSReg .ckpt path")
    parser.add_argument("--output", required=True, help="Output aframe .ckpt path")
    args = parser.parse_args()
    convert(args.input, args.output)
