"""
Convert a GaussianNLLRegressionAframe .ckpt checkpoint to a TorchScript .pt
file suitable for Triton inference (pytorch_libtorch backend).

The wrapper exposes model.score(X) -- negative mean predicted variance -- as a
single (batch_size, 1) output, which is what the Triton ensemble expects.

Usage:
    python scripts/ckpt_to_pt.py best.ckpt model.pt

If torch.jit.script fails (S4D uses complex FFT ops that sometimes break
scripting), the script falls back to torch.jit.trace. Verify the output is
non-constant and shape is (batch_size, 1). If both fail, use Path A
(aframe_model_dir) with Benedict's pre-built python-backend model instead.
"""

import argparse

import torch

from architectures.supervised import SupervisedS4Model
from train.model.regression import GaussianNLLRegressionAframe


class ScoreWrapper(torch.nn.Module):
    def __init__(self, model: GaussianNLLRegressionAframe):
        super().__init__()
        self.model = model

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # score() returns (N,); Triton expects (N, 1)
        return self.model.score(X).unsqueeze(-1)


def main():
    parser = argparse.ArgumentParser(
        description="Convert .ckpt to TorchScript .pt for Triton"
    )
    parser.add_argument("ckpt", help="Path to Lightning .ckpt file")
    parser.add_argument("out", help="Output TorchScript .pt path")
    # architecture hyperparams -- must match training config s4d_gaussian_nll.yaml
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--d-state", type=int, default=64)
    parser.add_argument("--d-output", type=int, default=2)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--kernel-length", type=float, default=4.0)
    parser.add_argument("--sample-rate", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    arch = SupervisedS4Model(
        num_ifos=2,
        d_output=args.d_output,
        d_model=args.d_model,
        d_state=args.d_state,
        n_layers=args.n_layers,
        dropout=args.dropout,
    )
    lit_model = GaussianNLLRegressionAframe(
        arch=arch,
        param_names=["chirp_mass"],
        y_mean=[1.2],
        y_std=[0.39],
        beta_nll=0.3,
        learning_rate=1e-3,
        pct_lr_ramp=0.115,
        weight_decay=1e-2,
    )

    ckpt = torch.load(args.ckpt, map_location="cpu")
    lit_model.load_state_dict(ckpt["state_dict"])
    lit_model.eval()

    wrapper = ScoreWrapper(lit_model)
    seq_len = int(args.kernel_length * args.sample_rate)
    dummy = torch.randn(args.batch_size, 2, seq_len)

    try:
        scripted = torch.jit.script(wrapper)
        print("torch.jit.script succeeded")
    except Exception as e:
        print(f"torch.jit.script failed ({e}), falling back to trace")
        scripted = torch.jit.trace(wrapper, dummy)

    scripted.save(args.out)
    print(f"Saved to {args.out}")

    out = scripted(dummy)
    print(f"Output shape: {out.shape}")  # expected: (batch_size, 1)
    print(f"Output range: [{out.min():.4f}, {out.max():.4f}]")


if __name__ == "__main__":
    main()
