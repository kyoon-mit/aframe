"""Standalone NaN diagnostic for the chirp_mass pre-merger regression run.

DOES NOT modify committed code. Instantiates the *exact* model + datamodule
from a LightningCLI YAML and runs the REAL Lightning training path, with an
instrumentation callback + a runtime monkeypatch on SnrRescaler (observe only).

Per training step it logs: loss, total grad-norm (pre-clip), min/median/max of
the *intrinsic* SNR that SnrRescaler divides by, max|X|, and min/max predicted
variance.  It stops and dumps a full report at the first non-finite loss so we
can distinguish a forward-pass/data inf (SnrRescaler division, bad X) from a
gradient / variance blow-up (X finite, grad-norm explodes, var collapses).

Usage (GPU node):
    uv run --no-sync python debug_nan.py --config <cfg.yaml> --steps 300 [--clip 1.0] [--anomaly]
"""
import argparse

import torch
import lightning.pytorch as pl
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.callbacks import Callback

import train.augmentations as aug

_SNR = {}


def _patch_snr_rescaler():
    """Wrap SnrRescaler.forward to record the intrinsic-SNR distribution + the
    rescaled response magnitude.  Pure observation; original behavior unchanged."""
    orig = aug.SnrRescaler.forward
    from ml4gw import gw

    def patched(self, responses, psds, target_snrs):
        num_freqs = responses.size(-1) // 2 + 1
        p = psds
        if p.size(-1) != num_freqs:
            if p.ndim == 2:
                p = p[None]; reshape = True
            else:
                reshape = False
            p = torch.nn.functional.interpolate(p, size=(num_freqs,))
            if reshape:
                p = p.view(-1, num_freqs)
        snrs = gw.compute_network_snr(responses, p, self.sample_rate, self.highpass, self.lowpass)
        _SNR["min"] = float(snrs.min()); _SNR["med"] = float(snrs.median()); _SNR["max"] = float(snrs.max())
        _SNR["n_zero"] = int((snrs <= 0).sum()); _SNR["n_tiny"] = int((snrs < 1e-3).sum())
        _SNR["n_nan"] = int((~torch.isfinite(snrs)).sum())
        out = orig(self, responses, psds, target_snrs)
        _SNR["resp_max"] = float(out.abs().max()); _SNR["resp_finite"] = bool(torch.isfinite(out).all())
        return out

    aug.SnrRescaler.forward = patched


def _snr_str():
    k = ["min", "med", "max", "n_zero", "n_tiny", "n_nan", "resp_max", "resp_finite"]
    return " ".join(f"{x}={_SNR.get(x)}" for x in k)


class Probe(Callback):
    def __init__(self, clip):
        self.clip = clip
        self.first_nan = None

    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        g = 0.0
        for p in pl_module.parameters():
            if p.grad is not None:
                g += float(p.grad.detach().norm()) ** 2
        self._gnorm = g ** 0.5

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        loss = float(outputs["loss"]) if isinstance(outputs, dict) else float(outputs)
        X, y, _ = batch
        xmax = float(X.abs().max()); xfin = bool(torch.isfinite(X).all())
        gnorm = getattr(self, "_gnorm", float("nan"))
        finite = (loss == loss) and abs(loss) != float("inf")
        flag = "" if finite else "  <<< NON-FINITE LOSS"
        if batch_idx < 12 or batch_idx % 10 == 0 or not finite:
            print(f"step {batch_idx:4d} | loss={loss:.4f} gnorm={gnorm:.3e} "
                  f"max|X|={xmax:.3e} Xfinite={xfin} | SNR {_snr_str()}{flag}", flush=True)
        if not finite and self.first_nan is None:
            self.first_nan = batch_idx
            print("\n[debug] ===== FIRST NON-FINITE LOSS =====", flush=True)
            print(f"  step={batch_idx}  loss={loss}", flush=True)
            print(f"  X finite={xfin}  max|X|={xmax:.3e}", flush=True)
            print(f"  pre-step grad norm={gnorm:.3e}", flush=True)
            print(f"  intrinsic-SNR/inject: {_snr_str()}", flush=True)
            print("  Interpretation:", flush=True)
            print("    * X non-finite OR n_tiny/n_zero>0 with huge resp_max  -> data/forward inf"
                  " (SnrRescaler target/snr division).", flush=True)
            print("    * X finite, grad norm huge in prior steps             -> gradient blow-up"
                  " (needs clipping / lower lr / saner dt_max).", flush=True)
            trainer.should_stop = True


def build(config_path):
    import tempfile, yaml
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    slim = {k: cfg[k] for k in ("seed_everything", "model", "data") if k in cfg}
    tmp = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
    yaml.safe_dump(slim, tmp); tmp.flush()
    return LightningCLI(run=False, args=["--config", tmp.name], save_config_callback=None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--clip", type=float, default=0.0)
    ap.add_argument("--anomaly", action="store_true")
    ap.add_argument("--seed", type=int, default=-1)
    args = ap.parse_args()

    if args.seed >= 0:
        pl.seed_everything(args.seed)
    _patch_snr_rescaler()
    cli = build(args.config)

    probe = Probe(args.clip)
    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1, logger=False, enable_checkpointing=False, enable_progress_bar=False,
        enable_model_summary=False, max_steps=args.steps, num_sanity_val_steps=0,
        limit_val_batches=0, callbacks=[probe], detect_anomaly=args.anomaly,
        gradient_clip_val=(args.clip or None),
    )
    print(f"[debug] device={'cuda' if torch.cuda.is_available() else 'cpu'} "
          f"steps={args.steps} clip={args.clip} anomaly={args.anomaly}", flush=True)
    trainer.fit(cli.model, datamodule=cli.datamodule)
    if probe.first_nan is None:
        print(f"\n[debug] RESULT: completed {args.steps} steps with FINITE loss (no NaN).", flush=True)
    else:
        print(f"\n[debug] RESULT: first non-finite loss at step {probe.first_nan}.", flush=True)


if __name__ == "__main__":
    main()
