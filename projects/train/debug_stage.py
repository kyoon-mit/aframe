"""Localize WHICH data-pipeline stage first produces a non-finite value.

Reuses the real model+datamodule from the YAML and the real trainer.fit path,
but monkeypatches each leaf stage of the on-the-fly injection pipeline to check
finiteness in order:

    psd_estimator -> waveform generator -> SnrRescaler(response+snr) -> whitener

At the first non-finite stage it prints the stage, the step, and diagnostic
stats (incl. the offending sample's prior parameters when it is the waveform).
DOES NOT modify committed code.
"""
import argparse

import torch
import lightning.pytorch as pl
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.callbacks import Callback

import train.augmentations as aug
from utils.preprocessing import PsdEstimator
from ml4gw.transforms import Whiten

STATE = {"step": -1, "prior": None, "reported": set()}


def _report(stage, **kw):
    key = stage
    if key in STATE["reported"]:
        return
    STATE["reported"].add(key)
    print(f"\n[LOCALIZE] >>> first NON-FINITE at stage '{stage}' (train step {STATE['step']}) <<<", flush=True)
    for k, v in kw.items():
        print(f"    {k} = {v}", flush=True)
    print(flush=True)


def _stats(t):
    fin = torch.isfinite(t)
    return (f"shape={tuple(t.shape)} finite={bool(fin.all())} "
            f"n_nonfinite={int((~fin).sum())} max|.|={float(t.abs()[fin].max()) if fin.any() else 'NA'}")


def patch():
    # ---- PSD estimator ----
    _psd = PsdEstimator.forward

    def psd_fwd(self, x, *a, **k):
        out = _psd(self, x, *a, **k)
        psd = out[1] if isinstance(out, (tuple, list)) else out
        if not torch.isfinite(psd).all():
            # which batch element, and min psd value
            bad = (~torch.isfinite(psd)).any(dim=tuple(range(1, psd.ndim)))
            _report("psd_estimator", psd=_stats(psd),
                    n_bad_elems=int(bad.sum()),
                    psd_min=float(psd[torch.isfinite(psd)].min()) if torch.isfinite(psd).any() else "NA",
                    psd_min_overall=float(torch.nan_to_num(psd, nan=float('inf')).min()),
                    input_x=_stats(x if isinstance(x, torch.Tensor) else x[0]))
        else:
            STATE["last_psd_min"] = float(psd.min())
        return out
    PsdEstimator.forward = psd_fwd

    # ---- waveform generator ----
    gen_cls = None
    import train.data.waveforms.generator.cbc_regression as cr
    gen_cls = cr.RegressionCBCGenerator
    _gen_call = gen_cls.__call__

    def gen_call(self, *a, **k):
        STATE["prior"] = k
        out = _gen_call(self, *a, **k)
        hc, hp = out
        if not (torch.isfinite(hc).all() and torch.isfinite(hp).all()):
            bad = (~torch.isfinite(hc)).any(dim=-1) | (~torch.isfinite(hp)).any(dim=-1)
            idx = int(torch.where(bad)[0][0]) if bad.any() else -1
            params = {kk: (float(vv[idx]) if torch.is_tensor(vv) and vv.ndim and idx >= 0 else vv)
                      for kk, vv in k.items()}
            _report("waveform_generator", hc=_stats(hc), hp=_stats(hp),
                    n_bad=int(bad.sum()), first_bad_idx=idx, first_bad_params=params)
        return out
    gen_cls.__call__ = gen_call

    # ---- SnrRescaler (intrinsic snr + rescaled response) ----
    _snr = aug.SnrRescaler.forward
    from ml4gw import gw

    def snr_fwd(self, responses, psds, target_snrs):
        if not torch.isfinite(responses).all():
            _report("response_pre_rescale(observed_strain)", responses=_stats(responses))
        num_freqs = responses.size(-1) // 2 + 1
        p = psds
        if p.size(-1) != num_freqs:
            if p.ndim == 2:
                p = p[None]; rs = True
            else:
                rs = False
            p = torch.nn.functional.interpolate(p, size=(num_freqs,))
            if rs:
                p = p.view(-1, num_freqs)
        snrs = gw.compute_network_snr(responses, p, self.sample_rate, self.highpass, self.lowpass)
        if not torch.isfinite(snrs).all():
            bad = ~torch.isfinite(snrs)
            idx = int(torch.where(bad)[0][0])
            _report("compute_network_snr", snrs=_stats(snrs), first_bad_idx=idx,
                    response_of_bad=_stats(responses[idx]),
                    psd_used=_stats(p), psd_min=float(p.min()),
                    response_all_finite=bool(torch.isfinite(responses).all()))
        out = _snr(self, responses, psds, target_snrs)
        if not torch.isfinite(out).all():
            _report("snr_rescaled_response", out=_stats(out), snrs=_stats(snrs))
        return out
    aug.SnrRescaler.forward = snr_fwd

    # ---- whitener ----
    _wh = Whiten.forward

    def wh_fwd(self, x, psd, *a, **k):
        out = _wh(self, x, psd, *a, **k)
        if not torch.isfinite(out).all():
            _report("whitener", out=_stats(out), in_x=_stats(x), in_psd=_stats(psd),
                    psd_min=float(torch.nan_to_num(psd, nan=float('inf')).min()))
        return out
    Whiten.forward = wh_fwd


class StepTracker(Callback):
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        STATE["step"] = batch_idx

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        loss = float(outputs["loss"]) if isinstance(outputs, dict) else float(outputs)
        if not (loss == loss):
            print(f"[LOCALIZE] loss became NaN at step {batch_idx}; "
                  f"stages reported: {sorted(STATE['reported'])}", flush=True)
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
    ap.add_argument("--steps", type=int, default=200)
    args = ap.parse_args()
    patch()
    cli = build(args.config)
    trainer = pl.Trainer(accelerator="gpu" if torch.cuda.is_available() else "cpu",
                         devices=1, logger=False, enable_checkpointing=False,
                         enable_progress_bar=False, enable_model_summary=False,
                         max_steps=args.steps, num_sanity_val_steps=0, limit_val_batches=0,
                         callbacks=[StepTracker()])
    print(f"[LOCALIZE] starting; will stop at first NaN loss or {args.steps} steps", flush=True)
    trainer.fit(cli.model, datamodule=cli.datamodule)
    print(f"[LOCALIZE] done. stages that went non-finite: {sorted(STATE['reported']) or 'NONE'}", flush=True)


if __name__ == "__main__":
    main()
