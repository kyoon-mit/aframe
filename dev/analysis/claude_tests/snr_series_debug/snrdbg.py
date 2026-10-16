import torch
from lightning.pytorch.callbacks import Callback

from train.experimental.snr_series.loss import snr_series


def _report(tag, pl_module, batch):
    X, X_clean = batch[0], batch[1]
    loss_fn = pl_module.denoiser_loss
    sig = X_clean.pow(2).sum(dim=(-2, -1)) > 0
    was_training = pl_module.training
    with torch.no_grad():
        true = snr_series(X, X_clean)
        out = {}
        for mode in ("train", "eval"):
            pl_module.train(mode == "train")
            h = pl_module(X)
            pred = snr_series(X, h, eps=loss_fn.eps)
            per_row = (pred - true).pow(2).mean(dim=(-2, -1))
            out[mode] = (h, pred, per_row)
    pl_module.train(was_training)

    print(f"\n=== {tag}  rows={len(X)} sig={int(sig.sum())}")
    print(f"X std {X.std():.3f}  |X|/row {X.flatten(1).norm(dim=1).mean():.1f}")
    print(f"clean norm (sig rows) {X_clean[sig].flatten(1).norm(dim=1).mean():.2f}")
    print(f"true peak sig {true[sig].amax(dim=(-2,-1)).mean():.2f}")
    for mode, (h, pred, per_row) in out.items():
        hn = h.flatten(1).norm(dim=1)
        print(
            f"[{mode}] |h| sig {hn[sig].mean():.3g} bkg {hn[~sig].mean():.3g}"
            f"  peak sig {pred[sig].amax(dim=(-2,-1)).mean():.2f}"
            f" bkg {pred[~sig].amax(dim=(-2,-1)).mean():.2f}"
            f"  mean snr {pred.mean():.2f}"
            f"  loss sig {per_row[sig].mean():.3f} bkg {per_row[~sig].mean():.3f}"
        )
        if mode == "eval":
            cos = torch.nn.functional.cosine_similarity(
                h.flatten(1), X.flatten(1), dim=1
            )
            print(f"[eval] cos(h, X) mean {cos.mean():.3f}")


class SNRDebug(Callback):
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if batch_idx in (0, 9):
            _report(f"train step {batch_idx}", pl_module, batch)

    def on_validation_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx=0):
        if batch_idx == 0:
            _report("val batch 0", pl_module, batch)


class SNRRows(Callback):
    """Per-row eval loss against the true peak SNR, all validation batches."""

    def on_validation_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx=0):
        X, X_clean = batch[0], batch[1]
        with torch.no_grad():
            true = snr_series(X, X_clean)
            pred = snr_series(X, pl_module(X), eps=pl_module.denoiser_loss.eps)
            per_row = (pred - true).pow(2).mean(dim=(-2, -1))
            tpeak = true.amax(dim=(-2, -1))
        for loss, peak in zip(per_row.tolist(), tpeak.tolist()):
            print(f"ROW {batch_idx} {peak:.2f} {loss:.3f}")


class RowShare(Callback):
    """How much the largest row carries, in loss and in gradient on the output."""

    def __init__(self, windows=((0, 20), (280, 300))):
        self.windows = windows

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        step = trainer.global_step
        if not any(a <= step < b for a, b in self.windows):
            return
        X, X_clean = batch[0], batch[1]
        eps = pl_module.denoiser_loss.eps
        with torch.no_grad():
            h = pl_module(X)
            true = snr_series(X, X_clean)
        with torch.enable_grad():
            h = h.detach().requires_grad_(True)
            per_row = (snr_series(X, h, eps=eps) - true).pow(2).mean(dim=(-2, -1))
            per_row.sum().backward()
        g = h.grad.flatten(1).norm(dim=1)
        r = per_row.detach()
        sig = X_clean.pow(2).sum(dim=(-2, -1)) > 0
        top = r.argmax()
        print(
            f"SHARE {step} mean {r.mean():.3f} median {r.median():.3f} "
            f"maxloss {r.max():.2f} loss_share {r.max() / r.sum():.3f} "
            f"grad_share {g.max() / g.sum():.3f} grad_ratio {g.max() / g.median():.1f} "
            f"top_is_bkg {int(not sig[top])}"
        )
