from typing import Callable, Optional

import torch

from train.model.regression_ky import (
    WarmupCosineAnnealingWarmRestarts,
    clamp_ssm_params,
    load_compatible_weights,
)
from train.losses import soft_pauc_loss
from train.model.supervised import SupervisedAframeS4


class SupervisedAframeS4CustomLR(SupervisedAframeS4):
    """
    SupervisedAframeS4 with a configurable epoch-based learning-rate
    schedule (default: warmup followed by cosine annealing with warm
    restarts) in place of the parent's step-based CosineAnnealingLR.

    Also logs per-parameter gradient norms and S4D kernel (A, dt) stats,
    matching GaussianNLLRegressionAframeCustomLR.

    Args:
        lr_scheduler:
            Callable mapping an optimizer to a learning-rate scheduler. If
            ``None``, WarmupCosineAnnealingWarmRestarts is used with
            ``warmup_epochs=8, T_0=10, T_mult=2, eta_min=1e-7``.
        lr_scheduler_interval:
            "epoch" or "step"; how often the scheduler is stepped.
        normalize_input:
            If True, divide each whitened channel by its own standard
            deviation before the network, matching the kyoon-dev models.
        warm_start_ckpt:
            Path to a checkpoint whose name/shape-compatible weights are
            loaded at init; incompatible tensors keep fresh initialization.
    """

    def __init__(
        self,
        *args,
        lr_scheduler: Optional[
            Callable[[torch.optim.Optimizer], object]
        ] = None,
        lr_scheduler_interval: str = "epoch",
        normalize_input: bool = False,
        warm_start_ckpt: Optional[str] = None,
        pauc_weight: float = 0.0,
        pauc_fpr_frac: float = 0.05,
        log_dt_min: float = -11.5,
        log_dt_max: float = 2.3,
        log_a_max: float = 4.6,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters(
            "lr_scheduler_interval",
            "normalize_input",
            "pauc_weight",
            "pauc_fpr_frac",
            "log_dt_min",
            "log_dt_max",
            "log_a_max",
        )
        self._lr_scheduler_factory = lr_scheduler
        if warm_start_ckpt is not None:
            load_compatible_weights(self, warm_start_ckpt)

    def train_step(self, batch):
        # BCE, optionally + a low-FAR partial-AUROC surrogate (opt-in)
        X, y, _ = batch
        logits = self(X)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y)
        if self.hparams.pauc_weight > 0:
            loss = loss + self.hparams.pauc_weight * soft_pauc_loss(
                logits, y, self.hparams.pauc_fpr_frac
            )
        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx):
        clamp_ssm_params(
            self,
            log_dt_bounds=(self.hparams.log_dt_min, self.hparams.log_dt_max),
            log_a_max=self.hparams.log_a_max,
        )

    def test_step(self, batch, _):
        """Per-batch detection scores for ClassificationTestCallback.

        Test batches mix injected (label 1, with snr) and pure-background
        (label 0) rows via ``waveform_prob``; the raw logit is the ranking
        statistic for the ROC / efficiency-vs-snr plots.
        """
        X, y, params = batch
        score = self(X).reshape(-1)
        out = {
            "score": score.detach().cpu(),
            "label": y.reshape(-1).detach().cpu(),
        }
        if isinstance(params, dict) and "snr" in params:
            out["snr"] = params["snr"].reshape(-1).detach().cpu()
        return out

    def forward(self, X):
        # divide each whitened channel by its own std (kyoon-dev
        # normalize_input), so inputs are exactly unit-variance
        if self.hparams.normalize_input:
            X = X / X.std(dim=-1, keepdim=True).clamp(min=1e-8)
        return self.model(X)

    def on_after_backward(self) -> None:
        for name, param in self.named_parameters():
            if param.grad is not None:
                self.log(
                    f"grad_norm/{name}",
                    param.grad.norm(),
                    on_step=True,
                    on_epoch=False,
                )
            if "log_A_real" in name:
                self.log(
                    f"ssm/A_real_mean/{name}",
                    -param.exp().mean(),
                    on_step=False,
                    on_epoch=True,
                )
                self.log(
                    f"ssm/A_real_max/{name}",
                    -param.exp().max(),
                    on_step=False,
                    on_epoch=True,
                )
            if "log_dt" in name:
                self.log(
                    f"ssm/dt_mean/{name}",
                    param.exp().mean(),
                    on_step=False,
                    on_epoch=True,
                )
                self.log(
                    f"ssm/dt_max/{name}",
                    param.exp().max(),
                    on_step=False,
                    on_epoch=True,
                )

    def configure_optimizers(self):
        world_size = (
            torch.distributed.get_world_size()
            if torch.distributed.is_initialized()
            else 1
        )
        lr = self.hparams.learning_rate * world_size
        self._logger.info(f"Scaled lr by {world_size} to {lr}")

        ssm_params, other_params = [], []
        for name, p in self.model.named_parameters():
            leaf = name.rsplit(".", 1)[-1]
            if leaf in self.SSM_PARAM_NAMES:
                ssm_params.append(p)
            else:
                other_params.append(p)

        optimizer = torch.optim.AdamW(
            [
                {
                    "params": other_params,
                    "lr": lr,
                    "weight_decay": self.hparams.weight_decay,
                },
                {
                    "params": ssm_params,
                    "lr": self.hparams.ssm_lr,
                    "weight_decay": 0.0,
                },
            ]
        )

        if self._lr_scheduler_factory is not None:
            scheduler = self._lr_scheduler_factory(optimizer)
        else:
            scheduler = WarmupCosineAnnealingWarmRestarts(
                optimizer,
                warmup_epochs=8,
                T_0=10,
                T_mult=2,
                eta_min=1e-7,
            )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": self.hparams.lr_scheduler_interval,
            },
        }


class DenoisedClassification(SupervisedAframeS4CustomLR):
    """Joint denoiser + detection classifier.

    Architecture returns ``(x_denoised, logit)`` (e.g.
    ``RegressionTimeDomainS4DenoiseRegress`` with ``d_output=1``). Trains on
    ``BCE(logit, y) + lambda_denoise * denoise(x_denoised, X_clean)`` using the
    ``DenoisingTimeDomainSupervisedAframeDataset`` batch
    ``(X, X_clean, y, params)``. Validation uses only the logit head, so the
    inherited ``TimeSlideAUROC`` path is unchanged. Optional low-FAR pAUC term
    via ``pauc_weight``.
    """

    def __init__(
        self,
        *args,
        denoiser_loss: Optional[torch.nn.Module] = None,
        lambda_denoise: float = 1.0,
        lambda_bce: float = 1.0,
        bce_schedule: Optional[list] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.denoiser_loss = denoiser_loss or torch.nn.MSELoss()
        self.lambda_denoise = lambda_denoise
        # base weight; the schedule multiplies it 0->1 over training
        self._base_lambda_bce = lambda_bce
        self.lambda_bce = lambda_bce
        # step schedule: (epoch, multiplier), applied at/after each epoch;
        # default = denoiser-only for 30 epochs, then joint
        self.bce_schedule = sorted(bce_schedule or [(0, 0.0), (30, 1.0)])

    def _norm(self, X):
        if self.hparams.normalize_input:
            X = X / X.std(dim=-1, keepdim=True).clamp(min=1e-8)
        return X

    def score(self, X):
        _, logit = self.model(self._norm(X))
        return logit

    def on_train_epoch_start(self):
        e = self.current_epoch
        sched = self.bce_schedule
        if e <= sched[0][0]:
            mult = sched[0][1]
        elif e >= sched[-1][0]:
            mult = sched[-1][1]
        else:
            for (e0, m0), (e1, m1) in zip(sched, sched[1:], strict=False):
                if e0 <= e <= e1:
                    mult = m0 + (m1 - m0) * (e - e0) / (e1 - e0)
                    break
        self.lambda_bce = self._base_lambda_bce * mult
        self.log("lambda/bce", self.lambda_bce, on_epoch=True)
        self.log("lambda/denoise", float(self.lambda_denoise), on_epoch=True)

    def train_step(self, batch):
        X, X_clean, y, _ = batch
        x_denoised, logit = self.model(self._norm(X))
        bce = torch.nn.functional.binary_cross_entropy_with_logits(logit, y)
        denoise = self.denoiser_loss(x_denoised, X_clean)
        loss = self.lambda_bce * bce + self.lambda_denoise * denoise
        if self.hparams.pauc_weight > 0:
            loss = loss + self.hparams.pauc_weight * soft_pauc_loss(
                logit, y, self.hparams.pauc_fpr_frac
            )
        self.log("train/bce", bce, on_step=False, on_epoch=True)
        self.log("train/loss_denoise", denoise, on_step=False, on_epoch=True)
        if hasattr(self.denoiser_loss, "alpha"):
            self.log(
                "denoiser_loss/alpha",
                float(self.denoiser_loss.alpha),
                on_step=False,
                on_epoch=True,
            )
        return loss
