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
            # a frozen denoiser must stay out of the optimizer entirely:
            # AdamW's weight decay would keep shrinking its weights even
            # with no gradient
            if not p.requires_grad:
                continue
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
        denoiser_ckpt: Optional[str] = None,
        freeze_denoiser: bool = False,
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

        # two-stage training: load a denoiser trained on its own by the
        # Denoiser task, then hold it fixed while the classifier learns
        self.freeze_denoiser = freeze_denoiser
        if denoiser_ckpt is not None:
            self._load_denoiser(denoiser_ckpt)
        if freeze_denoiser:
            self._freeze_denoiser()

    @property
    def denoiser(self):
        """The denoiser submodule, wherever the architecture keeps it.

        ``ClassificationTimeDomainS4DenoiseClassifyResNet`` wraps an
        ``S4ModelDenoiseRegress`` in its own ``.model``, so the denoiser
        sits one level deeper than in architectures that hold it directly.
        """
        if hasattr(self.model, "denoiser"):
            return self.model.denoiser
        return self.model.model.denoiser

    def _load_denoiser(self, ckpt_path):
        """Copy denoiser weights out of a standalone Denoiser checkpoint.

        The standalone task stores them under ``model.model.*`` (task ->
        TimeDomainS4Denoiser -> S4ModelSeq2Seq), while here the same stack
        lives at ``model.denoiser.*``, so the prefix is rewritten and only
        shape-matching tensors are taken.
        """
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        source = ckpt.get("state_dict", ckpt)
        target = self.denoiser.state_dict()
        compatible, skipped = {}, []
        for key, value in source.items():
            name = key
            for prefix in ("model.model.", "model.denoiser.", "model."):
                if name.startswith(prefix):
                    name = name[len(prefix) :]
                    break
            if name in target and target[name].shape == value.shape:
                compatible[name] = value
            else:
                skipped.append(key)
        missing = [k for k in target if k not in compatible]
        self.denoiser.load_state_dict(compatible, strict=False)
        self._logger.info(
            f"Loaded denoiser from {ckpt_path}: {len(compatible)} tensors, "
            f"skipped {len(skipped)}, left {len(missing)} fresh {missing}"
        )

    def _freeze_denoiser(self):
        for param in self.denoiser.parameters():
            param.requires_grad = False
        self.denoiser.eval()
        self._logger.info("Denoiser frozen; training classifier head only")

    def train(self, mode: bool = True):
        """Keep the frozen denoiser in eval mode.

        Lightning calls ``train()`` on the whole task at each epoch, which
        would otherwise switch the frozen denoiser's dropout back on and
        make the classifier chase a moving input.
        """
        super().train(mode)
        if self.freeze_denoiser:
            self.denoiser.eval()
        return self

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

    def test_step(self, batch, _):
        """Detection scores plus denoiser reconstruction on test batches.

        The denoising dataset emits ``(X, X_clean, y, params)``, one element
        wider than the parent's test batch, so the unpacking is overridden
        here. Denoiser RMS is reported alongside the ranking statistic: with
        ``waveform_prob=0`` the clean target is identically zero, so
        ``denoised_rms`` is whatever the denoiser produces from noise alone.
        """
        X, X_clean, y, params = batch
        x_denoised, logit = self.model(self._norm(X))
        score = logit.reshape(-1)

        denoised_rms = x_denoised.pow(2).mean(dim=-1).sqrt()
        target_rms = X_clean.pow(2).mean(dim=-1).sqrt()
        self.log("test/denoised_rms", denoised_rms.mean())
        self.log("test/target_rms", target_rms.mean())
        self.log("test/denoise_loss", self.denoiser_loss(x_denoised, X_clean))

        out = {
            "score": score.detach().cpu(),
            "label": y.reshape(-1).detach().cpu(),
            "denoised_rms": denoised_rms.detach().cpu(),
            "target_rms": target_rms.detach().cpu(),
        }
        if isinstance(params, dict) and "snr" in params:
            out["snr"] = params["snr"].reshape(-1).detach().cpu()
        return out

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


class StagedDenoisedClassification(DenoisedClassification):
    """Denoise first, then freeze the denoiser and train the classifier.

    One run, two phases, with the switch at ``freeze_epoch``:

    * before it, ``lambda_bce`` is zero, so only the denoiser trains, and
      the datamodule injects into every sample (``waveform_prob=1.0``)
      since a background-only row teaches a denoiser nothing;
    * from it onward the denoiser is frozen and the classifier trains,
      and ``waveform_prob`` drops to ``classify_waveform_prob`` so the
      classifier sees the background rows it needs to separate signal
      from noise.

    Freezing is three separate things, not just ``requires_grad``: the
    parameters are also dropped from the optimizer, since AdamW weight
    decay would keep shrinking them with no gradient, and the denoiser is
    kept in eval mode so its dropout does not leave the classifier
    chasing a moving input.

    ``bce_schedule`` is derived from ``freeze_epoch`` rather than given
    separately, so the two can never disagree about when the handover is.

    Args:
        freeze_epoch: epoch at which the denoiser freezes and the
            classifier starts training.
        denoise_waveform_prob: injection probability while denoising.
        classify_waveform_prob: injection probability once classifying.
    """

    def __init__(
        self,
        *args,
        freeze_epoch: int = 300,
        denoise_waveform_prob: float = 1.0,
        classify_waveform_prob: float = 0.5,
        **kwargs,
    ):
        # BCE is off until the freeze epoch, then full weight. Two points
        # one epoch apart make it a step rather than a ramp.
        kwargs.setdefault(
            "bce_schedule",
            [[0, 0.0], [freeze_epoch - 1, 0.0], [freeze_epoch, 1.0]],
        )
        super().__init__(*args, **kwargs)
        self.freeze_epoch = freeze_epoch
        self.denoise_waveform_prob = denoise_waveform_prob
        self.classify_waveform_prob = classify_waveform_prob

    @property
    def classifying(self) -> bool:
        return self.current_epoch >= self.freeze_epoch

    def _set_waveform_prob(self, prob: float) -> None:
        """Retarget the datamodule's injection rate.

        ``inject`` reads ``waveform_prob`` from hparams on every batch, so
        setting it here takes effect from the next batch on.
        """
        datamodule = getattr(self.trainer, "datamodule", None)
        if datamodule is None:
            return
        if datamodule.hparams.waveform_prob != prob:
            datamodule.hparams.waveform_prob = prob
            self._logger.info(f"waveform_prob set to {prob}")

    def on_train_epoch_start(self):
        super().on_train_epoch_start()

        if self.classifying:
            self._set_waveform_prob(self.classify_waveform_prob)
            # freeze once, on the first classifier epoch
            if not self.freeze_denoiser:
                self.freeze_denoiser = True
                self._freeze_denoiser()
                # the optimizer still holds the now-frozen tensors, so
                # rebuild it to drop them and their weight decay
                self.trainer.strategy.setup_optimizers(self.trainer)
                self._logger.info(
                    f"epoch {self.current_epoch}: denoiser frozen, "
                    "classifier training"
                )
        else:
            self._set_waveform_prob(self.denoise_waveform_prob)

        self.log("stage/classifying", float(self.classifying), on_epoch=True)
