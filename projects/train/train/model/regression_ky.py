import math
import warnings
from typing import Callable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from train.model.regression import GaussianNLLRegressionAframe


def clamp_ssm_params(module, log_dt_bounds=(-11.5, 2.3), log_a_max=4.6):
    """Keep the S4D kernel in a numerically sane region. Runaway log_dt /
    log_A_real (from a loud batch under Adam) silently kills the kernel:
    once dt or A overflow, gradients vanish and the layer never recovers."""
    with torch.no_grad():
        for name, param in module.named_parameters():
            if "log_dt" in name:
                param.clamp_(*log_dt_bounds)
            elif "log_A_real" in name:
                param.clamp_(max=log_a_max)


# config-owned buffers that a warm start must never override
WARM_START_EXCLUDE = ("y_mean", "y_std")


def load_compatible_weights(module, ckpt_path):
    """Warm-start from a checkpoint, loading only tensors whose name and
    shape match; everything else keeps its fresh initialization."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    source = ckpt["state_dict"]
    own = module.state_dict()
    loaded, skipped = [], []
    compatible = {}
    for key, value in source.items():
        if key in WARM_START_EXCLUDE:
            skipped.append(key)
        elif key in own and own[key].shape == value.shape:
            compatible[key] = value
            loaded.append(key)
        else:
            skipped.append(key)
    missing = [key for key in own if key not in compatible]
    module.load_state_dict(compatible, strict=False)
    module._logger.info(
        f"Warm start from {ckpt_path}: loaded {len(loaded)} tensors, "
        f"skipped {len(skipped)} incompatible {skipped}, "
        f"left {len(missing)} fresh {missing}"
    )


class WarmupCosineAnnealingWarmRestarts(torch.optim.lr_scheduler.LRScheduler):
    """Linear warmup followed by CosineAnnealingWarmRestarts (epoch-based)."""

    def __init__(
        self,
        optimizer,
        warmup_epochs,
        T_0,
        T_mult=2,
        eta_min=1e-8,
        warmup_start_factor=0.01,
        peak_decay=1.0,
        last_epoch=-1,
    ):
        self.warmup_epochs = warmup_epochs
        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.warmup_start_factor = warmup_start_factor
        # each restart scales the cosine amplitude by peak_decay**n_restart,
        # so successive peaks decay (e.g. 0.8 -> base*0.8, base*0.64, ...)
        self.peak_decay = peak_decay
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        e = self.last_epoch
        if e < self.warmup_epochs:
            alpha = self.warmup_start_factor + (
                1.0 - self.warmup_start_factor
            ) * e / max(1, self.warmup_epochs)
            return [base_lr * alpha for base_lr in self.base_lrs]
        t = e - self.warmup_epochs
        T_cur, T_i, n = self._cosine_position(t)
        decay = self.peak_decay**n
        return [
            self.eta_min
            + (base_lr - self.eta_min)
            * decay
            * (1 + math.cos(math.pi * T_cur / T_i))
            / 2
            for base_lr in self.base_lrs
        ]

    def _cosine_position(self, t):
        T_i, n = self.T_0, 0
        while t >= T_i:
            t -= T_i
            T_i *= self.T_mult
            n += 1
        return t, T_i, n


class GaussianNLLRegressionAframeCustomLR(GaussianNLLRegressionAframe):
    """
    GaussianNLLRegressionAframe with a two-group optimizer and a configurable
    epoch-based learning-rate schedule.

    Parameters whose names appear in SSM_PARAM_NAMES (the S4D state-space
    kernel parameters) are placed in their own optimizer group with learning
    rate ``ssm_lr`` and zero weight decay; all other parameters use
    ``learning_rate`` and ``weight_decay``. The schedule is built by the
    ``lr_scheduler`` factory (default: warmup followed by cosine annealing
    with warm restarts) and stepped per epoch.

    Adds the kyoon-dev training/validation metrics on top of the parent:
    a spread penalty on the loss (weighted by ``lambda_spread``), and
    validation logging of the Gaussian NLL, per-parameter MSE and mean
    predicted sigma, fraction of predictions within 1/2/5/10% relative
    error, and the variance of the prediction across validation views.

    Args:
        ssm_lr:
            Learning rate for the S4D kernel parameters.
        lambda_spread:
            Weight of the spread penalty, which discourages the predicted
            means from collapsing to a narrower distribution than the
            targets. 0 disables it.
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

    SSM_PARAM_NAMES = ("log_dt", "log_A_real", "A_imag")

    def __init__(
        self,
        *args,
        ssm_lr: float = 1e-4,
        lambda_spread: float = 0.0,
        lr_scheduler: Optional[
            Callable[[torch.optim.Optimizer], object]
        ] = None,
        lr_scheduler_interval: str = "epoch",
        normalize_input: bool = False,
        warm_start_ckpt: Optional[str] = None,
        # required by AframeBase but unused here (OneCycle is replaced by
        # lr_scheduler); defaulted so configs may omit it
        pct_lr_ramp: float = 0.0,
        log_dt_min: float = -11.5,
        log_dt_max: float = 2.3,
        log_a_max: float = 4.6,
        **kwargs,
    ):
        super().__init__(*args, pct_lr_ramp=pct_lr_ramp, **kwargs)
        self.save_hyperparameters(
            "ssm_lr",
            "lambda_spread",
            "lr_scheduler_interval",
            "normalize_input",
            "log_dt_min",
            "log_dt_max",
            "log_a_max",
        )
        self._lr_scheduler_factory = lr_scheduler
        if warm_start_ckpt is not None:
            load_compatible_weights(self, warm_start_ckpt)

    def on_train_batch_end(self, outputs, batch, batch_idx):
        clamp_ssm_params(
            self,
            log_dt_bounds=(self.hparams.log_dt_min, self.hparams.log_dt_max),
            log_a_max=self.hparams.log_a_max,
        )

    def forward(self, X):
        # divide each whitened channel by its own std (kyoon-dev
        # normalize_input), so inputs are exactly unit-variance
        if self.hparams.normalize_input:
            X = X / X.std(dim=-1, keepdim=True).clamp(min=1e-8)
        return self.model(X)

    def _spread_penalty(self, y_norm, mean):
        """Penalize predicted means whose spread collapses below the
        targets' spread."""
        return F.softplus(y_norm.detach().var(dim=0) - mean.var(dim=0)).mean()

    def _spread_raw(self, y_norm, mean):
        """Pre-softplus (target_var - pred_var), on an interpretable
        scale: 0 = matched spread, negative = pred over-dispersed
        (fine), positive = pred under-dispersed (collapsing)."""
        return (y_norm.detach().var(dim=0) - mean.var(dim=0)).mean()

    def on_after_backward(self) -> None:
        """Log per-parameter gradient norms and S4D kernel (A, dt) stats."""
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

    def train_step(self, batch):
        X, y, params = batch
        mask = ~torch.isnan(next(iter(params.values())))
        if not mask.any():
            warnings.warn(
                "All samples in batch have NaN parameters;"
                "skipping regression step.",
                stacklevel=2,
            )
            return torch.zeros(1, device=X.device, requires_grad=True)

        targets = torch.stack(
            [params[k][mask] for k in self.param_names], dim=1
        )
        mean, var = self._split(self(X[mask]))
        y_norm = self._normalize(targets)
        nll = self.criterion(mean, y_norm, var)
        spread = self._spread_penalty(y_norm, mean)
        self.log("train/gaussnll", nll, on_step=False, on_epoch=True)
        self.log("train/spread_penalty", spread, on_step=False, on_epoch=True)
        self.log(
            "train/spread_raw",
            self._spread_raw(y_norm, mean),
            on_step=False,
            on_epoch=True,
        )

        mean_phys = mean * self.y_std + self.y_mean
        sigma_phys = torch.sqrt(var) * self.y_std
        rel_err = (mean_phys - targets).abs() / targets.abs().clamp(min=1e-8)
        for i, name in enumerate(self.param_names):
            self.log(
                f"train/mse_{name}",
                F.mse_loss(mean_phys[:, i], targets[:, i]),
                on_step=False,
                on_epoch=True,
            )
            self.log(
                f"train/mae_{name}",
                F.l1_loss(mean_phys[:, i], targets[:, i]),
                on_step=False,
                on_epoch=True,
            )
            self.log(
                f"train/sigma_{name}",
                sigma_phys[:, i].mean(),
                on_step=False,
                on_epoch=True,
            )
            for pct in (1, 2, 5, 10):
                self.log(
                    f"train/within_{pct}pct_{name}",
                    (rel_err[:, i] < pct / 100.0).float().mean(),
                    on_step=False,
                    on_epoch=True,
                )

        return nll + self.hparams.lambda_spread * spread

    def validation_step(self, batch, _):
        _, _, X_inj, params = batch
        num_views, N, *shape = X_inj.shape

        mean, var = self._split(self(X_inj.view(num_views * N, *shape)))
        n_vars = self.n_vars
        mean = mean.view(num_views, N, n_vars)
        var = var.view(num_views, N, n_vars)

        targets = torch.stack(
            [params[name] for name in self.param_names], dim=1
        )
        y_norm = self._normalize(targets)

        nll = self.criterion(
            mean.reshape(-1, n_vars),
            y_norm.repeat(num_views, 1),
            var.reshape(-1, n_vars),
        )
        mean_flat = mean.reshape(-1, n_vars)
        spread = self._spread_penalty(y_norm, mean_flat)
        self.log("val/gaussnll", nll, on_epoch=True, sync_dist=True)
        self.log("val/spread_penalty", spread, on_epoch=True, sync_dist=True)
        self.log(
            "val/spread_raw",
            self._spread_raw(y_norm, mean_flat),
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            "val/loss",
            nll + self.hparams.lambda_spread * spread,
            on_epoch=True,
            sync_dist=True,
        )

        # average prediction over views
        mean_avg = mean.mean(dim=0)
        var_avg = var.mean(dim=0)

        mean_phys = mean_avg * self.y_std + self.y_mean
        sigma_phys = torch.sqrt(var_avg) * self.y_std
        rel_err = (mean_phys - targets).abs() / targets.abs().clamp(min=1e-8)

        for i, name in enumerate(self.param_names):
            self.log(
                f"val/mse_{name}",
                F.mse_loss(mean_phys[:, i], targets[:, i]),
                on_epoch=True,
                sync_dist=True,
            )
            # MAE logged for curiosity only; MSE is the monitored metric
            self.log(
                f"val/mae_{name}",
                F.l1_loss(mean_phys[:, i], targets[:, i]),
                on_epoch=True,
                sync_dist=True,
            )
            self.log(
                f"val/sigma_{name}",
                sigma_phys[:, i].mean(),
                on_epoch=True,
                sync_dist=True,
            )
            for pct in (1, 2, 5, 10):
                self.log(
                    f"val/within_{pct}pct_{name}",
                    (rel_err[:, i] < pct / 100.0).float().mean(),
                    on_epoch=True,
                    sync_dist=True,
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


class DenoisedGaussianNLLRegression(GaussianNLLRegressionAframeCustomLR):
    """Joint S4D denoiser + regressor.

    The architecture (``S4ModelDenoiseRegress``) returns ``(x_denoised, out)``:
    a cleaned I/Q sequence and the ``(N, 2*len(param_names))`` regression head.
    The total loss is ``lambda_denoise * denoise + lambda_regress * regress``,
    where ``regress`` is the parent's beta-NLL (+ spread penalty). ``lambda_
    regress`` follows a step schedule so the denoiser can train alone first
    (e.g. ``[(0, 0.0), (30, 1.0)]`` = denoiser-only for 30 epochs, then joint).
    All ``train/*`` regression metrics from the parent are logged unchanged,
    plus ``train/loss_denoise`` / ``train/loss_regress`` / the lambdas.
    """

    def __init__(
        self,
        *args,
        denoiser_loss: Optional[nn.Module] = None,
        lambda_denoise: float = 0.5,
        lambda_regress: float = 0.5,
        regress_schedule: Optional[List[Tuple]] = None,
        alpha_schedule: Optional[dict] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        # MSE between x_denoised and X_clean, both (B, d_input, L) whitened
        self.denoiser_loss = denoiser_loss or nn.MSELoss()
        self.lambda_denoise = lambda_denoise
        # base weight; the schedule multiplies it 0->1 over training
        self._base_lambda_regress = lambda_regress
        self.lambda_regress = lambda_regress
        # step schedule: (epoch, multiplier), applied at/after each epoch;
        # default = denoiser-only for 30 epochs, then joint
        self.regress_schedule = sorted(
            regress_schedule or [(0, 0.0), (30, 1.0)]
        )
        # denoiser_loss.alpha schedule: {mode, start, end, start_epoch,
        # end_epoch}. mode = constant/linear/cosine. Only applies when
        # denoiser_loss exposes a mutable .alpha (e.g. ScheduledMixtureLoss).
        self.alpha_schedule = alpha_schedule

    def on_train_epoch_start(self):
        e = self.current_epoch
        sched = self.regress_schedule
        if e <= sched[0][0]:
            mult = sched[0][1]
        elif e >= sched[-1][0]:
            mult = sched[-1][1]
        else:
            # linear interpolation between the two bracketing control points
            for (e0, m0), (e1, m1) in zip(sched, sched[1:], strict=False):
                if e0 <= e <= e1:
                    mult = m0 + (m1 - m0) * (e - e0) / (e1 - e0)
                    break
        self.lambda_regress = self._base_lambda_regress * mult
        self.log("lambda/regress", self.lambda_regress, on_epoch=True)
        self.log("lambda/denoise", float(self.lambda_denoise), on_epoch=True)

        if self.alpha_schedule is not None and hasattr(
            self.denoiser_loss, "alpha"
        ):
            s = self.alpha_schedule
            mode = s.get("mode", "constant")
            a0, a1 = s.get("start", 0.5), s.get("end", 0.5)
            e0, e1 = s.get("start_epoch", 0), s.get("end_epoch", 0)
            if mode == "constant" or e >= e1:
                alpha = a1 if e >= e1 else a0
            elif e <= e0:
                alpha = a0
            else:
                frac = (e - e0) / (e1 - e0)
                if mode == "cosine":
                    frac = 0.5 * (1 - math.cos(math.pi * frac))
                alpha = a0 + (a1 - a0) * frac
            self.denoiser_loss.alpha = alpha
            self.log("denoiser_loss/alpha", alpha, on_epoch=True)

    def _log_regression_metrics(self, mean, var, targets, y_norm):
        """Mirror the parent train_step's train/* metric set."""
        nll = self.criterion(mean, y_norm, var)
        spread = self._spread_penalty(y_norm, mean)
        self.log("train/gaussnll", nll, on_step=False, on_epoch=True)
        self.log("train/spread_penalty", spread, on_step=False, on_epoch=True)
        self.log(
            "train/spread_raw",
            self._spread_raw(y_norm, mean),
            on_step=False,
            on_epoch=True,
        )
        mean_phys = mean * self.y_std + self.y_mean
        sigma_phys = torch.sqrt(var) * self.y_std
        rel_err = (mean_phys - targets).abs() / targets.abs().clamp(min=1e-8)
        for i, name in enumerate(self.param_names):
            self.log(
                f"train/mse_{name}",
                F.mse_loss(mean_phys[:, i], targets[:, i]),
                on_step=False,
                on_epoch=True,
            )
            self.log(
                f"train/mae_{name}",
                F.l1_loss(mean_phys[:, i], targets[:, i]),
                on_step=False,
                on_epoch=True,
            )
            self.log(
                f"train/sigma_{name}",
                sigma_phys[:, i].mean(),
                on_step=False,
                on_epoch=True,
            )
            for pct in (1, 2, 5, 10):
                self.log(
                    f"train/within_{pct}pct_{name}",
                    (rel_err[:, i] < pct / 100.0).float().mean(),
                    on_step=False,
                    on_epoch=True,
                )
        return nll + self.hparams.lambda_spread * spread

    def train_step(self, batch):
        # X_clean = noise-free projected waveform, same (whitened) space as
        # the denoiser output; supplied by the datamodule (step 3)
        X, X_clean, y, params = batch

        x_denoised, out = self(X)

        # denoiser trains on ALL rows (noise-only target is ~0)
        loss_denoise = self.denoiser_loss(x_denoised, X_clean)

        # regression + all train/* metrics on injected rows only
        mask = ~torch.isnan(next(iter(params.values())))
        if mask.any():
            targets = torch.stack(
                [params[k][mask] for k in self.param_names], dim=1
            )
            mean, var = self._split(out[mask])
            loss_regress = self._log_regression_metrics(
                mean, var, targets, self._normalize(targets)
            )
        else:
            loss_regress = torch.zeros((), device=X.device)

        loss = (
            self.lambda_denoise * loss_denoise
            + self.lambda_regress * loss_regress
        )
        self.log(
            "train/loss_denoise",
            loss_denoise,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "train/loss_regress",
            loss_regress,
            on_step=False,
            on_epoch=True,
        )
        # scaled contributions to the total, so their sum == train/loss:
        # tells you directly whether the denoiser or regressor dominates
        self.log(
            "train/contrib_denoise",
            self.lambda_denoise * loss_denoise,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "train/contrib_regress",
            self.lambda_regress * loss_regress,
            on_step=False,
            on_epoch=True,
        )
        # ScheduledMixtureLoss sub-terms (raw, pre-alpha): use their ratio to
        # pick alpha so time and spectral terms are comparably weighted
        if hasattr(self.denoiser_loss, "last_time_term"):
            self.log(
                "train/denoise_time",
                self.denoiser_loss.last_time_term,
                on_step=False,
                on_epoch=True,
            )
            self.log(
                "train/denoise_spectral",
                self.denoiser_loss.last_spectral_term,
                on_step=False,
                on_epoch=True,
            )
        # train/loss is logged by AframeBase.training_step from the return
        return loss

    def validation_step(self, batch, _):
        # parent unpacks a single-tensor forward; ours returns a tuple, so
        # override to take the regression head and reuse the same val/* logs
        _, _, X_inj, params = batch
        num_views, N, *shape = X_inj.shape
        _, out = self(X_inj.view(num_views * N, *shape))
        mean, var = self._split(out)
        n_vars = self.n_vars
        mean = mean.view(num_views, N, n_vars)
        var = var.view(num_views, N, n_vars)

        targets = torch.stack(
            [params[name] for name in self.param_names], dim=1
        )
        y_norm = self._normalize(targets)
        nll = self.criterion(
            mean.reshape(-1, n_vars),
            y_norm.repeat(num_views, 1),
            var.reshape(-1, n_vars),
        )
        mean_flat = mean.reshape(-1, n_vars)
        spread = self._spread_penalty(y_norm, mean_flat)
        self.log("val/gaussnll", nll, on_epoch=True, sync_dist=True)
        self.log("val/spread_penalty", spread, on_epoch=True, sync_dist=True)
        self.log(
            "val/loss",
            nll + self.hparams.lambda_spread * spread,
            on_epoch=True,
            sync_dist=True,
        )
        mean_avg = mean.mean(dim=0)
        var_avg = var.mean(dim=0)
        mean_phys = mean_avg * self.y_std + self.y_mean
        sigma_phys = torch.sqrt(var_avg) * self.y_std
        rel_err = (mean_phys - targets).abs() / targets.abs().clamp(min=1e-8)
        for i, name in enumerate(self.param_names):
            self.log(
                f"val/mse_{name}",
                F.mse_loss(mean_phys[:, i], targets[:, i]),
                on_epoch=True,
                sync_dist=True,
            )
            self.log(
                f"val/mae_{name}",
                F.l1_loss(mean_phys[:, i], targets[:, i]),
                on_epoch=True,
                sync_dist=True,
            )
            self.log(
                f"val/sigma_{name}",
                sigma_phys[:, i].mean(),
                on_epoch=True,
                sync_dist=True,
            )
            for pct in (1, 2, 5, 10):
                self.log(
                    f"val/within_{pct}pct_{name}",
                    (rel_err[:, i] < pct / 100.0).float().mean(),
                    on_epoch=True,
                    sync_dist=True,
                )
