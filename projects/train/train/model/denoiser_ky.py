"""Pure denoiser training task: reconstruct the clean strain, nothing else.

No classifier head, no detection metric, no AUROC. The model sees whitened
noisy strain and is scored only on how well it reproduces the injected
waveform, so nothing pulls the representation toward a detection statistic.

Validation reuses the training injection pipeline rather than the timeslide
path, because the timeslide batch carries no clean target to score against.
Set ``waveform_prob=1.0`` so every row carries signal.
"""

import math
from typing import Callable, Optional

import torch

from train.model.base import AframeBase
from train.model.regression_ky import (
    WarmupCosineAnnealingWarmRestarts,
    clamp_ssm_params,
)


class Denoiser(AframeBase):
    """Train an architecture to map noisy whitened strain to clean strain.

    The loss is expected to expose ``last_time_term`` and
    ``last_spectral_term`` after each call (as ``ScheduledMixtureLoss`` does),
    which are logged separately so the two halves of a mixture loss can be
    compared. When the loss normalises its terms, the logged values are the
    normalised ones, since those are what the optimiser actually sees.

    Args:
        arch: denoiser architecture mapping (B, C, L) to (B, C, L).
        denoiser_loss: reconstruction loss. Defaults to MSE.
        learning_rate: base learning rate for non-SSM parameters.
        ssm_lr: separate, usually larger, learning rate for S4D state
            parameters, which are also excluded from weight decay.
        weight_decay: L2 strength for non-SSM parameters.
        normalize_input: divide each channel by its own standard deviation.
        lr_scheduler: factory taking an optimizer and returning a scheduler.
        log_dt_min, log_dt_max, log_a_max: bounds applied to the S4D
            parameters after every optimiser step to keep them stable.
    """

    SSM_PARAM_NAMES = ("log_dt", "log_A_real", "A_imag")

    def __init__(
        self,
        arch: torch.nn.Module,
        denoiser_loss: Optional[torch.nn.Module] = None,
        learning_rate: float = 1e-4,
        ssm_lr: float = 1e-3,
        weight_decay: float = 0.01,
        normalize_input: bool = True,
        lr_scheduler: Optional[
            Callable[[torch.optim.Optimizer], object]
        ] = None,
        lr_scheduler_interval: str = "epoch",
        log_dt_min: float = -11.5,
        log_dt_max: float = 2.3,
        log_a_max: float = 4.6,
        pct_lr_ramp: float = 0.0,
        verbose: bool = False,
        log_grad_every: int = 1,
        alpha_schedule: Optional[dict] = None,
    ) -> None:
        super().__init__(
            arch=arch,
            learning_rate=learning_rate,
            pct_lr_ramp=pct_lr_ramp,
            weight_decay=weight_decay,
            verbose=verbose,
        )
        self.denoiser_loss = denoiser_loss or torch.nn.MSELoss()
        self._lr_scheduler_factory = lr_scheduler
        self._alpha_schedule = alpha_schedule
        self.save_hyperparameters(
            "ssm_lr",
            "normalize_input",
            "lr_scheduler_interval",
            "log_dt_min",
            "log_dt_max",
            "log_a_max",
            "log_grad_every",
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if self.hparams.normalize_input:
            X = X / X.std(dim=-1, keepdim=True).clamp(min=1e-8)
        return self.model(X)

    def _shared_step(self, batch, stage: str) -> torch.Tensor:
        """Denoise one batch and log the loss and its components.

        Both the training and validation dataloaders run the injection
        pipeline, so each batch is ``(X, X_clean, y, params)``.
        """
        X, X_clean = batch[0], batch[1]
        denoised = self(X)
        loss = self.denoiser_loss(denoised, X_clean)

        on_step = stage == "train"
        self.log(
            f"{stage}/den_loss",
            loss,
            on_step=on_step,
            on_epoch=True,
            prog_bar=True,
        )
        # ScheduledMixtureLoss stashes its raw components; log them when present
        # so the time and frequency halves can be compared directly.
        for attribute, name in (
            ("last_time_term", "den_time_loss"),
            ("last_spectral_term", "den_freq_loss"),
            ("last_shape_term", "den_shape_loss"),
            ("last_gain_term", "den_gain_loss"),
            ("last_bkg_term", "den_bkg_loss"),
        ):
            value = getattr(self.denoiser_loss, attribute, None)
            if value is not None:
                self.log(
                    f"{stage}/{name}",
                    value,
                    on_step=on_step,
                    on_epoch=True,
                )

        # Absolute reconstruction error, independent of whatever loss is in
        # use: residuals accumulated over the kernel and both ifos to give
        # one number per event, then averaged over the events in the batch.
        # Neither is normalised by anything, so both stay comparable across
        # the normalized and un-normalized variants.
        #
        # L1 weights every sample equally; RSS lets large errors dominate.
        with torch.no_grad():
            residual = denoised - X_clean
            self.log(
                f"{stage}/l1_per_event",
                residual.abs().sum(dim=(-2, -1)).mean(),
            )
            self.log(
                f"{stage}/rss_per_event",
                residual.pow(2).sum(dim=(-2, -1)).mean(),
            )

            pred_rms = denoised.pow(2).mean().sqrt()
            target_rms = X_clean.pow(2).mean().sqrt()
            self.log(f"{stage}/pred_rms", pred_rms)
            self.log(f"{stage}/target_rms", target_rms)
            if stage == "train":
                # running mean over the epoch, read by the snr_tracking
                # alpha schedule at the start of the next one
                self._target_rms_total = (
                    getattr(self, "_target_rms_total", 0.0)
                    + float(target_rms)
                )
                self._target_rms_count = (
                    getattr(self, "_target_rms_count", 0) + 1
                )
            # signed: positive means the denoiser is over-predicting
            self.log(f"{stage}/diff_pred_target_rms", pred_rms - target_rms)

            # Loss-independent reconstruction quality. rho is the number that
            # actually says whether the waveform was recovered: it is scale
            # free and phase sensitive, so an edge impulse or a shrunken copy
            # both score near zero. gain is the realised amplitude ratio, and
            # early_frac says how much power sits at the leading window edge
            # (the failure mode of the magnitude-spectrum losses).
            p = denoised.reshape(denoised.shape[0], -1)
            t = X_clean.reshape(X_clean.shape[0], -1)
            tt = t.pow(2).sum(-1)
            sig = tt.sqrt() > 0.5
            if sig.any():
                ps, ts = p[sig], t[sig]
                pn = ps.pow(2).sum(-1).sqrt()
                tn = ts.pow(2).sum(-1).sqrt()
                self.log(
                    f"{stage}/rho",
                    ((ps * ts).sum(-1) / (pn * tn).clamp_min(1e-12)).mean(),
                )
                self.log(f"{stage}/gain", (pn / tn.clamp_min(1e-12)).mean())
            n_early = max(1, denoised.shape[-1] // 20)
            energy = denoised.pow(2).sum(-2)
            self.log(
                f"{stage}/early_frac",
                (
                    energy[..., :n_early].sum(-1)
                    / energy.sum(-1).clamp_min(1e-12)
                ).mean(),
            )

        if self.hparams.log_grad_every and stage == "train":
            self._accumulate_term_gradients(denoised, X_clean)

        return loss

    def _accumulate_term_gradients(self, denoised, X_clean) -> None:
        """Accumulate each loss term's gradient norm for this epoch.

        Runs on every ``log_grad_every`` batch; ``on_train_epoch_end``
        averages and logs the totals as ``grad/<term>``.
        """
        if self.global_step % self.hparams.log_grad_every:
            return

        from train.losses import term_gradient_norms

        stats = term_gradient_norms(self.denoiser_loss, denoised, X_clean)
        if not stats:
            return

        for key, value in stats.items():
            if key.endswith("_gradnorm"):
                name = key[: -len("_gradnorm")]
                self._grad_totals[name] = (
                    self._grad_totals.get(name, 0.0) + value
                )
        self._grad_batches += 1

    def on_train_epoch_start(self) -> None:
        self._grad_totals = {}
        self._grad_batches = 0
        count = getattr(self, "_target_rms_count", 0)
        if count:
            self._last_target_rms = self._target_rms_total / count
        self._target_rms_total = 0.0
        self._target_rms_count = 0
        self._apply_alpha_schedule()

    def _apply_alpha_schedule(self) -> None:
        """Move the loss's alpha along its schedule for this epoch.

        Takes {mode, start, end, start_epoch, end_epoch} with mode one of
        constant, linear, cosine or snr_tracking. Only applies when the loss
        exposes a mutable alpha, as ScheduledMixtureLoss does.

        ``snr_tracking`` keeps the two terms' gradient contributions equal
        instead of interpolating on epoch. The time term's gradient scales as
        the waveform amplitude and the log-magnitude spectral term's as its
        inverse, so their ratio goes as amplitude^-2; measured on this data
        ratio * target_rms^2 is about 14 across the curriculum. alpha is then
        set from the current epoch's target amplitude so that
        alpha * g_time and (1 - alpha) * g_spec stay comparable as the SNR
        curriculum moves the amplitude.
        """
        schedule = self._alpha_schedule
        if schedule is None or not hasattr(self.denoiser_loss, "alpha"):
            return

        epoch = self.current_epoch
        mode = schedule.get("mode", "constant")
        start, end = schedule.get("start", 0.5), schedule.get("end", 0.5)
        first, last = (
            schedule.get("start_epoch", 0),
            schedule.get("end_epoch", 0),
        )

        if mode == "snr_tracking":
            scale = schedule.get("ratio_scale", 14.0)
            amplitude = getattr(self, "_last_target_rms", None)
            if amplitude is None or amplitude <= 0:
                return  # no measurement yet; leave alpha where it is
            ratio = scale / (amplitude**2)
            alpha = ratio / (1.0 + ratio)
            alpha = min(max(alpha, start), end)
            self.log("denoiser_loss/grad_ratio_est", ratio, on_epoch=True)
        elif mode == "constant" or epoch >= last:
            alpha = end if epoch >= last else start
        elif epoch <= first:
            alpha = start
        else:
            fraction = (epoch - first) / (last - first)
            if mode == "cosine":
                fraction = 0.5 * (1 - math.cos(math.pi * fraction))
            alpha = start + (end - start) * fraction

        self.denoiser_loss.alpha = alpha
        self.log("denoiser_loss/alpha", alpha, on_epoch=True)

    def on_train_epoch_end(self) -> None:
        if not getattr(self, "_grad_batches", 0):
            return
        for name, total in self._grad_totals.items():
            self.log(f"grad/{name}", total / self._grad_batches)

    def train_step(self, batch) -> torch.Tensor:
        """AframeBase.training_step delegates here and logs train/loss."""
        return self._shared_step(batch, "train")

    def validation_step(self, batch, _) -> torch.Tensor:
        return self._shared_step(batch, "val")

    def score(self, X: torch.Tensor) -> torch.Tensor:
        """No detection statistic: a denoiser only reconstructs."""
        raise NotImplementedError(
            "Denoiser has no detection statistic; it only reconstructs."
        )

    def on_train_batch_end(self, outputs, batch, batch_idx) -> None:
        clamp_ssm_params(
            self,
            log_dt_bounds=(self.hparams.log_dt_min, self.hparams.log_dt_max),
            log_a_max=self.hparams.log_a_max,
        )

    def configure_optimizers(self):
        world_size = (
            torch.distributed.get_world_size()
            if torch.distributed.is_initialized()
            else 1
        )
        lr = self.hparams.learning_rate * world_size

        ssm_params, other_params = [], []
        for name, parameter in self.model.named_parameters():
            leaf = name.rsplit(".", 1)[-1]
            if leaf in self.SSM_PARAM_NAMES:
                ssm_params.append(parameter)
            else:
                other_params.append(parameter)

        param_groups = [
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

        # A loss may carry its own parameters (ShapeGainLoss with
        # learn_weights uses free log-variances). They live outside
        # self.model, so add them explicitly or they would never be updated.
        loss_params = list(self.denoiser_loss.parameters())
        if loss_params:
            param_groups.append(
                {"params": loss_params, "lr": lr, "weight_decay": 0.0}
            )

        optimizer = torch.optim.AdamW(param_groups)

        if self._lr_scheduler_factory is not None:
            scheduler = self._lr_scheduler_factory(optimizer)
        else:
            scheduler = WarmupCosineAnnealingWarmRestarts(
                optimizer, warmup_epochs=5, T_0=20, T_mult=2, eta_min=1e-7
            )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": self.hparams.lr_scheduler_interval,
            },
        }
