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

from train.losses import term_gradient_norms
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
        # per-term gradient norms, summed over an epoch and averaged at its
        # end; the grad_tracking schedule reads the result
        self._grad_norm_totals: dict[str, float] = {}
        self._grad_norm_batches = 0
        self._measured_grad_ratio: Optional[float] = None
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
        # ScheduledMixtureLoss stashes its raw components; log the ones
        # present so the halves of a mixture can be compared directly.
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
            # signed: positive means the denoiser is over-predicting
            self.log(f"{stage}/diff_pred_target_rms", pred_rms - target_rms)

            self._log_recovery(denoised, X_clean, stage)

        if stage == "train":
            self._accumulate_term_gradients(denoised, X_clean)

        return loss

    def _log_recovery(self, denoised, target, stage: str) -> None:
        """Whether the waveform came back, independent of the loss.

        ``rho`` is the normalised inner product with the clean target:
        scale free and phase sensitive, so a shrunken or misplaced copy
        scores near zero however small its loss. ``gain`` is the amplitude
        actually realised. Rows with no signal are skipped.
        """
        pred_flat = denoised.reshape(denoised.shape[0], -1)
        target_flat = target.reshape(target.shape[0], -1)

        target_norm = target_flat.pow(2).sum(-1).sqrt()
        carries_signal = target_norm > 0.5
        if not carries_signal.any():
            return

        pred_flat = pred_flat[carries_signal]
        target_flat = target_flat[carries_signal]
        pred_norm = pred_flat.pow(2).sum(-1).sqrt()
        target_norm = target_norm[carries_signal].clamp_min(1e-12)

        overlap = (pred_flat * target_flat).sum(-1)
        self.log(
            f"{stage}/rho",
            (overlap / (pred_norm * target_norm).clamp_min(1e-12)).mean(),
        )
        self.log(f"{stage}/gain", (pred_norm / target_norm).mean())

    def _accumulate_term_gradients(self, denoised, target) -> None:
        """Add this batch's per-term gradient norms to the epoch totals.

        Costs an extra backward pass, so ``log_grad_every`` thins it out;
        0 turns it off.
        """
        every = self.hparams.log_grad_every
        if not every or self.global_step % every:
            return

        stats = term_gradient_norms(self.denoiser_loss, denoised, target)
        if not stats:
            return

        suffix = "_gradnorm"
        for key, value in stats.items():
            if key.endswith(suffix):
                name = key[: -len(suffix)]
                self._grad_norm_totals[name] = (
                    self._grad_norm_totals.get(name, 0.0) + value
                )
        self._grad_norm_batches += 1

    def on_train_epoch_start(self) -> None:
        """Average the last epoch's measurements, clear them, set alpha."""
        if self._grad_norm_batches:
            time_norm = self._grad_norm_totals.get("time", 0.0)
            spectral_norm = self._grad_norm_totals.get("spectral", 0.0)
            if time_norm > 0 and spectral_norm > 0:
                self._measured_grad_ratio = spectral_norm / time_norm
        self._grad_norm_totals = {}
        self._grad_norm_batches = 0

        self._apply_alpha_schedule()

    def on_train_epoch_end(self) -> None:
        if not self._grad_norm_batches:
            return
        for name, total in self._grad_norm_totals.items():
            self.log(f"grad/{name}", total / self._grad_norm_batches)

    def _apply_alpha_schedule(self) -> None:
        """Move the loss's alpha along its schedule for this epoch.

        Takes ``{mode, start, end, start_epoch, end_epoch}``, and needs a
        loss with a mutable ``alpha``. ``constant``, ``linear`` and
        ``cosine`` interpolate from ``start`` to ``end`` over the epoch
        range. ``grad_tracking`` instead holds the two terms' gradient
        contributions equal, setting alpha from the measured ratio of
        their norms; ``start`` and ``end`` then act as bounds.
        """
        schedule = self._alpha_schedule
        if schedule is None or not hasattr(self.denoiser_loss, "alpha"):
            return

        mode = schedule.get("mode", "constant")
        start = schedule.get("start", 0.5)
        end = schedule.get("end", 0.5)

        if mode == "grad_tracking":
            ratio = self._measured_grad_ratio
            if ratio is None or ratio <= 0:
                return  # nothing measured yet; leave alpha alone
            self.log("denoiser_loss/grad_ratio", ratio, on_epoch=True)
            target = ratio / (1.0 + ratio)
            # jumping straight to the target is unstable, so optionally
            # approach it; momentum 0 takes the measurement as-is
            momentum = schedule.get("momentum", 0.0)
            previous = float(getattr(self.denoiser_loss, "alpha", target))
            alpha = momentum * previous + (1.0 - momentum) * target
        else:
            alpha = self._interpolated_alpha(mode, start, end, schedule)

        alpha = min(max(alpha, start), end)
        self.denoiser_loss.alpha = alpha
        self.log("denoiser_loss/alpha", alpha, on_epoch=True)

    def _interpolated_alpha(
        self, mode: str, start: float, end: float, schedule: dict
    ) -> float:
        """Alpha for the fixed-shape modes, from the epoch counter."""
        epoch = self.current_epoch
        first = schedule.get("start_epoch", 0)
        last = schedule.get("end_epoch", 0)

        if mode == "constant" or epoch >= last:
            return end if epoch >= last else start
        if epoch <= first:
            return start

        fraction = (epoch - first) / (last - first)
        if mode == "cosine":
            fraction = 0.5 * (1 - math.cos(math.pi * fraction))
        return start + (end - start) * fraction

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

        # A loss can carry parameters of its own (learned term weights, say).
        # They sit outside self.model, so without this they would never be
        # updated.
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
