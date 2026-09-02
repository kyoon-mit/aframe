"""Pure denoiser training task: reconstruct the clean strain, nothing else.

No classifier head, no detection metric, no AUROC. The model sees whitened
noisy strain and is scored only on how well it reproduces the injected
waveform, so nothing pulls the representation toward a detection statistic.

Validation reuses the training injection pipeline rather than the timeslide
path, because the timeslide batch carries no clean target to score against.
Set ``waveform_prob=1.0`` so every row carries signal.
"""

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
        self.save_hyperparameters(
            "ssm_lr",
            "normalize_input",
            "lr_scheduler_interval",
            "log_dt_min",
            "log_dt_max",
            "log_a_max",
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

        return loss

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
                optimizer, warmup_epochs=5, T_0=20, T_mult=2, eta_min=1e-7
            )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": self.hparams.lr_scheduler_interval,
            },
        }
