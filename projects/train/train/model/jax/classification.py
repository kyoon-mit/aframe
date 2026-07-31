import logging
from typing import Optional

import equinox as eqx
import jax
import jax.random as jr
import optax
import torch
from torch import nn

from architectures import Architecture
from architectures.base import JaxArchitecture
from train.metrics import TimeSlideAUROC
from train.model.supervised_ky import SupervisedAframeS4CustomLR
from train.utils.jax.convert import jax_array_to_tensor, tensor_to_jax_array
from train.utils.jax.load_model import load_model
from train.utils.jax.training import (
    ssm_param_labels,
    jax_apply_classification_training_step,
    jax_inference,
)

logger = logging.getLogger(__name__)


class JaxClassificationAframe(SupervisedAframeS4CustomLR):
    """JAX/equinox detection classifier trained with BCE.

    Inherits ``SupervisedAframeS4CustomLR`` so validation uses the same
    ``TimeSlideAUROC`` path as the S4D classifiers. The forward/backward
    pass runs in JAX with an optax optimizer under
    ``automatic_optimization = False``; the architecture outputs a single
    detection logit.
    """

    def __init__(
        self,
        arch: JaxArchitecture,
        metric: TimeSlideAUROC,
        metric_full: Optional[TimeSlideAUROC] = None,
        learning_rate: float = 1e-4,
        ssm_lr: Optional[float] = None,
        weight_decay: float = 0.0,
        clip_grad_norm: float = 10.0,
        max_steps: int = 500_000,  # accepted for config compat
        warmup_steps: int = 1000,  # (warmup now via lr_scheduler)
        lr_scheduler=None,
        lr_scheduler_interval: str = "epoch",
        normalize_input: bool = False,
        seed: int = 42,
        load_from_checkpoint: Optional[str] = None,
        reset_optimizer_on_load: bool = True,
    ) -> None:
        ssm_lr = learning_rate if ssm_lr is None else ssm_lr
        super().__init__(
            Architecture(),
            metric=metric,
            metric_full=metric_full,
            ssm_lr=ssm_lr,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            pct_lr_ramp=0.0,
            lr_scheduler=lr_scheduler,
            lr_scheduler_interval=lr_scheduler_interval,
            normalize_input=normalize_input,
        )
        self.automatic_optimization = False
        self._step_marker = nn.Parameter(torch.zeros(1))
        self._step_marker_ssm = nn.Parameter(torch.zeros(1))

        logger.info(f"JAX devices: {jax.devices()}")

        self.jax_model = arch
        self.jax_model_state = eqx.nn.State(self.jax_model)
        self.jax_model_filter_spec = jax.tree_util.tree_map(
            eqx.is_inexact_array, self.jax_model
        )

        # direction-only adamw; per-group lr applied in the jax step from the
        # torch scheduler. weight decay off for SSM (mixer) params, like S4D.
        def _wd_mask(params):
            labels = ssm_param_labels(params)
            return jax.tree_util.tree_map(lambda lab: lab == "other", labels)

        self.optimizer = optax.chain(
            optax.clip_by_global_norm(clip_grad_norm),
            optax.scale_by_adam(),
            optax.add_decayed_weights(weight_decay, mask=_wd_mask),
        )
        diff_model, _ = eqx.partition(
            self.jax_model, self.jax_model_filter_spec
        )
        self.opt_state = self.optimizer.init(diff_model)

        if load_from_checkpoint is not None:
            self.jax_model, self.jax_model_state, loaded_opt = load_model(
                load_from_checkpoint,
                self.jax_model,
                self.opt_state,
                self.jax_model_state,
            )
            if reset_optimizer_on_load:
                diff_model, _ = eqx.partition(
                    self.jax_model, self.jax_model_filter_spec
                )
                self.opt_state = self.optimizer.init(diff_model)
            else:
                self.opt_state = loaded_opt

        self.rng_key = jr.PRNGKey(seed)

    def configure_optimizers(self):
        # dummy 2-group torch optimizer driven by the S4D-style scheduler so
        # LearningRateMonitor reports both lrs; real update runs in JAX.
        optimizer = torch.optim.SGD(
            [
                {
                    "params": [self._step_marker],
                    "lr": self.hparams.learning_rate,
                },
                {"params": [self._step_marker_ssm], "lr": self.hparams.ssm_lr},
            ]
        )
        if self._lr_scheduler_factory is None:
            return optimizer
        scheduler = self._lr_scheduler_factory(optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": self.hparams.lr_scheduler_interval,
                # log under the same panel as the S4D AdamW models
                # (LearningRateMonitor uses this name over the optimizer
                # class, so the dummy SGD no longer spawns an lr-SGD panel)
                "name": "lr-AdamW",
            },
        }

    def on_train_epoch_end(self):
        sched = self.lr_schedulers()
        if sched is not None and self.hparams.lr_scheduler_interval == "epoch":
            sched.step()

    def _current_lrs(self):
        groups = self.optimizers().param_groups
        return float(groups[0]["lr"]), float(groups[1]["lr"])

    def on_train_batch_end(self, outputs, batch, batch_idx):
        pass

    def _maybe_normalize(self, X: torch.Tensor) -> torch.Tensor:
        if self.hparams.normalize_input:
            X = X / X.std(dim=-1, keepdim=True).clamp(min=1e-8)
        return X

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = tensor_to_jax_array(self._maybe_normalize(X))
        self.rng_key, step_k = jr.split(self.rng_key)
        keys = jr.split(step_k, X.shape[0])
        outputs, new_state = jax_inference(
            self.jax_model, X, self.jax_model_state, keys
        )
        self.jax_model_state = new_state
        return jax_array_to_tensor(outputs)

    def training_step(self, batch):
        X, y, _ = batch
        Xj = tensor_to_jax_array(self._maybe_normalize(X))
        yj = tensor_to_jax_array(y)

        self.rng_key, k = jr.split(self.rng_key)
        keys = jr.split(k, Xj.shape[0])
        lr_other, lr_ssm = self._current_lrs()
        (
            self.jax_model,
            self.jax_model_state,
            self.opt_state,
            metrics,
        ) = jax_apply_classification_training_step(
            self.jax_model,
            self.jax_model_filter_spec,
            self.jax_model_state,
            Xj,
            yj,
            self.opt_state,
            self.optimizer.update,
            lr_other,
            lr_ssm,
            keys,
        )

        self.log(
            "train/loss",
            float(metrics["loss"]),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        # lr is reported by LearningRateMonitor, same as the S4D models

        opt = self.optimizers()
        opt.step()
        return None
