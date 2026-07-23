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
        learning_rate: float = 1e-4,
        ssm_lr: Optional[float] = None,
        weight_decay: float = 0.0,
        max_steps: int = 500_000,
        warmup_steps: int = 1000,
        clip_grad_norm: float = 10.0,
        normalize_input: bool = False,
        seed: int = 42,
        load_from_checkpoint: Optional[str] = None,
        reset_optimizer_on_load: bool = True,
    ) -> None:
        ssm_lr = learning_rate if ssm_lr is None else ssm_lr
        super().__init__(
            Architecture(),
            metric=metric,
            ssm_lr=ssm_lr,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            pct_lr_ramp=0.0,
            normalize_input=normalize_input,
        )
        self.automatic_optimization = False
        self._step_marker = nn.Parameter(torch.zeros(1))

        logger.info(f"JAX devices: {jax.devices()}")

        self.jax_model = arch
        self.jax_model_state = eqx.nn.State(self.jax_model)
        self.jax_model_filter_spec = jax.tree_util.tree_map(
            eqx.is_inexact_array, self.jax_model
        )

        def _sched(peak):
            return optax.warmup_cosine_decay_schedule(
                init_value=0.0,
                peak_value=peak,
                warmup_steps=warmup_steps,
                decay_steps=max_steps,
                end_value=peak * 0.01,
            )

        self.scheduler = _sched(learning_rate)
        self.ssm_scheduler = _sched(ssm_lr)
        self.optimizer = optax.chain(
            optax.clip_by_global_norm(clip_grad_norm),
            optax.multi_transform(
                {
                    "ssm": optax.inject_hyperparams(optax.adamw)(
                        learning_rate=self.ssm_scheduler,
                        weight_decay=weight_decay,
                    ),
                    "other": optax.inject_hyperparams(optax.adamw)(
                        learning_rate=self.scheduler,
                        weight_decay=weight_decay,
                    ),
                },
                ssm_param_labels,
            ),
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
        return torch.optim.SGD([self._step_marker], lr=0.0)

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
            keys,
        )

        self.log(
            "train/loss",
            float(metrics["loss"]),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "train/lr",
            float(self.opt_state[1].hyperparams["learning_rate"]),
            on_step=True,
            on_epoch=True,
        )

        opt = self.optimizers()
        opt.step()
        return None
