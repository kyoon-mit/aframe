import logging

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import optax
import torch
import numpy as np

from train.metrics import TimeSlideAUROC
from train.model.base import AframeBase
from architectures.base import Architecture, JaxArchitecture

from train.utils.jax.param_count import print_param_tree
from train.utils.jax.load_model import load_model
from train.utils.jax.training import jax_apply_training_step, jax_inference


logger = logging.getLogger(__name__)


class JaxSupervisedAframe(AframeBase):
    """"""

    def __init__(
        self,
        arch: JaxArchitecture,
        metric: TimeSlideAUROC,
        learning_rate: float = 1e-4,
        pct_lr_ramp: float = 0.1,
        weight_decay: float = 0.0,
        max_steps: int = 500_000,
        clip_grad_norm: float = 10.0,
        warmup_steps: int = 1000,
        seed: int = 42,
        load_from_checkpoint: str | None = None,
    ):
        super().__init__(
            arch=Architecture(),
            metric=metric,
            learning_rate=learning_rate,
            pct_lr_ramp=pct_lr_ramp,
            weight_decay=weight_decay,
        )
        self.automatic_optimization = False

        logger.info(f"JAX devices: {jax.devices()}")

        key = jr.PRNGKey(seed)

        self.jax_model = arch
        self.jax_model_state = eqx.nn.State(self.jax_model)
        self.jax_model_filter_spec = jax.tree_util.tree_map(
            eqx.is_inexact_array, self.jax_model
        )

        self.scheduler = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=learning_rate,
            warmup_steps=warmup_steps,
            decay_steps=max_steps,
            end_value=learning_rate * 0.01,
        )
        self.optimizer = optax.chain(
            optax.clip_by_global_norm(clip_grad_norm),
            optax.inject_hyperparams(optax.adamw)(
                learning_rate=self.scheduler, weight_decay=weight_decay
            ),
        )

        diff_model, _ = eqx.partition(
            self.jax_model, self.jax_model_filter_spec
        )
        self.opt_state = self.optimizer.init(diff_model)

        if load_from_checkpoint is not None:
            self.jax_model, self.jax_model_state, self.opt_state = load_model(
                load_from_checkpoint,
                self.jax_model,
                self.opt_state,
                self.jax_model_state,
            )

        self.rng_key = key

        print_param_tree(self.jax_model, depth=4, color="blue")

    def configure_optimizers(self):  # type: ignore
        pass

    def training_step(self, batch):
        X_time, y = batch
        X_time = jnp.asarray(X_time.cpu().numpy())
        y = jnp.asarray(y.cpu().numpy())

        self.rng_key, k = jr.split(self.rng_key)
        batch_size = X_time.shape[0]
        keys = jr.split(k, batch_size)

        (
            self.jax_model,
            self.jax_model_state,
            self.opt_state,
            metrics,
        ) = jax_apply_training_step(
            self.jax_model,
            self.jax_model_filter_spec,
            self.jax_model_state,
            X_time,
            y,
            self.opt_state,
            self.optimizer.update,
            keys,
        )

        loss = metrics["loss"].item()
        self.log(
            "train/loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        self.log(
            "train/lr",
            float(self.opt_state[1].hyperparams["learning_rate"]),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        # Needed for lightning if self.automatic_optimization = False
        optimizers = self.optimizers()
        if isinstance(optimizers, (list, tuple)):
            for opt in optimizers:
                opt.step()
        else:
            optimizers.step()

        return torch.tensor(0.0)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X_time_j = jnp.asarray(X.cpu().numpy())

        self.rng_key, step_k = jr.split(self.rng_key)
        keys = jr.split(step_k, X_time_j.shape[0])

        logits, new_state = jax_inference(
            self.jax_model, X_time_j, self.jax_model_state, keys
        )

        return torch.from_numpy(np.array(logits, copy=True))

    def validation_step(self, batch, _):
        shift, X_bg, X_inj = batch

        y_bg = self.forward(X_bg)

        num_views, batch_size, *shape = X_inj.shape
        X_inj_flat = X_inj.view(num_views * batch_size, *shape)

        y_fg_flat = self.forward(X_inj_flat)
        y_fg = y_fg_flat.view(num_views, batch_size).mean(0)

        self.metric.update(shift, y_bg, y_fg)
        self.log(
            "val/valid_auroc",
            self.metric,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )
