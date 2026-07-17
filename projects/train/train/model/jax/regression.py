import logging
from typing import List, Optional

import equinox as eqx
import jax
import jax.random as jr
import optax
import torch
from torch import nn

from architectures import Architecture
from architectures.base import JaxArchitecture
from train.model.regression import GaussianNLLRegressionAframe
from train.utils.jax.convert import jax_array_to_tensor, tensor_to_jax_array
from train.utils.jax.load_model import load_model
from train.utils.jax.training import (
    jax_apply_regression_training_step,
    jax_inference,
)

logger = logging.getLogger(__name__)


class JaxRegressionAframe(GaussianNLLRegressionAframe):
    """JAX/equinox chirp-mass regression trained with GaussianNLL.

    Reuses the ``GaussianNLLRegressionAframe`` validation (which logs
    ``val/mse_{param}``) and normalization buffers, but runs the
    forward/backward pass in JAX with an optax optimizer while Lightning
    drives the loop under ``automatic_optimization = False``. The
    architecture output is ``(d_output,)`` with the means in the first
    half and pre-Softplus variances in the second half.
    """

    def __init__(
        self,
        arch: JaxArchitecture,
        param_names: List[str],
        beta_nll: float = 0.3,
        lambda_spread: float = 0.0,
        y_mean: Optional[List[float]] = None,
        y_std: Optional[List[float]] = None,
        normalize_input: bool = False,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.0,
        max_steps: int = 500_000,
        warmup_steps: int = 1000,
        clip_grad_norm: float = 10.0,
        seed: int = 42,
        load_from_checkpoint: Optional[str] = None,
        reset_optimizer_on_load: bool = True,
    ) -> None:
        super().__init__(
            arch=Architecture(),
            param_names=param_names,
            beta_nll=beta_nll,
            y_mean=y_mean,
            y_std=y_std,
            learning_rate=learning_rate,
            pct_lr_ramp=0.0,
            weight_decay=weight_decay,
        )
        self.automatic_optimization = False
        self.beta_nll = beta_nll
        self.lambda_spread = lambda_spread
        self.normalize_input = normalize_input

        # dummy torch parameter so optimizer.step() advances global_step
        self._step_marker = nn.Parameter(torch.zeros(1))

        logger.info(f"JAX devices: {jax.devices()}")

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
        # dummy optimizer; the real update is applied in JAX/optax
        return torch.optim.SGD([self._step_marker], lr=0.0)

    def _maybe_normalize(self, X: torch.Tensor) -> torch.Tensor:
        if self.normalize_input:
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
        X, _, params = batch
        mask = ~torch.isnan(next(iter(params.values())))
        targets = torch.stack(
            [params[k][mask] for k in self.param_names], dim=1
        )
        y_norm = self._normalize(targets).reshape(-1, self.n_vars)

        X = tensor_to_jax_array(self._maybe_normalize(X[mask]))
        y_norm = tensor_to_jax_array(y_norm)

        self.rng_key, k = jr.split(self.rng_key)
        keys = jr.split(k, X.shape[0])
        (
            self.jax_model,
            self.jax_model_state,
            self.opt_state,
            metrics,
        ) = jax_apply_regression_training_step(
            self.jax_model,
            self.jax_model_filter_spec,
            self.jax_model_state,
            X,
            y_norm,
            self.beta_nll,
            self.lambda_spread,
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
            sync_dist=True,
        )
        self.log(
            "train/gaussnll",
            float(metrics["nll"]),
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            "train/spread_penalty",
            float(metrics["spread"]),
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            "train/lr",
            float(self.opt_state[1].hyperparams["learning_rate"]),
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )

        opt = self.optimizers()
        opt.step()
        return None
