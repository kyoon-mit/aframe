import logging
from typing import List, Optional

import equinox as eqx
import jax
import jax.random as jr
import optax
import torch
import torch.nn.functional as F
from torch import nn

from architectures import Architecture
from architectures.base import JaxArchitecture
from train.model.regression_ky import GaussianNLLRegressionAframeCustomLR
from train.utils.jax.convert import jax_array_to_tensor, tensor_to_jax_array
from train.utils.jax.load_model import load_model
from train.utils.jax.training import (
    ssm_param_labels,
    jax_apply_regression_training_step,
    jax_inference,
)

logger = logging.getLogger(__name__)


class JaxRegressionAframe(GaussianNLLRegressionAframeCustomLR):
    """JAX/equinox chirp-mass regression trained with GaussianNLL.

    Inherits ``GaussianNLLRegressionAframeCustomLR`` so validation logs the
    exact same metrics as the S4D regression models (``val/gaussnll``,
    ``val/spread_penalty``, ``val/spread_raw``, ``val/loss``,
    ``val/mse_*``, ``val/mae_*``, ``val/sigma_*``,
    ``val/within_{1,2,5,10}pct_*``). The forward/backward pass runs in JAX
    with an optax optimizer while Lightning drives the loop under
    ``automatic_optimization = False``; the training step logs the same
    ``train/*`` metric set. The architecture output is ``(d_output,)`` with
    means in the first half and pre-Softplus variances in the second half.
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
        ssm_lr: Optional[float] = None,
        weight_decay: float = 0.0,
        max_steps: int = 500_000,
        warmup_steps: int = 1000,
        clip_grad_norm: float = 10.0,
        seed: int = 42,
        load_from_checkpoint: Optional[str] = None,
        reset_optimizer_on_load: bool = True,
    ) -> None:
        # ssm_lr defaults to learning_rate (single-rate behavior)
        ssm_lr = learning_rate if ssm_lr is None else ssm_lr
        super().__init__(
            Architecture(),
            param_names,
            beta_nll=beta_nll,
            y_mean=y_mean,
            y_std=y_std,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            ssm_lr=ssm_lr,
            lambda_spread=lambda_spread,
            normalize_input=normalize_input,
            lr_scheduler=None,
            warm_start_ckpt=None,
            pct_lr_ramp=0.0,
        )
        self.automatic_optimization = False
        self.beta_nll = beta_nll
        self.lambda_spread = lambda_spread

        # dummy torch parameter so optimizer.step() advances global_step
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

        # separate lr for the SSM (LinOSS mixer) params vs everything else
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
        # dummy optimizer; the real update is applied in JAX/optax
        return torch.optim.SGD([self._step_marker], lr=0.0)

    def on_train_batch_end(self, outputs, batch, batch_idx):
        # override the S4D SSM clamp (no torch kernel params here)
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
        X, _, params = batch
        mask = ~torch.isnan(next(iter(params.values())))
        targets = torch.stack(
            [params[k][mask] for k in self.param_names], dim=1
        )
        y_norm = self._normalize(targets)

        Xj = tensor_to_jax_array(self._maybe_normalize(X[mask]))
        yj = tensor_to_jax_array(y_norm.reshape(-1, self.n_vars))

        self.rng_key, k = jr.split(self.rng_key)
        keys = jr.split(k, Xj.shape[0])
        (
            self.jax_model,
            self.jax_model_state,
            self.opt_state,
            metrics,
        ) = jax_apply_regression_training_step(
            self.jax_model,
            self.jax_model_filter_spec,
            self.jax_model_state,
            Xj,
            yj,
            self.beta_nll,
            self.lambda_spread,
            self.opt_state,
            self.optimizer.update,
            keys,
        )

        mean = jax_array_to_tensor(metrics["mean"])
        var = jax_array_to_tensor(metrics["var"])

        self.log(
            "train/gaussnll",
            float(metrics["nll"]),
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "train/spread_penalty",
            float(metrics["spread"]),
            on_step=False,
            on_epoch=True,
        )
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

        self.log(
            "train/loss",
            float(metrics["loss"]),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "train/lr",
            float(self.scheduler(self.global_step)),
            on_step=True,
            on_epoch=True,
        )
        self.log(
            "train/ssm_lr",
            float(self.ssm_scheduler(self.global_step)),
            on_step=True,
            on_epoch=True,
        )

        opt = self.optimizers()
        opt.step()
        return None
