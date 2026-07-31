import logging
from typing import Callable, List, Optional

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
        clip_grad_norm: float = 10.0,
        max_steps: int = 500_000,  # accepted for config compat
        warmup_steps: int = 1000,  # (warmup now via lr_scheduler)
        lr_scheduler: Optional[
            Callable[[torch.optim.Optimizer], object]
        ] = None,
        lr_scheduler_interval: str = "epoch",
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
            lr_scheduler=lr_scheduler,
            lr_scheduler_interval=lr_scheduler_interval,
            warm_start_ckpt=None,
            pct_lr_ramp=0.0,
        )
        self.automatic_optimization = False
        self.beta_nll = beta_nll
        self.lambda_spread = lambda_spread

        # dummy torch params (one per lr group) so the S4D-style torch
        # scheduler + LearningRateMonitor drive/report both learning rates
        self._step_marker = nn.Parameter(torch.zeros(1))
        self._step_marker_ssm = nn.Parameter(torch.zeros(1))

        logger.info(f"JAX devices: {jax.devices()}")

        self.jax_model = arch
        self.jax_model_state = eqx.nn.State(self.jax_model)
        self.jax_model_filter_spec = jax.tree_util.tree_map(
            eqx.is_inexact_array, self.jax_model
        )

        # direction-only adamw (no lr): the per-step, per-group lr comes from
        # the torch scheduler and is applied inside the jax training step.
        # weight decay is off for the SSM (mixer) params, matching S4D.
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
        # dummy torch optimizer whose two param-group lrs are driven by the
        # same scheduler S4D uses; the real weight update is applied in JAX
        # (reading these lrs each step). group 0 = other, group 1 = ssm.
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
            },
        }

    def on_train_epoch_end(self):
        # manual optimization: step the lr scheduler ourselves (epoch)
        sched = self.lr_schedulers()
        if sched is not None and self.hparams.lr_scheduler_interval == "epoch":
            sched.step()

    def _current_lrs(self):
        groups = self.optimizers().param_groups
        return float(groups[0]["lr"]), float(groups[1]["lr"])

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
        lr_other, lr_ssm = self._current_lrs()
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
            lr_other,
            lr_ssm,
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
        # lr is reported by LearningRateMonitor, same as the S4D models

        opt = self.optimizers()
        opt.step()
        return None
