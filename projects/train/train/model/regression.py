import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
from lightning.pytorch.cli import LRSchedulerCallable
from numpy.typing import ArrayLike

from train.model.base import AframeBase
from train.utils.beta_nll_loss import BetaNLLLoss


def _log_gaussian_nll(
    task: pl.LightningModule,
    stage: str,
    nll: float,
    indiv_mse: ArrayLike,
    variance: ArrayLike,
) -> None:
    task.log(f"{stage}/gaussnll", nll, on_step=False, on_epoch=True, prog_bar=True)
    for i in range(len(indiv_mse)):
        task.log(f"{stage}/mse/out_{i}", indiv_mse[i], on_step=False, on_epoch=True)
        task.log(
            f"{stage}/sigma_{i}",
            torch.sqrt(variance[i].mean(dim=0)),
            on_step=False,
            on_epoch=True,
        )


def _log_within_percentile(
    task: pl.LightningModule,
    stage: str,
    mean_norm: torch.Tensor,
    y_target: torch.Tensor,
) -> None:
    mean_phys = mean_norm * task.y_std + task.y_mean
    rel_err = (mean_phys - y_target).abs() / y_target.abs().clamp(min=1e-8)
    for pct in [1, 2, 5, 10]:
        within = (rel_err < pct / 100.0).float()
        for i in range(within.shape[-1]):
            task.log(
                f"{stage}/within_{pct}pct/out_{i}",
                within[:, i].mean(),
                on_step=False,
                on_epoch=True,
            )


class RegressionAframe(AframeBase):
    """AframeBase + GaussianNLL loss, regression validation, and warmup+cosine optimizer.

    This is the base for all regression models (``LitS4DGaussianNLL``,
    ``LitLinOSSGaussianNLL``, etc.).  Classification models inherit from
    ``ClassificationAframe`` instead.

    Follows the same ``arch``-first convention as ``ClassificationAframe``:
    the architecture is pre-built and passed in; this class owns the
    training loop, loss, and optimizer logic.
    """

    def __init__(
        self,
        arch,
        d_output: int,
        learning_rate: float,
        weight_decay: float,
        warmup_steps: int = 1000,
        beta_nll: float = 0.5,
        lambda_spread: float = 0.0,
        y_mean: list[float] | None = None,
        y_std: list[float] | None = None,
        normalize_input: bool = False,
    ) -> None:
        super().__init__(arch)
        if d_output % 2 != 0:
            raise ValueError(
                f"d_output={d_output} must be even (n_vars means + n_vars variances)."
            )
        self.n_vars = d_output // 2
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.lambda_spread = lambda_spread
        self.normalize_input = normalize_input
        self.criterion = BetaNLLLoss(beta=beta_nll)
        self.var_activation = nn.Softplus()

        _y_mean = (
            torch.tensor(y_mean, dtype=torch.float32)
            if y_mean is not None
            else torch.zeros(self.n_vars)
        )
        _y_std = (
            torch.tensor(y_std, dtype=torch.float32)
            if y_std is not None
            else torch.ones(self.n_vars)
        )
        self.register_buffer("y_mean", _y_mean)
        self.register_buffer("y_std", _y_std)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.model(X)

    def _prepare_input(self, X: torch.Tensor) -> torch.Tensor:
        if self.normalize_input:
            X = X / X.std(dim=-1, keepdim=True).clamp(min=1e-8)
        return X

    def _normalize_target(self, y: torch.Tensor) -> torch.Tensor:
        return (y - self.y_mean) / self.y_std

    def _unnormalize_output(
        self, mean: torch.Tensor, sigma: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return mean * self.y_std + self.y_mean, sigma * self.y_std

    def compute_loss(self, batch):
        X, y_target, _ = batch
        outputs = self(self._prepare_input(X))
        mean = outputs[:, : self.n_vars]
        var = self.var_activation(outputs[:, self.n_vars :])
        y_norm = self._normalize_target(y_target)
        indiv_mse = nn.MSELoss(reduction="none")(mean, y_norm).T.mean(dim=1)
        nll = self.criterion(mean, y_norm, var)
        spread = F.softplus(y_norm.detach().var(dim=0) - mean.var(dim=0)).mean()
        loss = nll + self.lambda_spread * spread
        return loss, nll, spread, indiv_mse, var, mean, y_target

    def training_step(self, batch, batch_idx):
        loss, nll, spread, indiv_mse, var, _, _ = self.compute_loss(batch)
        _log_gaussian_nll(self, "train", nll, indiv_mse, var)
        self.log("train/spread_penalty", spread, on_step=False, on_epoch=True)
        self.log("train/loss", loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, nll, spread, indiv_mse, var, mean_norm, y_target = self.compute_loss(
            batch
        )
        _log_gaussian_nll(self, "val", nll, indiv_mse, var)
        _log_within_percentile(self, "val", mean_norm, y_target)
        self.log("val/spread_penalty", spread, on_step=False, on_epoch=True)
        self.log("val/loss", loss, on_step=False, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        X, y_target, _ = batch
        outputs = self(self._prepare_input(X))
        mean_norm = outputs[:, : self.n_vars]
        sigma_norm = torch.sqrt(self.var_activation(outputs[:, self.n_vars :]))
        mean_phys, sigma_phys = self._unnormalize_output(mean_norm, sigma_norm)
        return {
            "y_true": y_target.detach().cpu(),
            "y_pred": mean_phys.detach().cpu(),
            "y_sigma": sigma_phys.detach().cpu(),
        }

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        total_steps = self.trainer.estimated_stepping_batches
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1e-2,
            end_factor=1.0,
            total_iters=self.warmup_steps,
        )
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, total_steps - self.warmup_steps)
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup, cosine],
            milestones=[self.warmup_steps],
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }


class RegressionAframeS4D(RegressionAframe):
    """S4D sequence model trained with GaussianNLL for parameter estimation.

    Pass a pre-built ``S4Model`` (or compatible) as ``arch``.
    ``d_output`` must be even: first half = means, second half = pre-Softplus variances.

    Uses S4D-aware optimizer: SSM parameters (those with ``._optim``) get their
    own learning-rate group; all other parameters share ``base_lr``.
    """

    def __init__(
        self,
        arch,
        d_output: int,
        base_lr: float = 1e-4,
        weight_decay: float = 0.0,
        warmup_steps: int = 1000,
        beta_nll: float = 0.5,
        lambda_spread: float = 0.0,
        lr_scheduler: LRSchedulerCallable | None = None,
        lr_scheduler_interval: str = "epoch",
        y_mean: list[float] | None = None,
        y_std: list[float] | None = None,
        normalize_input: bool = False,
        log_gradients: bool = False,
    ) -> None:
        super().__init__(
            arch,
            d_output=d_output,
            learning_rate=base_lr,
            weight_decay=weight_decay,
            warmup_steps=warmup_steps,
            beta_nll=beta_nll,
            lambda_spread=lambda_spread,
            y_mean=y_mean,
            y_std=y_std,
            normalize_input=normalize_input,
        )
        self._lr_scheduler_factory = lr_scheduler
        self.log_gradients = log_gradients
        self.save_hyperparameters(ignore=["arch", "lr_scheduler"])

    def on_after_backward(self) -> None:
        if self.log_gradients:
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

    def configure_optimizers(self):
        hp = self.hparams
        all_params = list(self.model.parameters())
        default_params = [p for p in all_params if not hasattr(p, "_optim")]
        optim_params = [p for p in all_params if hasattr(p, "_optim")]
        param_groups = [
            {
                "params": default_params,
                "lr": hp.base_lr,
                "weight_decay": hp.weight_decay,
            }
        ]
        unique_hps = [
            dict(s)
            for s in sorted(set(frozenset(p._optim.items()) for p in optim_params))
        ]
        for ohp in unique_hps:
            group = {
                "params": [p for p in optim_params if getattr(p, "_optim") == ohp],
                "lr": ohp.get("lr", hp.base_lr),
            }
            group.update(ohp)
            param_groups.append(group)
        optimizer = torch.optim.AdamW(param_groups)
        if self._lr_scheduler_factory is None:
            return optimizer
        scheduler = self._lr_scheduler_factory(optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": hp.lr_scheduler_interval,
            },
        }
