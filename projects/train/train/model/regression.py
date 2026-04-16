import torch
import torch.nn as nn
import lightning.pytorch as pl
from numpy.typing import ArrayLike

from architectures.networks.s4d import S4Model
from architectures.networks.linoss import LinOSSModel


def _log_gaussian_nll(
    task: pl.LightningModule,
    stage: str,
    loss: float,
    indiv_mse: ArrayLike,
    variance: ArrayLike,
) -> None:
    task.log(f'{stage}/gaussnll', loss, on_step=False, on_epoch=True, prog_bar=True)
    for i in range(len(indiv_mse)):
        task.log(f'{stage}/mse/out_{i}', indiv_mse[i], on_step=False, on_epoch=True)
        task.log(f'{stage}/sigma_{i}', torch.sqrt(variance[i].mean(dim=0)), on_step=False, on_epoch=True)


class _RegressionBase(pl.LightningModule):
    """Shared GaussianNLL loss, logging, and step logic for regression tasks."""

    def __init__(self, d_output: int, learning_rate: float, weight_decay: float) -> None:
        super().__init__()
        if d_output % 2 != 0:
            raise ValueError(f'd_output={d_output} must be even (n_vars means + n_vars variances).')
        self.n_vars = d_output // 2
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.criterion = nn.GaussianNLLLoss(reduction='mean')
        self.var_activation = nn.Softplus()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def _prepare_input(self, X_sequence: torch.Tensor) -> torch.Tensor:
        """Subclasses override to transpose if the model expects (B, L, d_input)."""
        return X_sequence

    def compute_loss(self, batch):
        X_sequence, y_target, _ = batch
        outputs = self(self._prepare_input(X_sequence))
        mean = outputs[:, :self.n_vars]
        var = self.var_activation(outputs[:, self.n_vars:])
        indiv_mse = nn.MSELoss(reduction='none')(mean, y_target).T.mean(dim=1)  # (n_vars,)
        return self.criterion(mean, y_target, var), indiv_mse, var

    def training_step(self, batch, batch_idx):
        loss, indiv_mse, var = self.compute_loss(batch)
        _log_gaussian_nll(self, 'train', loss, indiv_mse, var)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, indiv_mse, var = self.compute_loss(batch)
        _log_gaussian_nll(self, 'val', loss, indiv_mse, var)
        return loss

    def test_step(self, batch, batch_idx):
        X_sequence, y_target, _ = batch
        outputs = self(self._prepare_input(X_sequence))
        mean = outputs[:, :self.n_vars]
        sigma = torch.sqrt(self.var_activation(outputs[:, self.n_vars:]))
        return {
            'y_true': y_target.detach().cpu(),
            'y_pred': mean.detach().cpu(),
            'y_sigma': sigma.detach().cpu(),
        }

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )


class LitS4DGaussianNLL(_RegressionBase):
    """S4D sequence model trained with GaussianNLLLoss for parameter estimation.

    Input batch: (X_sequence, y_target, z_observed)
      - X_sequence: (B, n_ifos, L) channels-first
      - y_target:   (B, n_vars)
      - z_observed: (B, n_obs)  not used in loss

    d_output must be even: first half means, second half pre-Softplus variances.
    """

    def __init__(
        self,
        d_input: int,
        d_output: int,
        d_model: int = 256,
        d_state: int = 64,
        n_layers: int = 4,
        dropout: float = 0.2,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        lr: float | None = None,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.0,
    ) -> None:
        super().__init__(d_output=d_output, learning_rate=learning_rate, weight_decay=weight_decay)
        self.save_hyperparameters()
        self.model = None
        self.configure_model()

    def configure_model(self) -> None:
        if self.model is not None:
            return
        # S4Model takes (B, d_input, L) channels-first — no transpose needed
        hp = self.hparams
        self.model = torch.compile(S4Model(
            d_input=hp.d_input, d_output=hp.d_output, d_model=hp.d_model,
            d_state=hp.d_state, n_layers=hp.n_layers, dropout=hp.dropout,
            dt_min=hp.dt_min, dt_max=hp.dt_max, lr=hp.lr,
        ))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def configure_optimizers(self):
        # S4D kernel parameters carry _optim dicts specifying per-param lr / weight_decay
        all_params = list(self.model.parameters())
        general_params = [p for p in all_params if not hasattr(p, '_optim')]
        special_hps = [p._optim for p in all_params if hasattr(p, '_optim')]
        unique_hps = [
            dict(s) for s in sorted(dict.fromkeys(frozenset(hp.items()) for hp in special_hps))
        ]
        optimizer = torch.optim.AdamW(
            general_params, lr=self.learning_rate, weight_decay=self.weight_decay
        )
        for hp in unique_hps:
            params = [p for p in all_params if getattr(p, '_optim', None) == hp]
            optimizer.add_param_group({'params': params, **hp})
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, self.trainer.estimated_stepping_batches
        )
        return {'optimizer': optimizer, 'lr_scheduler': {'scheduler': scheduler, 'interval': 'step'}}


class LitLinOSSGaussianNLL(_RegressionBase):
    """LinOSS sequence model trained with GaussianNLLLoss for parameter estimation.

    Input batch: (X_sequence, y_target, z_observed)
      - X_sequence: (B, n_ifos, L) channels-first
      - y_target:   (B, n_vars)
      - z_observed: (B, n_obs)  not used in loss

    LinOSSModel expects (B, L, d_input), so the input is transposed in _prepare_input.
    """

    def __init__(
        self,
        d_input: int,
        d_output: int,
        d_model: int = 64,
        ssm_size: int = 64,
        n_layers: int = 4,
        dropout: float = 0.0,
        discretization: str = 'IM',
        learning_rate: float = 1e-3,
        weight_decay: float = 0.0,
    ) -> None:
        super().__init__(d_output=d_output, learning_rate=learning_rate, weight_decay=weight_decay)
        self.save_hyperparameters()
        self.model = None
        self.configure_model()

    def configure_model(self) -> None:
        if self.model is not None:
            return
        hp = self.hparams
        self.model = torch.compile(LinOSSModel(
            d_input=hp.d_input, d_output=hp.d_output, d_model=hp.d_model,
            ssm_size=hp.ssm_size, n_layers=hp.n_layers, dropout=hp.dropout,
            discretization=hp.discretization,
        ))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def _prepare_input(self, X_sequence: torch.Tensor) -> torch.Tensor:
        # LinOSSModel expects (B, L, d_input); X_sequence arrives as (B, d_input, L)
        return X_sequence.transpose(1, 2)
