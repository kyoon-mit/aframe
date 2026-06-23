import io
import os
import shutil
import logging
from typing import Optional


import h5py
import s3fs
import torch
from botocore.exceptions import ClientError, ConnectTimeoutError
from lightning import pytorch as pl
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.cli import SaveConfigCallback
from lightning.pytorch.callbacks import ModelCheckpoint as PLModelCheckpoint
from lightning.pytorch.callbacks import EarlyStopping as PLEarlyStopping
from lightning.pytorch.utilities import grad_norm
from typing import Literal, Union
from pathlib import Path

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

log = logging.getLogger(__name__)
BOTO_RETRY_EXCEPTIONS = (ClientError, ConnectTimeoutError)


def get_save_dir(trainer) -> str:
    """
    Helper to get the correct save directory
    Uses wandb project/id if available.
    """
    save_dir = trainer.logger.save_dir or "lightning_logs"
    for logger in trainer.loggers:
        if isinstance(logger, WandbLogger):
            try:
                project = logger.experiment.project or "aframe"
                run_id = logger.experiment.id or "unknown"
                save_dir = os.path.join(save_dir, project, run_id)
            except Exception:
                pass
            break

    if not save_dir.startswith("s3://"):
        save_dir = os.path.abspath(save_dir)
        os.makedirs(save_dir, exist_ok=True)

    return save_dir


class AframeWandbLogger(WandbLogger):
    """Thin wrapper around WandbLogger with clean type annotations.

    WandbLogger.__init__ uses ForwardRef('Run') / ForwardRef('RunDisabled')
    for the `experiment` parameter, which breaks jsonargparse's get_type_hints
    call. This subclass re-declares __init__ without that parameter so the
    Lightning CLI can instantiate it from a YAML config.
    """

    def __init__(
        self,
        name: Optional[str] = None,
        save_dir: Union[str, Path] = ".",
        version: Optional[str] = None,
        offline: bool = False,
        dir: Optional[Union[str, Path]] = None,
        id: Optional[str] = None,
        anonymous: Optional[bool] = None,
        project: Optional[str] = None,
        log_model: Union[Literal["all"], bool] = False,
        prefix: str = "",
        checkpoint_name: Optional[str] = None,
    ):
        super().__init__(
            name=name,
            save_dir=save_dir,
            version=version,
            offline=offline,
            dir=dir,
            id=id,
            anonymous=anonymous,
            project=project,
            log_model=log_model,
            prefix=prefix,
            checkpoint_name=checkpoint_name,
            save_code=True,
        )

        self._offline = offline

    @property
    def experiment(self):
        # Accessing the parent's experiment property initializes the run
        exp = super().experiment
        # The run has a log_code method
        if not getattr(self, "_code_logged", False):
            if not getattr(self, "offline", False):
                exp.log_code("train")
            self._code_logged = True
        return exp


class WandbSaveConfig(SaveConfigCallback):
    """
    Override of `lightning.pytorch.cli.SaveConfigCallback` for use with WandB
    to ensure all the hyperparameters are logged to the WandB dashboard.
    """

    def get_wandb_logger(self, trainer) -> Optional[WandbLogger]:
        for logger in trainer.loggers:
            if isinstance(logger, WandbLogger):
                return logger

    def save_config(self, trainer, pl_module, stage) -> None:
        wandb_logger = self.get_wandb_logger(trainer)
        if stage == "fit" and wandb_logger is not None:
            # pop off unecessary trainer args
            config = self.config.as_dict()
            config_copy = config.copy()
            if "trainer" in config_copy:
                config_copy.pop("trainer")
            if "CONDOR_JOB_ID" in os.environ:
                config_copy["CONDOR_JOB_ID"] = os.environ["CONDOR_JOB_ID"]
            wandb_logger.experiment.config.update(config_copy)

        save_dir = get_save_dir(trainer)

        log.info(f"WandbSaveConfig save dir: {save_dir}")
        if wandb_logger is not None:
            wandb_logger.experiment.config.update(
                {"WandbSaveConfig_save_dir": save_dir}
            )

        config_path = os.path.join(save_dir, "config.yaml")
        if save_dir.startswith("s3://"):
            s3 = s3fs.S3FileSystem()
            with s3.open(config_path, "w") as f:
                self.parser.save(
                    self.config,
                    str(f),
                    skip_none=False,
                    overwrite=True,
                    multifile=False,
                )
        else:
            os.makedirs(save_dir, exist_ok=True)
            self.parser.save(
                self.config,
                config_path,
                skip_none=False,
                overwrite=True,
                multifile=False,
            )


class ModelCheckpoint(PLModelCheckpoint):
    def setup(self, trainer, pl_module, stage: str):
        super().setup(trainer, pl_module, stage)
        # override dirpath to save checkpoints in the desired run directory
        save_dir = get_save_dir(trainer)
        self.dirpath = os.path.join(save_dir, "checkpoints")

    def on_train_end(self, trainer, pl_module):
        torch.cuda.empty_cache()
        module = pl_module.__class__.load_from_checkpoint(
            self.best_model_path, arch=pl_module.model, metric=pl_module.metric
        )

        device = pl_module.device
        # Handle the case of loading training waveforms from disk
        if trainer.datamodule.waveforms_from_disk:
            [X], waveforms = next(iter(trainer.train_dataloader))
            X = X.to(device)
            waveforms = waveforms.to(device)
            X, y = trainer.datamodule.inject(X, waveforms)
        else:
            [X] = next(iter(trainer.train_dataloader))
            X = X.to(device)
            X, y = trainer.datamodule.inject(X)
        if isinstance(X, tuple):
            X = tuple(i.cpu() for i in X)
        else:
            X = X.cpu()
        trace = torch.jit.trace(module.model.to("cpu"), X)

        save_dir = get_save_dir(trainer)
        if not save_dir.startswith("s3://"):
            save_dir = os.path.abspath(save_dir)

        log.info(f"ModelCheckpoint save dir: {save_dir}")
        for logger in trainer.loggers:
            if isinstance(logger, WandbLogger):
                logger.experiment.config.update(
                    {"ModelCheckpoint_save_dir": save_dir},
                    allow_val_change=True,
                )
                break

        if save_dir.startswith("s3://"):
            s3 = s3fs.S3FileSystem()
            with s3.open(f"{save_dir}/model.pt", "wb") as f:
                torch.jit.save(trace, f)

            s3.copy(self.best_model_path, f"{save_dir}/best.ckpt")
        else:
            with open(os.path.join(save_dir, "model.pt"), "wb") as f:
                torch.jit.save(trace, f)
            shutil.copy(
                self.best_model_path, os.path.join(save_dir, "best.ckpt")
            )


class SaveAugmentedBatch(Callback):
    def on_train_start(self, trainer, pl_module):
        if trainer.global_rank == 0:
            # find device module is on
            device = pl_module.device
            save_dir = get_save_dir(trainer)
            if not save_dir.startswith("s3://"):
                save_dir = os.path.abspath(save_dir)

            log.info(f"SaveAugmentedBatch save dir: {save_dir}")
            for logger in trainer.loggers:
                if isinstance(logger, WandbLogger):
                    logger.experiment.config.update(
                        {"SaveAugmentedBatch_save_dir": save_dir},
                        allow_val_change=True,
                    )
                    break

            # build training batch by hand
            # Handle the case of loading training waveforms from disk
            if trainer.datamodule.waveforms_from_disk:
                [X], waveforms = next(iter(trainer.train_dataloader))
                X = X.to(device)
                waveforms = waveforms.to(device)
                X, y = trainer.datamodule.inject(X, waveforms)
            else:
                [X] = next(iter(trainer.train_dataloader))
                X = X.to(device)
                X, y = trainer.datamodule.inject(X)
            # If X is not a tuple, make it one for consistency
            # of format for saving to file below
            if not isinstance(X, tuple):
                X = (X,)

            # build val batch by hand
            [background, _, _], [signals] = next(
                iter(trainer.datamodule.val_dataloader())
            )
            background = background.to(device)
            signals = signals.to(device)
            X_bg, X_inj = trainer.datamodule.build_val_batches(
                background, signals
            )
            # Make background and injected validation data into
            # tuples for consistency if necessary
            if not isinstance(X_bg, tuple):
                X_bg = (X_bg,)
            if not isinstance(X_inj, tuple):
                X_inj = (X_inj,)

            if save_dir.startswith("s3://"):
                s3 = s3fs.S3FileSystem()
                with s3.open(f"{save_dir}/batch.hdf5", "wb") as s3_file:
                    with io.BytesIO() as f:
                        with h5py.File(f, "w") as h5file:
                            for i, x in enumerate(X):
                                h5file[f"input_{i}"] = x.cpu().numpy()
                            h5file["y"] = y.cpu().numpy()
                        s3_file.write(f.getvalue())

                with s3.open(f"{save_dir}/val_batch.hdf5", "wb") as s3_file:
                    with io.BytesIO() as f:
                        with h5py.File(f, "w") as h5file:
                            for i, (bg, inj) in enumerate(
                                zip(X_bg, X_inj, strict=True)
                            ):
                                h5file[f"X_bg_{i}"] = bg.cpu().numpy()
                                h5file[f"X_inj_{i}"] = inj.cpu().numpy()
                        s3_file.write(f.getvalue())
            else:
                with h5py.File(os.path.join(save_dir, "batch.hdf5"), "w") as f:
                    for i, x in enumerate(X):
                        f[f"input_{i}"] = x.cpu().numpy()
                    f["y"] = y.cpu().numpy()

                with h5py.File(
                    os.path.join(save_dir, "val_batch.hdf5"), "w"
                ) as f:
                    for i, (bg, inj) in enumerate(
                        zip(X_bg, X_inj, strict=True)
                    ):
                        f[f"X_bg_{i}"] = bg.cpu().numpy()
                        f[f"X_inj_{i}"] = inj.cpu().numpy()

            # while we're here let's log the wandb url
            # associated with the run
            maybe_wandb_logger = trainer.loggers[-1]
            if isinstance(maybe_wandb_logger, pl.loggers.WandbLogger):
                url = maybe_wandb_logger.experiment.url
                if save_dir.startswith("s3://"):
                    with s3.open(f"{save_dir}/wandb_url.txt", "wb") as s3_file:
                        s3_file.write(url.encode())
                else:
                    with open(
                        os.path.join(save_dir, "wandb_url.txt"), "w"
                    ) as f:
                        f.write(url)


class GradientTracker(Callback):
    def __init__(self, norm_type: int = 2):
        self.norm_type = norm_type

    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        norms = grad_norm(pl_module, norm_type=self.norm_type)
        total_norm = norms[f"grad_{float(self.norm_type)}_norm_total"]
        self.log(f"grad_norm_{self.norm_type}", total_norm)


class EarlyStopping(PLEarlyStopping):
    """Early stopping that honors the current config when resuming a run.

    Lightning saves early stopping's own state into the checkpoint — its
    patience, how long it has gone without improvement, and the best score seen
    so far — and restores all of it on resume. The surprising consequence is
    that the patience set in the config is silently ignored on any resumed run:
    the old patience and the old wait counter come back and training can stop
    almost immediately. That is not what early stopping is meant to do. This
    version reads its settings from the config every time and starts its counter
    and best score fresh, while leaving the rest of the resume (weights,
    optimizer, epoch) untouched.

    Also treats a negative ``patience`` as "never early-stop" (infinite
    patience). Lightning's raw behavior for ``patience=-1`` is the opposite — it
    would stop at the first non-improving check — so we remap it here.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.patience < 0:
            # Negative patience == infinite patience (never early-stop).
            self.patience = 2**31

    def load_state_dict(self, state_dict):
        # Skip restoring patience / wait_count / best_score so the values built
        # from the current config are kept instead of the checkpoint's stale ones.
        pass





# ---------------------------------------------------------------------------
# PlotParamEstCallback (imported from BNSReg/callbacks/plot_param_est.py)
# ---------------------------------------------------------------------------

# Bin edges and labels used by PlotParamEstCallback
BINS: dict[str, np.ndarray] = {
    'chirp_mass':  np.linspace(0.5,   3.0,  51),
    'mass_ratio':  np.linspace(0.0,   1.0,  51),
    'mass_1':      np.linspace(1.0,   2.5,  31),
    'mass_2':      np.linspace(1.0,   2.5,  31),
    'm1':          np.linspace(1.0,   2.5,  31),
    'm2':          np.linspace(1.0,   2.5,  31),
    'distance':    np.linspace(100.,  1000., 37),
    'a_1':         np.linspace(0.0,   1.0,  21),
    'a_2':         np.linspace(0.0,   1.0,  21),
    'tilt_1':      np.linspace(0.0,   3.2,  33),
    'tilt_2':      np.linspace(0.0,   3.2,  33),
    'psi':         np.linspace(0.0,   3.2,  33),
    'inclination': np.linspace(0.0,   3.2,  33),
    'phi_12':      np.linspace(0.0,   6.4,  33),
    'phi_jl':      np.linspace(0.0,   6.4,  33),
    'phic':        np.linspace(0.0,   6.4,  33),
    'phi':         np.linspace(-3.2,  3.2,  65),
    'dec':         np.linspace(-1.6,  1.6,  33),
    'cos_theta':   np.linspace(-1.0,  1.0,  41),
    'snr':         np.linspace(5.0,  50.0,  19),
    's1z':         np.linspace(-1.0,  1.0,  41),
    's2z':         np.linspace(-1.0,  1.0,  41),
    'chi1':        np.linspace(-1.0,  1.0,  41),
    'chi2':        np.linspace(-1.0,  1.0,  41),
}

VAR_LABELS: dict[str, str] = {
    'chirp_mass':  r'$\mathcal{M}_c\,[M_\odot]$',
    'mass_ratio':  r'$q$',
    'mass_1':      r'$m_1\,[M_\odot]$',
    'mass_2':      r'$m_2\,[M_\odot]$',
    'm1':          r'$m_1\,[M_\odot]$',
    'm2':          r'$m_2\,[M_\odot]$',
    'distance':    r'$d_L\,[\mathrm{Mpc}]$',
    'a_1':         r'$a_1$',
    'a_2':         r'$a_2$',
    'tilt_1':      r'$\theta_1\,[\mathrm{rad}]$',
    'tilt_2':      r'$\theta_2\,[\mathrm{rad}]$',
    'phi_12':      r'$\phi_{12}\,[\mathrm{rad}]$',
    'phi_jl':      r'$\phi_{JL}\,[\mathrm{rad}]$',
    'phic':        r'$\phi_c\,[\mathrm{rad}]$',
    'psi':         r'$\psi\,[\mathrm{rad}]$',
    'phi':         r'$\phi\,[\mathrm{rad}]$',
    'dec':         r'$\delta\,[\mathrm{rad}]$',
    'cos_theta':   r'$\cos\theta$',
    'inclination': r'$\iota\,[\mathrm{rad}]$',
    'snr':         r'SNR',
    's1z':         r'$s_{1z}$',
    's2z':         r'$s_{2z}$',
    'chi1':        r'$\chi_1$',
    'chi2':        r'$\chi_2$',
}


_H1L1_BASELINE = np.array([0.6953014731407166, -0.5535351634025574, -0.4584263563156128])


def _label(key: str) -> str:
    return VAR_LABELS.get(key, key)


def _sig_label(key: str) -> str:
    inner = _label(key).strip('$')
    return rf'$\sigma_{{{inner}}}$'


def _bins_for(key: str, data: np.ndarray) -> np.ndarray:
    if key in BINS:
        bins = BINS[key]
        if float(data.max()) < bins[0] or float(data.min()) > bins[-1]:
            lo, hi = float(data.min()), float(data.max())
            return np.linspace(lo, hi, len(bins))
        return bins
    lo, hi = float(data.min()), float(data.max())
    return np.linspace(lo, hi, 51)


def _snr_bins(snr_arr: np.ndarray) -> np.ndarray:
    a = int(np.floor(float(snr_arr.min())))
    b = int(np.ceil(float(snr_arr.max())))
    return np.arange(a, b + 1, dtype=float)


def _snr_xticks(ax, a: int, b: int) -> None:
    # ticks at every multiple of 5 within [a, b], plus the min and max
    mults = list(range(int(np.ceil(a / 5.0)) * 5, b + 1, 5))
    ticks = sorted(set(mults) | {int(a), int(b)})
    ax.set_xticks(ticks)
    ax.set_xlim(a - 0.5, b + 0.5)


def _hist_outline(ax, data: np.ndarray, bins, **kwargs) -> None:
    counts, edges = np.histogram(data, bins=bins)
    ax.stairs(counts, edges, **kwargs)


def _binned_stats(x, y, bins):
    centers = 0.5 * (bins[:-1] + bins[1:])
    med = np.full(len(centers), np.nan)
    q16 = np.full(len(centers), np.nan)
    q84 = np.full(len(centers), np.nan)
    mae = np.full(len(centers), np.nan)
    for i, (lo, hi) in enumerate(zip(bins[:-1], bins[1:])):
        mask = (x >= lo) & (x < hi)
        if mask.sum() < 2:
            continue
        yb = y[mask]
        med[i] = np.median(yb)
        q16[i] = np.percentile(yb, 16)
        q84[i] = np.percentile(yb, 84)
        mae[i] = np.mean(np.abs(yb))
    return centers, med, q16, q84, mae


def _mc_q_to_m1_m2(mc: np.ndarray, q: np.ndarray):
    m1 = mc * (1.0 + q) ** 0.2 / q ** 0.6
    return m1, q * m1


def _propagate_m1_m2_uncertainty(mc, q, sig_mc, sig_q):
    m1, m2 = _mc_q_to_m1_m2(mc, q)
    dm1_dmc = m1 / mc
    dm1_dq  = m1 * (1.0 / (5.0 * (1.0 + q)) - 3.0 / (5.0 * q))
    dm2_dmc = m2 / mc
    dm2_dq  = m1 + q * dm1_dq
    sig_m1 = np.sqrt((dm1_dmc * sig_mc) ** 2 + (dm1_dq * sig_q) ** 2)
    sig_m2 = np.sqrt((dm2_dmc * sig_mc) ** 2 + (dm2_dq * sig_q) ** 2)
    return sig_m1, sig_m2


def _angles_to_vec(dec: np.ndarray, phi: np.ndarray) -> np.ndarray:
    x = np.cos(dec) * np.cos(phi)
    y = np.cos(dec) * np.sin(phi)
    z = np.sin(dec)
    return np.stack([x, y, z], axis=-1)


def _vec_to_angles(vec: np.ndarray):
    dec = np.arcsin(np.clip(vec[:, 2], -1.0, 1.0))
    phi = np.arctan2(vec[:, 1], vec[:, 0])
    return dec, phi


class PlotParamEstCallback(Callback):
    def __init__(self, save_dir: str | Path = ''):
        super().__init__()
        self._save_dir_override = Path(save_dir) if save_dir else None
        self.save_dir = self._save_dir_override or Path('.')

        self.target_variables:   list[str] = []
        self.observed_variables: list[str] = []
        self.normalize_variables: bool = False
        self.var_scales: dict[str, tuple[float, float]] = {}
        self.normalize_range: tuple[float, float] = (-1.0, 1.0)

        self._y_true:     list[torch.Tensor] = []
        self._y_pred:     list[torch.Tensor] = []
        self._y_sigma:    list[torch.Tensor] = []
        self._snr:        list[torch.Tensor] = []
        self._y_pred_bg:  list[torch.Tensor] = []
        self._y_sigma_bg: list[torch.Tensor] = []

    def on_test_start(self, trainer, pl_module):
        self._y_true.clear()
        self._y_pred.clear()
        self._y_sigma.clear()
        self._snr.clear()
        self._y_pred_bg.clear()
        self._y_sigma_bg.clear()

        if self._save_dir_override:
            self.save_dir = self._save_dir_override
        elif hasattr(trainer.logger, 'save_dir') and trainer.logger.save_dir is not None:
            exp = getattr(trainer.logger, 'experiment', None)
            run_id = (
                getattr(exp, 'id', None)
                or getattr(trainer.logger, 'version', None)
                or getattr(trainer.logger, 'id', None)
            )
            if run_id is not None:
                base = Path(trainer.logger.save_dir)
                project = getattr(exp, 'project', None)
                if project:
                    base = base / project
                self.save_dir = base / run_id / 'plots'

        dm = trainer.datamodule
        cfg = getattr(dm, 'cfg', None)
        if cfg is not None:
            self.target_variables    = list(cfg.target_variables)
            self.observed_variables  = list(cfg.observed_variables) if hasattr(cfg, 'observed_variables') else []
            self.normalize_variables = cfg.normalize_variables
            self.normalize_range     = tuple(cfg.normalize_range)
            if self.normalize_variables:
                self.var_scales = dict(dm.test_dataset.var_scales)
        else:
            # RegressionTimeDomainDataset: target names come from hparams, and the
            # model returns physical (already un-normalized) values, so the callback
            # does no normalization of its own.
            self.target_variables    = list(getattr(dm.hparams, 'target_parameters', ['chirp_mass']))
            self.observed_variables  = []
            self.normalize_variables = False

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if outputs is None:
            return
        self._y_true.append(outputs['y_true'])
        self._y_pred.append(outputs['y_pred'])
        if 'y_sigma' in outputs:
            self._y_sigma.append(outputs['y_sigma'])
        if 'snr' in outputs:
            self._snr.append(outputs['snr'])
        if 'y_pred_bg' in outputs:
            self._y_pred_bg.append(outputs['y_pred_bg'])
        if 'y_sigma_bg' in outputs:
            self._y_sigma_bg.append(outputs['y_sigma_bg'])

    def on_test_epoch_end(self, trainer, pl_module):
        if not self._y_true:
            return

        y_true = torch.cat(self._y_true).numpy()
        y_pred = torch.cat(self._y_pred).numpy()
        y_sigma = torch.cat(self._y_sigma).numpy() if self._y_sigma else None
        snr = torch.cat(self._snr).numpy().reshape(-1) if self._snr else None
        y_pred_bg = torch.cat(self._y_pred_bg).numpy() if self._y_pred_bg else None
        y_sigma_bg = torch.cat(self._y_sigma_bg).numpy() if self._y_sigma_bg else None

        if self.normalize_variables and self.var_scales:
            y_true = self._unnormalize_values(y_true, self.target_variables)
            y_pred_for_unnorm = y_pred if y_pred.shape[1] == len(self.target_variables) else y_pred
            if y_pred.shape[1] == len(self.target_variables):
                y_pred = self._unnormalize_values(y_pred, self.target_variables)
            if y_sigma is not None:
                y_sigma = self._unnormalize_sigma(y_sigma, self.target_variables)

        self.save_dir.mkdir(parents=True, exist_ok=True)

        is_sky = 'dec' in self.target_variables and 'phi' in self.target_variables
        if is_sky:
            self._run_sky(y_true, y_pred, snr)
        else:
            self._run_param_est(y_true, y_pred, y_sigma, snr)
            if y_sigma_bg is not None:
                self._run_background(y_pred_bg, y_sigma_bg)
                if y_sigma is not None:
                    self._run_detection(y_sigma, y_sigma_bg)

    def _unnormalize_values(self, arr: np.ndarray, variables: list[str]) -> np.ndarray:
        lo, hi = self.normalize_range
        out = arr.copy()
        for i, var in enumerate(variables):
            if var in self.var_scales:
                vmin, vmax = self.var_scales[var]
                out[:, i] = vmin + (vmax - vmin) * (arr[:, i] - lo) / (hi - lo)
        return out

    def _unnormalize_sigma(self, arr: np.ndarray, variables: list[str]) -> np.ndarray:
        lo, hi = self.normalize_range
        out = arr.copy()
        for i, var in enumerate(variables):
            if var in self.var_scales:
                vmin, vmax = self.var_scales[var]
                out[:, i] = arr[:, i] * (vmax - vmin) / (hi - lo)
        return out

    def _get_snr(self, z_observed):
        if z_observed is not None and 'snr' in self.observed_variables:
            return z_observed[:, list(self.observed_variables).index('snr')]
        return None

    def _run_param_est(self, y_true, y_pred, y_sigma, snr):
        vars_ = self.target_variables
        true_d  = {v: y_true[:, i] for i, v in enumerate(vars_)}
        pred_d  = {v: y_pred[:, i] for i, v in enumerate(vars_)}
        sigma_d = {v: y_sigma[:, i] for i, v in enumerate(vars_)} if y_sigma is not None else {}

        vars_ext = list(vars_)
        if 'chirp_mass' in vars_ and 'mass_ratio' in vars_:
            mc_t, q_t = true_d['chirp_mass'], true_d['mass_ratio']
            mc_p, q_p = pred_d['chirp_mass'], pred_d['mass_ratio']
            m1t, m2t = _mc_q_to_m1_m2(mc_t, q_t)
            m1p, m2p = _mc_q_to_m1_m2(mc_p, q_p)
            for key, tr, pr in [('m1', m1t, m1p), ('m2', m2t, m2p)]:
                true_d[key] = tr
                pred_d[key] = pr
            if sigma_d:
                s_mc, s_q = sigma_d['chirp_mass'], sigma_d['mass_ratio']
                s_m1, s_m2 = _propagate_m1_m2_uncertainty(mc_p, q_p, s_mc, s_q)
                sigma_d['m1'] = s_m1
                sigma_d['m2'] = s_m2
            vars_ext = vars_ + ['m1', 'm2']

        self._save_csv_param_est(y_true, y_pred, y_sigma, snr)

        plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False})
        figs, names = [], []

        def stats_text(t, p, s=None):
            res = p - t
            den = np.where(np.abs(t) > 1e-12, np.abs(t), 1.0)
            rel = np.abs(res / den)
            L = [f"N      = {len(t)}",
                 f"MAE    = {np.mean(np.abs(res)):.4f}",
                 f"RMSE   = {np.sqrt(np.mean(res ** 2)):.4f}",
                 f"bias   = {np.mean(res):+.4f}",
                 f"medres = {np.median(res):+.4f}",
                 f"stdres = {np.std(res):.4f}",
                 (f"R      = {np.corrcoef(t, p)[0, 1]:.4f}" if len(t) > 1 else "R = n/a"),
                 f"med|rel|= {100 * np.median(rel):.2f}%"]
            for c in (0.01, 0.02, 0.05, 0.10):
                L.append(f"<{int(c * 100):>2d}%   = {100 * np.mean(rel < c):.1f}%")
            if s is not None:
                z = res / (np.asarray(s) + 1e-12)
                L += [f"<sig>  = {np.mean(s):.4f}",
                      f"z mean = {np.mean(z):+.2f}",
                      f"z std  = {np.std(z):.2f}",
                      f"|z|<1  = {100 * np.mean(np.abs(z) < 1):.1f}%",
                      f"|z|<2  = {100 * np.mean(np.abs(z) < 2):.1f}%"]
            return "\n".join(L)

        def pred_vs_true_ax(ax, t, p, s, lims, edges, title):
            centers = 0.5 * (edges[:-1] + edges[1:])
            med = np.full(len(centers), np.nan)
            q16 = np.full(len(centers), np.nan); q84 = np.full(len(centers), np.nan)
            q2 = np.full(len(centers), np.nan); q97 = np.full(len(centers), np.nan)
            for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
                m = (t >= lo) & (t < hi)
                if m.sum() < 2:
                    continue
                pb = p[m]
                med[i] = np.median(pb)
                q16[i], q84[i] = np.percentile(pb, [16, 84])
                q2[i], q97[i] = np.percentile(pb, [2.5, 97.5])
            ok = ~np.isnan(med)
            ax.fill_between(centers[ok], q2[ok], q97[ok], color='steelblue', alpha=0.18, label='95%')
            ax.fill_between(centers[ok], q16[ok], q84[ok], color='steelblue', alpha=0.35, label='68%')
            ax.plot(centers[ok], med[ok], color='steelblue', lw=1.6, label='Median')
            ax.plot(lims, lims, color='gray', ls=':', lw=1.2)
            ax.set_xlim(lims); ax.set_ylim(lims); ax.set_aspect('equal', adjustable='box')
            ax.set_title(title, fontsize=10)
            ax.text(0.03, 0.97, stats_text(t, p, s), transform=ax.transAxes, ha='left', va='top',
                    fontsize=6.5, family='monospace',
                    bbox=dict(boxstyle='round', fc='white', ec='0.7', alpha=0.85))
            ax.legend(frameon=False, fontsize=8, loc='lower right')

        for v in vars_ext:
            tr = np.asarray(true_d[v]); pr = np.asarray(pred_d[v])
            s_all = np.asarray(sigma_d[v]) if (sigma_d and v in sigma_d) else None
            lims = [float(tr.min()), float(tr.max())]
            edges = _bins_for(v, tr)

            # 1) predicted vs true over the full test SNR range (8-50).
            if snr is not None:
                blo, bhi = int(np.floor(snr.min())), int(np.ceil(snr.max()))
                bands = [(f'SNR {blo}-{bhi}', np.ones(len(tr), bool), f'snr{blo}-{bhi}')]
            else:
                bands = [('all', np.ones(len(tr), bool), 'all')]
            for blab, bm, suffix in bands:
                if bm.sum() < 2:
                    continue
                ss = s_all[bm] if s_all is not None else None
                fig, ax = plt.subplots(figsize=(5.5, 5.5))
                pred_vs_true_ax(ax, tr[bm], pr[bm], ss, lims, edges, f'{_label(v)} - {blab}')
                ax.set_xlabel('True ' + _label(v)); ax.set_ylabel('Pred ' + _label(v))
                fig.tight_layout()
                figs.append(fig); names.append(f'{v}_pred_vs_true_{suffix}')

            # 3) z-score histogram (counts) + fitted Gaussian + stats.
            if s_all is not None:
                z = (pr - tr) / (s_all + 1e-12); z = z[np.isfinite(z)]
                fig, ax = plt.subplots(figsize=(6, 4.5))
                _c, ze, _ = ax.hist(z, bins=60, color='steelblue', alpha=0.6, label='z-score')
                mu, sd = float(np.mean(z)), float(np.std(z))
                xs = np.linspace(ze[0], ze[-1], 400); bw = ze[1] - ze[0]
                ax.plot(xs, z.size * bw / (sd * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((xs - mu) / sd) ** 2),
                        'r-', lw=1.8, label=f'fit N({mu:.2f}, {sd:.2f})')
                ax.plot(xs, z.size * bw / np.sqrt(2 * np.pi) * np.exp(-0.5 * xs ** 2),
                        'k--', lw=1.0, alpha=0.7, label='N(0, 1)')
                txt = (f"mean = {mu:+.2f}\nstd  = {sd:.2f}\n"
                       f"|z|<1 = {100 * np.mean(np.abs(z) < 1):.1f}% (68)\n"
                       f"|z|<2 = {100 * np.mean(np.abs(z) < 2):.1f}% (95)\n"
                       f"|z|<3 = {100 * np.mean(np.abs(z) < 3):.1f}% (99.7)")
                ax.text(0.03, 0.97, txt, transform=ax.transAxes, ha='left', va='top',
                        fontsize=7, family='monospace',
                        bbox=dict(boxstyle='round', fc='white', ec='0.7', alpha=0.85))
                ax.set_xlabel(f'{_label(v)} z-score'); ax.set_ylabel('Counts')
                ax.legend(frameon=False, fontsize=8, loc='upper right')
                figs.append(fig); names.append(f'{v}_zscore')

        # 2) fraction within a relative-error cutoff, per SNR bin.
        if snr is not None:
            snr_bins = _snr_bins(snr)
            snr_a, snr_b = int(snr_bins[0]), int(snr_bins[-1])
            snr_centers = 0.5 * (snr_bins[:-1] + snr_bins[1:])
            cuts = [0.01, 0.02, 0.05, 0.10]
            cmap = plt.cm.viridis
            for v in vars_ext:
                tr = np.asarray(true_d[v]); pr = np.asarray(pred_d[v])
                denom = np.where(np.abs(tr) > 1e-8, np.abs(tr), 1.0)
                rel = np.abs((pr - tr) / denom)
                fig, ax = plt.subplots(figsize=(6.5, 4))
                for j, cut in enumerate(cuts):
                    frac = np.full(len(snr_centers), np.nan)
                    for i, (lo, hi) in enumerate(zip(snr_bins[:-1], snr_bins[1:])):
                        m = (snr >= lo) & (snr < hi)
                        if m.sum() > 0:
                            frac[i] = 100.0 * (rel[m] < cut).mean()
                    ok = ~np.isnan(frac)
                    ax.plot(snr_centers[ok], frac[ok], marker='o', ms=3,
                            color=cmap(j / (len(cuts) - 1)), label=f'{int(cut * 100)}%')
                ax.set_xlabel(_label('snr')); ax.set_ylabel('% within cutoff')
                ax.set_ylim(0, 100); ax.set_title(_label(v))
                ax.legend(frameon=False, fontsize=9, title='|rel. err|', loc='lower right')
                _snr_xticks(ax, snr_a, snr_b)
                figs.append(fig); names.append(f'{v}_frac_within_vs_snr')

        self._save_figs(figs, names)

    def _run_background(self, y_pred_bg, y_sigma_bg):
        """Background-only (noise) predictions: CSV + predicted-value/sigma distributions."""
        vars_ = self.target_variables
        cols = [f'{v}_pred' for v in vars_] + [f'sigma_{v}_pred' for v in vars_]
        pd.DataFrame(
            np.concatenate([y_pred_bg, y_sigma_bg], axis=1), columns=cols
        ).to_csv(self.save_dir / 'param_est_results_bkg.csv', index=False)
        figs, names = [], []
        for i, v in enumerate(vars_):
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.hist(y_pred_bg[:, i], bins=60, color='tomato', alpha=0.7)
            ax.set_xlabel(f'background pred {_label(v)}'); ax.set_ylabel('Counts')
            figs.append(fig); names.append(f'{v}_pred_bkg')
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.hist(y_sigma_bg[:, i], bins=60, color='tomato', alpha=0.7)
            ax.set_xlabel(f'background {_sig_label(v)}'); ax.set_ylabel('Counts')
            figs.append(fig); names.append(f'{v}_sigma_bkg')
        self._save_figs(figs, names)

    def _run_detection(self, y_sigma, y_sigma_bg):
        """Predicted uncertainty as a detection statistic (lower sigma => signal)."""
        vars_ = self.target_variables
        figs, names = [], []
        for i, v in enumerate(vars_):
            s_sig = np.asarray(y_sigma[:, i]); s_bg = np.asarray(y_sigma_bg[:, i])

            # 1) overlapping, semi-transparent sigma histograms: signal vs background
            lo = float(min(s_sig.min(), s_bg.min()))
            hi = float(max(np.percentile(s_sig, 99.5), np.percentile(s_bg, 99.5)))
            bins = np.linspace(lo, hi, 60)
            fig, ax = plt.subplots(figsize=(6.5, 4))
            ax.hist(s_sig, bins=bins, density=True, color='steelblue', alpha=0.5,
                    label=f'signal (N={len(s_sig)})')
            ax.hist(s_bg, bins=bins, density=True, color='tomato', alpha=0.5,
                    label=f'background (N={len(s_bg)})')
            ax.set_xlabel(_sig_label(v)); ax.set_ylabel('density')
            ax.legend(frameon=False, fontsize=9)
            figs.append(fig); names.append(f'{v}_sigma_sig_vs_bkg')

            # 2) ROC from a sigma threshold (ranking score = -sigma)
            scores = np.concatenate([-s_sig, -s_bg])
            labels = np.concatenate([np.ones(len(s_sig)), np.zeros(len(s_bg))])
            order = np.argsort(-scores, kind='mergesort')
            lab = labels[order]
            tpr = np.concatenate([[0.0], np.cumsum(lab) / max(len(s_sig), 1)])
            fpr = np.concatenate([[0.0], np.cumsum(1 - lab) / max(len(s_bg), 1)])
            auc = float(np.trapz(tpr, fpr))
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.plot(fpr, tpr, color='steelblue', lw=1.8, label=f'AUC = {auc:.4f}')
            ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.6)
            ax.set_xlim(0, 1); ax.set_ylim(0, 1)
            ax.set_xlabel('False positive rate'); ax.set_ylabel('True positive rate')
            ax.set_title(f'{_label(v)} sigma-based ROC')
            ax.legend(frameon=False, loc='lower right')
            figs.append(fig); names.append(f'{v}_roc_sigma')
        self._save_figs(figs, names)

    def _save_csv_param_est(self, y_true, y_pred, y_sigma, snr):
        vars_ = self.target_variables
        residual = y_pred - y_true
        cols = (
            [f'{v}_true'       for v in vars_] +
            [f'{v}_pred'       for v in vars_] +
            ([f'sigma_{v}_pred' for v in vars_] if y_sigma is not None else []) +
            [f'{v}_residual'   for v in vars_]
        )
        ordered = [y_true, y_pred]
        if y_sigma is not None:
            zscore = residual / (y_sigma + 1e-12)
            ordered.append(y_sigma)
            cols += [f'{v}_zscore' for v in vars_]
        ordered.append(residual)
        if y_sigma is not None:
            ordered.append(zscore)
        if snr is not None:
            cols += ['snr']
            ordered.append(np.asarray(snr).reshape(-1, 1))
        pd.DataFrame(
            np.concatenate(ordered, axis=1),
            columns=cols,
        ).to_csv(self.save_dir / 'param_est_results.csv', index=False)

    def _run_sky(self, y_true, y_pred, snr):
        dec_true = y_true[:, self.target_variables.index('dec')]
        phi_true = y_true[:, self.target_variables.index('phi')]

        if y_pred.shape[1] == 3:
            norm = np.linalg.norm(y_pred, axis=1, keepdims=True)
            y_pred_norm = y_pred / (norm + 1e-8)
            dec_pred, phi_pred = _vec_to_angles(y_pred_norm)
            cos_theta_pred = y_pred_norm @ _H1L1_BASELINE
        else:
            dec_pred = y_pred[:, self.target_variables.index('dec')]
            phi_pred = y_pred[:, self.target_variables.index('phi')]
            v_pred = _angles_to_vec(dec_pred, phi_pred)
            cos_theta_pred = v_pred @ _H1L1_BASELINE

        v_true = _angles_to_vec(dec_true, phi_true)
        cos_theta_true = v_true @ _H1L1_BASELINE
        ring_dist = np.abs(cos_theta_pred - cos_theta_true)

        v_pred_for_angle = _angles_to_vec(dec_pred, phi_pred)
        cos_sim = np.clip((v_pred_for_angle * v_true).sum(axis=1), -1 + 1e-6, 1 - 1e-6)
        angular_error_deg = np.degrees(np.arccos(cos_sim))

        self._save_csv_sky(dec_true, phi_true, dec_pred, phi_pred,
                           cos_theta_true, cos_theta_pred, ring_dist, angular_error_deg, snr)

        # Minimal notification
        print(f'Saved {len(dec_true)} rows → {self.save_dir / "sky_loc_results.csv"}')

    def _save_csv_sky(self, dec_true, phi_true, dec_pred, phi_pred,
                      cos_theta_true, cos_theta_pred, ring_dist, angular_error_deg, snr):
        data = {
            'dec_true':           dec_true,
            'phi_true':           phi_true,
            'dec_pred':           dec_pred,
            'phi_pred':           phi_pred,
            'dec_residual':       dec_pred - dec_true,
            'phi_residual':       phi_pred - phi_true,
            'cos_theta_true':     cos_theta_true,
            'cos_theta_pred':     cos_theta_pred,
            'ring_distance':      ring_dist,
            'angular_error_deg':  angular_error_deg,
        }
        if snr is not None:
            data['snr'] = snr
        pd.DataFrame(data).to_csv(self.save_dir / 'sky_loc_results.csv', index=False)
        print(f'Saved {len(dec_true)} rows → {self.save_dir / "sky_loc_results.csv"}')

    def _save_figs(self, figs, names):
        for fig, name in zip(figs, names):
            out = self.save_dir / f'{name}.png'
            fig.savefig(out, dpi=400, bbox_inches='tight')
            plt.close(fig)
            print(f'Saved {out}')


# (No external imports — use the local TestPredictionsCallback implementation above.)
