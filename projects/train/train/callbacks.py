import io
import json
import os
import shutil
from typing import Optional, Union, Literal
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py
import s3fs
import torch
from botocore.exceptions import ClientError, ConnectTimeoutError
from lightning import pytorch as pl
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.utilities import grad_norm

BOTO_RETRY_EXCEPTIONS = (ClientError, ConnectTimeoutError)


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
        # wandb decides where to put its run data with
        # os.access(save_dir, W_OK), which is False for a directory that
        # does not exist yet. Lightning creates save_dir, but only after
        # the logger is built, so on a fresh run wandb would report
        # "wasn't writable", fall back to the node's /tmp, and lose the
        # local run data when the job ends. Create it first.
        Path(save_dir).mkdir(parents=True, exist_ok=True)
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


class WandbSaveConfig(pl.cli.SaveConfigCallback):
    """
    Override of `lightning.pytorch.cli.SaveConfigCallback` for use with WandB
    to ensure all the hyperparameters are logged to the WandB dashboard.
    """

    def get_wandb_logger(self, trainer) -> Optional[WandbLogger]:
        for logger in trainer.loggers:
            if isinstance(logger, WandbLogger):
                return logger

    def save_config(self, trainer, _, stage) -> None:
        wandb_logger = self.get_wandb_logger(trainer)
        if stage == "fit" and wandb_logger is not None:
            # pop off unecessary trainer args
            config = self.config.as_dict()
            config.pop("trainer")
            wandb_logger.experiment.config.update(config)


class ModelCheckpoint(pl.callbacks.ModelCheckpoint):
    def on_train_end(self, trainer, pl_module):
        torch.cuda.empty_cache()
        module = pl_module.__class__.load_from_checkpoint(
            self.best_model_path, arch=pl_module.model, metric=pl_module.metric
        )

        device = pl_module.device
        # Handle the case of loading training waveforms from disk
        if trainer.datamodule.waveforms_from_disk:
            [X], (waveforms, params) = next(iter(trainer.train_dataloader))
            X = X.to(device)
            waveforms = waveforms.to(device)
            params = {k: v.to(device) for k, v in params.items()}
        else:
            [X] = next(iter(trainer.train_dataloader))
            X = X.to(device)
            waveforms, params = trainer.datamodule.waveform_sampler.sample(X)
        X, y, _ = trainer.datamodule.inject(X, waveforms, params)
        if isinstance(X, tuple):
            X = tuple(i.cpu() for i in X)
        else:
            X = X.cpu()
        trace = torch.jit.trace(module.model.to("cpu"), X)

        save_dir = trainer.logger.save_dir
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
            save_dir = trainer.logger.save_dir

            # build training batch by hand
            # Handle the case of loading training waveforms from disk
            if trainer.datamodule.waveforms_from_disk:
                [X], (waveforms, params) = next(iter(trainer.train_dataloader))
                X = X.to(device)
                waveforms = waveforms.to(device)
                params = {k: v.to(device) for k, v in params.items()}
            else:
                [X] = next(iter(trainer.train_dataloader))
                X = X.to(device)
                waveforms, params = trainer.datamodule.waveform_sampler.sample(
                    X
                )
            X, y, _ = trainer.datamodule.inject(X, waveforms, params)
            # If X is not a tuple, make it one for consistency
            # of format for saving to file below
            if not isinstance(X, tuple):
                X = (X,)

            # build val batch by hand
            [background, _, _], [signals, params] = next(
                iter(trainer.datamodule.val_dataloader())
            )
            background = background.to(device)
            signals = signals.to(device)
            params = {k: v.to(device) for k, v in params.items()}
            X_bg, X_inj, _ = trainer.datamodule.build_val_batches(
                background, signals, params
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


BINS: dict[str, np.ndarray] = {
    "chirp_mass": np.linspace(0.5, 3.0, 51),
    "mass_ratio": np.linspace(0.0, 1.0, 51),
    "mass_1": np.linspace(1.0, 2.5, 31),
    "mass_2": np.linspace(1.0, 2.5, 31),
    "m1": np.linspace(1.0, 2.5, 31),
    "m2": np.linspace(1.0, 2.5, 31),
    "distance": np.linspace(100.0, 1000.0, 37),
    "a_1": np.linspace(0.0, 1.0, 21),
    "a_2": np.linspace(0.0, 1.0, 21),
    "tilt_1": np.linspace(0.0, 3.2, 33),
    "tilt_2": np.linspace(0.0, 3.2, 33),
    "psi": np.linspace(0.0, 3.2, 33),
    "inclination": np.linspace(0.0, 3.2, 33),
    "phi_12": np.linspace(0.0, 6.4, 33),
    "phi_jl": np.linspace(0.0, 6.4, 33),
    "phic": np.linspace(0.0, 6.4, 33),
    "phi": np.linspace(-3.2, 3.2, 65),
    "dec": np.linspace(-1.6, 1.6, 33),
    "cos_theta": np.linspace(-1.0, 1.0, 41),
    "snr": np.linspace(5.0, 50.0, 19),
    "s1z": np.linspace(-1.0, 1.0, 41),
    "s2z": np.linspace(-1.0, 1.0, 41),
    "chi1": np.linspace(-1.0, 1.0, 41),
    "chi2": np.linspace(-1.0, 1.0, 41),
}

VAR_LABELS: dict[str, str] = {
    "chirp_mass": r"$\mathcal{M}_c\,[M_\odot]$",
    "mass_ratio": r"$q$",
    "mass_1": r"$m_1\,[M_\odot]$",
    "mass_2": r"$m_2\,[M_\odot]$",
    "m1": r"$m_1\,[M_\odot]$",
    "m2": r"$m_2\,[M_\odot]$",
    "distance": r"$d_L\,[\mathrm{Mpc}]$",
    "a_1": r"$a_1$",
    "a_2": r"$a_2$",
    "tilt_1": r"$\theta_1\,[\mathrm{rad}]$",
    "tilt_2": r"$\theta_2\,[\mathrm{rad}]$",
    "phi_12": r"$\phi_{12}\,[\mathrm{rad}]$",
    "phi_jl": r"$\phi_{JL}\,[\mathrm{rad}]$",
    "phic": r"$\phi_c\,[\mathrm{rad}]$",
    "psi": r"$\psi\,[\mathrm{rad}]$",
    "phi": r"$\phi\,[\mathrm{rad}]$",
    "dec": r"$\delta\,[\mathrm{rad}]$",
    "cos_theta": r"$\cos\theta$",
    "inclination": r"$\iota\,[\mathrm{rad}]$",
    "snr": r"SNR",
    "s1z": r"$s_{1z}$",
    "s2z": r"$s_{2z}$",
    "chi1": r"$\chi_1$",
    "chi2": r"$\chi_2$",
}


_H1L1_BASELINE = np.array(
    [0.6953014731407166, -0.5535351634025574, -0.4584263563156128]
)


def _label(key: str) -> str:
    return VAR_LABELS.get(key, key)


_PLOT_CFG_PATH = (
    "/n/holystore01/LABS/iaifi_lab/Lab/kyoon/aframe/dev/configs/"
    "plot_configs.json"
)
try:
    with open(_PLOT_CFG_PATH) as _f:
        _PLOT_SYMBOLS = {
            k: v["label"]
            for k, v in json.load(_f)["parameters"].items()
            if "label" in v
        }
except (OSError, ValueError, KeyError):
    _PLOT_SYMBOLS = {}


def _sym(key: str) -> str:
    """Units-free symbol (titles, z-score, sigma subscript). Taken straight
    from plot_configs.json's ``label`` -- no unit stripping needed."""
    return _PLOT_SYMBOLS.get(key, _label(key))


def _hat(key: str) -> str:
    """Predicted-quantity symbol: a hat over the base symbol."""
    sym = _sym(key)
    if r"\mathcal{M}" in sym:
        return sym.replace(r"\mathcal{M}", r"\hat{\mathcal{M}}")
    return "$\\hat{" + sym.strip("$") + "}$"


def _sig_label(key: str) -> str:
    inner = _sym(key).strip("$")
    return rf"$\sigma_{{{inner}}}$"


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
    for i, (lo, hi) in enumerate(zip(bins[:-1], bins[1:], strict=False)):
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
    m1 = mc * (1.0 + q) ** 0.2 / q**0.6
    return m1, q * m1


def _propagate_m1_m2_uncertainty(mc, q, sig_mc, sig_q):
    m1, m2 = _mc_q_to_m1_m2(mc, q)
    dm1_dmc = m1 / mc
    dm1_dq = m1 * (1.0 / (5.0 * (1.0 + q)) - 3.0 / (5.0 * q))
    dm2_dmc = m2 / mc
    dm2_dq = m1 + q * dm1_dq
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
    def __init__(self, save_dir: str | Path = "", dist_label: str = ""):
        super().__init__()
        self._save_dir_override = Path(save_dir) if save_dir else None
        self.save_dir = self._save_dir_override or Path(".")
        # e.g. "powerlaw" / "uniform" -- appended to the pred-vs-true title
        self.dist_label = dist_label

        self.target_variables: list[str] = []
        self.observed_variables: list[str] = []
        self.normalize_variables: bool = False
        self.var_scales: dict[str, tuple[float, float]] = {}
        self.normalize_range: tuple[float, float] = (-1.0, 1.0)

        self._y_true: list[torch.Tensor] = []
        self._y_pred: list[torch.Tensor] = []
        self._y_sigma: list[torch.Tensor] = []
        self._snr: list[torch.Tensor] = []
        self._y_pred_bg: list[torch.Tensor] = []
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
        elif (
            hasattr(trainer.logger, "save_dir")
            and trainer.logger.save_dir is not None
        ):
            exp = getattr(trainer.logger, "experiment", None)
            run_id = (
                getattr(exp, "id", None)
                or getattr(trainer.logger, "version", None)
                or getattr(trainer.logger, "id", None)
            )
            if run_id is not None:
                base = Path(trainer.logger.save_dir)
                project = getattr(exp, "project", None)
                if project:
                    base = base / project
                self.save_dir = base / run_id / "plots"

        dm = trainer.datamodule
        cfg = getattr(dm, "cfg", None)
        if cfg is not None:
            self.target_variables = list(cfg.target_variables)
            self.observed_variables = (
                list(cfg.observed_variables)
                if hasattr(cfg, "observed_variables")
                else []
            )
            self.normalize_variables = cfg.normalize_variables
            self.normalize_range = tuple(cfg.normalize_range)
            if self.normalize_variables:
                self.var_scales = dict(dm.test_dataset.var_scales)
        else:
            # target names come from hparams, and the model returns
            # physical (already un-normalized) values, so the callback
            # does no normalization of its own.
            self.target_variables = list(
                getattr(dm.hparams, "target_parameters", ["chirp_mass"])
            )
            self.observed_variables = []
            self.normalize_variables = False

    def on_test_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        if outputs is None:
            return
        self._y_true.append(outputs["y_true"])
        self._y_pred.append(outputs["y_pred"])
        if "y_sigma" in outputs:
            self._y_sigma.append(outputs["y_sigma"])
        if "snr" in outputs:
            self._snr.append(outputs["snr"])
        if "y_pred_bg" in outputs:
            self._y_pred_bg.append(outputs["y_pred_bg"])
        if "y_sigma_bg" in outputs:
            self._y_sigma_bg.append(outputs["y_sigma_bg"])

    def on_test_epoch_end(self, trainer, pl_module):
        if not self._y_true:
            return

        y_true = torch.cat(self._y_true).numpy()
        y_pred = torch.cat(self._y_pred).numpy()
        y_sigma = torch.cat(self._y_sigma).numpy() if self._y_sigma else None
        snr = torch.cat(self._snr).numpy().reshape(-1) if self._snr else None
        y_pred_bg = (
            torch.cat(self._y_pred_bg).numpy() if self._y_pred_bg else None
        )
        y_sigma_bg = (
            torch.cat(self._y_sigma_bg).numpy() if self._y_sigma_bg else None
        )

        if self.normalize_variables and self.var_scales:
            y_true = self._unnormalize_values(y_true, self.target_variables)
            if y_pred.shape[1] == len(self.target_variables):
                y_pred = self._unnormalize_values(
                    y_pred, self.target_variables
                )
            if y_sigma is not None:
                y_sigma = self._unnormalize_sigma(
                    y_sigma, self.target_variables
                )
            if y_pred_bg is not None and y_pred_bg.shape[1] == len(
                self.target_variables
            ):
                y_pred_bg = self._unnormalize_values(
                    y_pred_bg, self.target_variables
                )
            if y_sigma_bg is not None:
                y_sigma_bg = self._unnormalize_sigma(
                    y_sigma_bg, self.target_variables
                )

        self.save_dir.mkdir(parents=True, exist_ok=True)

        is_sky = (
            "dec" in self.target_variables and "phi" in self.target_variables
        )
        if is_sky:
            self._run_sky(y_true, y_pred, snr)
        else:
            self._run_param_est(
                y_true, y_pred, y_sigma, snr, y_pred_bg, y_sigma_bg
            )
            if y_sigma_bg is not None:
                self._run_background(y_pred_bg, y_sigma_bg)
                if y_sigma is not None:
                    self._run_detection(y_sigma, y_sigma_bg)

    def _unnormalize_values(
        self, arr: np.ndarray, variables: list[str]
    ) -> np.ndarray:
        lo, hi = self.normalize_range
        out = arr.copy()
        for i, var in enumerate(variables):
            if var in self.var_scales:
                vmin, vmax = self.var_scales[var]
                out[:, i] = vmin + (vmax - vmin) * (arr[:, i] - lo) / (hi - lo)
        return out

    def _unnormalize_sigma(
        self, arr: np.ndarray, variables: list[str]
    ) -> np.ndarray:
        lo, hi = self.normalize_range
        out = arr.copy()
        for i, var in enumerate(variables):
            if var in self.var_scales:
                vmin, vmax = self.var_scales[var]
                out[:, i] = arr[:, i] * (vmax - vmin) / (hi - lo)
        return out

    def _get_snr(self, z_observed):
        if z_observed is not None and "snr" in self.observed_variables:
            return z_observed[:, list(self.observed_variables).index("snr")]
        return None

    def _run_param_est(  # noqa: C901
        self, y_true, y_pred, y_sigma, snr, y_pred_bg=None, y_sigma_bg=None
    ):
        vars_ = self.target_variables
        true_d = {v: y_true[:, i] for i, v in enumerate(vars_)}
        pred_d = {v: y_pred[:, i] for i, v in enumerate(vars_)}
        sigma_d = (
            {v: y_sigma[:, i] for i, v in enumerate(vars_)}
            if y_sigma is not None
            else {}
        )
        # background (noise-only) predictions on the same events -- for the
        # bkg-only pred-vs-true panel
        pred_bg_d = (
            {v: y_pred_bg[:, i] for i, v in enumerate(vars_)}
            if y_pred_bg is not None
            else {}
        )
        sigma_bg_d = (
            {v: y_sigma_bg[:, i] for i, v in enumerate(vars_)}
            if y_sigma_bg is not None
            else {}
        )

        vars_ext = list(vars_)
        if "chirp_mass" in vars_ and "mass_ratio" in vars_:
            mc_t, q_t = true_d["chirp_mass"], true_d["mass_ratio"]
            mc_p, q_p = pred_d["chirp_mass"], pred_d["mass_ratio"]
            m1t, m2t = _mc_q_to_m1_m2(mc_t, q_t)
            m1p, m2p = _mc_q_to_m1_m2(mc_p, q_p)
            for key, tr, pr in [("m1", m1t, m1p), ("m2", m2t, m2p)]:
                true_d[key] = tr
                pred_d[key] = pr
            if sigma_d:
                s_mc, s_q = sigma_d["chirp_mass"], sigma_d["mass_ratio"]
                s_m1, s_m2 = _propagate_m1_m2_uncertainty(mc_p, q_p, s_mc, s_q)
                sigma_d["m1"] = s_m1
                sigma_d["m2"] = s_m2
            vars_ext = vars_ + ["m1", "m2"]

        self._save_csv_param_est(y_true, y_pred, y_sigma, snr)

        plt.rcParams.update(
            {
                "font.size": 11,
                "axes.spines.top": False,
                "axes.spines.right": False,
            }
        )
        figs, names = [], []

        def pred_vs_true_ax(ax, t, p, s, lims, edges, title):
            # band = median prediction +/- the model's predicted sigma
            # (1sigma, 2sigma) per true bin -- NOT empirical percentiles
            centers = 0.5 * (edges[:-1] + edges[1:])
            med = np.full(len(centers), np.nan)
            sig = np.full(len(centers), np.nan)
            for i, (lo, hi) in enumerate(
                zip(edges[:-1], edges[1:], strict=False)
            ):
                m = (t >= lo) & (t < hi)
                if m.sum() < 2:
                    continue
                med[i] = np.median(p[m])
                if s is not None:
                    sig[i] = np.median(s[m])
            ok = ~np.isnan(med)
            if s is not None:
                bok = ok & ~np.isnan(sig)
                ax.fill_between(
                    centers[bok],
                    (med - sig)[bok],
                    (med + sig)[bok],
                    color="steelblue",
                    alpha=0.35,
                    label=r"$1\sigma$",
                )
            ax.plot(
                centers[ok], med[ok], color="steelblue", lw=1.6, label="Median"
            )
            ax.plot(lims, lims, color="gray", ls=":", lw=1.2)
            ax.set_xlim(lims)
            ax.set_ylim(lims)
            ax.set_aspect("equal", adjustable="box")
            ax.set_title(title, fontsize=10)
            ax.legend(frameon=False, fontsize=8, loc="lower right")

        for v in vars_ext:
            tr = np.asarray(true_d[v])
            pr = np.asarray(pred_d[v])
            s_all = (
                np.asarray(sigma_d[v]) if (sigma_d and v in sigma_d) else None
            )
            lims = [float(tr.min()), float(tr.max())]
            edges = _bins_for(v, tr)

            # 1) predicted vs true over the full test SNR range (8-50).
            if snr is not None:
                blo, bhi = int(np.floor(snr.min())), int(np.ceil(snr.max()))
                bands = [
                    (
                        f"SNR {blo}-{bhi}",
                        np.ones(len(tr), bool),
                        f"snr{blo}-{bhi}",
                    )
                ]
            else:
                bands = [("all", np.ones(len(tr), bool), "all")]
            dist_tag = f" ({self.dist_label})" if self.dist_label else ""
            pr_bg = np.asarray(pred_bg_d[v]) if v in pred_bg_d else None
            s_bg = np.asarray(sigma_bg_d[v]) if v in sigma_bg_d else None
            for blab, bm, suffix in bands:
                if bm.sum() < 2:
                    continue
                ss = s_all[bm] if s_all is not None else None
                fig, ax = plt.subplots(figsize=(5.5, 5.5))
                pred_vs_true_ax(
                    ax,
                    tr[bm],
                    pr[bm],
                    ss,
                    lims,
                    edges,
                    f"{blab}{dist_tag}",
                )
                ax.set_xlabel(_sym(v))
                ax.set_ylabel(_hat(v))
                fig.tight_layout()
                figs.append(fig)
                names.append(f"{v}_pred_vs_true_{suffix}")

                # same panel, background (noise-only) predictions -- only
                # valid when bkg rows are paired 1:1 with sig rows (same
                # true values); the on-the-fly waveform_prob<1 dataloader
                # draws sig/bkg from unrelated rows with different counts,
                # so bm (sized to sig) can't index pr_bg (sized to bkg)
                if pr_bg is not None and len(pr_bg) == len(tr):
                    fig, ax = plt.subplots(figsize=(5.5, 5.5))
                    pred_vs_true_ax(
                        ax,
                        tr[bm],
                        pr_bg[bm],
                        s_bg[bm] if s_bg is not None else None,
                        lims,
                        edges,
                        f"{blab}{dist_tag} (bkg)",
                    )
                    ax.set_xlabel(_sym(v))
                    ax.set_ylabel(_hat(v))
                    fig.tight_layout()
                    figs.append(fig)
                    names.append(f"{v}_pred_vs_true_{suffix}_bkg")

            # 3) z-score histogram (counts) + fitted Gaussian.
            if s_all is not None:
                z = (pr - tr) / (s_all + 1e-12)
                z = z[np.isfinite(z)]
                zbins = np.arange(-5.0, 5.0 + 0.25, 0.25)
                fig, ax = plt.subplots(figsize=(6, 4.5))
                ax.hist(
                    z,
                    bins=zbins,
                    color="steelblue",
                    alpha=0.6,
                    label="z-score",
                )
                mu, sd = float(np.mean(z)), float(np.std(z))
                xs = np.linspace(-5.0, 5.0, 400)
                bw = 0.25
                ax.plot(
                    xs,
                    z.size
                    * bw
                    / (sd * np.sqrt(2 * np.pi))
                    * np.exp(-0.5 * ((xs - mu) / sd) ** 2),
                    "r-",
                    lw=1.8,
                    label=f"fit N({mu:.2f}, {sd:.2f})",
                )
                ax.plot(
                    xs,
                    z.size * bw / np.sqrt(2 * np.pi) * np.exp(-0.5 * xs**2),
                    "k--",
                    lw=1.0,
                    alpha=0.7,
                    label="N(0, 1)",
                )
                ax.set_xlim(-5, 5)
                ax.set_xticks(range(-5, 6))
                ax.set_xlabel(f"{_sym(v)} z-score")
                ax.set_ylabel("Counts")
                ax.legend(frameon=False, fontsize=8, loc="upper right")
                figs.append(fig)
                names.append(f"{v}_zscore")

        # 2) fraction within a relative-error cutoff, per SNR bin.
        if snr is not None:
            snr_bins = _snr_bins(snr)
            snr_a, snr_b = int(snr_bins[0]), int(snr_bins[-1])
            snr_centers = 0.5 * (snr_bins[:-1] + snr_bins[1:])
            cuts = [0.01, 0.02, 0.05, 0.10]
            # viridis-ish blues/greens for 1/2/5%, light orange for 10%
            # (avoid the harsh viridis yellow at the top end)
            cut_colors = ["#3b528b", "#21918c", "#5ec962", "#f4a259"]
            for v in vars_ext:
                tr = np.asarray(true_d[v])
                pr = np.asarray(pred_d[v])
                denom = np.where(np.abs(tr) > 1e-8, np.abs(tr), 1.0)
                rel = np.abs((pr - tr) / denom)
                fig, ax = plt.subplots(figsize=(6.5, 4))
                for j, cut in enumerate(cuts):
                    frac = np.full(len(snr_centers), np.nan)
                    for i, (lo, hi) in enumerate(
                        zip(snr_bins[:-1], snr_bins[1:], strict=False)
                    ):
                        m = (snr >= lo) & (snr < hi)
                        if m.sum() > 0:
                            frac[i] = 100.0 * (rel[m] < cut).mean()
                    ok = ~np.isnan(frac)
                    ax.plot(
                        snr_centers[ok],
                        frac[ok],
                        marker="o",
                        ms=3,
                        color=cut_colors[j],
                        label=f"{int(cut * 100)}%",
                    )
                ax.set_xlabel(_label("snr"))
                ax.set_ylabel("% within cutoff")
                ax.set_ylim(0, 100)
                dist_tag = f" ({self.dist_label})" if self.dist_label else ""
                ax.set_title(f"SNR {snr_a}-{snr_b}{dist_tag}")
                ax.legend(
                    frameon=False,
                    fontsize=9,
                    title="|rel. err|",
                    loc="lower right",
                )
                _snr_xticks(ax, snr_a, snr_b)
                figs.append(fig)
                names.append(f"{v}_frac_within_vs_snr")

        self._save_figs(figs, names)

    def _run_background(self, y_pred_bg, y_sigma_bg):
        """Background-only predictions: CSV + value/sigma distributions."""
        vars_ = self.target_variables
        cols = [f"{v}_pred" for v in vars_] + [
            f"sigma_{v}_pred" for v in vars_
        ]
        pd.DataFrame(
            np.concatenate([y_pred_bg, y_sigma_bg], axis=1), columns=cols
        ).to_csv(self.save_dir / "param_est_results_bkg.csv", index=False)
        figs, names = [], []
        for i, v in enumerate(vars_):
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.hist(y_pred_bg[:, i], bins=60, color="tomato", alpha=0.7)
            ax.set_xlabel(f"background pred {_label(v)}")
            ax.set_ylabel("Counts")
            figs.append(fig)
            names.append(f"{v}_pred_bkg")
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.hist(y_sigma_bg[:, i], bins=60, color="tomato", alpha=0.7)
            ax.set_xlabel(f"background {_sig_label(v)}")
            ax.set_ylabel("Counts")
            figs.append(fig)
            names.append(f"{v}_sigma_bkg")
        self._save_figs(figs, names)

    def _run_detection(self, y_sigma, y_sigma_bg):
        """Predicted sigma as a detection statistic (lower => signal)."""
        vars_ = self.target_variables
        figs, names = [], []
        for i, v in enumerate(vars_):
            s_sig = np.asarray(y_sigma[:, i])
            s_bg = np.asarray(y_sigma_bg[:, i])

            # 1) overlapping sigma histograms: signal vs background
            lo = float(min(s_sig.min(), s_bg.min()))
            hi = float(
                max(np.percentile(s_sig, 99.5), np.percentile(s_bg, 99.5))
            )
            bins = np.linspace(lo, hi, 60)
            fig, ax = plt.subplots(figsize=(6.5, 4))
            ax.hist(
                s_sig,
                bins=bins,
                density=True,
                color="steelblue",
                alpha=0.5,
                label=f"signal (N={len(s_sig)})",
            )
            ax.hist(
                s_bg,
                bins=bins,
                density=True,
                color="tomato",
                alpha=0.5,
                label=f"background (N={len(s_bg)})",
            )
            ax.set_xlabel(_sig_label(v))
            ax.set_ylabel("density")
            ax.legend(frameon=False, fontsize=9)
            figs.append(fig)
            names.append(f"{v}_sigma_sig_vs_bkg")

            # 2) ROC from a sigma threshold (ranking score = -sigma)
            scores = np.concatenate([-s_sig, -s_bg])
            labels = np.concatenate([np.ones(len(s_sig)), np.zeros(len(s_bg))])
            order = np.argsort(-scores, kind="mergesort")
            lab = labels[order]
            tpr = np.concatenate([[0.0], np.cumsum(lab) / max(len(s_sig), 1)])
            fpr = np.concatenate(
                [[0.0], np.cumsum(1 - lab) / max(len(s_bg), 1)]
            )
            auc = float(np.trapz(tpr, fpr))
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.plot(
                fpr, tpr, color="steelblue", lw=1.8, label=f"AUC = {auc:.4f}"
            )
            ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.6)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_xlabel("False positive rate")
            ax.set_ylabel("True positive rate")
            ax.set_title(f"{_label(v)} sigma-based ROC")
            ax.legend(frameon=False, loc="lower right")
            figs.append(fig)
            names.append(f"{v}_roc_sigma")

            # 3) Log-log version of ROC curve
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.loglog(
                fpr, tpr, color="steelblue", lw=1.8, label=f"AUC = {auc:.4f}"
            )
            ax.set_xlim(0.000001, 1)
            ax.set_ylim(0.1, 1)
            ax.set_xlabel("False positive rate")
            ax.set_ylabel("True positive rate")
            ax.set_title(f"{_label(v)} sigma-based ROC")
            ax.legend(frameon=False, loc="lower right")
            figs.append(fig)
            names.append(f"{v}_roc_loglog_sigma")
        self._save_figs(figs, names)

    def _save_csv_param_est(self, y_true, y_pred, y_sigma, snr):
        vars_ = self.target_variables
        residual = y_pred - y_true
        cols = (
            [f"{v}_true" for v in vars_]
            + [f"{v}_pred" for v in vars_]
            + (
                [f"sigma_{v}_pred" for v in vars_]
                if y_sigma is not None
                else []
            )
            + [f"{v}_residual" for v in vars_]
        )
        ordered = [y_true, y_pred]
        if y_sigma is not None:
            zscore = residual / (y_sigma + 1e-12)
            ordered.append(y_sigma)
            cols += [f"{v}_zscore" for v in vars_]
        ordered.append(residual)
        if y_sigma is not None:
            ordered.append(zscore)
        if snr is not None:
            cols += ["snr"]
            ordered.append(np.asarray(snr).reshape(-1, 1))
        pd.DataFrame(
            np.concatenate(ordered, axis=1),
            columns=cols,
        ).to_csv(self.save_dir / "param_est_results.csv", index=False)

    def _run_sky(self, y_true, y_pred, snr):
        dec_true = y_true[:, self.target_variables.index("dec")]
        phi_true = y_true[:, self.target_variables.index("phi")]

        if y_pred.shape[1] == 3:
            norm = np.linalg.norm(y_pred, axis=1, keepdims=True)
            y_pred_norm = y_pred / (norm + 1e-8)
            dec_pred, phi_pred = _vec_to_angles(y_pred_norm)
            cos_theta_pred = y_pred_norm @ _H1L1_BASELINE
        else:
            dec_pred = y_pred[:, self.target_variables.index("dec")]
            phi_pred = y_pred[:, self.target_variables.index("phi")]
            v_pred = _angles_to_vec(dec_pred, phi_pred)
            cos_theta_pred = v_pred @ _H1L1_BASELINE

        v_true = _angles_to_vec(dec_true, phi_true)
        cos_theta_true = v_true @ _H1L1_BASELINE
        ring_dist = np.abs(cos_theta_pred - cos_theta_true)

        v_pred_for_angle = _angles_to_vec(dec_pred, phi_pred)
        cos_sim = np.clip(
            (v_pred_for_angle * v_true).sum(axis=1), -1 + 1e-6, 1 - 1e-6
        )
        angular_error_deg = np.degrees(np.arccos(cos_sim))

        self._save_csv_sky(
            dec_true,
            phi_true,
            dec_pred,
            phi_pred,
            cos_theta_true,
            cos_theta_pred,
            ring_dist,
            angular_error_deg,
            snr,
        )

        # Minimal notification
        print(
            f"Saved {len(dec_true)} rows -> "
            f"{self.save_dir / 'sky_loc_results.csv'}"
        )

    def _save_csv_sky(
        self,
        dec_true,
        phi_true,
        dec_pred,
        phi_pred,
        cos_theta_true,
        cos_theta_pred,
        ring_dist,
        angular_error_deg,
        snr,
    ):
        data = {
            "dec_true": dec_true,
            "phi_true": phi_true,
            "dec_pred": dec_pred,
            "phi_pred": phi_pred,
            "dec_residual": dec_pred - dec_true,
            "phi_residual": phi_pred - phi_true,
            "cos_theta_true": cos_theta_true,
            "cos_theta_pred": cos_theta_pred,
            "ring_distance": ring_dist,
            "angular_error_deg": angular_error_deg,
        }
        if snr is not None:
            data["snr"] = snr
        pd.DataFrame(data).to_csv(
            self.save_dir / "sky_loc_results.csv", index=False
        )
        print(
            f"Saved {len(dec_true)} rows -> "
            f"{self.save_dir / 'sky_loc_results.csv'}"
        )

    def _save_figs(self, figs, names):
        for fig, name in zip(figs, names, strict=True):
            out = self.save_dir / f"{name}.png"
            fig.savefig(out, dpi=400, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved {out}")


class EMA(Callback):
    """Exponential moving average of model weights.

    Keeps a shadow copy of every trainable parameter, updated each training
    step as ``shadow = decay*shadow + (1-decay)*param``. Validation (and the
    checkpoint saved at validation end) use the shadow weights, so the
    monitored metric and the best checkpoint reflect the averaged model. Live
    training weights are restored on the next train batch. The shadow is
    stored in the checkpoint so it survives resume.

    JAX/equinox models are skipped (their weights live outside torch params).

    Args:
        decay: EMA decay. Higher = slower/smoother (0.999-0.9999 typical).
    """

    def __init__(self, decay: float = 0.999):
        super().__init__()
        if not 0.0 < decay < 1.0:
            raise ValueError(f"decay must be in (0, 1), got {decay}")
        self.decay = decay
        self.shadow: dict[str, torch.Tensor] = {}
        self._backup: dict[str, torch.Tensor] = {}

    def _is_jax(self, pl_module) -> bool:
        return hasattr(pl_module, "jax_model")

    def on_fit_start(self, trainer, pl_module) -> None:
        if self._is_jax(pl_module):
            return
        if self.shadow:
            # shadow restored from a checkpoint loads on cpu; move it onto
            # the module device so the in-place ema update matches params
            self.shadow = {
                name: t.to(pl_module.device) for name, t in self.shadow.items()
            }
            return
        self.shadow = {
            name: p.detach().clone()
            for name, p in pl_module.named_parameters()
            if p.requires_grad
        }

    @torch.no_grad()
    def on_train_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx
    ) -> None:
        if not self.shadow:
            return
        for name, p in pl_module.named_parameters():
            if name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(
                    p.detach(), alpha=1.0 - self.decay
                )

    def _load_shadow(self, pl_module) -> None:
        self._backup = {}
        for name, p in pl_module.named_parameters():
            if name in self.shadow:
                self._backup[name] = p.detach().clone()
                p.data.copy_(self.shadow[name])

    def _restore(self, pl_module) -> None:
        for name, p in pl_module.named_parameters():
            if name in self._backup:
                p.data.copy_(self._backup[name])
        self._backup = {}

    def on_validation_epoch_start(self, trainer, pl_module) -> None:
        if self.shadow:
            self._load_shadow(pl_module)

    def on_train_batch_start(
        self, trainer, pl_module, batch, batch_idx
    ) -> None:
        # restore live weights after validation swapped in the shadow
        if self._backup:
            self._restore(pl_module)

    def on_save_checkpoint(self, trainer, pl_module, checkpoint) -> None:
        if self.shadow:
            checkpoint["ema_shadow"] = self.shadow

    def on_load_checkpoint(self, trainer, pl_module, checkpoint) -> None:
        if "ema_shadow" in checkpoint:
            self.shadow = checkpoint["ema_shadow"]


class ClassificationTestCallback(Callback):
    """Detection diagnostics for a classifier on ``trainer.test()``.

    Collects the per-batch logit ``score`` + ``label`` (1 = injected,
    0 = background) + ``snr`` from ``test_step`` and, at epoch end, writes:
      - ROC (TPR vs FPR, linear + log-log)
      - detection efficiency vs SNR at fixed background FPR (1e-3/1e-2/1e-1)
      - a raw ``class_test_scores.csv``.
    """

    _FPRS = [1e-3, 1e-2, 1e-1]
    _FPR_COLORS = ["#3b528b", "#21918c", "#f4a259"]

    def __init__(self, save_dir: str | Path = "", dist_label: str = ""):
        super().__init__()
        self.save_dir = Path(save_dir) if save_dir else Path(".")
        self.dist_label = dist_label
        self._score: list[torch.Tensor] = []
        self._label: list[torch.Tensor] = []
        self._snr: list[torch.Tensor] = []

    def on_test_start(self, trainer, pl_module):
        self._score.clear()
        self._label.clear()
        self._snr.clear()

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if not outputs:
            return
        self._score.append(outputs["score"])
        self._label.append(outputs["label"])
        if "snr" in outputs:
            self._snr.append(outputs["snr"])

    def on_test_epoch_end(self, trainer, pl_module):
        if not self._score:
            return
        score = torch.cat(self._score).numpy()
        label = torch.cat(self._label).numpy().astype(int)
        snr = torch.cat(self._snr).numpy() if self._snr else None
        self.save_dir.mkdir(parents=True, exist_ok=True)

        pd.DataFrame(
            {
                "score": score,
                "label": label,
                "snr": snr if snr is not None else np.nan,
            }
        ).to_csv(self.save_dir / "class_test_scores.csv", index=False)

        tag = f" ({self.dist_label})" if self.dist_label else ""
        self._roc(score, label, tag)
        if snr is not None:
            self._eff_vs_snr(score, label, snr, tag)

    def _roc(self, score, label, tag):
        order = np.argsort(-score)
        lab = label[order]
        tp = np.cumsum(lab)
        fp = np.cumsum(1 - lab)
        n_pos, n_neg = int(lab.sum()), int((1 - lab).sum())
        if n_pos == 0 or n_neg == 0:
            return
        tpr = tp / n_pos
        fpr = fp / n_neg

        for logx, name in [(False, "roc"), (True, "roc_loglog")]:
            fig, ax = plt.subplots(figsize=(5.5, 5))
            ax.plot(fpr, tpr, color="steelblue", lw=1.8)
            if logx:
                ax.set_xscale("log")
                ax.set_yscale("log")
                ax.set_xlim(1e-4, 1)
            else:
                ax.plot([0, 1], [0, 1], color="gray", ls=":", lw=1.0)
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
            ax.set_xlabel("False positive rate")
            ax.set_ylabel("True positive rate")
            ax.set_title(f"ROC{tag}")
            ax.grid(alpha=0.3)
            fig.tight_layout()
            fig.savefig(self.save_dir / f"{name}.png", dpi=140)
            plt.close(fig)

    def _eff_vs_snr(self, score, label, snr, tag):
        bkg = score[label == 0]
        sig = score[label == 1]
        sig_snr = snr[label == 1]
        good = np.isfinite(sig_snr)
        sig, sig_snr = sig[good], sig_snr[good]
        if len(bkg) == 0 or len(sig) == 0:
            return
        a, b = int(np.floor(sig_snr.min())), int(np.ceil(sig_snr.max()))
        bins = np.arange(a, b + 2, 2)
        centers = 0.5 * (bins[:-1] + bins[1:])

        fig, ax = plt.subplots(figsize=(6.5, 4))
        for fpr, color in zip(self._FPRS, self._FPR_COLORS, strict=False):
            thr = np.quantile(bkg, 1 - fpr)
            eff = np.full(len(centers), np.nan)
            for i, (lo, hi) in enumerate(
                zip(bins[:-1], bins[1:], strict=False)
            ):
                m = (sig_snr >= lo) & (sig_snr < hi)
                if m.sum() > 0:
                    eff[i] = 100.0 * (sig[m] > thr).mean()
            ok = ~np.isnan(eff)
            ax.plot(
                centers[ok],
                eff[ok],
                marker="o",
                ms=3,
                color=color,
                label=f"FPR {fpr:g}",
            )
        ax.set_xlabel("SNR")
        ax.set_ylabel("Detection efficiency [%]")
        ax.set_ylim(0, 100)
        ax.set_title(f"SNR {a}-{b}{tag}")
        ax.legend(frameon=False, fontsize=9, loc="lower right")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(self.save_dir / "efficiency_vs_snr.png", dpi=140)
        plt.close(fig)


class DenoiserEvolutionCallback(Callback):
    """Log per-epoch denoiser reconstruction on a fixed batch.

    Adapted for the aframe denoiser: the model returns ``(x_denoised, out)``
    (only ``x_denoised`` is plotted), and the clean target lives in the train
    batch ``(X, X_clean, y, params)``, so a fixed reference batch is captured
    from the train dataloader once and reused every epoch. Each validation
    epoch draws target vs prediction (vs noisy input) per example and ifo,
    logged to wandb under one stable key (media step-slider) and to disk;
    frames are assembled into a GIF at fit end.

    Tensors are ``(B, C, L)`` (C = n_ifos).
    """

    def __init__(
        self,
        n_examples: int = 3,
        every_n_epochs: int = 1,
        sample_rate: Optional[float] = 256.0,
        window_begin: float = 0.0,
        plot_window: Optional[tuple] = None,
        show_input: bool = True,
        out_dir: Optional[str] = None,
        gif_fps: int = 4,
        max_gif_frames: int = 200,
    ):
        super().__init__()
        self.n_examples = n_examples
        self.every_n_epochs = every_n_epochs
        self.sample_rate = sample_rate
        self.window_begin = window_begin
        self.plot_window = tuple(plot_window) if plot_window else None
        self.show_input = show_input
        self.out_dir = Path(out_dir) if out_dir else None
        self.gif_fps = gif_fps
        self.max_gif_frames = max_gif_frames
        self._fixed_batch = None
        self._frame_paths = []

    def on_fit_start(self, trainer, pl_module) -> None:
        if self.out_dir is None:
            self.out_dir = Path(trainer.log_dir or ".") / "denoiser_evolution"
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def on_train_batch_start(
        self, trainer, pl_module, batch, batch_idx
    ) -> None:
        """Capture the first post-augmentation batch once.

        aframe injects/whitens on-GPU in ``on_after_batch_transfer``, so the
        training batch here is ``(X, X_clean, y, params)`` -- the noisy input
        and clean target -- unlike the raw dataloader output.
        """
        if self._fixed_batch is not None:
            return
        if not (isinstance(batch, (tuple, list)) and len(batch) >= 2):
            return
        X, X_clean = batch[0], batch[1]
        if not (torch.is_tensor(X) and torch.is_tensor(X_clean)):
            return
        n = min(self.n_examples, X.shape[0])
        self._fixed_batch = (
            X[:n].detach().clone(),
            X_clean[:n].detach().clone(),
        )

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        if trainer.sanity_checking:
            return
        if trainer.current_epoch % self.every_n_epochs != 0:
            return
        if self._fixed_batch is None:
            return

        noisy, target = self._fixed_batch
        was_training = pl_module.training
        pl_module.eval()
        with torch.no_grad():
            pred = pl_module(noisy)
            if isinstance(pred, (tuple, list)):
                pred = pred[0]  # x_denoised
        if was_training:
            pl_module.train()

        fig = self._draw(
            noisy.float().cpu().numpy(),
            target.float().cpu().numpy(),
            pred.float().cpu().numpy(),
            epoch=trainer.current_epoch,
        )
        self.out_dir.mkdir(parents=True, exist_ok=True)
        path = self.out_dir / f"epoch_{trainer.current_epoch:04d}.png"
        fig.savefig(path, dpi=110, bbox_inches="tight")
        self._frame_paths.append(path)

        if isinstance(trainer.logger, WandbLogger):
            import wandb

            trainer.logger.experiment.log(
                {"denoiser/evolution": wandb.Image(fig)},
                step=trainer.global_step,
            )
        plt.close(fig)

    def on_test_start(self, trainer, pl_module) -> None:
        if self.out_dir is None:
            self.out_dir = Path(trainer.log_dir or ".") / "denoiser_evolution"
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._test_batch = None

    def on_test_batch_start(
        self, trainer, pl_module, batch, batch_idx, dataloader_idx=0
    ) -> None:
        """Capture one post-augmentation test batch."""
        if getattr(self, "_test_batch", None) is not None:
            return
        if not (isinstance(batch, (tuple, list)) and len(batch) >= 2):
            return
        X, X_clean = batch[0], batch[1]
        if not (torch.is_tensor(X) and torch.is_tensor(X_clean)):
            return
        n = min(self.n_examples, X.shape[0])
        self._test_batch = (
            X[:n].detach().clone(),
            X_clean[:n].detach().clone(),
        )

    def on_test_end(self, trainer, pl_module) -> None:
        """Draw the same time / frequency panels on the test batch.

        With ``waveform_prob=0`` the clean target is identically zero, so
        this shows what the denoiser emits from noise alone.
        """
        batch = getattr(self, "_test_batch", None)
        if batch is None:
            return

        noisy, target = batch
        was_training = pl_module.training
        pl_module.eval()
        with torch.no_grad():
            pred = pl_module(noisy)
            if isinstance(pred, (tuple, list)):
                pred = pred[0]
        if was_training:
            pl_module.train()

        fig = self._draw(
            noisy.float().cpu().numpy(),
            target.float().cpu().numpy(),
            pred.float().cpu().numpy(),
            epoch=trainer.current_epoch,
        )
        self.out_dir.mkdir(parents=True, exist_ok=True)
        path = self.out_dir / "test.png"
        fig.savefig(path, dpi=110, bbox_inches="tight")
        print(f"[DenoiserEvolutionCallback] wrote {path}")

        if isinstance(trainer.logger, WandbLogger):
            import wandb

            trainer.logger.experiment.log({"denoiser/test": wandb.Image(fig)})
        plt.close(fig)

    def _assemble_gif(self, frame_paths, gif_name):
        from PIL import Image

        paths = [p for p in frame_paths if p.exists()]
        if len(paths) < 2:
            return None
        if len(paths) > self.max_gif_frames:
            idx = np.linspace(0, len(paths) - 1, self.max_gif_frames)
            paths = [paths[int(i)] for i in idx]
        frames = [
            Image.open(p).convert("P", palette=Image.ADAPTIVE) for p in paths
        ]
        gif_path = self.out_dir / gif_name
        frames[0].save(
            gif_path,
            save_all=True,
            append_images=frames[1:],
            duration=int(1000 / self.gif_fps),
            loop=0,
        )
        return gif_path

    def on_fit_end(self, trainer, pl_module) -> None:
        if not self._frame_paths:
            return
        gif_path = self._assemble_gif(self._frame_paths, "evolution.gif")
        if gif_path and isinstance(trainer.logger, WandbLogger):
            import wandb

            trainer.logger.experiment.log(
                {
                    "denoiser/evolution_gif": wandb.Video(
                        str(gif_path), fps=self.gif_fps
                    )
                }
            )

    def _time_axis(self, length):
        if self.sample_rate:
            t = self.window_begin + np.arange(length) / self.sample_rate
            return t, "time [s]"
        return np.arange(length), "sample"

    def _fft(self, x):
        fs = self.sample_rate or 1.0
        mag = np.abs(np.fft.rfft(x))
        freqs = np.fft.rfftfreq(x.shape[0], d=1.0 / fs)
        return freqs[1:], np.maximum(mag[1:], 1e-30)

    def _draw(self, noisy, target, pred, epoch):
        """noisy/target/pred: (N, C, L). Returns a Figure."""
        n_ex, n_ifos, length = target.shape
        sr = self.sample_rate or 1.0
        xlabel = "time to merger [s]" if self.sample_rate else "sample"
        lo, hi = 0, length
        if self.plot_window and self.sample_rate:
            begin, end = self.plot_window
            lo = max(0, int((begin - self.window_begin) * self.sample_rate))
            hi = min(length, int((end - self.window_begin) * self.sample_rate))
        sl = slice(lo, hi)

        n_rows = n_ex * n_ifos
        fig, axes = plt.subplots(
            n_rows, 2, figsize=(14, 4.4 * n_rows), squeeze=False
        )
        for e in range(n_ex):
            for k in range(n_ifos):
                row = e * n_ifos + k
                # merger = peak amplitude of the clean target; t=0 there,
                # so pre-merger is negative (relative time to coalescence).
                # With no injection the target is all zeros and has no
                # merger, so fall back to absolute time from kernel start.
                if np.any(target[e, k]):
                    merger_idx = int(np.argmax(np.abs(target[e, k])))
                else:
                    merger_idx = 0
                t = (np.arange(length) - merger_idx) / sr
                mse = float(np.mean((pred[e, k, sl] - target[e, k, sl]) ** 2))
                ax = axes[row][0]
                if self.show_input:
                    ax.plot(
                        t[sl],
                        noisy[e, k, sl],
                        lw=0.5,
                        color="0.6",
                        alpha=0.45,
                        label="noisy input",
                    )
                ax.plot(
                    t[sl], target[e, k, sl], lw=0.9, color="k", label="target"
                )
                ax.plot(
                    t[sl],
                    pred[e, k, sl],
                    lw=0.9,
                    color="tab:red",
                    label="prediction",
                )
                ax.set_ylabel(f"ex {e} / ifo {k}")
                ax.set_title(f"MSE = {mse:.3e}", fontsize=8, loc="right")
                if row == 0:
                    ax.legend(fontsize=7, ncol=3, loc="upper left")
                if row == n_rows - 1:
                    ax.set_xlabel(xlabel)

                axf = axes[row][1]
                if self.show_input:
                    f, m = self._fft(noisy[e, k, sl])
                    axf.loglog(
                        f,
                        m,
                        lw=0.6,
                        color="0.6",
                        alpha=0.45,
                        label="noisy input",
                    )
                f, m = self._fft(target[e, k, sl])
                axf.loglog(f, m, lw=0.9, color="k", label="target")
                f, m = self._fft(pred[e, k, sl])
                axf.loglog(f, m, lw=0.9, color="tab:red", label="prediction")
                axf.set_ylabel("|rfft|")
                if row == 0:
                    axf.set_title("abs(rfft) magnitude", fontsize=9)
                    axf.legend(fontsize=7, loc="upper right")
                if row == n_rows - 1:
                    axf.set_xlabel("frequency [Hz]")
        fig.suptitle(f"epoch {epoch}", fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.98])
        return fig
