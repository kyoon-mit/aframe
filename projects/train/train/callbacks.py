import io
import os
import shutil
import logging
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import h5py
import s3fs
import torch
from botocore.exceptions import ClientError, ConnectTimeoutError
from lightning import pytorch as pl
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.cli import SaveConfigCallback
from lightning.pytorch.callbacks import ModelCheckpoint as PLModelCheckpoint
from lightning.pytorch.utilities import grad_norm
from typing import Literal, Union
from pathlib import Path

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
            X, y, *_ = trainer.datamodule.inject(X, waveforms)
        else:
            [X] = next(iter(trainer.train_dataloader))
            X = X.to(device)
            X, y, *_ = trainer.datamodule.inject(X)
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
                X, y, *_ = trainer.datamodule.inject(X, waveforms)
            else:
                [X] = next(iter(trainer.train_dataloader))
                X = X.to(device)
                X, y, *_ = trainer.datamodule.inject(X)
            # If X is not a tuple, make it one for consistency
            # of format for saving to file below
            if not isinstance(X, tuple):
                X = (X,)

            # build val batch by hand
            [background, _, _], [signals, _params] = next(
                iter(trainer.datamodule.val_dataloader())
            )
            background = background.to(device)
            signals = signals.to(device)
            X_bg, X_inj, _params = trainer.datamodule.build_val_batches(
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


class WaveformWindowDiagram(Callback):
    """Log a one-shot diagram of the waveform slicing geometry to W&B.

    Shows the sliced waveform extent, valid window placement range, and one
    example sampled window (with whitened vs. cropped edges), all relative
    to the merger at t=0.

    Add to the YAML trainer callbacks list::

        - class_path: train.callbacks.WaveformWindowDiagram
    """

    def on_train_start(self, trainer, pl_module):
        if trainer.global_rank != 0:
            return

        dm = trainer.datamodule
        hp = dm.hparams
        kl = hp.kernel_length   # whitened kernel length (s)
        fd = hp.fduration        # whitening filter duration (s)
        lp = hp.left_pad         # min merger-to-left-edge gap (s, whitened)
        rp = hp.right_pad        # min merger-to-right-edge gap (s, whitened)

        # Convert pad constraints to the unwhitened timeline.
        # Each side loses fd/2 to whitening, so in the unwhitened frame:
        lp_u = lp + fd / 2
        rp_u = rp + fd / 2
        kernel_u = kl + fd       # unwhitened kernel duration (s)

        # Sliced-waveform extents relative to merger at t=0.
        wf_start = -(kernel_u - rp_u)   # = -(kl + fd/2 - rp)
        wf_end   =  (kernel_u - lp_u)   # = kl + fd/2 - lp

        # Valid window-start range (unwhitened kernel must fit inside slice).
        win_start_min = wf_start
        win_start_max = -lp_u            # merger always >= lp_u from left

        # Sample one concrete window (fixed seed for reproducibility).
        rng = np.random.default_rng(42)
        frac = rng.uniform(0, 1)
        win_start = win_start_min + frac * (win_start_max - win_start_min)
        win_end   = win_start + kernel_u

        # Whitened portion: fd/2 trimmed from each end.
        wh_start = win_start + fd / 2
        wh_end   = win_end   - fd / 2

        # ── figure ──────────────────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(13, 3.5))

        Y_WF  = 0.75
        Y_WIN = 0.35
        LW    = 14

        # Sliced waveform bar (blue)
        ax.plot([wf_start, wf_end], [Y_WF, Y_WF], linewidth=LW,
                color="steelblue", solid_capstyle="butt",
                label="Sliced waveform")

        # Valid window-start span (orange shading)
        ax.axvspan(win_start_min, win_start_max, alpha=0.18,
                   color="darkorange", label="Valid window-start zone")

        # Sampled window: cropped (faded green) then whitened (solid green)
        ax.plot([win_start, win_end], [Y_WIN, Y_WIN], linewidth=LW,
                color="green", alpha=0.25, solid_capstyle="butt",
                label="Unwhitened kernel (cropped)")
        ax.plot([wh_start, wh_end], [Y_WIN, Y_WIN], linewidth=LW,
                color="green", solid_capstyle="butt",
                label="Whitened kernel (model input)")

        # Merger line
        ax.axvline(0, color="red", linewidth=1.8, linestyle="--",
                   label="Merger (t = 0)")

        # ── annotations ─────────────────────────────────────────────────────
        def ann(x, y, label, ha="center", color="black", dy=0.09):
            ax.text(x, y + dy, label, ha=ha, va="bottom",
                    fontsize=7.5, color=color)

        ann(wf_start, Y_WF, f"{wf_start:.2f}s", color="steelblue")
        ann(wf_end,   Y_WF, f"{wf_end:.2f}s",   color="steelblue")

        mid_valid = (win_start_min + win_start_max) / 2
        jitter = win_start_max - win_start_min
        ax.annotate(
            "",
            xy=(win_start_max, Y_WF - 0.18),
            xytext=(win_start_min, Y_WF - 0.18),
            arrowprops=dict(arrowstyle="<->", color="darkorange", lw=1.5),
        )
        ax.text(mid_valid, Y_WF - 0.12,
                f"jitter = {jitter:.2f}s", ha="center",
                fontsize=7.5, color="darkorange")

        ann(win_start, Y_WIN, f"{win_start:.2f}s", color="grey")
        ann(win_end,   Y_WIN, f"{win_end:.2f}s",   color="grey")
        ann(wh_start, Y_WIN, f"{wh_start:.2f}s", color="darkgreen")
        ann(wh_end,   Y_WIN, f"{wh_end:.2f}s",   color="darkgreen")

        ax.set_xlim(wf_start - 0.4, wf_end + 0.4)
        ax.set_ylim(0.1, 1.05)
        ax.set_yticks([])
        ax.set_xlabel("Time relative to merger (s)", fontsize=10)
        ax.set_title(
            f"Waveform window geometry  —  "
            f"kernel_length={kl}s, fduration={fd}s, "
            f"left_pad={lp}s, right_pad={rp}s",
            fontsize=10,
        )
        ax.legend(loc="upper left", fontsize=8, framealpha=0.8)
        ax.grid(True, axis="x", alpha=0.3)
        plt.tight_layout()

        for logger in trainer.loggers:
            if isinstance(logger, WandbLogger):
                import wandb
                logger.experiment.log(
                    {"waveform_window_geometry": wandb.Image(fig)}
                )
                break

        plt.close(fig)
