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
