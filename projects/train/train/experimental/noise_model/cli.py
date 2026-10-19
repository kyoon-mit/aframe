"""Entry point for the noise model (dev/local/train/claude_tests/noise_model/run.sh
adds each run's name and folder):

    python -m train.experimental.noise_model fit --config base.yaml --config <judge or generator>.yaml --config <data>.yaml
    python -m train.experimental.noise_model test --config ... --ckpt_path <run folder>/CLAUDE-TESTS/<run name>/checkpoints/last.ckpt

Later config files override earlier ones. Without run.sh, wandb picks a random
run id and files land in the current folder.
"""

import lightning.pytorch as pl
import torch
from lightning.pytorch.cli import LightningCLI

import train.cli  # noqa: F401  (its torch.load override lets checkpoints rebuild their networks)
from train.callbacks import WandbSaveConfig


def main(args=None):
    torch.set_float32_matmul_precision("high")
    return LightningCLI(
        pl.LightningModule,
        pl.LightningDataModule,
        subclass_mode_model=True,
        subclass_mode_data=True,
        parser_kwargs={"parser_mode": "omegaconf"},
        save_config_callback=WandbSaveConfig,
        save_config_kwargs={"overwrite": True},
        seed_everything_default=101588,
        args=args,
    )
