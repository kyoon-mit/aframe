"""Training entry point for Project 8.

The aframe CLI requires a BaseAframeDataset and links gravitational-wave data
arguments into the model, neither of which a Project 8 datamodule has. This
builds the same Lightning CLI around the aframe models with the Project 8
datamodule instead:

    python -m train.experimental.project8 fit --config <config.yaml>
"""

import torch
from lightning.pytorch.cli import LightningCLI

import train.cli  # noqa: F401  (applies its torch.load override)
from train.callbacks import WandbSaveConfig
from train.experimental.project8.data import Project8DataModule
from train.model import AframeBase


def main(args=None):
    torch.set_float32_matmul_precision("high")
    return LightningCLI(
        AframeBase,
        Project8DataModule,
        subclass_mode_model=True,
        subclass_mode_data=True,
        parser_kwargs={"parser_mode": "omegaconf"},
        save_config_callback=WandbSaveConfig,
        save_config_kwargs={"overwrite": True},
        seed_everything_default=101588,
        args=args,
    )
