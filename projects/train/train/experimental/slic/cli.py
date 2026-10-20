"""Entry point for the SLIC score network:

    python -m train.experimental.slic fit --config <score.yaml>

The data section is an aframe dataset config, used as is; fit trains on its
whitened noise-only windows (``data.NoiseWindows``).
"""

import lightning.pytorch as pl
import torch
from lightning.pytorch.cli import LightningCLI

import train.cli  # noqa: F401  (applies its torch.load override)
from train.callbacks import WandbSaveConfig
from train.data import BaseAframeDataset
from train.experimental.slic.data import NoiseWindows


class ScoreCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        # the data links of train.cli.AframeCLI
        for name in ("sample_rate", "fduration", "ifos"):
            parser.link_arguments(
                f"data.init_args.{name}",
                f"data.init_args.waveform_sampler.init_args.{name}",
                apply_on="parse",
            )

    def before_fit(self):
        self.datamodule = NoiseWindows(self.datamodule)


def main(args=None):
    torch.set_float32_matmul_precision("high")
    return ScoreCLI(
        pl.LightningModule,
        BaseAframeDataset,
        subclass_mode_model=True,
        subclass_mode_data=True,
        parser_kwargs={"parser_mode": "omegaconf"},
        save_config_callback=WandbSaveConfig,
        save_config_kwargs={"overwrite": True},
        seed_everything_default=101588,
        args=args,
    )
