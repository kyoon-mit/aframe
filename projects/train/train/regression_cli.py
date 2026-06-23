"""CLI entry point for regression training.

Run:
    uv run python -m train.regression_cli fit --config regression.yaml

The YAML must wire up a LightningDataModule subclass (e.g.
RegressionTimeDomainDataset) and a LightningModule subclass (e.g.
LitLinOSSGaussianNLL or LitS4DGaussianNLL).
"""

import torch
from lightning.pytorch.cli import LightningCLI

from projects.train.train.callbacks import PlotParamEstCallback


class RegressionCLI(LightningCLI):
    def before_test(self):
        # This hook runs automatically ONLY during the `test` subcommand,
        # after classes are instantiated but before the test routine starts.
        self.trainer.callbacks.append(PlotParamEstCallback())


def main():
    torch.set_float32_matmul_precision("high")

    # By letting `run=True` (default), subcommands like fit/test are enabled.
    # LightningCLI automatically executes the appropriate stage after parsing.
    RegressionCLI(save_config_callback=None)


if __name__ == "__main__":
    main()