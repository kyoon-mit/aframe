"""Entry point for the raw-strain noise model:

    python -m train.experimental.raw_noise fit --config <config.yaml>

The data section is an aframe dataset config, used as is; fit trains on its
noise-only windows (``data.RawWindows``), built as the model's ``kind`` and
``scale`` say.
"""

from train.experimental.raw_noise.data import RawWindows
from train.experimental.slic.cli import ScoreCLI, main


def before_fit(self):
    hparams = self.model.hparams
    self.datamodule = RawWindows(self.datamodule, hparams.kind, hparams.scale)


ScoreCLI.before_fit = before_fit
main()
