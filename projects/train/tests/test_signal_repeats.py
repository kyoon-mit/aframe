from types import SimpleNamespace

import pytest
import torch
from lightning.pytorch.utilities.parsing import AttributeDict

from train.augmentations import WaveformProjector
from train.data.supervised.supervised import SupervisedAframeDataset

SAMPLE_RATE = 256
KERNEL = 512
WAVEFORM = 768


class Stub(SupervisedAframeDataset):
    """Just enough of the dataset to call ``inject``."""

    def __init__(self, batch_size, signal_repeats, training=True):
        self._hparams = AttributeDict(batch_size=batch_size, waveform_prob=1.0)
        self.trainer = SimpleNamespace(training=training)
        self.signal_repeats = signal_repeats
        self.swap_prob = self.mute_prob = 0
        self.swapper = self.muter = None
        self.waveforms_from_disk = True
        self.param_transforms = []
        self.inverter = self.reverser = lambda X: X
        self.dec = torch.distributions.Uniform(-1.0, 1.0)
        self.psi = torch.distributions.Uniform(0.0, 3.14)
        self.phi = torch.distributions.Uniform(-3.14, 3.14)
        self.snr_sampler = torch.distributions.Uniform(4.0, 100.0)
        self.projector = WaveformProjector(["H1", "L1"], SAMPLE_RATE, 20)

    def psd_estimator(self, X):
        psds = torch.rand(len(X), 2, KERNEL // 2 + 1) + 0.5
        return X, psds


def batch(stub, n_waveforms):
    X = torch.randn(stub.hparams.batch_size, 2, KERNEL)
    waveforms = torch.randn(n_waveforms, 2, WAVEFORM)
    params = {"mass_1": torch.rand(n_waveforms) + 1}
    return X, waveforms, params


@pytest.mark.parametrize(
    "batch_size,repeats,expected", [(32, 4, 8), (30, 4, 8), (32, 1, 32)]
)
def test_waveforms_per_batch(batch_size, repeats, expected):
    assert Stub(batch_size, repeats).waveforms_per_batch == expected


@pytest.mark.parametrize("batch_size", [32, 30])
def test_repeats_share_signal(batch_size):
    torch.manual_seed(0)
    stub = Stub(batch_size, signal_repeats=4)
    X, waveforms, params = batch(stub, stub.waveforms_per_batch)
    background = X.clone()
    X, _, _, out = stub.inject(X, waveforms, params)

    group = torch.arange(batch_size) // 4
    for key in ("mass_1", "dec", "psi", "phi", "snr"):
        for g in group.unique():
            vals = out[key][group == g]
            assert torch.all(vals == vals[0]), key
    # distinct signals across groups, distinct noise within one
    assert out["mass_1"][::4].unique().numel() == len(group.unique())
    assert not torch.allclose(background[0], background[1])
    assert torch.allclose(X - stub._clean_signal, background, atol=1e-4)


def test_no_repeats_outside_training():
    torch.manual_seed(0)
    stub = Stub(32, signal_repeats=4, training=False)
    X, waveforms, params = batch(stub, 32)
    _, _, _, out = stub.inject(X, waveforms, params)
    assert out["mass_1"].unique().numel() == 32


def test_project_once_matches_projecting_each_copy():
    torch.manual_seed(0)
    projector = WaveformProjector(["H1", "L1"], SAMPLE_RATE, 20)
    n, R = 3, 4
    dec, psi, phi = torch.rand(n), torch.rand(n), torch.rand(n)
    snrs = torch.rand(n) * 50 + 4
    hc, hp = torch.randn(n, WAVEFORM), torch.randn(n, WAVEFORM)
    psds = torch.rand(n * R, 2, WAVEFORM // 2 + 1) + 0.5

    rep = lambda t: t.repeat_interleave(R, dim=0)
    each = projector(
        rep(dec), rep(psi), rep(phi), rep(snrs), psds, cross=rep(hc), plus=rep(hp)
    )
    once = projector.rescaler(
        rep(projector(dec, psi, phi, cross=hc, plus=hp)), psds, rep(snrs)
    )
    assert torch.allclose(once, each, rtol=1e-5, atol=1e-5)
