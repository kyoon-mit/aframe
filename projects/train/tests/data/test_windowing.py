"""
Tests for window placement relative to the merger.

Covers:
  * WindowConfig property math, including negative / post-merger leads.
  * The BaseAframeDataset._pad_and_slice helper (zero-pad, no wrap-around).
  * slice_waveforms geometry for windows that straddle, precede, or follow
    the merger.
  * build_val_batches view slicing for a pre-merger window (the path that
    previously wrapped negative indices around).
"""

import logging

import pytest
import torch

from train.data.base import BaseAframeDataset
from train.data.supervised.time_domain import TimeDomainSupervisedAframeDataset
from train.data.waveforms.generator.generator import WaveformGenerator
from train.data.windowing import WindowConfig

from conftest import (
    IFOS,
    KERNEL_LENGTH,
    SAMPLE_RATE,
    base_dataset_kwargs,
    setup_dataset_manually,
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
class _TrivialGenerator(WaveformGenerator):
    """Concrete WaveformGenerator whose forward() returns zeros."""

    def forward(self, **params):
        n = len(next(iter(params.values())))
        return torch.zeros(n, 2, int(8 * SAMPLE_RATE))


def _make_dataset(data_dir, val_waveform_file, *, lead_min, lead_max):
    mock_prior = lambda n, device=None: {  # noqa: E731
        "mass_1": torch.ones(n),
        "mass_2": torch.ones(n),
    }
    sampler = _TrivialGenerator(
        ifos=IFOS,
        sample_rate=SAMPLE_RATE,
        val_waveform_file=val_waveform_file,
        training_prior=mock_prior,
    )
    kwargs = base_dataset_kwargs(
        data_dir,
        sampler,
        window_lead_min=lead_min,
        window_lead_max=lead_max,
    )
    dataset = TimeDomainSupervisedAframeDataset(**kwargs)
    setup_dataset_manually(dataset)
    return dataset


def _ramp_waveforms(n, length):
    """(n, 2, length) tensor whose value at index i is i+1 (0 == zero-pad)."""
    ramp = torch.arange(1, length + 1, dtype=torch.float32)
    return ramp.repeat(n, 2, 1)


# --------------------------------------------------------------------------- #
# WindowConfig property math
# --------------------------------------------------------------------------- #
class TestWindowConfig:
    def test_defaults_to_fixed_window(self):
        wc = WindowConfig(kernel_length=2.0, window_lead_min=-5.0)
        assert wc.window_lead_max == -5.0
        assert wc.sliding_range == 0.0

    def test_min_greater_than_max_raises(self):
        with pytest.raises(ValueError):
            WindowConfig(2.0, window_lead_min=1.0, window_lead_max=0.0)

    def test_pre_merger_pads(self):
        wc = WindowConfig(2.0, window_lead_min=-5.0, window_lead_max=-5.0)
        assert wc.right_pad == -5.0
        assert wc.left_pad == 7.0
        assert wc.sliding_range == 0.0

    def test_post_merger_pads(self):
        wc = WindowConfig(2.0, window_lead_min=3.0, window_lead_max=3.0)
        assert wc.right_pad == 3.0
        assert wc.left_pad == -1.0

    def test_sliding_range(self):
        wc = WindowConfig(2.0, window_lead_min=-0.5, window_lead_max=1.0)
        assert wc.sliding_range == 1.5

    def test_str_renders_for_negative_leads(self):
        wc = WindowConfig(2.0, window_lead_min=-5.0, window_lead_max=-5.0)
        text = str(wc)
        assert "merger" in text
        assert "-5.000s" in text


# --------------------------------------------------------------------------- #
# _pad_and_slice
# --------------------------------------------------------------------------- #
class TestPadAndSlice:
    slice_fn = staticmethod(TimeDomainSupervisedAframeDataset._pad_and_slice)

    def test_interior(self):
        x = torch.arange(10).float()
        out = self.slice_fn(x, 2, 5)
        assert torch.equal(out, torch.tensor([2.0, 3.0, 4.0]))

    def test_left_overflow(self):
        x = torch.arange(1, 6).float()  # [1,2,3,4,5]
        out = self.slice_fn(x, -2, 3)
        assert torch.equal(out, torch.tensor([0.0, 0.0, 1.0, 2.0, 3.0]))

    def test_right_overflow(self):
        x = torch.arange(1, 6).float()
        out = self.slice_fn(x, 3, 7)
        assert torch.equal(out, torch.tensor([4.0, 5.0, 0.0, 0.0]))

    def test_both_overflow(self):
        x = torch.arange(1, 4).float()  # [1,2,3]
        out = self.slice_fn(x, -2, 5)
        assert torch.equal(
            out, torch.tensor([0.0, 0.0, 1.0, 2.0, 3.0, 0.0, 0.0])
        )

    def test_fully_before_is_zeros(self):
        x = torch.arange(1, 6).float()
        out = self.slice_fn(x, -10, -5)
        assert torch.equal(out, torch.zeros(5))

    def test_no_wraparound(self):
        # negative start must NOT index from the end of the tensor
        x = torch.arange(1, 6).float()
        out = self.slice_fn(x, -1, 1)
        assert torch.equal(out, torch.tensor([0.0, 1.0]))


# --------------------------------------------------------------------------- #
# slice_waveforms geometry
# --------------------------------------------------------------------------- #
class TestSliceWaveforms:
    R = 3.0  # stored merger sits 3s from the right edge of the waveform
    T = int(8 * SAMPLE_RATE)  # 8s waveforms

    def _setup(self, data_dir, val_waveform_file, lead_min, lead_max):
        ds = _make_dataset(
            data_dir, val_waveform_file, lead_min=lead_min, lead_max=lead_max
        )
        ds.waveform_sampler.right_pad = self.R
        return ds

    def _expected(self, ds, waveforms):
        sr = ds.hparams.sample_rate
        signal_idx = waveforms.shape[-1] - int(
            ds.waveform_sampler.right_pad * sr
        )
        kernel_size = int(ds.window_config.kernel_length * sr) + ds.filter_size
        start = signal_idx - (kernel_size - ds.right_pad_size)
        stop = signal_idx + (kernel_size - ds.left_pad_size)
        return signal_idx, kernel_size, start, stop

    def test_straddling_matches_reference_and_keeps_merger(
        self, data_dir, val_waveform_file
    ):
        ds = self._setup(data_dir, val_waveform_file, 0.0, KERNEL_LENGTH)
        wf = _ramp_waveforms(4, self.T)
        signal_idx, kernel_size, start, stop = self._expected(ds, wf)

        out = ds.slice_waveforms(wf)
        ref = ds._pad_and_slice(wf, start, stop)
        assert torch.equal(out, ref)

        # merger marker (value signal_idx+1) lands at
        # kernel_size-right_pad_size
        merger_local = kernel_size - ds.right_pad_size
        assert out[0, 0, merger_local].item() == signal_idx + 1

    def test_pre_merger_excludes_merger_and_zero_pads(
        self, data_dir, val_waveform_file
    ):
        ds = self._setup(data_dir, val_waveform_file, -5.0, -5.0)
        wf = _ramp_waveforms(4, self.T)
        signal_idx, kernel_size, start, stop = self._expected(ds, wf)
        assert start < 0  # this is the case that used to misbehave

        out = ds.slice_waveforms(wf)
        ref = ds._pad_and_slice(wf, start, stop)
        assert torch.equal(out, ref)

        # the merger marker must NOT appear in a purely pre-merger window
        assert not (out == (signal_idx + 1)).any()
        # leading samples are zero-padded (window starts before the waveform)
        assert out[0, 0, 0].item() == 0.0

    def test_post_merger_matches_reference(self, data_dir, val_waveform_file):
        ds = self._setup(data_dir, val_waveform_file, 4.0, 4.0)
        wf = _ramp_waveforms(4, self.T)
        _, _, start, stop = self._expected(ds, wf)
        out = ds.slice_waveforms(wf)
        ref = ds._pad_and_slice(wf, start, stop)
        assert torch.equal(out, ref)


# --------------------------------------------------------------------------- #
# build_val_batches view slicing
# --------------------------------------------------------------------------- #
class TestBuildValBatches:
    R = 3.0

    def _inputs(self, ds):
        sr = ds.hparams.sample_rate
        sample_size = int(ds.sample_length * sr)
        stride = int(ds.hparams.valid_stride * sr)
        # background long enough to unfold into 5 kernels
        bg_len = sample_size + 4 * stride
        background = torch.randn(len(IFOS), bg_len)
        # 5 signals so they line up 1:1 with the unfolded background kernels
        signals = _ramp_waveforms(5, int(8 * sr))
        params = {"mass_1": torch.arange(5, dtype=torch.float32)}
        return background, signals, params

    def _reference_views(self, ds, X, signals):
        sr = ds.hparams.sample_rate
        kernel_size = X.size(-1)
        signal_idx = signals.shape[-1] - int(
            ds.waveform_sampler.right_pad * sr
        )
        max_start = int(signal_idx - ds.left_pad_size)
        n = ds.hparams.num_valid_views
        if n == 1:
            step = 0
        else:
            step = (kernel_size - ds.left_pad_size - ds.right_pad_size) / (
                n - 1
            )
        views = []
        for i in range(n):
            start = max_start - int(i * step)
            views.append(
                ds._pad_and_slice(signals, start, start + kernel_size)
            )
        return views

    @pytest.mark.parametrize(
        "lead_min,lead_max", [(0.0, KERNEL_LENGTH), (-5.0, -5.0)]
    )
    def test_views_match_padded_reference(
        self, data_dir, val_waveform_file, lead_min, lead_max
    ):
        ds = _make_dataset(
            data_dir, val_waveform_file, lead_min=lead_min, lead_max=lead_max
        )
        ds.waveform_sampler.right_pad = self.R
        background, signals, params = self._inputs(ds)

        # call the base implementation directly to inspect the raw
        # (un-whitened) injected views, which is the geometry this test targets
        X, X_inj, _, _ = BaseAframeDataset.build_val_batches(
            ds, background, signals, params
        )

        refs = self._reference_views(ds, X, signals)
        assert X_inj.shape[0] == ds.hparams.num_valid_views
        for i, ref in enumerate(refs):
            assert torch.allclose(X_inj[i] - X, ref)
        assert not torch.isnan(X_inj).any()


# --------------------------------------------------------------------------- #
# short-waveform warning
# --------------------------------------------------------------------------- #
class TestShortWaveformWarning:
    def test_warns_for_pre_merger_window(
        self, data_dir, val_waveform_file, caplog
    ):
        ds = _make_dataset(
            data_dir, val_waveform_file, lead_min=-50.0, lead_max=-50.0
        )
        # short stored waveform that cannot fill the 50s-pre-merger window
        ds.val_waveforms = torch.zeros(1, len(IFOS), int(8 * SAMPLE_RATE))
        ds.waveform_sampler.right_pad = 3.0
        with caplog.at_level(logging.WARNING):
            ds._warn_if_waveform_too_short()
        assert any("before the merger" in r.message for r in caplog.records)
