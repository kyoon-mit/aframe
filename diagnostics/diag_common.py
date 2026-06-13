"""Shared building blocks for the regression diagnostic dump scripts.

Both diagnostics need the same three things:

  1. the trained model plus its preprocessing (resample -> PSD -> whiten -> input
     norm), loaded exactly as ``regression_infer.py`` loads them, so the numbers
     match the sensitive-volume pipeline;
  2. one background segment loaded as a contiguous strain array, optionally with
     the injection population added on top (the "foreground");
  3. a way to score an arbitrary batch of fixed-length windows and read back the
     *physical* chirp-mass mean and sigma (not the negated statistic).

This module provides those three things so the two diagnostic scripts stay short
and focused on what they actually plot.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import jsonargparse
import numpy as np
import torch
import torch.nn as nn
from ml4gw.transforms import Whiten

from train.regression_infer import RegressionSequence
from utils.preprocessing import PsdEstimator


# --------------------------------------------------------------------------- #
# config                                                                      #
# --------------------------------------------------------------------------- #

def load_infer_config(config_path: str) -> dict:
    """Read a ``regression_infer`` YAML into a plain dict (no model building)."""
    parser = jsonargparse.ArgumentParser()
    parser.add_argument("--config", action=jsonargparse.ActionConfigFile)
    # accept any of the regression_infer keys without validating against main()
    from train.regression_infer import main as _infer_main
    parser.add_function_arguments(_infer_main)
    cfg = parser.parse_args(["--config", config_path])
    cfg_dict = vars(cfg)
    cfg_dict.pop("config", None)
    return cfg_dict


# --------------------------------------------------------------------------- #
# scorer                                                                       #
# --------------------------------------------------------------------------- #

@dataclass
class Geometry:
    """Window timing, all derived from the inference config."""

    raw_sample_rate: float        # rate of the background files on disk
    sample_rate: float            # rate the model runs at (after resample)
    inference_sampling_rate: float
    sample_length: float          # full window length fed in: psd + fduration + kernel (s)
    kernel_length: float
    fduration: float
    psd_length: float
    integration_window_length: float
    window_offset: float

    @property
    def window_samples_raw(self) -> int:
        return int(self.sample_length * self.raw_sample_rate)

    @property
    def stride_raw(self) -> int:
        return int(self.raw_sample_rate / self.inference_sampling_rate)

    @property
    def right_edge_offset(self) -> float:
        """Seconds from a window's start to the kernel's *right edge*.

        The window is [psd_length | fduration/2 | kernel | fduration/2]; PSD is
        estimated on the first psd_length, whitening trims fduration/2 from each
        side of the tail, so the kernel the model actually sees ends this many
        seconds after the window start. (Matches ``verify_model_response.py``.)
        """
        return self.psd_length + self.fduration / 2 + self.kernel_length

    def window_start_sample(self, t0: float, coalescence_gps: float, e: float) -> int:
        """Sample index of the window whose kernel right-edge sits ``e`` seconds
        from coalescence (e=0 -> kernel ends exactly at the merger; e<0 ->
        kernel ends e seconds *before* the merger, i.e. pre-merger)."""
        gps = coalescence_gps + e - self.right_edge_offset
        return int(round((gps - t0) * self.raw_sample_rate))


def windows_for_edges(strain, t0, geom: "Geometry", center_gps: float, e_values):
    """Slice the windows whose kernel right-edge lands at ``center_gps + e`` for
    each e in ``e_values``. Windows off the segment are dropped.

    Returns (windows (S, n_ifos, W), e_kept (S,)) or (None, None) if none fit.
    """
    W = geom.window_samples_raw
    n = strain.shape[1]
    wins, kept = [], []
    for e in e_values:
        s = geom.window_start_sample(t0, center_gps, e)
        if s < 0 or s + W > n:
            continue
        wins.append(strain[:, s : s + W])
        kept.append(e)
    if not wins:
        return None, None
    return np.stack(wins).astype(np.float32), np.asarray(kept)


class Scorer:
    """Loads the model + preprocessing and scores batches of windows.

    ``score(windows)`` takes raw-rate windows ``(B, n_ifos, window_samples_raw)``
    and returns physical ``(mean, sigma)`` for the first target parameter
    (chirp mass), each shape ``(B,)``.
    """

    def __init__(self, cfg: dict, device: str = "cuda") -> None:
        self.geom = Geometry(
            raw_sample_rate=cfg.get("raw_sample_rate") or cfg["sample_rate"],
            sample_rate=cfg["sample_rate"],
            inference_sampling_rate=cfg["inference_sampling_rate"],
            sample_length=cfg["kernel_length"] + cfg["fduration"] + cfg["psd_length"],
            kernel_length=cfg["kernel_length"],
            fduration=cfg["fduration"],
            psd_length=cfg["psd_length"],
            integration_window_length=cfg["integration_window_length"],
            window_offset=cfg.get("window_offset", 0.0),
        )
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        from train.model.regression import LitS4DGaussianNLL, LitLinOSSGaussianNLL
        cls_map = {
            "LitS4DGaussianNLL": LitS4DGaussianNLL,
            "LitLinOSSGaussianNLL": LitLinOSSGaussianNLL,
        }
        model = cls_map[cfg["model_class"]].load_from_checkpoint(
            cfg["checkpoint"], strict=False
        )
        model.eval()
        self.model = model.to(self.device)
        self.n_vars = model.n_vars
        self.softplus = nn.Softplus()

        # de-normalization buffers (physical = norm * y_std + y_mean)
        self.y_mean = model.y_mean.detach().cpu().numpy()
        self.y_std = model.y_std.detach().cpu().numpy()

        # resampler (raw rate -> model rate), if needed
        self.resampler = None
        if self.geom.raw_sample_rate != self.geom.sample_rate:
            import torchaudio
            self.resampler = torchaudio.transforms.Resample(
                int(self.geom.raw_sample_rate), int(self.geom.sample_rate)
            ).to(self.device)

        window_length = cfg["kernel_length"] + cfg["fduration"]
        self.psd_estimator = PsdEstimator(
            window_length,
            cfg["sample_rate"],
            cfg["fftlength"],
            average="median",
            fast=cfg.get("highpass") is not None,
        ).to(self.device)
        self.whitener = Whiten(
            cfg["fduration"],
            cfg["sample_rate"],
            cfg.get("highpass"),
            cfg.get("lowpass"),
        ).to(self.device)

    @torch.no_grad()
    def score(self, windows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """windows: (B, n_ifos, window_samples_raw) -> (mean_phys, sigma_phys).

        Guards a quirk in ``PsdEstimator.forward``: a batch of exactly 2 is
        treated as a ``[psd_source, data]`` pair and collapsed to 1 output. We
        never mean that here (each window is independent), so pad a size-2 batch
        to 3 with a throwaway copy and drop it from the result.
        """
        n_in = len(windows)
        pad = n_in == 2
        if pad:
            windows = np.concatenate([windows, windows[:1]], axis=0)
        x = torch.from_numpy(np.ascontiguousarray(windows)).float().to(self.device)
        if self.resampler is not None:
            x = self.resampler(x)
        x, psds = self.psd_estimator(x)
        x = self.whitener(x, psds)
        x = self.model._prepare_input(x)
        out = self.model(x)
        mean_norm = out[:, : self.n_vars]
        sigma_norm = torch.sqrt(self.softplus(out[:, self.n_vars :]))
        mean_phys = mean_norm.cpu().numpy() * self.y_std + self.y_mean
        sigma_phys = sigma_norm.cpu().numpy() * self.y_std
        m, s = mean_phys[:, 0], sigma_phys[:, 0]
        if pad:
            m, s = m[:n_in], s[:n_in]
        return m, s

    @torch.no_grad()
    def score_batched(
        self, windows: np.ndarray, batch_size: int = 256
    ) -> tuple[np.ndarray, np.ndarray]:
        """Same as ``score`` but chunks large inputs to fit in GPU memory."""
        means, sigmas = [], []
        for i in range(0, len(windows), batch_size):
            m, s = self.score(windows[i : i + batch_size])
            means.append(m)
            sigmas.append(s)
        return np.concatenate(means), np.concatenate(sigmas)


# --------------------------------------------------------------------------- #
# segment loading                                                             #
# --------------------------------------------------------------------------- #

@dataclass
class Segment:
    """One background segment, with and without injections added."""

    background: np.ndarray                # (n_ifos, n_samples) raw rate
    foreground: Optional[np.ndarray]      # same, + injections; None if no injections
    t0: float                             # GPS start of the array
    injection_set: object                 # InterferometerResponseSet for this segment


def load_segment(cfg: dict, background_fname: str) -> Segment:
    """Load one segment at zero time-shift and add the injection population.

    Zero shift means the foreground injections sit at their true coalescence
    times -- which is what both diagnostics want (no slide, signals intact).
    """
    n_ifos = len(cfg["ifos"])
    seq = RegressionSequence(
        background_fname=background_fname,
        injection_set_fname=cfg["injection_set_fname"],
        ifos=cfg["ifos"],
        shifts=[0.0] * n_ifos,
        sample_length=cfg["kernel_length"] + cfg["fduration"] + cfg["psd_length"],
        inference_sampling_rate=cfg["inference_sampling_rate"],
        batch_size=cfg.get("batch_size", 256),
    )
    background = seq._load_shifted()  # (n_ifos, n) -- zero shift => full segment
    foreground = None
    if seq.injection_set is not None:
        injected = seq.injection_set.inject(background.copy(), seq.t0)
        foreground = injected[:, : background.shape[1]]
    return Segment(
        background=background,
        foreground=foreground,
        t0=seq.t0,
        injection_set=seq.injection_set,
    )


def list_background_files(cfg: dict) -> list[str]:
    import glob
    files = sorted(glob.glob(str(Path(cfg["background_dir"]) / "*.hdf5")))
    if not files:
        raise FileNotFoundError(f"No HDF5 files in {cfg['background_dir']}")
    return files
