"""Signal-only windows with the time axis warped by tau = (t_c - t)^(5/8)."""
import os

import torch

import train.data.supervised.supervised as _sup
from train.data.supervised.denoiser import DenoiserOnlyAframeDataset


class WarpedDenoiserOnlyAframeDataset(DenoiserOnlyAframeDataset):
    """``DenoiserOnlyAframeDataset`` whose ``X_clean`` is resampled on a grid
    uniform in tau = (t_c - t)^(5/8), where the leading-order chirp is a tone.

    Args:
        merger_offset:
            Seconds by which the window is moved before the merger. 0 keeps
            the usual placement (merger 3.5-4.0 s into the window, set by
            ``left_pad``/``right_pad``); 25.5 puts the window end 25.0-25.5 s
            before the merger.
        warp_rate:
            Sample rate of the warped output. ``None`` keeps ``sample_rate``.
            When lower, the signal is low-passed at its Nyquist frequency first.
        upsample:
            Band-limited upsampling factor before linear interpolation.

    With the merger in the window, t_c is each detector's amplitude peak and
    the output ends there; otherwise t_c comes from the window's position.
    """

    def __init__(self, *args, merger_offset: float = 0.0, warp_rate: float | None = None,
                 upsample: int = 8, **kwargs):
        super().__init__(*args, **kwargs)
        self.merger_offset = merger_offset
        self.warp_rate = warp_rate
        self.upsample = upsample

    @property
    def offset_size(self) -> int:
        return int(round(self.merger_offset * self.hparams.sample_rate))

    def slice_waveforms(self, waveforms: torch.Tensor) -> torch.Tensor:
        signal_idx = waveforms.shape[-1] - int(self.waveform_sampler.right_pad * self.hparams.sample_rate)
        signal_idx -= self.offset_size
        kernel_size = int(self.hparams.kernel_length * self.hparams.sample_rate) + self.filter_size
        start = signal_idx - (kernel_size - self.right_pad_size)
        stop = signal_idx + (kernel_size - self.left_pad_size)
        if start < 0:
            raise ValueError(f"merger_offset {self.merger_offset} s puts the window before the waveform start")
        return waveforms[..., start:stop]

    def inject(self, X, waveforms, params):
        original = _sup.sample_kernels
        captured = {}

        def sample_kernels(*a, **k):
            kernels, idx = original(*a, return_idx=True, **k)
            captured["idx"] = idx
            return kernels

        _sup.sample_kernels = sample_kernels
        try:
            X, X_clean, y, params_out = super().inject(X=X, waveforms=waveforms, params=params)
        finally:
            _sup.sample_kernels = original
        idx = captured["idx"].to(X_clean.device).float()
        if idx.ndim > 1:
            idx = idx[:, 0]
        fs = self.hparams.sample_rate
        # merger position in the whitened window, from where the kernel was cut
        t_c = (self.right_slack_size() - idx + self.offset_size) / fs
        t_c = t_c[:, None].expand(-1, X_clean.shape[1])
        if self.merger_offset == 0:
            geo = t_c
            t_c = self.peak_time(X_clean)
            if self._debug():
                d = (t_c - geo).abs()
                print(f"WARP_DEBUG peak-geometric t_c: median {d.median():.4f} s, max {d.max():.4f} s; "
                      f"t_c range {t_c.min():.3f}-{t_c.max():.3f} s")
        elif self._debug():
            print(f"WARP_DEBUG premerger t_c range {t_c.min():.3f}-{t_c.max():.3f} s (window 0-4 s)")
        Xw = self.warp(X_clean, t_c)
        if self._debug():
            print(f"WARP_DEBUG X_clean {tuple(X_clean.shape)} -> {tuple(Xw.shape)}")
        return X, Xw, y, params_out

    def _debug(self) -> bool:
        """True for the first two batches when WARP_DEBUG is set."""
        if not os.environ.get("WARP_DEBUG"):
            return False
        self._ndebug = getattr(self, "_ndebug", 0) + 1
        return self._ndebug <= 6

    def right_slack_size(self) -> int:
        """Merger index in the whitened window when the kernel starts at slice index 0."""
        kernel_size = int(self.hparams.kernel_length * self.hparams.sample_rate) + self.filter_size
        return kernel_size - self.right_pad_size - self.filter_size // 2

    def peak_time(self, X: torch.Tensor) -> torch.Tensor:
        """Time (s) of the analytic-signal amplitude peak, per row and detector."""
        L = X.shape[-1]
        F = torch.fft.fft(X, dim=-1)
        h = torch.zeros(L, device=X.device)
        h[0] = 1
        h[1:(L + 1) // 2] = 2
        if L % 2 == 0:
            h[L // 2] = 1
        env = torch.fft.ifft(F * h, dim=-1).abs()
        return env.argmax(dim=-1).float() / self.hparams.sample_rate

    def warp(self, X: torch.Tensor, t_c: torch.Tensor) -> torch.Tensor:
        fs = self.hparams.sample_rate
        B, C, L = X.shape
        out_rate = self.warp_rate or fs
        n_out = int(round(L * out_rate / fs))
        F = torch.fft.rfft(X, dim=-1)
        if out_rate < fs:
            F[..., int(L * (out_rate / 2) / fs) + 1:] = 0
        up = self.upsample
        Xu = torch.fft.irfft(F, n=L * up, dim=-1) * up
        t_end = L / fs
        tau_hi = t_c.clamp(min=1e-6) ** 0.625
        tau_lo = (t_c - t_end).clamp(min=0) ** 0.625
        k = torch.linspace(0, 1, n_out, device=X.device)
        tau = tau_hi[..., None] - (tau_hi - tau_lo)[..., None] * k
        t = t_c[..., None] - tau.clamp(min=0) ** 1.6
        pos = (t * fs * up).clamp(0, L * up - 1)
        i0 = pos.floor().long().clamp(max=L * up - 2)
        w = pos - i0
        a = torch.gather(Xu, -1, i0)
        b = torch.gather(Xu, -1, i0 + 1)
        return a * (1 - w) + b * w
