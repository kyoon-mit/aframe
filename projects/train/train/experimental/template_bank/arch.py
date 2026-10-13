"""Template-bank network: waveform templates, network SNR, max over time.

The whitened strain of every detector is correlated with every template of
a fixed bank by FFT: a long convolution, as in S4D, with the kernel taken
from the waveform model instead of generated from tones. The one-sided
(analytic) spectrum gives both quadratures at once, so |z|^2 is the
matched-filter SNR^2 at any phase. Network SNR^2 is rho^2 in the first
detector plus the largest rho^2 of the others within the light travel time,
maximized over the lags where a merger can sit. Templates are whitened once
with an average PSD and have unit norm; the correlation runs at
``sample_rate / decimate``.

The learned part reads the J peak SNR^2 values: a posterior over the bank's
chirp-mass grid, ``beta * SNR^2`` summed over the spins at each chirp mass
(log-sum-exp) plus a per-chirp-mass bias, and a detection logit from the
log mean of that posterior's unnormalized weights.
"""

import math

import numpy as np
import torch
import torch.nn as nn
from ml4gw.spectral import truncate_inverse_power_spectrum


class TemplateBankNet(nn.Module):
    """Args:
        bank_file: npz from ``template_bank.bank`` (templates, grid, PSD).
        num_ifos: detectors in the input.
        sample_rate: of the input, in Hz.
        kernel_length: input length, in seconds.
        decimate: correlate at ``sample_rate / decimate``.
        max_lag: largest offset, in seconds, between the template's merger
            and the data's that is searched, either way.
        coincidence: light travel time between detectors, in seconds.
        fduration, highpass: of the whitening filter, as in the data module.
        beta: initial weight on SNR^2 in the posterior (0.5 is the
            Gaussian-noise likelihood).
        n_bands: frequency bands of the chi^2 test in ``newsnr2``.
        detection_hidden: if positive, the detection logit comes from a
            two-hidden-layer MLP of this width over ``detection_inputs``
            instead of the affine map of the log mean posterior weight.
    """

    def __init__(
        self,
        bank_file: str,
        num_ifos: int = 2,
        sample_rate: float = 2048.0,
        kernel_length: float = 4.0,
        decimate: int = 2,
        max_lag: float = 0.3,
        coincidence: float = 0.01,
        fduration: float = 1.0,
        highpass: float = 20.0,
        beta: float = 0.5,
        n_bands: int = 16,
        detection_hidden: int = 0,
    ):
        super().__init__()
        bank = np.load(bank_file)
        length = int(kernel_length * sample_rate)
        self.decimate = decimate
        self.max_lag = int(max_lag * sample_rate / decimate)
        self.coincidence = int(coincidence * sample_rate / decimate)
        # internal length: no wraparound at the kept lags
        self.n = length // decimate + 2 ** math.ceil(math.log2(self.max_lag + 1))
        self.n_full = self.n * decimate
        self.f = self.n // 2 + 1

        templates = torch.tensor(bank["templates"], dtype=torch.float64)
        spectra = torch.fft.rfft(templates, n=self.n_full)[:, : self.f]   # (J, F)
        psd = torch.tensor(bank["psd"], dtype=torch.float64)[None]        # (1, I, f)
        psd = nn.functional.interpolate(psd, size=(self.n_full // 2 + 1,), mode="linear")
        inv_asd = truncate_inverse_power_spectrum(psd, fduration, sample_rate, highpass) ** -0.5
        inv_asd = torch.nan_to_num(inv_asd, posinf=0.0)[0, :, : self.f]    # (I, F)
        whitened = spectra[None] * inv_asd[:, None]                       # (I, J, F)
        parseval = torch.full((self.f,), 2.0, dtype=torch.float64)
        parseval[0] = 1.0
        sigma = ((whitened.abs() ** 2) * parseval).sum(-1).div(self.n_full).sqrt()
        # analytic weights; 1 / decimate restores unit-variance noise in z
        analytic = torch.full((self.f,), 2.0, dtype=torch.float64)
        analytic[0] = analytic[-1] = 1.0
        filters = whitened.conj() / sigma[..., None] * analytic / decimate
        self.register_buffer("filters", filters.to(torch.complex64), persistent=False)

        grid_index = torch.tensor(bank["grid_index"], dtype=torch.long)
        spins = np.unique(bank["spin"])
        spin_index = torch.tensor(np.searchsorted(spins, bank["spin"]), dtype=torch.long)
        n_grid = len(bank["grid_log_mc"])
        # (G, S) template index per chirp mass and spin, -1 where missing
        table = torch.full((n_grid, len(spins)), -1, dtype=torch.long)
        table[grid_index, spin_index] = torch.arange(len(grid_index))
        self.register_buffer("table", table, persistent=False)
        self.register_buffer("template_grid", grid_index, persistent=False)
        self.register_buffer(
            "grid_log_mc", torch.tensor(bank["grid_log_mc"], dtype=torch.float32), persistent=False
        )
        lags = torch.cat([torch.arange(self.n - self.max_lag, self.n), torch.arange(self.max_lag + 1)])
        self.register_buffer("lags", lags, persistent=False)

        # chi^2 bands: n_bands slices of each filter's power, per detector
        power = filters.abs() ** 2
        share = power.cumsum(-1) / power.sum(-1, keepdim=True)
        self.n_bands = n_bands
        band = (share * n_bands).long().clamp(max=n_bands - 1)
        self.register_buffer("band", band, persistent=False)                # (I, J, F)

        self.log_beta = nn.Parameter(torch.tensor(math.log(beta)))
        self.bias = nn.Parameter(torch.zeros(n_grid))
        self.det_scale = nn.Parameter(torch.tensor(1.0))
        self.det_shift = nn.Parameter(torch.tensor(0.0))
        self.detection_hidden = detection_hidden
        if detection_hidden > 0:
            width = len(self.FEATURES) + 2
            self.det_net = nn.Sequential(
                nn.Linear(width, detection_hidden), nn.GELU(),
                nn.Linear(detection_hidden, detection_hidden), nn.GELU(),
                nn.Linear(detection_hidden, 1),
            )

    @torch.no_grad()
    def snr2_series(self, X: torch.Tensor) -> torch.Tensor:
        """(B, I, L) whitened strain -> (B, I, J, lags) SNR^2 per detector."""
        spectrum = torch.fft.rfft(X.float(), n=self.n_full)[..., : self.f]   # (B, I, F)
        z = torch.fft.ifft(spectrum[:, :, None] * self.filters[None], n=self.n)
        z = z[..., self.lags]
        return z.real**2 + z.imag**2

    @torch.no_grad()
    def peak_snr2(self, X: torch.Tensor) -> torch.Tensor:
        """(B, I, L) whitened strain -> (B, J) peak network SNR^2."""
        rho2 = self.snr2_series(X)
        net = rho2[:, 0]
        if rho2.shape[1] > 1:
            B, I, J, T = rho2.shape
            others = nn.functional.max_pool1d(
                rho2[:, 1:].reshape(-1, 1, T), 2 * self.coincidence + 1, stride=1, padding=self.coincidence
            ).reshape(B, I - 1, J, T)
            net = net + others.sum(1)
        return net.amax(-1)

    FEATURES = (
        "rho2_first", "rho2_second", "chi2_first", "chi2_second", "lag_offset",
        "cos_phase_offset", "sin_phase_offset", "log_chirp_mass",
        "max_rho2_first", "max_rho2_second",
    )

    @torch.no_grad()
    def newsnr2(self, X: torch.Tensor, features: bool = False):
        """Loudest template's network SNR^2, each detector's SNR reweighted
        by its chi^2 over ``n_bands`` bands of equal template power (Allen
        2005; reweighting of Babak et al. 2013), and the plain network SNR^2,
        both (B,); and every template's peak network SNR^2, (B, J). With
        ``features``, also (B, len(FEATURES)) per-window features of the
        loudest template: each detector's SNR^2 and reduced chi^2, the
        second detector's lag offset over the coincidence window, the phase
        offset, the template's log chirp mass, and each detector's largest
        SNR^2 over all templates and lags."""
        spectrum = torch.fft.rfft(X.float(), n=self.n_full)[..., : self.f]
        prod = spectrum[:, :, None] * self.filters[None]                  # (B, I, J, F)
        z = torch.fft.ifft(prod, n=self.n)[..., self.lags]
        rho2 = z.real**2 + z.imag**2                                       # (B, I, J, T)
        B, I, J, T = rho2.shape
        if I != 2:
            raise ValueError("newsnr2 expects two detectors")
        other, where = nn.functional.max_pool1d(
            rho2[:, 1].reshape(-1, 1, T), 2 * self.coincidence + 1, stride=1,
            padding=self.coincidence, return_indices=True,
        )
        net = rho2[:, 0] + other.reshape(B, J, T)
        best = net.reshape(B, -1).argmax(-1)
        rows = torch.arange(B, device=X.device)
        j, t = best // T, best % T
        t_other = where.reshape(B, J, T)[rows, j, t]
        lag = torch.stack([self.lags[t], self.lags[t_other]], 1)          # (B, I)
        k = torch.arange(self.f, device=X.device)
        phase = torch.exp(2j * math.pi * k * lag[..., None] / self.n)
        contrib = prod[rows, :, j] * phase / self.n                        # (B, I, F)
        band = self.band[:, j].transpose(0, 1)                             # (B, I, F)
        zb_re = torch.zeros(B, I, self.n_bands, device=X.device).scatter_add_(-1, band, contrib.real)
        zb_im = torch.zeros(B, I, self.n_bands, device=X.device).scatter_add_(-1, band, contrib.imag)
        zb = torch.complex(zb_re, zb_im)
        total = zb.sum(-1, keepdim=True)
        chi2 = self.n_bands * ((zb - total / self.n_bands).abs() ** 2).sum(-1)
        reduced = chi2 / (2 * self.n_bands - 2)
        weight = torch.where(reduced > 1, ((1 + reduced**3) / 2) ** (-1 / 3), torch.ones_like(reduced))
        rho2_det = total[..., 0].abs() ** 2
        peak = net.amax(-1)
        if not features:
            return (rho2_det * weight).sum(1), peak.amax(-1), peak
        phase_offset = torch.angle(total[:, 0, 0]) - torch.angle(total[:, 1, 0])
        feats = torch.stack(
            [
                rho2_det[:, 0], rho2_det[:, 1], reduced[:, 0], reduced[:, 1],
                (t_other - t).float() / max(self.coincidence, 1),
                torch.cos(phase_offset), torch.sin(phase_offset),
                self.grid_log_mc[self.template_grid[j]],
                rho2[:, 0].amax((1, 2)), rho2[:, 1].amax((1, 2)),
            ],
            1,
        )
        return (rho2_det * weight).sum(1), peak.amax(-1), peak, feats

    def forward(self, X: torch.Tensor):
        """Chirp-mass logits over the grid (B, G) and a detection logit (B,)."""
        if self.detection_hidden <= 0:
            return self.head(self.peak_snr2(X))
        new, plain, peak, feats = self.newsnr2(X, features=True)
        logits, _ = self.head(peak)
        return logits, self.detect(feats, new, plain)

    @staticmethod
    def detection_inputs(feats, new, plain):
        """Features on comparable scales: logs of SNR^2 and chi^2, the
        offsets as they are, the chirp mass standardized over the prior."""
        f = dict(zip(TemplateBankNet.FEATURES, feats.unbind(-1)))
        return torch.stack(
            [
                torch.log1p(f["rho2_first"]), torch.log1p(f["rho2_second"]),
                torch.log(f["chi2_first"].clamp_min(1e-3)), torch.log(f["chi2_second"].clamp_min(1e-3)),
                f["lag_offset"], f["cos_phase_offset"], f["sin_phase_offset"],
                (f["log_chirp_mass"] - 0.43) / 0.34,
                torch.log1p(f["max_rho2_first"]), torch.log1p(f["max_rho2_second"]),
                torch.log1p(new), torch.log1p(plain),
            ],
            -1,
        )

    def detect(self, feats, new, plain):
        """Detection logit from the MLP over the per-window features."""
        return self.det_net(self.detection_inputs(feats, new, plain))[..., 0]

    def head(self, peak: torch.Tensor):
        """(B, J) peak network SNR^2 -> logits (B, G), detection logit (B,)."""
        weights = self.log_beta.exp() * peak
        padded = torch.cat([weights, torch.full_like(weights[:, :1], -torch.inf)], 1)
        logits = torch.logsumexp(padded[:, self.table], -1) + self.bias    # (B, G)
        evidence = torch.logsumexp(logits, -1) - math.log(logits.shape[-1])
        return logits, self.det_scale * evidence + self.det_shift

    def estimate(self, logits: torch.Tensor):
        """Log chirp mass at the posterior peak, refined by a parabola
        through the peak and its neighbours, and the posterior spread."""
        g = self.grid_log_mc
        j = logits.argmax(-1)
        jj = j.clamp(1, len(g) - 2)
        rows = torch.arange(len(j), device=j.device)
        y0, y1, y2 = logits[rows, jj - 1], logits[rows, jj], logits[rows, jj + 1]
        x0, x1, x2 = g[jj - 1], g[jj], g[jj + 1]
        d = (x0 - x1) * (x0 - x2) * (x1 - x2)
        a = (x2 * (y1 - y0) + x1 * (y0 - y2) + x0 * (y2 - y1)) / d
        b = (x2**2 * (y0 - y1) + x1**2 * (y2 - y0) + x0**2 * (y1 - y2)) / d
        vertex = -b / (2 * a)
        inner = (j > 0) & (j < len(g) - 1) & (a < 0) & (vertex > x0) & (vertex < x2)
        log_mc = torch.where(inner, vertex, g[j])
        probs = torch.softmax(logits, -1)
        mean = (probs * g).sum(-1, keepdim=True)
        spread = (probs * (g - mean) ** 2).sum(-1).sqrt()
        return log_mc, spread
