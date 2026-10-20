"""Whitened H1/L1 templates of one test window, built as the data pipeline
builds ``X_clean`` (``DenoiserOnlyAframeDataset`` with the oracle data config).

Steps, all float64:

1. bilby spins to LAL spins (``bilby_spins_to_lalsim``, f_ref 50 Hz);
2. IMRPhenomPv2 through ``TimeDomainCBCWaveformGenerator`` (f_min 20 Hz,
   f_ref 50 Hz, 2048 Hz, coalescence 2 s before the end of ``duration`` s);
3. projection onto H1 and L1 with the window's dec, psi, phi;
4. rescaling to ``snr`` by the pipeline's network SNR of the 11264-sample
   slice from 4.5 s before to 1.0 s after coalescence (PSD stretched to the
   slice's bins by nearest neighbour, f >= 20 Hz);
5. with ``s = 8192 - 2048 * tc``: the generator output moved earlier by
   ``s - floor(s)`` samples with an FFT phase ramp (its first and last
   ``TAPER`` samples tapered first), then the pipeline's circular roll by
   each detector's whole-sample delay (truncated as ml4gw does) and cut of
   the 10240-sample raw kernel at offset ``floor(s)`` of the moved slice;
6. ``Whiten(1.0, 2048, 20.0)`` with the window's PSDs.

The scale of step 4 does not depend on ``tc``. For whole ``s`` in 0..1023
the template is the pipeline's ``X_clean`` of the same waveform.
"""

import math

import torch
from ml4gw import gw
from ml4gw.constants import MSUN, C
from ml4gw.spectral import truncate_inverse_power_spectrum
from ml4gw.waveforms import IMRPhenomPv2
from ml4gw.waveforms.cbc import utils
from ml4gw.waveforms.conversion import (
    bilby_spins_to_lalsim,
    chirp_mass_and_mass_ratio_to_components,
)
from ml4gw.waveforms.generator import (
    EXTRA_CYCLES,
    EXTRA_TIME_FRACTION,
    TimeDomainCBCWaveformGenerator,
)

PARAMETERS = (
    "chirp_mass",
    "mass_ratio",
    "a_1",
    "a_2",
    "tilt_1",
    "tilt_2",
    "phi_12",
    "phi_jl",
    "phase",
    "snr",
    "tc",
)
FIXED = ("dec", "psi", "phi", "theta_jn")

IFOS = ("H1", "L1")
SAMPLE_RATE = 2048
F_MIN = 20.0
F_REF = 50.0
HIGHPASS = 20.0
RIGHT_PAD = 2.0
FDURATION = 1.0
DURATION = 7.0  # seconds of generator output
DISTANCE = 100.0  # Mpc; removed by the SNR rescaling
SLICE = 11264  # samples kept by slice_waveforms
SLICE_MERGER = 9216  # coalescence sample in the slice
KERNEL = 10240  # raw kernel
WINDOW = 8192  # whitened window
PAD = int(FDURATION * SAMPLE_RATE) // 2
TAPER = 1024  # generator output samples tapered at each end before the shift
MAX_OFFSET = 1023  # kernel offsets s run over 0..1023
TC_MIN = (WINDOW - MAX_OFFSET) / SAMPLE_RATE
TC_MAX = WINDOW / SAMPLE_RATE


def whole_sample_delays(dec, phi):
    """Per-detector delays in whole samples, as ``ml4gw.gw.shift_responses``
    computes them from float32 sky angles."""
    _, vertices = gw.get_ifo_geometry(*IFOS)
    theta = torch.pi / 2 - torch.as_tensor(dec, dtype=torch.float32).view(-1)
    phi = torch.as_tensor(phi, dtype=torch.float32).view(-1)
    omega = torch.column_stack(
        [
            torch.sin(theta) * torch.cos(phi),
            torch.sin(theta) * torch.sin(phi),
            torch.cos(theta),
        ]
    ).view(-1, 1, 3)
    dt = -(omega * vertices).sum(axis=-1)
    dt *= SAMPLE_RATE / C
    return torch.trunc(dt).type(torch.int64)[0]


@torch.no_grad()
def chirp_lengths(mass_1, mass_2, s1z, s2z):
    """Samples of each row's frequency grid in
    ``TimeDomainCBCWaveformGenerator`` (its computation, row by row)."""
    mass_1, mass_2 = mass_1.double() * MSUN, mass_2.double() * MSUN
    f_isco = utils.frequency_isco(mass_1, mass_2)
    f_min = torch.minimum(f_isco, torch.tensor(F_MIN, device=f_isco.device))
    tchirp = utils.chirp_time_bound(f_min, mass_1, mass_2, s1z, s2z)
    spin = utils.final_black_hole_spin_bound(s1z, s2z)
    tmerge = utils.merge_time_bound(mass_1, mass_2)
    tmerge = tmerge + utils.ringdown_time_bound(mass_1 + mass_2, spin)
    fstart = utils.chirp_start_frequency_bound(
        (1.0 + EXTRA_TIME_FRACTION) * tchirp, mass_1, mass_2
    )
    tchirp = utils.chirp_time_bound(fstart, mass_1, mass_2, s1z, s2z)
    length = (tchirp + tmerge + 2.0 * EXTRA_CYCLES / f_min) * SAMPLE_RATE
    return 2 ** torch.ceil(torch.log(torch.round(length)) / math.log(2))


class WindowTemplate(torch.nn.Module):
    """Templates for one window.

    Args:
        psds: (2, 2049) PSDs of the window, H1 then L1.
        dec, psi, phi, theta_jn: the window's fixed values.
        duration: seconds of generator output.

    ``forward(theta)`` takes (B, 11) float64 values in ``PARAMETERS`` order
    and returns (B, 2, 8192) float64. Rows are independent: a row whose
    waveform has NaN values comes back NaN, as does its gradient, and
    ``finite`` flags it.
    """

    def __init__(self, psds, dec, psi, phi, theta_jn, duration=DURATION):
        super().__init__()
        f64 = torch.float64
        psds = torch.as_tensor(psds, dtype=f64).view(1, len(IFOS), -1)
        self.generator = TimeDomainCBCWaveformGenerator(
            IMRPhenomPv2(),
            sample_rate=SAMPLE_RATE,
            duration=duration,
            f_min=F_MIN,
            f_ref=F_REF,
            right_pad=RIGHT_PAD,
        )
        self.size = int(duration * SAMPLE_RATE)
        merger = self.size - int(RIGHT_PAD * SAMPLE_RATE)
        self.start = merger - SLICE_MERGER  # slice start in the output
        if self.start < TAPER:
            raise ValueError(f"duration {duration} s is too short")
        theta_jn = torch.tensor(float(theta_jn), dtype=f64)
        self.register_buffer("theta_jn", theta_jn)

        # antenna factors (cross, plus) x (H1, L1)
        tensors, _ = gw.get_ifo_geometry(*IFOS)
        antenna = gw.compute_antenna_responses(
            torch.tensor([math.pi / 2 - float(dec)], dtype=f64),
            torch.tensor([float(psi)], dtype=f64),
            torch.tensor([float(phi)], dtype=f64),
            tensors.to(f64),
            ["cross", "plus"],
        )[0]
        self.register_buffer("antenna", antenna)
        delays = whole_sample_delays(float(dec), float(phi))
        self.register_buffer("delays", delays)

        # network SNR of the slice: nearest-neighbour PSD, f >= 20 Hz
        num_freqs = SLICE // 2 + 1
        snr_psd = torch.nn.functional.interpolate(psds, size=(num_freqs,))
        freqs = torch.fft.rfftfreq(SLICE, 1 / SAMPLE_RATE, dtype=f64)
        mask = (freqs >= HIGHPASS).to(f64)
        self.register_buffer("snr_weight", mask / snr_psd / SAMPLE_RATE**2)

        # phase ramp bins and end tapers of the generator output
        bins = torch.arange(self.size // 2 + 1, dtype=f64)
        self.register_buffer("bins", bins)
        ramp = torch.hann_window(2 * TAPER, periodic=False, dtype=f64)
        taper = torch.ones(self.size, dtype=f64)
        taper[:TAPER], taper[-TAPER:] = ramp[:TAPER], ramp[TAPER:]
        self.register_buffer("taper", taper)

        # slice sample of kernel sample k at offset 0, per detector
        positions = torch.arange(KERNEL)[None] - delays[:, None]
        self.register_buffer("positions", positions)

        # whitening filter of the raw kernel
        kernel_psd = torch.nn.functional.interpolate(
            psds, size=(KERNEL // 2 + 1,), mode="linear"
        )
        window = torch.hann_window(int(FDURATION * SAMPLE_RATE), dtype=f64)
        kernel_psd = truncate_inverse_power_spectrum(
            kernel_psd, window, SAMPLE_RATE, HIGHPASS, None
        )
        self.register_buffer("inv_asd", torch.nan_to_num(kernel_psd**-0.5))

    def polarizations(self, theta):
        """Cross and plus from the generator, (B, size) each. Rows that
        share a frequency grid are generated together; each row equals the
        generator's output for that row alone. Rows without a finite grid
        length are NaN."""
        p = dict(zip(PARAMETERS, theta.unbind(-1)))
        mass_1, mass_2 = chirp_mass_and_mass_ratio_to_components(
            p["chirp_mass"], p["mass_ratio"]
        )
        incl, s1x, s1y, s1z, s2x, s2y, s2z = bilby_spins_to_lalsim(
            self.theta_jn.expand_as(p["phase"]),
            p["phi_jl"],
            p["tilt_1"],
            p["tilt_2"],
            p["phi_12"],
            p["a_1"],
            p["a_2"],
            mass_1,
            mass_2,
            F_REF,
            p["phase"],
        )
        values = dict(
            chirp_mass=p["chirp_mass"],
            mass_ratio=p["mass_ratio"],
            mass_1=mass_1,
            mass_2=mass_2,
            s1x=s1x,
            s1y=s1y,
            s1z=s1z,
            s2x=s2x,
            s2y=s2y,
            s2z=s2z,
            distance=torch.full_like(p["phase"], DISTANCE),
            phic=p["phase"],
            inclination=incl,
        )
        lengths = chirp_lengths(mass_1, mass_2, s1z, s2z)
        finite = torch.isfinite(lengths)
        order, outputs = [], []
        for length in torch.unique(lengths[finite]):
            rows = torch.where(lengths == length)[0]
            group = {k: v[rows] for k, v in values.items()}
            order.append(rows)
            outputs.append(torch.stack(self.generator(**group), 1))
        rows = torch.where(~finite)[0]
        if len(rows):
            order.append(rows)
            shape = (len(rows), 2, self.size)
            outputs.append(theta.new_full(shape, torch.nan))
        order = torch.cat(order)
        pols = torch.cat(outputs)[torch.argsort(order)]
        return pols[:, 0], pols[:, 1]

    def network_snr(self, responses):
        """Pipeline network SNR of (B, 2, 11264) slices, (B,)."""
        df = SAMPLE_RATE / SLICE
        power = torch.fft.rfft(responses, dim=-1).abs() ** 2
        return (4 * df * (power * self.snr_weight).sum(-1)).sum(-1).sqrt()

    def forward(self, theta):
        theta = torch.as_tensor(theta, dtype=torch.float64)
        snr, tc = theta[:, -2], theta[:, -1]
        cross, plus = self.polarizations(theta)
        pols = torch.stack([cross, plus], dim=1)  # (B, 2, size)
        responses = torch.einsum("pi,bpt->bit", self.antenna, pols)
        a = self.start
        scale = snr / self.network_snr(responses[..., a : a + SLICE])

        # moved[j] = responses[j + frac]
        offset = WINDOW - SAMPLE_RATE * tc
        whole = torch.floor(offset.detach())
        frac = (offset - whole)[:, None, None]
        phase = (2 * math.pi / self.size) * frac * self.bins
        moved = torch.fft.irfft(
            torch.fft.rfft(responses * self.taper, dim=-1)
            * torch.polar(torch.ones_like(phase), phase),
            n=self.size,
            dim=-1,
        )[..., a : a + SLICE]

        # pipeline roll by the delays and cut at offset floor(s)
        index = self.positions[None] + whole.long()[:, None, None]
        kernel = torch.gather(moved, -1, torch.remainder(index, SLICE))
        return self.whiten(kernel * scale[:, None, None])

    def whiten(self, kernel):
        """``ml4gw.spectral.whiten`` of the raw kernel, kept in float64."""
        kernel = kernel - kernel.mean(-1, keepdim=True)
        kernel_fft = torch.fft.rfft(kernel, norm="forward", dim=-1)
        kernel_fft = kernel_fft * self.inv_asd
        white = torch.fft.irfft(kernel_fft, n=KERNEL, norm="forward", dim=-1)
        return white[..., PAD:-PAD] / SAMPLE_RATE**0.5

    @staticmethod
    def finite(*tensors):
        """(B,) True where every value of every tensor's row is finite."""
        ok = None
        for t in tensors:
            row = torch.isfinite(t).flatten(1).all(1)
            ok = row if ok is None else ok & row
        return ok
