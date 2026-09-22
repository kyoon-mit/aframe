import torch

# G M_sun / c^3 in seconds, so a chirp mass in solar masses enters the
# phase as a time
MSUN_SECONDS = 4.925490947641267e-06


def inspiral_phase(
    freqs: torch.Tensor,
    chirp_mass: torch.Tensor,
    coalescence_time: torch.Tensor,
) -> torch.Tensor:
    """Leading-order inspiral phase, as a function of frequency.

        Psi(f) = 2 pi f t_c + (3 / 128) (pi M f)^(-5/3)

    with the chirp mass M in seconds. This is the 0PN stationary-phase
    term: one term in the chirp mass and one in the coalescence time,
    with no dependence on mass ratio or spin.

    Args:
        freqs: (F,) rfft frequencies in Hz.
        chirp_mass: (B,) chirp masses in solar masses.
        coalescence_time: (B,) seconds from the kernel start to the merger.

    Returns:
        (B, F) phase in radians. The DC bin is zero, where the expression
        diverges.
    """
    mass = (chirp_mass * MSUN_SECONDS).unsqueeze(-1)
    f = freqs.unsqueeze(0).clamp_min(1e-12)
    phase = 2 * torch.pi * f * coalescence_time.unsqueeze(-1)
    phase = phase + (3.0 / 128.0) * (torch.pi * mass * f) ** (-5.0 / 3.0)
    return torch.where(freqs.unsqueeze(0) > 0, phase, torch.zeros_like(phase))


def _rotation(
    length: int,
    chirp_mass: torch.Tensor,
    coalescence_time: torch.Tensor,
    sample_rate: float,
    sign: int,
) -> torch.Tensor:
    """``exp(i sign Psi)`` on the rfft grid, as a complex (B, F) tensor.

    The phase runs to thousands of radians at the low end of the band and
    to tens of thousands through the coalescence-time term, which single
    precision cannot carry: its spacing there is coarse enough to shift
    the demodulation by an appreciable fraction of a radian. So the phase
    is built in double and only the unit-modulus result is cast back.
    """
    dtype = torch.complex64 if chirp_mass.dtype == torch.float32 else None
    freqs = torch.fft.rfftfreq(
        length,
        1 / sample_rate,
        device=chirp_mass.device,
        dtype=torch.float64,
    )
    phase = inspiral_phase(
        freqs, chirp_mass.double(), coalescence_time.double()
    )
    rotation = torch.exp(sign * 1j * phase)
    return rotation if dtype is None else rotation.to(dtype)


def heterodyne(
    x: torch.Tensor,
    chirp_mass: torch.Tensor,
    coalescence_time: torch.Tensor,
    sample_rate: float,
) -> torch.Tensor:
    """Demodulate by the inspiral phase, real channels to I and Q.

    A chirp at this chirp mass has its phase winding removed and is left
    a slowly varying envelope near zero frequency, while broadband noise
    stays broadband. Channel ``c`` becomes channels ``2c`` and ``2c + 1``.

    Args:
        x: (B, C, L) real.
        chirp_mass: (B,) solar masses.
        coalescence_time: (B,) seconds.
        sample_rate: Hz.

    Returns:
        (B, 2C, L) real, interleaved in phase and quadrature.
    """
    length = x.shape[-1]
    rotation = _rotation(length, chirp_mass, coalescence_time, sample_rate, 1)
    spectrum = torch.fft.rfft(x, dim=-1) * rotation.unsqueeze(1)
    # a one-sided spectrum carries only positive frequencies, so the
    # inverse is complex and holds the analytic signal
    analytic = torch.fft.ifft(
        torch.nn.functional.pad(spectrum, (0, length - spectrum.shape[-1])),
        dim=-1,
    )
    return torch.stack([analytic.real, analytic.imag], dim=2).flatten(1, 2)


def dehierodyne(
    z: torch.Tensor,
    chirp_mass: torch.Tensor,
    coalescence_time: torch.Tensor,
    sample_rate: float,
) -> torch.Tensor:
    """Inverse of ``heterodyne``: I and Q back to real channels."""
    length = z.shape[-1]
    analytic = torch.complex(z[:, 0::2], z[:, 1::2])
    rotation = _rotation(length, chirp_mass, coalescence_time, sample_rate, -1)
    spectrum = torch.fft.fft(analytic, dim=-1)[..., : rotation.shape[-1]]
    spectrum = spectrum * rotation.unsqueeze(1)
    return torch.fft.irfft(spectrum, n=length, dim=-1)
