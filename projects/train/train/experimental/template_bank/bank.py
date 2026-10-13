"""Build a template bank from the injection waveform file.

Templates are waveforms from the file itself (h_plus), so they use the same
waveform model as the injections. One template per point of a log chirp-mass
grid and per target effective spin, mass ratio near ``q_target``, smallest
in-plane spin among the candidates. Each is aligned so its merger (peak of
h_plus^2 + h_cross^2) sits at ``merge_index`` of a ``length``-sample frame,
and the start is tapered.

usage: python -m train.experimental.template_bank.bank WAVEFORMS PSDS OUT [SPINS]
SPINS: comma-separated target effective spins, default -0.1,0,0.1
"""

import sys

import h5py
import numpy as np


def chirp_mass(m1, m2):
    return (m1 * m2) ** 0.6 / (m1 + m2) ** 0.2


def build_bank(
    waveform_file: str,
    mc_min: float = 0.85,
    mc_max: float = 2.75,
    step: float = 0.005,
    spins: tuple = (-0.1, 0.0, 0.1),
    spin_tol: float = 0.03,
    q_target: float = 0.8,
    q_tol: float = 0.1,
    length: int = 8192,
    merge_index: int = 7680,
    taper: float = 0.25,
    sample_rate: float = 2048.0,
) -> dict:
    """Templates (J, length) and their chirp mass, effective spin, grid index."""
    f = h5py.File(waveform_file, "r")
    p = f["parameters"]
    m1, m2 = p["mass_1"][:], p["mass_2"][:]
    a1, a2, t1, t2 = p["a_1"][:], p["a_2"][:], p["tilt_1"][:], p["tilt_2"][:]
    mc = chirp_mass(m1, m2)
    q = np.minimum(m1, m2) / np.maximum(m1, m2)
    chi = (m1 * a1 * np.cos(t1) + m2 * a2 * np.cos(t2)) / (m1 + m2)
    inplane = a1 * np.abs(np.sin(t1)) + a2 * np.abs(np.sin(t2))

    grid = np.exp(np.arange(np.log(mc_min), np.log(mc_max) + 1e-9, step))
    rows, grid_index, spin = [], [], []
    for g, center in enumerate(grid):
        near = np.where(
            (np.abs(np.log(mc / center)) < step / 4) & (np.abs(q - q_target) < q_tol)
        )[0]
        for s in spins:
            cand = near[np.abs(chi[near] - s) < spin_tol]
            if len(cand):
                rows.append(cand[np.argmin(np.abs(chi[cand] - s) / spin_tol + inplane[cand])])
                grid_index.append(g)
                spin.append(s)
    rows = np.array(rows)
    order = np.argsort(rows)  # h5py reads need increasing indices
    plus = np.empty((len(rows), f["waveforms/plus"].shape[-1]))
    cross = np.empty_like(plus)
    plus[order] = f["waveforms/plus"][rows[order]]
    cross[order] = f["waveforms/cross"][rows[order]]

    window = np.ones(length)
    ramp = int(taper * sample_rate)
    window[:ramp] = 0.5 * (1 - np.cos(np.pi * np.arange(ramp) / ramp))
    templates = np.zeros((len(rows), length), np.float32)
    for j in range(len(rows)):
        peak = int(np.argmax(plus[j] ** 2 + cross[j] ** 2))
        padded = np.pad(plus[j], (length, length))
        start = peak + length - merge_index
        templates[j] = padded[start : start + length] * window
    return {
        "templates": templates,
        "chirp_mass": mc[rows],
        "spin": np.array(spin),
        "true_spin": chi[rows],
        "mass_ratio": q[rows],
        "grid_index": np.array(grid_index),
        "grid_log_mc": np.log(grid),
        "rows": rows,
        "merge_index": merge_index,
        "sample_rate": sample_rate,
    }


def average_psd(psd_file: str, n: int = 4000) -> np.ndarray:
    """Mean one-sided PSD per detector, (num_ifos, num_freqs)."""
    with h5py.File(psd_file, "r") as f:
        return f["psds"][:n].mean(0)


if __name__ == "__main__":
    waveform_file, psd_file, out = sys.argv[1:4]
    spins = tuple(float(s) for s in sys.argv[4].split(",")) if len(sys.argv) > 4 else (-0.1, 0.0, 0.1)
    bank = build_bank(waveform_file, spins=spins)
    bank["psd"] = average_psd(psd_file)
    bank["psd_file"] = psd_file
    bank["waveform_file"] = waveform_file
    np.savez(out, **bank)
    print(f"{len(bank['templates'])} templates on {len(bank['grid_log_mc'])} chirp masses -> {out}")
