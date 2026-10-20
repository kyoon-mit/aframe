"""TARP coverage (Lemos, Coogan et al. 2023, arXiv:2302.03026) of ``pe.py``
outputs, as slicgw ``scripts/run_tarp.py``: random references, euclidean
distance, values scaled by the range of the true values, ``n_sims // 5``
credibility bins, 1000 bootstrap draws, and a null band from 10000 sets of
uniform values. Plot as Figure 3 of Legin et al. (2024).

    python -m train.experimental.slic.coverage --pe <pe.h5> [<pe.h5> ...] \\
        [--pe <pe.h5> ...] --labels <name> [<name> ...] --out <folder>

Each ``--pe`` is one set of runs, drawn as one curve. All sets use the
smallest set's number of windows.
"""

# TARP functions follow github.com/Ciela-Institute/tarp drp.py (MIT).

import argparse
from pathlib import Path

import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

NUM_BOOTSTRAP = 1000
NUM_NULL = 10000


def tarp_coverage(samples, theta, references, num_alpha_bins, norm=True):
    """Expected coverage and credibility values, (bins + 1,) each.

    Args:
        samples: (n_samples, n_sims, n_dims).
        theta: (n_sims, n_dims) true values.
        references: (n_sims, n_dims) in the unit cube.
    """
    theta = theta[None]
    if norm:
        low = theta.min(1, keepdims=True)
        high = theta.max(1, keepdims=True)
        samples = (samples - low) / (high - low + 1e-10)
        theta = (theta - low) / (high - low + 1e-10)
    sample_dist = np.sqrt(((references[None] - samples) ** 2).sum(-1))
    theta_dist = np.sqrt(((references - theta) ** 2).sum(-1))
    f = (sample_dist < theta_dist).sum(0) / len(samples)
    h, alpha = np.histogram(f, density=True, bins=num_alpha_bins, range=(0, 1))
    ecp = np.cumsum(h) * (alpha[1] - alpha[0])
    return np.concatenate([[0], ecp]), alpha


def tarp_bootstrap(
    samples, theta, num_alpha_bins, num_bootstrap=NUM_BOOTSTRAP, rng=None
):
    """(num_bootstrap, bins + 1) coverage over windows drawn with
    replacement, each with new random references; and alpha."""
    rng = np.random.default_rng(rng)
    num_sims, num_dims = theta.shape
    ecp = np.empty((num_bootstrap, num_alpha_bins + 1))
    for i in range(num_bootstrap):
        idx = rng.integers(0, num_sims, size=num_sims)
        references = rng.uniform(0, 1, size=(num_sims, num_dims))
        ecp[i], alpha = tarp_coverage(
            samples[:, idx], theta[idx], references, num_alpha_bins
        )
    return ecp, alpha


def null_band(num_sims, num_alpha_bins, num_draws=NUM_NULL, rng=None):
    """Mean and standard deviation of the coverage of uniform values, and
    alpha."""
    rng = np.random.default_rng(rng)
    ecp = np.empty((num_draws, num_alpha_bins + 1))
    for i in range(num_draws):
        f = rng.uniform(0, 1, size=num_sims)
        h, alpha = np.histogram(
            f, density=True, bins=num_alpha_bins, range=(0, 1)
        )
        ecp[i] = np.concatenate([[0], np.cumsum(h) * (alpha[1] - alpha[0])])
    return ecp.mean(0), ecp.std(0), alpha


def load(files):
    """Samples (n_samples, n_sims, 11) and true values (n_sims, 11) of
    ``pe.py`` outputs, windows in file order."""
    samples, theta = [], []
    for path in files:
        with h5py.File(path, "r") as f:
            samples.append(f["samples"][:])
            theta.append(f["theta_true"][:])
    samples = np.concatenate(samples)  # (sims, steps, walkers, dims)
    theta = np.concatenate(theta)
    samples = samples.transpose(1, 2, 0, 3)
    return samples.reshape(-1, *theta.shape), theta


def plot(curves, null, path):
    """Coverage curves (label, alpha, mean, std) over the null band."""
    null_mean, null_std, null_alpha = null
    plt.figure(figsize=(8, 6))
    for label, alpha, mean, std in curves:
        plt.plot(alpha, mean, label=label)
        plt.fill_between(alpha, mean - std, mean + std, alpha=0.6)
    plt.plot(null_alpha, null_mean, color="grey", label="Null Hypothesis")
    for k, shade in ((1, 0.4), (3, 0.2)):
        plt.fill_between(
            null_alpha,
            null_mean - k * null_std,
            null_mean + k * null_std,
            alpha=shade,
            color="grey",
        )
    plt.plot([0.0, 1.0], [0.0, 1.0], "k--")
    plt.xlabel("Expected Coverage Probability")
    plt.ylabel("Coverage")
    plt.legend()
    plt.savefig(path, bbox_inches="tight", dpi=175)
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--pe", nargs="+", action="append", required=True)
    parser.add_argument("--labels", nargs="+", required=True)
    parser.add_argument("--out", required=True, help="folder")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    if len(args.pe) != len(args.labels):
        parser.error("one label per --pe")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    sets = [load(files) for files in args.pe]
    num_sims = min(len(theta) for _, theta in sets)
    num_alpha_bins = num_sims // 5
    print(f"Using {num_sims} windows per set, {num_alpha_bins} bins.")
    null = null_band(num_sims, num_alpha_bins, rng=rng)

    curves, arrays = [], {}
    for label, (samples, theta) in zip(args.labels, sets):
        ecp, alpha = tarp_bootstrap(
            samples[:, :num_sims], theta[:num_sims], num_alpha_bins, rng=rng
        )
        curves.append((label, alpha, ecp.mean(0), ecp.std(0)))
        arrays[f"{label}/ecp"] = ecp
        arrays[f"{label}/alpha"] = alpha
    plot(curves, null, out / "coverage.png")
    np.savez(
        out / "coverage.npz",
        null_mean=null[0],
        null_std=null[1],
        null_alpha=null[2],
        **arrays,
    )
    print(f"Plot saved to {out / 'coverage.png'}")


if __name__ == "__main__":
    main()
