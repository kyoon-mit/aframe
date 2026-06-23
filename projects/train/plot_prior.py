"""Visualize the effective prior distributions from a training config YAML.

Usage
-----
    cd projects/train
    uv run python plot_prior.py \
        --config configs/ai4gw/chirp_mass_1s_d64_s64_l4.yaml \
        --n_samples 5000 \
        --output /tmp/prior_check.png
"""

import argparse
import importlib
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import yaml


def _import(dotted_path: str):
    module_path, _, name = dotted_path.rpartition(".")
    return getattr(importlib.import_module(module_path), name)


def _instantiate(class_path: str, init_args: dict):
    cls = _import(class_path)
    return cls(**init_args)


def load_prior_from_config(config_path: str):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    prior_cfg = (
        cfg["data"]["init_args"]
        ["waveform_sampler"]["init_args"]
        ["training_prior"]["init_args"]
    )
    conversion_fn_path = prior_cfg["conversion_function"]
    conversion_fn = _import(conversion_fn_path)

    priors = {}
    for param, spec in prior_cfg["priors"].items():
        priors[param] = _instantiate(spec["class_path"], spec.get("init_args", {}))

    from train.prior import AframePrior
    return AframePrior(priors=priors, conversion_function=conversion_fn)


def plot_prior(samples: dict[str, torch.Tensor], output: str):
    params_to_plot = [
        ("chirp_mass", r"Chirp mass $\mathcal{M}_c$ [$M_\odot$]"),
        ("mass_ratio", r"Mass ratio $q$"),
        ("mass_1",     r"$m_1$ [$M_\odot$]"),
        ("mass_2",     r"$m_2$ [$M_\odot$]"),
        ("distance",   r"Distance [Mpc]"),
        ("a_1",        r"Spin $a_1$"),
    ]
    present = [(k, lbl) for k, lbl in params_to_plot if k in samples]

    n_panels = len(present) + 1  # +1 for m1-m2 scatter
    ncols = 3
    nrows = (n_panels + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = axes.flatten()

    for i, (key, label) in enumerate(present):
        ax = axes[i]
        vals = samples[key].numpy()
        ax.hist(vals, bins=60, density=True, color="steelblue", alpha=0.8, edgecolor="none")
        ax.set_xlabel(label, fontsize=11)
        ax.set_ylabel("Density", fontsize=10)
        ax.set_title(f"{key}", fontsize=11)

    # m1-m2 scatter
    scatter_ax = axes[len(present)]
    if "mass_1" in samples and "mass_2" in samples:
        scatter_ax.scatter(
            samples["mass_1"].numpy(),
            samples["mass_2"].numpy(),
            s=2, alpha=0.3, color="steelblue",
        )
        scatter_ax.set_xlabel(r"$m_1$ [$M_\odot$]", fontsize=11)
        scatter_ax.set_ylabel(r"$m_2$ [$M_\odot$]", fontsize=11)
        scatter_ax.set_title("m1 vs m2 (coupling)", fontsize=11)
        # diagonal line m1=m2
        lim = scatter_ax.get_xlim()
        scatter_ax.plot(lim, lim, "k--", lw=0.8, label="m1=m2")
        scatter_ax.legend(fontsize=9)

    for ax in axes[n_panels:]:
        ax.set_visible(False)

    fig.suptitle(f"Prior distributions  (N={len(next(iter(samples.values())))})", fontsize=13)
    fig.tight_layout()
    fig.savefig(output, dpi=150, bbox_inches="tight")
    print(f"Saved → {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to training YAML config")
    parser.add_argument("--n_samples", type=int, default=5000)
    parser.add_argument("--output", default="prior_check.png")
    args = parser.parse_args()

    print(f"Loading prior from {args.config} ...")
    prior = load_prior_from_config(args.config)

    print(f"Sampling {args.n_samples} points ...")
    samples = prior(args.n_samples)
    samples = {k: v.cpu() for k, v in samples.items()}

    print("Parameters available:", list(samples.keys()))
    for k, v in samples.items():
        print(f"  {k:20s}  min={v.min():.4f}  max={v.max():.4f}  mean={v.mean():.4f}")

    plot_prior(samples, args.output)


if __name__ == "__main__":
    main()
