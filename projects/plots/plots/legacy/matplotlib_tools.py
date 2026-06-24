"""Matplotlib rendering of the sensitive-volume plots.

This is a drop-in alternative to the bokeh rendering in ``tools.py`` so the
legacy pipeline can emit static PNGs instead of an interactive HTML file.
"""

import matplotlib.pyplot as plt

PALETTE = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
]


def plot_err_bands(ax, x, y, err, color, alpha=0.4, **kwargs):
    ax.fill_between(x, y - err, y + err, color=color, alpha=alpha, **kwargs)


def make_grid(combos):
    num_plots = len(combos)
    if num_plots not in [1, 4]:
        raise ValueError(
            f"Only support 2x2 or 1x1 grids, can't plot {num_plots} combos"
        )

    if num_plots == 1:
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        m1, m2 = combos[0]
        ax.set_title(f"Log Normal $m_1={m1}$, $m_2={m2}$")
        ax.set_xscale("log")
        ax.set_xlabel("False Alarm Rate [yr$^{-1}$]")
        ax.set_ylabel("Sensitive Volume [Gpc$^3$]")
        return fig, [ax]

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    axes = axes.flatten()

    for i, (ax, combo) in enumerate(zip(axes, combos, strict=True)):
        m1, m2 = combo
        ax.set_title(f"Log Normal $m_1={m1}$, $m_2={m2}$")
        ax.set_xscale("log")

        # y-axis label on left column
        if i % 2 == 0:
            ax.set_ylabel("Sensitive Volume [Gpc$^3$]")

        # x-axis label on bottom row
        if i >= 2:
            ax.set_xlabel("False Alarm Rate [yr$^{-1}$]")
        else:
            ax.set_xticklabels([])

    return fig, axes


def plot_sensitive_volume(
    output_path,
    mass_combos,
    fars,
    aframe_sv,
    aframe_err,
    gwtc3_sv,
    gwtc3_err,
    model_name="aframe",
):
    """Render the per-mass-combo SV vs. FAR grid and save it as a PNG.

    Mirrors the bokeh grid produced in ``main.py`` but uses matplotlib.
    """
    fig, axes = make_grid(mass_combos)

    for i, ax in enumerate(axes):
        color = PALETTE[0]
        # only label aframe once so the legend isn't repeated per panel
        label = model_name if i == 0 else None
        ax.plot(
            fars,
            aframe_sv[i],
            linewidth=1.5,
            color=color,
            label=label,
            alpha=1 if i == 0 else 0.5,
            zorder=5 if i == 0 else 1,
        )
        plot_err_bands(ax, fars, aframe_sv[i], aframe_err[i], color=color)

        for pipeline, color in zip(gwtc3_sv.keys(), PALETTE[1:], strict=False):
            m1, m2 = mass_combos[i]
            mass_key = f"{m1}-{m2}"
            sv = gwtc3_sv[pipeline][mass_key]
            err = gwtc3_err[pipeline][mass_key]

            label = pipeline if i == 0 else None
            ax.plot(fars, sv, linewidth=1, color=color, label=label)
            plot_err_bands(ax, fars, sv, err, color=color)

    # legend only on the top-left panel
    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(
        handles,
        labels,
        loc="upper left",
        fontsize=8,
        ncol=2,
        handlelength=1.5,
        borderpad=0.4,
        labelspacing=0.3,
    )

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
