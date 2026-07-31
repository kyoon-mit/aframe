"""Step 4: sensitive volume via aframe's legacy SV pipeline (the real plots).

Thin wrapper around ``plots.legacy.main`` -- the same code kyoon-dev used --
pointed at one aggregated tag dir from step 3. Writes aframe's
``sensitive_volume.h5`` + ``sensitive_volume.html``, and saves the same bokeh
figure to ``sensitive_volume.png`` with bokeh's native export (what the
figure's download button produces).

    uv run python step4_sv_plot.py --dir <.../aggregated/boxcar>
"""

import argparse
from pathlib import Path

import plots.legacy.main as legacy
from bokeh.io import export_png
from bokeh.resources import INLINE
from bokeh.models import LogTicker, LogTickFormatter
from priors.priors import end_o3_ratesandpops_bns
from selenium import webdriver
from selenium.webdriver.chrome.service import Service

REJECTED = "/n/holystore01/LABS/iaifi_lab/Lab/kyoon/DATA/aframe_data/test/rejected-parameters.hdf5"  # noqa: E501
MASS_COMBOS = [
    (1.4, 1.4),
    (1.6, 1.6),
    (1.8, 1.8),
    (2.0, 2.0),
]  # BNS combo present in aframe's catalog comparison
IFOS = ["H1", "L1"]


def _ensure_axis_space(grid):
    """Give each Bokeh plot enough border to render the y-axis cleanly."""

    def walk(obj):
        if hasattr(obj, "renderers") and hasattr(obj, "x_range"):
            yield obj
        elif isinstance(obj, (list, tuple)):
            for child in obj:
                yield from walk(child)

    for plot in walk(grid):
        plot.min_border_left = max(plot.min_border_left, 80)
        plot.min_border_bottom = max(plot.min_border_bottom, 50)
        plot.min_border_right = max(plot.min_border_right, 20)
        plot.min_border_top = max(plot.min_border_top, 20)
        plot.yaxis.axis_label_standoff = 8
        plot.yaxis.major_label_text_font_size = "9pt"

        plot.x_range.start = 1.0
        if getattr(plot, "xaxis", None):
            xaxis = plot.xaxis[0]
            xaxis.ticker = LogTicker(
                base=10, mantissas=[1], minor_thresholds=[]
            )
            xaxis.formatter = LogTickFormatter(base=10, mantissas=[1])
            xaxis.major_label_text_font_size = "9pt"


def make_webdriver():
    """Headless chromium for bokeh's export_png."""
    options = webdriver.ChromeOptions()
    options.binary_location = "/usr/bin/chromium-browser"
    flags = [
        "--headless",
        "--no-sandbox",
        "--disable-gpu",
        "--force-device-scale-factor=3",
    ]
    for flag in flags:
        options.add_argument(flag)
    return webdriver.Chrome(
        service=Service("/usr/bin/chromedriver"), options=options
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dir",
        required=True,
        help="aggregated tag dir with background.hdf5/foreground.hdf5",
    )
    parser.add_argument("--rejected", default=REJECTED)
    parser.add_argument("--max-far", type=float, default=365.0)
    args = parser.parse_args()

    # capture the bokeh grid that legacy main() builds and saves internally
    captured = {}
    original_save = legacy.save

    def capturing_save(grid, *save_args, **save_kwargs):
        captured["grid"] = grid
        _ensure_axis_space(grid)
        # inline bokeh JS so the .html renders with no internet (e.g. in the
        # vscode sandbox); default CDN resources leave a blank page offline.
        save_kwargs.setdefault("resources", INLINE)
        return original_save(grid, *save_args, **save_kwargs)

    legacy.save = capturing_save
    tag_dir = Path(args.dir)
    try:
        legacy.main(
            background=tag_dir / "background.hdf5",
            foreground=tag_dir / "foreground.hdf5",
            rejected_params=Path(args.rejected),
            ifos=IFOS,
            mass_combos=MASS_COMBOS,
            source_prior=end_o3_ratesandpops_bns,
            output_dir=tag_dir,
            max_far=args.max_far,
        )
    finally:
        legacy.save = original_save
    print(f"wrote {tag_dir / 'sensitive_volume.html'}")

    # native bokeh png of the same figure (toolbar not rendered)
    grid = captured["grid"]
    grid.toolbar_location = None
    _ensure_axis_space(grid)
    driver = make_webdriver()
    png_path = tag_dir / "sensitive_volume.png"
    export_png(grid, filename=str(png_path), webdriver=driver, scale_factor=3)
    driver.quit()
    print(f"wrote {png_path}")


if __name__ == "__main__":
    main()
