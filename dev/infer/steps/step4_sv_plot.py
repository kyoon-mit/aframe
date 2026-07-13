"""Sensitive volume vs FAR using aframe's own SensitiveVolumePlot.

uv run python make_sv.py --results <dir with background/foreground.hdf5>
"""

import argparse

from bokeh.io import output_file, save
from ledger.events import EventSet, RecoveredInjectionSet
from ledger.injections import InjectionParameterSet
from priors.priors import end_o3_ratesandpops_bns
from plots.vizapp.pages.summary.sv import SensitiveVolumePlot

REJECTED = "/n/holystore01/LABS/iaifi_lab/Lab/kyoon/DATA/aframe_data/test/rejected-parameters.hdf5"  # noqa: E501
MASS_COMBOS = [(1.5, 1.5)]  # BNS combo present in aframe's catalog comparison


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    parser.add_argument("--rejected", default=REJECTED)
    args = parser.parse_args()

    background = EventSet.read(f"{args.results}/background.hdf5")
    foreground = RecoveredInjectionSet.read(f"{args.results}/foreground.hdf5")
    rejected = InjectionParameterSet.read(args.rejected)
    source_prior, _ = end_o3_ratesandpops_bns()

    plot = SensitiveVolumePlot(
        background=background,
        foreground=foreground,
        rejected_params=rejected,
        mass_combos=MASS_COMBOS,
        source_prior=source_prior,
    )
    # aframe's plot.save() has a bug (self.y); save the bokeh grid directly.
    out = f"{args.results}/sensitive_volume.html"
    output_file(out, title="Sensitive Volume")
    save(plot.grid)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
