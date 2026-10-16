"""Sensitive volume of several statistics against den-reg and the O3 pipelines.

usage: sv_compare.py LABEL=AGG_DIR [LABEL=AGG_DIR ...]
AGG_DIR holds sensitive_volume.h5 and gwtc-3_pipeline_sv.h5 (step 4 output).
"""

import sys
from pathlib import Path

import h5py
import numpy as np

COMBOS = ("1.4-1.4", "1.6-1.6", "1.8-1.8", "2.0-2.0")
runs = [a.split("=", 1) for a in sys.argv[1:]]
ref = h5py.File(Path(runs[0][1]) / "gwtc-3_pipeline_sv.h5", "r")
fars = ref["far"][:]
for i, far in enumerate(fars):
    print(f"\nFAR {far:.0f}/yr: sensitive volume (Gpc^3), and fraction of MBTA's")
    print(f"{'statistic':34s} " + " ".join(f"{c:>18s}" for c in COMBOS))
    for label, d in runs:
        f = Path(d) / "sensitive_volume.h5"
        if not f.exists():
            print(f"{label:34s} (missing)")
            continue
        sv = h5py.File(f, "r")
        cells = []
        for c in COMBOS:
            v, e = sv[c]["sv"][i], sv[c]["err"][i]
            cells.append(f"{v:.2e}±{e:.0e} {100 * v / ref['mbta'][c]['sv'][i]:4.1f}%")
        print(f"{label:34s} " + " ".join(f"{x:>18s}" for x in cells))
    for pipe in ("mbta", "gstlal", "pycbc_hyperbank"):
        print(f"{pipe:34s} " + " ".join(f"{ref[pipe][c]['sv'][i]:18.2e}" for c in COMBOS))
