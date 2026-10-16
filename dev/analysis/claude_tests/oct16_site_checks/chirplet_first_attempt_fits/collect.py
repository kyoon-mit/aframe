"""Load fit results; best fit per (Mc, M) among float32-valid fits."""

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
F32_TOL = 1e-3  # a fit counts as float32-valid if its float32 match is within this of float64


def load_runs(outdir="runs"):
    recs = []
    for p in sorted((HERE / outdir).glob("*.json")):
        r = json.load(open(p))
        r["_tag"] = p.stem
        recs.append(r)
    return recs


def valid(r):
    return (r["match_final"] - r["match_final_float32"]) <= F32_TOL


def best_by(recs, key=lambda r: (r["Mc"], r["M"]), filt=None):
    out = {}
    for r in recs:
        if not valid(r) or (filt is not None and not filt(r)):
            continue
        k = key(r)
        if k not in out or r["match_final"] > out[k]["match_final"]:
            out[k] = r
    return out


def fitted_kernel(r, outdir="runs"):
    """Unit-norm fitted kernel of a run (float64)."""
    import core as C
    z = np.load(HERE / outdir / f"{r['_tag']}.npz")
    K = C.kernel_of(z["theta"], z["a"], r["M"])
    return K / np.linalg.norm(K)
