"""Check that fits/summary.json tables equal the tables rebuilt from the raw run files by report.py's own code."""

import ast
import glob
import json
import sys
import types
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
FITS = Path(__file__).resolve().parent.parent / "fits"


def pick(path, names, ns):
    """Execute only the named top-level functions and assignments of a source file."""
    body = []
    for n in ast.parse(Path(path).read_text()).body:
        if isinstance(n, ast.FunctionDef) and n.name in names:
            body.append(n)
        elif isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in names for t in n.targets):
            body.append(n)
    exec(compile(ast.Module(body=body, type_ignores=[]), str(path), "exec"), ns)
    missing = [k for k in names if k not in ns]
    assert not missing, missing
    return ns


cv = pick(FITS / "common.py", ["MCS", "NS"], {})
C = types.SimpleNamespace(MCS=cv["MCS"], NS=cv["NS"], OUT_DIR=FITS)
ns = pick(FITS / "report.py", ["STARTS", "THRESH", "load", "tables"],
          {"np": np, "json": json, "glob": glob, "C": C, "D": FITS})
T = ns["tables"](*ns["load"]())
S = json.load(open(FITS / "summary.json"))["tables"]

KEYS = ["pencil_reflected_tone_check", "pencil_clipped_no_fit", "fourier_bestN_undamped",
        "chirp_placement_no_fit_f32", "varpro_best_f32_valid", "varpro_best_f64_pure_ls",
        "pencil_growing_allowed_not_s4d"]
n, worst = 0, 0.0
for Mc in C.MCS:
    m = str(Mc)
    assert T[m]["int_tau_df_dominant"] == S[m]["int_tau_df_dominant"]
    assert sorted(T[m]["rows"]) == sorted(int(k) for k in S[m]["rows"])
    for p, row in T[m]["rows"].items():
        srow = S[m]["rows"][str(p)]
        for k in KEYS:
            assert (k in row) == (k in srow), (Mc, p, k)
            if k in row:
                worst = max(worst, abs(row[k] - srow[k]))
                n += 1
        assert ("random_restarts" in row) == ("random_restarts" in srow), (Mc, p)
        if "random_restarts" in row:
            worst = max(worst, abs(row["random_restarts"]["best_f32_on_path"]
                                   - srow["random_restarts"]["best_f32_on_path"]))
            n += 1
        assert ("pencil_reflected_tone_check" in row) == (p in C.NS)
print(f"{n} plotted values compared, max abs difference {worst:.3g}")
