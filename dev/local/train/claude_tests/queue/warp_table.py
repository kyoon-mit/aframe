"""4 x 3 table: S4D 64x32x4 read-outs x (unwarped merger, warped merger, warped premerger), val within 1/2/5/10%.

Modes: last (latest epoch), best (best epoch), matched (per row, best epoch up to
the smallest epoch all started columns have reached).
"""
import glob, json, os, sys
from wandb.sdk.internal import datastore
from wandb.proto import wandb_internal_pb2 as pb

R = "/home/kyoon/SSM-BNS/aframe/dev/runs/claude_tests"
CACHE = f"{R}/queue/warp_table_cache.json"
ROWS = [("real, mean", "sig_only_reg_s4d_64x32x4"), ("real, max", "sig_only_reg_s4d_max_64x32x4"),
        ("complex, mean", "sig_only_reg_s4d_complex_mean_64x32x4"), ("complex, energy max", "sig_only_reg_s4d_complex_energy_max_64x32x4")]
COLS = [("merger", ""), ("merger, warped", "_warp_merger"), ("premerger, warped", "_warp_premerger")]
KEYS = [f"val/within_{p}pct_chirp_mass" for p in (1, 2, 5, 10)]

try:
    cache = json.load(open(CACHE))
except (OSError, ValueError):
    cache = {}


def scan(f):
    st = os.stat(f)
    sig = f"{st.st_size}:{st.st_mtime}"
    if cache.get(f, {}).get("sig") == sig:
        return cache[f]["rows"]
    out = []
    ds = datastore.DataStore(); ds.open_for_scan(f)
    while (d := ds.scan_data()) is not None:
        r = pb.Record(); r.ParseFromString(d)
        if r.WhichOneof("record_type") != "history":
            continue
        row = {(i.key or "/".join(i.nested_key)): i.value_json for i in r.history.item}
        if KEYS[0] in row:
            out.append({k: json.loads(row[k]) for k in KEYS + ["epoch"] if k in row})
    cache[f] = {"sig": sig, "rows": out}
    return out


def rows(run):
    """Val rows by epoch; a resumed run's later file overrides repeated epochs."""
    by_epoch = {}
    for f in sorted(glob.glob(f"{R}/{run}/wandb/run-*/*.wandb")):
        for r in scan(f):
            by_epoch[int(r.get("epoch", -1))] = r
    return [by_epoch[e] for e in sorted(by_epoch)]


def cell(h, upto=None, best=False):
    if upto is not None:
        h = [r for r in h if r.get("epoch", -1) <= upto]
    if not h:
        return "none"
    if best:
        v = [max(r.get(k, 0) for r in h) for k in KEYS]
    else:
        v = [h[-1].get(k, float("nan")) for k in KEYS]
    return " / ".join(f"{100 * x:.1f}" for x in v) + f" (ep {int(h[-1].get('epoch', -1))})"


mode = sys.argv[1] if len(sys.argv) > 1 else "last"
print(f"mode {mode}")
print(f"| read-out | " + " | ".join(c for c, _ in COLS) + " |")
print("|---|" + "---|" * len(COLS))
for label, base in ROWS:
    hs = [rows(base + suf) for _, suf in COLS]
    upto = None
    if mode == "matched":
        started = [int(h[-1].get("epoch", -1)) for h in hs if h]
        upto = min(started) if started else None
    cells = [cell(h, upto, best=mode in ("best", "matched")) if h else "not started" for h in hs]
    print(f"| {label} | " + " | ".join(cells) + " |")
json.dump(cache, open(CACHE, "w"))
