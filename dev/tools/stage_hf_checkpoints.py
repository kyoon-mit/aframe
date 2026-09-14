"""Stage denoiser checkpoints for upload to a Hugging Face model repo.

Layout produced under ``--out``::

    denoiser/<family>/<run_name>/<checkpoint>.ckpt   original filename kept
    denoiser/<family>/<run_name>/config.yaml
    denoiser/<family>/<run_name>/wandb.json
    results.csv

``<family>`` is ``merger`` or ``premerger``.

``wandb.json`` records metrics at each checkpoint's own epoch, which is not
the run summary: the summary is always the last epoch, while the monitored
checkpoint sits at whatever epoch minimised ``val/den_loss``.

Usage::

    python stage_hf_checkpoints.py --list
    python stage_hf_checkpoints.py --runs NAME [NAME ...] --out hf_staging
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RUN_ROOTS = [
    REPO / "dev/runs/denoiser_4s/train/merger/fix_loss_grad",
    REPO / "dev/runs/denoise/premerger",
    REPO / "dev/runs/denoise/merger",
]
CONFIG_ROOTS = [
    REPO / "dev/local/train/denoiser/premerger",
    REPO / "dev/local/train/denoiser/merger/fix_loss_grad",
    REPO / "dev/configs/denoise/premerger",
    REPO / "dev/configs/denoise/merger",
]
ENTITY = "kyoon-mit-massachusetts-institute-of-technology"
PROJECT = "DENOISER-SCAN"
METRICS = ["val/rho", "val/gain", "val/den_loss"]


def find_runs() -> dict[str, Path]:
    """Map run name to the directory holding its checkpoints."""
    runs: dict[str, Path] = {}
    for root in RUN_ROOTS:
        if not root.is_dir():
            continue
        for path in sorted(root.glob("*/")):
            checkpoints = sorted(path.rglob("*.ckpt"))
            if checkpoints:
                runs[path.name] = path
    return runs


def find_config(name: str) -> Path | None:
    for root in CONFIG_ROOTS:
        candidate = root / f"{name}.yaml"
        if candidate.is_file():
            return candidate
    return None


def family(entry: dict) -> str:
    """'premerger' or 'merger', taken from where the config lives."""
    config = entry.get("config")
    if config is not None and "premerger" in config.parts:
        return "premerger"
    return "premerger" if entry["name"].startswith("premerger") else "merger"


def checkpoint_epoch(path: Path) -> int | None:
    """Epoch a checkpoint was written at, read from the file itself."""
    try:
        import torch

        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return None
    epoch = payload.get("epoch")
    return int(epoch) - 1 if isinstance(epoch, int) else None


def fetch_history(name: str):
    """Per-epoch metrics for one run, or None if wandb is unreachable."""
    try:
        import wandb

        run = wandb.Api().run(f"{ENTITY}/{PROJECT}/{name}")
        frame = run.history(
            keys=["epoch", *METRICS], samples=100000, pandas=True
        ).dropna()
        return run, frame
    except Exception as error:
        print(f"  wandb lookup failed for {name}: {error}", file=sys.stderr)
        return None, None


def metrics_at(frame, epoch: int | None) -> dict:
    if frame is None or epoch is None or not len(frame):
        return {}
    row = frame.loc[(frame["epoch"] - epoch).abs().idxmin()]
    if abs(int(row["epoch"]) - epoch) > 1:
        return {}
    return {
        "epoch": int(row["epoch"]),
        "val_rho": round(float(row["val/rho"]), 4),
        "val_gain": round(float(row["val/gain"]), 4),
        "val_den_loss": round(float(row["val/den_loss"]), 6),
    }


def describe(name: str, directory: Path) -> dict:
    run, frame = fetch_history(name)
    entry = {
        "name": name,
        "state": run.state if run is not None else "unknown",
        "url": run.url if run is not None else "",
        "config": find_config(name),
        "checkpoints": {},
    }
    for path in sorted(directory.rglob("*.ckpt")):
        epoch = checkpoint_epoch(path)
        entry["checkpoints"][path.name] = {
            "path": path,
            "size_mb": round(path.stat().st_size / 1e6, 1),
            **metrics_at(frame, epoch),
        }
    return entry


def flag(entry: dict) -> str:
    """Mark runs that are probably not worth uploading."""
    if entry["state"] in ("running",):
        return "SKIP: still running, checkpoints will change"
    if entry["state"] in ("crashed", "failed"):
        return "CHECK: did not finish"
    best = max(
        (c.get("val_rho", -9) for c in entry["checkpoints"].values()),
        default=-9,
    )
    if best < 0.1:
        return "CHECK: rho below 0.1"
    if entry["config"] is None:
        return "CHECK: no config found"
    return ""


def command_list(entries: list[dict]) -> None:
    for entry in entries:
        note = flag(entry)
        print(f"\n{entry['name']}  [{entry['state']}]{'  ' + note if note else ''}")
        if entry["config"] is None:
            print("    config: NOT FOUND")
        for filename, info in entry["checkpoints"].items():
            epoch = info.get("epoch")
            rho = info.get("val_rho")
            gain = info.get("val_gain")
            print(
                f"    {filename:20s} {info['size_mb']:6.1f} MB"
                + (f"  epoch {epoch:4d}" if epoch is not None else "  epoch    ?")
                + (f"  rho {rho:.4f}" if rho is not None else "  rho      ?")
                + (f"  gain {gain:.3f}" if gain is not None else "")
            )


def command_stage(entries: list[dict], out: Path) -> None:
    rows = []
    for entry in entries:
        if entry["config"] is None:
            sys.exit(f"{entry['name']}: no config found, refusing to stage")
        target = out / "denoiser" / family(entry) / entry["name"]
        target.mkdir(parents=True, exist_ok=True)
        shutil.copy2(entry["config"], target / "config.yaml")

        record = {
            "url": entry["url"],
            "project": PROJECT,
            "run_id": entry["name"],
            "state": entry["state"],
            "checkpoints": {},
        }
        for filename, info in entry["checkpoints"].items():
            shutil.copy2(info["path"], target / filename)
            record["checkpoints"][filename] = {
                key: value for key, value in info.items()
                if key not in ("path", "size_mb")
            }
            rows.append(
                {
                    "family": family(entry),
                    "run": entry["name"],
                    "checkpoint": filename,
                    "epoch": info.get("epoch", ""),
                    "val_rho": info.get("val_rho", ""),
                    "val_gain": info.get("val_gain", ""),
                    "val_den_loss": info.get("val_den_loss", ""),
                    "size_mb": info["size_mb"],
                }
            )
        (target / "wandb.json").write_text(json.dumps(record, indent=2) + "\n")
        print(
            f"staged {family(entry)}/{entry['name']}"
            f" ({len(entry['checkpoints'])} checkpoints)"
        )

    with (out / "results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {out}/results.csv ({len(rows)} checkpoints)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="show stageable runs")
    parser.add_argument("--runs", nargs="+", default=[], help="run names to stage")
    parser.add_argument("--out", type=Path, default=Path("hf_staging"))
    args = parser.parse_args()

    available = find_runs()
    if not available:
        sys.exit("no run directories with checkpoints found")

    if args.list or not args.runs:
        command_list([describe(n, d) for n, d in available.items()])
        if not args.runs:
            return

    unknown = [n for n in args.runs if n not in available]
    if unknown:
        sys.exit(f"unknown runs: {', '.join(unknown)}")
    command_stage([describe(n, available[n]) for n in args.runs], args.out)


if __name__ == "__main__":
    main()
