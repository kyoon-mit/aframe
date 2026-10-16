"""Stage the Claude test runs for the Hugging Face model repo, by website date.

Layout produced under ``--out``::

    claude_tests/README.md                    index by date, with update links
    claude_tests/results.csv                  one row per run
    claude_tests/runs.json                    the run map used
    claude_tests/<date>/<run>/config.yaml     training config in the repo, if any
    claude_tests/<date>/<run>/config_saved.yaml   saved in the run folder at its latest launch
    claude_tests/<date>/<run>/base_config.yaml    if the run folder has one
    claude_tests/<date>/<run>/wandb_config.yaml   values wandb recorded, if the map lists them
    claude_tests/<date>/<run>/repo_<name>.yaml    other repo configs the map lists
    claude_tests/<date>/<run>/checkpoints/...     from <wandb project>/<run id>/checkpoints/
    claude_tests/<date>/<run>/run.json        status, best value, updates, files
    claude_tests/<date>/<name>/<file>         files outside run folders (map "extras")

``<date>`` is the date of the earliest BNS update that reports the run, or
``unreported``. Runs mapped to ``exclude`` are listed in the README only.
Checkpoints are hard links to the run folders when possible.

Usage::

    python stage_hf_claude_tests.py stage --map claude_tests_hf_map.json --out DIR
    uv run --with huggingface_hub python stage_hf_claude_tests.py upload --out DIR

Staging needs torch (every checkpoint is loaded once on CPU as a check).
Uploading needs huggingface_hub and 'hf auth login' with a write token.
"""

from __future__ import annotations

import argparse
import csv
import filecmp
import json
import os
import re
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RUNS = REPO / "dev/runs/claude_tests"
CONFIGS = REPO / "dev/local/train/claude_tests"
SITE_DOCS = REPO.parent / "WEB/kyoon-mit.github.io/docs/_updates"
SITE_URL = "https://kyoon-mit.github.io/projects/bns/updates/"
WANDB = "https://wandb.ai/kyoon-mit-massachusetts-institute-of-technology"
HF_REPO = "kyoon-mit/SSM-BNS-WIP"
CKPT_DIR = re.compile(r"^([^/]+)/([^/]+)/checkpoints/")


def update_titles() -> dict[str, str]:
    """Map update file stem to its front-matter title."""
    titles = {}
    for path in sorted(SITE_DOCS.glob("*.md")):
        match = re.search(r'^title:\s*"?(.*?)"?\s*$', path.read_text(), re.M)
        titles[path.stem] = match.group(1) if match else path.stem
    return titles


def link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def staged_name(rel: str, taken: set[str]) -> str:
    """Checkpoint path inside the staged run folder: <project>/<id>/checkpoints/x -> checkpoints/x."""
    short = CKPT_DIR.sub("checkpoints/", rel)
    return short if short not in taken else rel


def wandb_link(name: str, rels: list[str]) -> str:
    """wandb page of the run, from the <project>/<id>/checkpoints/ folder that holds its checkpoints."""
    for rel in rels:
        match = CKPT_DIR.match(rel)
        if match:
            return f"{WANDB}/{match.group(1)}/runs/{match.group(2)}"
    return f"{WANDB}/CLAUDE-TESTS/runs/{name}"


def stage_configs(src: Path, dst: Path, listed: list[str], repo_configs: list[Path]) -> None:
    """config_saved.yaml, base_config.yaml, wandb_config.yaml and other repo configs of a run."""
    saved = src / "config.yaml"
    if saved.is_file() and not (repo_configs and filecmp.cmp(saved, repo_configs[0], shallow=False)):
        link_or_copy(saved, dst / "config_saved.yaml")
    if (src / "base_config.yaml").is_file():
        link_or_copy(src / "base_config.yaml", dst / "base_config.yaml")
    wandb_configs = sorted(p for p in map(Path, listed) if "wandb" in p.parts and p.name == "config.yaml")
    if wandb_configs and wandb_configs[-1].is_file():
        link_or_copy(wandb_configs[-1], dst / "wandb_config.yaml")
    for path in map(Path, listed):
        inside_run = src in path.parents or path.parent == src
        if path.is_file() and not inside_run and path not in repo_configs:
            link_or_copy(path, dst / f"repo_{path.name}")


def check_checkpoint(path: Path) -> dict:
    import torch

    torch.set_num_threads(1)
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as error:  # noqa: BLE001
        return {"loads": False, "error": str(error)[:200]}
    return {"loads": True, "epoch": ckpt.get("epoch"), "global_step": ckpt.get("global_step")}


def fmt_best(best: dict) -> str:
    value, epoch = best.get("value"), best.get("epoch")
    if value is None:
        return "n/a"
    shown = f"{100 * value:.1f}%" if "within" in best.get("metric", "") else f"{value:.4g}"
    return f"{best.get('metric', '')} {shown}" + (f" (epoch {epoch})" if epoch is not None else "")


def stage_extras(base: Path, extras: list[dict], bad: list[str]) -> list[dict]:
    """Copy files outside run folders that runs need or that updates report."""
    staged = []
    for extra in extras:
        for src in sorted((RUNS / extra["src"]).glob(extra["glob"])):
            dst = base / extra["dest"] / src.name
            link_or_copy(src, dst)
            info = check_checkpoint(src) if src.suffix in (".ckpt", ".pt") else {"loads": True}
            if not info["loads"]:
                bad.append(f"{extra['dest']}/{src.name}: {info['error']}")
            staged.append({"file": f"{extra['dest']}/{src.name}", "folder": extra["dest"].split("/")[0],
                           "bytes": src.stat().st_size, "note": extra["note"]})
    return staged


def command_stage(map_path: Path, out: Path) -> None:
    plan = json.loads(map_path.read_text())
    runs = plan["runs"]
    base = out / "claude_tests"
    base.mkdir(parents=True, exist_ok=True)
    titles = update_titles()
    rows, bad = [], []
    for entry in runs:
        name, folder = entry["name"], entry["folder"]
        if folder == "exclude":
            continue
        src, dst = RUNS / name, base / folder / name
        dst.mkdir(parents=True, exist_ok=True)
        repo_configs = sorted(CONFIGS.glob(f"*/{name}.yaml"))
        if repo_configs:
            link_or_copy(repo_configs[0], dst / "config.yaml")
        stage_configs(src, dst, entry.get("configs", []), repo_configs)
        files, taken = [], set()
        ckpts = [c for c in sorted(src.rglob("*.ckpt")) if "wandb" not in c.relative_to(src).parts]
        for ckpt in ckpts:
            rel = staged_name(str(ckpt.relative_to(src)), taken)
            taken.add(rel)
            link_or_copy(ckpt, dst / rel)
            info = check_checkpoint(ckpt)
            if not info["loads"]:
                bad.append(f"{name}/{rel}: {info['error']}")
            files.append({"file": rel, "bytes": ckpt.stat().st_size, **info})
        record = {
            "name": name,
            "folder": folder,
            "description": entry.get("description", ""),
            "status": entry.get("status", ""),
            "best": entry.get("best", {}),
            "updates": [{"title": titles.get(u, u), "url": SITE_URL + u + "/"} for u in entry.get("updates", [])],
            "wandb": wandb_link(name, [str(c.relative_to(src)) for c in ckpts]),
            "repo_config": str(repo_configs[0].relative_to(REPO)) if repo_configs else None,
            "checkpoints": files,
        }
        (dst / "run.json").write_text(json.dumps(record, indent=1) + "\n")
        rows.append(record)
    extras = stage_extras(base, plan.get("extras", []), bad)
    shutil.copy2(map_path, base / "runs.json")
    write_results(base, rows)
    write_readme(base, rows, [r for r in runs if r["folder"] == "exclude"], titles, extras)
    print(f"staged {len(rows)} runs, {sum(len(r['checkpoints']) for r in rows)} checkpoints in {base}")
    for line in bad:
        print("DOES NOT LOAD:", line)


def write_results(base: Path, rows: list[dict]) -> None:
    with open(base / "results.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["folder", "run", "status", "metric", "best", "best_epoch", "checkpoints", "updates"])
        for r in sorted(rows, key=lambda r: (r["folder"], r["name"])):
            best = r["best"] or {}
            writer.writerow([r["folder"], r["name"], r["status"], best.get("metric", ""), best.get("value", ""),
                             best.get("epoch", ""), " ".join(c["file"] for c in r["checkpoints"]),
                             " ".join(u["url"] for u in r["updates"])])


def write_readme(base: Path, rows: list[dict], excluded: list[dict], titles: dict[str, str],
                 extras: list[dict]) -> None:
    lines = [
        "# Claude test runs: checkpoints and configs by website date",
        "",
        "Checkpoints and configs of the test runs logged to the wandb projects CLAUDE-TESTS and DENOISER-SCAN,",
        f"grouped by the date of the BNS website update that first reports each run ({SITE_URL}).",
        "Runs that no update reports are in `unreported/`.",
        "",
        "Each run folder holds:",
        "",
        "- `config.yaml`: the training config in the aframe repo",
        "  (`dev/local/train/claude_tests/<family>/<run>.yaml`, branch reg-dev-latest), when it has one;",
        "- `config_saved.yaml`: the config Lightning saved in the run folder at its latest launch.",
        "  For runs without `config.yaml` a later test may have overwritten it; `base_config.yaml`",
        "  (the launch config) and `wandb_config.yaml` (the values wandb recorded during training)",
        "  then give the training settings, and `repo_*.yaml` are related configs from the repo;",
        "- the checkpoints, and `run.json` (status, best validation value, the updates that report",
        "  the run, the wandb link, each checkpoint's epoch).",
        "",
        "To resume a run that has `config.yaml`, from aframe/projects/train:",
        "`python -m train fit --config config.yaml --ckpt_path checkpoints/last.ckpt`.",
        "",
    ]
    for folder in sorted({r["folder"] for r in rows}, key=lambda f: (f == "unreported", f)):
        group = sorted((r for r in rows if r["folder"] == folder), key=lambda r: r["name"])
        lines += [f"## {folder}", ""]
        if folder != "unreported":
            stems = sorted({u["url"].rstrip("/").rsplit("/", 1)[-1] for r in group for u in r["updates"]
                            if u["url"].rstrip("/").rsplit("/", 1)[-1].startswith(folder)})
            lines += ["Reported in: " + "; ".join(f"[{titles.get(s, s)}]({SITE_URL}{s}/)" for s in stems), ""]
        lines += ["| run | what it is | status | best | checkpoints |", "|---|---|---|---|---|"]
        for r in group:
            ck = ", ".join(f"{c['file'].rsplit('/', 1)[-1]} (epoch {c.get('epoch')})" for c in r["checkpoints"])
            lines.append(f"| [{r['name']}]({r['wandb']}) | {r['description']} | {r['status']} | {fmt_best(r['best'])} | {ck} |")
        lines.append("")
        here = [e for e in extras if e["folder"] == folder]
        if here:
            lines += ["Other files:", ""]
            lines += [f"- `{e['file']}` ({e['bytes'] / 1e6:.1f} MB): {e['note']}" for e in here]
            lines.append("")
    if excluded:
        lines += ["## Not uploaded", ""]
        lines += [f"- {r['name']}: {r.get('status') or r.get('description', '')}" for r in excluded]
        lines.append("")
    (base / "README.md").write_text("\n".join(lines))


def command_upload(out: Path, repo: str) -> None:
    from huggingface_hub import HfApi

    api = HfApi()
    who = api.whoami()["name"]
    api.create_repo(repo, repo_type="model", private=True, exist_ok=True)
    print(f"uploading {out / 'claude_tests'} to {repo}/claude_tests as {who}")
    url = api.upload_folder(
        repo_id=repo,
        repo_type="model",
        folder_path=str(out / "claude_tests"),
        path_in_repo="claude_tests",
        commit_message="Add the Claude test runs' checkpoints and configs, by website date",
    )
    print(url)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    stage = sub.add_parser("stage")
    stage.add_argument("--map", type=Path, required=True)
    stage.add_argument("--out", type=Path, required=True)
    upload = sub.add_parser("upload")
    upload.add_argument("--out", type=Path, required=True)
    upload.add_argument("--repo", default=HF_REPO)
    args = parser.parse_args()
    if args.command == "stage":
        command_stage(args.map, args.out)
    else:
        command_upload(args.out, args.repo)


if __name__ == "__main__":
    sys.exit(main())
