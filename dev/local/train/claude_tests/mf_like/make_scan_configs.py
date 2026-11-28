"""Writes the MNet-Shallow scan configs to scan/: two template starts,
three learning rates, five schedules and seven template counts, from the
first-pass config. Prints one config path per line, in launch order."""

import copy
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
BASE = HERE / "mnet_shallow_h1_t1_train_snr4_prob0.5.yaml"
RUNS = Path("/home/kyoon/SSM-BNS/aframe/dev/runs/claude_tests/mf_like/scan")
TEMPLATES = Path(
    "/home/kyoon/SSM-BNS/aframe/dev/runs/claude_tests/mf_like/templates/"
    "h1_signal_prior_4096.npz"
)
COUNTS = [1, 4, 16, 64, 256, 1024, 4096]
RATES = ["1e-5", "1e-4", "1e-3"]
VALIDATE_EVERY = 5
SCHEDULES = {
    "constant": {
        "lr_scheduler": {
            "class_path": "torch.optim.lr_scheduler.ConstantLR",
            "init_args": {"factor": 1.0, "total_iters": 0},
        },
    },
    "one_cycle": {"lr_scheduler": None, "pct_lr_ramp": 0.115},
    "cosine_restarts": {
        "lr_scheduler": {
            "class_path": "train.model.WarmupCosineAnnealingWarmRestarts",
            "init_args": {
                "warmup_epochs": 10,
                "T_0": 20,
                "T_mult": 2,
                "eta_min": 1e-7,
                "warmup_start_factor": 0.01,
                "peak_decay": 0.9,
            },
        },
    },
    "exponential": {
        "lr_scheduler": {
            "class_path": "torch.optim.lr_scheduler.ExponentialLR",
            "init_args": {"gamma": 0.95},
        },
    },
    "plateau": {
        "lr_scheduler": {
            "class_path": "torch.optim.lr_scheduler.ReduceLROnPlateau",
            "init_args": {"mode": "max"},
        },
        "lr_scheduler_monitor": "validation/TimeSlideAUROC",
        "lr_scheduler_frequency": VALIDATE_EVERY,
    },
}
STARTS = {"start_random": None, "start_signals": str(TEMPLATES)}


def config(base, count, start, schedule, rate):
    name = (
        f"mnet_shallow_h1_t{count}_{start}_{schedule}_lr{rate}"
        "_train_snr4_prob0.5"
    )
    root = str(RUNS / name)
    c = copy.deepcopy(base)
    t = c["trainer"]
    t["default_root_dir"] = root
    t["check_val_every_n_epoch"] = VALIDATE_EVERY
    t["logger"]["init_args"].update(name=name, id=name, save_dir=root)
    m = c["model"]["init_args"]
    m["arch"]["init_args"]["num_templates"] = count
    m["arch"]["init_args"]["bias"] = True
    m["learning_rate"] = float(rate)
    m["template_file"] = STARTS[start]
    m["lr_scheduler_interval"] = "epoch"
    m.update(copy.deepcopy(SCHEDULES[schedule]))
    return name, c


def main():
    base = yaml.safe_load(BASE.read_text())
    out = HERE / "scan"
    out.mkdir(exist_ok=True)
    for rate in RATES:
        for schedule in SCHEDULES:
            for start in STARTS:
                for count in COUNTS:
                    name, c = config(base, count, start, schedule, rate)
                    path = out / f"{name}.yaml"
                    path.write_text(yaml.safe_dump(c, sort_keys=False))
                    print(path)


if __name__ == "__main__":
    main()
