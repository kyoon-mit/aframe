"""Write injected windows of a data config's test (O3b) or validation (O3a)
loader to HDF5, with what ``forward.WindowTemplate`` needs to rebuild them.

Runs the config's own data module and injection code on CPU. Wrapped from
here, unchanged: ``sample_kernels`` (records each window's kernel offset) and
the module's ``apply_transforms`` (records the PSDs). The waveform file row
is found by matching the window's parameters.

    python -m train.experimental.slic.dump_windows \\
        --config <data.yaml> --out <file.h5> --count 16 [--split val]

Output (N windows):
    X, X_clean          (N, 2, 8192) float32, as the loader returns them
    psds                (N, 2, 2049) float64
    bank_row            (N,) row of the waveform file
    kernel_offset       (N,) s, in 0..1023
    params/<name>       (N,) float64: forward.PARAMETERS and forward.FIXED
                        (tc = (8192 - s) / 2048; dec, psi, phi, snr as drawn)
    bank/<column>       (N,) float64 parameters of the waveform file row
"""

import argparse
import time
from types import SimpleNamespace

import h5py
import lightning.pytorch as pl
import numpy as np
import torch
import yaml
from jsonargparse import ArgumentParser

import train.data.supervised.supervised as supervised
from train.experimental.slic.forward import (
    FIXED,
    PARAMETERS,
    SAMPLE_RATE,
    WINDOW,
)

MATCH_COLUMNS = (
    "a_1",
    "a_2",
    "mass_1",
    "mass_2",
    "phase",
    "phi_12",
    "phi_jl",
    "ra",
    "redshift",
    "theta_jn",
    "tilt_1",
    "tilt_2",
)
SPINS = ("a_1", "a_2", "tilt_1", "tilt_2", "phi_12", "phi_jl")


def build_data_module(config: str, split: str):
    """The config's data module, set up for one split without a trainer."""
    with open(config) as f:
        section = yaml.safe_load(f)["data"]
    parser = ArgumentParser()
    parser.add_subclass_arguments(pl.LightningDataModule, "data")
    for key in ("sample_rate", "fduration", "ifos"):  # as train.cli.AframeCLI
        parser.link_arguments(
            f"data.init_args.{key}",
            f"data.init_args.waveform_sampler.init_args.{key}",
            apply_on="parse",
        )
    cfg = parser.parse_object({"data": section})
    dm = parser.instantiate_classes(cfg).data
    dm.trainer = SimpleNamespace(
        training=False,
        testing=split == "test",
        validating=split == "val",
        sanity_checking=False,
        device_ids=[0],
        accelerator=None,
    )
    dm._logger = dm.get_logger(1, 0)
    dm.train_fnames, dm.valid_fnames = dm.train_val_split()
    dm.build_transforms()
    return dm


class Recorder:
    """Kernel offsets and PSDs of the last injected batch."""

    def __init__(self, dm):
        self.offsets = self.psds = None
        sample_kernels = supervised.sample_kernels
        apply_transforms = dm.apply_transforms

        def record_offsets(x, kernel_size, coincident=True):
            kernels, idx = sample_kernels(
                x,
                kernel_size=kernel_size,
                coincident=coincident,
                return_idx=True,
            )
            self.offsets = idx.clone()
            return kernels

        def record_psds(X, psds):
            self.psds = psds.clone()
            return apply_transforms(X, psds)

        supervised.sample_kernels = record_offsets
        dm.apply_transforms = record_psds


def bank_rows(params, bank):
    """Waveform file row of each window, by exact float32 match."""
    table = np.stack([bank[k].astype(np.float32) for k in MATCH_COLUMNS], 1)
    rows = []
    for i in range(len(params["mass_1"])):
        query = np.array(
            [params[k][i] for k in MATCH_COLUMNS], dtype=np.float32
        )
        hits = np.flatnonzero((table == query).all(1))
        if len(hits) != 1:
            raise ValueError(f"window {i}: {len(hits)} matching bank rows")
        rows.append(int(hits[0]))
    return np.array(rows)


def dump(config, out, count, split, seed):
    """Writes ``count`` windows to ``out``; returns seconds spent in the
    loader and injection."""
    pl.seed_everything(seed)
    dm = build_data_module(config, split)
    recorder = Recorder(dm)
    loader = dm.test_dataloader() if split == "test" else dm.val_dataloader()
    bank_file = dm.waveform_sampler.training_waveform_files[0]
    with h5py.File(bank_file, "r") as f:
        bank = {k: f["parameters"][k][:] for k in f["parameters"]}

    keep = {k: [] for k in ("X", "X_clean", "psds", "offsets", "params")}
    start = time.time()
    total = 0
    for batch in loader:
        batch = dm.on_before_batch_transfer(batch, 0)
        X, X_clean, y, params = dm.on_after_batch_transfer(batch, 0)
        if not (y[:, 0] > 0).all() or len(recorder.offsets) != len(y):
            raise ValueError("expected a signal in every window")
        keep["X"].append(X.numpy())
        keep["X_clean"].append(X_clean.numpy())
        keep["psds"].append(recorder.psds.double().numpy())
        keep["offsets"].append(recorder.offsets.numpy())
        keep["params"].append({k: v.numpy() for k, v in params.items()})
        total += len(y)
        if total >= count:
            break
    elapsed = time.time() - start

    drawn = {
        k: np.concatenate([p[k] for p in keep["params"]])[:count]
        for k in keep["params"][0]
    }
    offsets = np.concatenate(keep["offsets"])[:count]
    rows = bank_rows(drawn, bank)
    mass_1, mass_2 = bank["mass_1"][rows], bank["mass_2"][rows]
    values = {
        "chirp_mass": (mass_1 * mass_2) ** 0.6 / (mass_1 + mass_2) ** 0.2,
        "mass_ratio": mass_2 / mass_1,
        "tc": (WINDOW - offsets) / SAMPLE_RATE,
    }
    for k in SPINS + ("phase", "theta_jn"):
        values[k] = bank[k][rows]
    for k in ("snr", "dec", "psi", "phi"):
        values[k] = drawn[k].astype(np.float64)

    with h5py.File(out, "w") as f:
        for k in ("X", "X_clean", "psds"):
            f[k] = np.concatenate(keep[k])[:count]
        f["bank_row"] = rows
        f["kernel_offset"] = offsets
        for k in PARAMETERS + FIXED:
            f[f"params/{k}"] = values[k]
        for k, v in bank.items():
            f[f"bank/{k}"] = v[rows]
        f.attrs.update(
            config=str(config),
            split=split,
            seed=seed,
            bank_file=str(bank_file),
            seconds=elapsed,
        )
    return elapsed


def read_window(path, index):
    """One window of a dump: arrays, ``theta`` (11,) in PARAMETERS order
    and the FIXED values as floats."""
    with h5py.File(path, "r") as f:
        keys = ("X", "X_clean", "psds", "bank_row", "kernel_offset")
        window = {k: f[k][index] for k in keys}
        window["theta"] = np.array(
            [f[f"params/{k}"][index] for k in PARAMETERS]
        )
        window.update({k: float(f[f"params/{k}"][index]) for k in FIXED})
    return window


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", required=True, help="yaml, data section")
    parser.add_argument("--out", required=True)
    parser.add_argument("--count", type=int, default=16)
    parser.add_argument("--split", choices=["test", "val"], default="test")
    parser.add_argument(
        "--seed", type=int, default=None, help="default: the config's"
    )
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    seed = args.seed
    if seed is None:
        with open(args.config) as f:
            seed = yaml.safe_load(f)["seed_everything"]
    elapsed = dump(args.config, args.out, args.count, args.split, seed)
    print(
        f"{args.count} windows: {elapsed:.1f} s in the loader "
        f"({elapsed / args.count:.3f} s per window) -> {args.out}"
    )


if __name__ == "__main__":
    main()
