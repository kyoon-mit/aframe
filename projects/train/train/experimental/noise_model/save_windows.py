"""Save whitened windows to an HDF5 file, for judges to read as a ``saved`` source.

From the ``first`` or ``second`` source of a data config (with ``--part test``,
the default, each window once, until the part runs out):

    python -m train.experimental.noise_model.save_windows --data <data.yaml> --source first --count 24000 --out <file.h5>

From a trained generator (the data config only sets the window shape):

    python -m train.experimental.noise_model.save_windows --data <data.yaml> \
        --generator <run folder>/CLAUDE-TESTS/<run name>/checkpoints/last.ckpt --count 24000 --out <file.h5>

With ``--loud 10``, only windows with a value beyond 10 (in units of whitened
noise) are kept, for judges that study the loud windows alone.
"""

import argparse
from pathlib import Path

import h5py
import torch
import yaml

import train.cli  # noqa: F401  (its torch.load override lets the checkpoint rebuild its network)
from train.experimental.noise_model.data import NoiseData
from train.experimental.noise_model.tasks import GeneratorTask


def generated(task: GeneratorTask, batch_size: int, num_ifos: int, size: int, steps: int):
    while True:
        yield task.diffusion.generate(batch_size, num_ifos, size, steps)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", required=True, help="config with a data section like the judges'")
    parser.add_argument("--out", required=True)
    parser.add_argument("--count", type=int, required=True)
    made_from = parser.add_mutually_exclusive_group(required=True)
    made_from.add_argument("--source", choices=["first", "second"])
    made_from.add_argument("--generator", help="GeneratorTask checkpoint")
    parser.add_argument("--part", choices=["fit", "test"], default="test", help="with --source")
    parser.add_argument("--steps", type=int, default=18, help="generation steps, with --generator")
    parser.add_argument("--loud", type=float, help="keep only windows with a value beyond this")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    with open(args.data) as f:
        data = NoiseData(**yaml.safe_load(f)["data"]["init_args"])
    num_ifos, size = len(data.hparams.ifos), data.whitening.window_size
    if args.generator:
        task = GeneratorTask.load_from_checkpoint(args.generator, map_location=device).eval()
        batches = generated(task, args.batch_size, num_ifos, size, args.steps)
    else:
        which = {"first": 0, "second": 1}[args.source]
        if which >= len(data.sources):
            parser.error(f"{args.data} has no {args.source} source")
        batches = data.windows(which, args.part, args.batch_size, 10**9, device)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.out, "w") as f:
        # grows as windows arrive, one window per chunk for reading single rows
        saved = f.create_dataset(
            "windows", shape=(0, num_ifos, size), maxshape=(None, num_ifos, size),
            chunks=(1, num_ifos, size), dtype="float32",
        )
        written = drawn = 0
        for batch_number, windows in enumerate(batches):
            drawn += len(windows)
            if args.loud is not None:
                windows = windows[windows.abs().amax(dim=(1, 2)) > args.loud]
            take = min(len(windows), args.count - written)
            if take:
                saved.resize(written + take, axis=0)
                saved[written:] = windows[:take].cpu().numpy()
                written += take
            if batch_number % 50 == 0 or written == args.count:
                print(f"{written} of {args.count} windows written, {drawn} drawn", flush=True)
            if written == args.count:
                break
        if written < args.count:
            print(f"the source ran out: {written} windows written, {drawn} drawn", flush=True)
        f.attrs["made_from"] = args.generator or f"{args.source} source of {args.data}, {args.part} part"
        f.attrs["ifos"] = ",".join(data.hparams.ifos)
        f.attrs["sample_rate"] = data.hparams.sample_rate
        f.attrs["windows_drawn"] = drawn
        if args.loud is not None:
            f.attrs["loud"] = args.loud


if __name__ == "__main__":
    main()
