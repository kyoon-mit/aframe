"""Compare the sources of a data config without training anything.

For each source, from up to ``--count`` windows of its ``--part`` part (test,
the default: each window once; fit: random draws, which can overlap or
repeat):

- how many windows per hour of data hold a value beyond each of ``--values``
  (in units of whitened noise);
- how often per hour of searched merger times the template bank's loudest
  network SNR passes each of ``--snrs``. The bank searches only the merger
  times within ``max_lag`` (0.3 s) of its templates' own merger time (3.75 s
  into the window), 0.6 s per window, so this rate is per hour of those times.

These are the rates that set false alarms in a search. Each window counts at
most once, so they are rates of loud events as long as such events are rarer
than one per window.

    python -m train.experimental.noise_model.compare --data <data.yaml> --bank-file <bank.npz> --count 20000 --out <compare.json>
"""

import argparse
import json
import math
from pathlib import Path

import torch
import yaml

from train.experimental.noise_model.data import NoiseData
from train.experimental.template_bank.arch import TemplateBankNet


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", required=True, help="config with a data section like the judges'")
    parser.add_argument("--bank-file", required=True, help="npz from template_bank.bank")
    parser.add_argument("--out", required=True)
    parser.add_argument("--count", type=int, default=20000)
    parser.add_argument("--part", choices=["fit", "test"], default="test")
    parser.add_argument("--values", type=float, nargs="+", default=[5, 10, 20, 50])
    parser.add_argument("--snrs", type=float, nargs="+", default=[5, 6, 7, 8, 9, 10, 12])
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    with open(args.data) as f:
        data = NoiseData(**yaml.safe_load(f)["data"]["init_args"])
    settings = data.hparams
    # decimate=1: correlate at the full sample rate
    bank = TemplateBankNet(
        args.bank_file, len(settings.ifos), settings.sample_rate, settings.kernel_length, decimate=1,
        fduration=settings.fduration, highpass=settings.highpass,
    ).to(device)
    searched_seconds_per_window = (2 * bank.max_lag + 1) * bank.decimate / settings.sample_rate
    batches = math.ceil(args.count / args.batch_size)

    report = {}
    for which, name in enumerate(["first", "second"][: len(data.sources)]):
        largest_value, loudest_snr = [], []
        for windows in data.windows(which, args.part, args.batch_size, batches, device):
            largest_value.append(windows.abs().amax(dim=(1, 2)).cpu())
            loudest_snr.append(bank.peak_snr2(windows).amax(-1).sqrt().cpu())
        largest_value = torch.cat(largest_value)[: args.count]
        loudest_snr = torch.cat(loudest_snr)[: args.count]
        data_hours = len(largest_value) * settings.kernel_length / 3600
        searched_hours = len(loudest_snr) * searched_seconds_per_window / 3600
        report[name] = {
            "windows": len(largest_value),
            "hours_of_data": data_hours,
            "hours_of_merger_times_searched": searched_hours,
            "windows_per_hour_with_a_value_beyond": {
                v: (largest_value > v).sum().item() / data_hours for v in args.values
            },
            "loudest_network_snr_beyond_per_searched_hour": {
                s: (loudest_snr > s).sum().item() / searched_hours for s in args.snrs
            },
        }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=1)
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
