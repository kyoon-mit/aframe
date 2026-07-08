"""Plot the raw parameter distributions from an injection/prior hdf5 file.

Makes one histogram per parameter (raw counts, not normalized) using the bin
range, bin size, color, and alpha from ../configs/plot_configs.json, and writes
a metadata.info text file describing the input.

Example:
    python plot_priors_train.py \\
        --input /path/to/train_waveforms.hdf5 \\
        --dest  /path/to/output_plots
"""

import argparse
import json
import os
from pathlib import Path

import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEFAULT_CONFIG = (
    Path(__file__).resolve().parent.parent / "configs" / "plot_configs.json"
)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input", required=True, help="input hdf5 file")
    parser.add_argument("--dest", required=True, help="output plot folder")
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help="json with per-parameter bins/color/alpha",
    )
    return parser.parse_args()


def chirp_mass(mass_1, mass_2):
    """Compute the chirp mass from the component masses."""
    return (mass_1 * mass_2) ** (3 / 5) / (mass_1 + mass_2) ** (1 / 5)


def mass_ratio(mass_1, mass_2):
    """Compute the mass ratio from the component masses."""
    # if mass_1.shape != mass_2.shape:
    #     raise ValueError("mass_1 and mass_2 must have the same shape")
    # if np.any(mass_1 <= 0) or np.any(mass_2 <= 0):
    #     raise ValueError("component masses must be positive")
    # Require caller to provide mass_1 as the primary (>= mass_2).
    if np.any(mass_1 < mass_2):
        raise ValueError("mass_1 must be >= mass_2 for all entries")
    return mass_2 / mass_1


def _helper_param_iter(group, derived_dict):
    """Yield (name, values) for live group datasets then derived arrays."""
    for name in group:
        yield name, group[name][:]
    for name, arr in derived_dict.items():
        yield name, arr


def plot_parameters(parameter_group, parameter_config, dest_dir):
    """One raw-count histogram per parameter, saved as <name>.png."""
    derived = {}
    if "chirp_mass" not in parameter_group:
        mass1 = parameter_group["mass_1"][:]
        mass2 = parameter_group["mass_2"][:]
        derived["chirp_mass"] = chirp_mass(mass1, mass2)
    if "mass_ratio" not in parameter_group:
        mass1 = parameter_group["mass_1"][:]
        mass2 = parameter_group["mass_2"][:]
        derived["mass_ratio"] = mass_ratio(mass1, mass2)

    # iterate parameters (live h5 datasets first, then derived ones)
    for parameter_name, values in _helper_param_iter(parameter_group, derived):
        settings = parameter_config.get(parameter_name)
        if settings is None:
            # parameter not in the config: use the data range and 50 bins
            minimum, maximum = float(values.min()), float(values.max())
            settings = {
                "min": minimum,
                "max": maximum,
                "bin_size": (maximum - minimum) / 50 or 1e-2,
                "color": "blue",
                "alpha": 1.0,
                "label": parameter_name,
            }
        bin_edges = np.arange(
            settings["min"],
            settings["max"] + settings["bin_size"],
            settings["bin_size"],
        )
        figure, axis = plt.subplots(figsize=(6, 4))
        axis.hist(
            values,
            bins=bin_edges,
            color=settings["color"],
            alpha=settings["alpha"],
            label=settings["label"],
        )
        axis.set_xlabel(settings["xlabel"])
        axis.set_ylabel("count")
        axis.legend(loc="upper right", fontsize=8)
        figure.tight_layout()
        output_path = os.path.join(dest_dir, f"{parameter_name}.png")
        figure.savefig(output_path, dpi=140)
        plt.close(figure)
        print("wrote", output_path, flush=True)


def write_metadata(hdf5_file, input_path, dest_dir):
    """Write duration/length/geometry/keys of the input to metadata.info."""
    attributes = dict(hdf5_file.attrs)
    duration = float(attributes["duration"])
    sample_rate = float(attributes["sample_rate"])
    length = int(attributes["length"])
    samples_per_injection = int(round(duration * sample_rate))
    number_of_samples = length * samples_per_injection

    top_level_keys = list(hdf5_file.keys())
    parameter_keys = list(hdf5_file["parameters"].keys())
    waveform_keys = (
        list(hdf5_file["waveforms"].keys()) if "waveforms" in hdf5_file else []
    )

    lines = [
        f"file_name: {os.path.basename(input_path)}",
        f"duration: {duration}",
        f"length: {length}",
        f"num_injections: {int(attributes['num_injections'])}",
        f"right_pad: {float(attributes['right_pad'])}",
        f"sample_rate: {sample_rate}",
        f"samples_per_injection: {samples_per_injection}",
        f"number_of_samples: {number_of_samples}",
        f"h5_top_level_keys: {top_level_keys}",
        f"h5_parameter_keys: {parameter_keys}",
        f"h5_waveform_keys: {waveform_keys}",
    ]
    metadata_path = os.path.join(dest_dir, "metadata.info")
    with open(metadata_path, "w") as metadata_file:
        metadata_file.write("\n".join(lines) + "\n")
    print("wrote", metadata_path, flush=True)


def main():
    arguments = parse_arguments()
    os.makedirs(arguments.dest, exist_ok=True)
    with open(arguments.config) as config_file:
        parameter_config = json.load(config_file)["parameters"]
    with h5py.File(arguments.input, "r") as hdf5_file:
        plot_parameters(
            hdf5_file["parameters"], parameter_config, arguments.dest
        )
        write_metadata(hdf5_file, arguments.input, arguments.dest)


if __name__ == "__main__":
    main()
