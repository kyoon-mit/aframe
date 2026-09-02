"""Plot parameter distributions and metadata from a validation hdf5 file.

Same histograms as plot_priors_train.py (including the source/detector
frame overlay for redshift-scaled masses), plus what validation files add
on top of a training file:

  * snr           network SNR of each accepted signal
  * ifo_snrs      per-interferometer SNR, split into one histogram per ifo
  * metadata      the ifos attribute and the rejection-sampling acceptance
                  rate (length / num_injections)

Example:
    python plot_priors_val.py \\
        --input /path/to/val_waveforms.hdf5 \\
        --dest  /path/to/output_plots
"""

import json
import os

import h5py

from plot_priors_train import parse_arguments, plot_parameters


def load_parameters(parameter_group, ifos):
    """Read all parameters into a dict of 1D arrays.

    The 2D ifo_snrs dataset is split into one snr_<ifo> array per
    interferometer so every entry can be histogrammed the same way.
    """
    parameters = {}
    for name in parameter_group:
        values = parameter_group[name][:]
        if name == "ifo_snrs":
            for column, ifo in enumerate(ifos):
                parameters[f"snr_{ifo.lower()}"] = values[:, column]
        else:
            parameters[name] = values
    return parameters


def write_metadata(hdf5_file, input_path, dest_dir):
    """Write every file attribute plus derived rejection info."""
    attributes = dict(hdf5_file.attrs)
    duration = float(attributes["duration"])
    sample_rate = float(attributes["sample_rate"])
    length = int(attributes["length"])
    num_injections = int(attributes["num_injections"])
    samples_per_injection = int(round(duration * sample_rate))

    lines = [f"file_name: {os.path.basename(input_path)}"]
    for key in sorted(attributes):
        value = attributes[key]
        if hasattr(value, "tolist"):
            value = value.tolist()
        lines.append(f"{key}: {value}")
    lines += [
        f"samples_per_injection: {samples_per_injection}",
        f"number_of_samples: {length * samples_per_injection}",
        f"rejected: {num_injections - length}",
        f"acceptance_rate: {length / num_injections:.4f}",
        f"h5_top_level_keys: {list(hdf5_file.keys())}",
        f"h5_parameter_keys: {list(hdf5_file['parameters'].keys())}",
        f"h5_waveform_keys: {list(hdf5_file['waveforms'].keys())}",
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
        ifos = [str(ifo) for ifo in hdf5_file.attrs["ifos"]]
        parameters = load_parameters(hdf5_file["parameters"], ifos)
        plot_parameters(parameters, parameter_config, arguments.dest)
        write_metadata(hdf5_file, arguments.input, arguments.dest)


if __name__ == "__main__":
    main()
