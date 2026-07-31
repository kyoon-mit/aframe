"""Plot parameter distributions and metadata from a TEST (SV) injection set.

Same output as plot_priors_val.py, but handles what the SV test set adds on
top of a validation file:

  * ``shift``           the per-interferometer time shift, stored (N, n_ifos)
                        like ifo_snrs, so it is split into one shift_<ifo>
                        histogram per interferometer
  * ``injection_time``  GPS time of each injection

Any other per-ifo 2D dataset is split the same way, so this also covers test
sets that grow new column-per-ifo fields later.

Example:
    python plot_priors_test.py \\
        --input /path/to/test/end_o3_ratesandpops_bns_snr4.hdf5 \\
        --dest  /path/to/output_plots \\
        --config /path/to/plot_configs.json
"""

import json
import os

import h5py

from plot_priors_train import parse_arguments, plot_parameters
from plot_priors_val import write_metadata


def load_parameters(parameter_group, ifos):
    """Read all parameters into a dict of 1D arrays.

    Any (N, n_ifos) dataset is split into one array per interferometer:
    ``ifo_snrs`` becomes snr_<ifo> (matching the validation naming), and
    everything else keeps its own name, e.g. ``shift`` -> shift_<ifo>.
    """
    parameters = {}
    for name in parameter_group:
        values = parameter_group[name][:]
        if values.ndim == 2 and values.shape[1] == len(ifos):
            stem = "snr" if name == "ifo_snrs" else name
            for column, ifo in enumerate(ifos):
                parameters[f"{stem}_{ifo.lower()}"] = values[:, column]
        else:
            parameters[name] = values
    return parameters


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
