"""Lower the sample rate of strain files by a whole-number factor.

Reads every ``background-*.hdf5`` in --source, downsamples each detector by
--factor with an anti-aliasing filter, and writes the new files to --dest.

Example (turn 4096 Hz files into 2048 Hz -> factor 2):
    python downsample_background.py \\
        --source /path/to/O3b_H1_L1_4096Hz \\
        --dest   /path/to/O3b_H1_L1_2048Hz \\
        --factor 2
"""

import argparse
import glob
import os

import h5py
from scipy.signal import resample


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--source", required=True, help="folder of input files"
    )
    parser.add_argument("--dest", required=True, help="folder for new files")
    parser.add_argument(
        "--factor", type=int, default=2, help="how much slower to make it"
    )
    parser.add_argument(
        "--detectors",
        nargs="+",
        default=["H1", "L1"],
        help="detectors to copy",
    )
    return parser.parse_args()


def main():
    arguments = parse_arguments()
    os.makedirs(arguments.dest, exist_ok=True)
    input_files = sorted(glob.glob(f"{arguments.source}/background-*.hdf5"))
    for input_path in input_files:
        file_name = os.path.basename(input_path)
        with (
            h5py.File(input_path, "r") as source_file,
            h5py.File(f"{arguments.dest}/{file_name}", "w") as dest_file,
        ):
            for detector in arguments.detectors:
                input_dataset = source_file[detector]
                resampled = resample(
                    input_dataset[:], len(input_dataset[:]) // arguments.factor
                )
                output_dataset = dest_file.create_dataset(
                    detector, data=resampled.astype(input_dataset.dtype)
                )
                for attr_name, attr_value in input_dataset.attrs.items():
                    output_dataset.attrs[attr_name] = attr_value
                # time-step grows by the same factor the rate shrinks
                output_dataset.attrs["dx"] = (
                    input_dataset.attrs["dx"] * arguments.factor
                )
        new_sample_rate = 1 / (input_dataset.attrs["dx"] * arguments.factor)
        print(
            f"{file_name}: {len(resampled)} samples "
            f"@ {new_sample_rate:.0f} Hz",
            flush=True,
        )
    print(f"wrote {len(input_files)} files -> {arguments.dest}", flush=True)


if __name__ == "__main__":
    main()
