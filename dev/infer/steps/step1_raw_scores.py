"""Step 1: run inference and save the RAW (un-integrated) model scores.

Same arguments as aframe's ``infer`` (it reuses aframe's parser), but also
writes ``timeseries.hdf5`` holding the model output before any integration or
clustering, plus the metadata step 2 needs to rebuild events from it.

This is the only step that needs Triton/GPU. Once the raw scores are cached,
step 2 can re-apply any integration method for free.

Example (one branch):
    uv run python step1_raw_scores.py \\
        --config /path/to/infer_merger_s4_id2.yaml \\
        --client.address ${TRITON}:8001 \\
        --data.triton_address ${TRITON}:8001 \\
        --data.background_fname /path/to/background-....hdf5 \\
        --data.shifts [0,1] \\
        --outdir /path/to/results/branch_0
"""

import os

import h5py
import numpy as np

from infer.cli import build_parser
from infer.main import infer


def save_raw_scores(
    output_dir,
    background_scores,
    foreground_scores,
    sequence,
    postprocessor_config,
):
    """Write the raw score timeseries + everything step 2 needs."""
    output_path = os.path.join(output_dir, "timeseries.hdf5")
    with h5py.File(output_path, "w") as output_file:
        output_file.create_dataset(
            "background_ts", data=np.asarray(background_scores)
        )
        if foreground_scores is not None:
            output_file.create_dataset(
                "foreground_ts", data=np.asarray(foreground_scores)
            )
        output_file.attrs["t0"] = float(sequence.t0)
        output_file.attrs["duration"] = float(sequence.duration)
        output_file.attrs["sample_rate"] = float(sequence.sample_rate)
        output_file.attrs["shifts"] = np.asarray(
            sequence.shifts, dtype=float
        ) / float(sequence.sample_rate)
        output_file.attrs["ifos"] = list(sequence.ifos)
        output_file.attrs["background_fname"] = str(sequence.background_fname)
        for key, value in postprocessor_config.items():
            output_file.attrs[key] = value
    print(f"wrote {output_path}", flush=True)


def main():
    parser = build_parser()
    cfg = parser.parse_args()

    # capture the postprocessor/data settings before the classes are built,
    # so step 2 can rebuild a postprocessor without re-reading the yaml
    postprocessor_config = {
        "psd_length": float(cfg.postprocessor.psd_length),
        "fduration": float(cfg.postprocessor.fduration),
        "inference_sampling_rate": float(cfg.data.inference_sampling_rate),
        "integration_window_length": float(
            cfg.postprocessor.integration_window_length
        ),
        "cluster_window_length": float(
            cfg.postprocessor.cluster_window_length
        ),
        "injection_set_fname": str(cfg.data.injection_set_fname),
    }
    output_dir = cfg.outdir
    os.makedirs(output_dir, exist_ok=True)

    cfg = parser.instantiate_classes(cfg)
    with cfg.client:
        background, foreground, background_scores, foreground_scores = infer(
            cfg.client, cfg.data, cfg.postprocessor, return_timeseries=True
        )

    # keep the usual event files so nothing downstream breaks
    background.write(os.path.join(output_dir, "background.hdf5"))
    foreground.write(os.path.join(output_dir, "foreground.hdf5"))
    save_raw_scores(
        output_dir,
        background_scores,
        foreground_scores,
        cfg.data,
        postprocessor_config,
    )


if __name__ == "__main__":
    main()
