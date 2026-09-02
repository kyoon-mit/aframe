"""Step 1: run inference and save the RAW model-output timeseries.

Same arguments as aframe's ``infer`` (it reuses aframe's parser), but instead
of aframe's inline integrate/cluster (which assumes a 1-D discriminator) it
captures the raw discriminator timeseries straight from the streaming client
and writes ``timeseries.hdf5`` -- the model output before any postprocessing,
plus the metadata step 2 needs to rebuild events from it.

This is the only step that needs Triton/GPU. Once the raw scores are cached,
step 2 applies any integration method AND any metric for free -- so the served
discriminator emits the full model output ([mass, sigma], two channels), never
a single collapsed statistic. background_ts/foreground_ts are (N,) for a
1-channel serve or (N, 2) for the [mass, sigma] serve.

Example (one branch):
    uv run python step1_raw_scores.py \\
        --config /path/to/infer_merger_s4_test7.yaml \\
        --client.address ${TRITON}:8001 \\
        --data.triton_address ${TRITON}:8001 \\
        --data.background_fname /path/to/background-....hdf5 \\
        --data.shifts [0,1] \\
        --outdir /path/to/results/branch_0
"""

import os
import time

import h5py
import numpy as np
from tqdm import tqdm

from infer.cli import build_parser


def stream_scores(client, sequence):
    """Stream one sequence and return the RAW (background_ts, foreground_ts).

    Replicates infer()'s request loop but stops before the integrate/cluster
    step, so it works for a multi-channel discriminator. Returns whatever the
    server emits per window: (N,) for 1-channel, (N, C) for C channels.
    """
    for i, (background_window, injected_window) in enumerate(tqdm(sequence)):
        sequence_start = i == 0
        sequence_end = i == len(sequence) - 1
        client.infer(
            np.stack([background_window, background_window]),
            request_id=i,
            sequence_id=sequence.id,
            sequence_start=sequence_start,
            sequence_end=sequence_end,
        )
        if injected_window is not None:
            client.infer(
                np.stack([background_window, injected_window]),
                request_id=i,
                sequence_id=sequence.id + 1,
                sequence_start=sequence_start,
                sequence_end=sequence_end,
            )
        # warm up: block on the first response so inference errors surface now
        if not i:
            while not sequence.started:
                client.get()
                time.sleep(1e-2)

    result = client.get()
    while result is None:
        result = client.get()
        time.sleep(1e-1)
    background_ts, foreground_ts = result
    return background_ts, foreground_ts


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
    parser.add_argument(
        "--true_injection_set_fname",
        default=None,
        help=(
            "Provenance override: the ORIGINAL injection set this run's "
            "data.injection_set_fname was cached/extracted from (see "
            "step0_cache_injections.py). Recorded in timeseries.hdf5 "
            "instead of the (possibly cached) --data.injection_set_fname, "
            "so provenance still points at the real source."
        ),
    )
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
        "injection_set_fname": str(
            cfg.true_injection_set_fname or cfg.data.injection_set_fname
        ),
    }
    output_dir = cfg.outdir
    os.makedirs(output_dir, exist_ok=True)

    cfg = parser.instantiate_classes(cfg)
    with cfg.client:
        background_scores, foreground_scores = stream_scores(
            cfg.client, cfg.data
        )

    save_raw_scores(
        output_dir,
        background_scores,
        foreground_scores,
        cfg.data,
        postprocessor_config,
    )


if __name__ == "__main__":
    main()
