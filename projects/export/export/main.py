import io
import logging
import re
import shutil
from pathlib import Path
from typing import Optional

import h5py
import hermes.quiver as qv
import torch

from export.snapshotter import add_streaming_input_preprocessor
from utils.s3 import open_file


def scale_model(model, instances):
    """
    Scale the model to the number of instances per GPU desired
    at inference time
    """
    # TODO: should quiver handle this under the hood?
    try:
        model.config.scale_instance_group(instances)
    except ValueError:
        model.config.add_instance_group(count=instances)


def _install_python_aframe(
    repository_directory: str,
    aframe_model_dir: str,
    *,
    batch_size: int,
    num_ifos: int,
    seq_len: int,
    output_name: str,
) -> None:
    """Overwrite the repo's ``aframe`` model with a Triton python backend.

    Copies the version directory (``model.py`` etc.) from ``aframe_model_dir``
    and writes a python-backend ``config.pbtxt`` with the target geometry,
    reusing the source config's ``backend``/``instance_group``/``optimization``
    blocks but rewriting the input/output dims to ``(batch_size, num_ifos,
    seq_len)`` and ``(batch_size, 1)``. This is the no-compile path: the model
    is served as-is rather than exported to TensorRT/ONNX.
    """
    src = Path(aframe_model_dir)
    dst = Path(repository_directory) / "aframe"

    # copy the version dir (model.py, empty model.plan, ...); per-run files
    # (config.yaml/weights.*) are added later by make_sv_run.py.
    src_version = src / "1"
    dst_version = dst / "1"
    if dst_version.exists():
        shutil.rmtree(dst_version)
    shutil.copytree(
        src_version,
        dst_version,
        ignore=shutil.ignore_patterns(
            "__pycache__", "config.yaml", "weights.*"
        ),
    )

    cfg = (src / "config.pbtxt").read_text()
    cfg = re.sub(
        r"input\s*\{.*?\}",
        "input {\n"
        '  name: "X"\n'
        "  data_type: TYPE_FP32\n"
        f"  dims: {batch_size}\n"
        f"  dims: {num_ifos}\n"
        f"  dims: {seq_len}\n"
        "}",
        cfg,
        count=1,
        flags=re.DOTALL,
    )
    cfg = re.sub(
        r"output\s*\{.*?\}",
        "output {\n"
        f'  name: "{output_name}"\n'
        "  data_type: TYPE_FP32\n"
        f"  dims: {batch_size}\n"
        "  dims: 1\n"
        "}",
        cfg,
        count=1,
        flags=re.DOTALL,
    )
    (dst / "config.pbtxt").write_text(cfg)
    logging.info(
        f"Installed python-backend aframe from {src} "
        f"(input dims [{batch_size}, {num_ifos}, {seq_len}])"
    )


def export(
    repository_directory: str,
    num_ifos: int,
    kernel_length: float,
    inference_sampling_rate: float,
    sample_rate: float,
    batch_size: int,
    fduration: float,
    psd_length: float,
    preprocessor: torch.nn.Module,
    weights: Optional[str] = None,
    batch_file: Optional[str] = None,
    streams_per_gpu: int = 1,
    num_outputs: Optional[int] = 1,
    aframe_instances: Optional[int] = None,
    preproc_instances: Optional[int] = None,
    platform: qv.Platform = qv.Platform.TENSORRT,
    aframe_model_dir: Optional[str] = None,
    clean: bool = False,
    verbose: bool = False,
    **kwargs,
) -> None:
    """
    Export a aframe architecture to a model repository
    for streaming inference, including adding a model
    for caching input snapshot state on the server.

    Args:
        repository_directory:
            Directory to which to save the models and their
            configs
        num_ifos:
            The number of interferometers contained along the
            channel dimension used to train aframe
        kernel_length:
            Length of segment in seconds that the network sees
        inference_sampling_rate:
            The rate at which kernels are sampled from the
            h(t) timeseries. This, along with the `sample_rate`,
            dictates the size of the update expected at the
            snapshotter model
        sample_rate:
            Rate at which the input kernel has been sampled, in Hz
        batch_size:
            Number of kernels per batch
        fduration:
            Length of the time-domain whitening filter in seconds
        psd_length:
            Length of background time in seconds to use for PSD
            calculation
        preprocessor:
            PyTorch module that defines how data from the snapshotter
            is manipulated before being given the model
        weights:
            File Like object or Path representing
            a set of trained weights that will be
            exported to a model_repository. Supports
            local and S3 paths. Not required (and ignored)
            when ``aframe_model_dir`` is given.
        batch_file:
            Path to file containing a batch of data from model
            training. This is used to determine the input size
            of the model. File structure is assumed to match
            the structure of the file written during training.
            Not required when ``aframe_model_dir`` is given, in which
            case the input shape is derived from ``kernel_length`` and
            ``sample_rate``.
        streams_per_gpu:
            The number of snapshot states to host per GPU during
            inference
        num_outputs:
            The number of neural network outputs. Default is set to 1
        aframe_instances:
            The number of concurrent execution instances of the
            aframe architecture to host per GPU during inference
        platform:
            The backend framework platform used to host the
            aframe architecture on the inference service. Right
            now only `"onnxruntime_onnx"` is supported.
        aframe_model_dir:
            Path to a directory containing a pre-built Triton
            python-backend ``aframe`` model (``config.pbtxt`` and a
            ``1/`` version dir with ``model.py``). When given, the
            compile step (loading ``weights`` and exporting the graph to
            TensorRT/ONNX) is skipped: the snapshotter/preprocessor are
            still rebuilt for the requested geometry, and this model is
            installed as the ``aframe`` step. Used for architectures that
            aren't TRT/ONNX-friendly (e.g. S4D's FFT) and are served via
            a python backend instead.
        clean:
            Whether to clear the repository directory before starting
            export
        verbose:
            If True, log at `DEBUG` verbosity, otherwise log at
            `INFO` verbosity.
        **kwargs:
            Key word arguments specific to the export platform
    """

    # instantiate a model repository at the
    # indicated location. Split up the preprocessor
    # and the neural network (which we'll call aframe)
    # to export/scale them separately, and start by
    # seeing if either already exists in the model repo
    repo = qv.ModelRepository(repository_directory, clean)
    try:
        aframe = repo.models["aframe"]
    except KeyError:
        aframe = repo.add("aframe", platform=platform)

    # if we specified a number of instances we want per-gpu
    # for each model at inference time, scale them now
    if aframe_instances is not None:
        scale_model(aframe, aframe_instances)

    # determining the number of neural networks for
    # naming the outputs
    if num_outputs < 1:
        raise ValueError("num_outputs must be >= 1")
    output_names = (
        ["discriminator"]
        if num_outputs == 1
        else [f"discriminator_{i}" for i in range(num_outputs)]
    )

    skip_compile = aframe_model_dir is not None

    if skip_compile:
        # No graph to compile: the aframe model is served by a Triton python
        # backend (installed below). Register its I/O on the quiver config so
        # the ensemble can be wired, and derive the input shape directly from
        # the geometry rather than a training batch file.
        if num_outputs != 1:
            raise ValueError(
                "aframe_model_dir (skip-compile) only supports num_outputs=1"
            )
        seq_len = int(kernel_length * sample_rate)
        input_shape_dict = {"X": (batch_size, num_ifos, seq_len)}
        aframe.config.add_input("X", input_shape_dict["X"], dtype="float32")
        aframe.config.add_output(
            output_names[0], (batch_size, 1), dtype="float32"
        )
        aframe.config.write()
    else:
        # load in the model graph
        logging.info("Initializing model graph")
        with open_file(weights, "rb") as f:
            graph = nn = torch.jit.load(f, map_location="cpu")

        graph.eval()
        logging.info(f"Initialize:\n{nn}")

        # Infer the shape of each input from the batch file.
        # Assumes the output is stored as "y"
        with open_file(batch_file, "rb") as f:
            batch_file = h5py.File(io.BytesIO(f.read()))
            input_shape_dict = {
                key: (batch_size,) + tuple(batch_file[key].shape[1:])
                for key in batch_file.keys()
                if key != "y"
            }

        # the network will have some different keyword
        # arguments required for export depending on
        # the target inference platform
        # TODO: hardcoding these kwargs for now, but worth
        # thinking about a more robust way to handle this
        kwargs = {}
        if platform == qv.Platform.ONNX:
            kwargs["opset_version"] = 13

            # turn off graph optimization because of this error
            # https://github.com/triton-inference-server/server/issues/3418
            aframe.config.optimization.graph.level = -1
        elif platform == qv.Platform.TENSORRT:
            kwargs["use_fp16"] = False

        aframe.export_version(
            graph,
            input_shapes=input_shape_dict,
            output_names=output_names,
            **kwargs,
        )

    # now try to create an ensemble that has a snapshotter
    # at the front for streaming new data to
    ensemble_name = "aframe-stream"
    try:
        # first see if we have an existing
        # ensemble with the given name
        ensemble = repo.models[ensemble_name]
    except KeyError:
        # if we don't, create one
        ensemble = repo.add(ensemble_name, platform=qv.Platform.ENSEMBLE)
        preproc_model = add_streaming_input_preprocessor(
            ensemble,
            input_shape_dict,
            batch_size,
            num_ifos,
            psd_length=psd_length,
            sample_rate=sample_rate,
            kernel_length=kernel_length,
            inference_sampling_rate=inference_sampling_rate,
            fduration=fduration,
            preprocessor=preprocessor,
            preproc_instances=preproc_instances,
            streams_per_gpu=streams_per_gpu,
        )
        for key in input_shape_dict.keys():
            ensemble.pipe(preproc_model.outputs[key], aframe.inputs[key])

        # export the ensemble model, which basically amounts
        # to writing its config and creating an empty version entry
        # this takes care of number of neural network outputs and its name
        for name in output_names:
            ensemble.add_output(aframe.outputs[name])
        ensemble.export_version(None)
    else:
        # if there does already exist an ensemble by
        # the given name, make sure it has aframe
        # and the snapshotter as a part of its models
        if aframe not in ensemble.models:
            raise ValueError(
                "Ensemble model '{}' already in repository "
                "but doesn't include model 'aframe'".format(ensemble_name)
            )
        # TODO: checks for snapshotter and preprocessor

    # keep snapshot states around for a long time in case there are
    # unexpected bottlenecks which throttle update for a few seconds
    snapshotter = repo.models["snapshotter"]
    snapshotter.config.sequence_batching.max_sequence_idle_microseconds = int(
        6e10
    )

    # Limit the number of threads the snapshotter uses to avoid system limits
    snapshotter.config.parameters["intra_op_thread_count"].string_value = "2"

    snapshotter.config.write()

    # Finally, in the no-compile path, swap the placeholder aframe (used only
    # to wire the ensemble above) for the python-backend model files.
    if skip_compile:
        _install_python_aframe(
            repository_directory,
            aframe_model_dir,
            batch_size=batch_size,
            num_ifos=num_ifos,
            seq_len=seq_len,
            output_name=output_names[0],
        )
