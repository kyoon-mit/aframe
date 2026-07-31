"""Re-trace a trained S4D model ON GPU into a Triton repo's model.pt.

Tracing on CPU bakes `torch.arange(L, device=A.device)` (s4d.py) as a CPU
constant, so the libtorch server errors on GPU with a cuda/cpu device mismatch.
Tracing on cuda bakes it on cuda:0 (where Triton serves), fixing the mismatch.

Accepts either a plain net state_dict (`.pt`) or a Lightning checkpoint
(`.ckpt`), whose `model.` prefix is stripped and whose `y_std` is folded in so
the served statistic is sigma in physical chirp-mass units. That scaling is a
constant, so it changes no ranking and no sensitive volume.

    uv run python trace_model_gpu.py --weights <.ckpt|.pt> --out <model.pt>
"""

import argparse

import torch
import torch.nn as nn
import yaml

from architectures.supervised import (
    SupervisedS4Model,
    SupervisedS4ModelPrenorm,
)
from ml4gw.nn.ssm.s4d import S4Model

# architecture class_path leaf name in the run config -> where it lives, so it
# can be imported and built with the config's own init_args (no dims by hand).
# S4Model is the plain regression net (takes d_input); the Supervised* classes
# take num_ifos. Anything not listed still works if it is importable from
# architectures.supervised.
ARCH_MODULE = {
    "S4Model": "ml4gw.nn.ssm.s4d",
}


def arch_from_config(config_path):
    """Return (class_path, init_args, outputs) for the arch in a config.yaml.

    Walks the yaml for the model's ``arch`` block (a ``class_path`` +
    ``init_args``) and infers the served ``outputs`` from its ``d_output``:
    a single logit -> ``score`` (classifier), two channels -> ``mass_sigma``
    (regression mean + sigma). Raises if no arch block is found.
    """
    with open(config_path) as config_file:
        config = yaml.safe_load(config_file)

    def find_arch(node):
        if isinstance(node, dict):
            class_path = str(node.get("class_path", ""))
            init_args = node.get("init_args", {})
            if (
                class_path.startswith("architectures.")
                or class_path.rsplit(".", 1)[-1] in ARCH_MODULE
            ) and "d_output" in init_args:
                return class_path, init_args
            for value in node.values():
                found = find_arch(value)
                if found:
                    return found
        elif isinstance(node, list):
            for item in node:
                found = find_arch(item)
                if found:
                    return found
        return None

    found = find_arch(config)
    if not found:
        raise SystemExit(f"no arch block found in {config_path}")
    class_path, init_args = found
    outputs = (
        "score" if int(init_args.get("d_output", 1)) == 1 else "mass_sigma"
    )
    return class_path, dict(init_args), outputs


def build_net_from_config(class_path, init_args):
    """Import and instantiate the arch class with the config's own init_args.

    Supplies the two-detector input the served kernel has: classes that take
    ``num_ifos`` get 2, the plain ``S4Model`` gets ``d_input=2``. Every other
    hyperparameter (dims, dropout, dt, prenorm, ...) comes from the config, so
    the built net matches the trained one exactly.
    """
    import importlib
    import inspect

    leaf = class_path.rsplit(".", 1)[-1]
    module_name = ARCH_MODULE.get(leaf, "architectures.supervised")
    net_class = getattr(importlib.import_module(module_name), leaf)

    kwargs = dict(init_args)
    parameters = inspect.signature(net_class.__init__).parameters
    if "num_ifos" in parameters:
        kwargs.setdefault("num_ifos", 2)
    elif "d_input" in parameters:
        kwargs.setdefault("d_input", 2)
    return net_class(**kwargs)


DEFAULT_WEIGHTS = "/n/holystore01/LABS/iaifi_lab/Lab/kyoon/MODEL/aframe/reg-dev-latest/template/reg/merger_4s/chirp_mass_snr_8_50_60-64s_d64_s64_l4_on_disk_id2/1/chirp_mass_id2.pt"  # noqa: E501
DEFAULT_OUT = "/n/holystore01/LABS/iaifi_lab/Lab/kyoon/MODEL/aframe/reg-dev-latest/triton_ts/merger_4s/chirp_mass_snr_8_50_60-64s_d64_s64_l4_on_disk_id2/aframe/1/model.pt"  # noqa: E501


class Served(nn.Module):
    """Whitened X (B, 2, L) -> discriminator score.

    Two score functions are available (the discriminator IS just a score
    function, so swapping it is the only model-side change needed):

    - ``sigma``: legacy single channel, -sigma (B, 1). Larger = more confident.
    - ``mass_sigma``: two channels [mass, sigma] (B, 2) in physical chirp-mass
      units. Caching BOTH lets any metric (-sigma, mass/sigma, mass-k*sigma,
      ...) be computed offline in postprocessing without ever re-serving.
    - ``score``: the net's raw output (B, 1) passed straight through -- for a
      classification model, this is the detection logit (higher = more
      signal-like), which is the ranking statistic directly.

    sigma (not variance) is emitted: with sigma < 1 here, variance = sigma^2 is
    smaller and would integrate more weakly, so sigma is the better channel.
    """

    def __init__(self, net, y_mean=0.0, y_std=1.0, outputs="sigma"):
        super().__init__()
        self.net = net
        self.softplus = nn.Softplus()
        self.outputs = outputs
        self.register_buffer("y_mean", torch.as_tensor(y_mean).float())
        self.register_buffer("y_std", torch.as_tensor(y_std).float())

    def forward(self, x):
        x = x / x.std(dim=-1, keepdim=True).clamp(min=1e-8)
        out = self.net(x)
        if self.outputs == "score":
            return out  # (B, 1) classifier logit; higher = more signal-like
        sigma = self.y_std * torch.sqrt(self.softplus(out[:, 1:2]))
        if self.outputs == "sigma":
            return -sigma
        mass = out[:, 0:1] * self.y_std + self.y_mean
        return torch.cat([mass, sigma], dim=1)


def load_weights(path):
    """Return (net state_dict, y_mean, y_std) from a .ckpt or a plain .pt."""
    if not path.endswith(".ckpt"):
        sd = torch.load(path, map_location="cpu", weights_only=True)
        return sd, 0.0, 1.0

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state_dict = checkpoint["state_dict"]
    net_state = {
        key[len("model.") :]: value
        for key, value in state_dict.items()
        if key.startswith("model.")
    }
    # regression ckpts carry target normalization; classification ones do not
    y_mean = (
        float(state_dict["y_mean"].item()) if "y_mean" in state_dict else 0.0
    )
    y_std = float(state_dict["y_std"].item()) if "y_std" in state_dict else 1.0
    print(
        f"lightning ckpt: epoch {checkpoint.get('epoch')}, "
        f"y_mean {y_mean}, y_std {y_std}, {len(net_state)} net tensors"
    )
    return net_state, y_mean, y_std


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument(
        "--config",
        default=None,
        help="run config.yaml: auto-sets --arch and dims from its model block "
        "(any flag you also pass explicitly still wins)",
    )
    parser.add_argument("--d-model", type=int, default=None)
    parser.add_argument("--d-state", type=int, default=None)
    parser.add_argument("--n-layers", type=int, default=None)
    parser.add_argument(
        "--arch",
        choices=["regression", "classification", "classification_prenorm"],
        default=None,
        help="regression = S4Model (2 outputs); classification = "
        "SupervisedS4Model; classification_prenorm = "
        "SupervisedS4ModelPrenorm (both 1 logit). Inferred from --config.",
    )
    parser.add_argument(
        "--outputs",
        choices=["sigma", "mass_sigma", "score"],
        default=None,
        help="discriminator: -sigma, [mass, sigma] (2ch), or raw score (cls). "
        "Inferred from --config (score for cls, mass_sigma for reg).",
    )
    parser.add_argument(
        "--kernel-size",
        type=int,
        default=8192,
        help="samples per inference window (4 s at 2048 Hz)",
    )
    return parser.parse_args()


def main():
    arguments = parse_arguments()
    assert torch.cuda.is_available(), "run on a GPU node"
    device = torch.device("cuda")

    # Preferred path: build the net straight from the run config's arch block,
    # so dims / dropout / dt / decoder variant all match the trained model and
    # nothing has to be passed by hand.
    if arguments.config:
        class_path, init_args, outputs = arch_from_config(arguments.config)
        arguments.outputs = arguments.outputs or outputs
        net = build_net_from_config(class_path, init_args)
        print(
            f"config: {class_path} outputs={arguments.outputs} "
            f"init_args={init_args}"
        )
    else:
        # backward-compatible manual path (no config given)
        arguments.arch = arguments.arch or "regression"
        arguments.outputs = arguments.outputs or "sigma"
        d_model = arguments.d_model or 64
        d_state = arguments.d_state or 64
        n_layers = arguments.n_layers or 4
        if arguments.arch in ("classification", "classification_prenorm"):
            cls = (
                SupervisedS4ModelPrenorm
                if arguments.arch == "classification_prenorm"
                else SupervisedS4Model
            )
            net = cls(
                num_ifos=2,
                d_output=1,
                d_model=d_model,
                d_state=d_state,
                n_layers=n_layers,
                dropout=0.2,
                dt_min=1e-3,
                dt_max=5.0,
            )
        else:
            net = S4Model(
                d_input=2,
                d_output=2,
                d_model=d_model,
                d_state=d_state,
                n_layers=n_layers,
                dropout=0.2,
                dt_min=1e-3,
                dt_max=5.0,
            )
    net_state, y_mean, y_std = load_weights(arguments.weights)
    net.load_state_dict(net_state, strict=True)
    served = (
        Served(net, y_mean=y_mean, y_std=y_std, outputs=arguments.outputs)
        .eval()
        .to(device)
    )

    with torch.no_grad():
        example = torch.randn(2, 2, arguments.kernel_size, device=device)
        traced = torch.jit.trace(served, example)
        # verify on GPU at a different batch (the failure mode we are fixing)
        check = traced(torch.randn(8, 2, arguments.kernel_size, device=device))
    expected_channels = 2 if arguments.outputs == "mass_sigma" else 1
    assert check.shape[1] == expected_channels, check.shape
    print(
        f"GPU trace OK ({arguments.outputs}):",
        tuple(check.shape),
        "finite:",
        bool(torch.isfinite(check).all()),
    )
    if arguments.outputs == "mass_sigma":
        print(
            "  mass range:",
            (float(check[:, 0].min()), float(check[:, 0].max())),
            "sigma range:",
            (float(check[:, 1].min()), float(check[:, 1].max())),
        )
    elif arguments.outputs == "score":
        print(
            "  score range:",
            (float(check.min()), float(check.max())),
        )
    torch.jit.save(traced, arguments.out)
    print("wrote", arguments.out)


if __name__ == "__main__":
    main()
