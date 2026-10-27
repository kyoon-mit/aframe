"""Chirp-mass accuracy of the Gaussian-noise den-reg checkpoints by SNR.

Windows: the S4D Gaussian config's validation path (Gaussian copies of the
validation background, validation waveforms, SNR power law -3 from 4), the
same windows for every checkpoint. Each network runs twice on every window:
in eval mode (no dropout), and with dropout on as in training, with one
dropout draw. Runs on the GPU if one is visible, in chunks of 32 windows and
within a tenth of its memory, and on the CPU otherwise. For the Mamba-2
checkpoint, also the 1/e memory 1 / (dt A) of each denoiser head on the first
batch, in seconds.

usage: eval_by_snr.py <out.npz> <passes> <name=config:ckpt> [...]
"""

import sys

import lightning.pytorch as pl
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from jsonargparse import ArgumentParser

from train.experimental.slic.dump_windows import build_data_module

SAMPLE_RATE, KERNEL_LENGTH = 2048.0, 4.0
BANDS = [(4, 5), (5, 6), (6, 8), (8, 12), (12, 100), (8, 100), (4, 100)]
CHUNK = 32


def build_net(config, ckpt):
    with open(config) as f:
        model = yaml.safe_load(f)["model"]["init_args"]
    arch = model["arch"]
    arch["init_args"].update(num_ifos=2, sample_rate=SAMPLE_RATE, kernel_length=KERNEL_LENGTH)
    if "compile" in arch["init_args"]:
        arch["init_args"]["compile"] = False
    parser = ArgumentParser()
    parser.add_subclass_arguments(torch.nn.Module, "arch")
    net = parser.instantiate_classes(parser.parse_object({"arch": arch})).arch
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    net.load_state_dict({k[len("model."):]: v for k, v in state["state_dict"].items()
                         if k.startswith("model.")})
    print(f"{ckpt}: epoch {state['epoch']}", flush=True)
    return net.eval(), state["state_dict"]["y_mean"], state["state_dict"]["y_std"]


def chirp_mass(net, X, y_mean, y_std, dropout):
    net.train(dropout)
    with torch.no_grad():
        est = torch.cat([net(x)[1][:, 0] for x in X.split(CHUNK)])
    net.eval()
    return (est.cpu() * y_std[0] + y_mean[0]).numpy()


def mamba_memory(net, X):
    """1/e memory of every denoiser head at every sample, in seconds."""
    out = []

    def hook(layer):
        def f(_, inputs, proj):
            dt = F.softplus(proj[..., -layer.nheads:] + layer.dt_bias)
            out.append((1 / (dt * layer.A_log.exp())).flatten(0, 1).cpu() / SAMPLE_RATE)
        return f

    layers = net.model.denoiser.layers
    handles = [layer.in_proj.register_forward_hook(hook(layer)) for layer in layers]
    with torch.no_grad():
        net(X[:CHUNK])
    for h in handles:
        h.remove()
    return torch.stack(out)  # (layers, samples, heads)


def main():
    out, passes = sys.argv[1], int(sys.argv[2])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(0.1)
    nets = {}
    for arg in sys.argv[3:]:
        name, paths = arg.split("=", 1)
        config, ckpt = paths.split(":", 1)
        nets[name] = build_net(config, ckpt)
        nets[name][0].to(device)
    data_config = sys.argv[3].split("=", 1)[1].split(":", 1)[0]
    with open(data_config) as f:
        pl.seed_everything(yaml.safe_load(f)["seed_everything"])
    dm = build_data_module(data_config, "val")
    rows = {"snr": [], "true": []}
    rows |= {f"{name}{tag}": [] for name in nets for tag in ("", "_dropout")}
    memory = None
    for p in range(passes):
        for batch in dm.val_dataloader():
            batch = dm.on_before_batch_transfer(batch, 0)
            X, _, _, params = dm.on_after_batch_transfer(batch, 0)
            keep = ~torch.isnan(params["chirp_mass"])
            X = X[keep].to(device)
            rows["snr"].append(params["snr"][keep].numpy())
            rows["true"].append(params["chirp_mass"][keep].numpy())
            for name, (net, y_mean, y_std) in nets.items():
                rows[name].append(chirp_mass(net, X, y_mean, y_std, False))
                rows[f"{name}_dropout"].append(chirp_mass(net, X, y_mean, y_std, True))
                if memory is None and hasattr(net.model.denoiser, "layers"):
                    memory = mamba_memory(net, X).numpy()
        print(f"pass {p + 1}/{passes}, {sum(map(len, rows['snr']))} windows", flush=True)
    rows = {k: np.concatenate(v) for k, v in rows.items()}
    np.savez(out, memory=memory if memory is not None else np.zeros(0), **rows)

    for tag, mode in (("", "eval mode"), ("_dropout", "dropout on")):
        print(f"share within 1% / 2% / 5% of the true chirp mass, {mode}")
        for lo, hi in BANDS:
            k = (rows["snr"] >= lo) & (rows["snr"] < hi)
            line = f"  SNR {lo}-{hi}: n={k.sum():5d}"
            for name in nets:
                e = np.abs(rows[name + tag][k] - rows["true"][k]) / rows["true"][k]
                line += f" | {name} " + " / ".join(f"{(e < t).mean():.3f}" for t in (0.01, 0.02, 0.05))
            print(line)
    if memory is not None:
        print("Mamba-2 denoiser 1/e memory (s), median and 99th percentile over samples:")
        for i, m in enumerate(memory):
            print(f"  layer {i}: " + ", ".join(
                f"head {h}: {np.median(m[:, h]):.4f} / {np.percentile(m[:, h], 99):.4f}"
                for h in range(m.shape[1])))


if __name__ == "__main__":
    main()
