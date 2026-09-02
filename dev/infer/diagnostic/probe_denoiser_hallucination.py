"""Does the denoiser extract signal, or does it hallucinate a chirp?

A denoiser trained on injected strain can succeed for the wrong reason. If it
has learned the *shape* of a BNS chirp rather than how to *extract* one, it
will happily paint that shape onto anything it is shown, including pure
detector noise. Standard validation never catches this, because every
validation kernel contains an injection.

This probe removes the injection. Whitening is linear, so

    whiten(noise + signal) = whiten(noise) + whiten(signal)

and for a kernel with no injection the clean target is identically zero. An
honest signal extractor should therefore return approximately zero. A model
that returns chirp-like structure is telling on itself.

Two blocks are plotted on shared axes, matching the layout of
`DenoiserEvolutionCallback` (time domain left, abs(rfft) right, one row per
example and interferometer):

    BG   background-only input, target is the zero line
    INJ  injected reference batch, for scale comparison

The headline number is `out/in RMS` on the background block, and the ratio of
background output RMS to injected output RMS. Near zero means extraction.
Near the injected level means the model has memorised a waveform.

Usage
-----
    uv run python dev/infer/diagnostic/probe_denoiser_hallucination.py \\
        --ckpt   .../checkpoints/last.ckpt \\
        --config .../merger_4s_den_reg_cls_64s64n4_resnet14_local.yaml \\
        --n-examples 4
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

# Colours shared with the evolution plots, so the two read the same way.
COLOR_INPUT = "0.6"
COLOR_TARGET = "k"
COLOR_PRED_BG = "tab:red"
COLOR_PRED_INJ = "tab:blue"


# --------------------------------------------------------------------------
# arguments
# --------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--ckpt", required=True, type=Path, help="checkpoint to probe"
    )
    p.add_argument(
        "--config",
        required=True,
        type=Path,
        help="training config the checkpoint came from",
    )
    p.add_argument(
        "--n-examples",
        type=int,
        default=4,
        help="strains to plot per block",
    )
    p.add_argument(
        "--sample-rate",
        type=float,
        default=None,
        help="override the config sample rate",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="output png (default: next to the checkpoint)",
    )
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------


def _load_trusting(path, device):
    """torch.load with weights_only disabled.

    torch >= 2.6 refuses to unpickle arbitrary classes, and these checkpoints
    carry project and ml4gw objects in their saved hyperparameters. The file
    is our own training output, so full unpickling is appropriate here.
    """
    return torch.load(path, map_location=device, weights_only=False)


def load_model(ckpt, config_path, device):
    """Rebuild the model from config, then load the trained weights.

    `arch` is a constructor argument that is not written into the checkpoint
    hyperparameters, so `load_from_checkpoint` alone cannot reconstruct the
    module. Instantiating from the same config the run used, then loading the
    state dict, sidesteps that.
    """
    from jsonargparse import ArgumentParser

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    arch_cfg = cfg["model"]["init_args"]["arch"]
    module_path, _, class_name = arch_cfg["class_path"].rpartition(".")
    arch_cls = getattr(
        __import__(module_path, fromlist=[class_name]), class_name
    )

    parser = ArgumentParser()
    parser.add_class_arguments(arch_cls, "arch")
    model = parser.instantiate_classes(
        parser.parse_object({"arch": arch_cfg.get("init_args", {})})
    ).arch

    # The LightningModule stores the network under `model.`; strip that prefix
    # so the weights land on the bare architecture.
    state = _load_trusting(ckpt, device)
    weights = {
        key[len("model.") :]: value
        for key, value in state["state_dict"].items()
        if key.startswith("model.")
    }
    missing, unexpected = model.load_state_dict(weights, strict=False)
    if missing:
        print(f"[warn] {len(missing)} missing keys, e.g. {missing[:3]}")
    if unexpected:
        print(
            f"[warn] {len(unexpected)} unexpected keys, e.g. {unexpected[:3]}"
        )

    epoch = state.get("epoch")
    if epoch is not None:
        print(f"loaded checkpoint from epoch {epoch}")

    return model.eval().to(device)


def load_datamodule(config_path, model, device):
    """Rebuild the datamodule exactly as the trainer would have built it."""
    from jsonargparse import ArgumentParser

    from train.data.supervised.time_domain import (
        DenoisingTimeDomainSupervisedAframeDataset,
    )

    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    data_args = cfg["data"]["init_args"]

    parser = ArgumentParser()
    parser.add_class_arguments(
        DenoisingTimeDomainSupervisedAframeDataset, "data"
    )
    dm = parser.instantiate_classes(
        parser.parse_object({"data": data_args})
    ).data

    # The datamodule reaches into its trainer for world size, rank, device
    # and accelerator. A stand-in supplies exactly those, avoiding a full
    # Trainer around what is only a bare architecture here.
    import lightning.pytorch as pl

    class _TrainerStub:
        world_size = 1
        global_rank = 0
        num_devices = 1
        device_ids = [0]
        # on_after_batch_transfer branches on these to pick the training
        # augmentation path, which is the one that performs injection.
        training = True
        testing = False
        validating = False
        sanity_checking = False
        predicting = False

        def __init__(self, device):
            self.lightning_module = type(
                "_ModuleStub", (), {"device": device}
            )()
            self.accelerator = (
                pl.accelerators.CUDAAccelerator()
                if device.type == "cuda"
                else pl.accelerators.CPUAccelerator()
            )

    dm.trainer = _TrainerStub(device)

    dm.setup("fit")
    dm.build_transforms()

    # Move every transform, including buffers nested inside them such as the
    # spectral-density window, onto the target device. The transforms are
    # plain attributes rather than registered submodules, so walk both.
    def move_modules(obj, device, depth=0):
        """Recursively push every nested nn.Module onto the device."""
        if depth > 3:
            return
        for attribute in vars(obj).values():
            if isinstance(attribute, torch.nn.Module):
                attribute.to(device)
                move_modules(attribute, device, depth + 1)

    move_modules(dm, device)

    sample_rate = float(data_args["sample_rate"])
    return dm, sample_rate


# --------------------------------------------------------------------------
# inference
# --------------------------------------------------------------------------


def denoise(model, x):
    """Run the model and keep only the denoised strain."""
    out = model(x)
    return out[0] if isinstance(out, (tuple, list)) else out


def unwrap_strain(raw_batch):
    """Pull the strain tensor out of the dataloader's nested batch.

    The strain loader emits ``[X]`` (and ``([X], (waveforms, params))`` when
    waveforms come from disk), so unwrap until a tensor appears.
    """
    item = raw_batch
    while isinstance(item, (tuple, list)):
        item = item[0]

    # The strain dataset is itself batched, and the DataLoader wraps that in
    # a further batch dimension of one; drop it to get (B, ifos, samples).
    while item.ndim > 3 and item.shape[0] == 1:
        item = item.squeeze(0)
    return item


def background_only(dm, model, raw_batch, n_examples, device):
    """Whiten raw background with no injection, then denoise it."""
    background = unwrap_strain(raw_batch).to(device)

    with torch.no_grad():
        kernels, psd = dm.psd_estimator(background)
        whitened = dm.whitener(kernels, psd)[:n_examples]
        predicted = denoise(model, whitened)

    return whitened.float().cpu().numpy(), predicted.float().cpu().numpy()


def to_device(item, device):
    """Recursively move tensors inside an arbitrarily nested batch."""
    if torch.is_tensor(item):
        return item.to(device)
    if isinstance(item, (tuple, list)):
        return type(item)(to_device(sub, device) for sub in item)
    return item


def injected_reference(dm, model, raw_batch, n_examples, device):
    """Denoise a normal injected batch, for side-by-side scale.

    Runs the datamodule's own on-device augmentation, which is what performs
    the injection and produces the paired clean target.
    """
    with torch.no_grad():
        batch = dm.on_after_batch_transfer(to_device(raw_batch, device), 0)
        if not (isinstance(batch, (tuple, list)) and len(batch) >= 2):
            return None

        noisy, clean = batch[0], batch[1]
        injected = (clean.abs().amax(dim=(1, 2)) > 0).nonzero().flatten()
        injected = injected[:n_examples]
        if not len(injected):
            return None

        noisy = noisy[injected].to(device)
        clean = clean[injected].to(device)
        predicted = denoise(model, noisy)

    return (
        noisy.float().cpu().numpy(),
        predicted.float().cpu().numpy(),
        clean.float().cpu().numpy(),
    )


# --------------------------------------------------------------------------
# plotting
# --------------------------------------------------------------------------


def spectrum(x, sample_rate):
    """One-sided magnitude spectrum, DC dropped, floored for log axes."""
    magnitude = np.abs(np.fft.rfft(x))
    freqs = np.fft.rfftfreq(x.shape[0], d=1.0 / sample_rate)
    return freqs[1:], np.maximum(magnitude[1:], 1e-30)


def rms(x, axis=-1):
    return np.sqrt(np.mean(x**2, axis=axis))


def draw_background_row(ax_time, ax_freq, noisy, predicted, t, sample_rate):
    """One background-only strain: prediction against a zero target."""
    ax_time.plot(
        t,
        noisy,
        lw=0.5,
        color=COLOR_INPUT,
        alpha=0.45,
        label="background input",
    )
    ax_time.axhline(
        0.0, lw=0.9, color=COLOR_TARGET, label="target (zero, no injection)"
    )
    ax_time.plot(t, predicted, lw=0.9, color=COLOR_PRED_BG, label="prediction")
    ax_time.set_title(
        f"prediction RMS = {rms(predicted):.3e}   (input {rms(noisy):.3e})",
        fontsize=8,
        loc="right",
    )

    for series, color, alpha, label in (
        (noisy, COLOR_INPUT, 0.45, "background input"),
        (predicted, COLOR_PRED_BG, 1.0, "prediction"),
    ):
        freqs, magnitude = spectrum(series, sample_rate)
        ax_freq.loglog(
            freqs, magnitude, lw=0.7, color=color, alpha=alpha, label=label
        )


def draw_injected_row(ax_time, ax_freq, noisy, predicted, target, sample_rate):
    """One injected strain, time axis referenced to the merger."""
    merger = int(np.argmax(np.abs(target)))
    t = (np.arange(target.shape[0]) - merger) / sample_rate

    ax_time.plot(
        t, noisy, lw=0.5, color=COLOR_INPUT, alpha=0.45, label="noisy input"
    )
    ax_time.plot(t, target, lw=0.9, color=COLOR_TARGET, label="target")
    ax_time.plot(
        t, predicted, lw=0.9, color=COLOR_PRED_INJ, label="prediction"
    )
    ax_time.set_title(
        f"MSE = {np.mean((predicted - target) ** 2):.3e}",
        fontsize=8,
        loc="right",
    )

    for series, color, alpha, label in (
        (noisy, COLOR_INPUT, 0.45, "noisy input"),
        (target, COLOR_TARGET, 1.0, "target"),
        (predicted, COLOR_PRED_INJ, 1.0, "prediction"),
    ):
        freqs, magnitude = spectrum(series, sample_rate)
        ax_freq.loglog(
            freqs, magnitude, lw=0.7, color=color, alpha=alpha, label=label
        )


def draw(bg, injection, sample_rate, title):
    """Background block above, injected block below, shared figure."""
    bg_noisy, bg_pred = bg
    n_bg, n_ifos, length = bg_noisy.shape
    n_inj = injection[0].shape[0] if injection else 0
    n_rows = (n_bg + n_inj) * n_ifos

    fig, axes = plt.subplots(
        n_rows, 2, figsize=(14, 3.6 * n_rows), squeeze=False
    )
    t = np.arange(length) / sample_rate
    row = 0

    for example in range(n_bg):
        for ifo in range(n_ifos):
            ax_time, ax_freq = axes[row]
            draw_background_row(
                ax_time,
                ax_freq,
                bg_noisy[example, ifo],
                bg_pred[example, ifo],
                t,
                sample_rate,
            )
            ax_time.set_ylabel(f"background {example} / ifo {ifo}")
            ax_freq.set_ylabel("|rfft|")
            if row == 0:
                ax_time.legend(fontsize=7, ncol=3, loc="upper left")
                ax_freq.legend(fontsize=7, loc="upper right")
                ax_freq.set_title("abs(rfft) magnitude", fontsize=9)
            row += 1

    if injection:
        inj_noisy, inj_pred, inj_target = injection
        for example in range(n_inj):
            for ifo in range(n_ifos):
                ax_time, ax_freq = axes[row]
                draw_injected_row(
                    ax_time,
                    ax_freq,
                    inj_noisy[example, ifo],
                    inj_pred[example, ifo],
                    inj_target[example, ifo],
                    sample_rate,
                )
                ax_time.set_ylabel(f"injected {example} / ifo {ifo}")
                ax_freq.set_ylabel("|rfft|")
                if example == 0 and ifo == 0:
                    ax_time.legend(fontsize=7, ncol=3, loc="upper left")
                    ax_freq.legend(fontsize=7, loc="upper right")
                row += 1

    axes[-1][0].set_xlabel("time [s] (injected rows: time to merger)")
    axes[-1][1].set_xlabel("frequency [Hz]")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    return fig


# --------------------------------------------------------------------------
# verdict
# --------------------------------------------------------------------------


def report(bg, injection, ckpt):
    """Print the numbers that decide extraction versus hallucination."""
    bg_noisy, bg_pred = bg
    input_rms = rms(bg_noisy)
    output_rms = rms(bg_pred)
    ratio = output_rms / np.maximum(input_rms, 1e-30)

    print("\n=== denoiser hallucination probe ===")
    print(f"checkpoint  : {ckpt}")
    print(f"examples    : {bg_noisy.shape[0]} x {bg_noisy.shape[1]} ifos")
    print(f"input  RMS  : {input_rms.mean():.4e}")
    print(f"output RMS  : {output_rms.mean():.4e}")
    print(f"out/in      : mean {ratio.mean():.4f}, max {ratio.max():.4f}")

    if not injection:
        print("\nNo injected reference batch available for comparison.")
        return

    injected_rms = rms(injection[1])
    leakage = output_rms.mean() / max(injected_rms.mean(), 1e-30)
    print(f"injected output RMS : {injected_rms.mean():.4e}")
    print(f"background/injected : {leakage:.4f}")
    print(
        "\nA signal extractor drives the background output far below the "
        "injected output, so this ratio should be small. A ratio near one "
        "means the model emits comparable structure with or without a "
        "signal present, which is a learned shape rather than extraction."
    )


# --------------------------------------------------------------------------


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device)
    model = load_model(args.ckpt, args.config, device)
    dm, config_sample_rate = load_datamodule(args.config, model, device)
    sample_rate = args.sample_rate or config_sample_rate

    raw_batch = next(iter(dm.train_dataloader()))
    bg = background_only(dm, model, raw_batch, args.n_examples, device)
    injection = injected_reference(
        dm, model, raw_batch, args.n_examples, device
    )

    report(bg, injection, args.ckpt)

    out = args.out or args.ckpt.parent / "denoiser_hallucination_probe.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig = draw(
        bg,
        injection,
        sample_rate,
        f"denoiser hallucination probe  |  {args.ckpt.name}",
    )
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
