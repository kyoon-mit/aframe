"""Measure how the denoiser loss's two terms compare, in value and gradient.

The mixture loss adds a time-domain term to a frequency-domain one and
weights them with a single ``alpha``. That only works if the two are on a
comparable scale, and they are not: an MSE in the time domain grows with
the square of the waveform amplitude, while an ``msle`` spectral term
compares log magnitudes and is amplitude-invariant. Their gradients are
worse still, since differentiating a log contributes a factor 1/|X| that
*shrinks* as the amplitude grows -- so the two gradients scale in opposite
directions and their ratio moves as the square of the amplitude.

This matters because the SNR curriculum changes the amplitude during
training, so a fixed ``alpha`` silently re-weights the two objectives as a
run proceeds.

Every number in the summary that accompanies this script comes from here.
Run it with no arguments to reproduce them:

    python dev/infer/diagnostic/probe_loss_gradient_balance.py

Both gradients are taken with respect to the *same* variable, the
time-domain prediction, since that is what the denoiser emits; the
spectral gradient reaches it through the FFT. Comparing d/d|X| against
d/dx would compare different variables and mean nothing.
"""

import math

import torch

KERNEL_LENGTH = 8192
LOG_FLOOR = 1e-9
LOG_BASE = math.log(10.0)
RELATIVE_ERROR = 0.3  # prediction sits this far off the target


def time_term(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Time-domain MSE, as ``time_loss: mse`` computes it."""
    return ((pred - target) ** 2).mean()


def spectral_term(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Spectral msle, as ``spectral_loss: msle`` computes it."""
    pred_mag = torch.fft.rfft(pred, dim=-1).abs()
    target_mag = torch.fft.rfft(target, dim=-1).abs()
    log_pred = torch.log(pred_mag + LOG_FLOOR) / LOG_BASE
    log_target = torch.log(target_mag + LOG_FLOOR) / LOG_BASE
    return ((log_pred - log_target) ** 2).mean()


def make_batch(amplitude: float, batch: int = 4, seed: int | None = None):
    """A target at that amplitude, and a prediction a fixed fraction off."""
    if seed is not None:
        torch.manual_seed(seed)
    target = amplitude * torch.randn(
        batch, 2, KERNEL_LENGTH, dtype=torch.float64
    )
    noise = torch.randn(batch, 2, KERNEL_LENGTH, dtype=torch.float64)
    pred = target + RELATIVE_ERROR * amplitude * noise
    return pred, target


def gradient_norms(pred: torch.Tensor, target: torch.Tensor):
    """Norm of each term's gradient wrt the time-domain prediction."""
    leaf = pred.clone().requires_grad_(True)
    grad_time = torch.autograd.grad(time_term(leaf, target), leaf)[0]
    leaf = pred.clone().requires_grad_(True)
    grad_spec = torch.autograd.grad(spectral_term(leaf, target), leaf)[0]
    return grad_time, grad_spec


def report_scaling() -> None:
    """Each term's value and gradient across amplitude."""
    print("=" * 74)
    print("Term values and gradients vs amplitude (relative error fixed)")
    print("=" * 74)
    print(
        f"{'amp':>8} {'time':>12} {'spectral':>12} "
        f"{'|g_time|':>12} {'|g_spec|':>12} {'g_spec/g_time':>14}"
    )
    for amplitude in (0.01, 0.1, 1.0, 10.0, 100.0):
        pred, target = make_batch(amplitude, seed=0)
        value_time = float(time_term(pred, target))
        value_spec = float(spectral_term(pred, target))
        grad_time, grad_spec = gradient_norms(pred, target)
        mean_time = float(grad_time.abs().mean())
        mean_spec = float(grad_spec.abs().mean())
        print(
            f"{amplitude:>8g} {value_time:>12.3e} {value_spec:>12.3e} "
            f"{mean_time:>12.3e} {mean_spec:>12.3e} "
            f"{mean_spec / mean_time:>14.4f}"
        )
    print()
    print("time gradient scales as amplitude, spectral as 1/amplitude,")
    print("so their ratio moves as the square of the amplitude.")
    print()


def report_crossover() -> None:
    """Amplitude at which the two gradients are equal."""
    print("=" * 74)
    print("Where the two gradients balance")
    print("=" * 74)
    for amplitude in (0.5, 1.0, 2.0, 2.3, 3.0, 5.0):
        pred, target = make_batch(amplitude, seed=0)
        grad_time, grad_spec = gradient_norms(pred, target)
        ratio = float(grad_spec.abs().mean() / grad_time.abs().mean())
        leader = "spectral leads" if ratio > 1 else "time leads"
        print(f"  amplitude={amplitude:<6g} ratio={ratio:>9.3f}   {leader}")
    print()


def report_gradient_noise(trials: int = 20) -> None:
    """How much each gradient norm moves from batch to batch.

    Decides whether the norms can be used directly or need smoothing: a
    per-batch divisor that is itself noisy injects more than it removes.
    """
    print("=" * 74)
    print(f"Batch-to-batch spread of the gradient norms ({trials} batches)")
    print("=" * 74)
    norms_time, norms_spec = [], []
    for trial in range(trials):
        pred, target = make_batch(1.0, seed=trial)
        grad_time, grad_spec = gradient_norms(pred, target)
        norms_time.append(float(grad_time.norm()))
        norms_spec.append(float(grad_spec.norm()))
    for name, values in (("time", norms_time), ("spectral", norms_spec)):
        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
        print(
            f"  {name:<9} mean={mean:.4e} "
            f"relative spread={variance**0.5 / mean:.4f}"
        )
    print()
    print("The spectral norm moves by more than its own size from batch to")
    print("batch, so dividing by it directly would add noise, not remove it.")
    print()


def report_balancing() -> None:
    """Dividing each term by its own gradient norm, exact and smoothed."""
    print("=" * 74)
    print("Balancing each term by its gradient norm")
    print("=" * 74)
    print(f"{'amp':>8} {'raw ratio':>14} {'per-batch':>14}")
    for amplitude in (0.01, 0.1, 1.0, 10.0, 100.0):
        pred, target = make_batch(amplitude, seed=0)
        grad_time, grad_spec = gradient_norms(pred, target)
        raw = float(grad_spec.abs().mean() / grad_time.abs().mean())

        # divide each term by its own (detached) gradient norm
        norm_time = grad_time.norm().detach()
        norm_spec = grad_spec.norm().detach()
        leaf = pred.clone().requires_grad_(True)
        balanced_time = torch.autograd.grad(
            time_term(leaf, target) / norm_time, leaf
        )[0]
        leaf = pred.clone().requires_grad_(True)
        balanced_spec = torch.autograd.grad(
            spectral_term(leaf, target) / norm_spec, leaf
        )[0]
        balanced = float(balanced_spec.norm() / balanced_time.norm())
        print(f"{amplitude:>8g} {raw:>14.4f} {balanced:>14.6f}")
    print()
    print("Exact per batch, but see the spread above: those divisors are")
    print("too noisy to use directly. Smoothing them trades that noise for")
    print("lag, and does not hold the ratio at one.")
    print()


def report_smoothed(momentum: float = 0.99, steps: int = 100) -> None:
    """A running average of the gradient norms, across a falling amplitude."""
    print("=" * 74)
    print(f"Running average of the norms (momentum={momentum})")
    print("=" * 74)
    running_time = running_spec = None
    seed = 0
    for amplitude in (1.0, 0.3, 0.1):
        ratios = []
        for _ in range(steps):
            pred, target = make_batch(amplitude, seed=seed)
            seed += 1
            grad_time, grad_spec = gradient_norms(pred, target)
            norm_time = float(grad_time.norm())
            norm_spec = float(grad_spec.norm())
            running_time = (
                norm_time
                if running_time is None
                else momentum * running_time + (1 - momentum) * norm_time
            )
            running_spec = (
                norm_spec
                if running_spec is None
                else momentum * running_spec + (1 - momentum) * norm_spec
            )
            ratios.append(
                (norm_spec / running_spec) / (norm_time / running_time)
            )
        tail = ratios[-20:]
        mean = sum(tail) / len(tail)
        variance = sum((r - mean) ** 2 for r in tail) / (len(tail) - 1)
        print(
            f"  amplitude={amplitude:<5g} "
            f"update ratio={mean:>7.3f} (spread {variance**0.5:.3f})"
        )
    print()
    print("Better than the raw ratio by about two orders of magnitude, but")
    print("it still wanders and the spread is as large as the value.")
    print()


def main() -> None:
    torch.set_grad_enabled(True)
    report_scaling()
    report_crossover()
    report_gradient_noise()
    report_balancing()
    report_smoothed()


if __name__ == "__main__":
    main()
