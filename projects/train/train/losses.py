import math

import torch
import torch.nn as nn
import torch.nn.functional as F

LOSS_FUNCTIONS = (
    "mse",
    "mae",
    "rmse",
    "huber",
    "smooth_l1",
    "smae",
    "logcosh",
)


def loss_helper(
    loss: str,
    pred: torch.Tensor,
    target: torch.Tensor,
    delta: float = 1.0,
) -> torch.Tensor:
    """Scalar discrepancy between two tensors, selected by name.

    All options are means over every element, so they stay comparable in
    magnitude and can be swapped without retuning the mixing weight. See
    LOSS_FUNCTIONS for the choices; ``delta`` is the huber/smooth_l1
    transition point (ignored otherwise).
    """
    if loss == "mse":
        return F.mse_loss(pred, target)
    if loss == "mae":
        return F.l1_loss(pred, target)
    if loss == "rmse":
        # clamp keeps the sqrt gradient finite when the error reaches zero
        return F.mse_loss(pred, target).clamp_min(1e-12).sqrt()
    if loss == "huber":
        return F.huber_loss(pred, target, delta=delta)
    if loss in ("smooth_l1", "smae"):
        return F.smooth_l1_loss(pred, target, beta=delta)
    if loss == "logcosh":
        # log(cosh(d)) as |d| + log1p(exp(-2|d|)) - log(2); no overflow
        d = (pred - target).abs()
        return (d + torch.log1p(torch.exp(-2.0 * d)) - math.log(2.0)).mean()
    raise ValueError(f"Unknown loss {loss!r}. Choose one of {LOSS_FUNCTIONS}.")


class TermStashMixin:
    """Stash per-term loss values for logging, optionally keeping the graph.

    Terms are detached by default, since they are logged rather than
    backpropagated. ``term_gradient_norms`` flips ``keep_term_graph`` for a
    single call so each term can be differentiated separately.
    """

    keep_term_graph = False

    def _stash(self, **terms) -> None:
        for name, value in terms.items():
            setattr(
                self,
                f"last_{name}_term",
                value if self.keep_term_graph else value.detach(),
            )


@torch.enable_grad()
def term_gradient_norms(
    loss_fn: nn.Module,
    pred: torch.Tensor,
    target: torch.Tensor,
) -> dict:
    """Gradient norm of each loss term wrt the prediction.

    Term values do not say which term drives the update; gradient norms do.
    For a time-vs-spectral mixture that balance moves with waveform
    amplitude, which the SNR curriculum changes mid-run.

    Differentiates wrt the prediction, not the parameters: it is what the
    terms share, and costs one backward through the loss only.

    Picks up any ``last_<name>_term`` attribute. Returns
    ``{name_value, name_gradnorm}``; empty if the loss exposes no terms.
    """
    prediction = pred.detach().clone().requires_grad_(True)
    target = target.detach()

    # Terms are stashed detached, since they exist to be logged and holding
    # their graphs would leak memory every step. Flip the flag for this one
    # call so they can be differentiated, then restore it. A loss without
    # the mixin has nothing to flip, so return nothing.
    previous = getattr(loss_fn, "keep_term_graph", None)
    if previous is None:
        return {}
    loss_fn.keep_term_graph = True
    try:
        loss_fn(prediction, target)
        terms = {
            attribute[len("last_") : -len("_term")]: getattr(
                loss_fn, attribute
            )
            for attribute in dir(loss_fn)
            if attribute.startswith("last_") and attribute.endswith("_term")
        }
        stats = {}
        differentiable = [
            (name, value)
            for name, value in terms.items()
            if torch.is_tensor(value) and value.requires_grad
        ]
        for index, (name, value) in enumerate(differentiable):
            stats[f"{name}_value"] = float(value.detach())
            # the graph is shared across terms, so keep it until the last one
            (gradient,) = torch.autograd.grad(
                value,
                prediction,
                retain_graph=index < len(differentiable) - 1,
                allow_unused=True,
            )
            stats[f"{name}_gradnorm"] = (
                0.0 if gradient is None else float(gradient.norm())
            )
        # a term that carries no gradient (an inactive branch, say) still has
        # a value worth logging
        for name, value in terms.items():
            stats.setdefault(f"{name}_value", float(value.detach()))
    finally:
        loss_fn.keep_term_graph = previous
    return stats


class ScheduledMixtureLoss(TermStashMixin, nn.Module):
    """Mixture of a time-domain and a frequency-domain denoiser loss.

    Aframe tensors are ``(B, C, L)``, so the FFT runs over the last dim ``L``.
    ``alpha`` is externally mutable so the training task can schedule it
    epoch-by-epoch (0 = pure spectral, 1 = pure time-domain).

    Both terms are computed per row and then averaged, so every signal
    carries equal weight regardless of its SNR. When ``density`` is set, each
    row's error is divided by that row's own "error of predicting zero"
    before averaging -- mean(err_row / scale_row), not
    mean(err_row) / mean(scale_row).

    Args:
        alpha: initial weight for the time term. Updated in place each epoch.
        density: if True, divide each row's error by that row's discrepancy
            against a zero prediction, making both terms dimensionless
            "error relative to predicting nothing" so alpha=0.5 is equal
            weight. If False, no scaling; rely on alpha for balance.
        time_loss: time-domain discrepancy, one of LOSS_FUNCTIONS.
        spectral_loss: 'mse' (default) compares |FFT| directly; 'msle'
            compares log(|FFT| + log_floor) to compress dynamic range so loud
            bins stop dominating; any LOSS_FUNCTIONS name applies that
            discrepancy to plain magnitudes.
        log_floor: floor inside the log for 'msle'.
        log_base: base of the logarithm for 'msle'. Changing it rescales the
            spectral term by 1/ln(base)**2 -- base 10 makes it ~5.3x smaller
            than natural log -- so it shifts where alpha balances the two
            terms rather than changing the shape of either.
        huber_delta: transition point for 'huber'/'smooth_l1', both terms.
    """

    def __init__(
        self,
        alpha: float = 0.5,
        density: bool = True,
        time_loss: str = "mse",
        spectral_loss: str = "mse",
        log_floor: float = 1e-6,
        log_base: float = 10.0,
        huber_delta: float = 1.0,
    ):
        super().__init__()
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        if time_loss not in LOSS_FUNCTIONS:
            raise ValueError(
                f"time_loss must be one of {LOSS_FUNCTIONS}, got {time_loss!r}"
            )
        if spectral_loss not in LOSS_FUNCTIONS + ("msle",):
            raise ValueError(
                f"spectral_loss must be 'msle' or one of {LOSS_FUNCTIONS}, "
                f"got {spectral_loss!r}"
            )
        if log_floor <= 0.0:
            raise ValueError(f"log_floor must be > 0, got {log_floor}")
        if log_base <= 1.0:
            raise ValueError(f"log_base must be > 1, got {log_base}")
        if huber_delta <= 0.0:
            raise ValueError(f"huber_delta must be > 0, got {huber_delta}")
        self.alpha = alpha  # mutable; updated each epoch by the task
        self.density = density
        self.time_loss = time_loss
        self.spectral_loss = spectral_loss
        self.log_floor = log_floor
        self.log_base = log_base
        self._log_base_scale = math.log(log_base)
        self.huber_delta = huber_delta

    def _row_term(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """Per-row MSE, optionally divided per row by the row's own
        error-of-predicting-zero, then averaged over the batch.

        pred, target: (..., L). Reduces over every dim but the batch, so a
        (B, C, L) or (B, C, F) input gives one number per (B, C) pair.
        """
        err = (pred - target).pow(2).mean(dim=tuple(range(1, target.ndim)))
        if self.density:
            scale = (
                target.pow(2)
                .mean(dim=tuple(range(1, target.ndim)))
                .clamp_min(1e-9)
            )
            err = err / scale
        return err.mean()

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """pred, target: (B, C, L). Returns scalar loss."""
        time_term = self._row_term(pred, target)

        pred_mag = torch.fft.rfft(pred, dim=-1).abs()
        target_mag = torch.fft.rfft(target, dim=-1).abs()

        if self.spectral_loss == "msle":
            # log_b(x) = ln(x) / ln(b)
            pred_mag = (
                torch.log(pred_mag + self.log_floor) / self._log_base_scale
            )
            target_mag = (
                torch.log(target_mag + self.log_floor) / self._log_base_scale
            )
        spectral_term = self._row_term(pred_mag, target_mag)

        mix = self.alpha * time_term + (1.0 - self.alpha) * spectral_term

        # expose raw components (pre-alpha) so the task can log them and
        # pick alpha from their relative scale
        self._stash(time=time_term, spectral=spectral_term)
        return mix


class ShapeGainLoss(TermStashMixin, nn.Module):
    """Scale-free shape term plus an explicit gain term.

    Written against a measured failure of the magnitude-spectrum mixture: the
    trained ``ScheduledMixtureLoss`` models put ~98% of their output power in
    the first 5% of the kernel, with correlation to the target consistent with
    zero, because ``|FFT|`` discards phase and so cannot tell a chirp from a
    window-edge impulse. Two independent quantities are therefore supervised:

    ``shape``  1 - rho, where rho is the normalized inner product between
        prediction and target over (channel, time). Phase sensitive, so an
        edge impulse cannot satisfy it, and scale invariant, so it cannot be
        reduced by shrinking the output.
    ``gain``   (log c*)**2, where c* is the least-squares rescaling of the
        prediction onto the target. Zero exactly at the right amplitude and
        indifferent to shape, so it does not fight the shape term.

    A small absolute time-domain anchor (``lambda_time``) keeps the output on
    an absolute scale. Rows whose target is essentially empty (background,
    ``||target|| < sig_thresh``) are excluded from shape and gain, which are
    undefined there, and are instead driven to zero by ``lambda_bkg``.

    Args:
        lambda_gain: weight on the gain term. Larger forces amplitude harder;
            some shrinkage is statistically correct at low SNR, so this is the
            knob for that trade-off.
        lambda_time: weight on the plain per-row MSE anchor.
        lambda_bkg: weight on the mean square of the prediction for noise-only
            rows. Only active when the batch contains such rows.
        sig_thresh: ||target|| below which a row counts as noise-only.
        learn_weights: if True, replace the fixed weights above with learned
            uncertainty weighting (Kendall et al.): each term k is scaled by
            exp(-s_k) and regularized by +s_k, with s_k a free parameter. The
            +s_k prevents the trivial s_k -> +inf solution. lambda_* then act
            as the initial relative scales.
        eps: numerical floor.
    """

    TERMS = ("shape", "gain", "time", "bkg")

    def __init__(
        self,
        lambda_gain: float = 0.3,
        lambda_time: float = 1.0,
        lambda_bkg: float = 1.0,
        sig_thresh: float = 0.5,
        learn_weights: bool = False,
        eps: float = 1e-8,
    ):
        super().__init__()
        for name, value in (
            ("lambda_gain", lambda_gain),
            ("lambda_time", lambda_time),
            ("lambda_bkg", lambda_bkg),
        ):
            if value < 0.0:
                raise ValueError(f"{name} must be >= 0, got {value}")
        if sig_thresh < 0.0:
            raise ValueError(f"sig_thresh must be >= 0, got {sig_thresh}")
        self.lambda_gain = lambda_gain
        self.lambda_time = lambda_time
        self.lambda_bkg = lambda_bkg
        self.sig_thresh = sig_thresh
        self.learn_weights = learn_weights
        self.eps = eps
        if learn_weights:
            # s_k = log(sigma_k**2); init so exp(-s_k) matches the fixed
            # weights, keeping the two modes comparable at step 0
            init = [
                -math.log(max(w, 1e-6))
                for w in (1.0, lambda_gain, lambda_time, lambda_bkg)
            ]
            self.log_var = nn.Parameter(torch.tensor(init))

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """pred, target: (B, C, L). Returns scalar loss."""
        p = pred.reshape(pred.shape[0], -1)
        t = target.reshape(target.shape[0], -1)

        pt = (p * t).sum(-1)
        pp = p.pow(2).sum(-1)
        tt = t.pow(2).sum(-1)

        sig = tt.sqrt() > self.sig_thresh
        zero = pred.new_zeros(())

        if sig.any():
            rho = pt[sig] / (pp[sig].sqrt() * tt[sig].sqrt() + self.eps)
            shape_term = (1.0 - rho).mean()
            # c* = <p,t>/<p,p>, the best rescale of pred onto target
            c_star = pt[sig] / pp[sig].clamp_min(self.eps)
            gain_term = torch.log(c_star.clamp_min(self.eps)).pow(2).mean()
        else:
            shape_term = zero
            gain_term = zero

        time_term = (p - t).pow(2).mean(-1).mean()
        if (~sig).any():
            bkg_term = p[~sig].pow(2).mean()
        else:
            bkg_term = zero

        terms = torch.stack([shape_term, gain_term, time_term, bkg_term])
        if self.learn_weights:
            total = (torch.exp(-self.log_var) * terms + self.log_var).sum()
        else:
            weights = terms.new_tensor(
                [1.0, self.lambda_gain, self.lambda_time, self.lambda_bkg]
            )
            total = (weights * terms).sum()

        # exposed for logging, matching the names the task looks for
        self._stash(shape=shape_term, gain=gain_term,
                    time=time_term, bkg=bkg_term)
        return total


class ApproximateRecoveryLoss(TermStashMixin, nn.Module):
    """Shape/gain objective weighted toward where the signal actually is.

    Exact recovery is not reachable in the SNR 4-8 regime: with the whole-
    window correlation at rho ~ 0.2, the measured gain sits at ~0.2 as well,
    which is the Wiener optimum c* = rho rather than a defect. Approximate
    recovery is the reachable goal, and it needs a different emphasis:

    ``energy_weight`` weights each sample's shape contribution by the local
        energy of the target, so the loss stops spending capacity on the long
        quiet stretch before the merger, where there is nothing to recover
        and the model can only add noise. Whole-window rho dilutes exactly
        this distinction.
    ``calibrate_gain`` divides the prediction by its own realised gain before
        the shape comparison. Because a correct minimum-variance estimator
        shrinks by a known factor, the scale can be restored after the fact;
        this makes the shape term indifferent to a systematic shrink instead
        of fighting it.

    For detection and parameter estimation a scale error is nearly free while
    a phase error is fatal, so shape carries the objective and gain is kept
    only as a weak anchor.

    Args:
        lambda_gain: weight on the (log gain)**2 term. Deliberately small.
        lambda_time: weight on the plain MSE anchor.
        lambda_bkg: weight on driving noise-only rows to zero.
        energy_weight: if True, weight the shape term by target local energy.
        smooth_ms: width in milliseconds of the smoothing applied to the
            target energy envelope before it is used as a weight.
        calibrate_gain: if True, remove the realised gain before comparing
            shape, so shrinkage does not penalise an otherwise good fit.
        sig_thresh: ||target|| below which a row counts as noise only.
        sample_rate: needed to convert smooth_ms into samples.
        eps: numerical floor.
    """

    def __init__(
        self,
        lambda_gain: float = 0.05,
        lambda_time: float = 0.1,
        lambda_bkg: float = 1.0,
        energy_weight: bool = True,
        smooth_ms: float = 50.0,
        calibrate_gain: bool = True,
        sig_thresh: float = 0.5,
        sample_rate: float = 2048.0,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.lambda_gain = lambda_gain
        self.lambda_time = lambda_time
        self.lambda_bkg = lambda_bkg
        self.energy_weight = energy_weight
        self.calibrate_gain = calibrate_gain
        self.sig_thresh = sig_thresh
        self.eps = eps
        self.kernel = max(1, int(smooth_ms * 1e-3 * sample_rate) | 1)

    def _envelope(self, target: torch.Tensor) -> torch.Tensor:
        """Smoothed per-sample energy of the target, normalised to mean one."""
        e = target.pow(2).sum(1, keepdim=True)
        w = F.avg_pool1d(e, self.kernel, stride=1, padding=self.kernel // 2,
                         count_include_pad=False)
        w = w / w.mean(dim=-1, keepdim=True).clamp_min(self.eps)
        return w

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        tt_row = target.reshape(target.shape[0], -1).pow(2).sum(-1).sqrt()
        sig = tt_row > self.sig_thresh
        zero = pred.new_zeros(())

        if sig.any():
            p, t = pred[sig], target[sig]
            pf = p.reshape(p.shape[0], -1)
            tf = t.reshape(t.shape[0], -1)
            pn = pf.norm(dim=-1).clamp_min(self.eps)
            tn = tf.norm(dim=-1).clamp_min(self.eps)
            gain = pn / tn
            gain_term = torch.log(gain.clamp_min(self.eps)).pow(2).mean()

            if self.calibrate_gain:
                # compare shape only: remove the systematic scale first
                p = p / gain.detach().clamp_min(self.eps).view(-1, 1, 1)

            if self.energy_weight:
                w = self._envelope(t)
                pw, tw = p * w.sqrt(), t * w.sqrt()
            else:
                pw, tw = p, t
            pwf = pw.reshape(pw.shape[0], -1)
            twf = tw.reshape(tw.shape[0], -1)
            rho = (pwf * twf).sum(-1) / (
                pwf.norm(dim=-1) * twf.norm(dim=-1)
            ).clamp_min(self.eps)
            shape_term = (1.0 - rho).mean()
            time_term = (p - t).pow(2).mean()
        else:
            shape_term = gain_term = time_term = zero

        if (~sig).any():
            bkg_term = pred[~sig].pow(2).mean()
        else:
            bkg_term = zero

        total = (
            shape_term
            + self.lambda_gain * gain_term
            + self.lambda_time * time_term
            + self.lambda_bkg * bkg_term
        )
        self._stash(shape=shape_term, gain=gain_term,
                    time=time_term, bkg=bkg_term)
        return total


class CorrelationDenoiseLoss(nn.Module):
    """Scale-invariant matched-filter denoiser loss.

    Plain MSE between the denoised and clean whitened strain collapses to
    "predict zero" for low-SNR (SNR 4-8) signals: their power is tiny next to
    the noise, so silence already yields a low MSE and the denoiser never
    learns to recover the buried chirp. Matched filtering, the optimal
    low-SNR detector, cares only about the template SHAPE, not its amplitude.

    This loss rewards template overlap: ``1 - NCC`` where NCC is the
    normalized cross-correlation (cosine similarity over time) between
    denoised and clean, per (sample, detector). Being scale-free it does not
    reward zero output. It is applied only to rows that actually carry signal
    (``||clean|| ~ optimal SNR > sig_thresh``); a small MSE term over all rows
    (``mse_weight``) both suppresses noise-only rows to zero and keeps the
    recovered amplitude calibrated so the regressor can read absolute scale.
    """

    def __init__(
        self,
        mse_weight: float = 0.1,
        sig_thresh: float = 0.5,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.mse_weight = mse_weight
        self.sig_thresh = sig_thresh
        self.eps = eps

    def forward(
        self, inputs: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        i = inputs - inputs.mean(dim=-1, keepdim=True)
        t = targets - targets.mean(dim=-1, keepdim=True)
        t_norm = t.norm(dim=-1)  # (B, C) ~ optimal matched-filter SNR
        i_norm = i.norm(dim=-1)
        ncc = (i * t).sum(dim=-1) / (i_norm * t_norm + self.eps)  # (B, C)
        sig = t_norm > self.sig_thresh
        if sig.any():
            corr = (1.0 - ncc[sig]).mean()
        else:
            corr = inputs.new_zeros(())
        if self.mse_weight:
            return corr + self.mse_weight * F.mse_loss(inputs, targets)
        return corr


class SNRWeightedMSELoss(nn.Module):
    """Per-row SNR-focal MSE.

    Plain MSE is dominated by the loud (high-SNR) rows, so the denoiser nails
    the easy signals and neglects the SNR 4-8 rows that are the actual
    unsolved regime. Here each row's reconstruction error is weighted by
    ``(ref/(snr+ref))**gamma``, shifting capacity onto low-SNR signals while
    keeping high-SNR rows nonzero. ``snr`` is read from the clean-target norm
    (``||whitened h|| ~ optimal SNR``); noise-only rows (norm ~ 0) get the max
    weight so noise is still driven to zero. Weights are renormalized to mean
    1 so the ``lambda_denoise`` scale stays comparable to a plain MSE.
    """

    def __init__(
        self, gamma: float = 1.0, ref_snr: float = 8.0, eps: float = 1e-8
    ):
        super().__init__()
        self.gamma = gamma
        self.ref_snr = ref_snr
        self.eps = eps

    def forward(
        self, inputs: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        row_mse = ((inputs - targets) ** 2).mean(dim=-1)  # (B, C)
        snr = targets.norm(dim=-1)  # (B, C)
        w = (self.ref_snr / (snr + self.ref_snr)) ** self.gamma
        w = w / w.mean().clamp(min=self.eps)
        return (w * row_mse).mean()


class MixtureMSESpectralLoss(nn.Module):
    """MSE (time domain) + spectral MSE (frequency domain) mixture.

    Spectral term compares the magnitude of the real FFT along the sequence
    axis. Aframe tensors are ``(B, C, L)``, so the FFT is taken over the last
    dim ``L``.

    Args:
        alpha: Weight for the time-domain MSE. Spectral term weighted by
            ``(1 - alpha)``. Defaults to ``0.5``.
    """

    def __init__(self, alpha: float = 0.5):
        super().__init__()
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        self.alpha = alpha
        self.mse = nn.MSELoss()

    def forward(
        self, inputs: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        mse_loss = self.mse(inputs, targets)
        inputs_mag = torch.abs(torch.fft.rfft(inputs, dim=-1))
        targets_mag = torch.abs(torch.fft.rfft(targets, dim=-1))
        spectral_loss = self.mse(inputs_mag, targets_mag)
        return self.alpha * mse_loss + (1.0 - self.alpha) * spectral_loss


def soft_pauc_loss(
    logits: torch.Tensor,
    y: torch.Tensor,
    fpr_frac: float = 0.05,
    beta: float = 10.0,
) -> torch.Tensor:
    """Differentiable partial-AUROC surrogate at low false-positive rate.

    Detection at low FAR is set by the loudest background. This penalizes
    signals that fail to outscore the top ``fpr_frac`` fraction of background
    logits in the batch (a soft ranking hinge). Returns 0 if the batch lacks
    positives or negatives.

    Args:
        logits: raw detection logits, shape ``(N, 1)`` or ``(N,)``.
        y: binary labels, same shape.
        fpr_frac: fraction of loudest negatives to rank against (low-FAR
            focus; smaller = stricter tail).
        beta: softplus sharpness; higher approaches a hard hinge.
    """
    logits = logits.reshape(-1)
    y = y.reshape(-1)
    pos = logits[y > 0.5]
    neg = logits[y <= 0.5]
    if pos.numel() == 0 or neg.numel() == 0:
        return logits.new_zeros(())
    k = max(1, int(neg.numel() * fpr_frac))
    top_neg = torch.topk(neg, k).values
    # pairwise: want pos > top_neg; penalize (top_neg - pos) > 0
    diff = top_neg.view(-1, 1) - pos.view(1, -1)  # (k, P)
    return torch.nn.functional.softplus(beta * diff).mean() / beta
