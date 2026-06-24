import corner
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from scipy.stats import gaussian_kde
from sklearn.metrics import roc_auc_score, roc_curve

from .types import BatchedParams, BatchedTarget, CustomMetric, ImageLog, metric


@metric(type="Accumulated", stages=("val", "test"))
def efficiency_vs_snr(
    target: BatchedTarget, pred: BatchedTarget, params: BatchedParams, **kwargs
) -> ImageLog:
    """Compute efficiency vs SNR."""
    if params is None or "snr" not in params:
        fig = plt.figure(figsize=(6, 4))
        plt.text(0.5, 0.5, "SNR not available", ha="center", va="center")
        plt.tight_layout()
        image = plt.gcf()
        plt.close()
        return ImageLog(value=image, caption="Detection Efficiency vs SNR")

    target_fpr = 0.001
    bin_step = 1

    # Support both (N,) raw logits and (N, C) class log-probs
    if pred.ndim > 1:
        pred = pred[..., -1]
    pred = pred.flatten()
    target = target.flatten()
    # params only contains foreground entries (one per injected signal),
    # so index directly rather than masking with target == 1.
    s_snr_signal = np.array(params["snr"]).flatten()
    s_score_signal = pred[target == 1].flatten()

    fpr, _, thresholds = roc_curve(target, pred)
    # Threshold corresponding to target FPR
    idx = np.argmin(np.abs(fpr - target_fpr))
    threshold = thresholds[idx]

    # Bin range based on signal SNR
    bin_start = np.floor(s_snr_signal.min())
    bin_stop = np.ceil(s_snr_signal.max()) + bin_step  # include last bin
    bins = np.arange(bin_start, bin_stop, bin_step)
    bin_centers = 0.5 * (bins[1:] + bins[:-1])

    efficiencies = []
    errors = []
    counts = []
    for i in range(len(bins) - 1):
        in_bin = (s_snr_signal >= bins[i]) & (s_snr_signal < bins[i + 1])
        n_in_bin = int(np.sum(in_bin))
        counts.append(n_in_bin)
        if n_in_bin == 0:
            efficiencies.append(np.nan)
            errors.append(np.nan)
            continue
        detected = np.sum(s_score_signal[in_bin] > threshold)
        eff = detected / n_in_bin
        err = np.sqrt(eff * (1 - eff) / n_in_bin)

        efficiencies.append(eff)
        errors.append(err)

    efficiencies = np.array(efficiencies)
    errors = np.array(errors)
    counts = np.array(counts, dtype=float)

    fig, ax1 = plt.subplots(figsize=(10, 6))
    valid_mask = ~(np.isnan(efficiencies) | np.isnan(errors))
    ax1.errorbar(
        bin_centers[valid_mask],
        efficiencies[valid_mask],
        yerr=errors[valid_mask],
        color="steelblue",
        label="Efficiency",
    )
    ax1.set_xlabel("SNR")
    ax1.set_ylabel("Detection Efficiency")
    ax1.set_ylim(0, 1.1)
    ax1.set_xscale("log")
    x_ticks = [8, 9, 10, 11, 12, 15, 20, 25, 30, 40, 50, 70, 100]
    ax1.set_xticks(x_ticks)
    ax1.set_xticklabels([str(x) for x in x_ticks])
    ax1.grid(True, alpha=0.3)
    ax1.set_title(f"Detection Efficiency vs SNR for FPR = {target_fpr:.3f}")

    ax2 = ax1.twinx()
    ax2.bar(
        bin_centers,
        counts,
        width=bin_step * 0.5,
        color="darkorange",
        alpha=0.1,
        label="Event count",
    )
    ax2.set_ylabel("Events per bin", color="darkorange")
    ax2.tick_params(axis="y", labelcolor="darkorange")
    ax2.set_ylim(bottom=0)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower right")

    plt.tight_layout()
    image = plt.gcf()
    plt.close()

    return ImageLog(
        value=image,
        caption="Detection Efficiency vs SNR",
    )


@metric(type="Accumulated", stages=("val", "test"))
def roc_plot(target: BatchedTarget, pred: BatchedTarget, **kwargs) -> ImageLog:
    """Compute ROC curve."""
    num_false_samples = np.sum(target == 0)
    # set max fpr to 10 / num_false_samples, which is the smallest
    # non-zero FPR we can measure given the number of negative samples
    max_fpr = (
        min(0.001, 10 / num_false_samples) if num_false_samples > 0 else 0.001
    )
    plt.figure(figsize=(8, 6))
    fpr, tpr, _ = roc_curve(target, pred[:, -1])
    auc_full = roc_auc_score(target, pred[:, -1])
    auc_small = roc_auc_score(target, pred[:, -1], max_fpr=max_fpr)
    plt.plot(fpr, tpr)

    # draw a line at 0.5 TPR
    plt.axhline(y=0.5, color="red", linestyle="--", alpha=0.2)
    # draw a line at x=0.001 FPR
    plt.axvline(x=0.001, color="blue", linestyle="--", alpha=0.2)

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    auc_label = f"AUC={auc_full:.4f}" if not np.isnan(auc_full) else "AUC=nan"
    small_label = (
        f"(AUC@{max_fpr}={auc_small:.4f})" if not np.isnan(auc_small) else ""
    )
    plt.title(f"ROC Curves {auc_label} {small_label}")
    # plt.ylim(0.75, 1.)
    plt.xlim(1e-6, 1.5)
    plt.xscale("log")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    image = plt.gcf()
    plt.close()

    return ImageLog(
        value=image,
        caption="ROC Curve",
    )


@metric(type="Accumulated", stages=("val", "test"))
def classification_visualization(
    target: BatchedTarget, pred: BatchedTarget, params: BatchedParams, **kwargs
) -> ImageLog:
    """Create a single RGB image where each pixel represents a sample.
    Samples are sorted by SNR (low to high).
    Pixel colors encode both classification correctness and SNR:
    - Green hue: correctly classified
    - Red hue: incorrectly classified
    - Brightness: darker = lower SNR (more important), lighter = higher SNR
    Classification is done at FPR=0.001.
    """
    from PIL import Image

    # Classify at FPR=0.001 using ROC curve threshold
    assert pred.shape[-1] == 2
    target_fpr = 0.001
    target = target.flatten()
    pred_scores = pred[:, -1]  # probability of positive class
    fpr, _, thresholds = roc_curve(target, pred_scores)
    idx = np.argmin(np.abs(fpr - target_fpr))
    threshold = thresholds[idx]

    # Get predicted classes using threshold
    pred_classes = (pred_scores > threshold).astype(int)
    correct = pred_classes == target
    sample_indices = params["index"].flatten()
    sample_snr = params["snr"].flatten()
    # replace nan with 0
    sample_snr = np.where(np.isnan(sample_snr), 0.0, sample_snr)

    # Sort by SNR (ascending - low SNR first)
    sort_idx = np.argsort(sample_snr, stable=True)
    correct_sorted = correct[sort_idx]
    snr_sorted = sample_snr[sort_idx]

    # Determine the size of the number of samples
    num_samples = len(sample_indices)
    # Create a square-ish image
    img_width = int(np.ceil(np.sqrt(num_samples)))
    img_height = int(np.ceil(num_samples / img_width))

    # Normalize SNR for visualization (using log scale, clamped)
    # Lower SNR -> darker (0.2), Higher SNR -> lighter (1.0)
    snr_min = np.nanpercentile(snr_sorted, 1)
    snr_max = np.nanpercentile(snr_sorted, 99)
    snr_normalized = (snr_sorted - snr_min) / (snr_max - snr_min + 1e-8)
    snr_normalized = np.clip(snr_normalized, 0, 1)
    # Invert so low SNR is darker (more visible)
    brightness = 0.3 + 0.7 * (1 - snr_normalized)

    # Initialize image with gray (unprocessed samples)
    img = np.ones((img_height, img_width, 3)) * 0.5

    # Fill in the pixels for each sample
    for i in range(num_samples):
        row = i // img_width
        col = i % img_width

        if correct_sorted[i]:
            # Green for correct, modulated by SNR
            color = np.array([0.0, brightness[i], 0.0])
        else:
            # Red for incorrect, modulated by SNR
            color = np.array([brightness[i], 0.0, 0.0])

        img[row, col] = color

    # Convert to PIL Image
    img_array = np.array(img * 255, dtype=np.uint8)
    image = Image.fromarray(img_array, mode="RGB")

    accuracy = np.mean(correct)
    return ImageLog(
        value=image,
        caption=f"Classification Visualization (Accuracy: {accuracy:.2%})",
    )


@metric(type="Accumulated", stages=("val", "test"))
def accuracy_by_snr(
    target: BatchedTarget, pred: BatchedTarget, params: BatchedParams, **kwargs
) -> ImageLog:
    """
    Plot accuracy vs. SNR threshold.
    Accurcy is computed the following way:
    for each SNR threshold classify all samples with SNR >= threshold
    as positive class,
    and compute the accuracy.
    """
    is_signal = np.array(params["label"] == 1)
    predicted_snr = pred
    # normalize predicted_snr to [0, 1]
    norm_predicted_snr = (predicted_snr - predicted_snr.min()) / (
        predicted_snr.max() - predicted_snr.min() + 1e-8
    )
    # copy last dim twice to (N, 2)
    norm_predicted_snr = np.tile(norm_predicted_snr, (1, 2))
    plot = roc_plot(is_signal, norm_predicted_snr, **kwargs)

    return ImageLog(
        value=plot.value,
        caption="Accuracy by SNR",
    )


@metric(type="Accumulated", stages=("val", "test"))
def decision_threshold_vs_detected_events(
    target: BatchedTarget, pred: BatchedTarget, **kwargs
) -> ImageLog:
    """Plot the decision threshold vs. number of detected events."""
    # Assuming pred contains log-probabilities (based on auc_roc using exp).
    # We flatten to handle both (N, C) and (N, T, C) cases consistently.
    scores = np.exp(pred[..., -1]).flatten()
    targets = target.flatten()

    thresholds = np.linspace(0, 1, 101)
    num_detected_events = []

    # Convert to numpy for loop efficiency and plotting
    scores_np = np.array(scores)

    for t in thresholds:
        num_detected_events.append(np.sum(scores_np > t))

    plt.figure(figsize=(10, 5))
    plt.plot(thresholds, num_detected_events)

    # plot line at number of y_true==1
    n_true_positive = np.sum(targets == 1)
    plt.axhline(
        y=n_true_positive,
        color="red",
        linestyle="--",
        alpha=0.2,
        label="Number of events with injected signal",
    )

    plt.xlabel("Decision Threshold")
    plt.ylabel("Number of Events Classified as Signal")
    plt.title("Decision threshold vs. Number of Detected Events")
    plt.legend()

    # x ticks every 0.1
    plt.xticks(np.arange(0, 1.1, 0.1))
    plt.grid(True, alpha=0.3)
    plt.yscale("log")
    plt.tight_layout()

    image = plt.gcf()
    plt.close()

    return ImageLog(
        value=image,
        caption="Decision threshold vs. Number of Detected Events",
    )


@metric(type="Accumulated", stages=("val", "test"))
def distribution_comparison(
    target: BatchedTarget, pred: BatchedTarget, **kwargs
) -> ImageLog:
    """Plot density distributions of target vs. predicted values.
    Values are standardized (z-score normalization) for better comparison.
    """
    target = np.atleast_2d(target)
    pred = np.atleast_2d(pred)

    # Handle case where target is (N,) and np.atleast_2d makes it (1, N)
    if target.shape[0] == 1 and target.shape[1] > 1:
        target = target.T
    if pred.shape[0] == 1 and pred.shape[1] > 1:
        pred = pred.T

    num_features = target.shape[-1]

    # Define grid layout (rows, cols)
    cols = 2
    rows = (num_features + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows))
    # Flatten axes array for easy iteration if multiple, otherwise wrap in list
    axes = np.array(axes).flatten()

    for i in range(num_features):
        ax = axes[i]
        t_feat = target[:, i]
        p_feat = pred[:, i]

        # Remove any NaN or infinite values per feature
        valid_mask = np.isfinite(t_feat) & np.isfinite(p_feat)
        t_feat = t_feat[valid_mask]
        p_feat = p_feat[valid_mask]

        if len(t_feat) > 1:
            kde_target = gaussian_kde(t_feat)
            kde_pred = gaussian_kde(p_feat)

            x_min = min(t_feat.min(), p_feat.min())
            x_max = max(t_feat.max(), p_feat.max())
            x_range = np.linspace(x_min, x_max, 500)

            ax.plot(
                x_range,
                kde_target(x_range),
                label="Target",
                alpha=0.7,
                linewidth=2,
            )
            ax.plot(
                x_range,
                kde_pred(x_range),
                label="Predicted",
                alpha=0.7,
                linewidth=2,
            )
            ax.fill_between(x_range, kde_target(x_range), alpha=0.2)
            ax.fill_between(x_range, kde_pred(x_range), alpha=0.2)

        ax.set_xlabel("Value")
        ax.set_ylabel("Density")
        t_mean = t_feat.mean()
        p_mean = p_feat.mean()
        ax.set_title(
            f"Feature {i}\nTarget μ={t_mean:.2f} | Pred μ={p_mean:.2f}"
        )
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Hide unused subplots if num_features is odd
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()

    image = plt.gcf()
    # Note: plt.close() should be called after the caller is done
    # with the figure object
    plt.close()
    return ImageLog(
        value=image,
        caption=f"Distribution Comparison: {num_features} Features",
    )


@metric(type="Accumulated", stages=("val", "test"))
def regression_corner_plot(
    target: BatchedTarget, pred: BatchedTarget, **kwargs
) -> ImageLog:
    """Create corner plots showing the distribution of prediction errors.
    Shows differences (pred - target) for each output dimension.
    """
    # Calculate differences for each output dimension
    diffs = (
        pred - target
    )  # Shape: (N, D) where D is number of output dimensions

    # Remove any NaN or infinite values
    valid_mask = np.all(np.isfinite(diffs), axis=1) & np.all(
        target != 0, axis=-1
    )
    diffs_clean = diffs[valid_mask]

    if len(diffs_clean) == 0:
        plt.figure(figsize=(10, 6))
        plt.text(0.5, 0.5, "No valid data to plot", ha="center", va="center")
        plt.title("Regression Corner Plot")
        image = plt.gcf()
        plt.close()
        return ImageLog(
            value=image, caption="Regression Corner Plot (no valid data)"
        )

    # Create labels for each dimension
    n_dims = diffs_clean.shape[1] if len(diffs_clean.shape) > 1 else 1
    if n_dims == 1:
        diffs_clean = diffs_clean.reshape(-1, 1)

    labels = [f"$\\hat{{y}}_{{{i}}} - y_{{{i}}}$" for i in range(n_dims)]

    fig = plt.figure(figsize=(10, 10))

    # Create corner plot
    figure = corner.corner(
        np.array(diffs_clean),
        labels=labels,
        bins=100,
        quantiles=[0.16, 0.5, 0.84],
        show_titles=True,
        title_kwargs={"fontsize": 10},
        label_kwargs={"fontsize": 10},
        title_fmt=".3f",
        hist_kwargs={"density": True},
        fig=fig,
    )
    figure.suptitle("Prediction Errors Distribution", fontsize=12, y=0.98)

    # Save and close
    plt.tight_layout()
    image = plt.gcf()
    plt.close()

    return ImageLog(
        value=image,
        caption="Regression Corner Plot: Prediction Errors",
    )


@metric(type="Accumulated", stages=("val", "test"))
def true_vs_false_distribution(
    target: BatchedTarget,
    pred: BatchedTarget,
    **kwargs,
) -> ImageLog:
    """Plot the distribution of predicted values for true vs. false samples."""
    # For binary classification, pred has shape (N, 2)
    # extract positive class probability
    if pred.ndim > 1 and pred.shape[-1] == 2:
        pred = pred[..., -1]
    target = target.flatten()
    exp_true_pred = np.exp(np.array(pred)).flatten()

    true_values = exp_true_pred[target == 1]
    false_values = exp_true_pred[target == 0]

    plt.figure(figsize=(10, 6))
    eps = 1e-7
    lo = float(np.clip(exp_true_pred.min(), eps, 1 - eps))
    hi = float(np.clip(exp_true_pred.max(), eps, 1 - eps))
    logit_lo = np.log(lo / (1 - lo))
    logit_hi = np.log(hi / (1 - hi))
    logit_bins = np.linspace(logit_lo, logit_hi, 500)
    bins = 1 / (1 + np.exp(-logit_bins))  # sigmoid: back to probability space

    plt.hist(
        true_values,
        bins=bins,
        alpha=0.6,
        label="True Samples",
        color="g",
        histtype="stepfilled",
    )
    plt.hist(
        false_values,
        bins=bins,
        alpha=0.6,
        label="False Samples",
        color="r",
        histtype="stepfilled",
    )

    # Add vertical line at threshold corresponding to 0.001 FPR
    target_fpr = 0.001
    fpr, _, thresholds = roc_curve(target, exp_true_pred)
    idx = np.argmin(np.abs(fpr - target_fpr))
    threshold_at_fpr = thresholds[idx]
    plt.axvline(
        x=threshold_at_fpr,
        color="blue",
        linestyle="--",
        alpha=0.6,
        label=f"Threshold @ FPR={target_fpr}",
    )

    plt.xscale("logit")
    plt.yscale("log")
    plt.xlabel("Predicted Probability (logit scale)")
    plt.ylabel("Density")
    plt.title("Distribution of Predicted Values: True vs. False Samples")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    image = plt.gcf()
    plt.close()

    return ImageLog(
        value=image,
        caption="True vs. False Samples Distribution",
    )


@metric(type="Accumulated", stages=("val", "test"))
def gll_true_vs_false_distribution(
    pred: BatchedTarget, params: BatchedParams, **kwargs
) -> ImageLog:
    """Plot true_vs_false_distribution of uncertainty.

    Input: second half of the prediction tensor.
    """
    _, pred_y_std = np.split(pred, 2, axis=-1)
    labels = params["label"].flatten()
    # if pred_y_std last dim > 1, map over last dim and return a dict
    # {<i>: distribution_comparison(labels, pred_y_std[..., i])}
    if pred_y_std.shape[-1] > 1:
        images = {}
        for i in range(pred_y_std.shape[-1]):
            images[i] = true_vs_false_distribution(labels, pred_y_std[..., i])
        return images
    return true_vs_false_distribution(labels, pred_y_std)


@metric(type="Accumulated", stages=("val", "test"))
def gll_regression_corner_plot(
    target: BatchedTarget,
    pred: BatchedTarget,
    **kwargs,
) -> ImageLog:
    pred_y_mean, _ = np.split(pred, 2, axis=-1)
    return regression_corner_plot(target, pred_y_mean)


@metric(type="Accumulated", stages=("val", "test"))
def gll_distribution_comparison(
    target: BatchedTarget,
    pred: BatchedTarget,
    **kwargs,
) -> ImageLog:
    pred_y_mean, _ = np.split(pred, 2, axis=-1)
    return distribution_comparison(target, pred_y_mean)


@metric(type="Accumulated", stages=("val", "test"))
def bg_fg_response_comparison(
    target: BatchedTarget,
    pred: BatchedTarget,
    params: BatchedParams,
    **kwargs,
) -> ImageLog:
    """Bar chart of the bg-subtracted positive-class score per event.

    The validation step sets pred['label'] = log(sigmoid(exp(fg) - exp(bg))),
    so exp(pred[..., -1]) = sigmoid(fg_prob - bg_prob) in [0, 1].
    Values > 0.5 mean the foreground score exceeds the background (blue);
    values < 0.5 mean the reverse (red).
    Events are sorted by SNR (low→high).  A secondary y-axis (0–100) shows
    the SNR of each event with '_' markers.
    """
    scores = np.exp(
        np.array(pred[..., -1]).flatten()
    )  # sigmoid(fg-bg) in [0, 1]

    snr_raw = np.array(params["snr"]).flatten()
    snr_for_sort = np.where(np.isnan(snr_raw), 0.0, snr_raw)
    snr_display = np.clip(snr_for_sort, 0, 100)

    # Sort by SNR ascending
    sort_idx = np.argsort(snr_for_sort, stable=True)
    scores_sorted = scores[sort_idx]
    snr_sorted = snr_display[sort_idx]

    # Subsample uniformly across the SNR range for readability
    n_total = len(scores_sorted)
    max_events = 500
    if n_total > max_events:
        sub_idx = np.round(np.linspace(0, n_total - 1, max_events)).astype(int)
        scores_sorted = scores_sorted[sub_idx]
        snr_sorted = snr_sorted[sub_idx]

    n_events = len(scores_sorted)
    x = np.arange(n_events)

    fig, ax1 = plt.subplots(figsize=(14, 6))

    colors = np.where(scores_sorted >= 0.5, "steelblue", "firebrick")
    ax1.bar(x, scores_sorted, color=colors, alpha=0.7, width=1.0)
    ax1.set_ylim(0, 1)
    ax1.axhline(y=0.5, color="black", linewidth=0.8, linestyle="--")
    ax1.set_xlabel(
        f"Event index (sorted by SNR, showing {n_events}/{n_total})"
    )
    ax1.set_ylabel("sigmoid(FG − BG)  [positive-class probability]")
    ax1.grid(True, alpha=0.3, axis="y")

    # Secondary y-axis: SNR (0–100) with '_' markers
    ax2 = ax1.twinx()
    ax2.plot(
        x,
        snr_sorted,
        marker="_",
        linestyle="none",
        color="darkorange",
        markersize=6,
        markeredgewidth=1.5,
        alpha=0.85,
        label="SNR",
    )
    ax2.set_ylim(0, 100)
    ax2.set_ylabel("SNR", color="darkorange")
    ax2.tick_params(axis="y", labelcolor="darkorange")

    plt.title("FG vs BG Response Difference per Event (sorted by SNR)")
    plt.tight_layout()
    image = plt.gcf()
    plt.close()

    return ImageLog(
        value=image,
        caption="FG vs BG Response Difference per Event",
    )


@metric(type="Accumulated", stages=("val", "test"))
def fg_bg_score_distribution(
    target: BatchedTarget,
    pred: BatchedTarget,
    **kwargs,
) -> ImageLog:
    """Histogram of detection scores split by foreground / background.

    Matches the style of the offline analysis plot: both distributions
    are drawn as step histograms on shared bins (50 bins spanning the
    full score range), background dashed and foreground solid, with a
    log y-axis so the tail separation is easy to read.
    """
    target = target.flatten()
    # pred may be (N,) raw logits or (N, 2) log-probs; always use the
    # final column as the detection score.
    if pred.ndim > 1 and pred.shape[-1] >= 2:
        scores = pred[:, -1].flatten()
    else:
        scores = pred.flatten()

    fg_scores = scores[target == 1]
    bg_scores = scores[target == 0]

    combined_bins = np.linspace(scores.min(), scores.max(), 50)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(
        bg_scores,
        bins=combined_bins,
        histtype="step",
        color="royalblue",
        linestyle="--",
        label="background",
    )
    ax.hist(
        fg_scores,
        bins=combined_bins,
        histtype="step",
        color="royalblue",
        label="foreground",
    )
    ax.set_xlabel("Score")
    ax.set_ylabel("Count")
    ax.set_title("Detection Scores: Foreground vs Background Distributions")
    ax.set_yscale("log")
    ax.legend()
    plt.tight_layout()

    image = plt.gcf()
    plt.close()

    return ImageLog(
        value=image,
        caption="FG vs BG detection score distributions",
    )


@metric(type="Accumulated", stages=("val", "test"))
def snr_vs_score_scatter(
    target: BatchedTarget, pred: BatchedTarget, params: BatchedParams, **kwargs
) -> ImageLog:
    """Scatter plot of foreground detection score vs. network SNR.

    A vertical dashed line marks the maximum background score, indicating
    where the background distribution ends relative to the foreground.
    """
    if params is None or "snr" not in params:
        plt.figure(figsize=(10, 6))
        plt.text(0.5, 0.5, "SNR not available", ha="center", va="center")
        plt.tight_layout()
        image = plt.gcf()
        plt.close()
        return ImageLog(
            value=image, caption="Detection Statistic vs. SNR (no SNR data)"
        )

    target = target.flatten()
    if pred.ndim > 1 and pred.shape[-1] >= 2:
        scores = pred[:, -1].flatten()
    else:
        scores = pred.flatten()

    fg_d = scores[target == 1]
    bkg_d = scores[target == 0]
    snr = np.array(params["snr"]).flatten()

    plt.figure(figsize=(10, 6))
    plt.scatter(fg_d, snr, alpha=0.1, s=3, color="royalblue")

    max_bkg = bkg_d.max()
    plt.axvline(
        x=max_bkg,
        color="red",
        linestyle="--",
        label=f"Max Background Score: {max_bkg:.2f}",
        linewidth=0.5,
    )

    plt.xlabel("Detection Statistic (Score)")
    plt.ylabel("Network SNR")
    plt.yscale("log")
    plt.title("Detection Statistic vs. SNR for Foreground Events")
    plt.grid(True, which="both", ls="--", lw=0.5, alpha=0.5)
    plt.legend()
    plt.tight_layout()

    image = plt.gcf()
    plt.close()

    return ImageLog(
        value=image,
        caption="Detection Statistic vs. SNR scatter",
    )


@metric(type="Accumulated", stages=("val", "test"))
def recoveries_vs_snr(
    target: BatchedTarget, pred: BatchedTarget, params: BatchedParams, **kwargs
) -> ImageLog:
    """Plot number of recovered injections per SNR bin at FPR=0.001."""
    if params is None or "snr" not in params:
        fig = plt.figure(figsize=(6, 4))
        plt.text(0.5, 0.5, "SNR not available", ha="center", va="center")
        plt.tight_layout()
        image = plt.gcf()
        plt.close()
        return ImageLog(value=image, caption="Recoveries vs SNR")

    target_fpr = 0.001
    bin_step = 1

    if pred.ndim > 1:
        pred = pred[..., -1]
    pred = pred.flatten()
    target = target.flatten()

    s_snr_signal = np.array(params["snr"]).flatten()
    s_score_signal = pred[target == 1].flatten()

    fpr, _, thresholds = roc_curve(target, pred)
    idx = np.argmin(np.abs(fpr - target_fpr))
    threshold = thresholds[idx]

    bin_start = np.floor(s_snr_signal.min())
    bin_stop = np.ceil(s_snr_signal.max()) + bin_step
    bins = np.arange(bin_start, bin_stop, bin_step)

    total_counts = []
    recovered_counts = []
    for i in range(len(bins) - 1):
        in_bin = (s_snr_signal >= bins[i]) & (s_snr_signal < bins[i + 1])
        total_counts.append(int(np.sum(in_bin)))
        recovered_counts.append(
            int(np.sum(s_score_signal[in_bin] > threshold))
        )

    total_counts = np.array(total_counts, dtype=float)
    recovered_counts = np.array(recovered_counts, dtype=float)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.stairs(
        total_counts,
        bins,
        color="steelblue",
        alpha=0.4,
        fill=True,
        label="Total injections",
    )
    ax.stairs(
        recovered_counts,
        bins,
        color="steelblue",
        alpha=0.9,
        fill=True,
        label="Recovered",
    )
    ax.set_xlabel("SNR")
    ax.set_ylabel("Count")
    ax.set_yscale("log")
    ax.set_xscale("log")
    x_ticks = [8, 9, 10, 11, 12, 15, 20, 25, 30, 40, 50, 70, 100]
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([str(x) for x in x_ticks])
    ax.set_title(f"Recoveries vs SNR at FPR = {target_fpr:.3f}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    image = plt.gcf()
    plt.close()

    return ImageLog(value=image, caption="Recoveries vs SNR")


class RecoveriesVsSnrCallback(CustomMetric):
    """Lightning callback that logs recovery counts per SNR bin to WandB.

    Instantiated with no arguments so it can be referenced directly in
    a jsonargparse YAML config::

        - class_path: train.metric.plotting.RecoveriesVsSnrCallback
    """

    def __init__(self) -> None:
        super().__init__(
            metric=recoveries_vs_snr,
            metric_name="recoveries_vs_snr",
            type="Accumulated",
            stages=("val", "test"),
        )


class FgBgScoreDistributionCallback(CustomMetric):
    """Lightning callback that logs the FG/BG score histogram to WandB.

    Instantiated with no arguments so it can be referenced directly in
    a jsonargparse YAML config::

        - class_path: train.metric.plotting.FgBgScoreDistributionCallback
    """

    def __init__(self) -> None:
        super().__init__(
            metric=fg_bg_score_distribution,
            metric_name="fg_bg_score_distribution",
            type="Accumulated",
            stages=("val", "test"),
        )


class EfficiencyVsSnrCallback(CustomMetric):
    """Lightning callback that logs the detection efficiency vs SNR to WandB.

    Requires injection parameters (especially SNR) to be threaded through
    the validation pipeline via ``val_params_tensor``.

    Instantiated with no arguments so it can be referenced directly in
    a jsonargparse YAML config::

        - class_path: train.metric.plotting.EfficiencyVsSnrCallback
    """

    def __init__(self) -> None:
        super().__init__(
            metric=efficiency_vs_snr,
            metric_name="efficiency_vs_snr",
            type="Accumulated",
            stages=("val", "test"),
        )


class SnrVsScoreScatterCallback(CustomMetric):
    """Lightning callback that logs a scatter plot of SNR vs detection score.

    Instantiated with no arguments so it can be referenced directly in
    a jsonargparse YAML config::

        - class_path: train.metric.plotting.SnrVsScoreScatterCallback
    """

    def __init__(self) -> None:
        super().__init__(
            metric=snr_vs_score_scatter,
            metric_name="snr_vs_score_scatter",
            type="Accumulated",
            stages=("val", "test"),
        )


@metric(type="Accumulated", stages=("val", "test"))
def value_distribution(
    target: BatchedTarget, pred: BatchedTarget, params: BatchedParams, **kwargs
) -> ImageLog:
    """Plot target and predicted value distributions split by signal
    (label=1) vs background (label=0)."""
    target = np.asarray(target)
    pred = np.asarray(pred)
    if target.ndim == 1:
        target = target[:, None]
    if pred.ndim == 1:
        pred = pred[:, None]

    snr = np.asarray(params["snr"]).flatten()
    is_signal = np.isfinite(snr)
    is_bg = ~is_signal

    n_components = pred.shape[-1]

    plt.figure(figsize=(10, 6))

    plotted = 0
    for i in range(n_components):
        t = target[:, i]
        p = pred[:, i]
        tgt = t[np.isfinite(t)]
        sig = p[is_signal & np.isfinite(p)]
        bg = p[is_bg & np.isfinite(p)]
        if tgt.size == 0 and sig.size == 0 and bg.size == 0:
            continue
        plotted += 1

        combined = np.concatenate([x for x in [tgt, sig, bg] if x.size > 0])
        lo = float(np.min(combined))
        hi = float(np.max(combined))
        if lo == hi:
            hi = lo + 1e-6
        bins = np.linspace(lo, hi, 100)

        suffix = f" (comp {i})" if n_components > 1 else ""
        if tgt.size:
            plt.hist(
                tgt,
                bins=bins,
                histtype="step",
                linewidth=1.5,
                label=f"Target{suffix} (μ={tgt.mean():.3g})",
            )
        if sig.size:
            plt.hist(
                sig,
                bins=bins,
                histtype="step",
                linewidth=1.5,
                label=f"Pred Signal{suffix} (μ={sig.mean():.3g})",
            )
        if bg.size:
            plt.hist(
                bg,
                bins=bins,
                histtype="step",
                linewidth=1.5,
                label=f"Pred Background{suffix} (μ={bg.mean():.3g})",
            )

    if plotted == 0:
        plt.text(
            0.5,
            0.5,
            "No valid samples",
            ha="center",
            va="center",
            transform=plt.gca().transAxes,
        )
    else:
        plt.legend()

    plt.xlabel("Value")
    plt.ylabel("Count")
    plt.title("Value Distribution: Target, Pred Signal, Pred Background")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    image = plt.gcf()
    plt.close()

    return ImageLog(
        value=image,
        caption="Value Distribution: Target, Pred Signal, Pred Background",
    )


class ValueDistributionCallback(CustomMetric):
    """Plot target and predicted value distributions split by label.

    Instantiated with no arguments::

        - class_path: train.metric.plotting.ValueDistributionCallback
    """

    def __init__(self) -> None:
        super().__init__(
            metric=value_distribution,
            metric_name="value_distribution",
            type="Accumulated",
            stages=("val", "test"),
        )


def chirp_mass_error_vs_snr(
    target: BatchedTarget,
    pred: BatchedTarget,
    params: BatchedParams,
    all_outputs: dict | None = None,
    **kwargs,
) -> ImageLog:
    """Scatter of relative chirp-mass error vs SNR, coloured by relative
    uncertainty.

    Left y-axis: (pred - true) / true per sample (scatter, alpha=0.05).
    Right y-axis: fraction of samples within ±5 % error per log-spaced SNR
    bin.
    Colour encodes pred_std / pred (relative uncertainty); colourbar uses
    5–95th percentile range to avoid outlier saturation.
    """
    snr = np.asarray(params["snr"]).flatten()
    cm_pred = np.asarray(pred).flatten()
    cm_true = np.asarray(target).flatten()

    valid = (
        np.isfinite(snr)
        & np.isfinite(cm_pred)
        & np.isfinite(cm_true)
        & (cm_true != 0)
    )
    snr = snr[valid]
    cm_pred = cm_pred[valid]
    cm_true = cm_true[valid]

    rel_error = (cm_pred - cm_true) / cm_true

    has_std = (
        all_outputs is not None
        and "chirp_mass_std" in all_outputs
        and all_outputs["chirp_mass_std"] is not None
    )
    if has_std:
        cm_std = np.asarray(all_outputs["chirp_mass_std"]).flatten()[valid]
        rel_uncertainty = cm_std / np.maximum(np.abs(cm_pred), 1e-6)
    else:
        rel_uncertainty = np.zeros_like(cm_pred)

    snr_bins = np.logspace(np.log10(3.5), np.log10(100), 100)
    bin_centers = (snr_bins[:-1] + snr_bins[1:]) / 2
    bin_indices = np.digitize(snr, snr_bins)

    within_5pct = []
    for i in range(1, len(snr_bins)):
        mask = bin_indices == i
        if np.sum(mask) > 0:
            within_5pct.append(float(np.mean(np.abs(rel_error[mask]) <= 0.05)))
        else:
            within_5pct.append(np.nan)

    fig, ax1 = plt.subplots(figsize=(20, 8))

    if has_std:
        vmin, vmax = np.percentile(rel_uncertainty, [5, 95])
        scatter = ax1.scatter(  # noqa: F841
            snr,
            rel_error,
            alpha=0.05,
            zorder=5,
            c=rel_uncertainty,
            cmap="hot",
            vmin=vmin,
            vmax=vmax,
        )
        sm = ScalarMappable(cmap="hot", norm=Normalize(vmin=vmin, vmax=vmax))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax1, pad=0.1)
        cbar.set_label("Relative Chirp Mass Uncertainty")
    else:
        ax1.scatter(snr, rel_error, alpha=0.05, zorder=5, label="Samples")

    log_ticks = [4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 30, 50, 70, 100]
    ax1.set_xscale("log")
    ax1.set_xticks(log_ticks)
    ax1.set_xticklabels([str(t) for t in log_ticks])
    ax1.set_xlim(3.5, 100)
    ax1.set_ylim(-1.1, 1.1)
    ax1.set_yticks(np.linspace(-1, 1, 11))
    ax1.axhline(
        0.05,
        color="r",
        linestyle="--",
        zorder=0,
        linewidth=1,
        label="±5% error bound",
    )
    ax1.axhline(-0.05, color="r", linestyle="--", zorder=0, linewidth=1)
    ax1.set_ylabel("Chirp Mass Relative Prediction Error")
    ax1.set_xlabel("SNR (Log Scale)")
    ax1.grid(True, which="both", linestyle="--", linewidth=0.5, zorder=0)

    ax2 = ax1.twinx()
    ax2.plot(
        bin_centers,
        within_5pct,
        color="green",
        linewidth=1,
        alpha=0.8,
        label="Ratio within 5% error",
        zorder=1,
    )
    ax2.set_ylabel("Ratio of Samples Within 5% Error", color="green")
    ax2.tick_params(axis="y", labelcolor="green")
    ax2.set_yticks(np.linspace(0, 1, 11))
    ax2.set_ylim(-0.05, 1.05)
    ax2.grid(False)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower right")

    plt.title(f"Chirp Mass Relative Prediction Error vs SNR (N={valid.sum()})")
    plt.tight_layout()
    image = plt.gcf()
    plt.close()

    return ImageLog(
        value=image, caption="Chirp Mass Relative Prediction Error vs SNR"
    )


def true_vs_predicted_scatter(
    target: BatchedTarget,
    pred: BatchedTarget,
    params: BatchedParams | None = None,
    **kwargs,
) -> ImageLog:
    """Scatter of true vs predicted value, one panel per output dimension.

    Points are drawn with a low alpha so density is visible, and coloured
    by SNR when ``params['snr']`` is available (colourbar clipped to the
    5-95th percentile to avoid outlier saturation). A dashed ``y = x`` line
    marks perfect prediction.
    """
    target = np.asarray(target)
    pred = np.asarray(pred)
    if target.ndim == 1:
        target = target[:, None]
    if pred.ndim == 1:
        pred = pred[:, None]

    n_dims = pred.shape[-1]

    snr = None
    if params is not None and "snr" in params and params["snr"] is not None:
        snr = np.asarray(params["snr"]).flatten()

    cols = min(n_dims, 3)
    rows = (n_dims + cols - 1) // cols
    fig, axes = plt.subplots(
        rows, cols, figsize=(6 * cols, 5 * rows), squeeze=False
    )
    axes = axes.flatten()

    scatter = None
    for i in range(n_dims):
        ax = axes[i]
        t = target[:, i].flatten()
        p = pred[:, i].flatten()

        valid = np.isfinite(t) & np.isfinite(p)
        c = None
        if snr is not None and snr.shape[0] == valid.shape[0]:
            valid &= np.isfinite(snr)
            c = snr[valid]
        t = t[valid]
        p = p[valid]

        if t.size == 0:
            ax.text(0.5, 0.5, "No valid data", ha="center", va="center")
            continue

        if c is not None and c.size > 0:
            vmin, vmax = np.percentile(c, [5, 95])
            scatter = ax.scatter(
                t,
                p,
                c=c,
                cmap="viridis",
                vmin=vmin,
                vmax=vmax,
                alpha=0.15,
                s=6,
            )
        else:
            ax.scatter(t, p, alpha=0.15, s=6, color="steelblue")

        lo = float(min(t.min(), p.min()))
        hi = float(max(t.max(), p.max()))
        ax.plot([lo, hi], [lo, hi], "r--", linewidth=1, label="y = x")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal", adjustable="box")
        suffix = f" (out {i})" if n_dims > 1 else ""
        ax.set_xlabel(f"True{suffix}")
        ax.set_ylabel(f"Predicted{suffix}")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left")

    for j in range(n_dims, len(axes)):
        axes[j].axis("off")

    if scatter is not None:
        cbar = fig.colorbar(scatter, ax=axes[:n_dims].tolist(), pad=0.02)
        cbar.set_label("SNR")
        cbar.solids.set(alpha=1.0)

    fig.suptitle("True vs Predicted")
    image = plt.gcf()
    plt.close()

    return ImageLog(value=image, caption="True vs Predicted (coloured by SNR)")


class TrueVsPredictedScatterCallback(CustomMetric):
    """Plot true vs predicted values (coloured by SNR) at validation end.

    Expects the model's validation_step to return a dict with:
    - 'targets': true values
    - 'outputs': predicted values (physical means)
    - 'params': dict containing 'snr' (optional)

    Instantiated with no arguments::

        - class_path: train.metric.plotting.TrueVsPredictedScatterCallback
    """

    def __init__(self) -> None:
        super().__init__(
            metric=true_vs_predicted_scatter,
            metric_name="true_vs_predicted_scatter",
            type="Accumulated",
            stages=("val", "test"),
        )

    def log_metric(self, trainer, pl_module, outputs, stage):
        if not isinstance(outputs, dict):
            return
        if "targets" not in outputs or "outputs" not in outputs:
            return

        try:
            image_log = true_vs_predicted_scatter(
                target=outputs["targets"],
                pred=outputs["outputs"],
                params=outputs.get("params") or {},
            )
            image_log.log(
                trainer,
                pl_module,
                f"{stage}/{self.metric_name}",
                prog_bar=False,
                batch_size=len(outputs["targets"]),
            )
        except Exception as e:
            print(f"Error logging metric {self.metric_name}: {e}")


class ChirpMassErrorVsSnrCallback(CustomMetric):
    """Plot chirp-mass relative error vs SNR at validation epoch end.

    Expects the model's validation_step to return a dict with:
    - 'targets': true chirp masses (physical)
    - 'outputs': predicted chirp masses (physical means)
    - 'params': dict containing 'snr'
    - 'all_outputs': dict containing 'chirp_mass_std' (optional)

    Instantiated with no arguments::

        - class_path: train.metric.plotting.ChirpMassErrorVsSnrCallback
    """

    def __init__(self) -> None:
        super().__init__(
            metric=chirp_mass_error_vs_snr,
            metric_name="chirp_mass_error_vs_snr",
            type="Accumulated",
            stages=("val", "test"),
        )

    def log_metric(self, trainer, pl_module, outputs, stage):
        if not isinstance(outputs, dict):
            return
        if "targets" not in outputs or "outputs" not in outputs:
            return

        try:
            image_log = chirp_mass_error_vs_snr(
                target=outputs["targets"],
                pred=outputs["outputs"],
                params=outputs.get("params") or {},
                all_outputs=outputs.get("all_outputs"),
            )
            image_log.log(
                trainer,
                pl_module,
                f"{stage}/{self.metric_name}",
                prog_bar=False,
                batch_size=len(outputs["targets"]),
            )
        except Exception as e:
            print(f"Error logging metric {self.metric_name}: {e}")
