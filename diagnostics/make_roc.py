"""ROC comparison per model: the time-shift + clustering pipeline vs the model
applied exactly at its trained alignment.

Two curves on one plot, for each model:

  - "slide + cluster (pipeline)" : from the real ``regression_infer`` run on
    time-shifted background. Signal = recovered-injection detection statistics
    (``foreground.hdf5``, missed injections count as undetected); background =
    clustered time-slide events (``background.hdf5``). This is the algorithm that
    does NOT know where the signal is -> look-elsewhere + recovery cost.
  - "oracle (fixed at trained e)" : from ``diag_test_step`` (``test_step.csv``).
    Signal = -sigma on the injection, background = -sigma on the identical
    signal-free noise, both at the trained kernel alignment.

Detection statistic is -sigma in both (higher = more confident). Both use the
powerlaw(4, 50) SNR prior, so the injection populations match.

Run:  python diagnostics/make_roc.py
"""

from pathlib import Path

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/n/holystore01/LABS/iaifi_lab/Lab/kyoon/aframe_linoss")
DIAG = ROOT / "runs/regression_sv/diag"
OUT = ROOT / "diagnostics"
SENT = -1e29  # missed-injection sentinel guard

MODELS = [
    dict(folder="3s_premerger", pipeline="premerger_59-60s_ft_sv",
         oracle="premerger_59-60s_ft", title="Pre-merger 1s (59-60s, id11)"),
    dict(folder="merger_4s", pipeline="merger_60-64s_ft_sv",
         oracle="merger_60-64s_ft", title="Merger 4s (60-64s)"),
    dict(folder="merger_1s", pipeline="merger_63-64s_ft_sv",
         oracle="merger_63-64s_ft", title="Merger 1s (63-64s)"),
]


def roc(signal, background):
    """Return (fpr, tpr) sweeping the detection-statistic threshold, plus AUC.

    TPR = fraction of signal at/above threshold (missed signal stays below any
    finite threshold and counts as undetected). FPR = fraction of background
    at/above threshold.
    """
    signal = np.sort(np.asarray(signal, dtype=np.float64))
    background = np.sort(np.asarray(background, dtype=np.float64))
    thr = np.unique(np.concatenate([signal, background]))
    tpr = (len(signal) - np.searchsorted(signal, thr, "left")) / len(signal)
    fpr = (len(background) - np.searchsorted(background, thr, "left")) / len(background)
    # sort by fpr ascending and pin the (0,0)/(1,1) endpoints for a clean curve/AUC
    o = np.argsort(fpr)
    fpr, tpr = fpr[o], tpr[o]
    fpr = np.concatenate([[0.0], fpr, [1.0]])
    tpr = np.concatenate([[0.0], tpr, [1.0]])
    return fpr, tpr, float(np.trapz(tpr, fpr))


def load_pipeline(run):
    with h5py.File(DIAG / run / "background.hdf5") as f:
        bg = f["parameters/detection_statistic"][:]
    with h5py.File(DIAG / run / "foreground.hdf5") as f:
        fg = f["parameters/detection_statistic"][:]
    return fg, bg  # signal, background (missed fg kept as -1e30 = undetected)


def load_oracle(run):
    import pandas as pd
    df = pd.read_csv(DIAG / run / "test_step.csv")
    sig = -df.fg_chirp_sigma.to_numpy()
    bg = -df.bg_chirp_sigma.to_numpy()
    return sig[np.isfinite(sig)], bg[np.isfinite(bg)]


def main():
    for m in MODELS:
        try:
            p_sig, p_bg = load_pipeline(m["pipeline"])
            o_sig, o_bg = load_oracle(m["oracle"])
        except FileNotFoundError as e:
            print("SKIP", m["folder"], "-", e)
            continue

        fpr_o, tpr_o, auc_o = roc(o_sig, o_bg)
        fpr_p, tpr_p, auc_p = roc(p_sig, p_bg)

        fig, ax = plt.subplots(figsize=(6.2, 6))
        ax.plot(fpr_o, tpr_o, color="tab:green", lw=2.5,
                label=f"oracle: fixed at trained e   (AUC={auc_o:.3f})")
        ax.plot(fpr_p, tpr_p, color="tab:orange", lw=2.5,
                label=f"slide + cluster pipeline   (AUC={auc_p:.3f})")
        ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="chance")
        ax.set_xscale("log")
        ax.set_xlim(1e-4, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate (injection efficiency)")
        ax.set_title(f"{m['title']}\nROC: pipeline vs model at trained alignment")
        ax.legend(loc="lower right", fontsize=9)
        ax.grid(alpha=0.3, which="both")
        fig.tight_layout()
        p = OUT / m["folder"] / "roc_pipeline_vs_oracle.png"
        fig.savefig(p, dpi=140, bbox_inches="tight")
        plt.close(fig)
        print(f"saved {p}  | oracle AUC={auc_o:.3f} (n_sig={len(o_sig)}, n_bg={len(o_bg)})"
              f"  pipeline AUC={auc_p:.3f} (n_sig={len(p_sig)}, n_bg={len(p_bg)})")


if __name__ == "__main__":
    main()
