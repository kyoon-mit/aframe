"""Train the scorers for one model, under several objectives.

objectives
----------
classify : binary cross-entropy, mixed negatives (the original approach).
rank     : pairwise margin-ranking loss on hard negatives -- push every signal
           above every (loud) background, i.e. optimise the ordering directly.
snr      : regress the injection SNR (0 for background); a monotone-in-loudness
           target, so the score ranks like the boxcar response does.
detect   : differentiable "recall at fixed FAR" -- per batch take the soft
           alpha-quantile of background scores as a threshold tau and push
           signals above it / background below it.  This is the closest
           surrogate to the actual metric (signals recovered above the FAR
           threshold); CNN only.

Each objective trains a TinyCNN and (except ``detect``) an sklearn feature
model, saved under ``<out>/<model>/<objective>/``.
"""

import json
import logging
import pickle

import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import build_training_set
from .features import window_features
from .models import TinyCNN, score_windows


def _device(arg):
    if arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return arg


def _new(args, device):
    model = TinyCNN(channels=args.channels, kernel=args.kernel).to(device)
    return model, torch.optim.Adam(model.parameters(), lr=args.lr)


def _epochs(model, opt, step_fn, args):
    rng = np.random.default_rng(args.seed)
    for _ in range(args.epochs):
        model.train()
        step_fn(rng)


def train_cnn_classify(Xtr, ytr, args, device):
    model, opt = _new(args, device)
    pw = torch.tensor(
        [(ytr == 0).sum() / max((ytr == 1).sum(), 1)], device=device
    )
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pw)
    Xt, yt = torch.as_tensor(Xtr), torch.as_tensor(ytr)

    def step(rng):
        for idx in _batches(rng, len(Xt), args.batch):
            opt.zero_grad()
            loss = loss_fn(model(Xt[idx].to(device)), yt[idx].to(device))
            loss.backward()
            opt.step()

    _epochs(model, opt, step, args)
    return model


def train_cnn_snr(Xtr, target_tr, args, device):
    model, opt = _new(args, device)
    loss_fn = nn.MSELoss()
    Xt = torch.as_tensor(Xtr)
    tt = torch.as_tensor(np.log1p(target_tr))

    def step(rng):
        for idx in _batches(rng, len(Xt), args.batch):
            opt.zero_grad()
            loss = loss_fn(model(Xt[idx].to(device)), tt[idx].to(device))
            loss.backward()
            opt.step()

    _epochs(model, opt, step, args)
    return model


def train_cnn_rank(Xtr, ytr, args, device):
    model, opt = _new(args, device)
    pos = torch.as_tensor(Xtr[ytr == 1])
    neg = torch.as_tensor(Xtr[ytr == 0])

    def step(rng):
        op, on = rng.permutation(len(pos)), rng.permutation(len(neg))
        m = min(len(pos), len(neg))
        for i in range(0, m, args.batch):
            ip, ineg = op[i : i + args.batch], on[i : i + args.batch]
            k = min(len(ip), len(ineg))  # pair equal numbers of pos/neg
            ip, ineg = ip[:k], ineg[:k]
            sp = model(pos[ip].to(device))
            sn = model(neg[ineg].to(device))
            loss = torch.relu(args.margin - (sp - sn)).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()

    _epochs(model, opt, step, args)
    return model


def train_cnn_detect(Xtr, ytr, args, device):
    model, opt = _new(args, device)
    pos = torch.as_tensor(Xtr[ytr == 1])
    neg = torch.as_tensor(Xtr[ytr == 0])
    alpha = args.target_fpr

    def step(rng):
        op, on = rng.permutation(len(pos)), rng.permutation(len(neg))
        m = min(len(pos), len(neg))
        for i in range(0, m, args.batch):
            sp = model(pos[op[i : i + args.batch]].to(device))
            sn = model(neg[on[i : i + args.batch]].to(device))
            tau = torch.quantile(sn.detach(), 1 - alpha)
            temp = 0.25 * torch.cat([sp, sn]).detach().std() + 1e-3
            # missed signals above tau + false alarms above tau
            loss = (
                torch.sigmoid((tau - sp) / temp).mean()
                + torch.sigmoid((sn - tau) / temp).mean()
            )
            opt.zero_grad()
            loss.backward()
            opt.step()

    _epochs(model, opt, step, args)
    return model


def _batches(rng, n, batch):
    perm = rng.permutation(n)
    for i in range(0, n, batch):
        yield perm[i : i + batch]


CNN_TRAINERS = {
    "classify": lambda X, y, snr, a, d: train_cnn_classify(X, y, a, d),
    "rank": lambda X, y, snr, a, d: train_cnn_rank(X, y, a, d),
    "snr": lambda X, y, snr, a, d: train_cnn_snr(X, snr, a, d),
    "detect": lambda X, y, snr, a, d: train_cnn_detect(X, y, a, d),
}


def fit_feature_model(Ftr, ytr, snr_tr, Fva, yva, objective):
    """Returns (model, task, kind, val_auc) or None (detect has no model)."""
    if objective == "detect":
        return None
    if objective == "snr":
        pipe = make_pipeline(StandardScaler(), GradientBoostingRegressor())
        pipe.fit(Ftr, np.log1p(snr_tr))
        auc = roc_auc_score(yva, pipe.predict(Fva))
        return pipe, "regressor", "gbr", auc
    # classify / rank -> pick the better classifier by val AUC
    best = None
    for kind, clf in [
        ("logreg", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ("gbdt", GradientBoostingClassifier()),
    ]:
        pipe = make_pipeline(StandardScaler(), clf)
        pipe.fit(Ftr, ytr)
        auc = roc_auc_score(yva, pipe.decision_function(Fva))
        if best is None or auc > best[3]:
            best = (pipe, "classifier", kind, auc)
    return best


def train_model(run_dir, name, args, out_root):
    device = _device(args.device)
    summary = {"name": name, "objectives": {}}

    for objective in args.objectives:
        neg_mode = "hard" if objective == "rank" else "mixed"
        X, y, snr, stats = build_training_set(run_dir, args, neg_mode)

        rng = np.random.default_rng(args.seed)
        perm = rng.permutation(len(X))
        X, y, snr = X[perm], y[perm], snr[perm]
        nva = int(len(X) * 0.15)
        Xva, yva = X[:nva], y[:nva]
        Xtr, ytr, snr_tr = X[nva:], y[nva:], snr[nva:]

        obj_dir = out_root / name / objective
        obj_dir.mkdir(parents=True, exist_ok=True)

        logging.info(
            "[%s/%s] training CNN on %s (%d/%d)",
            name,
            objective,
            device,
            len(Xtr),
            len(Xva),
        )
        cnn = CNN_TRAINERS[objective](Xtr, ytr, snr_tr, args, device)
        cnn_auc = roc_auc_score(
            yva, score_windows(cnn, Xva, device, sigmoid=False)
        )
        torch.save(
            {
                "state_dict": cnn.state_dict(),
                "stats": stats,
                "L": X.shape[1],
                "channels": args.channels,
                "kernel": args.kernel,
                "objective": objective,
            },
            obj_dir / "cnn.pt",
        )

        feat = fit_feature_model(
            window_features(Xtr),
            ytr,
            snr_tr,
            window_features(Xva),
            yva,
            objective,
        )
        feat_auc = None
        if feat is not None:
            model, task, kind, feat_auc = feat
            with open(obj_dir / "features.pkl", "wb") as fh:
                pickle.dump(
                    {
                        "model": model,
                        "task": task,
                        "kind": kind,
                        "stats": stats,
                    },
                    fh,
                )

        summary["objectives"][objective] = {
            "cnn_val_auc": float(cnn_auc),
            "feature_val_auc": None if feat_auc is None else float(feat_auc),
            "feature_kind": None if feat is None else feat[2],
            "n_train": int(len(Xtr)),
        }
        logging.info(
            "[%s/%s] val AUC cnn=%.4f%s",
            name,
            objective,
            cnn_auc,
            ""
            if feat_auc is None
            else f"  features({feat[2]})={feat_auc:.4f}",
        )

    with open(out_root / name / "train_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    return summary
