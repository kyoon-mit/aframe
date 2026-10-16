"""Collect (i) own heads, (ii) fresh read-outs, (iii) the template bank; paired differences on the same
test windows; and a crude SNR-scaling prediction from the kernel matches. Writes results/summary.json
and prints markdown tables."""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
D, R = HERE / "data", HERE / "results"
NETS = ["d64", "band", "d512", "free"]
TAGS = NETS + ["bank", "d64_exact", "d512_exact"]
BANDS = [(4, 8), (8, 10), (10, 12), (12, 16), (16, 25), (25, 100.01)]
LOGGED_BEST = {"d64": 0.0966, "d512": 0.2365, "band": 0.1053, "free": 0.4194}          # kernels.md / calibration
LOGGED_PM10 = {"d64": 0.0763, "d512": 0.2143, "band": 0.0938, "free": 0.4000}          # calibration_table.json


def lab(which):
    z = np.load(D / which / "labels.npz")
    return z["chirp_mass"].astype(np.float64), z["snr"].astype(np.float64)


MCA, SNRA = lab("testA")
MCB, SNRB = lab("testB")
MC = np.concatenate([MCA, MCB])
SNR = np.concatenate([SNRA, SNRB])


def correct(tag, method):
    """Per-window within-1% on test A and on test A + B for a stored estimate."""
    ea = np.load(R / f"est_testA__{tag}__{method}.npy")
    eb = np.load(R / f"est_testB__{tag}__{method}.npy")
    return np.abs(ea - MCA) / MCA < 0.01, np.abs(np.concatenate([ea, eb]) - MC) / MC < 0.01


def summarize(wa, wab):
    out = {"testA_within1": float(wa.mean()), "testA_se": float(np.sqrt(wa.mean() * (1 - wa.mean()) / len(wa))), "bands": {}}
    for lo, hi in BANDS:
        m = (SNR >= lo) & (SNR < hi)
        p = float(wab[m].mean())
        out["bands"][f"{lo}-{hi:g}"] = {"within1": p, "se": float(np.sqrt(p * (1 - p) / m.sum())), "n": int(m.sum())}
    return out


def paired(w1, w2):
    """Mean of w2 - w1 and its standard error over the same windows."""
    d = w2.astype(float) - w1.astype(float)
    return float(d.mean()), float(d.std(ddof=1) / np.sqrt(len(d)))


def main():
    out = {"methods": {}, "paired": {}, "selection": {}}
    for tag in TAGS:
        sel = json.load(open(R / f"select__{tag}.json"))
        best = sel["best_mlp"]
        out["selection"][tag] = {k: {"es_within1": v["train"]["es_within1"], "testA_within1": v["testA"]["within1"],
                                     "best_epoch": v["train"]["best_epoch"], "params": v["train"]["params"]}
                                 for k, v in sel["configs"].items()}
        out["selection"][tag]["chosen"] = best
        m = {}
        if tag != "bank":
            m["own_head"] = correct(tag, "own")
        m["linear_refit"] = correct(tag, "linear")
        m["mlp"] = correct(tag, best)
        raw = R / f"est_testA__{tag}__{best}__raw.npy"
        if raw.exists():
            m["mlp_raw_features"] = correct(tag, f"{best}__raw")
        if (R / f"est_testA__{tag}__{best}__n128000.npy").exists():
            m["mlp_128k"] = correct(tag, f"{best}__n128000")
        for n in (256000, 512000):
            for name in ("linear", best):
                if (R / f"est_testA__{tag}__{name}__n{n}.npy").exists():
                    m[f"{'linear' if name == 'linear' else 'mlp'}_{n // 1000}k"] = correct(tag, f"{name}__n{n}")
                if (R / f"est_testA__{tag}__{name}__n{n}__es8.npy").exists():
                    m[f"{'linear' if name == 'linear' else 'mlp'}_{n // 1000}k_es8"] = correct(tag, f"{name}__n{n}__es8")
        out["methods"][tag] = {k: summarize(*v) for k, v in m.items()}
        out["methods"][tag]["mlp_config"] = best
        if "own_head" in m:
            for k in ("linear_refit", "mlp", "mlp_128k", "linear_256k", "mlp_256k", "linear_512k", "mlp_512k",
                      "linear_512k_es8", "mlp_512k_es8"):
                if k in m:
                    out["paired"][f"{tag}:{k}-own_head"] = {
                        "testA": paired(m["own_head"][0], m[k][0]),
                        **{f"{lo}-{hi:g}": paired(m["own_head"][1][(SNR >= lo) & (SNR < hi)], m[k][1][(SNR >= lo) & (SNR < hi)])
                           for lo, hi in BANDS}}
    # crude prediction from kernel matches: bank accuracy at SNR scaled by (best-channel match / bank's)
    mc_json = json.load(open(R / "match_compare.json"))
    bb = out["selection"]["bank"]["chosen"]
    bank_method = f"{bb}__n512000" if (R / f"est_testA__bank__{bb}__n512000.npy").exists() else bb
    _, wab_bank = correct("bank", bank_method)
    edges = np.array([4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 9, 10, 11, 12, 14, 16, 20, 25, 35, 50, 100.01])
    centers, acc = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        s = (SNR >= lo) & (SNR < hi)
        centers.append(float(np.median(SNR[s])))
        acc.append(float(wab_bank[s].mean()))
    centers, acc = np.array(centers), np.array(acc)
    out["bank_curve"] = {"method": bank_method, "snr_median": centers.tolist(), "within1": acc.tolist()}
    m_bank = mc_json["bank"]["best_channel_match_median"]
    pred = {}
    for tag in NETS:
        m_net = mc_json[tag]["inband_best_channel_match_median"]
        eff = SNR * m_net / m_bank
        p = np.interp(np.log(eff), np.log(centers), acc)          # clipped at the curve's ends
        below = eff < centers[0]
        pred[tag] = {"match_ratio": m_net / m_bank, "bands": {}}
        for lo, hi in BANDS:
            s = (SNR >= lo) & (SNR < hi)
            pred[tag]["bands"][f"{lo}-{hi:g}"] = {"predicted": float(p[s].mean()), "share_below_curve": float(below[s].mean())}
        sa = slice(0, len(SNRA))
        pred[tag]["testA_predicted"] = float(p[sa].mean())
        pred[tag]["testA_share_below_curve"] = float(below[sa].mean())
    out["match_prediction"] = pred
    json.dump(out, open(R / "summary.json", "w"), indent=1)

    # tables
    def row(name, s):
        b = s["bands"]
        return (f"| {name} | {100 * s['testA_within1']:.1f} ± {100 * s['testA_se']:.1f} | " +
                " | ".join(f"{100 * b[k]['within1']:.1f}" for k in ("4-8", "8-10", "10-12", "12-16", "16-25", "25-100.01")) + " |")

    print("| network / read-out | test A within 1% (SNR >= 4) | 4-8 | 8-10 | 10-12 | 12-16 | 16-25 | 25-100 |")
    print("|---|---|---|---|---|---|---|---|")
    for tag in TAGS:
        for k, s in out["methods"][tag].items():
            if k == "mlp_config":
                continue
            print(row(f"{tag} {k}", s))
    n = {k: v["n"] for k, v in out["methods"]["free"]["own_head"]["bands"].items()}
    print("windows per band (test A + B):", n)
    print(json.dumps(out["paired"], indent=0)[:4000])
    print(json.dumps(pred, indent=1))


if __name__ == "__main__":
    main()
