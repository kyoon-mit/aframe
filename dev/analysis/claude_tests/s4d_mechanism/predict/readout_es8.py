"""Early-stopping variant: the chosen MLP and the linear head on N training windows, stopped by within 1%
on data/es8 (10,000 windows from the validation background with SNR >= 8, the training SNR law; same
waveform rows as data/es, new noise, sky and SNR draws) instead of data/es (SNR >= 4). Everything else as
readout_big.py.

usage: readout_es8.py N TAG...
"""
import json
import sys

import numpy as np

import readout as R
import readout_big as B


def main(n, tags):
    for tag in tags:
        sel = json.load(open(R.R / f"select__{tag}.json"))
        xtr, mctr = B.load_train(tag, n)
        xes, mces, _, _ = R.features("es8", tag)
        out = {"tag": tag, "n_train": n, "early_stop_set": "es8", "configs": {}}
        for name in ("linear", sel["best_mlp"]):
            model, info = B.train(R.CONFIGS[name], xtr, mctr, xes, mces)
            res, ea, eb = R.evaluate(model, tag)
            out["configs"][name] = {"train": info, **res}
            np.save(R.R / f"est_testA__{tag}__{name}__n{n}__es8.npy", ea)
            np.save(R.R / f"est_testB__{tag}__{name}__n{n}__es8.npy", eb)
            print(f"{tag} {name} n={n} es8: es8 {info['es_within1']:.4f} (ep {info['best_epoch']}/{info['epochs_run']}, {info['seconds']} s) "
                  f"testA {res['testA']['within1']:.4f} bands " +
                  " ".join(f"{k}:{v['within1']:.3f}" for k, v in res["testAB_bands"].items()), flush=True)
        json.dump(out, open(R.R / f"es8__{tag}__n{n}.json", "w"), indent=1)


if __name__ == "__main__":
    main(int(sys.argv[1]), sys.argv[2:])
