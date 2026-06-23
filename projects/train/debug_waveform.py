"""Characterize the waveform-generator NaN.

(1) Reproduce the exact failing prior sample -> expect NaN waveform.
(2) Same sample but with mass_ratio nudged away from 1.0 -> expect finite.
(3) Monte-Carlo N draws from the *real* training prior, generate waveforms,
    count NaN waveforms and report the mass_ratio distribution of the NaN
    samples (to prove the NaNs cluster at q->1) and whether capping q removes
    them.

Uses the real RegressionCBCGenerator from the YAML. No committed code changed.
"""
import argparse

import torch
from lightning.pytorch.cli import LightningCLI


def build(config_path):
    import tempfile, yaml
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    slim = {k: cfg[k] for k in ("seed_everything", "model", "data") if k in cfg}
    tmp = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
    yaml.safe_dump(slim, tmp); tmp.flush()
    return LightningCLI(run=False, args=["--config", tmp.name], save_config_callback=None)


def finite_frac(hc, hp):
    ok = torch.isfinite(hc).all(dim=-1) & torch.isfinite(hp).all(dim=-1)
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--n", type=int, default=20000)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cli = build(args.config)
    dm = cli.datamodule
    gen = dm.waveform_sampler.to(dev)
    prior = gen.training_prior
    print(f"[wf] device={dev}", flush=True)

    # ----- (1)+(2) controlled mass_ratio sweep on the exact bad sample -----
    from train.conversion import precessing_to_lalsimulation_parameters
    bad = dict(chirp_mass=1.4080100059509277,
               a_1=0.07369086891412735, a_2=0.16629241406917572,
               tilt_1=0.7859686613082886, tilt_2=1.2205051183700562,
               phi_12=2.278597354888916, phi_jl=4.323770999908447,
               inclination=0.14806629717350006, phic=5.626690864562988,
               distance=427.12310791015625)

    def run_one(q):
        raw = {k: torch.tensor([v], device=dev, dtype=torch.float32) for k, v in bad.items()}
        raw["mass_ratio"] = torch.tensor([q], device=dev, dtype=torch.float32)
        conv = precessing_to_lalsimulation_parameters(dict(raw))
        hc, hp = gen(**conv)
        return bool(finite_frac(hc, hp).all()), float(conv["mass_1"][0]), float(conv["mass_2"][0])

    for q in [0.9996640682220459, 0.9999, 0.999, 0.995, 0.99, 0.95, 0.8, 0.5]:
        try:
            fin, m1, m2 = run_one(q)
            print(f"[wf] q={q:.7f} -> waveform_finite={fin}  (m1={m1:.5f} m2={m2:.5f} dm={m1-m2:.2e})", flush=True)
        except Exception as e:
            print(f"[wf] q={q}: EXC {type(e).__name__}: {e}", flush=True)

    # ----- (3) Monte-Carlo over the real prior -----
    print(f"\n[wf] Monte-Carlo {args.n} draws from real prior ...", flush=True)
    torch.manual_seed(0)
    bs = 2000
    n_bad = 0
    bad_q = []
    all_q = []
    for i in range(0, args.n, bs):
        s = prior(bs, device=dev)
        hc, hp = gen(**s)
        ok = finite_frac(hc, hp)
        q = s["mass_ratio"].detach().float().cpu()
        all_q.append(q)
        bad_mask = ~ok.cpu()
        n_bad += int(bad_mask.sum())
        if bad_mask.any():
            bad_q.append(q[bad_mask])
    all_q = torch.cat(all_q)
    print(f"[wf] total={args.n} nan_waveforms={n_bad} frac={n_bad/args.n:.3e}", flush=True)
    if bad_q:
        bq = torch.cat(bad_q)
        print(f"[wf] mass_ratio of NaN samples: min={bq.min():.5f} max={bq.max():.5f} "
              f"mean={bq.mean():.5f}  (all >0.99? {bool((bq>0.99).all())})", flush=True)
        for thr in [0.999, 0.998, 0.995, 0.99, 0.98]:
            print(f"      NaN with q>{thr}: {int((bq>thr).sum())}/{len(bq)}", flush=True)
    # how many samples would a q<=0.99 cap remove from the prior?
    print(f"[wf] prior draws with q>0.99: {int((all_q>0.99).sum())}/{args.n} "
          f"({float((all_q>0.99).float().mean())*100:.2f}%)", flush=True)
    print(f"[wf] prior draws with q>0.999: {int((all_q>0.999).sum())}/{args.n}", flush=True)


if __name__ == "__main__":
    main()
