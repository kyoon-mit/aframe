"""Run all chirplet fits with 8 single-thread worker processes (resumable: finished fits are skipped)."""

import argparse
import multiprocessing as mp
import os
import time

os.environ["CHIRPLET_THREADS"] = "1"

ap = argparse.ArgumentParser()
ap.add_argument("--phase", type=int, default=1)
ap.add_argument("--workers", type=int, default=8)
ap.add_argument("--outdir", type=str, default="runs")
args = ap.parse_args()

MCS = [1.0, 1.2, 1.5, 2.0]


def jobs(phase):
    out = []
    if phase == 1:
        Ms, walls = [1, 2, 4, 8, 16, 32], {}
        track = ["track_n", "track_e", "track_n_j1", "track_n_j2", "track_n_j3"]
        rand = [f"random_{j}" for j in range(8)]
        dom = ["track_n_dom", "track_e_dom"]
    else:
        Ms, walls = [64, 128], {64: 600.0, 128: 900.0}
        track = ["track_n", "track_e"]
        rand = ["random_0", "random_1"]
        dom = ["track_n_dom"]
    for M in Ms:
        for Mc in MCS:
            starts = track + rand + (dom if Mc == 1.0 else [])
            for s in starts:
                out.append({"Mc": Mc, "M": M, "start": s, "wall": walls.get(M, 600.0)})
    # largest M first for load balance within a phase
    out.sort(key=lambda j: -j["M"])
    return out


def work(job):
    import fit as F
    from pathlib import Path
    od = Path(__file__).resolve().parent / args.outdir
    try:
        r = F.run_job(job, od)
        return (job, r["match_final"], None)
    except Exception as e:  # noqa: BLE001
        return (job, None, repr(e))


if __name__ == "__main__":
    from pathlib import Path
    (Path(__file__).resolve().parent / args.outdir).mkdir(exist_ok=True)
    js = jobs(args.phase)
    print(f"phase {args.phase}: {len(js)} jobs, {args.workers} workers", flush=True)
    t0 = time.time()
    ctx = mp.get_context("spawn")
    with ctx.Pool(args.workers, maxtasksperchild=4) as pool:
        for i, (job, m, err) in enumerate(pool.imap_unordered(work, js)):
            if err:
                print(f"ERROR {job}: {err}", flush=True)
            if (i + 1) % 10 == 0:
                print(f"[{i + 1}/{len(js)}] {time.time() - t0:.0f}s", flush=True)
    print(f"phase {args.phase} done {time.time() - t0:.0f}s", flush=True)
