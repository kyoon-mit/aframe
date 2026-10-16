"""Recheck PREDICTION: own-head accuracy from stored estimates, own head recomputed from stored peak energies with the
checkpoint's linear layer (own code), waveform-row disjointness, SNR shares, fresh read-out vs own head."""
import json, glob, math, sys
import numpy as np, torch, h5py
P = "/tmp/claude-1024/-home-kyoon-SSM-BNS/82effc75-bb98-44ad-af20-e41e39a7258d/scratchpad/s4dmath/predict"
H = "/tmp/claude-1024/-home-kyoon-SSM-BNS/82effc75-bb98-44ad-af20-e41e39a7258d/scratchpad/s4dmath/harness"
out = {}
labA = np.load(f"{P}/data/testA/labels.npz")
labH = np.load(f"{H}/data/noisy_labels.npz")
out["testA_labels_equal_harness"] = bool(np.array_equal(labA["chirp_mass"], labH["chirp_mass"]) and np.array_equal(labA["waveform_row"], labH["waveform_row"]))
mc = labA["chirp_mass"].astype(np.float64); snr = labA["snr"]
w1 = lambda e: (np.abs(e - mc) / mc < 0.01)
CK = {"d64": "s4d_energy_max_256ch_dstate64_softmax_train_snr8_prob0.96__ck_266", "d512": "s4d_energy_max_256ch_dstate512_softmax_train_snr8_prob0.96__ck_288",
      "band": "s4d_energy_max_256ch_dstate512_bandinit_softmax_train_snr8_prob0.96__ck_122", "free": "free_kernel_energy_max_256ch_softmax_train_snr8_prob0.96__ck_248"}
grid = torch.arange(math.log(0.85), math.log(2.75) + 1e-9, 0.005).float()
for tag, ck in CK.items():
    e = np.load(f"{P}/data/testA/est__{tag}.npy")
    eh = np.load(f"{H}/data/est_noisy__{ck}.npy")
    pe = torch.tensor(np.load(f"{P}/data/testA/pe__{tag}.npy"))
    sd = torch.load(f"{H}/ckpt/{ck}.ckpt", map_location="cpu", weights_only=False)["state_dict"]
    z = torch.log1p(pe); z = (z - z.mean(-1, keepdim=True)) / z.std(-1, keepdim=True).clamp_min(1e-6)
    logits = z @ sd["model.out.weight"].float().T + sd["model.out.bias"].float()
    j = logits.argmax(-1); jj = j.clamp(1, len(grid) - 2); r = torch.arange(len(j))
    y0, y1, y2 = logits[r, jj - 1], logits[r, jj], logits[r, jj + 1]; x0, x1, x2 = grid[jj - 1], grid[jj], grid[jj + 1]
    d = (x0 - x1) * (x0 - x2) * (x1 - x2)
    a = (x2 * (y1 - y0) + x1 * (y0 - y2) + x0 * (y2 - y1)) / d; b = (x2**2 * (y0 - y1) + x1**2 * (y2 - y0) + x0**2 * (y1 - y2)) / d
    v = -b / (2 * a); inner = (j > 0) & (j < len(grid) - 1) & (a < 0) & (v > x0) & (v < x2)
    mine = torch.where(inner, v, grid[j]).exp().double().numpy()
    out[tag] = dict(own_head_testA_w1=float(w1(e).mean()), harness_f16_w1=float(w1(eh).mean()), flips_vs_harness=int((w1(e) != w1(eh)).sum()),
                    my_head_from_pe_w1=float(w1(mine).mean()), my_head_vs_stored_max_rel=float(np.max(np.abs(mine - e) / e)),
                    my_head_flips_vs_stored=int((w1(mine) != w1(e)).sum()))
    bj = json.load(open(f"{P}/results/big__{tag}__n512000.json"))["configs"]
    out[tag]["fresh_512k"] = {k: v["testA"]["within1"] for k, v in bj.items()}
bj = json.load(open(f"{P}/results/big__bank__n512000.json"))["configs"]
out["bank_fresh_512k"] = {k: v["testA"]["within1"] for k, v in bj.items()}
# row disjointness
rowsA = set(labA["waveform_row"].tolist())
tr = set()
for f in sorted(glob.glob(f"{P}/data/train*/labels.npz")):
    tr |= set(np.load(f)["waveform_row"].tolist())
es = set(np.load(f"{P}/data/es/labels.npz")["waveform_row"].tolist())
tb = set(np.load(f"{P}/data/testB/labels.npz")["waveform_row"].tolist())
with h5py.File("/home/kyoon/SSM-BNS/DATA/aframe/train/end_o3_ratesandpops_bns_uniform_chirp.hdf5", "r") as f:
    nrows = f["waveforms/cross"].shape[0]
out["rows"] = dict(file_rows=int(nrows), train_distinct=len(tr), train_and_testA=len(tr & rowsA), train_and_es=len(tr & es), train_and_testB=len(tr & tb),
                   es_and_testA=len(es & rowsA), testB_and_testA=len(tb & rowsA), es_and_testB=len(es & tb))
out["testA_snr_shares"] = dict(lt8=float((snr < 8).mean()), ge16=float((snr >= 16).mean()))
json.dump(out, open("/tmp/claude-1024/-home-kyoon-SSM-BNS/82effc75-bb98-44ad-af20-e41e39a7258d/scratchpad/s4dmath/verify/v_predict.json", "w"), indent=1)
print(json.dumps(out, indent=1))
