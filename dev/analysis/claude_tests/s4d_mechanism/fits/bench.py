import sys, time, json
import numpy as np
sys.argv = ["varpro.py", "--device", sys.argv[1], "--threads", sys.argv[2]]
exec(open("varpro.py").read().split("def main():")[0])
ker = np.load(C.OUT_DIR / "kernels.npz"); trk = np.load(C.OUT_DIR / "tracks.npz")
tj = json.load(open(C.OUT_DIR / "tracks.json"))
K = ker["K_1.2"]
for p in (256, 1024):
    a0, b0, _ = chirp_start(trk, 1.2, p, tj["1.2"]["energy_frac"]["pos"])
    obj = Objective(K); th = theta_from(a0, b0)
    obj(th)
    if DEV.type == "cuda": torch.cuda.synchronize()
    t = time.time(); n = 20
    for _ in range(n): f, g = obj(th)
    if DEV.type == "cuda": torch.cuda.synchronize()
    print(sys.argv[2], DEV, p, f"{(time.time()-t)/n*1000:.1f} ms/eval", "match", np.sqrt(1-f),
          "maxmem MB", torch.cuda.max_memory_allocated()/1e6 if DEV.type=="cuda" else "")
