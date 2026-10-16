import glob, h5py
tot = 0
val = {"background-1238412673-14008.hdf5", "background-1238384700-27833.hdf5"}
for f in sorted(glob.glob("/home/kyoon/SSM-BNS/DATA/O3a_H1_L1_2048Hz/background/*.hdf5")):
    name = f.split("/")[-1]
    with h5py.File(f, "r") as h:
        lens = {k: h[k].shape[-1] for k in h.keys()}
    dur = min(lens.values()) / 2048
    if name not in val:
        tot += dur
    print(name, lens, dur)
print("train total s", tot, "days", tot / 86400)
