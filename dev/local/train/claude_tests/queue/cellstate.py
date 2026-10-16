"""State of each table run: last epoch, best within 1% and its epoch, when early stopping fires."""
import sys, subprocess
sys.argv = ["x", "none"]
sys.path.insert(0, "/home/kyoon/SSM-BNS/aframe/dev/runs/claude_tests")
import importlib.util
spec = importlib.util.spec_from_file_location("wt", "/home/kyoon/SSM-BNS/aframe/dev/runs/claude_tests/warp_table.py")
src = open(spec.origin).read().split('mode = sys.argv[1]')[0]
ns = {}; exec(compile(src, "warp_table.py", "exec"), ns)
R = ns["R"]; K = ns["KEYS"]

def cycle_end_after(epoch, warmup=10, T_0=40, T_mult=2):
    end, T_i = warmup + T_0 - 1, T_0
    while end < epoch:
        T_i *= T_mult; end += T_i
    return end

bases = ["sig_only_reg_s4d_dropout0_64x32x4", "sig_only_reg_s4d_max_dropout0_64x32x4",
         "sig_only_reg_s4d_complex_mean_dropout0_64x32x4", "sig_only_reg_s4d_complex_energy_max_dropout0_64x32x4"]
runs = [b + s for b in bases for s in ("", "_warp_merger", "_warp_premerger")] + [
    "sig_only_reg_resnet_energy_max", "free_kernel_time_energy_max_256ch_softmax_train_snr4_2048hz_prob0.96",
    "sig_only_reg_free_kernel", "sig_only_reg_resnet", "sig_only_reg_dilated_cnn_energy_max"]
for r in runs:
    h = ns["rows"](r)
    if not h:
        print(f"{r}: no val rows"); continue
    v = [x.get(K[0], 0) for x in h]
    b = max(range(len(v)), key=lambda i: (v[i], -i))
    be = int(h[b].get("epoch", -1)); last = int(h[-1].get("epoch", -1))
    pat = 160 if r.startswith("free_kernel_time") else 80
    stop = max(be + pat, cycle_end_after(be))
    best = " / ".join(f"{100*h[b].get(k,0):.1f}" for k in K)
    log = f"{R}/{r}/train.log"
    es = subprocess.run(["grep", "-a", "-c", "Signaling Trainer to stop", log], capture_output=True, text=True).stdout.strip()
    print(f"{r}: last ep {last}, best {best} at ep {be}, ES fires at ep {stop}, ES msgs {es}")
