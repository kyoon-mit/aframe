"""Runtime, epoch and step of the three noisy long-conv runs (read-only wandb API)."""
import wandb
api = wandb.Api(timeout=30)
P = "kyoon-mit-massachusetts-institute-of-technology/CLAUDE-TESTS/"
for i in ("free_kernel_energy_max_256ch_softmax_train_snr8_prob0.96",
          "s4d_energy_max_256ch_dstate64_softmax_train_snr8_prob0.96",
          "s4d_energy_max_256ch_dstate512_softmax_train_snr8_prob0.96"):
    try:
        r = api.run(P + i)
        s = r.summary
        rt, ep, st = s.get("_runtime"), s.get("epoch"), s.get("trainer/global_step")
        print(i, "| state", r.state, "| heartbeat", r.heartbeat_at, "| runtime", rt, "| epoch", ep, "| step", st)
        if rt and st:
            ep_done = st / 1564
            print(f"   steps/1564 = {ep_done:.3f}; runtime/(floor(steps/1564)*1628) = {1000*rt/(int(ep_done)*1628):.1f} ms;"
                  f" runtime/(steps + 64*floor(steps/1564)) = {1000*rt/(st + 64*int(ep_done)):.1f} ms")
    except Exception as e:
        print(i, "ERROR", type(e).__name__, str(e)[:200])
