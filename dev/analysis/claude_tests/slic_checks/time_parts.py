"""Time template forward, VJP and network score for 64 rows on GPU."""
import sys, time
import torch
from train.experimental.slic.pe import load_windows, network_score
from train.experimental.slic.task import load_score_net

windows, ckpt = sys.argv[1], sys.argv[2]
dev = torch.device("cuda")
templates, data, truth = load_windows(windows, [0], dev)
net = load_score_net(ckpt, map_location=dev)
score_fn = network_score(net)
tpl = templates[0]
rows0 = truth[0][None].repeat(64, 1)

def timed(fn, n=5):
    fn(); torch.cuda.synchronize()
    t = time.time()
    for _ in range(n):
        out = fn()
    torch.cuda.synchronize()
    return (time.time() - t) / n, out

def fwd_nograd():
    with torch.no_grad():
        return tpl(rows0)

def fwd_bwd():
    rows = rows0.clone().requires_grad_(True)
    h = tpl(rows)
    (g,) = torch.autograd.grad(h, rows, grad_outputs=torch.ones_like(h))
    return h.detach()

t1, h = timed(fwd_nograd)
t2, _ = timed(fwd_bwd)
t3, _ = timed(lambda: score_fn(data[0] - h))
torch.cuda.reset_peak_memory_stats()
fwd_bwd(); torch.cuda.synchronize()
mem = torch.cuda.max_memory_allocated() / 2**30
print(f"64 rows: template fwd {t1:.3f} s, fwd+vjp {t2:.3f} s, net score {t3:.3f} s, fwd+vjp peak {mem:.1f} GiB")

with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
    fwd_bwd(); torch.cuda.synchronize()
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))
