"""Re-draw the SNR labels of the saved sets with ml4gw's PowerLaw as harness._draws does; count tail fraction over seeds."""
import numpy as np, torch
from ml4gw.distributions import Cosine, PowerLaw
torch.set_num_threads(4)
D = "/tmp/claude-1024/-home-kyoon-SSM-BNS/82effc75-bb98-44ad-af20-e41e39a7258d/scratchpad/s4dmath/harness/data"
def draw(n, seed):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        Cosine().sample((n,)); torch.distributions.Uniform(0.0, 3.14159, validate_args=False).sample((n,))
        torch.distributions.Uniform(-3.14159, 3.14159, validate_args=False).sample((n,))
        return PowerLaw(minimum=4.0, maximum=100.0, index=-3).sample((n,)).numpy()
for s, n, seed in (("noisy", 20000, 2002), ("clean", 4000, 1001)):
    a = draw(n, seed); b = np.load(f"{D}/{s}_snr.npy")
    print(s, "labels equal redraw:", np.array_equal(a, b), "frac>20", (a > 20).mean())
fr = np.array([(draw(20000, k) > 20).mean() for k in range(3000, 3400)])
print("400 other seeds, n=20000: mean %.4f%% sd %.4f%%, min %.4f%%, share <= 3.395%%: %d" % (100 * fr.mean(), 100 * fr.std(), 100 * fr.min(), (fr <= 0.03395).sum()))
