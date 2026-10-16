"""Waveform chunk loader with 0 workers: same chunks served, memory held by one chunk only."""
import resource, glob, torch, time
from train.data.waveforms.loader import Hdf5WaveformLoader, ChunkedWaveformDataset
files = sorted(glob.glob("/home/kyoon/SSM-BNS/DATA/aframe/train/end_o3_ratesandpops_bns_uniform_chirp.hdf5"))
def run(workers):
    torch.manual_seed(0)
    wl = Hdf5WaveformLoader(files, batch_size=2000, batches_per_epoch=4, channels=["cross", "plus"], path="waveforms")
    dl = torch.utils.data.DataLoader(wl, num_workers=workers, pin_memory=False, persistent_workers=workers > 0)
    ds = ChunkedWaveformDataset(dl, batch_size=8, batches_per_chunk=63)
    t0 = time.time(); n = 0; shapes = set()
    for i, b in enumerate(ds):
        w = b[0] if isinstance(b, (list, tuple)) else b
        shapes.add(tuple(w.shape) if hasattr(w, "shape") else type(w).__name__); n += 1
        if n >= 200: break
    return n, shapes, time.time() - t0
n, shapes, dt = run(0)
print(f"workers 0: {n} batches in {dt:.1f} s, shapes {shapes}, peak RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1048576:.2f} GB")
