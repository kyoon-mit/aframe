"""Per-batch elapsed times of the three noisy runs from the progress bars in their training logs."""
import re, sys
import numpy as np
base = '/home/kyoon/SSM-BNS/aframe/dev/runs/claude_tests/'
runs = ['free_kernel_energy_max_256ch_softmax_train_snr8_prob0.96',
        's4d_energy_max_256ch_dstate64_softmax_train_snr8_prob0.96',
        's4d_energy_max_256ch_dstate512_softmax_train_snr8_prob0.96']
logged = [238, 260, 284]
def secs(t):
    s = 0
    for p in t.split(':'):
        s = s * 60 + int(p)
    return s
pat = re.compile(r'Epoch (\d+): 100%\|[^|]*\| 1564/1564 \[([0-9:]+)<00:00')
for r, lg in zip(runs, logged):
    data = open(base + r + '/train.log', 'rb').read().decode('utf-8', 'replace')
    first, last = {}, {}
    for m in pat.finditer(data):
        e, s = int(m.group(1)), secs(m.group(2))
        first.setdefault(e, s)
        last[e] = s
    va = [secs(m.group(1)) for m in re.finditer(r'Validation DataLoader 0: 100%\|[^|]*\| 64/64 \[([0-9:]+)<00:00', data)]
    n = len(first)
    f = np.array([first[e] for e in sorted(first)]); l = np.array([last[e] for e in sorted(last)])
    print(f'{r[:30]}: epochs {n}, val bars {len(va)}')
    print(f'   training loop only: {1000*f.sum()/(n*1564):.1f} ms per training batch')
    print(f'   last refresh of the epoch bar: {1000*l.sum()/(n*1628):.1f} ms per batch (1,628 per epoch); logged {lg}; diff {100*(1000*l.sum()/(n*1628)-lg)/lg:+.1f}%')
    print(f'   training loop + validation bars: {1000*(f.sum()+np.sum(va[:n]))/(n*1628):.1f} ms per batch; diff {100*(1000*(f.sum()+np.sum(va[:n]))/(n*1628)-lg)/lg:+.1f}%')
    print(f'   mean (last - first) {np.mean(l-f):.2f} s, mean val bar {np.mean(va):.2f} s')
