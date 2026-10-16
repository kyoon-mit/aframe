"""Build the edit list for glossary.yml, apply it to a copy, and check the copy."""
import collections, json, re, sys
import yaml

SRC = '/home/kyoon/SSM-BNS/WEB/kyoon-mit.github.io/docs/_data/glossary.yml'
OUTDIR = '/tmp/claude-1024/-home-kyoon-SSM-BNS/82effc75-bb98-44ad-af20-e41e39a7258d/scratchpad/oct16fix/glossary/'
block = open(OUTDIR + 'append_block.yml', encoding='utf-8').read()
assert block.startswith('\n') and block.endswith('\n')

last = '    2004), section 3.1.8, equation 3.5: "f(E x) ≤ E f(x)".'
edits = [
    (last, last + '\n' + block.rstrip('\n'),
     'Append 14 new entries (architecture, ReLU, residual block, 1 × 1 convolution, tensor, weight decay, warmup, patience, callback, seed, Hann window, main lobe, real FFT, detector frame) after the last entry of the file.'),
    ('  also: [early stop, stopped early]',
     '  also: [early stop, stopped early, stopping rule]',
     'early-stopping: the 16 October status and dropout pages call our early stopping "the stopping rule".'),
    ('  also: [warm restarts, cosine schedule]',
     '  also: [warm restarts, cosine schedule, learning-rate cycle, cosine cycles]',
     'warm-restart-cosine: the status page and #oct16-stopping write "learning-rate cycle" and "cosine cycles".'),
    ('  also: [timestep, Δt, dt]',
     '  also: [timestep, Δt, dt, time step per channel]',
     'step-size: the 15 October lecture calls S4D\'s Δt "one time step per channel"; the bare "time step" is not added because other pages use it for a sample of a time series.'),
    ('  also: [χ_eff]',
     '  also: [χ_eff, effective spins]',
     'effective-spin: what-s4d-computes and the derivations write the plural, which the singular form does not match.'),
    ('- id: matrix-pencil\n  term: matrix-pencil method\n  group: Signal processing',
     '- id: matrix-pencil\n  term: matrix-pencil method\n  also: [matrix-pencil, matrix pencil]\n  group: Signal processing',
     'matrix-pencil: several pages write "matrix-pencil fits" or "Matrix pencil", which the term "matrix-pencil method" does not match.'),
    ('- id: taper\n  term: taper\n  group: Signal processing',
     '- id: taper\n  term: taper\n  also: [tapered]\n  group: Signal processing',
     'taper: the 15 October lecture first writes "tapered".'),
    ('  also: [mean pooling, max pooling, mean pool, max pool, pooling layer]',
     '  also: [mean pooling, max pooling, mean pool, max pool, pooling layer, pooled value, pooled values]',
     'pooling: the dropout page and its derivations speak of "the pooled value(s)"; the bare "pooled" is not added because other pages use it for combined test sets.'),
    ('  also: [long convolutions, long-convolution layer, long kernel, long kernels]',
     '  also: [long convolutions, long-convolution layer, long kernel, long kernels, long-convolution]',
     'long-convolution: the 15 October lecture and #oct16-s4d-fft-offset write "our long-convolution code", which no current spelling matches.'),
    ("  also: [channel's zeros, right-half-plane zeros, zeros in the right half plane]",
     "  also: [channel's zeros, channel’s zeros, right-half-plane zeros, zeros in the right half plane]",
     "filter-zero: kramdown turns the apostrophe of \"channel's zeros\" on what-s4d-computes into ’, so the straight-apostrophe spelling never matches; add the typographic one (as for Jensen’s inequality)."),
]

text = open(SRC, encoding='utf-8').read()
new = text
for old, rep, why in edits:
    n = new.count(old)
    assert n == 1, (old, n)
    assert text.count(old) == 1, ('not unique in original', old)
    new = new.replace(old, rep)
open(OUTDIR + 'glossary.yml', 'w', encoding='utf-8').write(new)
json.dump([{'file': SRC, 'old_str': o, 'new_str': r, 'why': w} for o, r, w in edits],
          open(OUTDIR + 'edits.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)

# checks
d0 = yaml.safe_load(text)
d = yaml.safe_load(new)
print('parses: original', len(d0), 'entries; edited', len(d), 'entries')
ids = [e['id'] for e in d]
dup = [k for k, c in collections.Counter(ids).items() if c > 1]
print('duplicate ids:', dup)
allowed = ['id', 'term', 'also', 'only_also', 'case', 'group', 'short', 'def', 'source']
required = {'id', 'term', 'group', 'short', 'source'}
groups = set(e['group'] for e in d0)
bad = []
for e in d:
    ks = list(e.keys())
    if not required <= set(ks):
        bad.append((e['id'], 'missing', required - set(ks)))
    if set(ks) - set(allowed):
        bad.append((e['id'], 'unknown', set(ks) - set(allowed)))
    order = [k for k in allowed if k in ks]
    if ks != order:
        bad.append((e['id'], 'key order', ks))
    if e['group'] not in groups:
        bad.append((e['id'], 'new group', e['group']))
    for k in ('term', 'short', 'source'):
        if not isinstance(e[k], str) or not e[k].strip():
            bad.append((e['id'], 'empty', k))
    if 'also' in e and not all(isinstance(a, str) and a for a in e['also']):
        bad.append((e['id'], 'also not strings', e['also']))
print('key problems:', bad)
newids = [e['id'] for e in d if e['id'] not in set(x['id'] for x in d0)]
print('new ids:', newids)
keysets = collections.Counter(tuple(e.keys()) for e in d0)
for e in d:
    if e['id'] in newids:
        ks = tuple(e.keys())
        print(f"  {e['id']:24s} keys={ks} used by {keysets.get(ks, 0)} existing entries")
# every form (term and also) must be new: not equal (case-insensitively) to an existing form of another id
def forms(e):
    return ([] if e.get('only_also') else [e['term']]) + list(e.get('also') or [])
owner = {}
for e in d0:
    for f in forms(e):
        owner.setdefault(f.lower(), set()).add(e['id'])
clash = []
for e in d:
    e0 = next((x for x in d0 if x['id'] == e['id']), None)
    old = set(forms(e0)) if e0 else set()
    for f in forms(e):
        if f in old:
            continue
        o = owner.get(f.lower(), set()) - {e['id']}
        if o:
            clash.append((e['id'], f, o))
print('form clashes with other ids:', clash)
# em dashes and en dashes in the added text
added = new[len(text):] if new.startswith(text) else None
print('em dash in edited file beyond original count:', new.count('—') - text.count('—'))
print('"roughly"/"about" in added block:', [w for w in ('roughly', ' about ') if w in block])
