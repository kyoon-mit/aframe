import json, re, sys, collections
import yaml

SRC = "/home/kyoon/SSM-BNS/WEB/kyoon-mit.github.io/docs/_data/glossary.yml"
W = "/tmp/claude-1024/-home-kyoon-SSM-BNS/82effc75-bb98-44ad-af20-e41e39a7258d/scratchpad/oct16fix/glossary/"
orig = open(SRC, encoding="utf-8").read()
block = open(W + "new_entries.yml", encoding="utf-8").read()

anchor = ('    Boyd and Vandenberghe, *Convex Optimization* (Cambridge University Press,\n'
          '    2004), section 3.1.8, equation 3.5: "f(E x) ≤ E f(x)".')
edits = [
    ("- id: effective-spin\n  term: effective spin\n  also: [χ_eff]",
     "- id: effective-spin\n  term: effective spin\n  also: [χ_eff, effective spins]"),
    ("- id: matrix-pencil\n  term: matrix-pencil method\n  group: Signal processing",
     "- id: matrix-pencil\n  term: matrix-pencil method\n  also: [matrix pencil, matrix-pencil fit, matrix-pencil fits]\n  group: Signal processing"),
    ("- id: taper\n  term: taper\n  group: Signal processing",
     "- id: taper\n  term: taper\n  also: [tapered]\n  group: Signal processing"),
    ("  also: [mean pooling, max pooling, mean pool, max pool, pooling layer]",
     "  also: [mean pooling, max pooling, mean pool, max pool, pooling layer, pools, pooled value, pooled values]"),
    ("  term: warm-restart cosine schedule\n  also: [warm restarts, cosine schedule]",
     "  term: warm-restart cosine schedule\n  also: [warm restarts, cosine schedule, learning-rate cycle, cosine cycles]"),
    ("  term: step size\n  also: [timestep, Δt, dt]",
     "  term: step size\n  also: [timestep, Δt, dt, time step]"),
    ("  also: [long convolutions, long-convolution layer, long kernel, long kernels]",
     "  also: [long convolutions, long-convolution layer, long-convolution, long kernel, long kernels]"),
    ("  term: early stopping\n  also: [early stop, stopped early]",
     "  term: early stopping\n  also: [early stop, stopped early, stops early, stopping rule]"),
    (anchor, anchor + "\n" + block.rstrip("\n")),
]

text = orig
for i, (old, new) in enumerate(edits, 1):
    n = text.count(old)
    n0 = orig.count(old)
    assert n == 1 and n0 == 1, (i, n, n0, old[:60])
    text = text.replace(old, new)
open(W + "glossary.patched.yml", "w", encoding="utf-8").write(text)
json.dump([{"old_str": o, "new_str": n} for o, n in edits], open(W + "edits.json", "w"), ensure_ascii=False, indent=1)
print("edits applied:", len(edits), "; old_str each found once in the current file")

# --- parse and structural checks
d0 = yaml.safe_load(orig)
d = yaml.safe_load(text)
print("entries before/after:", len(d0), len(d))
ids = [e["id"] for e in d]
dup = [k for k, v in collections.Counter(ids).items() if v > 1]
print("duplicate ids:", dup)
ORDER = ["id", "term", "also", "only_also", "case", "group", "short", "def", "source"]
REQ = {"id", "term", "group", "short", "source"}
groups = {"Gravitational waves", "Detectors and data", "Searching for signals", "Signal processing",
          "Statistics", "Machine learning", "State-space models", "This project", "Project 8"}
bad = []
for k, e in enumerate(d):
    keys = list(e.keys())
    if not REQ <= set(keys) or not set(keys) <= set(ORDER):
        bad.append((e["id"], "keys", keys))
    if keys != [x for x in ORDER if x in keys]:
        bad.append((e["id"], "order", keys))
    if e["group"] not in groups:
        bad.append((e["id"], "group", e["group"]))
    for f in ("term", "short", "source"):
        if not isinstance(e[f], str):
            bad.append((e["id"], f, type(e[f])))
    if "also" in e and not (isinstance(e["also"], list) and all(isinstance(a, str) for a in e["also"])):
        bad.append((e["id"], "also", e["also"]))
print("structural problems:", bad)
new_ids = [e["id"] for e in d if e["id"] not in set(x["id"] for x in d0)]
print("new ids:", new_ids)
pos = {e["id"]: k for k, e in enumerate(d)}
for nid in new_ids:
    k = pos[nid]
    nb = [d[j] for j in (k - 2, k - 1, k + 1, k + 2) if 0 <= j < len(d)]
    mine = set(d[k].keys())
    shapes = [tuple(x.keys()) for x in nb]
    print(f"  {nid:16s} keys={tuple(d[k].keys())}  same key set as a neighbour: {any(set(s) == mine for s in shapes)}")
changed = [e["id"] for e in d if e["id"] in set(x["id"] for x in d0) and e != next(x for x in d0 if x["id"] == e["id"])]
print("existing entries changed:", changed)
for e in d:
    if e["id"] in changed:
        old = next(x for x in d0 if x["id"] == e["id"])
        diff = {k: (old.get(k), e.get(k)) for k in set(old) | set(e) if old.get(k) != e.get(k)}
        print("   ", e["id"], diff)

# --- spellings: no form shared between two ids (case-insensitive)
form_owner = collections.defaultdict(set)
for e in d:
    for f in ([] if e.get("only_also") else [e["term"]]) + list(e.get("also") or []):
        form_owner[f.lower()].add(e["id"])
clash = {f: o for f, o in form_owner.items() if len(o) > 1}
print("forms owned by more than one id:", clash)

# --- wording checks on the added text only
added = block + "".join(n for o, n in edits[:-1])
for pat, name in [("—", "em dash"), (r"\broughly\b", "roughly"), (r"\babout\b", "about"), (r"\bapprox", "approx")]:
    print(f"'{name}' in added text:", len(re.findall(pat, added)))
long = [l for l in block.split("\n") if len(l) > 80 and "](" not in l]
print("added lines over 80 chars (excluding link lines):", long)
