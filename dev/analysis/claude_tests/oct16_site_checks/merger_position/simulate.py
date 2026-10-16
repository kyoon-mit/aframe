"""Apply the proposed edits to scratch copies, checking each old string occurs exactly once."""
import difflib
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from edits import EDITS  # noqa: E402

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sim")
os.makedirs(out, exist_ok=True)
texts = {}
for f, old, new, why in EDITS:
    if f not in texts:
        texts[f] = open(f, encoding="utf-8").read()
ok = True
orig = dict(texts)
for i, (f, old, new, why) in enumerate(EDITS):
    n = texts[f].count(old)
    if n != 1:
        print(f"EDIT {i} in {os.path.basename(f)}: old string occurs {n} times")
        ok = False
        continue
    # each old string must also occur exactly once in the ORIGINAL file
    if orig[f].count(old) != 1:
        print(f"EDIT {i}: not unique in original")
        ok = False
    texts[f] = texts[f].replace(old, new)
    for bad in ("—",):
        if bad in new:
            print(f"EDIT {i}: contains an em dash")
            ok = False
print("all edits apply once:", ok)
for f, t in texts.items():
    p = os.path.join(out, os.path.basename(f))
    open(p, "w", encoding="utf-8").write(t)
    # headings need a blank line before them
    lines = t.split("\n")
    for j, line in enumerate(lines):
        if line.startswith("#") and j > 0 and lines[j - 1].strip() != "" and not lines[j - 1].startswith("{%"):
            if line.startswith("## ") or line.startswith("### "):
                print(f"  {os.path.basename(f)}:{j + 1}: heading without blank line before: {line[:60]}")
    d = difflib.unified_diff(orig[f].split("\n"), t.split("\n"), lineterm="", n=0)
    open(p + ".diff", "w", encoding="utf-8").write("\n".join(d))
# write the edits as JSON for the final answer
json.dump([{"file": f, "old_str": o, "new_str": n, "why": w} for f, o, n, w in EDITS],
          open(os.path.join(out, "edits.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("wrote", out)
