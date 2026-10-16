"""Build the exact edits, check each old_str occurs once in the current site files, dry-run them on copies."""
import json
import sys
from pathlib import Path

SITE = Path("/home/kyoon/SSM-BNS/WEB/kyoon-mit.github.io/docs")
W = Path(sys.argv[1])
OUT = W / "v3_dryrun"
B = SITE / "projects/bns/brainstorm.md"
D = SITE / "projects/bns/derivations.md"
G = SITE / "_data/glossary.yml"

item = (W / "v3_brainstorm_item.html").read_text().rstrip("\n")
sec = (W / "v3_derivation_section.md").read_text().rstrip("\n")
ent = (W / "v3_glossary_entries.yml").read_text().rstrip("\n")

b_old = ("  <li><span class=\"checklist__box\" aria-label=\"to do\"></span><span class=\"checklist__text\"><b>Heterodyne near the true chirp mass, then the joint denoiser and regressor,</b> small model, 256 Hz anti-aliased, pre-merger window. Scan the chirp-mass error ε = 0, 0.1%, 1%, 3%. This is DINGO's refinement setting in our architecture.</span></li>")
d_old = "## How the trained channels were measured {#oct16-s4d-channel-analysis}"
g_old = "    Derived on the [derivations page](/projects/bns/derivations/#oct15-in-step-time)."

edits = [
    {"file": str(B), "old_str": b_old, "new_str": item + "\n" + b_old,
     "why": "Adds the chirplet-kernel To-do item after the Mamba item in 'To try'."},
    {"file": str(D), "old_str": d_old, "new_str": sec + "\n\n" + d_old,
     "why": "Adds derivation section oct16-chirplet-count (in-step time of a linear chirp, piece counts, chirplet constructions and fits) after the resonator-fit section."},
    {"file": str(G), "old_str": g_old, "new_str": g_old + "\n" + ent,
     "why": "Adds glossary entries 'chirplet' (with chains and kernels) and 'linear chirp' after 'in-step time'."},
]
ok = True
for e in edits:
    txt = Path(e["file"]).read_text()
    n = txt.count(e["old_str"])
    print(Path(e["file"]).name, "occurrences of old_str:", n)
    ok &= n == 1
    new = txt.replace(e["old_str"], e["new_str"], 1)
    (OUT / Path(e["file"]).name).write_text(new)
print("all unique:", ok)
json.dump(edits, open(OUT / "edits.json", "w"), indent=1, ensure_ascii=False)
