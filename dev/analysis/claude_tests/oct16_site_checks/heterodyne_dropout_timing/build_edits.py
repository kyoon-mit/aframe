"""Build the drafted edits, check each old_str occurs exactly once, apply them to copies in the scratch dir, lint new text."""
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DOCS = Path("/home/kyoon/SSM-BNS/WEB/kyoon-mit.github.io/docs")
DER = DOCS / "projects/bns/derivations.md"
ART = DOCS / "_updates/2026-10-15-oscillators-and-chirps.md"
SEP = DOCS / "_updates/2026-09-23-denoiser-scans.md"

het = (HERE / "text/het_grid_worst.md").read_text().rstrip("\n")
step = (HERE / "text/step_time.md").read_text().rstrip("\n")
D = "{{ d | relative_url }}#oct15-het-grid-worst"

stop_h = "## When to stop training: patience and the learning-rate cycle {#oct16-stopping}"
res_h = "## One S4D channel: transfer function, resonance width and phase {#oct16-s4d-resonance-curve}"

reb_old = ("which lies outside the grid). That is one S4D channel's worth. The leftover also lies where the "
           "standard start already puts long-lived oscillators: at 1.2 M☉, while the chirp rises from 80 to "
           "512 Hz the leftover stays between 3.2 and 21 Hz, and a channel of 32 oscillators lasting 1 s spans "
           "0 to 31 Hz.")
reb_new = reb_old.replace("which lies outside the grid).", f"which lies outside the grid; [derivation]({D})).") \
                 .replace("spans 0 to 31 Hz.", f"spans 0 to 31 Hz ([derivation]({D})).")
ver_old = ("(we repeated the counts for both signs of the error). At 1.2 M☉ the leftover stays between 3.2 and "
           "21 Hz.")
ver_new = (f"(we repeated the counts for both signs of the error; [derivation]({D})). At 1.2 M☉ the leftover "
           f"stays between 3.2 and 21 Hz ([derivation]({D})).")
cav_old = "No test with an error of 1 to 10% exists. The experiment that decides the"
cav_new = ("Every one of these denoiser runs, with heterodyning or without, had\n"
           "dropout 0.2 inside every S4D block, the setting that broke our small S4D\n"
           "chirp-mass regressors on clean signals\n"
           "([16 October]({{ u | append: '2026-10-16-why-dropout-broke-our-s4d/' | relative_url }})),\n"
           "and none has been rerun without it.\n" + cav_old)
inc = "{% include notice-snr-powerlaw.html %}"
notice = ("Correction, 16 October: every network on this page had dropout 0.2 inside every S4D block, which the "
          "page did not say. That includes the denoisers with and without heterodyning and both parts of the joint "
          "denoiser and chirp-mass regressor. Dropout 0.2 is what broke our small S4D chirp-mass regressors on "
          "clean signals; with dropout 0 they put 99.9% of clean validation events within 1% of the true chirp "
          "mass. None of the runs below has been repeated without dropout, so whether and how much it changed the "
          "overlaps, the gain from heterodyning or the joint model's chirp-mass fractions is not known. See "
          "[why dropout broke our S4D]({{ '/projects/bns/updates/2026-10-16-why-dropout-broke-our-s4d/' | relative_url }}).")
sep_new = inc + "\n\n<blockquote markdown=\"1\">\n" + notice + "\n</blockquote>"

grid_old = ("Cycles for windows of 3.5 / 4 s. The last row is the grid of the AFRAME BNS\n"
            "search ([how much phase heterodyning leaves](#proj-heterodyne)).")
grid_new = grid_old + ("\nThe leftover at the first grid's two worst errors, a true chirp mass 6.74%\n"
                       "above or 6.31% below a grid mass, is worked out\n"
                       "[below](#oct15-het-grid-worst).")

edits = [
    (DER, stop_h, het + "\n\n" + stop_h, "(i) new derivation of the ten-mass grid's worst-case numbers, before oct16-stopping"),
    (DER, grid_old, grid_new, "(i) pointer from the oct15-het-grid grids table to its new worst-case addition"),
    (DER, res_h, "\n" + step + "\n\n" + res_h, "(iii) new oct16-step-time section before oct16-s4d-resonance-curve, with blank lines"),
    (ART, reb_old, reb_new, "(i) link the rebuttal's worst-case numbers to the new derivation"),
    (ART, ver_old, ver_new, "(i) link the verdict's worst-case numbers to the new derivation"),
    (ART, cav_old, cav_new, "(ii) one-sentence dropout caveat where the 0.900 to 0.917 result is cited"),
    (SEP, inc, sep_new, "(ii) correction notice at the top of the 23 September update"),
]

texts = {p: p.read_text() for p in (DER, ART, SEP)}
ok = True
for path, old, new, why in edits:
    n = texts[path].count(old)
    print(f"{path.name}: old_str count {n} :: {why}")
    ok &= n == 1
for anchor in ("{#oct15-het-grid-worst}", "{#oct16-step-time}"):
    print(anchor, "already present:", anchor in texts[DER])
if not ok:
    sys.exit("old_str not unique")

patched = dict(texts)
for path, old, new, why in edits:
    patched[path] = patched[path].replace(old, new, 1)
out = HERE / "patched"
out.mkdir(exist_ok=True)
for p, t in patched.items():
    (out / p.name).write_text(t)

# lint the inserted text
bad = []
for path, old, new, why in edits:
    added = new.replace(old, "")
    for pat, label in (("—", "em dash"), ("–", "en dash"), (r"\broughly\b", "roughly"), (r"\babout\b", "about")):
        if re.search(pat, added):
            bad.append((why, label))
    if added.count("$") % 2:
        bad.append((why, "odd number of $"))
print("lint problems:", bad or "none")

# blank line before every heading in the patched derivations page
lines = patched[DER].split("\n")
miss = [lines[i] for i in range(1, len(lines)) if lines[i].startswith("#") and lines[i - 1].strip() != ""]
print("headings without a blank line before them:", miss or "none")
for a in ("{#oct15-het-grid-worst}", "{#oct16-step-time}"):
    i = next(k for k, l in enumerate(lines) if a in l)
    print(a, "line", i + 1, "| before:", repr(lines[i - 1][:40]), "| next heading:", next(l for l in lines[i + 1:] if l.startswith("## "))[:60])

json.dump([{"file": str(p), "old_str": o, "new_str": n, "why": w} for p, o, n, w in edits],
          open(HERE / "edits.json", "w"), ensure_ascii=False, indent=1)
print("wrote", HERE / "edits.json")
