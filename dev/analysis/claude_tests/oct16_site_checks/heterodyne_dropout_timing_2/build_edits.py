"""Build the edits as JSON, check each old_str is unique, apply them to copies, and lint the result."""
import json, os, re, sys

W = "/tmp/claude-1024/-home-kyoon-SSM-BNS/82effc75-bb98-44ad-af20-e41e39a7258d/scratchpad/oct16fix/sourcing3"
SITE = "/home/kyoon/SSM-BNS/WEB/kyoon-mit.github.io/docs"
DER = f"{SITE}/projects/bns/derivations.md"
ART = f"{SITE}/_updates/2026-10-15-oscillators-and-chirps.md"
SEP = f"{SITE}/_updates/2026-09-23-denoiser-scans.md"

grid = open(f"{W}/sec_het_grid_worst.md").read().rstrip("\n")
step = open(f"{W}/sec_step_time.md").read().rstrip("\n")
D = "{{ d | relative_url }}"

edits = []

# (i) new derivation section, before the oct16-stopping heading
h_stop = "## When to stop training: patience and the learning-rate cycle {#oct16-stopping}"
edits.append(dict(file=DER, old_str=h_stop, new_str=grid + "\n\n" + h_stop,
                  why="(i) New section deriving the ten-mass grid's worst-case leftover numbers used in section 8 of the 15 October update; placed just before the oct16-stopping heading as ordered."))

# (iii) new derivation section, before the resonance-curve heading, with the missing blank line
old_iii = ("49, 129, 289, 289, 289, 289, 609).\n"
           "## One S4D channel: transfer function, resonance width and phase {#oct16-s4d-resonance-curve}")
new_iii = ("49, 129, 289, 289, 289, 289, 609).\n\n" + step + "\n\n"
           "## One S4D channel: transfer function, resonance width and phase {#oct16-s4d-resonance-curve}")
edits.append(dict(file=DER, old_str=old_iii, new_str=new_iii,
                  why="(iii) New oct16-step-time section documenting the 15 October step-time measurement; also puts the blank line kramdown needs before the oct16-s4d-resonance-curve heading."))

# (i) links in the article: rebuttal
old_r1 = ("(3 to 7 at a 1% error; we computed both signs; corrected on 16 October, since the first "
          "version also took an error of 6.74% below, which lies outside the grid)")
new_r1 = ("(3 to 7 at a 1% error; we computed both signs; corrected on 16 October, since the first "
          "version also took an error of 6.74% below, which lies outside the grid; "
          f"[derivation]({D}#oct15-het-grid-worst))")
edits.append(dict(file=ART, old_str=old_r1, new_str=new_r1,
                  why="(i) Links the rebuttal's worst-case numbers (6.74%/6.31%, 4.0 to 4.2%, 4.9 to 5.0, 21 to 34, 476 to 744, 3 to 7) to the new derivation."))
old_r2 = "and a channel of 32 oscillators lasting 1 s spans 0 to 31 Hz."
new_r2 = f"and a channel of 32 oscillators lasting 1 s spans 0 to 31 Hz ([derivation]({D}#oct15-het-grid-worst))."
edits.append(dict(file=ART, old_str=old_r2, new_str=new_r2,
                  why="(i) Links the rebuttal's 3.2 to 21 Hz and 0 to 31 Hz to the new derivation (which cites #s4d-tones for the 0 to 31 Hz)."))

# (i) links in the article: verdict
old_v1 = "and 3 to 7 at a 1% error (we repeated the counts for both signs of the error). At 1.2 M☉ the leftover stays between 3.2 and 21 Hz."
new_v1 = ("and 3 to 7 at a 1% error (we repeated the counts for both signs of the error; "
          f"[derivation]({D}#oct15-het-grid-worst)). At 1.2 M☉ the leftover stays between 3.2 and 21 Hz "
          f"([derivation]({D}#oct15-het-grid-worst)).")
edits.append(dict(file=ART, old_str=old_v1, new_str=new_v1,
                  why="(i) Links the verdict's worst-case numbers and its 3.2 to 21 Hz to the new derivation."))

# (i) optional: the objection's numbers (first version, +-6.74%)
old_o1 = "and 3 to 7 at a 1% error (we computed these with the chirps aligned at the merger, as in the table above)."
new_o1 = ("and 3 to 7 at a 1% error (we computed these with the chirps aligned at the merger, as in the table above; "
          f"[derivation]({D}#oct15-het-grid-worst)).")
edits.append(dict(file=ART, old_str=old_o1, new_str=new_o1,
                  why="(i, optional) Links the objection's numbers to the new derivation: its 4.2% is the first-order value given there (4.213%), its 4.9 matches 4.902, and its 23 to 34 is the +6.740% column of the new table."))

# (ii)(a) caveat sentence in the article's section 8
old_a = ("window, heterodyning with a random chirp-mass error of 0.1% still gave 0.873\n"
         "([23 September]({{ u | append: '2026-09-23-denoiser-scans/' | relative_url }})).\n"
         "No test with an error of 1 to 10% exists.")
new_a = ("window, heterodyning with a random chirp-mass error of 0.1% still gave 0.873\n"
         "([23 September]({{ u | append: '2026-09-23-denoiser-scans/' | relative_url }})).\n"
         "All three runs, like the other runs of that update, had dropout 0.2 inside\n"
         "every layer, the setting that made our S4D chirp-mass regressors fail on\n"
         "clean signals\n"
         "([16 October]({{ u | append: '2026-10-16-why-dropout-broke-our-s4d/' | relative_url }}));\n"
         "whether it also held these denoisers back has not been tested.\n"
         "No test with an error of 1 to 10% exists.")
edits.append(dict(file=ART, old_str=old_a, new_str=new_a,
                  why="(ii)(a) The 0.900 and 0.917 runs (and the 0.873 run) had dropout 0.2 (wandb configs of the DENOISER-SCAN runs); one-sentence caveat where the article cites them."))

# (ii)(b) correction notice at the top of the 23 September update
old_b = ("{% include notice-snr-powerlaw.html %}\n\n"
         "Two weeks of scans on the 4-second merger window and a 16-second pre-merger")
notice = ("<blockquote markdown=\"1\">\n"
          "Correction, 16 October: every run on this page had dropout 0.2 inside every layer of its S4D networks. "
          "The same setting made our S4D chirp-mass regressors fail on clean signals; with dropout 0 they put 99.9% "
          "of clean validation events within 1% of the true chirp mass "
          "([why dropout broke our S4D]({{ '/projects/bns/updates/2026-10-16-why-dropout-broke-our-s4d/' | relative_url }})). "
          "None of these denoiser runs has been repeated without dropout, so whether it lowered the overlaps below "
          "is not known. Since every run had it, dropout is not one of the settings in which the compared runs differ.\n"
          "</blockquote>")
new_b = ("{% include notice-snr-powerlaw.html %}\n\n" + notice + "\n\n"
         "Two weeks of scans on the 4-second merger window and a 16-second pre-merger")
edits.append(dict(file=SEP, old_str=old_b, new_str=new_b,
                  why="(ii)(b) Correction notice in the blockquote style of the 14 October signal-only page: every DENOISER-SCAN run on this page had dropout 0.2 (wandb configs), none rerun without it."))

# checks
ok = True
texts = {f: open(f, encoding="utf-8").read() for f in (DER, ART, SEP)}
for e in edits:
    n = texts[e["file"]].count(e["old_str"])
    if n != 1:
        ok = False
        print("NOT UNIQUE", n, e["file"], e["old_str"][:80])
patched = dict(texts)
for e in edits:
    t = patched[e["file"]]
    assert t.count(e["old_str"]) == 1, ("sequential", e["old_str"][:60])
    patched[e["file"]] = t.replace(e["old_str"], e["new_str"])
os.makedirs(f"{W}/patched", exist_ok=True)
for f, t in patched.items():
    open(f"{W}/patched/{os.path.basename(f)}", "w", encoding="utf-8").write(t)

# lint the patched files
ids = set(re.findall(r"\{#([A-Za-z0-9_-]+)\}", patched[DER]))
print("duplicate ids:", [i for i in ids if patched[DER].count("{#" + i + "}") > 1])
for f in (DER, ART, SEP):
    lines = patched[f].split("\n")
    infence = False
    for k, ln in enumerate(lines):
        if ln.startswith("```"):
            infence = not infence
        if not infence and re.match(r"^#{1,6} ", ln) and k > 0 and lines[k - 1].strip() != "" and not lines[k - 1].startswith("---"):
            print("NO BLANK BEFORE HEADING", os.path.basename(f), k + 1, ln[:70])
# anchors used in new text
for e in edits:
    for a in re.findall(r"\]\(#([A-Za-z0-9_-]+)\)", e["new_str"]):
        if a not in ids:
            print("MISSING local anchor", a)
    for a in re.findall(r"relative_url \}\}#([A-Za-z0-9_-]+)\)", e["new_str"]):
        if a not in ids:
            print("MISSING derivations anchor", a)
    if "—" in e["new_str"] or "–" in e["new_str"]:
        print("DASH in", e["why"][:40])
    for bad in (" roughly", " about ", "approximately"):
        if bad in e["new_str"].replace(e["old_str"], ""):
            print("VAGUE WORD", bad, e["why"][:40])
print("all old_str unique:", ok)
json.dump(edits, open(f"{W}/edits.json", "w"), indent=1, ensure_ascii=False)
print("edits:", len(edits))
