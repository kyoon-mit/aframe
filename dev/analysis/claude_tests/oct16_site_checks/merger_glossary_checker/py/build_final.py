"""Build the corrected edit set from the drafts, apply it to snapshot copies, and check it."""
import json
import re
import shutil
from pathlib import Path

import yaml

CK = Path("/tmp/claude-1024/-home-kyoon-SSM-BNS/82effc75-bb98-44ad-af20-e41e39a7258d/scratchpad/oct16fix/checker")
SITE = "/home/kyoon/SSM-BNS/WEB/kyoon-mit.github.io/docs"
DER = f"{SITE}/projects/bns/derivations.md"
LEC = f"{SITE}/_updates/2026-10-15-oscillators-and-chirps.md"
WHAT = f"{SITE}/_updates/2026-10-16-what-s4d-computes.md"
STAT = f"{SITE}/_updates/2026-10-16-status-dropout-and-signal-only.md"
GLOS = f"{SITE}/_data/glossary.yml"

merger = json.load(open(CK / "merger_edits_draft.json"))
gloss = json.load(open(CK / "glossary_edits_draft.json"))

# ---------------------------------------------------------------- derivations
D0_NEW = r"""Setting $\tau = 4$ s gives the
frequency 4 s before merger and the cycles after it, and $\tau = 3.5$ s the
same for the last 3.5 s:

<div markdown="1">

| chirp mass | frequency 4 s before merger | cycles in the last 4 s | frequency 3.5 s before merger | cycles in the last 3.5 s |
|---|---|---|---|---|
| 0.85 M☉ | 99 Hz | 636 | 105 Hz | 585 |
| 1.2 M☉ | 80 Hz | 513 | 84 Hz | 472 |
| 1.8 M☉ | 62 Hz | 398 | 65 Hz | 366 |
| 2.75 M☉ | 48 Hz | 305 | 50 Hz | 281 |

</div>

In our 4 s windows the merger falls 3.5 to 4 s after the first sample
([where the merger falls](#oct16-merger-position)), so a window holds the
last 3.5 to 4 s of the chirp. Over 0.85 to 2.75 M☉ the chirp in a window
therefore starts at 48 to 105 Hz and makes 281 to 636 cycles (280 to 636 in
one detector, where the merger can move by a further 21.0 ms). Since
$f \propto \tau^{-3/8}$ and $N \propto \tau^{5/8}$, each 3.5 s frequency is
$(4/3.5)^{3/8} = 1.0513$ times the 4 s one, and each 3.5 s count
$(3.5/4)^{5/8} = 0.9199$ times.

These are leading-order values; post-Newtonian corrections are not included.

**Check.** Python (numpy and scipy), with
$GM_\odot/c^3 = 4.925490947\times10^{-6}$ s ([constants](#ck-chirp)): the
closed form for $N$ and a numerical integral of $f$ over the time before
merger agree to better than $10^{-14}$ (relative). At 4 s, 99.41, 80.14,
62.20 and 47.72 Hz and 636.23, 512.88, 398.07 and 305.43 cycles; at 3.5 s,
104.52, 84.25, 65.39 and 50.17 Hz and 585.29, 471.81, 366.19 and 280.98
cycles. For one detector the count starts at the window's first sample,
3.4795 to 4.0210 s before the merger, and stops at the merger or at the
window's end, whichever comes first: 279.95 cycles at the fewest (2.75 M☉)
and 636.23 at the most (0.85 M☉).

## Where the merger falls in our 4 s windows {#oct16-merger-position}

**Claim** (15 and 16 October updates): in the training, validation and test
windows of our 4 s runs, signal-only and noisy, the merger falls 3.5 to 4 s
after the window's first sample, at one of 1,024 positions, each equally
likely. A window therefore holds the last 3.5 to 4 s of the chirp and up to
0.5 s after the merger. In each detector the merger moves by at most a
further 21.0 ms either way. The merger runs with a time warp use the same
windows, warped up to each detector's amplitude peak; the early runs with a
time warp use windows that end 25.0 to 25.5 s before the merger.

**Setup.** Our training code (built on aframe and ml4gw) cuts every window
the same way.

- The stored waveforms (IMRPhenomPv2, generated with PyCBC) are 32 s long at
  2048 Hz, 65,536 samples, with the coalescence placed 2 s before the end, at
  sample 61,440. That sample is the merger here.
- A window is 4 s of whitened data, 8,192 samples, numbered here from 0; it
  ends 4 s after its first sample, one sample after its last. The whitening
  filter is 1 s long, so the code whitens a 5 s segment (10,240 samples) and
  drops 0.5 s (1,024 samples) at each end.
- The run settings ask for the merger at least 3.5 s after the window's start
  and no later than its end.

**Derivation.** From each stored waveform the code first cuts the 11,264
samples from 9,216 before the merger to 2,048 after it, so that the merger is
sample 9,216 of the cut: $9{,}216 = 10{,}240 - 1{,}024$, the segment less half
the filter (no time is required after the merger), and
$2{,}048 = 10{,}240 - 8{,}192$, the segment less the 3.5 s (7,168 samples)
required before the merger and half the filter. It then takes the 5 s
segment that starts at sample $k$ of this cut, with $k$ drawn with equal
probability from the 1,024 whole numbers 0 to 1,023
($11{,}264 - 10{,}240 = 1{,}024$ choices). The merger is at sample
$9{,}216 - k$ of the segment and, once the first 1,024 samples are dropped,
at sample $8{,}192 - k$ of the window: from sample 7,169, 3.5005 s after the
first, to sample 8,192, one past the last, at the window's end. So the window
ends 0 to 0.4995 s after the merger.

Projecting onto each detector also moves the waveform, by the difference
between the wave's arrival at the detector and at the Earth's centre for the
source's direction, cut to whole samples toward zero. That difference is at
most the detector's distance from the Earth's centre over $c$, 21.24 ms for
H1 and 21.26 ms for L1, so the shift is at most 43 samples (21.0 ms): in
each detector the merger falls 3.4795 to 4.0210 s after the first sample.
When it falls after the window's end, the window misses at most the last
21.0 ms of that detector's chirp.

The signal-only runs read the whitened signal alone, the noisy runs the same
signal added to real noise before whitening, and their validation and test
windows are made by the same code. The merger runs with a time warp take the
same window and resample it, from its first sample up to $t_c$, on a grid
uniform in $(t_c - t)^{5/8}$ ([time warp](#time-warp)), with $t_c$ the peak
of each detector's whitened amplitude (the size of the analytic signal), so
their input ends at that peak. In each of the first batches logged by three
of these runs, the peak lay a median of 10.7 to 12.2 ms from the merger
placed as above, and at most 35 to 63 ms. The early runs with a time warp
move the cut 25.5 s (52,224 samples) earlier: the window then starts
29.0005 to 29.5 s and ends 25.0005 to 25.5 s before the merger. The stored
waveforms start 30 s before the merger, so this is the earliest window they
allow.

**Check.** Python, with the training code's own functions (its cut of the
stored waveforms, and ml4gw's choice of segment, detector shift and
whitening): each stored waveform was replaced by the numbers 0, 1, 2, …, so
that every window sample shows which stored sample it came from. The segment
choice drew all 1,024 values of $k$ in 20,000 draws, and over these values
the merger falls at window samples 7,169 to 8,192 (3.500488 to 4.000000 s,
mean 3.750244 s). The detector shift equals the arrival-time difference cut
toward zero (tested on 4,096 directions) and reached 43 samples either way;
over 4,000,000 directions drawn as in training, the merger fell 3.479492 to
4.020996 s after the first sample in a detector, and after the window's end
in 1.04% of detector windows, the share computed exactly for directions
spread evenly over the sky. With the early runs' cut it fell 29.000488 to
29.500000 s after the first sample. A single nonzero sample, whitened with
the 1 s filter and the 20 Hz highpass, peaks at its own position less 1,024,
with a flat noise spectrum and with a steeply varying one. In 192 stored
waveforms (three blocks of 64, chirp masses 0.87 to 2.75 M☉), the amplitude
$\sqrt{h_+^2 + h_\times^2}$ peaks 4 to 23 samples (2.0 to 11.2 ms) before
the merger sample, and around the merger sample the median of eight
successive phase steps gives a frequency of 963 to 1024 Hz, close to the
1024 Hz Nyquist frequency of the stored data.

## S4D tones: frequency and memory {#s4d-tones}"""

D6_NEW = r"""| 2.75 M☉ | 47.72 Hz | 305.4 | 114.5 |

</div>

A channel's kernel is the same in every window, so a channel that is to be
the matched filter in all our windows must cover this longest case. The
chirp in a window whose merger falls 3.5 s in has $(3.5/4)^{5/8} = 0.9199$
of each count: 585.3, 471.8, 366.2 and 281.0 cycles and $\int\tau\,df$ =
219.5, 176.9, 137.3 and 105.4 at the four chirp masses, so that chirp alone
would need at least 106 to 220 resonators.

**Derivation.** With $\tau \propto f^{-8/3}$,"""

D7_NEW = r"""with $\tau_1$ = 4 s this gives the table, and since $f_1 \propto \tau_1^{-3/8}$,
$\tau_1 f_1 \propto \tau_1^{5/8}$ gives the 3.5 s values. A channel's output
at the merger adds its kernel times the data back to the window's first
sample, so a kernel matched to the last 4 s is also the matched filter of the
shorter stretch in a window whose merger falls earlier, while one matched
only to the last 3.5 s misses the first 0.5 s of chirp in a window whose
merger falls at its end. In general, integrating by parts,"""

D8_NEW = r"""$2\pi\int\tau\,df$ = 1,208.4 at 1.2 M☉. The frequencies match the
[cycles](#cycles) section. The 3.5 s values were recomputed for this page
from the closed form and by quadrature, which agree to better than
$10^{-14}$ (relative): $\int\tau\,df$ = 219.48, 176.93, 137.32 and 105.37,
and 585.29, 471.81, 366.19 and 280.98 cycles. In one detector the shortest
stretch is 3.4795 s ([where the merger falls](#oct16-merger-position)),
which gives 104.98 at 2.75 M☉. A merger that falls after the window's end
in one detector, by up to 21.0 ms, cuts the top of the band, so
$\int\tau\,df$ stays below its 4 s value: 230.40 at 0.85 M☉ for a merger
21.0 ms after the end."""

D4B = {
    "file": DER,
    "old_str": "M☉, for windows of\n3.5 and 4 s and both signs of $\\epsilon$; the leftover grows monotonically",
    "new_str": "M☉, for the last\n3.5 and 4 s before the merger and both signs of $\\epsilon$; the leftover grows monotonically",
    "why": "#oct15-het-grid Check: the same wording fix as in its claim, ranges note and grid note (the windows are 4 s; the stretch before the merger is 3.5 to 4 s). Numbers unchanged.",
}

# ---------------------------------------------------------------- update pages
W10_NEW = r"""([derivation]({{ d | relative_url }}#oct16-s4d-three-eighths)). In our 4 s
windows the merger falls 3.5 to 4 s after the first sample
([derivation]({{ d | relative_url }}#oct16-merger-position)), so a window
holds the last 3.5 to 4 s of the chirp. A channel's kernel is the same in
every window, so to be the matched filter in all of them it must cover the
longest case, the last 4 s:"""

W11_NEW = r"""One channel needs at least 115 to 239 resonators over our range of chirp
masses (the chirp of a window whose merger falls 3.5 s in would need at
least 106 to 220 on its own); the clean network has 16 per channel, the
noisy ones 32 or 256."""

L18_NEW = r"""\tfrac83\int\tau\,df$. A channel is the same in every window, so to be the
filter in all of them it must cover the longest case, the last 4 s before
the merger, with 305 to 636 cycles: that is at least 115 to 239 oscillators
for chirp masses 2.75 to 0.85 M☉, and 193 at 1.2 M☉, where
$\int\tau\,df = 192.3$. The chirp of a window whose merger falls 3.5 s in
would need at least 106 to 220 on its own
([derivation]({{ d | relative_url }}#oct16-s4d-three-eighths)). The bound is
close to what is needed."""

L21B = {
    "file": LEC,
    "old_str": "far fewer than the original 305 to 636 but",
    "new_str": "far fewer than the original 281 to 636 but",
    "why": "Section 8, for consistency with sections 2 and 10 as edited: our windows hold 280.98 (2.75 M☉, merger 3.5 s in) to 636.23 (0.85 M☉, 4 s) cycles, and the 23 to 25 leftover cycles it is compared with were computed over both 3.5 and 4 s.",
}

# ---------------------------------------------------------------- glossary
G0_NEW = gloss[0]["new_str"]
fixes = [
    (
        """    [arXiv:1512.03385](https://arxiv.org/abs/1512.03385), section 3.2: "We
    adopt residual learning to every few stacked layers. A building block is
    shown in Fig. 2." Its shortcut connections "simply perform identity
    mapping, and their outputs are added to the outputs of the stacked
    layers". Our networks: the ml4gw and aframe code.""",
        """    [arXiv:1512.03385](https://arxiv.org/abs/1512.03385), section 3.2: "We
    adopt residual learning to every few stacked layers. A building block is
    shown in Fig. 2." Section 1: "the shortcut connections simply perform
    identity mapping, and their outputs are added to the outputs of the
    stacked layers". Our networks: the ml4gw and aframe code.""",
    ),
    (
        """    [arXiv:1312.4400](https://arxiv.org/abs/1312.4400), section 3.2: "The""",
        """    [arXiv:1312.4400](https://arxiv.org/abs/1312.4400), section 3.1: "The""",
    ),
    (
        """    the gradients. The signal-only comparison of 16 October used λ = 0.01 for
    every network.""",
        """    the gradients. The signal-only comparison of 16 October used λ = 0.01 for
    every network, and none for the S4D's own parameters (its resonators'
    frequencies and damping rates, and each channel's time step).""",
    ),
    (
        """    1e-2)". Our value: run configurations.""",
        """    1e-2)". Our value: run configurations; the zero for the S4D parameters:
    the aframe training code.""",
    ),
]
fixes.append((
    """# documentation; statements about our pipeline were checked against the
# aframe and ml4gw code.""",
    """# documentation; statements on our pipeline were checked against the
# aframe and ml4gw code.""",
))
for old, new in fixes:
    assert G0_NEW.count(old) == 1, old
    G0_NEW = G0_NEW.replace(old, new)

# ---------------------------------------------------------------- assemble
final = []
why_fix = {}
for i, e in enumerate(merger):
    e = dict(e)
    if i == 0:
        e["new_str"] = D0_NEW
        e["why"] = ("#cycles: adds the 3.5 s columns, what our windows hold (with the per-detector range), and a Check; "
                    "then a new section {#oct16-merger-position} deriving where the merger falls from the pipeline "
                    "arithmetic, with its check. Checker fixes: window end defined (first sample + 4 s) so 'ends 0 to "
                    "0.4995 s after the merger' is consistent; 1.25% corrected to 1.04% (exact for an isotropic sky; "
                    "MC of 4,000,000 gives 1.039%); MC mean replaced by the exact 3.750244 s; frequency estimator stated; "
                    "'one thread' dropped; 'time-warped' (not a glossary spelling) reworded so 'time warp' links.")
    elif i == 6:
        e["new_str"] = D6_NEW
        e["why"] = e["why"] + " Checker fix: the bound is necessary, not sufficient, so 'would need at least', not 'would do'."
    elif i == 7:
        e["new_str"] = D7_NEW
        e["why"] = e["why"] + " Checker fix: 'longest windows' (all windows are 4 s) reworded."
    elif i == 8:
        e["new_str"] = D8_NEW
        e["why"] = (e["why"] + " Checker fixes: 230.40 is the value for a merger 21.0 ms after the end, not a maximum "
                    "(the largest over all positions is 238.59, the 4 s value); 'one thread' dropped; agreement stated "
                    "as better than 1e-14 (rerun gives 3.6e-15).")
    elif i == 10:
        e["new_str"] = W10_NEW
        e["why"] = e["why"] + " Checker fix: reworded to avoid saying '3.5 to 4 s' twice in one sentence."
    elif i == 11:
        e["new_str"] = W11_NEW
        e["why"] = "Adds the count for the shortest window, 105.37 to 219.48 rounded up; checker fix: 'at least', since it is a lower bound."
    elif i == 18:
        e["new_str"] = L18_NEW
        e["why"] = e["why"] + " Checker fix: 'would need at least 106 to 220', a lower bound."
    final.append(e)
    if i == 4:
        final.append(D4B)
    if i == 21:
        final.append(L21B)

g0 = dict(gloss[0])
g0["new_str"] = G0_NEW
g0["why"] = (gloss[0]["why"] + " Checker fixes: He et al.'s identity-mapping quote is from section 1, not 3.2; "
             "the Network In Network 1x1 sentence is in section 3.1 (MLP Convolution Layers), not 3.2; weight decay: "
             "the S4D kernel parameters (log_dt, log_A_real, A_imag) are in an AdamW group with weight_decay 0.0.")
final.append(g0)
final.extend(gloss[1:])

# ---------------------------------------------------------------- apply to copies and check
work = CK / "applied"
if work.exists():
    shutil.rmtree(work)
work.mkdir()
copies = {}
for f in {e["file"] for e in final}:
    dst = work / Path(f).name
    shutil.copy(CK / "snap" / Path(f).name, dst)
    copies[f] = dst
live_same = {f: open(f).read() == open(CK / "snap" / Path(f).name).read() for f in copies}
print("live files unchanged since snapshot:", live_same)

problems = []
for n, e in enumerate(final):
    orig = open(CK / "snap" / Path(e["file"]).name).read()
    if orig.count(e["old_str"]) != 1:
        problems.append(f"edit {n}: old_str occurs {orig.count(e['old_str'])} times in the original")
    live = open(e["file"]).read()
    if live.count(e["old_str"]) != 1:
        problems.append(f"edit {n}: old_str occurs {live.count(e['old_str'])} times in the live file")
    cur = copies[e["file"]].read_text()
    if cur.count(e["old_str"]) != 1:
        problems.append(f"edit {n}: old_str occurs {cur.count(e['old_str'])} times at apply time")
    copies[e["file"]].write_text(cur.replace(e["old_str"], e["new_str"], 1))
    new = e["new_str"]
    added = new.replace(e["old_str"], "") if e["old_str"] in new else new
    for bad in ["—", "–", " roughly", " about ", "Roughly", "About "]:
        if bad in added:
            problems.append(f"edit {n}: contains {bad!r}")
    # blank line before every heading inside new_str
    lines = new.split("\n")
    for j, ln in enumerate(lines):
        if ln.startswith("#") and not ln.startswith("# ") and j > 0 and lines[j - 1].strip() != "":
            problems.append(f"edit {n}: no blank line before heading {ln[:40]!r}")
        if re.match(r"^##+ ", ln) and j > 0 and lines[j - 1].strip() != "":
            problems.append(f"edit {n}: no blank line before heading {ln[:40]!r}")
    # $ balance per paragraph (ignoring $$)
    if not e["file"].endswith(".yml"):
        for para in new.split("\n\n"):
            if para.replace("$$", "").count("$") % 2:
                problems.append(f"edit {n}: odd number of $ in a paragraph: {para[:60]!r}")
print("problems:", problems or "none")

# anchors
der = copies[DER].read_text()
ids = set(re.findall(r"\{#([A-Za-z0-9_-]+)\}", der))
for f, p in copies.items():
    txt = p.read_text()
    if f == DER:
        links = re.findall(r"\]\(#([A-Za-z0-9_-]+)\)", txt)
    elif f.endswith(".md"):
        links = re.findall(r"relative_url \}\}#([A-Za-z0-9_-]+)\)", txt)
        assert "{%- assign d = '/projects/bns/derivations/' -%}" in txt, f
    else:
        links = re.findall(r"/projects/bns/derivations/#([A-Za-z0-9_-]+)\)", txt)
    missing = sorted(set(l for l in links if l not in ids))
    print(Path(f).name, "derivation links:", len(links), "missing:", missing)
print("oct16-merger-position defined once:", der.count("{#oct16-merger-position}") == 1)

# glossary parse and schema
g = yaml.safe_load(copies[GLOS].read_text())
groups = {"Detectors and data", "Gravitational waves", "Machine learning", "Project 8", "Searching for signals",
          "Signal processing", "State-space models", "Statistics", "This project"}
idl = [x["id"] for x in g]
print("glossary entries:", len(g), "unique ids:", len(set(idl)) == len(idl),
      "groups ok:", all(x["group"] in groups for x in g),
      "required keys:", all(all(k in x for k in ("id", "term", "group", "short", "source")) for x in g))
spell = {}
for x in g:
    for s in [x["term"]] + list(x.get("also") or []):
        spell.setdefault(s.lower(), set()).add(x["id"])
dups = {s: v for s, v in spell.items() if len(v) > 1}
print("spellings shared by two ids:", dups)

json.dump(final, open(CK / "final_edits.json", "w"), ensure_ascii=False, indent=1)
print("edits:", len(final))
