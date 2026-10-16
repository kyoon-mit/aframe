"""Build the caption, alt-text and methods edits, check each old_str occurs once, apply them to copies, and dump JSON."""
import json
import shutil
import sys
from pathlib import Path

SITE = Path("/home/kyoon/SSM-BNS/WEB/kyoon-mit.github.io/docs")
PAGE = SITE / "_updates/2026-10-16-what-s4d-computes.md"
DERIV = SITE / "projects/bns/derivations.md"
SIM = Path(sys.argv[1])

E = []

E.append((PAGE,
 'alt="Match against the number of resonators divided by the integral of tau df, for four chirp masses; all curves jump from below 0.5 to above 0.85 near 1"',
 'alt="Match against the number of resonators divided by the integral of tau df, for chirp masses 1.0, 1.2, 1.5 and 2.0 solar masses; the 1.2, 1.5 and 2.0 curves jump from below 0.5 to above 0.85 as the ratio passes 1; the 1.0 curve, counted over both signs of frequency, jumps from 0.50 to 0.78 near 1 and reaches 0.97 at 1.6"',
 "Figure 1 alt text: the old 'all curves jump ... to above 0.85 near 1' is false for 1.0 M☉, which the figure draws at 0.4998 (x 0.798), 0.7774 (0.958), 0.8030 (1.117), 0.8305 (1.277), 0.9672 (1.596); 1.2/1.5/2.0 go 0.4673/0.4569/0.4801 to 0.9022/0.8606/0.9321."))

E.append((PAGE,
 "caption=\"Resonators laid along the chirp, one per band of equal ∫τ df, ring time τ(f), only the weights solved, evaluated in float32. Horizontal axis: the number of resonators p over ∫τ df (the figure's 'poles per hold-time cell'). Left: match; right: 1 − match. For 1.0 M☉ (dashed) the count is summed over both signs of frequency.\"",
 "caption=\"Resonators laid along the chirp, one per band of equal ∫τ df, ring time τ(f), only the weights solved (the figure's 'no fit'), evaluated in float32. Horizontal axis: the number of resonators over ∫τ df on the dominant sign of frequency (the figure's 'resonators per hold-time slot'). Left: match; right: 1 − match. For 1.0 M☉ (dashed, open circles) the count is summed over both signs of frequency.\"",
 "Figure 1 caption: the replot dropped the symbol p and renamed 'poles per hold-time cell' to 'resonators per hold-time slot'; its title says 'no fit' where the caption says only the weights are solved (same series, the derivation's 'Along the chirp, no fit'), so the caption names it; 1.0 M☉ is the dashed line with open circles."))

E.append((PAGE,
 'alt="Match against number of resonators for every fitting method at chirp masses 1.0, 1.2, 1.5 and 2.0 solar masses"',
 'alt="Match (top row) and 1 − match (bottom row, log scale) against the number of resonators for every fitting method, at chirp masses 1.0, 1.2, 1.5 and 2.0 solar masses"',
 "Figure 2 alt text: the figure has two rows, match on top and 1 − match on a log scale below; the old alt text named only the match."))

E.append((PAGE,
 "caption=\"Grey squares: the earlier matrix-pencil fits, growing poles reflected; they level off at 0.88 to 0.92. Green triangles: the same poles clipped to |z| = 1. Orange: resonators along the chirp with no fit (figure 1). Yellow: the strongest Fourier tones. Blue line and stars ('VarPro'): all parameters fitted, valid in float32. Open circles and pink: fits valid only in float64, or with growing poles; not usable in S4D. Vertical lines: 32 resonators, and ∫τ df for each kernel.\"",
 "caption=\"Top row: match; bottom row: 1 − match. Grey dashed lines: the earlier matrix-pencil fits, with growing resonators reflected (open squares; they level off at 0.88 to 0.92) or clipped to |z| = 1 and not refitted (filled triangles). Grey dash-dot line, open triangles: the same fits with growing resonators allowed, not usable in S4D. Dotted line, diamonds: the strongest Fourier tones (the figure's 'Fourier components'). Thin dark line, small dots: resonators along the chirp, only the weights solved (figure 1). Thick navy line (the figure's 'optimized decaying resonators'): all parameters fitted by VarPro from three fixed starts (the two matrix-pencil fits and the layout along the chirp), the best result that holds in float32. Navy stars: the same from 8 to 24 random starts. Open navy circles: the same fits in float64 alone; above the thick line they do not hold in float32 and are not usable in S4D. Vertical lines: 32 resonators, and ∫τ df over the dominant sign of frequency.\"",
 "Figure 2 caption: no green, orange, yellow, blue or pink is drawn any more (replot.py fig_fits: refl INK3 dashed open squares; clip INK3 dashed filled triangles; grow INK3 dash-dot open down-triangles; four INK2 dotted filled diamonds; chirp INK thin solid small dots; vp ACCENT thick solid; rand ACCENT stars; f64 ACCENT open circles); the legend says 'Fourier components', 'optimized decaying resonators', '3 fixed starts (2 matrix pencil, 1 chirp), best step valid in float32', 'random starts (best of 8 to 24)', 'match in float64 only (best of 3 starts)', and the vertical-line label says 'dominant sign of frequency'. Values checked in summary.json: reflected at 2,048 = 0.8754 to 0.9166."))

E.append((PAGE,
 'alt="Ring time of every resonator against the time before merger at its frequency, four S4D networks, with the line ring time equal to tau"',
 'alt="Ring time of each resonator between 20 Hz and the Nyquist frequency against the time before merger at its frequency for chirp mass 1.2 solar masses, four S4D networks, with lines where ring time equals tau for chirp masses 0.85, 1.2 and 2.75 solar masses"',
 "Figure 3 alt text: the panels show only resonators between 20 Hz and Nyquist (e.g. 671 of 1,024), the x axis is tau at 1.2 M☉, and there are now three ring-time-equals-tau lines (1.2 solid, 0.85 and 2.75 dashed)."))

E.append((PAGE,
 "caption=\"Each dot is one resonator between 20 Hz and the Nyquist frequency: its ring time against τ(f) at its frequency for 1.2 M☉, coloured by its share of its channel's energy; grey, the same network at a fresh start. Orange line and band: ring time equal to τ (0.85 to 2.75 M☉). Green: the in-step time. Dots near the orange line could hold their note until the merger. 'Tones' in the panel titles are resonators.\"",
 "caption=\"Each dot is one resonator between 20 Hz and the Nyquist frequency: its ring time against τ(f) at its frequency for 1.2 M☉, shaded by its share of its channel's energy, darker for a larger share; light grey, the same network at a fresh start. Solid red line: ring time equal to τ(f) for 1.2 M☉; dashed red lines: the same for 0.85 M☉ (upper) and 2.75 M☉ (lower). Dash-dot line: the in-step time for 1.2 M☉. Dots near the solid red line could hold their note until the merger.\"",
 "Figure 3 caption: the orange line and band are now a solid red line (--warn, 1.2 M☉) and two dashed red lines (0.85 above, 2.75 below, since τ scales as (1.2/Mc)^(5/3)); the in-step time is a dark navy (INK) dash-dot line, not green; trained dots run from grey-blue (INK3, share 1e-5) to dark navy (INK, share 1) on a log scale, fresh-start dots are light grey (RULE2); panel titles now say 'resonators', so the 'Tones' sentence is dropped."))

E.append((DERIV,
 "[doi:10.1137/0710036](https://doi.org/10.1137/0710036)), best of 24\n  random starts,\n  keeping only results whose match holds when the kernel is evaluated in\n  float32, the precision S4D computes in. The kept 32-resonator fits have\n  cancellation ratios of 12 to 24.",
 "[doi:10.1137/0710036](https://doi.org/10.1137/0710036)), from three\n  fixed starts (the two matrix-pencil fits and the layout along the chirp)\n  and from random starts (24 at 32 resonators, 8 at 64 and 128), keeping\n  only results whose match holds when the kernel is evaluated in float32,\n  the precision S4D computes in; the claim gives the best over all starts.\n  The kept 32-resonator fits have cancellation ratios of 12 to 24.",
 "Methods for figure 2 (linked from the page as 'methods'): figure 2 draws three fixed starts (thick line) and 'random starts (best of 8 to 24)' (stars). The raw restart summaries hold 24 seeds at 32 resonators (seeds 0-7 and 8-23) and 8 at 64 and 128 (seeds 0-7); the claim's 64- and 128-resonator values (0.389 ... 0.942) are the fixed-start results, which beat the random starts there, and the 32-resonator values are the random-start best. 'best of 24 random starts' is true only at 32."))

bad = []
copies = {}
for path, old, new, why in E:
    text = path.read_text()
    n = text.count(old)
    if n != 1:
        bad.append((path.name, n, old[:60]))
    for s in (new,):
        if "—" in s or "–" in s:
            bad.append(("dash", path.name, s[:60]))
    if path == PAGE and new.count('"') != 2:
        bad.append(("quotes", new[:60]))
assert not bad, bad
for path in {p for p, *_ in E}:
    dst = SIM / path.name
    shutil.copy(path, dst)
    copies[path] = dst
for path, old, new, why in E:
    dst = copies[path]
    t = dst.read_text()
    assert t.count(old) == 1
    dst.write_text(t.replace(old, new))
json.dump([{"file": str(p), "old_str": o, "new_str": n, "why": w} for p, o, n, w in E],
          open(SIM / "edits.json", "w"), ensure_ascii=False, indent=1)
print("all", len(E), "old_str found exactly once; copies edited in", SIM)
