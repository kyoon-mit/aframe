"""Build the proposed edits (v2), check each old_str occurs exactly once in the
current site files, apply them to copies in patched_v2/, write edits_v2.json.

Reads the site files only; the site repository is not modified.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SITE = "/home/kyoon/SSM-BNS/WEB/kyoon-mit.github.io/docs"
DER = f"{SITE}/projects/bns/derivations.md"
ART = f"{SITE}/_updates/2026-10-15-oscillators-and-chirps.md"
GLO = f"{SITE}/_data/glossary.yml"

new_section = open(os.path.join(HERE, "new_section_v2.md")).read()
assert new_section.endswith("\n") and not new_section.endswith("\n\n")

L = "{{ d | relative_url }}#oct16-nonlinearity-linear-part"
SC = "{{ d | relative_url }}#semi-coherent"
E = []

# 1. derivations: the new section, before the heading of #oct16-s4d-fft-offset
H512 = "## The 512 Hz bin and the GPU offset {#oct16-s4d-fft-offset}"
E.append(dict(file=DER, old_str=H512, new_str=new_section + "\n" + H512,
              why="New section {#oct16-nonlinearity-linear-part}, directly after {#oct16-s4d-gelu-linear} and before the heading of {#oct16-s4d-fft-offset}; the line before the old heading is already blank, and the new section ends with a blank line before it."))

# 2. derivations: #oct16-s4d-gelu-linear
E.append(dict(file=DER,
    old_str=("[$N^{1/4}$ section](#semi-coherent), which lists a GELU among the\n"
             "nonlinearities that turn a sum into a sum of powers: that holds for its even\n"
             "part only. It holds exactly for the noisy one-layer networks, which square\n"
             "each channel."),
    new_str=("[$N^{1/4}$ section](#semi-coherent), which listed a GELU among the\n"
             "nonlinearities that turn a sum into a sum of powers: that holds for its even\n"
             "part only. It holds exactly for the noisy one-layer networks, which square\n"
             "each channel. The even part adds noise but no response linear in the\n"
             "signal, so for pieces with independent Gaussian noise a sum in step after a\n"
             "GELU keeps the fraction 0.838 to 0.910 of the SNR at noise spreads 0.5 to 2\n"
             "([next section](#oct16-nonlinearity-linear-part))."),
    why="{#oct16-s4d-gelu-linear} agrees with the new section (odd part exactly y/2, gain 1/2, even part adds no linear response), but it says the N^{1/4} section 'lists' a GELU, which that section no longer does since its 16 October correction, and it does not say that gain 1/2 is not an SNR factor of 1/2; this changes 'lists' to 'listed' and adds the SNR fraction with a link."))

# 3. derivations: #semi-coherent, noise-only spread
E.append(dict(file=DER,
    old_str=("as powers, $P = \\sum_k z_k^2$ has mean $N + \\rho^2$ and standard deviation\n"
             "about $\\sqrt{2N}$, so the signal stands out by $\\rho^2/\\sqrt{2N}$."),
    new_str=("as powers, $P = \\sum_k z_k^2$ has mean $N + \\rho^2$ and, in noise alone,\n"
             "standard deviation $\\sqrt{2N}$, so the signal stands out by $\\rho^2/\\sqrt{2N}$."),
    why="Precision, tied to the SNR definition of the new section (mean shift over the spread in noise alone): with no signal the standard deviation of P is exactly sqrt(2N), not 'about' (with the signal it is sqrt(2N + 4 rho^2)). Independent of the other edits; skip it if only the nonlinearity wording is wanted."))

# 4. derivations: #semi-coherent, the correction note
E.append(dict(file=DER,
    old_str=("applied to the pieces before they are added (a square, an absolute value)\n"
             "turns the sum into this kind. (Corrected 16 October: an earlier version also\n"
             "listed a threshold and a GELU. Those keep a part linear in a weak signal and\n"
             "cost only a constant factor, a GELU passing the linear response with gain 1/2;\n"
             "see [the GELU section](#oct16-s4d-gelu-linear).)"),
    new_str=("applied to the pieces before they are added (a square, an absolute value, an\n"
             "energy) turns the sum into this kind. (Corrected 16 October: an earlier\n"
             "version also listed a threshold and a GELU. A GELU and a threshold at zero\n"
             "are not even: they keep a part linear in a weak signal, the GELU with gain\n"
             "1/2 ([GELU section](#oct16-s4d-gelu-linear)), and pieces added in step after\n"
             "them keep most of the SNR. For the 600 pieces above at SNR 8 the exact SNR\n"
             "of the sum is 7.737 after a GELU and 6.271 after a threshold at zero,\n"
             "against 8 without them and 1.848 after a square\n"
             "([section on nonlinear functions](#oct16-nonlinearity-linear-part)). A\n"
             "threshold on the size $\\lvert z_k\\rvert$ is even and does turn the sum into\n"
             "this kind.)"),
    why="{#semi-coherent} agrees with the new section on even functions (its rho^2/sqrt(2N) is r2 = 1 for the square), but its correction note says 'a threshold' keeps a linear part, which holds for a threshold at zero and not for a threshold on |z| (even, r = 0), and 'cost only a constant factor' gives no number; for its own pieces (one sign, so equal weights are in step) the new section gives exact values: 7.737 (GELU), 6.271 (threshold at zero), 1.848 (square) against 8."))

# 5. article: main text of the first obstacle
E.append(dict(file=ART,
    old_str=("adding in step. It does not: a GELU passes a weak signal with an average\n"
             "gain of one half, and a sum in step taken after it keeps 84 to 91% of the\n"
             "SNR for noise spreads of 0.5 to 2 (a ReLU, the function $\\max(0, x)$,\n"
             "keeps 86%, a threshold at zero 80%). Only an even function, such as a\n"
             "square or an energy, turns the sum into a sum of powers (our derivation and\n"
             "Monte Carlo check, 15 October; the sentence of the\n"
             "[derivation of the power-sum loss]({{ d | relative_url }}#semi-coherent)\n"
             "that lists a threshold and a GELU beside a square is wrong)."),
    new_str=("adding in step. It does not: a GELU passes a weak signal with an average\n"
             "gain of one half, and a sum in step taken after it keeps 84 to 91% of the\n"
             "SNR it would have without the GELU, when the pieces added carry independent\n"
             "Gaussian noise with a spread of 0.5 to 2 (a ReLU, the function\n"
             "$\\max(0, x)$, keeps 86%, and a threshold at zero 80%, at any spread). Only\n"
             "an even function, such as a square or an energy, loses the part linear in\n"
             "the signal at every spread, and what it leaves is a sum of powers\n"
             f"([derivation]({L}); the\n"
             f"[derivation of the power-sum loss]({SC}),\n"
             "which listed a GELU beside a square, has been corrected)."),
    why="Links the numbers to the new section. Keeps 84 to 91% (r = 0.838159 to 0.909548 for 0.5 <= sigma <= 2, minimum at 1.678), 86% (0.856429) and 80% (0.797885); adds the hypothesis of the analysis (independent Gaussian noise per piece) and that the ReLU and threshold values hold at any spread; replaces 'only an even function turns the sum into a sum of powers' by what is proved (only even functions lose the linear part at every spread; x^3 - 3x loses it at sigma = 1 alone); the #semi-coherent sentence was corrected on 16 October, so 'is wrong' is out of date; avoids a bare 'threshold' after 'threshold at zero', which the glossary script would link to 'detection threshold'."))

# 6. article: objection
E.append(dict(file=ART,
    old_str=("For Gaussian noise of any spread, a sum taken after the function keeps "
             "between 84 and 100% of the SNR it would have without it (85% at unit spread; "
             "we computed the moments by numerical integration)."),
    new_str=("For Gaussian noise of any spread, independent from piece to piece, a sum in "
             "step taken after the function keeps between 84 and 100% of the SNR it would "
             f"have without it (85% at unit spread; [derivation]({L}))."),
    why="Bare numbers with only 'we computed the moments by numerical integration'; the derivation gives r between 0.838159 (sigma = 1.678) and 1 at any spread, 0.850463 at sigma = 1, for a sum in step over pieces with independent noise; without 'in step' the statement is false for an oscillating signal added with equal weights (a power sum)."))

# 7. article: rebuttal
E.append(dict(file=ART,
    old_str="(85% at unit spread; we checked by numerical integration)",
    new_str=f"(85% at unit spread; [derivation]({L}))",
    why="Rebuttal quotes 84 to 100% and 85% with only 'we checked by numerical integration'; link to the derivation, whose check reruns that integration."))

# 8. article: verdict
E.append(dict(file=ART,
    old_str=("and a sum in step taken after it keeps 84 to 100% of the SNR (checked "
             "numerically); the single-layer rows have no such step."),
    new_str=("and a sum in step taken after it, over pieces with independent noise, keeps "
             f"84 to 100% of the SNR ([derivation]({L})); the single-layer rows have no "
             "such step."),
    why="Verdict quotes 84 to 100% with 'checked numerically' only; link it and state the independence hypothesis."))

# 9. article: note added 16 October
E.append(dict(file=ART,
    old_str=("Added 16 October: a separate derivation confirms the first objection (a sum "
             "in step after a GELU keeps 84 to 91% of the SNR for noise spreads of 0.5 to "
             "2, and tends to 100% as the spread falls; after a ReLU 86%, after a threshold "
             "at zero 80%; only even functions force a sum of powers), and the sentence it "
             "quotes has been rewritten;"),
    new_str=(f"Added 16 October: a separate [derivation]({L}) confirms the first objection "
             "(a sum in step after a GELU keeps 84 to 91% of the SNR for noise spreads of "
             "0.5 to 2, and tends to 100% as the spread falls; after a ReLU 86%, after a "
             "threshold at zero 80%; only even functions lose the linear part at every "
             "spread, leaving a sum of powers), and the sentence it quotes has been "
             f"rewritten, as has the [derivation of the power-sum loss]({SC});"),
    why="'A separate derivation' had no link; 'only even functions force a sum of powers' becomes the proved statement; records that #semi-coherent was corrected too."))

# 10. glossary: threshold at zero
GN = "- id: group-norm\n  term: group norm\n"
gl_new = (
    "- id: threshold-at-zero\n"
    "  term: threshold at zero\n"
    "  also: [step function, Heaviside function, Heaviside step function]\n"
    "  group: Machine learning\n"
    "  short: >-\n"
    "    The function that is 1 where its input is above zero and 0 elsewhere,\n"
    "    the Heaviside step function. The sign function, +1 above zero and −1\n"
    "    below, is the same up to a scale and a shift. A threshold on the size of\n"
    "    x (1 where x is above a level or below minus that level) is instead an\n"
    "    even function of x.\n"
    "  source: >-\n"
    "    NIST Digital Library of Mathematical Functions, equation 1.16.13\n"
    "    ([dlmf.nist.gov/1.16.iv](https://dlmf.nist.gov/1.16.iv)): H(x) = 1 for\n"
    "    x > 0 and H(x) = 0 for x ≤ 0.\n"
    "\n"
)
E.append(dict(file=GLO, old_str=GN, new_str=gl_new + GN,
    why="'threshold at zero' (article section 5 and the new section) has no entry, and the bare word 'threshold' is an alias of 'detection threshold', so on the article page, where the first 'threshold' is 'a threshold at zero 80%', the script would link it to 'detection threshold'. As the longer form at the same start it now links here. Inserted after the GELU entry."))

texts, ok = {}, True
for e in E:
    f = e["file"]
    texts.setdefault(f, open(f).read())
    n = texts[f].count(e["old_str"])
    print(f"{os.path.basename(f):40s} occurrences {n}: {e['old_str'][:70]!r}")
    if n != 1:
        ok = False
        continue
    texts[f] = texts[f].replace(e["old_str"], e["new_str"], 1)
if not ok:
    sys.exit("an old_str does not occur exactly once")
out = os.path.join(HERE, "patched_v2")
os.makedirs(out, exist_ok=True)
for f, t in texts.items():
    open(os.path.join(out, os.path.basename(f)), "w").write(t)
json.dump(E, open(os.path.join(HERE, "edits_v2.json"), "w"), indent=1, ensure_ascii=False)
print("ok; patched copies in", out)
