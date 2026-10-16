"""Build the proposed edits, check each old_str occurs once, apply them to copies.

Reads the site files (read only) and writes patched copies and edits.json into
this folder; the site repository is not modified.
"""
import json
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SITE = "/home/kyoon/SSM-BNS/WEB/kyoon-mit.github.io/docs"
DER = f"{SITE}/projects/bns/derivations.md"
ART = f"{SITE}/_updates/2026-10-15-oscillators-and-chirps.md"
GLO = f"{SITE}/_data/glossary.yml"

new_section = open(os.path.join(HERE, "new_section.md")).read()
assert new_section.endswith("\n")

L = "{{ d | relative_url }}#oct16-nonlinearity-linear-part"
SC = "{{ d | relative_url }}#semi-coherent"

E = []

# 1. derivations: new section before #oct16-s4d-fft-offset
H512 = "## The 512 Hz bin and the GPU offset {#oct16-s4d-fft-offset}"
E.append(dict(file=DER, old_str=H512, new_str=new_section + "\n" + H512,
              why="New derivation section {#oct16-nonlinearity-linear-part}, inserted directly after {#oct16-s4d-gelu-linear} and before the heading of {#oct16-s4d-fft-offset} (blank line kept before both headings)."))

# 2. derivations: #oct16-s4d-gelu-linear, tense and pointer to the new section
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
                       "GELU keeps the fraction 0.84 to 0.91 of the SNR at noise spreads 0.5 to 2\n"
                       "([next section](#oct16-nonlinearity-linear-part))."),
              why="{#oct16-s4d-gelu-linear} agrees with the new section (gain 1/2, odd part exactly y/2) but says the N^{1/4} section 'lists' a GELU, which it no longer does since its 16 October correction, and it gives no SNR cost for the even part; this changes 'lists' to 'listed' and adds the cost with a link."))

# 3. derivations: #semi-coherent correction note made precise
E.append(dict(file=DER,
              old_str=("applied to the pieces before they are added (a square, an absolute value)\n"
                       "turns the sum into this kind. (Corrected 16 October: an earlier version also\n"
                       "listed a threshold and a GELU. Those keep a part linear in a weak signal and\n"
                       "cost only a constant factor, a GELU passing the linear response with gain 1/2;\n"
                       "see [the GELU section](#oct16-s4d-gelu-linear).)"),
              new_str=("applied to the pieces before they are added (a square, an absolute value, an\n"
                       "energy, a threshold on $\\lvert x\\rvert$) turns the sum into this kind\n"
                       "([nonlinearity section](#oct16-nonlinearity-linear-part)). (Corrected 16\n"
                       "October: an earlier version also listed a threshold and a GELU. A threshold\n"
                       "at zero and a GELU are not even. They keep a part linear in a weak signal,\n"
                       "the GELU with gain 1/2 ([GELU section](#oct16-s4d-gelu-linear)), and pieces\n"
                       "added in step after them keep a fixed fraction of the SNR: 0.80 after a\n"
                       "threshold at zero, 0.84 to 0.91 after a GELU at noise spreads 0.5 to 2.)"),
              why="{#semi-coherent} agrees on even functions (square: r2 = 1 gives its rho^2/sqrt(2N)), but its correction note says 'a threshold' keeps a linear part, which is true only for a one-sided threshold (a threshold on |x| is even and gives a power sum), omits that the constant factor needs weights in step, and gives no factor; this states which threshold, the in-step condition and the factors, with links."))

# 4. article: main text of the first obstacle
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
                       "gain of one half, and a sum in step taken after it, over pieces with\n"
                       "independent Gaussian noise, keeps 84 to 91% of the SNR for noise spreads of\n"
                       "0.5 to 2 (a ReLU, the function $\\max(0, x)$, keeps 86% and a threshold at\n"
                       "zero 80%, at any spread). Only an even function, such as a square or an\n"
                       "energy, loses the part linear in the signal at every noise spread, and what\n"
                       "it leaves is a sum of powers\n"
                       f"([derivation]({L}); the\n"
                       f"[derivation of the power-sum loss]({SC}),\n"
                       "which listed a GELU beside a square, was corrected on 16 October)."),
              why="Links the numbers to the new section; adds the analysis's hypothesis (independent Gaussian noise per piece) and that the ReLU and threshold values hold at any spread; replaces 'only an even function turns the sum into a sum of powers' by what is proved (only even functions lose the linear part at every spread; x^3 - 3x loses it at sigma = 1 alone); the #semi-coherent sentence was corrected on 16 October, so 'is wrong' is out of date; drops the bare word 'threshold', which the glossary script would link to 'detection threshold'."))

# 5. article: objection
E.append(dict(file=ART,
              old_str=("For Gaussian noise of any spread, a sum taken after the function keeps "
                       "between 84 and 100% of the SNR it would have without it (85% at unit spread; "
                       "we computed the moments by numerical integration)."),
              new_str=("For Gaussian noise of any spread, independent from piece to piece, a sum in "
                       "step taken after the function keeps between 84 and 100% of the SNR it would "
                       f"have without it (85% at unit spread; [derivation]({L}))."),
              why="Objection quotes 84 to 100% and 85% without a source; the derivation gives r between 0.838159 (sigma = 1.678) and 1 at any spread, 0.850463 at sigma = 1, for a sum in step over pieces with independent noise (a plain sum of an oscillation gives a power sum instead)."))

# 6. article: rebuttal
E.append(dict(file=ART,
              old_str="(85% at unit spread; we checked by numerical integration)",
              new_str=f"(85% at unit spread; [derivation]({L}))",
              why="Rebuttal quotes 84 to 100% and 85% with only 'we checked by numerical integration'; link to the derivation, whose check reruns that integration."))

# 7. article: verdict
E.append(dict(file=ART,
              old_str=("and a sum in step taken after it keeps 84 to 100% of the SNR (checked "
                       "numerically); the single-layer rows have no such step."),
              new_str=("and a sum in step taken after it, over pieces with independent noise, keeps "
                       f"84 to 100% of the SNR ([derivation]({L})); the single-layer rows have no "
                       "such step."),
              why="Verdict quotes 84 to 100% with 'checked numerically' only; link it and state the independence hypothesis."))

# 8. article: note added 16 October
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
              why="'A separate derivation' had no link; 'only even functions force a sum of powers' is replaced by the proved statement; records that #semi-coherent was corrected too."))

# 9. glossary: ReLU and step function (threshold at zero)
GN = "- id: group-norm\n  term: group norm\n"
gl_new = (
    "- id: relu\n"
    "  term: ReLU\n"
    "  case: true\n"
    "  also: [ReLUs, rectified linear unit]\n"
    "  group: Machine learning\n"
    "  short: >-\n"
    "    The function max(0, x): positive values pass unchanged and negative ones\n"
    "    become zero. A common nonlinear function between the layers of a neural\n"
    "    network; the GELU is a smooth version of it.\n"
    "  source: >-\n"
    "    PyTorch 2.10 documentation of `torch.nn.ReLU`: \"Applies the rectified\n"
    "    linear unit function element-wise\", ReLU(x) = max(0, x). Hendrycks and\n"
    "    Gimpel (2016), [arXiv:1606.08415](https://arxiv.org/abs/1606.08415),\n"
    "    abstract: the GELU \"weights inputs by their value, rather than gates\n"
    "    inputs by their sign as in ReLUs\".\n"
    "\n"
    "- id: step-function\n"
    "  term: step function\n"
    "  also: [threshold at zero, step at zero, Heaviside step, Heaviside function]\n"
    "  group: Machine learning\n"
    "  short: >-\n"
    "    The function that is 1 where its input is above zero and 0 elsewhere,\n"
    "    also called a threshold at zero. The sign function, +1 above zero and\n"
    "    −1 below, is the same up to a scale and a shift. A threshold on the\n"
    "    size of x instead is an even function of x.\n"
    "  source: >-\n"
    "    NIST Digital Library of Mathematical Functions, equation 1.16.13\n"
    "    ([dlmf.nist.gov/1.16.iv](https://dlmf.nist.gov/1.16.iv)), the Heaviside\n"
    "    function: H(x) = 1 for x > 0 and H(x) = 0 for x ≤ 0.\n"
    "\n"
)
E.append(dict(file=GLO, old_str=GN, new_str=gl_new + GN,
              why="ReLU is new to the site (first used in the article and the new section) and has no glossary entry. 'threshold at zero' would otherwise be auto-linked to 'detection threshold' (also: [threshold]); as the longer form at the same start it now links to its own entry. Check first that a parallel glossary task has not added id relu."))

# apply to copies
texts = {}
ok = True
for e in E:
    f = e["file"]
    if f not in texts:
        texts[f] = open(f).read()
    n = texts[f].count(e["old_str"])
    print(f"{os.path.basename(f):40s} occurrences {n}: {e['old_str'][:60]!r}")
    if n != 1:
        ok = False
        continue
    texts[f] = texts[f].replace(e["old_str"], e["new_str"], 1)
if not ok:
    sys.exit("an old_str does not occur exactly once")
outdir = os.path.join(HERE, "patched")
os.makedirs(outdir, exist_ok=True)
for f, t in texts.items():
    open(os.path.join(outdir, os.path.basename(f)), "w").write(t)
json.dump(E, open(os.path.join(HERE, "edits.json"), "w"), indent=1, ensure_ascii=False)
print("ok; patched copies in", outdir)
