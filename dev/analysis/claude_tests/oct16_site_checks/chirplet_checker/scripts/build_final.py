"""Build the corrected edits from the drafter's v3 edits by exact, asserted replacements."""
import json
import sys

SP = "/tmp/claude-1024/-home-kyoon-SSM-BNS/82effc75-bb98-44ad-af20-e41e39a7258d/scratchpad/oct16fix"
E = json.load(open(SP + "/chirplet/v3_dryrun/edits.json"))


def rep(s, old, new):
    n = s.count(old)
    assert n == 1, (n, old[:80])
    return s.replace(old, new)


# ---------------- brainstorm item ----------------
b = E[0]["new_str"]
b = rep(b, "(Mann and Haykin 1995, <i>IEEE Trans. Signal Process.</i> 43, 2745,",
        "(Mann and Haykin 1995, IEEE Trans. Signal Process. 43, 2745,")
b = rep(b, "At leading order, pieces of linear chirp follow our chirps to within an eighth of a cycle with 10 to 12 per window up to 1024 Hz,",
        "At leading order, 10 to 12 pieces of linear chirp per 4 s window follow our chirps to within an eighth of a cycle up to 1024 Hz,")
b = rep(b, "at least as many as the bank through the same ending",
        "at least as many as the 256-template bank through the same ending")
b = rep(b, "12 chirplets keep 0.98 to 0.99 of the SNR; laid along a leading-order chirp with no fit, they match the same three at 0.79 to 0.82, where free kernels start at 0.10 and reach at most 0.81 after training (<a href=\"{{ '/projects/bns/derivations/' | relative_url }}#oct16-chirplet-count\">derivation</a>).",
        "12 chirplets keep 0.98 to 0.99 of the matched filter's SNR; laid along a leading-order chirp with no fit, they match the same three at 0.79 to 0.82 (<a href=\"{{ '/projects/bns/derivations/' | relative_url }}#oct16-chirplet-count\">derivation</a>), where the free kernels started at 0.10 and reached at most 0.81 after training (<a href=\"{{ u | append: '2026-10-13-learned-long-kernels/' | relative_url }}\">13 Oct</a>).")
b = rep(b, "and a smaller drop than theirs on test noise from the second half of O3",
        "and a smaller drop than the free kernels' on test noise from the second half of O3")
b = rep(b, "which training hardly moved; without noise,",
        "which training hardly moved (<a href=\"{{ '/projects/bns/derivations/' | relative_url }}#oct16-s4d-channel-analysis\">measured</a>); without noise,")
b = rep(b, "against 0.95 to 0.96 from a start along the chirp.",
        "against 0.955 and 0.962 from a start along the chirp.")

# ---------------- derivation section ----------------
d = E[1]["new_str"]
d = rep(d, "([3/8 section](#oct16-s4d-three-eighths)). Over the last 4 s before the\nmerger:\n",
        "([3/8 section](#oct16-s4d-three-eighths)). Over the last 4 s before the\n"
        "merger, rounded up to whole pieces, that is 10 to 12 linear pieces up to\n"
        "1024 Hz, against at least 114 to 234 resonators:\n")
d = rep(d, "(Mann and Haykin 1995,\n*IEEE Trans. Signal Process.* 43, 2745,\n",
        "(Mann and Haykin 1995,\nIEEE Trans. Signal Process. 43, 2745,\n")
d = rep(d, "- *Cut pieces*, each the tangent linear chirp at its midpoint: 9 to 12 of\n  them keep 0.958 to 0.959 (up to 512, 900 or 1024 Hz); steady pieces made\n  the same way, 24 to 34 of them, keep 0.973.\n",
        "- *Cut pieces*, each the tangent linear chirp at its midpoint in time: 9\n  to 12 of them keep 0.958 to 0.959 (up to 512, 900 or 1024 Hz); steady\n  pieces made the same way, 24 to 34 of them, keep 0.973.\n")
d = rep(d, "- *Chirplets along the chirp, no fit:* on the leading-order chirps of the\n  resonator fits (20 to 900 Hz), 9 to 10 chirplets reach a match of 0.9, 10\n  to 11 reach 0.95 and 14 reach 0.99, where for 0.9 the same chirps need 112\n  to 219 resonators and 442 to 800 Fourier tones\n",
        "- *Chirplets along the chirp, amplitudes solved:* on the leading-order\n  chirps of the resonator fits (20 to 900 Hz), 9 to 10 chirplets reach a\n  match of 0.9, 10 to 11 reach 0.95 and 14 reach 0.99, where for 0.9 the\n  same chirps need 112 to 219 resonators and 442 to 800 Fourier tones\n")
d = rep(d, "g(t) = A\\,e^{-(t - t_k)^2/2\\sigma^2}\\exp",
        "g(t) = A\\,e^{-(t - t_k)^2/(2\\sigma^2)}\\exp")
d = rep(d, "amplitude $A$ and phase $\\phi$. Two sets of\nchirps:\n",
        "amplitude $A$ and phase $\\phi$. Two sets of\ntargets:\n")
d = rep(d, "with the chirp's frequency and rate at each centre and a width that\n  is a fixed fraction of the local spacing in time (0.3, 0.4, 0.5, 0.6, 0.8\n  or 1.0, the best kept for each $P$; it is 0.4 from 10 to 16 chirplets).\n  Only the complex amplitudes are solved, by least squares. Taking instead\n  the chirp's own amplitude and phase at each centre, with nothing solved,\n  needs the same numbers of chirplets for 0.9 and 0.95, and 14 to 15 for\n  0.99.\n",
        "with the chirp's frequency and rate at each centre and a width\n  $\\sigma$ that is a fixed fraction of the local spacing in time (0.3, 0.4,\n  0.5, 0.6, 0.8 or 1.0, the best kept for each $P$; it is 0.4 from 10 to 16\n  chirplets). Only the complex amplitudes are solved, by least squares.\n  Taking instead the chirp's own amplitude and phase at each centre, with\n  nothing solved, needs the same numbers of chirplets for 0.9 and 0.95, and\n  14 to 15 for 0.99.\n")
d = rep(d, "the match is the largest\n  over the kernel's lag, both signs of frequency and the masses from 5%\n  below to 3% above the kernel's chirp mass. For the second number only the\n  complex amplitudes are re-solved, by least squares, at the best mass and\n  lag.\n",
        "the match is the largest\n  over the kernel's lag, both signs of frequency and all 256 masses (the\n  best lies 0.8% to 1.5% below the kernel's chirp mass). For the second\n  number only the complex amplitudes are re-solved, by least squares, at\n  the best mass and lag.\n")
d = rep(d, "$\\pi/4$, for $|s| \\le (3/4\\ddot f)^{1/3}$: a total time\n$T_{\\rm lin} = 2\\,(3/4\\ddot f)^{1/3} = (6/\\ddot f)^{1/3}$.",
        "$\\pi/4$, for $|s| \\le \\big(3/(4\\ddot f)\\big)^{1/3}$: a total time\n$T_{\\rm lin} = 2\\,\\big(3/(4\\ddot f)\\big)^{1/3} = (6/\\ddot f)^{1/3}$.")
d = rep(d, "The interval\n  where the two differ by at most $\\pi/4$ has length $(6/\\ddot f)^{1/3}$ to\n  within 0.02% to 0.49% for 0.85, 1.2 and 2.75 M☉ at 50, 100 and 300 Hz;",
        "The interval\n  where the two differ by at most $\\pi/4$ is 0.02% to 0.49% shorter than\n  $(6/\\ddot f)^{1/3}$ for 0.85, 1.2 and 2.75 M☉ at 50, 100 and 300 Hz;")
d = rep(d, "The fit's\n  exact gradient agrees with finite differences to a relative $10^{-8}$ at\n  the start along the track.",
        "The fit's\n  exact gradient agrees with central finite differences of a separately\n  written objective to a relative $2.2\\times10^{-8}$ or better in every\n  component above a thousandth of the largest, at the start along the track\n  (8 chirplets, 1.2 M☉).")
d = rep(d, "length instead of padding it, gives 0.809 to 0.838 before the amplitudes\n  are re-solved.\n",
        "length instead of padding it, gives 0.809 to 0.838 before the amplitudes\n  are re-solved.\n"
        "- A third code, written separately again to check the first two (root\n"
        "  finding for the in-step lengths, counts integrated on a grid, least\n"
        "  squares through a QR decomposition), reproduced every number in the\n"
        "  three tables, the cut-piece and chirplet matches and counts, the 800,\n"
        "  685, 557 and 442 Fourier tones, every saved fit rebuilt from its\n"
        "  parameters, and the start with no fit (0.788 to 0.824, and 0.924 to\n"
        "  0.955 with the amplitudes re-solved), taking the best over all 256\n"
        "  masses.\n")

# ---------------- glossary ----------------
g = E[2]["new_str"]
g = rep(g, """  source: >-
    Mann and Haykin, IEEE Trans. Signal Process. 43, 2745 (1995),
    [doi:10.1109/78.482123](https://doi.org/10.1109/78.482123). In the words
    of Glotin, Ricard and Balestriero 2016,
    [arXiv:1611.08749](https://arxiv.org/abs/1611.08749), section 2: "A
    chirplet can be seen as a complex sinus with increasing or decreasing
    frequency over time modulated by a Gaussian window to have a localized
    support in the time and Fourier domain." Chains: Chassande-Mottin and
    Pai, Phys. Rev. D 73, 042003 (2006),
    [arXiv:gr-qc/0512137](https://arxiv.org/abs/gr-qc/0512137), section IV:
    "CCs are signals whose (instantaneous) frequency is a broken line." Their
    chirplets are cut pieces, with no Gaussian envelope. How many cover our
    windows: the
    [derivations page](/projects/bns/derivations/#oct16-chirplet-count).""",
        """  source: >-
    Mann and Haykin, IEEE Trans. Signal Process. 43, 2745 (1995),
    [doi:10.1109/78.482123](https://doi.org/10.1109/78.482123). In the words
    of Glotin, Ricard and Balestriero 2016,
    [arXiv:1611.08749](https://arxiv.org/abs/1611.08749), section 2: "A
    chirplet can be seen as a complex sinus with increasing or decreasing
    frequency over time modulated by a Gaussian window to have a localized
    support in the time and Fourier domain." Chirplet chains (CCs):
    Chassande-Mottin and Pai, Phys. Rev. D 73, 042003 (2006),
    [arXiv:gr-qc/0512137](https://arxiv.org/abs/gr-qc/0512137), section IV:
    "CCs are signals whose (instantaneous) frequency is a broken line." Their
    chirplets are cut pieces, with no Gaussian envelope (section IV.A): "A
    chirplet is a short piece of signal whose frequency varies linearly
    between two successive nodes of the grid." How many cover our windows:
    the [derivations page](/projects/bns/derivations/#oct16-chirplet-count).""")
g = rep(g, """    A wave whose frequency changes at a constant rate, so that its phase
    grows with the square of time. Over a short enough stretch, any chirp is
    close to one.""",
        """    A wave whose frequency changes at a constant rate, so that its phase is
    a quadratic function of time. Over a short enough stretch, any smooth
    chirp is close to one.""")
g = rep(g, """  source: >-
    Mann and Haykin, IEEE Trans. Signal Process. 43, 2745 (1995),
    [doi:10.1109/78.482123](https://doi.org/10.1109/78.482123), abstract: "We
    propose the use of quadratic chirp functions (which we will call q-chirps
    for short)". How long a binary's chirp stays within an eighth of a cycle
    of one: the
    [derivations page](/projects/bns/derivations/#oct16-chirplet-count).""",
        """  source: >-
    Chassande-Mottin and Pai, Phys. Rev. D 73, 042003 (2006),
    [arXiv:gr-qc/0512137](https://arxiv.org/abs/gr-qc/0512137), section V.B:
    "a linear chirp (i.e., a chirp whose frequency is a linear function of
    time)". How long a binary's chirp stays within an eighth of a cycle of
    one: the
    [derivations page](/projects/bns/derivations/#oct16-chirplet-count).""")

out = []
whys = [
    "Adds the chirplet-kernel item to 'To try', after the Mamba item. Corrected from the draft: 0.9551 rounds to 0.96, so '0.95 to 0.96' became '0.955 and 0.962'; the free-kernel 0.10 and 0.81 now link to the 13 Oct update that measured them (the derivation link did not cover them); 'training hardly moved' links to its measurement; '10 to 12' says whole pieces; 'theirs' made unambiguous; journal not italic, as elsewhere on the site.",
    "Adds the derivation section oct16-chirplet-count (Claim, Setup, Derivation, Check) between the resonator fits and the channel analysis. Corrected from the draft: KaTeX exponents written unambiguously ((2 sigma^2), 3/(4 fddot)); the no-fit start is the best over all 256 masses (checked equal to the draft's -5%..+3% window); the in-step length is shorter by 0.02% to 0.49%; the gradient agreement is 2.2e-8, not 1e-8; 'no fit' label renamed where amplitudes are solved; rounding to whole pieces stated; a third, separate code's check added.",
    "Adds glossary entries 'chirplet' and 'linear chirp' (Signal processing) after 'in-step time'. Corrected from the draft: the linear-chirp source now quotes a definition that says what a linear chirp is (Chassande-Mottin and Pai 2006, section V.B) instead of Mann and Haykin's 'quadratic chirp functions' line; the chirplet source adds the verified section IV.A quote behind 'cut pieces'; 'phase grows with the square of time' made exact.",
]
for e, new, why in zip(E, (b, d, g), whys):
    out.append({"file": e["file"], "old_str": e["old_str"], "new_str": new, "why": why})
json.dump(out, open(sys.argv[1], "w"), indent=1, ensure_ascii=False)
print("ok")
