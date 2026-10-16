"""Build the corrected edit set from the two drafts, apply it to copies, check uniqueness."""
import json
import re
import os
import shutil

SC = '/tmp/claude-1024/-home-kyoon-SSM-BNS/82effc75-bb98-44ad-af20-e41e39a7258d/scratchpad/oct16fix'
OUT = SC + '/adv2'
N = json.load(open(SC + '/nonlin/edits_v2.json'))
S = json.load(open(SC + '/sourcing3/edits_final.json'))


def sub(s, old, new):
    assert s.count(old) == 1, (s.count(old), old[:80])
    return s.replace(old, new)


# ---------------- N1: the nonlinearity section ----------------
t = N[0]['new_str']
t = sub(t, "added with equal weights\n  (which are in step there, since every piece's signal is positive), the",
        "added with equal weights\n  (which are in step there, since every piece carries the same signal), the")
t = sub(t, "For a\nGaussian input, the correlation of a function's output with its input, at\nany time lag, is this same $g$ times the input's own correlation (Bussgang",
        "For a\nGaussian input, the cross-correlation of a function's output with its\ninput, at any time lag, is this same $g$ times the input's autocorrelation\n(Bussgang")
t = sub(t, "$\\sigma \\to \\infty$, $c \\to 1$, the bracket tends to $\\pi/2 - 1$, and",
        "$\\sigma \\to \\infty$, $c \\to 1$, the term in parentheses tends to\n$\\pi/2 - 1$, and")
t = sub(t, "(a result of Sheppard 1899, Phil. Trans. R.\nSoc. Lond. A 192, 101,",
        "(a result of Sheppard 1899, Phil. Trans. R.\nSoc. Lond. A 192, 101, §27,")
t = sub(t, "*Pieces whose signals have one sign.* If every $s_k$ has the same sign, as\nfor the pieces of the [$N^{1/4}$ section](#semi-coherent), equal weights are\nin step, and the mean of $\\sum_k \\phi(z_k)$ moves by\n$\\sum_k \\big(m(s_k) - m(0)\\big)$ exactly. For $N$ pieces that each carry $a$,\n",
        "**Pieces that all carry the same signal.** If each of $N$ pieces carries the\nsame signal $a$, as in the [$N^{1/4}$ section](#semi-coherent), equal weights\nare in step, and the mean of $\\sum_k \\phi(z_k)$ moves by\n$N\\big(m(a) - m(0)\\big)$ exactly, so\n")
for lab in ["The average gain.", "The sum in step.", "GELU and ReLU.", "ReLU, closed form.",
            "Threshold at zero, closed form.", "GELU, closed form.",
            "Even functions keep nothing linear.",
            "Only even functions lose the linear part at every spread."]:
    t = sub(t, "*" + lab + "*", "**" + lab + "**")
t = sub(t, "  1, 1.678 and 2 and the ReLU agree with it to 0.04% (at most 2.4 standard\n  errors in the first run, 1.5 in the second). The threshold at zero gives\n  $6.374 \\pm 0.005$ and $6.369 \\pm 0.003$ against 6.371, its exact mean\n  shift for this cosine averaged over 4,000 phases, which is\n  $r\\rho = 6.383$ with its $s^3$ term. With equal weights the GELU, the\n  ReLU, the square and the absolute value agree with $r_2\\rho^2/\\sqrt{2K}$\n  to 0.5% (at most 1.4 standard errors), and the threshold at zero gives\n  $-0.003 \\pm 0.004$ and $-0.001 \\pm 0.003$ against 0.",
        "  1, 1.678 and 2 and the ReLU agree with it to 0.04% (at most 2.4 standard\n  errors in the first run, 1.5 in the second). The threshold at zero gives\n  $6.374 \\pm 0.005$ and $6.369 \\pm 0.003$ against 6.371, its exact value\n  for this cosine averaged over 4,000 phases (the first-order\n  $r\\rho = 6.383$ lowered by its $s^3$ term). With equal weights, divided by\n  the exact spread in the same way, the GELU, the ReLU, the square and the\n  absolute value agree with $r_2\\rho^2/\\sqrt{2K}$ to 0.5% (at most 1.4\n  standard errors), and the threshold at zero gives $-0.003 \\pm 0.004$ and\n  $-0.001 \\pm 0.003$ against 0.")
t = sub(t, "- By numerical integration, $m(a) - m(-a) - a$ is at most $9\\times10^{-16}$\n  for the GELU and the ReLU at $\\sigma$ = 0.5, 1 and 2 and $a$ = 0.1, 1 and\n  3, and",
        "- By numerical integration, $m(a) - m(-a) - a$ is at most\n  $1.4\\times10^{-15}$ (rounding) in two codes for the GELU and the ReLU at\n  $\\sigma$ = 0.5, 1 and 2 and $a$ = 0.1, 1 and 3, and")
t = sub(t, "- The analysis of 15 October behind the lecture, with its own integration\n  and Monte Carlo codes, gave the same $r$ to six digits.\n",
        "- The analysis of 15 October behind the 15 October update, with its own\n  integration and Monte Carlo codes, gave the same $r$ to six digits.\n- A third code, written for this check (scipy quadrature), reproduces the\n  closed forms, the minimum, $g$, $h$, $r_2$, the exact SNRs of the 600\n  pieces and the threshold's 6.375 and 6.371 to the digits shown.\n")
assert '*' + 'The' not in t.replace('**The', '')  # no italic labels left


def primes_inline(s):
    """Inline $...$ is not protected from kramdown's smart quotes: write primes as \\prime."""
    parts = re.split(r'(\$\$.*?\$\$)', s, flags=re.S)
    out = []
    nfix = 0
    for k, part in enumerate(parts):
        if k % 2 == 1:
            out.append(part)
            continue

        def fix(m):
            nonlocal nfix
            body = m.group(1)
            if "'" in body:
                nfix += 1
                body = body.replace("''", "^{\\prime\\prime}").replace("'", "^\\prime")
            return '$' + body + '$'
        out.append(re.sub(r'(?<![\$\\])\$(?!\$)([^$]+?)(?<!\\)\$(?!\$)', fix, part))
    return ''.join(out), nfix


t, nfix = primes_inline(t)
print('inline spans with primes rewritten:', nfix)
assert nfix == 13
t = sub(t, "**Check.** Python with numpy, scipy, mpmath and PyTorch on the CPU, two codes\n(`check_nonlin.py` and `check_nonlin_mp.py`):",
        "**Check.** Python with numpy, scipy, mpmath and PyTorch on the CPU, three\ncodes (`check_nonlin.py`, `check_nonlin_mp.py` and a third written to check\nthis section):")
t = sub(t, "- By numerical integration, $m(a) - m(-a) - a$ is at most\n  $1.4\\times10^{-15}$ (rounding) in two codes for the GELU and the ReLU at\n  $\\sigma$ = 0.5, 1 and 2 and $a$ = 0.1, 1 and 3, and $E[n(n^3 - 3n)]$ is $-0.5625$, 0 and 36 at $\\sigma$ = 0.5, 1 and 2,\n  as $3\\sigma^4 - 3\\sigma^2$ gives.",
        "- By numerical integration, $m(a) - m(-a) - a$ is at most\n  $1.4\\times10^{-15}$ (rounding) in two codes for the GELU and the ReLU at\n  $\\sigma$ = 0.5, 1 and 2 and $a$ = 0.1, 1 and 3, and $E[n(n^3 - 3n)]$ is\n  $-0.5625$, 0 and 36 at $\\sigma$ = 0.5, 1 and 2, as $3\\sigma^4 - 3\\sigma^2$\n  gives.")
t = sub(t, "- The analysis of 15 October behind the 15 October update, with its own\n  integration and Monte Carlo codes, gave the same $r$ to six digits.\n- A third code, written for this check (scipy quadrature), reproduces the\n",
        "- Our analysis of 15 October, with its own integration and Monte Carlo\n  codes, gave the same $r$ to six digits.\n- The third code (scipy quadrature) reproduces the\n")
N[0]['new_str'] = t

# ---------------- N4: #semi-coherent correction note ----------------
t = N[3]['new_str']
t = sub(t, "applied to the pieces before they are added (a square, an absolute value, an\nenergy) turns the sum into this kind. (Corrected 16 October: an earlier\nversion also listed a threshold and a GELU.",
        "applied to the pieces before they are added (a square, an absolute value)\nturns the sum into this kind. (Corrected 16 October: an earlier version also\nlisted a threshold and a GELU.")
t = sub(t, " A GELU and a threshold at zero\nare not even: they keep a part linear in a weak signal, the GELU with gain\n1/2 ([GELU section](#oct16-s4d-gelu-linear)), and pieces added in step after\nthem keep most of the SNR. For the 600 pieces above",
        " A GELU and a threshold at zero are\nnot even: they keep a part linear in a weak signal, the GELU with gain 1/2\n([GELU section](#oct16-s4d-gelu-linear)), and pieces added in step after\nthem keep, to first order, the fraction 0.838 to 0.910 of the SNR (GELU,\nnoise spreads 0.5 to 2) or 0.798 (threshold at zero). For the 600 pieces above")
N[3]['new_str'] = t

# ---------------- N5: article, first obstacle ----------------
t = N[4]['new_str']
t = sub(t, "$\\max(0, x)$, keeps 86%, and a threshold at zero 80%, at any spread). Only\nan even function, such as a square or an energy, loses the part linear in\nthe signal at every spread, and what it leaves is a sum of powers\n",
        "$\\max(0, x)$, keeps 86%, and a threshold at zero 80%, at any spread). The\nchannels of one layer share their input, so their noises are in general\ncorrelated, a case these numbers do not cover. Only an even function, such\nas a square or an energy, loses the part linear in the signal at every\nspread, and what it leaves is a sum of powers\n")
N[4]['new_str'] = t

# ---------------- N10: glossary entry ----------------
t = N[9]['new_str']
t = sub(t, "  short: >-\n    The function that is 1 where its input is above zero and 0 elsewhere,\n    the Heaviside step function. The sign function, +1 above zero and −1\n    below, is the same up to a scale and a shift. A threshold on the size of\n    x (1 where x is above a level or below minus that level) is instead an\n    even function of x.\n",
        "  short: >-\n    The function that is 1 where its input is above zero and 0 elsewhere\n    (the Heaviside step function). The sign function, +1 above zero and −1\n    below, is the same up to a scale and a shift; a threshold on the size of\n    the input, 1 where it is above a level or below minus that level, is\n    instead an even function.\n")
N[9]['new_str'] = t

# ---------------- S1: heterodyne section ----------------
t = S[0]['new_str']
t = sub(t, "So 21 to 34 tones at the grid's worst errors, and 3 to 7 at 1%, with the\nsame count for either sign. The first version of the rebuttal also took\n$\\epsilon = -6.740\\%$, which no chirp mass in the grid's range reaches, since\nnone is more than 6.315% below its nearest grid mass; there $q = 4.27\\%$ and\nthe counts are 34, 30, 27 and 23.",
        "So 21 to 34 tones at the grid's worst errors, and 3 to 7 at an error of 1%,\nwhere both signs give the same count. The first version of the rebuttal\nalso took $\\epsilon = -6.740\\%$, which no chirp mass in the grid's range\nreaches, since none is more than 6.315% below its nearest grid mass; there\n$q = 4.27\\%$ and the counts are 34, 30, 27 and 23.")
t = sub(t, "Two earlier check\nscripts written for the lecture give the same counts in every case they\ncover.",
        "Two earlier check\nscripts, written for the 15 October update, give the same counts in every\ncase they cover, and a third code written for this check reproduces the\nworst errors, $q$, the leftover frequencies, the in-step ratios and every\ncount in the table.")
S[0]['new_str'] = t

# ---------------- S2: step-time section ----------------
t = S[1]['new_str']
t = sub(t, "So 83.5 to 90.0% of the\nelapsed time per batch is neither the whitening arithmetic nor the network,\nand what fills it was not measured. These numbers come from one measurement.",
        "So 83.5 to 90.0% of the\nelapsed time per batch is outside the timed data preparation and training\nstep, and what fills it was not measured. These numbers come from one\nmeasurement.")
t = sub(t, "channel), on noisy data from both LIGO detectors at 2048 Hz, with 32 windows\nper batch and 1,564 training batches and 64 validation batches per epoch\n(run configurations). A short script timed the GPU work of one batch on\nrandom numbers of the real shapes, without reading any data:",
        "channel), on noisy data from both LIGO detectors sampled at 2048 Hz, which\nthe network halves to 1,024 Hz before its kernels, with 32 windows per\nbatch and 1,564 training batches and 64 validation batches per epoch (run\nconfigurations). A short script timed the GPU work of one batch on random\nnumbers of the real shapes, without reading any data:")
t = sub(t, "| network | data preparation | training step | sum | elapsed time per batch | neither |",
        "| network | data preparation | training step | sum | elapsed time per batch | outside the timed parts |")
t = sub(t, "The elapsed time per batch of each real\nrun is its run time recorded in Weights and Biases divided by the number of\nbatches it had run, 1,628 per epoch counting validation.",
        "The elapsed time per batch of each real\nrun was read from its run time recorded in Weights and Biases, with 1,628\nbatches per epoch counting validation.")
t = sub(t, "The last column is 1 minus the sum divided by the elapsed time per batch;\nthe log rounds it to 80 to 90%. Since the timed parts are upper bounds\n(below), these fractions are lower bounds.",
        "The last column is 1 minus the sum divided by the elapsed time per batch.")
t = sub(t, "**Caveats** (as logged). This is one measurement. The GPU was shared with 8\nrunning jobs, so the absolute times are upper bounds. The 83.5 to 90.0% that\nis neither the whitening arithmetic nor the network was not broken down: it\nis data loading (cutting windows from the background files and reading\nwaveforms from disk, on the processor rather than the GPU) or the sharing of\nthe GPU, in proportions not measured.",
        "**Caveats.** This is one measurement. The GPU was shared with 8 running\njobs, so the absolute times are upper bounds. The remaining 83.5 to 90.0%\nwas not broken down. It holds everything outside the two timed parts, among\nthem the data loading (cutting windows from the background files and\nreading waveforms from disk, on the processor rather than the GPU) and the\nwaits caused by sharing the GPU; their shares were not measured.")
t = sub(t, "channels, state sizes 64 and 512). From the free-kernel run's record in\nWeights and Biases, 97,683 s over 395,899 training batches and the 64\nvalidation batches of each of its 253 completed epochs give 237.0 ms per\nbatch, against 238 ms in the log; the S4D runs were interrupted and resumed,\nand their elapsed times were not recomputed. The last column was recomputed\nfrom the others. The GPU timing was not repeated.",
        "channels, state sizes 64 and 512, the rate halved to 1,024 Hz). The\nfree-kernel run ran without a break, and its record in Weights and Biases,\n97,683 s over 395,899 training batches and the 64 validation batches of\neach of its 253 completed epochs, gives 237.0 ms per batch. The S4D runs\nwere interrupted and resumed, so their records cannot be checked the same\nway. The epoch timers of the progress bars in the three runs' training\nlogs, read at their last update in each epoch, give 236.1, 254.5 and\n281.8 ms per batch over 253, 331 and 344 epochs, 0.8 to 2.1% below the\nelapsed times above. The last column was recomputed from the others. The\nGPU timing was not repeated.")
S[1]['new_str'] = t

# ---------------- S8: 23 September correction notice ----------------
t = S[7]['new_str']
t = sub(t, "None of these denoiser runs has been repeated without dropout, so whether it lowered the overlaps below is not known.",
        "None of these runs has been repeated without dropout, so whether it lowered the overlaps below, or the chirp-mass accuracy of the joint denoiser and regressor, is not known.")
S[7]['new_str'] = t

edits = []
for i, e in enumerate(N, 1):
    edits.append(dict(file=e['file'], old_str=e['old_str'], new_str=e['new_str'], tag='N%d' % i))
for i, e in enumerate(S, 1):
    edits.append(dict(file=e['file'], old_str=e['old_str'], new_str=e['new_str'], tag='S%d' % i))

# apply to copies, in order, checking uniqueness each time
files = sorted(set(e['file'] for e in edits))
os.makedirs(OUT + '/patched', exist_ok=True)
os.makedirs(OUT + '/orig', exist_ok=True)
txt = {}
for f in files:
    shutil.copy(f, OUT + '/orig/' + os.path.basename(f))
    txt[f] = open(f).read()
for e in edits:
    c = txt[e['file']].count(e['old_str'])
    c0 = open(e['file']).read().count(e['old_str'])
    print(e['tag'], os.path.basename(e['file']), 'count now', c0, 'count at apply', c)
    assert c == 1 and c0 == 1
    txt[e['file']] = txt[e['file']].replace(e['old_str'], e['new_str'])
for f in files:
    open(OUT + '/patched/' + os.path.basename(f), 'w').write(txt[f])
json.dump(edits, open(OUT + '/final_edits.json', 'w'), indent=1, ensure_ascii=False)
print('wrote', len(edits), 'edits')

# ---- final tidy of N4 wrapping (whole new_str), then re-apply everything ----
N4_FINAL = """applied to the pieces before they are added (a square, an absolute value)
turns the sum into this kind. (Corrected 16 October: an earlier version also
listed a threshold and a GELU. A GELU and a threshold at zero are not even:
they keep a part linear in a weak signal, the GELU with gain 1/2
([GELU section](#oct16-s4d-gelu-linear)), and pieces added in step after
them keep, to first order, the fraction 0.838 to 0.910 of the SNR (GELU,
noise spreads 0.5 to 2) or 0.798 (threshold at zero). For the 600 pieces
above at SNR 8 the exact SNR of the sum is 7.737 after a GELU and 6.271
after a threshold at zero, against 8 without them and 1.848 after a square
([section on nonlinear functions](#oct16-nonlinearity-linear-part)). A
threshold on the size $\\lvert z_k\\rvert$ is even and does turn the sum into
this kind.)"""
assert N4_FINAL.replace('\n', ' ') == edits[3]['new_str'].replace('\n', ' '), 'N4 text changed beyond wrapping'
edits[3]['new_str'] = N4_FINAL
txt = {f: open(f).read() for f in files}
for e in edits:
    assert txt[e['file']].count(e['old_str']) == 1
    txt[e['file']] = txt[e['file']].replace(e['old_str'], e['new_str'])
for f in files:
    open(OUT + '/patched/' + os.path.basename(f), 'w').write(txt[f])
json.dump(edits, open(OUT + '/final_edits.json', 'w'), indent=1, ensure_ascii=False)
print('final N4 rewrapped; wrote', len(edits))
