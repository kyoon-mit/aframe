"""Proposed edits (file, old, new, why). Applied only to scratch copies by simulate.py."""
SITE = "/home/kyoon/SSM-BNS/WEB/kyoon-mit.github.io/docs"
DER = f"{SITE}/projects/bns/derivations.md"
WS4D = f"{SITE}/_updates/2026-10-16-what-s4d-computes.md"
LEC = f"{SITE}/_updates/2026-10-15-oscillators-and-chirps.md"
STAT = f"{SITE}/_updates/2026-10-16-status-dropout-and-signal-only.md"

NEW_SECTION = r"""## Where the merger falls in our 4 s windows {#oct16-merger-position}

**Claim** (15 and 16 October updates): in the training, validation and test
windows of our 4 s runs, signal-only and noisy, the merger falls 3.5 to 4 s
after the window's first sample, at one of 1,024 positions, each equally
likely. A window therefore holds the last 3.5 to 4 s of the chirp and up to
0.5 s after the merger. In each detector the merger moves by at most a
further 21.0 ms either way. The time-warped merger runs use the same windows,
warped up to each detector's amplitude peak; the early time-warped runs use
windows that end 25.0 to 25.5 s before the merger.

**Setup.** Our training code (built on aframe and ml4gw) cuts every window
the same way.

- The stored waveforms (IMRPhenomPv2, generated with PyCBC) are 32 s long at
  2048 Hz, 65,536 samples, with the coalescence placed 2 s before the end, at
  sample 61,440. That sample is the merger here.
- A window is 4 s of whitened data, 8,192 samples. The whitening filter is
  1 s long, so the code whitens a 5 s segment (10,240 samples) and drops
  0.5 s (1,024 samples) at each end.
- The run settings ask for the merger at least 3.5 s after the window's start
  and no later than its end.

**Derivation.** From each stored waveform the code first cuts the 11,264
samples from 9,216 before the merger to 2,048 after it:
$9{,}216 = 10{,}240 - 1{,}024$, the segment less half the filter (no time is
required after the merger), and $2{,}048 = 10{,}240 - 8{,}192$, the segment
less the 3.5 s (7,168 samples) required before the merger and half the
filter. It then takes the 5 s segment that starts at sample $k$ of this cut,
with $k$ drawn with equal probability from the 1,024 whole numbers 0 to
1,023 ($11{,}264 - 10{,}240 = 1{,}024$ choices). The merger is at sample
$9{,}216 - k$ of the segment and, once the first 1,024 samples are dropped,
at sample $8{,}192 - k$ of the window, counting the first sample as 0: at
samples 7,169 to 8,192, which is 3.5005 to 4.0000 s after the first sample.
The last sample is at 3.9995 s, so the window ends 0 to 0.4995 s after the
merger.

Projecting onto each detector then moves the waveform by the difference
between the wave's arrival at the detector and at the Earth's centre, for the
source's direction, cut to whole samples toward zero. That difference is at
most the detector's distance from the Earth's centre over $c$, 21.24 ms for
H1 and 21.26 ms for L1, so the shift is at most 43 samples (21.0 ms): in each
detector the merger falls 3.4795 to 4.0210 s after the first sample. When it
falls after the window's end, the window misses at most the last 21.0 ms of
that detector's chirp.

The signal-only runs read the whitened signal alone, the noisy runs the same
signal added to real noise before whitening, and their validation and test
windows are made by the same code. The time-warped merger runs take the same
window and resample it on a grid uniform in $(t_c - t)^{5/8}$
([time warp](#time-warp)) from its first sample up to $t_c$, the peak of each
detector's whitened amplitude (the size of the analytic signal), so their
input ends at that peak; in the first batches logged by three of these runs,
the peak lay a median 10.7 to 12.2 ms and at most 35 to 63 ms from the merger
placed as above. The early time-warped runs move the cut 25.5 s (52,224
samples) earlier: the window then starts 29.0005 to 29.5 s and ends 25.0005
to 25.5 s before the merger. The stored waveforms start 30 s before the
merger, so this is the earliest window they allow.

**Check.** Python, one thread, with the training code's own functions (its
cut of the stored waveforms, and ml4gw's choice of segment, detector shift
and whitening): each stored waveform was replaced by the numbers 0, 1, 2, …,
so that every window sample shows which stored sample it came from. Over
16,384 windows all 1,024 values of $k$ occurred, and the merger fell at
window samples 7,169 to 8,192 (3.500488 to 4.000000 s, mean 3.7514 s). With
source directions drawn as in training, the shift reached 43 samples, the
merger fell 3.479980 to 4.020996 s after the first sample in a detector, and
after the last sample in 1.25% of the detector windows. With the early runs'
cut it fell 29.000488 to 29.500000 s after the first sample. A single nonzero
sample, whitened with the 1 s filter, a flat noise spectrum and the 20 Hz
highpass, peaks at its own position less 1,024. In 192 stored waveforms
(three storage blocks of 64, chirp masses 0.87 to 2.75 M☉), the amplitude
$\sqrt{h_+^2 + h_\times^2}$ peaks 4 to 23 samples (2.0 to 11.2 ms) before the
merger sample, where the frequency has reached 963 to 1024 Hz, the Nyquist
frequency of the stored data.

## S4D tones: frequency and memory {#s4d-tones}"""

EDITS = [
    # ---------------- derivations.md ----------------
    (DER,
     r"""Setting $\tau = 4$ s gives the
frequency 4 s before merger and the cycles after it:

<div markdown="1">

| chirp mass | frequency 4 s before merger | cycles in the last 4 s |
|---|---|---|
| 0.85 M☉ | 99 Hz | 636 |
| 1.2 M☉ | 80 Hz | 513 |
| 1.8 M☉ | 62 Hz | 398 |
| 2.75 M☉ | 48 Hz | 305 |

</div>

These are leading-order values; post-Newtonian corrections are not included.

## S4D tones: frequency and memory {#s4d-tones}""",
     r"""Setting $\tau = 4$ s gives the
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

Our 4 s windows hold the last 3.5 to 4 s before the merger, which falls 3.5
to 4 s after a window's first sample
([where the merger falls](#oct16-merger-position)). Over 0.85 to 2.75 M☉,
the chirp in one of our windows therefore starts at 48 to 105 Hz and makes
281 to 636 cycles. Since $f \propto \tau^{-3/8}$ and $N \propto \tau^{5/8}$,
each 3.5 s frequency is $(4/3.5)^{3/8} = 1.0513$ times the 4 s one, and each
3.5 s count $(3.5/4)^{5/8} = 0.9199$ times.

These are leading-order values; post-Newtonian corrections are not included.

**Check.** Python (numpy and scipy, one thread), with
$GM_\odot/c^3 = 4.925490947\times10^{-6}$ s: the closed form for $N$ and a
numerical integral of $f$ over the time before merger agree to a relative
$3\times10^{-15}$. At 4 s, 99.41, 80.14, 62.20 and 47.72 Hz and 636.23,
512.88, 398.07 and 305.43 cycles; at 3.5 s, 104.52, 84.25, 65.39 and
50.17 Hz and 585.29, 471.81, 366.19 and 280.98 cycles.

""" + NEW_SECTION,
     "#cycles is cited for 'our windows' but covered only the last 4 s; adds the 3.5 s case (merger 3.5 s in), the window relation, a check, and a new derivation of where the merger falls (pipeline arithmetic), with its check."),
    (DER,
     r"""In our 4 s windows the chirp starts at 48 to 99 Hz ([cycles](#cycles)); it""",
     r"""In our 4 s windows the chirp starts at 48 to 105 Hz ([cycles](#cycles)); it""",
     "Start frequency over merger positions 3.5 to 4 s: 47.72 Hz (2.75 M☉, 4 s) to 104.52 Hz (0.85 M☉, 3.5 s); the 2.55 M☉ clause stays true."),
    (DER,
     r"""over chirp masses 0.85 to 2.75 M☉ and windows
of 3.5 to 4 s ending at the merger, heterodyning""",
     r"""over chirp masses 0.85 to 2.75 M☉ and the
last 3.5 to 4 s before the merger, the stretch of chirp our windows hold
([where the merger falls](#oct16-merger-position)), heterodyning""",
     "Our windows are 4 s and end 0 to 0.5 s after the merger; what is 3.5 to 4 s is the stretch before the merger. Numbers unchanged."),
    (DER,
     r"""Ranges over chirp masses 0.85 to 2.75 M☉, windows of 3.5 and 4 s, and both""",
     r"""Ranges over chirp masses 0.85 to 2.75 M☉, the last 3.5 and 4 s, and both""",
     "Same wording fix as the claim."),
    (DER,
     r"""Cycles for windows of 3.5 / 4 s. The last row is the grid of the AFRAME BNS""",
     r"""Cycles for the last 3.5 / 4 s before the merger. The last row is the grid of the AFRAME BNS""",
     "Same wording fix."),
    (DER,
     r"""same time. In the last 4 s before the merger:""",
     r"""same time. In the last 4 s before the merger, the longest stretch of chirp
our windows hold ([where the merger falls](#oct16-merger-position)):""",
     "Ties the table to the windows."),
    (DER,
     r"""| 2.75 M☉ | 47.72 Hz | 305.4 | 114.5 |

</div>

**Derivation.** With $\tau \propto f^{-8/3}$,""",
     r"""| 2.75 M☉ | 47.72 Hz | 305.4 | 114.5 |

</div>

A channel's kernel is the same in every window, so a channel that is to be
the matched filter in all our windows must cover this longest case. The
chirp in a window whose merger falls 3.5 s in has $(3.5/4)^{5/8} = 0.9199$
of each count: 585.3, 471.8, 366.2 and 281.0 cycles and $\int\tau\,df$ =
219.5, 176.9, 137.3 and 105.4 at the four chirp masses, so 106 to 220
resonators would do for that chirp alone.

**Derivation.** With $\tau \propto f^{-8/3}$,""",
     "Recomputed counts for the shortest stretch (merger 3.5 s in) and why the 4 s count is the per-channel requirement."),
    (DER,
     r"""with $\tau_1$ = 4 s this gives the table. In general, integrating by parts,""",
     r"""with $\tau_1$ = 4 s this gives the table, and since $f_1 \propto \tau_1^{-3/8}$,
$\tau_1 f_1 \propto \tau_1^{5/8}$ gives the 3.5 s values. A channel's output
at the merger adds its kernel times the data back to the window's first
sample, so a kernel matched to the last 4 s is also the matched filter of the
shorter stretch in a window whose merger falls earlier, while one matched
only to the last 3.5 s misses the earliest 0.5 s of chirp in the longest
windows. In general, integrating by parts,""",
     "Derivation of the 3.5 s values and of the longest-case requirement."),
    (DER,
     r"""$2\pi\int\tau\,df$ = 1,208.4 at 1.2 M☉. The frequencies match the
[cycles](#cycles) section.""",
     r"""$2\pi\int\tau\,df$ = 1,208.4 at 1.2 M☉. The frequencies match the
[cycles](#cycles) section. The 3.5 s values were recomputed for this page
(one thread) from the closed form and by quadrature, which agree to a
relative $3\times10^{-15}$: $\int\tau\,df$ = 219.48, 176.93, 137.32 and
105.37, and 585.29, 471.81, 366.19 and 280.98 cycles. In one detector the
shortest stretch is 3.4795 s ([where the merger falls](#oct16-merger-position)),
which gives 104.98 at 2.75 M☉; a merger that falls up to 21.0 ms after the
window's end, in one detector, cuts the top of the band and leaves at most
230.40 (0.85 M☉), below the 4 s value.""",
     "Check for the new numbers."),
    # ---------------- what-s4d-computes.md ----------------
    (WS4D,
     r"""  over frequency: 3/8 of the chirp's cycles, 115 to 239 in our 4 s windows.""",
     r"""  over frequency: 3/8 of the chirp's cycles, 115 to 239 for the last 4 s
  before the merger, the longest stretch of chirp our windows hold.""",
     "The windows hold the last 3.5 to 4 s; 115 to 239 is the 4 s count, which a channel shared by all windows must meet."),
    (WS4D,
     r"""([derivation]({{ d | relative_url }}#oct16-s4d-three-eighths)). In our 4 s
windows, which end at the merger:""",
     r"""([derivation]({{ d | relative_url }}#oct16-s4d-three-eighths)). Our 4 s
windows hold the last 3.5 to 4 s before the merger, which falls 3.5 to 4 s
after a window's first sample
([derivation]({{ d | relative_url }}#oct16-merger-position)). A channel's
kernel is the same in every window, so to be the matched filter in all of
them it must cover the longest case, the last 4 s:""",
     "False: the windows end 0 to 0.5 s after the merger (merger at window samples 7,169 to 8,192)."),
    (WS4D,
     r"""One channel needs at least 115 to 239 resonators over our range of chirp
masses; the clean network has 16 per channel, the noisy ones 32 or 256.""",
     r"""One channel needs at least 115 to 239 resonators over our range of chirp
masses (the chirp of a window whose merger falls 3.5 s in would need 106 to
220 on its own); the clean network has 16 per channel, the noisy ones 32 or
256.""",
     "States the shortest-window count, 105.4 to 219.5 rounded up."),
    # ---------------- oscillators-and-chirps.md ----------------
    (LEC,
     r"""them to be the filter of one of our chirps; the 476 to 1,184 of the first""",
     r"""them to be the filter of the last 4 s of one of our chirps, the longest
stretch a window holds; the 476 to 1,184 of the first""",
     "The count is for the last 4 s; a window can hold as little as 3.5 s."),
    (LEC,
     r"""to be the filter of one of our chirps (section 4)""",
     r"""to be the filter of the last 4 s of one of our chirps, the longest stretch a window holds (section 4)""",
     "Same as the revision box."),
    (LEC,
     r"""ending
at the last moment of the window, the signal of two neutron stars spiralling
into each other.""",
     r"""ending
in the window's last half second (section 1), the signal of two neutron
stars spiralling into each other.""",
     "False: the merger falls 3.5 to 4 s after the window starts, so the signal ends 0 to 0.5 s before the window's end."),
    (LEC,
     r"""3.5 to 4 s after the window starts (run configuration). Unless stated""",
     r"""3.5 to 4 s after the window starts
([derivation]({{ d | relative_url }}#oct16-merger-position)). Unless stated""",
     "The run configuration is not a public source; link the new derivation."),
    (LEC,
     r"""(ours hold 305 to 636)""",
     r"""(ours hold 281 to 636)""",
     "Range over our windows: 280.98 (2.75 M☉, merger 3.5 s in) to 636.23 (0.85 M☉, 4 s)."),
    (LEC,
     r"""chirp starts at 48 to 99 Hz, so the 50 Hz column applies inside the window""",
     r"""chirp starts at 48 to 105 Hz, so the 50 Hz column applies inside the window""",
     "Start frequency over merger positions: 47.72 to 104.52 Hz; the 2.55 M☉ clause stays true."),
    (LEC,
     r"""\tfrac83\int\tau\,df$. For our windows, with 305 to 636 cycles, that is at
least 115 to 239 oscillators for chirp masses 2.75 to 0.85 M☉, and 193 at
1.2 M☉, where $\int\tau\,df = 192.3$. The bound is close to what is needed.""",
     r"""\tfrac83\int\tau\,df$. A channel is the same in every window, so to be the
filter in all of them it must cover the longest case, the last 4 s before
the merger, with 305 to 636 cycles: that is at least 115 to 239 oscillators
for chirp masses 2.75 to 0.85 M☉, and 193 at 1.2 M☉, where
$\int\tau\,df = 192.3$. The chirp of a window whose merger falls 3.5 s in
would need 106 to 220 on its own
([derivation]({{ d | relative_url }}#oct16-s4d-three-eighths)). The bound is
close to what is needed.""",
     "Our windows do not all hold 305 to 636 cycles; the 4 s count is the requirement for a channel shared by all windows."),
    (LEC,
     r"""over a window of 3.5 to 4 s it adds up to the following number of cycles""",
     r"""over the last 3.5 to 4 s before the merger, the stretch a window holds, it
adds up to the following number of cycles""",
     "Wording: windows are 4 s; the stretch before the merger is 3.5 to 4 s. Numbers unchanged."),
    (LEC,
     r"""Chirp masses 0.85 to 2.75 M☉, windows of 3.5 and 4 s before the merger,""",
     r"""Chirp masses 0.85 to 2.75 M☉, the last 3.5 and 4 s before the merger,""",
     "Same wording fix."),
    (LEC,
     r"""neutron star signal by adding its 305 to 636 cycles in step with a copy of
the expected wave; at SNR 8 each cycle carries an SNR of 0.32 to 0.46""",
     r"""neutron star signal by adding its 281 to 636 cycles in step with a copy of
the expected wave; at SNR 8 each cycle carries an SNR of 0.32 to 0.48""",
     "Cycles in our windows 280.98 to 636.23; 8/sqrt(N) = 0.4773 to 0.3172."),
    # ---------------- status page ----------------
    (STAT,
     r"""our data allow, 25 s before merger, warped and sampled at 256 Hz. All twelve""",
     r"""our data allow, ending 25.0 to 25.5 s before the merger
([derivation]({{ d | relative_url }}#oct16-merger-position)), warped and
sampled at 256 Hz. All twelve""",
     "merger_offset 25.5 s: the window ends 25.0005 to 25.5 s before the merger."),
    (STAT,
     r"""| ending | merger | merger, warped | 25 s before merger, warped |""",
     r"""| ending | merger | merger, warped | 25.0 to 25.5 s before merger, warped |""",
     "Same."),
    (STAT,
     r"""its position (25 s before merger),""",
     r"""its position (ending 25.0 to 25.5 s before merger),""",
     "Same."),
]
