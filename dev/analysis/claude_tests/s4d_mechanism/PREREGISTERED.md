# Pre-registered predictions (written 2026-10-15 21:08 EDT by the main session, before any test ran)

Claim under test (stated to the owner): on clean signals the trained S4D (sig_only_reg_s4d_max_dropout0_64x32x4)
reads chirp mass like a spectrogram: each first-layer resonator responds when the chirp passes its frequency,
and the pattern of those local responses (frequency and sweep rate) fixes the chirp mass. It does NOT add the
whole chirp in step (no long-range phase coherence).

H_local (my claim) versus H_global (the alternative: the channels act as long in-step partial templates).

T4  Peak times of first-layer channels. For every first-layer channel whose dominant resonance lies in 20-150 Hz,
    find the time of the peak of its output envelope on clean whitened chirps (before any nonlinearity).
    H_local: peak time follows the time the chirp passes the channel's frequency (rank correlation > 0.7 across
             channels, peaks spread over seconds).
    H_global: peaks cluster within 0.1 s of the merger for most such channels.
T2  Shuffle 0.5 s segments (8 segments, tapered, random order). Coherent match of the shuffled signal with its
    own template must fall below 0.5 (check that the manipulation really breaks in-step structure).
    H_local: S4D within-1% falls by at most 30% of its clean value.  H_global: falls by at least 70%.
T3  Random constant phase per 0.5 s segment (analytic signal times exp(i phi_k), tapered). Same check and the same
    thresholds as T2.
Refutation: if T2 and T3 both drop S4D by >= 70% and T4 shows merger-clustered peaks, my claim is wrong.
Mixed results (30-70% drops, or T4 between) are reported as inconclusive, not as support.
References run through the same tests, with no prediction required: free kernels (sig_only_reg_free_kernel),
aframe ResNet (sig_only_reg_resnet), and a coherent template bank.
