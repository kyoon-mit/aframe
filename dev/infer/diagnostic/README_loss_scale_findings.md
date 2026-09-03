# Time and spectral terms are not on a comparable scale

Findings on `ScheduledMixtureLoss`, to be reproduced on the other machine.

## Code version

| | |
|---|---|
| branch | `reg-dev-latest` |
| commit | `6a46d28ec74d2f645a2bef3e1c0a4eb857b722d1` (`6a46d28`) |
| `projects/train/train/losses.py` blob | `75d22b8d89b4e339e7815a69858507652aee667d` |
| pushed to | `github.com/kyoon-mit/aframe` |

```bash
git fetch origin reg-dev-latest
git checkout 6a46d28
cd projects/train && source .venv/bin/activate
python ../../dev/infer/diagnostic/probe_loss_gradient_balance.py
```

Every number below comes from that script. It is self-contained (no data,
no checkpoint, CPU only, about a minute) and uses fixed seeds, so it
should reproduce exactly.

## What was measured

`alpha` weights a time-domain term against a frequency-domain one, on the
assumption that the two are comparable. They are not, and the mismatch is
not constant: it moves with the waveform amplitude, which the SNR
curriculum changes during a run.

Both gradients are taken with respect to the same variable, the
time-domain prediction, since that is what the denoiser emits. The
spectral gradient reaches it through the FFT.

### Values and gradients against amplitude

Relative error held fixed at 30%, so only the amplitude changes:

| amplitude | time | spectral | `\|g_time\|` | `\|g_spec\|` | ratio |
|---|---|---|---|---|---|
| 0.01 | 9.01e-06 | 2.85e-02 | 7.31e-08 | 5.72e-03 | **78263** |
| 0.1 | 9.01e-04 | 2.85e-02 | 7.31e-07 | 5.72e-04 | 783 |
| 1 | 9.01e-02 | 2.85e-02 | 7.31e-06 | 5.72e-05 | 7.83 |
| 10 | 9.01e+00 | 2.85e-02 | 7.31e-05 | 5.72e-06 | 0.078 |
| 100 | 9.01e+02 | 2.85e-02 | 7.31e-04 | 5.72e-07 | **0.0008** |

The time term grows with the square of the amplitude while the spectral
term does not move at all, because a log difference cancels any common
scale.

The gradients are the more serious half. They scale in *opposite*
directions -- time as amplitude, spectral as 1/amplitude, the latter from
the factor 1/|X| that differentiating a log contributes -- so their ratio
moves as the square of the amplitude, across eight orders of magnitude
here.

### Where they balance

The two gradients are equal near amplitude 2.6. Below that the spectral
term drives the update; above it the time term does. A run whose
curriculum lowers the SNR therefore drifts from time-led to spectral-led
with no change to the config.

### Can the gradient norms be used to balance them

Dividing each term by its own gradient norm pins the ratio at exactly
1.000000 at every amplitude tested. But those divisors are far too noisy
to use per batch:

| term | mean norm | relative spread over 20 batches |
|---|---|---|
| time | 2.34e-03 | 0.004 |
| spectral | 2.29e-02 | **1.019** |

The spectral norm varies by more than its own size from batch to batch.
Dividing by it would inject more noise than it removes -- the same trap as
dividing by a per-batch statistic of the target.

Smoothing the norms with a running average (momentum 0.99) trades that
noise for lag, and does not hold the ratio at one:

| amplitude | update ratio | spread |
|---|---|---|
| 1.0 | 0.53 | 0.21 |
| 0.3 | 2.66 | 1.43 |
| 0.1 | 3.01 | 2.60 |

Roughly two orders of magnitude better than the raw ratio, but it still
wanders between about 0.5 and 3, and the spread is as large as the value.

## What is already in this commit, and what is not

`6a46d28` normalises the **time term by a running mean of the target
scale**, replacing the previous division by a statistic of the current
batch. That fixes the *values*: over a hundredfold amplitude range the
time-to-spectral value ratio stays near 3.1, where before it swung from
3266 to 0.003. The buffer is part of the state dict, so a resumed run
keeps its normalisation, and it is frozen in eval mode.

It does **not** address the gradient ratio in the table above. That
normalisation is a single constant on one term, so it shifts the values
without changing how the two gradients scale against each other. Gradient
balancing is not implemented -- the measurements above are the case for
and against it, not a change that has been made.

## Two other things worth knowing

**Parseval holds, as expected, and the 1/L is real.** On raw O3a H1 strain
and on whitened noise, `sum(x^2)` equals `(1/L) * sum(|X|^2)` to 5e-16
with the one-sided weighting (interior bins doubled, DC and Nyquist not).
The 1/L is needed because torch's forward transform is unnormalised;
`norm="ortho"` would remove it.

**Rescaling the spectrum is not a free change.** Multiplying `|X|` by L
leaves the msle *value* untouched but divides its gradient by L, so it
acts as a learning-rate change on that term and nothing else. It is not
neutral near zero, though: because `log_floor` is an absolute threshold
that is *not* rescaled alongside the magnitudes, scaling moves every bin
relative to it. Multiplying by L pushes bins above the floor and scores a
given leak more harshly; dividing by L pushes them under it, and at
`log_floor: 1e-3` with L = 8192 the loss collapses to about 4e-10, which
would be optimising nothing. `log_floor` is the honest knob for where
"effectively zero" sits; a scale factor conflates it with gradient
strength.

## Open question

Whether to balance by gradient norm at all. The measurements say it works
in principle and only approximately in practice. If it is added it should
*replace* the running target scale rather than sit on top of it -- two
normalisations competing over the same term is how the current situation
arose.
