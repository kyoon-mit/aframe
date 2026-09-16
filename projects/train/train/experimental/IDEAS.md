# Denoising at very low SNR

Ideas for the pre-merger denoiser, where a 4 s window ending 16 s before
coalescence holds about 8% of the event's SNR. An event of SNR 4 has SNR
0.3 in that window. The waveform is an order of magnitude below the noise
per sample.

This document lists the ideas, what each one changes, and the papers the
idea comes from. Nothing here is implemented. Code that tries one of these
goes in this directory.

## Why the usual denoising literature does not apply

Image denoising, speech enhancement, and most work labelled "time series
denoising" assume the signal power is comparable to or larger than the
noise power. U-Nets, transformers, and convolutional autoencoders are
built for that regime: they learn local structure from the data in front
of them. At SNR 0.3 the data in front of them carries almost no local
structure. What makes the signal recoverable at all is that it belongs
to a known family (a chirp with two masses) and that it is coherent over
thousands of cycles. The estimator has to use that prior knowledge. This
is closer to matched filtering and to posterior estimation than to
denoising, and the relevant literature is under those names.

Two references frame the problem. The optimal linear detector for a known
signal in Gaussian noise is the matched filter (Turin 1960), and its
output SNR is set by the signal's total energy over the whole observation,
not by any per-sample ratio. Owen and Sathyaprakash (1999) apply this to
compact binaries and quantify how much SNR is lost when the template does
not match the signal, which is the quantity a learned denoiser is
implicitly trying to minimise.

- Turin, G. L. "An introduction to matched filters." *IRE Transactions
  on Information Theory* 6.3 (1960): 311-329.
- Owen, B. J., and Sathyaprakash, B. S. "Matched filtering of
  gravitational waves from inspiraling compact binaries: Computational
  cost and template placement." *Physical Review D* 60.2 (1999): 022002.

## 1. Condition the denoiser on chirp mass and SNR

The network currently sees the strain and nothing else. Given the chirp
mass it would know which waveform family to look for; given the SNR it
would know how far above the noise to expect it. Both are available in
the training batch already, under `params["chirp_mass"]` and
`params["snr"]`, and are discarded.

The mechanism is feature-wise linear modulation (FiLM): a small network
maps the conditioning vector to a per-layer scale and shift, applied to
the hidden activations of each S4D block. FiLM was introduced for visual
reasoning and is now the standard way to condition a network on side
information without changing its input shape.

The difficulty is that chirp mass and SNR are not known at inference.
Three responses, in order of preference:

1. Train with conditioning dropout, so the model also works with no
   conditioning. At inference it runs unconditioned; the conditioning
   was a training signal. This is how classifier-free guidance trains
   diffusion models (Ho and Salimans 2022).
2. Feed a coarse estimate from a cheap upstream regressor. The existing
   `DenoisedGaussianNLLRegression` already regresses chirp mass.
3. Use the conditioning only in a teacher model and distil to an
   unconditioned student.

- Perez, E., et al. "FiLM: Visual reasoning with a general conditioning
  layer." *AAAI* 2018.
- Ho, J., and Salimans, T. "Classifier-free diffusion guidance."
  arXiv:2207.12598 (2022).

Cost: about 40 lines. `cond_dim: int = 0` on `TimeDomainS4Denoiser`,
zero meaning no FiLM and existing configs unchanged. `_shared_step`
passes the params through.

## 2. Curriculum on the lower SNR bound

Train first on events the model can reconstruct, then lower the floor.
This is curriculum learning (Bengio et al. 2009). The sampler already
supports it: `SnrSampler` interpolates `start_params` to `end_params`
over `decay_steps`. With the log-uniform distribution now in use, only
`minimum` needs to move.

```yaml
start_params: {minimum: 40.0}   # kernel SNR about 3
end_params:   {minimum: 4.0}
decay_steps:  39100             # about 100 epochs
```

The known problem is that the sampler's step counter is not saved in the
checkpoint, so a resumed run restarts the curriculum. Either accept that
or set `decay_steps` from the resumed step by hand.

- Bengio, Y., et al. "Curriculum learning." *ICML* 2009.

Cost: config only.

## 3. Heterodyne the strain

Multiply the whitened strain by `exp(-i phi(t))`, where `phi(t)` is the
phase of a reference chirp, then low-pass. A signal near the reference
chirp mass is moved to near zero frequency, where a narrow filter keeps
it and removes most of the broadband noise. The network then sees a slow
complex signal instead of a fast real one.

This is how the gravitational-wave parameter-estimation community makes
long inspirals tractable: Cornish (2010) introduced the heterodyned
likelihood, and Zackay, Dai and Venumadhav (2018) developed it into
relative binning, which is now standard in LIGO analyses. The idea is
older than either; it is the basis of every radio receiver.

There is already a heterodyne path in
`train/data/supervised/time_frequency_domain.py`. Read that before
writing a new one.

The reference chirp mass has to come from somewhere. A fixed value works
if the training prior is narrow; otherwise it needs idea 1.

- Cornish, N. J. "Fast Fisher matrices and lazy likelihoods."
  arXiv:1007.4820 (2010).
- Zackay, B., Dai, L., and Venumadhav, T. "Relative binning and fast
  likelihood evaluation for gravitational wave parameter estimation."
  arXiv:1806.08792 (2018).

Cost: moderate. Mostly data pipeline. The model input becomes two
channels per detector (in-phase and quadrature).

## 4. Coarse localisation, then fine denoising

A 4 s window cannot see the coherence of a 32 s inspiral. A first network
on the full 32 s at low sample rate estimates the chirp track, or just the
chirp mass and coalescence time. The denoiser then runs on the 4 s window
conditioned on that estimate, through idea 1.

This is a two-stage cascade, and the argument for it is the one made for
coherent integration in pulsar searches and in cryo-EM: at low SNR the
information is in the correlation across many samples, so integrate first
and estimate second.

Cost: high. It is idea 1 plus a new regression task on long windows. The
pieces exist but it is a pipeline, not a patch.

## 5. A generative prior over clean waveforms

Rather than learning a map from noisy to clean, learn the distribution of
clean waveforms and denoise by sampling from the posterior given the
noisy observation. Score-based diffusion models do this: the score
network learns the gradient of the log-density of clean data, and the
noisy observation enters as a likelihood term at sampling time (Song et
al. 2021; Kawar et al. 2022). The prior supplies what the data cannot.

Dax et al. (2021) apply neural posterior estimation to gravitational-wave
parameters and reach the accuracy of stochastic sampling; the same
machinery, pointed at the waveform itself rather than its parameters, is
this idea.

- Song, Y., et al. "Score-based generative modeling through stochastic
  differential equations." *ICLR* 2021.
- Kawar, B., et al. "Denoising diffusion restoration models."
  *NeurIPS* 2022.
- Dax, M., et al. "Real-time gravitational wave science with neural
  posterior estimation." *Physical Review Letters* 127.24 (2021): 241103.

Cost: the highest. A new training task, a sampler, and inference that is
tens to hundreds of forward passes per event. Only worth trying if the
cheaper ideas plateau.

## Related but not the same problem

Noise2Noise (Lehtinen et al. 2018) and Noise2Void (Krull et al. 2019)
train denoisers without clean targets by exploiting the independence of
noise across observations. We have clean targets, so the method does not
apply, but the underlying point does: at low SNR the model should learn
the noise statistics as much as the signal. The residual formulation,
where the network predicts the noise and the waveform is what remains, is
that point made in a different way.

- Lehtinen, J., et al. "Noise2Noise: Learning image restoration without
  clean data." *ICML* 2018.
- Krull, A., Buchholz, T.-O., and Jug, F. "Noise2Void: Learning
  denoising from single noisy images." *CVPR* 2019.

The gravitational-wave detection literature bounds what is learnable at a
given SNR. George and Huerta (2018) showed convolutional networks match
matched filtering for detection above SNR about 10, and the MLGWSC-1
challenge (Schäfer et al. 2023) compared learned and classical searches on
common data. Neither is denoising, but they show the SNR at which a
learned model stops being able to tell signal from noise, which is a
floor for any denoiser.

- George, D., and Huerta, E. A. "Deep neural networks to enable real-time
  multimessenger astrophysics." *Physical Review D* 97.4 (2018): 044039.
- Schäfer, M. B., et al. "First machine learning gravitational-wave search
  mock data challenge." *Physical Review D* 107.2 (2023): 023021.

## Order of attack

1 and 2 together first. They are a day of work, and they address the
actual failure, which is that at SNR 0.3 the model has no way to know
which waveform it is looking for. 3 next if 1 shows that knowing the
chirp mass helps. 4 and 5 only if those plateau.
