# SLIC checks (20 October 2026)

Scripts behind the 20 October BNS update and the derivations
`oct20-gaussian-denoiser` and `oct20-pe-cost`. Run from `projects/train`
with its environment, e.g. `.venv/bin/python -I <script> ...`.

- `time_parts.py <windows.h5> <ckpt>`: GPU time of the waveforms with their
  gradients and of the noise network for 64 points of one window.
- `gauss_vs_net.py <score.yaml> <ckpt> <out.npz>`: denoising loss of the
  noise network against white Gaussian models on the validation windows.
- `color_vs_net.py <score.yaml> <ckpt> <out.npz>`: the same with Gaussian
  models of the mean and the median training spectrum.
- `check_formulas.py`: simulation check of the Gaussian losses and the
  sampler cost arithmetic.

The SLIC code is in `projects/train/train/experimental/slic/`, its
configuration and launch scripts in `dev/local/train/claude_tests/slic/`.
