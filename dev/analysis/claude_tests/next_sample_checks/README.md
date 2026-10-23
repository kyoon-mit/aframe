# Next-sample noise model checks (23 October 2026)

Scripts behind the 23 October BNS update. Run from `projects/train` with its
environment, e.g. `.venv/bin/python <script> ...`. The model is
`projects/train/train/experimental/next_sample/`, its configuration and
launch script `dev/local/train/claude_tests/next_sample/`; the checkpoint
used is epoch 288 of `next_sample_k3_shared_mean_asinh`.

- `test1_likelihood.py <ckpt> <out dir>`: log-likelihood of every
  back-to-back 4 s window of the O3a validation files and of all O3b files,
  under the network and under Gaussians with the mean or typical spectrum of
  the training windows (Levinson-Durbin, orders 0 to 8191).
- `test2_detection.py <config> <ckpt> <out dir> [windows]`: matched filter,
  network likelihood ratio and Gaussian likelihood ratio with the true
  template on 20,000 injected O3b test windows.
- `test2_spread.py <config> <ckpt> <out.npz>`: the network likelihood ratio
  on the first 4,000 of those windows, per detector and per 1/16 s block.
- `test2_curvature.py <config> <ckpt>`: the network log-likelihood along
  scaled copies of the template, against a unit Gaussian.
- `summarize.py <test1.npz> <test2.npz> <spread.npz>`: the tables; its
  output is `summary.txt`.
