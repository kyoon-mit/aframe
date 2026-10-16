#!/bin/bash
cd /home/kyoon/SSM-BNS/aframe/projects/train
run() {
  CUDA_VISIBLE_DEVICES=0 WANDB_MODE=disabled .venv/bin/python -m train test --config /home/kyoon/SSM-BNS/aframe/dev/runs/claude_tests/ft_m3_snr8_rop_from_d128_n6_ep634/config.yaml --config /home/kyoon/SSM-BNS/aframe/dev/local/train/denoiser/claude_tests/test/den_reg_sched_alpha_0.5_floor_1e-2_c_rho_5_lambda_amp_3_powerlaw_snr_d128_powerlaw.yaml --trainer.devices=1 --trainer.strategy=auto --trainer.logger=false "--trainer.callbacks=[{'class_path':'train.callbacks.PlotParamEstCallback','init_args':{'save_dir':'/home/kyoon/SSM-BNS/aframe/dev/runs/claude_tests/ft_m3_snr8_rop_from_d128_n6_ep634/test_last/$1'}}]" "--data.snr_sampler=$2" --ckpt_path /home/kyoon/SSM-BNS/aframe/dev/runs/claude_tests/ft_m3_snr8_rop_from_d128_n6_ep634/test_last/last.ckpt > /home/kyoon/SSM-BNS/aframe/dev/local/train/denoiser/claude_tests/test/ft_m3_snr8_rop_from_d128_n6_ep634_$1.log 2>&1
  echo "$1 exit $?"
}
run snr4_powerlaw "{'class_path':'train.augmentations.SnrSampler','init_args':{'distribution':'ml4gw.distributions.PowerLaw','constant_params':{'index':-3.0,'maximum':50.0},'start_params':{'minimum':4.0},'end_params':{'minimum':4.0},'decay_steps':1}}"
run snr8_powerlaw "{'class_path':'train.augmentations.SnrSampler','init_args':{'distribution':'ml4gw.distributions.PowerLaw','constant_params':{'index':-3.0,'maximum':50.0},'start_params':{'minimum':8.0},'end_params':{'minimum':8.0},'decay_steps':1}}"
run snr4_uniform "{'class_path':'train.augmentations.SnrSampler','init_args':{'distribution':'torch.distributions.Uniform','constant_params':{'low':4.0,'validate_args':false},'start_params':{'high':50.0},'end_params':{'high':50.0},'decay_steps':1}}"
