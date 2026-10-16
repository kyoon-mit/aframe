#!/bin/bash
# usage: make_test.sh RUN_NAME CKPT TAG [GPU]
# Writes and starts RUN_DIR/TAG/run_tests.sh: 50k-window test on the three
# populations, in parallel, as for den-reg test_ep339.
set -e
R=/home/kyoon/SSM-BNS/aframe/dev/runs/claude_tests/$1
T=$R/$3
GPU=${4:-1}
mkdir -p $T
cp $2 $T/test.ckpt
cp $R/config.yaml $T/run_config_snapshot.yaml
OVERLAY=/home/kyoon/SSM-BNS/aframe/dev/local/train/denoiser/claude_tests/test/den_reg_sched_alpha_0.5_floor_1e-2_c_rho_5_lambda_amp_3_powerlaw_snr_d128_powerlaw.yaml
cat > $T/run_tests.sh <<SH
#!/bin/bash
cd /home/kyoon/SSM-BNS/aframe/projects/train
run() {
  mkdir -p $T/\$1
  CUDA_VISIBLE_DEVICES=$GPU WANDB_MODE=disabled HDF5_USE_FILE_LOCKING=FALSE .venv/bin/python -m train test --config $T/run_config_snapshot.yaml --config $OVERLAY --trainer.devices=1 --trainer.strategy=auto --trainer.default_root_dir=$T/\$1 --data.batch_size=32 --data.test_batches=1563 --data.batches_per_epoch=1563 --data.persistent_workers=false --data.max_num_workers=4 "--trainer.callbacks=[{'class_path':'train.callbacks.PlotParamEstCallback','init_args':{'save_dir':'$T/\$1'}}]" "--data.snr_sampler=\$2" --ckpt_path $T/test.ckpt > $T/\$1.log 2>&1
  echo "\$1 exit \$?"
}
run snr4_powerlaw "{'class_path':'train.augmentations.SnrSampler','init_args':{'distribution':'ml4gw.distributions.PowerLaw','constant_params':{'index':-3.0,'maximum':50.0},'start_params':{'minimum':4.0},'end_params':{'minimum':4.0},'decay_steps':1}}" &
run snr8_powerlaw "{'class_path':'train.augmentations.SnrSampler','init_args':{'distribution':'ml4gw.distributions.PowerLaw','constant_params':{'index':-3.0,'maximum':50.0},'start_params':{'minimum':8.0},'end_params':{'minimum':8.0},'decay_steps':1}}" &
run snr4_uniform "{'class_path':'train.augmentations.SnrSampler','init_args':{'distribution':'torch.distributions.Uniform','constant_params':{'low':4.0,'validate_args':false},'start_params':{'high':50.0},'end_params':{'high':50.0},'decay_steps':1}}" &
wait
SH
chmod +x $T/run_tests.sh
setsid nohup bash $T/run_tests.sh > $T/run_tests.out 2>&1 < /dev/null &
echo "started $T"
