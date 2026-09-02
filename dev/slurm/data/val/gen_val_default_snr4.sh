#!/bin/bash
#SBATCH --job-name=gen_val
#SBATCH --partition=test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/DATA/aframe_data/slurm_logs/slurm-gen_val-%j.out
#SBATCH --error=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/DATA/aframe_data/slurm_logs/slurm-gen_val-%j.err

# Submit with:  sbatch dev/slurm/data/val/gen_val_default_snr4.sh
#
# Validation waveforms: projected onto H1/L1 with SNR >= 4, drawn from the
# astrophysical BNS prior (end_o3_ratesandpops_bns). Produces the
# projected+SNR WaveformSet the mainline-aframe validation path expects.
#
# Memory: rejection sampling holds the FULL output in RAM —
# 2 ifos x NUM_SIGNALS x 65536 samples x 8 B (~1 GB per 1k signals), plus
# one MAX_NUM_SAMPLES batch and the worker pool. 50k signals -> ~52 GB
# output, so request 128G. Scale --mem if NUM_SIGNALS changes.
# MAX_NUM_SAMPLES only caps per-batch memory; the loop always runs until
# NUM_SIGNALS accepted signals exist.
#
# PSD: passing the O3a DIRECTORY averages the median PSD of every background
# file (duration-weighted) instead of trusting a single segment, and the PSD
# is truncated to the 2048 Hz waveform band (the raw files are 4096 Hz).

export AFRAME_CONTAINER_ROOT=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/DATA/aframe_data/containers
export SINGULARITYENV_PYTHONWARNINGS="ignore::SyntaxWarning"
export SINGULARITYENV_CURL_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
export SINGULARITYENV_SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
export SINGULARITYENV_REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
export NO_PKGCONFIG=1

POOL=${SLURM_CPUS_PER_TASK:-4}
NUM_SIGNALS=50000
# NUM_SIGNALS=1000
MAX_NUM_SAMPLES=2048
PSD_DIR=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/DATA/O3a_H1_L1_4096Hz

# --bind mounts the host repo over the container's editable /opt/aframe so
# local code edits are used without rebuilding data.sif.
singularity run \
    --bind /n/holystore01/LABS/iaifi_lab/Lab/kyoon/aframe:/opt/aframe \
    $AFRAME_CONTAINER_ROOT/data.sif \
    python -m data validation_waveforms \
    --num_signals ${NUM_SIGNALS} \
    --max_num_samples ${MAX_NUM_SAMPLES} \
    --prior priors.priors.end_o3_ratesandpops_bns \
    --ifos='["H1", "L1"]' \
    --minimum_frequency 20 \
    --reference_frequency 50 \
    --sample_rate 2048 \
    --waveform_duration 32 \
    --waveform_approximant IMRPhenomPv2 \
    --right_pad 2 \
    --highpass 20 \
    --lowpass 1024 \
    --snr_threshold 4 \
    --psd ${PSD_DIR} \
    --pool ${POOL} \
    --output_file /n/holystore01/LABS/iaifi_lab/Lab/kyoon/DATA/aframe_data/val/end_o3_ratesandpops_bns_snr4.hdf5
    # --output_file /n/holystore01/LABS/iaifi_lab/Lab/kyoon/aframe/dev/slurm/data/val/end_o3_ratesandpops_bns_snr4.hdf5
