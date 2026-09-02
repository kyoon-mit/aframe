#!/bin/bash
#SBATCH --job-name=val_chunk
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --array=0-49
#SBATCH --output=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/DATA/aframe_data/slurm_logs/slurm-val_chunk-%A_%a.out
#SBATCH --error=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/DATA/aframe_data/slurm_logs/slurm-val_chunk-%A_%a.err

export AFRAME_CONTAINER_ROOT=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/DATA/aframe_data/containers
export SINGULARITYENV_PYTHONWARNINGS="ignore::SyntaxWarning"
export SINGULARITYENV_CURL_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
export SINGULARITYENV_SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
export SINGULARITYENV_REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
export NO_PKGCONFIG=1

OUT_DIR=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/DATA/aframe_data/val/chunks
mkdir -p ${OUT_DIR}

singularity run \
    --bind /n/holystore01/LABS/iaifi_lab/Lab/kyoon/aframe:/opt/aframe \
    $AFRAME_CONTAINER_ROOT/data.sif \
    python -m data validation_waveforms \
    --num_signals 1000 \
    --max_num_samples 4096 \
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
    --psd /n/holystore01/LABS/iaifi_lab/Lab/kyoon/DATA/O3a_H1_L1_4096Hz \
    --pool ${SLURM_CPUS_PER_TASK} \
    --output_file ${OUT_DIR}/chunk-${SLURM_ARRAY_TASK_ID}.hdf5
