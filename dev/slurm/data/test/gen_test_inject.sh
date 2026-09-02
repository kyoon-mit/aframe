#!/bin/bash
#SBATCH --job-name=gen_test_injections
#SBATCH --partition=test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/DATA/aframe_data/slurm_logs/slurm-gen_test-%j.out
#SBATCH --error=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/DATA/aframe_data/slurm_logs/slurm-gen_test-%j.err

# Submit with:  sbatch containers/gen_test.sh
# Common astrophysical SV injection set (InterferometerResponseSet) over O3b.

export AFRAME_CONTAINER_ROOT=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/DATA/aframe_data/containers
export SINGULARITYENV_PYTHONWARNINGS="ignore::SyntaxWarning"
export SINGULARITYENV_CURL_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
export SINGULARITYENV_SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
export SINGULARITYENV_REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
export NO_PKGCONFIG=1

# --bind mounts the host repo over the container's editable /opt/aframe so the
# local scripts/code are used without rebuilding data.sif.
POOL=${SLURM_CPUS_PER_TASK:-4}
singularity run \
    --bind /n/holystore01/LABS/iaifi_lab/Lab/kyoon/aframe:/opt/aframe \
    $AFRAME_CONTAINER_ROOT/data.sif \
    python /opt/aframe/scripts/generate_test_injection_set.py \
    --background_dir /n/holystore01/LABS/iaifi_lab/Lab/kyoon/DATA/O3b_H1_L1_4096Hz \
    --output_dir /n/holystore01/LABS/iaifi_lab/Lab/kyoon/DATA/aframe_data/test \
    --prior end_o3_ratesandpops_bns \
    --sample_rate 2048 \
    --waveform_duration 32 \
    --right_pad 2 \
    --spacing 64 \
    --buffer 64 \
    --snr_threshold 4 \
    --n_shifts 36 \
    --pool ${POOL} \
    --seed 42
