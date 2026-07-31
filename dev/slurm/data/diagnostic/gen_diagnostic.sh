#!/bin/bash
#SBATCH --job-name=gen_diagnostic
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=2-00:00:00
#SBATCH --output=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/DATA/aframe_data/slurm_logs/slurm-gen_diagnostic-%j.out
#SBATCH --error=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/DATA/aframe_data/slurm_logs/slurm-gen_diagnostic-%j.err

# All generation settings live in dev/configs/waveform_configs.json
# under the "diagnostic" key.
# Submit with:  sbatch dev/slurm/data/diagnostic/gen_diagnostic.sh

export AFRAME_CONTAINER_ROOT=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/DATA/aframe_data/containers
export SINGULARITYENV_PYTHONWARNINGS="ignore::SyntaxWarning"
export SINGULARITYENV_CURL_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
export SINGULARITYENV_SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
export SINGULARITYENV_REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
export NO_PKGCONFIG=1

singularity run \
    --bind /n/holystore01/LABS/iaifi_lab/Lab/kyoon/aframe:/opt/aframe \
    $AFRAME_CONTAINER_ROOT/data.sif \
    python /opt/aframe/dev/data/generate_diagnostic_set.py \
    --config /opt/aframe/dev/configs/waveform_configs.json \
    --pool ${SLURM_CPUS_PER_TASK:-4}
