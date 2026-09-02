#!/bin/bash
#SBATCH --job-name=gen_train_uniform_chirp
#SBATCH --partition=test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/DATA/aframe_data/slurm_logs/slurm-gen_train_uniform_chirp-%j.out
#SBATCH --error=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/DATA/aframe_data/slurm_logs/slurm-gen_train_uniform_chirp-%j.err

# Submit with:  sbatch containers/gen_train_uniform_chirp.sh

export AFRAME_CONTAINER_ROOT=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/DATA/aframe_data/containers
export SINGULARITYENV_PYTHONWARNINGS="ignore::SyntaxWarning"
# Repoint CA bundle vars to the path that exists inside the (Debian-based)
# container; the host's RHEL path /etc/ssl/certs/ca-bundle.crt isn't present there.
export SINGULARITYENV_CURL_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
export SINGULARITYENV_SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
export SINGULARITYENV_REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
export NO_PKGCONFIG=1

# Workers = the cores Slurm allocated (falls back to 4 if run outside Slurm).
POOL=${SLURM_CPUS_PER_TASK:-4}
NUM_SIGNALS=200000
# NUM_SIGNALS=1000

# --bind mounts the host repo over the container's editable /opt/aframe, so
# local code edits (e.g. the parallel from_parameters fix) are used without
# rebuilding data.sif. Drop the bind only once the fix is baked into the image.
singularity run \
    --bind /n/holystore01/LABS/iaifi_lab/Lab/kyoon/aframe:/opt/aframe \
    $AFRAME_CONTAINER_ROOT/data.sif \
    python -m data training_waveforms \
    --num_signals ${NUM_SIGNALS} \
    --waveform_duration 32 \
    --sample_rate 2048 \
    --prior priors.priors.end_o3_ratesandpops_bns_uniform_chirp \
    --minimum_frequency 20 \
    --reference_frequency 50 \
    --waveform_approximant IMRPhenomPv2 \
    --right_pad 2 \
    --pool ${POOL} \
    --output_file /n/holystore01/LABS/iaifi_lab/Lab/kyoon/DATA/aframe_data/train/end_o3_ratesandpops_bns_uniform_chirp.hdf5
    # --output_file /n/holystore01/LABS/iaifi_lab/Lab/kyoon/aframe/dev/slurm/data/train/end_o3_ratesandpops_bns_uniform_chirp.hdf5
