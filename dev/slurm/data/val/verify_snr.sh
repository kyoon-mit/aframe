cd /n/holystore01/LABS/iaifi_lab/Lab/kyoon/DATA/aframe_data
sbatch --job-name=verify_snr --partition=test --cpus-per-task=4 --mem=16G --time=01:00:00 \
  --output=slurm_logs/slurm-verify_snr-%j.out --error=slurm_logs/slurm-verify_snr-%j.err \
  --wrap "export NO_PKGCONFIG=1; singularity exec \
    --bind /n/holystore01/LABS/iaifi_lab/Lab/kyoon/aframe:/opt/aframe \
    containers/data.sif /opt/env/bin/python /opt/aframe/dev/data/verify_snr.py \
    --input val/end_o3_ratesandpops_bns_snr4.hdf5 \
    --background /n/holystore01/LABS/iaifi_lab/Lab/kyoon/DATA/O3a_H1_L1_4096Hz \
    --dest val --highpass 20 --lowpass 1024 \
    --recompute True"
