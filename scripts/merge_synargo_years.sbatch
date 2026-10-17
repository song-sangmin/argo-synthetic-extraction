#!/bin/bash
#SBATCH --account=m4839
#SBATCH --qos=shared
#SBATCH --constraint=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --job-name=synargo_merge
#SBATCH --output=/pscratch/sd/s/sangsong/synargo_logs/%x_%j.out
#
# Join the 10 yearly files into one file with a running profile number per float.
# Run after all extract_synargo_years.sbatch tasks have finished.
#
# Usage (from the folder with merge_synargo_years.py):
#   sbatch merge_synargo_years.sbatch

set -eo pipefail

module load python
conda activate synArgo # any env with xarray, netCDF4, pandas

echo "Job $SLURM_JOB_ID on $(hostname)"
echo "Started: $(date)"

python -u merge_synargo_years.py

echo "Finished: $(date)"
