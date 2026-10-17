#!/bin/bash
#SBATCH --account=m4839
#SBATCH --qos=shared
#SBATCH --constraint=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --job-name=synSOCAT
#SBATCH --output=/pscratch/sd/s/sangsong/synargo_logs/%x_%j.out
#
# Extract the synthetic profile matched to each SOCAT sample, one year at a time,
# into a single file with dimensions (sample_datetag, nVertLevels).
# Any arguments after the script name are passed on to extract_SOCAT.py.
#
# Usage (from the folder with extract_SOCAT.py, with the SOCAT index csv copied there):
#   sbatch extract_SOCAT.sbatch                                  # all years, defaults
#   sbatch extract_SOCAT.sbatch --years 2014 --filename test.nc  # quick test on one year
#   sbatch extract_SOCAT.sbatch --outpath /global/cfs/.../ --filename synSOCAT_matches.nc

set -eo pipefail

module load python
conda activate synArgo

echo "Job $SLURM_JOB_ID on $(hostname)"
echo "Started: $(date)"

python -u extract_SOCAT.py "$@"

echo "Finished: $(date)"
