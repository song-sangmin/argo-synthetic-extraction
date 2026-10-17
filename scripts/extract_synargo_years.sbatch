#!/bin/bash
#SBATCH --account=m4839
#SBATCH --qos=shared
#SBATCH --constraint=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --array=2014-2023
#SBATCH --job-name=synargo
#SBATCH --output=/pscratch/sd/s/sangsong/synargo_logs/%x_%A_%a.out
#
# Extract 10-day synthetic Argo profiles, one array task per year.
# Each task reads one Synthetic_Argo_Data_<yr>.nc file and writes one output file.
#
# Usage (from the folder with extract_synargo_year.py, after running build_deployINDEX.py once):
#   sbatch extract_synargo_years.sbatch                 # all years
#   sbatch --array=2019 extract_synargo_years.sbatch    # rerun a single year
#   sbatch --array=2014,2016 extract_synargo_years.sbatch

set -eo pipefail

module load python
conda activate synArgo 

YEAR=$SLURM_ARRAY_TASK_ID
echo "Job $SLURM_ARRAY_JOB_ID task $YEAR on $(hostname)"
echo "Started: $(date)"

python -u extract_synargo_year.py "$YEAR"

echo "Finished: $(date)"
