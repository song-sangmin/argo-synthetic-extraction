
# Instructions

Current files use working_dir = '/global/cfs/cdirs/m4839/sangsong/crusoe_carbon_validation/'

The file 'ar_index_global_prof.txt' contains the date and time of all float profiles, downloaded from Coriolis.


## To run once during setup: 

Make environment (or change in .sbatch files):
```
cd /global/cfs/cdirs/m4839/sangsong/crusoe_carbon_validation/
conda env create -f environment.yml     # create synArgo env
```

Optional to change: logs currently stored in "/pscratch/sd/s/sangsong/synargo_logs". 
```
mkdir -p /pscratch/sd/s/sangsong/synargo_logs     # create logs folder once
```

## Colocate real world floats to synthetic data 

Use `build_realworldINDEX.py`, set --outpath to choose where results ('float_deployment_realworld_annualrecords.parquet') should be saved. Set arguments --startyear and --endyear to specify years to search deployments from. (Run time <10 min) 

```
cd /global/cfs/cdirs/m4839/sangsong/crusoe_carbon_validation/
module load python && conda activate synArgo

python build_realworldINDEX.py             # for default search 2000-2023

python build_realworldINDEX.py --startyear 2023 --endyear 2023 --outpath '/global/cfs/cdirs/m4839/sangsong/crusoe_carbon_validation/synArgo_59record_2023-2024deployments/'  
```

python build_realworldINDEX.py --startyear 2023 --endyear 2024 --outpath '/global/cfs/cdirs/m4839/sangsong/crusoe_carbon_validation/synArgo_59record_2000-2023deployments/'  


The file 'float_deployment_realworld_annualrecords.parquet' has record indices for each float by year. 


## Main run to extract annual data

Use `extract_synargo_years.sbatch` to retrieve annnual records. Set --outpath argument to the save location you specified in previous step. (Run time <1 hour for 10 years) 

```
cd /global/cfs/cdirs/m4839/sangsong/crusoe_carbon_validation/
module load python && conda activate synArgo


sbatch --array=2014 extract_synargo_years.sbatch         # ex: run one year

sbatch --array=2015-2023 extract_synargo_years.sbatch    # rerun select years if needed

sbatch extract_synargo_years.sbatch --outpath  '/global/cfs/cdirs/m4839/sangsong/crusoe_carbon_validation/synArgo_59record_2000-2023deployments/'    

```

### Checking logs during run if debugging (optional)

```
tail -f /pscratch/sd/s/sangsong/synargo_logs/synargo_<ARRAYJOBID>_<YEAR>.out
tail -f /pscratch/sd/s/sangsong/synargo_logs/synargo_59325157_2015.out
```

If notebook crashes the tail command can cause Terminal to lag. 

## Merging years into a single file 

Use `merge_synargo_years.sbatch` and set --path argument to the folder where extracted annual files are saved. Default output is saved as `synArgo_realworld_matches_merged.nc`, or set file name with --out argument.

```
sbatch merge_synargo_years.sbatch --path '/global/cfs/cdirs/m4839/sangsong/crusoe_carbon_validation/synArgo_59record_2000-2023deployments/' 

```

## Synthetic SOCAT: colocate SOCAT samples to synthetic floats, then extract

1. Locally, export the SOCAT samples (one row per sample: sample_datetag, datetime, latitude,
   longitude) and upload the csv to the working directory, e.g.
   `socat_trainval_samples_2014-2023_acc<date>.csv` (SOCAT rows of the P2 combined trainval file).
2. Use `build_socatINDEX.py` to match each sample to the nearest synthetic particle at its model
   time step. It writes `socatINDEX_syn_colocation.csv`, the default `--index` of `extract_SOCAT.py`.
   (Reads only the records with samples from each Synthetic_Argo_Data_<yr>.nc.)
3. Use `extract_SOCAT.sbatch` to pull the matched synthetic profiles.

```
cd /global/cfs/cdirs/m4839/sangsong/crusoe_carbon_validation/
module load python && conda activate synArgo

python build_socatINDEX.py --socat socat_trainval_samples_2014-2023_acc<date>.csv
python build_socatINDEX.py --socat <file> --years 2014 --out test_socatINDEX.csv    # quick test

sbatch extract_SOCAT.sbatch --filename synSOCAT-daily_matches.nc
```

## Accessing dataframes without nan padding

from merge_synargo_years import to_profile_dataframe
ds = xr.open_dataset('.../synArgo_realworld_matches_10day_2014-2023.nc')

df_2d = to_profile_dataframe(ds, variables='2d') # surface/per-profile variables only 