"""
[Run after build_realworldINDEX.py]

Extract one year of 10-day synthetic Argo profiles for the real-world floats.

Reads realworldINDEX parquet and pulls each float's records from Synthetic_Argo_Data_<yr>.nc
Writes one file with dimensions (fnum, profnum, nVertLevels). 

'profnum' keeps track of float's profiles within the year (0, 1, 2, ...); 
floats with fewer profiles that year are padded with NaN. 
Actual profile times are in time(fnum, profnum).

Usage (normally called by extract_synargo_years.sbatch):
    python extract_synargo_year.py 2014
"""

import argparse
import os
from datetime import datetime

import numpy as np
import pandas as pd
import xarray as xr

######### User options ##########

default_data_path = '/global/cfs/cdirs/m4839/ktchris/Synthetic_Argo_Dataset/'
default_outpath = '/global/cfs/cdirs/m4839/sangsong/crusoe_carbon_validation/' + 'synArgo_59record_2000-2023deployments/'
default_index = 'float_deployment_realworld_annualrecords.parquet'

##################################

def build_profile_indices(yearINDEX, yrcol):
    """Record and particle index for every (fnum, profnum) slot.

    Floats with fewer profiles are padded with record 0; `valid` marks the
    real profiles so the padding can be masked to NaN afterwards.
    """
    nfloats = len(yearINDEX)
    max_prof = yearINDEX[yrcol].map(len).max()

    record_idx = np.zeros((nfloats, max_prof), dtype=int)
    valid = np.zeros((nfloats, max_prof), dtype=bool)
    for i, records in enumerate(yearINDEX[yrcol]):
        record_idx[i, :len(records)] = records
        valid[i, :len(records)] = True

    particle_idx = np.repeat(yearINDEX['nParticles_index'].to_numpy()[:, None], max_prof, axis=1)
    return record_idx, particle_idx, valid


def extract_year(yr, deployINDEX, data_path):
    yrcol = f'{yr}_records'
    synargo_4hourly = xr.open_dataset(os.path.join(data_path, f'Synthetic_Argo_Data_{yr}.nc'))

    # Retrieve floats that have data for this given year
    yearINDEX = deployINDEX[deployINDEX[yrcol].notna()]
    print(f'Processing {len(yearINDEX)} valid floats...')

    # Floats profile at different times, so line them up by profile number
    # instead of time, and keep each float's times as a variable.
    record_idx, particle_idx, valid = build_profile_indices(yearINDEX, yrcol)
    dims = ('fnum', 'profnum')
    coords = {'fnum': yearINDEX.index.to_numpy(), 'profnum': np.arange(record_idx.shape[1])}
    record_da = xr.DataArray(record_idx, dims=dims, coords=coords)
    particle_da = xr.DataArray(particle_idx, dims=dims, coords=coords)
    valid_da = xr.DataArray(valid, dims=dims, coords=coords)

    times = np.where(valid, synargo_4hourly['record'].values[record_idx], np.datetime64('NaT'))
    annualDS = xr.Dataset(coords={**coords,
                                  'nParticles_index': ('fnum', yearINDEX['nParticles_index'].to_numpy()),
                                  'time': (dims, times)})

    # fnum comes from deployINDEX, so skip the model's fnum variable
    float_vars = [var for var in synargo_4hourly.data_vars
                  if {'record', 'nParticles'} <= set(synargo_4hourly[var].dims) and var != 'fnum']

    # Read each variable whole (one sequential read is much faster than many
    # small scattered reads per float), then pick out all floats' profiles at once
    for nvar, var in enumerate(float_vars, start=1):
        da = synargo_4hourly[var].compute() # change from .load()
        annualDS[var] = (da.isel(record=record_da, nParticles=particle_da)
                           .drop_vars('record') # model times are already in 'time'
                           .where(valid_da))
        del da
        print(f'  {nvar}/{len(float_vars)} variables ({var}): ', datetime.now().strftime('%Y%m%d %H:%M:%S'))

    synargo_4hourly.close()
    print('Writing file...')
    return annualDS.transpose('fnum', 'profnum', 'nVertLevels', ...)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('year', type=int)
    parser.add_argument('--index', default=None,
                        help=f'deployINDEX parquet file (default: <outpath>/{default_index})')
    parser.add_argument('--datapath', default=default_data_path)
    parser.add_argument('--outpath', default=default_outpath)
    args = parser.parse_args()

    if args.index is None:
        args.index = os.path.join(args.outpath, default_index)
        
    print(f'Retrieving float data for year {args.year}: ', datetime.now().strftime('%Y%m%d %H:%M'))
    deployINDEX = pd.read_parquet(args.index)

    annualDS = extract_year(args.year, deployINDEX, args.datapath)

    os.makedirs(args.outpath, exist_ok=True)
    outfile = os.path.join(args.outpath, f'synArgo_realworld_matches_10day_yr{args.year}.nc')
    annualDS.to_netcdf(outfile)

    print(f'Saved {int(annualDS.time.notnull().sum())} profiles from '
          f'{annualDS.sizes["fnum"]} floats to {outfile}')
    print(f'Finished retrieving year {args.year}: ', datetime.now().strftime('%Y%m%d %H:%M'))


if __name__ == "__main__":
    main()
