"""
Extract the synthetic profile matched to each SOCAT sample.

Reads in the SOCAT index from build_socatINDEX.py (includes sample_datetag identifier), 
which stores the index of the nearest synthetic particle for each sample.

For each year, pull every sample's matched (record, nParticles) profile from 
Synthetic_Argo_Data_<yr>.nc, then write one file with dimensions (sample_datetag, nVertLevels).
Model time of each sample is in time(sample_datetag).

Usage (normally called by extract_SOCAT.sbatch):
    python extract_SOCAT.py
    python extract_SOCAT.py --years 2014 2015 --filename test.nc
"""
import argparse
import os
from datetime import datetime

import numpy as np
import pandas as pd
import xarray as xr

default_index = '/global/cfs/cdirs/m4839/sangsong/github-repos/argo-synthetic-extraction/outputs/socatINDEX_syn_colocation.csv'  # build_socatINDEX.py output
default_data_path = '/global/cfs/cdirs/m4839/ktchris/Synthetic_Argo_Dataset/'
default_outpath = '/global/cfs/cdirs/m4839/sangsong/github-repos/argo-synthetic-extraction/outputs/'
default_filename = 'synSOCAT_matches.nc'
default_levels = [0, 1, 2]


def extract_year(yr, yearINDEX, data_path, levels):
    synargo_4hourly = xr.open_dataset(os.path.join(data_path, f'Synthetic_Argo_Data_{yr}.nc'))

    # An isel with DataArray indexers picks one (record, nParticles) pair per
    # sample and puts the results along a new 'sample_datetag' dim
    tags = {'sample_datetag': yearINDEX['sample_datetag'].to_numpy()}
    record_idx = yearINDEX['record_index'].astype(int).to_numpy()
    particle_idx = yearINDEX['nParticles_index'].astype(int).to_numpy()
    record_da = xr.DataArray(record_idx, dims='sample_datetag', coords=tags)
    particle_da = xr.DataArray(particle_idx, dims='sample_datetag', coords=tags)

    yearDS = xr.Dataset(coords={**tags,
                                'record_index': ('sample_datetag', record_idx),
                                'nParticles_index': ('sample_datetag', particle_idx),
                                'time': ('sample_datetag', synargo_4hourly['record'].values[record_idx])})

    float_vars = [var for var in synargo_4hourly.data_vars
                  if {'record', 'nParticles'} <= set(synargo_4hourly[var].dims)]

    # Read each variable whole (one sequential read is much faster than many
    # small scattered reads per sample), then pick out all samples at once
    for nvar, var in enumerate(float_vars, start=1):
        da = synargo_4hourly[var]
        if 'nVertLevels' in da.dims:
            da = da.isel(nVertLevels=levels)
        # compute() rather than load(): load() also keeps the data in
        # synargo_4hourly, so every variable would stay in memory
        da = da.compute()
        yearDS[var] = (da.isel(record=record_da, nParticles=particle_da)
                         .drop_vars('record', errors='ignore')) # model times are already in 'time'
        del da
        print(f'  {nvar}/{len(float_vars)} variables ({var}): ', datetime.now().strftime('%Y%m%d %H:%M:%S'))

    synargo_4hourly.close()
    return yearDS


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--index', default=default_index, help='SOCAT index csv with synthetic matches')
    parser.add_argument('--data-path', default=default_data_path)
    parser.add_argument('--outpath', default=default_outpath, help='Folder to save output in')
    parser.add_argument('--filename', default=default_filename, help='Output netCDF name')
    parser.add_argument('--years', type=int, nargs='+', default=None,
                        help='Only extract these years (default: every year in the index)')
    parser.add_argument('--levels', type=int, nargs='+', default=default_levels,
                        help='nVertLevels to keep (default: 0 1 2)')
    args = parser.parse_args()

    socatINDEX = pd.read_csv(args.index, index_col=0)

    # Drop samples without a matched synthetic float
    socatINDEX = socatINDEX.dropna(subset=['nParticles_index'])
    if not socatINDEX['sample_datetag'].is_unique:
        raise ValueError('sample_datetag values in the index are not unique')
    socatINDEX['year'] = socatINDEX['year'].astype(int)
    if args.years is not None:
        socatINDEX = socatINDEX[socatINDEX['year'].isin(args.years)]

    samples_by_year = []
    for yr, yearINDEX in socatINDEX.groupby('year'):
        if not os.path.exists(os.path.join(args.data_path, f'Synthetic_Argo_Data_{yr}.nc')):
            print(f'No synthetic file for {yr}, skipping {len(yearINDEX)} samples')
            continue
        print(f'Retrieving {len(yearINDEX)} samples for year {yr}: ', datetime.now().strftime('%Y%m%d %H:%M:%S'))
        samples_by_year.append(extract_year(yr, yearINDEX, args.data_path, args.levels))

    socatDS = xr.concat(samples_by_year, dim='sample_datetag')
    socatDS = socatDS.transpose('sample_datetag', 'nVertLevels', ...)

    print('Writing file...')
    os.makedirs(args.outpath, exist_ok=True)
    outfile = os.path.join(args.outpath, args.filename)
    socatDS.to_netcdf(outfile)

    print(f'Saved {socatDS.sizes["sample_datetag"]} samples to {outfile}: ',
          datetime.now().strftime('%Y%m%d %H:%M:%S'))


if __name__ == "__main__":
    main()
