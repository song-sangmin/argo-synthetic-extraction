"""
[Run after extract_synargo_years.sbatch]

Join the yearly 10-day files into one dataset with a running profile number.
After running, 'profnum' runs continuously from the float's first profile 
(0, 1, 2, ... across all years), and writes one file with dimensions (fnum, profnum, nVertLevels). 
Floats with fewer profiles than longest-lived float are padded with NaN (time is NaT).

Usage (normally called by merge_synargo_years.sbatch):
    python merge_synargo_years.py

To convert to a DataFrame without the padding, see to_profile_dataframe().
"""
import argparse
import os
from datetime import datetime

import numpy as np
import pandas as pd
import xarray as xr


default_path = '/global/cfs/cdirs/m4839/sangsong/crusoe_carbon_validation/' + 'synArgo_59record_2000-2023deployments/'

year_range = range(2014, 2024)


def yearly_file(path, yr):
    return os.path.join(path, f'synArgo_realworld_matches_10day_yr{yr}.nc')


def count_profiles(path, years):
    """Profiles per float (rows) per year (columns); time is NaT for padding."""
    counts = {}
    for yr in years:
        with xr.open_dataset(yearly_file(path, yr)) as ds:
            counts[yr] = ds.time.notnull().sum('profnum').to_series()
    return pd.DataFrame(counts).fillna(0).astype(int).sort_index()


def merge_years(path, years):
    counts = count_profiles(path, years)
    # Running profile number at which each float's profiles for each year start
    offsets = counts.cumsum(axis=1) - counts
    fnums = counts.index.to_numpy()
    max_prof = int(counts.sum(axis=1).max())
    print(f'{len(fnums)} floats, {counts.values.sum()} profiles, up to {max_prof} per float')

    # Empty (fnum, profnum, ...) grid for each variable, using the first year as a template
    with xr.open_dataset(yearly_file(path, years[0])) as template:
        grids = {}
        for var in template.data_vars:
            extra_dims = template[var].dims[2:] # e.g. nVertLevels
            shape = (len(fnums), max_prof) + tuple(template.sizes[d] for d in extra_dims)
            grids[var] = (('fnum', 'profnum') + extra_dims, np.full(shape, np.nan),
                          template[var].attrs)
        nVertLevels = template['nVertLevels'].values
        global_attrs = template.attrs
    times = np.full((len(fnums), max_prof), np.datetime64('NaT'), dtype='datetime64[ns]')
    nParticles_index = pd.Series(-1, index=fnums)

    for yr in years:
        with xr.open_dataset(yearly_file(path, yr)) as ds:
            ds = ds.load()
            # Real profiles in this year's file: (float, local profnum) pairs.
            # Each float's profiles fill profnum 0..n-1 in time order, padding after.
            yr_fi, yr_pi = np.nonzero(ds.time.notnull().values)
            rows = np.searchsorted(fnums, ds.fnum.values)[yr_fi]
            cols = offsets[yr].to_numpy()[rows] + yr_pi

            for var, (dims, grid, attrs) in grids.items():
                grid[rows, cols] = ds[var].transpose(*(('fnum', 'profnum') + dims[2:])).values[yr_fi, yr_pi]
            times[rows, cols] = ds.time.values[yr_fi, yr_pi]
            nParticles_index[ds.fnum.values] = ds.nParticles_index.values
        print(f'  {yr}: {len(rows)} profiles from {counts[yr].gt(0).sum()} floats: ',
              datetime.now().strftime('%Y%m%d %H:%M:%S'))

    assert np.count_nonzero(~np.isnat(times)) == counts.values.sum(), 'profiles lost or overwritten'

    return xr.Dataset({var: (dims, grid, attrs) for var, (dims, grid, attrs) in grids.items()},
                      coords={'fnum': fnums,
                              'profnum': np.arange(max_prof),
                              'nVertLevels': nVertLevels,
                              'nParticles_index': ('fnum', nParticles_index.to_numpy()),
                              'time': (('fnum', 'profnum'), times)},
                      attrs=global_attrs)


def to_profile_dataframe(ds, variables=None):
    """DataFrame with one row per real profile (and level), skipping NaN padding.

    Converting the (fnum, profnum) grid directly with ds.to_dataframe() would
    also make rows for every padded slot (~2/3 of the grid).
    :param variables: optional list of variables to keep, e.g. only 2D variables
                      to get one row per profile instead of one row per level
    """

    if variables == '2d':
        ds = ds[['lat', 'lon', 'ifrac', 'MLD', 'atmPressure', 'CO2_gas_flux', 
                'pCO2surface', 'dpCO2', 'atmosphericCO2', 'wind_speed']]
    elif variables is not None:
        ds = ds[variables]

    stacked = ds.stack(prof=('fnum', 'profnum'))
    stacked = stacked.isel(prof=stacked.time.notnull().values)
    return stacked.to_dataframe()


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--path', default=default_path, help='Folder with the yearly files')
    parser.add_argument('--out', default=None,
                        help='Output file (default: <path>/synArgo_realworld_matches_merged.nc)')
    args = parser.parse_args()

    
    out = args.out or os.path.join(args.path, 'synArgo_realworld_matches_merged.nc')

    print('Merging yearly files: ', datetime.now().strftime('%Y%m%d %H:%M:%S'))
    allDS = merge_years(args.path, list(year_range))

    print('Writing file...')
    # Compression mostly shrinks the NaN padding
    encoding = {var: {'zlib': True, 'complevel': 4} for var in allDS.data_vars}
    allDS.to_netcdf(out, encoding=encoding)
    print(f'Saved to {out}: ', datetime.now().strftime('%Y%m%d %H:%M:%S'))


if __name__ == "__main__":
    main()
