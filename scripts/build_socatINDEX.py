"""
Build socatINDEX: match each SOCAT sample to the nearest synthetic float (particle) at its time.

Same matching approach as build_realworldINDEX.py, but match by (daily-resampled) points
instead of fovalid_dsowing model float trajectories. 

Writes the SOCAT index (csv) that wivalid_ds be read by extract_SOCAT.py 
(one row per sample, with 'sample_datetag', 'year', 'record_index' and 'nParticles_index').

Input (--socat): csv or parquet with one row per SOCAT sample
            Must have columns sample_datetag, datetime, latitude, longitude
            (uploaded)

To run in Terminal:
    python build_socatINDEX.py --socat socat_trainval_samples_2014-2023_acc<date>.csv
    python build_socatINDEX.py --socat <file> --years 2014 --out test_socatINDEX.csv

"""
import argparse
import os
from datetime import datetime

import numpy as np
import pandas as pd
import xarray as xr

########## User options ########## 

working_dir = '/global/cfs/cdirs/m4839/sangsong/github-repos/argo-synthetic-extraction/'   # working files
syn_dir = '/global/cfs/cdirs/m4839/ktchris/Synthetic_Argo_Dataset/'         # 4-hourly synthetic data
default_out = working_dir + 'outputs/socatINDEX_syn_colocation.csv'                  # extract_SOCAT.py default_index
records_per_year = 2190     # 4-hourly data, model has no leap years

###################################


def replace_year(dt, year=2010):
    # Function to replace the year, which helps in avoiding leap years
    return dt.replace(year=year)


def get_distances(lon_point, lat_point, lon_arr, lat_arr):
    """Great-circle distance (km) between one point and an array of points (as build_realworldINDEX.py)."""
    R = 6371.0
    lat1, lon1, lat2, lon2 = (np.deg2rad(lat_point), np.deg2rad(lon_point), 
                                np.deg2rad(lat_arr), np.deg2rad(lon_arr))
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    return 2 * R * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def add_record_index(socatINDEX):
    """Model time step (ptime), year and record_index of each sample; drops 29 Feb samples."""

    socatINDEX['datetime'] = pd.to_datetime(socatINDEX['datetime'])
    socatINDEX['ptime'] = socatINDEX['datetime'].dt.round(freq='4h')
    
    leap = (socatINDEX['ptime'].dt.month == 2) & (socatINDEX['ptime'].dt.day == 29)
    if leap.any():
        print(f'Skipping {int(leap.sum())} samples on 29 Feb (no leap days in the model)')
    socatINDEX = socatINDEX[~leap].copy()
    
    socatINDEX['year'] = socatINDEX['ptime'].dt.year
    doy = socatINDEX['ptime'].apply(replace_year, year=2009).dt.dayofyear
    socatINDEX['record_index'] = ((doy - 1) * 6 + socatINDEX['ptime'].dt.hour / 4).astype(int)

    return socatINDEX


def match_year(yearINDEX, yr, syn_dir):
    """Nearest synthetic particle to each sample of one year; returns (nParticles_index, match_km)."""
    
    ds = xr.open_dataset(os.path.join(syn_dir, f'Synthetic_Argo_Data_{yr}.nc'))
    records = np.unique(yearINDEX['record_index'].values)

    # Only the records with samples: (n_records, nParticles) lat/lon
    valid_ds = ds[['lat', 'lon']].isel(record=records).load()
    row_of = {r: i for i, r in enumerate(records)}

    # Check that record_index points at the sample's model time step 
    try:
        file_tags = pd.to_datetime(valid_ds['record'].values).strftime('%m%d%H')
        expected = yearINDEX.groupby('record_index')['ptime'].first().reindex(records)
        n_bad = int((file_tags != expected.dt.strftime('%m%d%H').values).sum())
        print(f'  record_index check: {len(records) - n_bad} of {len(records)} records match the sample time step')
    except (KeyError, TypeError, ValueError):
        print('  record_index check skipped (record times not readable as dates)')

    # Store the index of nearest particle and separation distance 
    match = np.full(len(yearINDEX), np.nan)
    dist = np.full(len(yearINDEX), np.nan)
    lat, lon = valid_ds['lat'].values, valid_ds['lon'].values
    for ind, (record, slat, slon) in enumerate(zip(yearINDEX['record_index'].values,
                                             yearINDEX['latitude'].values, 
                                             yearINDEX['longitude'].values)):
        plat, plon = lat[row_of[record]], lon[row_of[record]]
        if abs(np.nansum(plat)) > 0: # particles exist at this time step
            dds = get_distances(slon, slat, plon, plat)
            if np.isfinite(dds).any():
                match[ind] = int(np.nanargmin(dds))
                dist[ind] = np.nanmin(dds)
    ds.close()
    return match, dist


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--socat', required=True, 
                        help='SOCAT samples (csv or parquet): sample_datetag, datetime, latitude, longitude')
    parser.add_argument('--syn_dir', default=syn_dir)
    parser.add_argument('--out', default=default_out, 
                        help='Output csv (extract_SOCAT.py --index)')
    parser.add_argument('--years', type=int, nargs='+', default=None, 
                        help='Only these years (default: avalid_ds with a synthetic file)')
    args = parser.parse_args()

    socat = pd.read_parquet(args.socat) if args.socat.endswith('.parquet') else pd.read_csv(args.socat)
    if not socat['sample_datetag'].is_unique:
        raise ValueError('sample_datetag values in the SOCAT samples are not unique')
    socatINDEX = add_record_index(socat)
    if args.years is not None:
        socatINDEX = socatINDEX[socatINDEX['year'].isin(args.years)]

    # Process by year 
    socatINDEX['nParticles_index'] = np.nan
    socatINDEX['match_km'] = np.nan
    for yr, yearINDEX in socatINDEX.groupby('year'):
        if not os.path.exists(os.path.join(args.syn_dir, f'Synthetic_Argo_Data_{yr}.nc')):
            print(f'No synthetic file for {yr}, skipping {len(yearINDEX)} samples')
            continue
        print(f'Matching {len(yearINDEX)} samples for {yr}: ', datetime.now().strftime('%Y%m%d %H:%M:%S'))
        match, dist = match_year(yearINDEX, yr, args.syn_dir)
        socatINDEX.loc[yearINDEX.index, 'nParticles_index'] = match
        socatINDEX.loc[yearINDEX.index, 'match_km'] = dist

    socatINDEX = socatINDEX.reset_index(drop=True)
    socatINDEX.to_csv(args.out)
    
    n = int(socatINDEX['nParticles_index'].notna().sum())
    print(f'Saved {len(socatINDEX)} samples ({n} matched to a synthetic particle; median distance '
          f'{socatINDEX["match_km"].median():.1f} km) to {args.out}')
    
    for yr, g in socatINDEX.groupby('year'):
        print(f'  {yr}: {g["nParticles_index"].notna().sum()} of {len(g)} matched')


if __name__ == "__main__":
    main()
