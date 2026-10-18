"""
Build realworldINDEX: ~10-day record indices for each float, split by year.

For each real-world deployment, take every nth record (for approx. 10 days, n=59) 
over the float's real lifetime, then split into year columns ('2014_records', ...) 

Yearly columns hold indices for that year's Synthetic_Argo_Data_<yr>.nc file. 
Subsampling before splits keeps cycles continuous across year boundaries.

To run in Terminal:
    python build_realworldINDEX.py
"""
import argparse
import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm
import os

########## User options ##########

record_interval = int(59)                  # 60 for exactly 10 days 
extract_years = range(2014, 2024)          # study period to extract
search_years = [2000, 2023]                # add buffer before study period to search deployments

# Choose folder 
save_folder = '/global/cfs/cdirs/m4839/sangsong/crusoe_carbon_validation/' + f'synArgo_{record_interval}record_{search_years[0]}-{search_years[1]}deployments/'

default_colocation = 'float_deployment_realworld_colocation.csv'            # intermediary output file
default_annualrecords = 'float_deployment_realworld_annualrecords.parquet'  # final output file

# Optional to change
records_per_year = 2190     # 4-hourly data, model has no leap years
d_hours = 4                 # hourly interval for full data

working_dir = '/global/cfs/cdirs/m4839/sangsong/crusoe_carbon_validation/'   # working files
syn_dir = '/global/cfs/cdirs/m4839/ktchris/Synthetic_Argo_Dataset/'          # 4-hourly synthetic data
default_global_index = working_dir + 'ar_index_global_prof.txt'

#####################################

def replace_year(dt, year=2010):
    # Function to replace the year, which helps in avoiding leap years
    return dt.replace(year=year)

def get_distances(lon_point, lat_point, lon_arr, lat_arr):  
    """Calculate distance between one point and an array of points.
    PARAMETERS
    ----------
    lon_point, lat_point: coordinates for a single location, float
    lon_arr, lat_arr: coordinates from which to compute the distance from lon_point/lat_point, array/array-like or float
    
    RETURNS
    -------
    distance : array/float, unit: kilometers
    ** Added np.clip() on input to np.arccos(), to fix error where numpy calculates (sin^2 + cos^2) to > 1 and np.arccos(>1) gives nan.
    """
    
    # approximate radius of earth in km
    R = 6371.0

    lat1, lon1, lat2, lon2 = np.deg2rad(lat_point), np.deg2rad(lon_point) , np.deg2rad(lat_arr), np.deg2rad(lon_arr)

    dlon = lon2 - lon1
    dlat = lat2 - lat1
    
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    all_distances = 2*R*np.arctan2(np.sqrt(a), np.sqrt(1 - a)) 

    return all_distances

def colocate_realworld_floats(global_index = default_global_index,
                              syn_dir = '/global/cfs/cdirs/m4839/ktchris/Synthetic_Argo_Dataset/',
                              startyear = search_years[0],
                             endyear = search_years[1]):
    """ From Katy's Useful_codebits.ipynb """

    # Open global index of all float profiles in 'ar_index_global_prof.txt'
    # Create a datetime column rather individual than y, m, d columns
    df = pd.read_csv(default_global_index,sep=',',comment='#',header=0,usecols=[0,1,2,3])
    df['datetime'] = pd.to_datetime(df.pop('date'),format='%Y%m%d%H%M%S')
    
    # Create an index column that contains the float number
    fnums = []
    fnames = df.pop('file')
    for filenum in range(len(fnames)):
        fnums.append(int(fnames.iloc[filenum].split('/')[1]))
    df['fnum'] = fnums
    
    # Round the datetimes to model 4-hourly time steps
    # Skip any deployments on leap day (since that doesn't exist in our model)
    df['ptime'] = df['datetime'].dt.round(freq='4h')
    df = df.where(~((df['ptime'].dt.month==2)&(df['ptime'].dt.day==29))).dropna()
    
    # Group profiles by float
    float_groups = df.groupby('fnum')
    
    # Get the deployment and final profile dates for each float
    minimal = float_groups.min()
    maximal = float_groups.max()
    
    # Calculate the deployment lifetime of each float (rounded to the nearest 4 hours)
    minimal['deplife'] = maximal['datetime'] - minimal['datetime']
    minimal['deplife'] = minimal['deplife'].dt.round(freq='4h')
    
    # Select only deployments that occured within your years of interest
    deploy = minimal.where((minimal['datetime']>= str(startyear)+'-01-01') & (minimal['datetime'] < str(endyear+1)+'-01-01')).dropna()
    
    # Save the deployment latitude and longitude for each float
    for fnumber in tqdm(deploy.index.values):
        oneflt = float_groups.get_group(fnumber)
        deploy.loc[fnumber,['latitude','longitude']] = oneflt[['latitude','longitude']].where(oneflt['datetime'] == deploy['datetime'].loc[fnumber]).dropna().values[0,:]
        del oneflt
    
    del df, fnums, fnames, minimal, maximal
    
    # Create a column for year the float was deployed
    deploy['year'] = deploy.datetime.dt.year # subsample 9 days 20 hours
    
    # Create a column for the time-index value that the float was deployed
    # (e.g. what index to use if searching for this float)
    deploy['doy'] = deploy['ptime'].apply(replace_year,year=2009).dt.dayofyear
    deploy['hour'] = deploy['ptime'].dt.hour
    deploy['record_index'] = ((deploy.pop('doy')-1)*6 + deploy.pop('hour')/4).astype(int)

    # ========= Match up floats 
    
    fltmatch = np.full(deploy['record_index'].shape,np.nan)
    
    # Open each of the saved yearly consolidated files
    sfiles = sorted([syn_dir+f for f in os.listdir(syn_dir) if ".nc" in f])
    sdata = xr.open_mfdataset(sfiles,concat_dim='record',combine='nested',data_vars=['lat','lon'],compat='no_conflicts')
    
    # Load the latitudes and longitudes and get the maximum time step
    # This helps with collocating with the real floats
    slat_slon = sdata[['lat','lon']].load()
    maxdate = sdata['record'].max().values
    
    # Go through each real world float
    for fdep in tqdm(range(0,len(deploy))):
        # Get the time, duration, and location of float deployment
        flt = deploy.iloc[fdep]
        starttime = flt['ptime']
        fltfile = syn_dir+'floats/Synth{}_{}.nc'.format(int(flt.name),starttime.strftime("%Y%m%d%H%M"))
        
        # Select the synthetic timestep for deployemnt
        tstep_ll = slat_slon.sel(record=starttime)
        if abs(np.sum(tstep_ll['lat'])) > 0:
            # Get the nearest synthetic float to the real world deployment
            dds = get_distances(flt['longitude'],flt['latitude'], 
                                            tstep_ll['lon'].values, tstep_ll['lat'].values)
            closept = np.where(dds == np.min(dds))[0]
            
            fltmatch[fdep] = int(closept[0])
            
    deploy['nParticles_index'] = fltmatch
    deploy.to_csv(default_colocation)


def get_record_list(single_float, step=record_interval):
    """Record indices from deployment to end of life, every `step` records.
    Indices count from the first record of the deployment year.
    """
    start_record = int(single_float['record_index'])

    ndays = float(single_float['deplife'].split('days')[0])
    nhours = float(single_float['deplife'].split('days')[1].split(':')[0])

    nrecords = ndays*(24/d_hours) + (nhours/d_hours)
    record_list = [int(x) for x in np.arange(start_record, start_record + nrecords + 1, step)]
    return record_list


def split_yearly_records(record_list, start_year=2014, extract_years=extract_years):
    """Split records counted from the deployment year into per-year indices.
    Years with no records are None.
    """
    records_by_year = {k:[] for k in extract_years}

    for record in record_list:
        year_offset = int(record) // records_per_year
        year = start_year + year_offset
        adjusted_index = int(record % records_per_year)
        if year in extract_years:
            records_by_year[year].append(adjusted_index)
    for k, v in records_by_year.items():
        if len(v) == 0:
            records_by_year[k] = None
    return records_by_year


def add_yearly_columns(realworldINDEX, extract_years=extract_years):
    """Add year-indexed columns of 10-day record indices to dataframe
    """
    for yr in extract_years:
        realworldINDEX[f'{yr}_records'] = pd.Series([None]*len(realworldINDEX),
                                                 index=realworldINDEX.index, dtype=object)

    for fnum, single_float in realworldINDEX.iterrows():
        record_list = get_record_list(single_float)
        start_year = int(single_float['year'])
        yearly_records = split_yearly_records(record_list, start_year, extract_years)

        for yr, indices in yearly_records.items():
            realworldINDEX.at[fnum, f'{yr}_records'] = indices

    return realworldINDEX


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--outpath', default = save_folder)
    parser.add_argument('--deployments', default=None)
    parser.add_argument('--annrecords', default=None)
    parser.add_argument('--syn_dir', default=syn_dir)
    parser.add_argument('--startyear', type=int, default=search_years[0])
    parser.add_argument('--endyear', type=int, default=search_years[1])

    args = parser.parse_args()
    
    if args.deployments is None:
        args.deployments = os.path.join(args.outpath, default_colocation)
    if args.annrecords is None:
        args.annrecords = os.path.join(args.outpath, default_annualrecords)
        
    colocate_realworld_floats(syn_dir = args.syn_dir, 
                                        startyear=args.startyear, 
                                        endyear=args.endyear)
    deployments = pd.read_csv(args.deployments)
    deployments = deployments.dropna(subset=['nParticles_index'])
    deployments['fnum'] = deployments['fnum'].astype(int)
    deployments['nParticles_index'] = deployments['nParticles_index'].astype(int)
    deployments = deployments.set_index('fnum')

    realworldINDEX = add_yearly_columns(deployments)
    realworldINDEX.to_parquet(args.out)

    print(f'Saved realworldINDEX for {len(realworldINDEX)} floats to {args.out}')
    for yr in extract_years:
        print(f'  {yr}: {realworldINDEX[f"{yr}_records"].notna().sum()} floats')


if __name__ == "__main__":
    main()
