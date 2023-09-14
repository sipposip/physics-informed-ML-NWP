#! /pfs/nobackup/home/s/sebsc/miniconda3/envs/nn-reanalysis-env/bin/python
#SBATCH -A SNIC2019-3-611
#SBATCH --time=24:00:00
#SBATCH -N 1
#SBATCH --chdir=/home/s/sebsc/pfs/nn_reanalysis

"""
compute Northern Hemisphere RMSE per forecast (and per leadtime)
todo 64 bit?
"""

import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
import xarray as xr
import matplotlib
import gc
matplotlib.use('agg')
from pylab import plt
from dask.diagnostics import ProgressBar


pbar = ProgressBar()
pbar.register()

inpath = '/climstorage/sebastian/nn_reanalysis/era5_benchmarkdata/'

startyear = 1979
endyear = 2018
test_startyear=2017
test_endyear=2018
nettype = sys.argv[1]
resolution = sys.argv[2]
i_train = int(sys.argv[3])
tres = 6  # hours
# specify variables and levels. level can be a single number, or a list (not tuple!) of numbers
# here we need this only for filenames
var_and_lev_list = [['geopotential', [1000, 700, 500, 300]], ['temperature', [850]],
                    ['toa_incident_solar_radiation', 'single']]
# for filenames
var_lev_str = ''.join(str(e) for a in var_and_lev_list for e in a).replace(' ', ''). \
    replace('[', '').replace(']', '_').replace(',', '_')


net_fc_path = f'/proj/bolinc/users/x_sebsc/nn_reanalysis/eval/{nettype}_{resolution}/'

#open fcs
init_dates = pd.date_range(f'{test_startyear}0101-0600', f'{test_endyear}1231-1200', freq=f'{tres}h')

fc_ifiles = [f'{net_fc_path}/fc_{nettype}_{resolution}_itrain{i_train}_{date.strftime("%Y%m%d%H")}.nc' for date in init_dates]

# # xr.set_options(file_cache_maxsize=1)
fcs = xr.open_mfdataset(fc_ifiles, concat_dim='time_init', combine='nested',
                        engine='netcdf4')

fcs = fcs['__xarray_dataarray_variable__']

fcs['time_valid'] = fcs['time_valid'].compute()


# # save comboined files to disk as a single file, and reload again (otherwise we get oom problems)
# # due to a bug in xarray, we need to convert time_valid to float for this
# fcs['time_valid'] = fcs['time_valid'].astype('float')
# fcs.to_netcdf(f'{net_fc_path}/fc_combined_{nettype}_{resolution}_itrain{i_train}.nc')
# fcs = xr.open_dataarray(f'{net_fc_path}/fc_combined_{nettype}_{resolution}_itrain{i_train}.nc',
#                         chunks={'time_init':100, 'lev':1})
# # and convert back to datetime
# fcs['time_valid'] = fcs['time_valid'].astype('datetime64[ns]')


# select only northern hemisphere

fcs = fcs.sel(lat=slice(0,90))


# the first timetsep in each forecast is the analysis

# score function from WeatherBench

def compute_weighted_rmse(da_fc, da_true, mean_dims=xr.ALL_DIMS):
    """
    Compute the RMSE with latitude weighting from two xr.DataArrays.
    Args:
        da_fc (xr.DataArray): Forecast. Time coordinate must be validation time.
        da_true (xr.DataArray): Truth.
        mean_dims: dimensions over which to average score
    Returns:
        rmse: Latitude weighted root mean squared error
    """
    error = da_fc - da_true
    weights_lat = np.cos(np.deg2rad(error.lat))
    weights_lat /= weights_lat.mean()
    rmse = np.sqrt(((error)**2 * weights_lat).mean(mean_dims))
    return rmse


truth = fcs[:,0]

# we want to have valid_time as coordinate, which for compatibility with
# WeatheBench we simply call "time"
truth = truth.rename({'time_init':'time'})
truth['time'] = truth['time_valid']

vars = fcs.lev.values

res = []
for fcday in tqdm(fcs.fcday.values):
    fc = fcs.sel(fcday=fcday)


    fc = fc.rename({'time_init': 'time'})
    fc['time'] = fc['time_valid']

    # both truth and fc have  "time" dimension, and in the scoring
    # functions they are automatically aligned, so dont need to crop the last/first couple
    # of timesteps

    for ivar, varname in enumerate(vars):
        rmse = compute_weighted_rmse(fc.isel(lev=ivar), truth.isel(lev=ivar),
                                     mean_dims=('lat','lon'))

        _res = pd.DataFrame({'rmse':rmse.values,
                             'fcday':fcday,
                             'var':varname,
                             'time_valid':rmse.time,
                             'member':i_train}, index=rmse.time.values)
        res.append(_res)
        plt.close('all')
        gc.collect()

res = pd.concat(res)

res.to_pickle(f'data/fc_res_per_fc_NH_{nettype}_{resolution}_itrain{i_train}.pkl')
res.to_csv(f'data/fc_res__per_fc_NH_{nettype}_{resolution}_itrain{i_train}.csv')

