

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

os.system(f'mkdir -p plots')


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


# the following is needed on kebnekaise,
# but not on tetralith. (on tetralith it actually doesnt work with the following, but it works
# without it)

# # save comboined files to disk as a single file, and reload again (otherwise we get oom problems)
# # due to a bug in xarray, we need to convert time_valid to float for this
# fcs['time_valid'] = fcs['time_valid'].astype('float')
# fcs.to_netcdf(f'{net_fc_path}/fc_combined_{nettype}_{resolution}_itrain{i_train}.nc')
# fcs = xr.open_dataarray(f'{net_fc_path}/fc_combined_{nettype}_{resolution}_itrain{i_train}.nc',
#                         chunks={'time_init':100, 'lev':1})
#
# # and convert back to datetime
# fcs['time_valid'] = fcs['time_valid'].astype('datetime64[ns]')

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

def compute_weighted_acc(da_fc, da_true, mean_dims=xr.ALL_DIMS):
    """
    Compute the ACC with latitude weighting from two xr.DataArrays.
    WARNING: Does not work if datasets contain NaNs
    Args:
        da_fc (xr.DataArray): Forecast. Time coordinate must be validation time.
        da_true (xr.DataArray): Truth.
        mean_dims: dimensions over which to average score
    Returns:
        acc: Latitude weighted acc
    """

    clim = da_true.mean('time')
    try:
        t = np.intersect1d(da_fc.time, da_true.time)
        fa = da_fc.sel(time=t) - clim
    except AttributeError:
        t = da_true.time.values
        fa = da_fc - clim
    a = da_true.sel(time=t) - clim

    weights_lat = np.cos(np.deg2rad(da_fc.lat))
    weights_lat /= weights_lat.mean()
    w = weights_lat

    fa_prime = fa - fa.mean()
    a_prime = a - a.mean()

    acc = (
            np.sum(w * fa_prime * a_prime) /
            np.sqrt(
                np.sum(w * fa_prime ** 2) * np.sum(w * a_prime ** 2)
            )
    )
    return acc

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

    fc_pers = truth.copy()
    fc_pers['time'] = fc['time']

    # both truth and fc have  "time" dimension, and in the scoring
    # functions they are automatically aligned, so dont need to crop the last/first couple
    # of timesteps

    for ivar, varname in enumerate(vars):
        rmse = compute_weighted_rmse(fc.isel(lev=ivar), truth.isel(lev=ivar))
        acc = compute_weighted_acc(fc.isel(lev=ivar), truth.isel(lev=ivar))
        rmse_pers = compute_weighted_rmse(fc_pers.isel(lev=ivar), truth.isel(lev=ivar))
        acc_pers = compute_weighted_acc(fc_pers.isel(lev=ivar), truth.isel(lev=ivar))

        _res = pd.DataFrame({'rmse':rmse.values,
                             'acc':acc.values,
                             'rmse_pers': rmse_pers.values,
                             'acc_pers': acc_pers.values,
                             'fcday':fcday,
                             'var':varname,
                             'member':i_train}, index=[0])
        res.append(_res)

        #error maps
        rmse_2d = np.sqrt((fc.isel(lev=ivar) - truth.isel(lev=ivar))**2).mean(('time'))
        rmse_2d.to_netcdf(f'data/fc_res_2d_{nettype}_{resolution}_{varname}_itrain{i_train}_day{fcday}.nc')
        plt.figure()
        rmse_2d.plot.contourf()
        plt.savefig(f'plots/fcskill_2d_{nettype}_{resolution}_{varname}_itrain{i_train}_day{fcday}.svg')

        plt.close('all')
        gc.collect()

res = pd.concat(res)

res.to_pickle(f'data/fc_res_{nettype}_{resolution}_itrain{i_train}.pkl')
res.to_csv(f'data/fc_res_{nettype}_{resolution}_itrain{i_train}.csv')



for varname in vars:

    sub = res[res['var']==varname]
    sub = sub.set_index('fcday')
    plt.figure()
    ax = plt.subplot(211)
    sub[['rmse','rmse_pers']].plot(ax=ax)
    ax = plt.subplot(212)
    sub[['acc', 'acc_pers']].plot(ax=ax)
    plt.savefig(f'plots/fcskill_{nettype}_{resolution}_{varname}_itrain{i_train}.svg')

    plt.close('all')