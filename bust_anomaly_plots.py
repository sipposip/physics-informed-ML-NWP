"""
run on misu160
"""


import matplotlib
matplotlib.use('agg')
import xarray as xr
import cartopy.crs as ccrs
import pandas as pd
from pylab import plt
from dask.diagnostics import ProgressBar

pbar = ProgressBar()
pbar.register()

resolution='1.40625'

era5_file = '/proj/bolinc/users/x_sebsc/nn_reanalysis/era5_netcdf/era5_geopotential_500hpa_2017-1028_highres.nc'

# we have to chunk by level and time, otherwise things get very slow when subselecting levels
data = xr.open_dataset(era5_file)
data = data['z']


# select only northern hemisphere
data = data.sel(lat=slice(0,90))

clim = data.groupby('time.month').mean('time')

for nettype in ('unet_base', 'unet_sphereconv', 'unet_sphereconv_hemconv_shared',
                'unet_hemconv_sharedweights'):


    for fcday in range(1,11):
        dates_valid = pd.read_csv(f'data/bust_dates_{nettype}_fcday{fcday}.csv', index_col=0)

        dates_valid = pd.DatetimeIndex(dates_valid['0'].values)
        # we want to have init dates

        dates = dates_valid - pd.to_timedelta(f'{fcday}d')




        busts_full = data.sel(time=dates.values)

        busts_anom = busts_full.groupby('time.month') - clim
        bust_anom_mean = busts_anom.mean('time')

        bust_anom_mean.load()

        fig, axis = plt.subplots(1, 1, subplot_kw=dict(projection=ccrs.Orthographic(0, 90)))

        p = bust_anom_mean.plot(
            ax=axis,
            transform = ccrs.PlateCarree(),
            cbar_kwargs={'label': 'z500 anomaly $[m^2/s^2]$'}
            )
        axis.coastlines()
        axis.gridlines()
        plt.title(f'fcday={fcday}')
        plt.tight_layout()
        plt.savefig(f'bust_anomaly_z500_{nettype}_fcday{fcday}.png', dpi=300, bbox='tight')
        plt.savefig(f'bust_anomaly_z500_{nettype}_fcday{fcday}.svg')
        plt.close('all')