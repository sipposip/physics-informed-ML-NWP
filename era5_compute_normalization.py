

# run on misu160

import os
import xarray as xr
from dask.diagnostics import ProgressBar

inpath = '/climstorage/sebastian/nn_nwp_point/era5_benchmarkdata/'

startyear = 1979
endyear = 2016
tres = 6 # hours
# specify variables and levels. level can be a single number, or a list (not tuple!) of numbers
var_and_lev_list = [['geopotential', [1000,700,500,300]], ['temperature', [850]],
                    ['toa_incident_solar_radiation','single']]
long_to_shortname = {'geopotential': 'z', 'temperature': 't',
                     '2m_temperature': 't2m', 'u_component_of_wind': 'u',
                     'v_component_of_wind': 'v', 'total_cloud_cover': 'tcc',
                     'toa_incident_solar_radiation': 'tisr',
                     'relative_humidity': 'r'}
# resolution = '2.8125'
resolution='1.40625'
outpath = f'/climstorage/sebastian/nn_reanalysis/era5_benchmarkdata/tfrecord_{resolution}/'

os.system(f'mkdir -p {outpath}')
# for filenames
var_lev_str = ''.join(str(e) for a  in var_and_lev_list for e in a).replace(' ','').\
    replace('[','').replace(']','_').replace(',','_')


# constants. these we open only once
constants = xr.open_dataset(f'{inpath}/constants_{resolution}deg.nc')
# stack the different const variables as levels
constants = xr.concat([constants[key] for key in ['orography', 'lsm']], dim='level')
# add empty time dimension
constants = constants.expand_dims('time', 0)

# open all desired years for all desired variables
res = []
for var, level in var_and_lev_list:
    ifiles = [f'{inpath}/{var}_{year}_{resolution}deg.nc' for year in range(startyear, endyear + 1)]
    # we have to chunk by level and time, otherwise things get very slow when subselecting levels
    data = xr.open_mfdataset(ifiles, combine='nested', concat_dim='time', chunks={'time':1})
    data = data[long_to_shortname[var]]
    if level == 'single':
        # add empty lev dimension
        data = data.expand_dims('level', 1)
    else:
        data = data.sel(level=level)
        # remove level coordinate (we dont need it)
        del data['level']
    res.append(data)

data = xr.concat(res, dim='level')

data = data[::tres]
# in tisr the first 7 timesteps (in the hourly data) are missing. th
if tres != 6:
    raise NotImplementedError('for tres!=6 the cropping of the initial nans has to be adjusted!')
data = data[2:]

# add constants (expanded over time dimension). This is not necessary here, but then
# we can use the same code as in era5_to_tfrecord.py
constants_expanded = xr.concat([constants for _ in range(data.shape[0])], dim='time')
data = xr.concat([data, constants_expanded], dim='level')

# normalization per level

with ProgressBar():
    norm_mean = data.mean(('lat', 'lon', 'time')).compute()
    norm_std = data.std(('lat', 'lon', 'time')).compute()

norm_mean.to_netcdf(f'{outpath}/norm_mean_geopotential_level{var_lev_str}_{startyear}-{endyear}_{resolution}.deg_tres{tres}.nc')
norm_std.to_netcdf(f'{outpath}/norm_std_geopotential_level{var_lev_str}_{startyear}-{endyear}_{resolution}.deg_tres{tres}.nc')
