"""
preprocess data
concatanates different years, normalizes the data,
and converts it to tfrecord files

normalization is NOT done here, but when the data is actually read in for training
"""

import os
import pickle
from tqdm import trange
import tensorflow as tf
import numpy as np
import xarray as xr

inpath = '/climstorage/sebastian/nn_nwp_point/era5_benchmarkdata/'

startyear = 1979
endyear = 2018
tres = 6  # hours
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
# records_per_tffile = 1500
if resolution == '2.8125':
    records_per_tffile = 2000
else:
    records_per_tffile = 500
os.system(f'mkdir -p {outpath}')

# for filenames
var_lev_str = ''.join(str(e) for a in var_and_lev_list for e in a).replace(' ', ''). \
    replace('[', '').replace(']', '_').replace(',', '_')

nyears = endyear - startyear + 1

# constants. these we open only once
constants = xr.open_dataset(f'{inpath}/constants_{resolution}deg.nc')
# stack the different const variables as levels
constants = xr.concat([constants[key] for key in ['orography', 'lsm']], dim='level')
# add empty time dimension
constants = constants.expand_dims('time', 0)

# in principle, we could open  all data with mfdataset and loop over it
# howver, for large datasets this is very slow, much slower
# then opening only part of a dataset and looping over that.
# (the time to read a single timeslice gets much longer when many files are open with mfdataset)
# therefore we loop over years, and process each year sepereately
# for this, we have to repeat the dataprocessing from above

for year in trange(startyear, endyear + 1):
    res = []
    for var, level in var_and_lev_list:
        ifile = f'{inpath}/{var}_{year}_{resolution}deg.nc'
        data = xr.open_dataset(ifile, chunks={'time': 1})
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
    if year == 1979:
        # in tisr the first 7 timesteps (in the hourly data) are missing. th
        if tres != 6:
            raise NotImplementedError('for tres!=6 the cropping of the initial nans has to be adjusted!')
        data = data[2:]
    # add constants (expanded over time dimension)
    constants_expanded = xr.concat([constants for _ in range(data.shape[0])], dim='time')
    data = xr.concat([data, constants_expanded], dim='level')

    n_samples, nlev, nlat, nlon = data.shape
    # write out the shape to a seperate file
    pickle.dump((nlev, nlat, nlon), open(f'{outpath}/shape_{var_lev_str}_{resolution}deg_tres{tres}.pkl', 'wb'))

    n_tfrecord_files_per_year = int(np.ceil(n_samples / records_per_tffile))
    nsamples_thisyear = len(data)
    for i_file in trange(n_tfrecord_files_per_year):
        filename = f'{outpath}/{var_lev_str}_{year}_{i_file:04d}_{resolution}deg_tres{tres}.tfr'
        # just to make sure, remove file in case it exists
        if os.path.exists(filename):
            os.remove(filename)
        # now loop over the data
        writer = tf.io.TFRecordWriter(filename)
        for i in trange(i_file * records_per_tffile, (i_file + 1) * records_per_tffile):
            if i < nsamples_thisyear:
                data_i = data[i]
                # add doy information
                doy = data_i['time'].dt.dayofyear.values
                hour = data_i['time'].dt.hour.values
                # load data
                data_i = data_i.values.flatten()
                if np.any(np.isnan(data_i)):
                    raise ValueError(f"found nan in data in year {year}, stopping!")

                example = tf.train.Example(features=tf.train.Features(feature={
                    'data': tf.train.Feature(float_list=tf.train.FloatList(value=data_i)),
                    'doy': tf.train.Feature(float_list=tf.train.FloatList(value=[doy])),
                    'hour': tf.train.Feature(float_list=tf.train.FloatList(value=[hour])),
                }))
                writer.write(example.SerializeToString())
    data.close()