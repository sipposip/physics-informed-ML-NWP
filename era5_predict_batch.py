

import os
import glob
import pickle
import sys
import numpy as np
import tensorflow as tf
import xarray as xr
import pandas as pd




from custom_layers import PolarPaddingConv2d, SphericalConv2D, SphericalHemConv2D_shared, SphericalHemConv2D,\
    HemishpereConv2D


nettype = sys.argv[1]
resolution = sys.argv[2]
i_train = int(sys.argv[3])



leadtime = 6 # h
startyear = 1979
endyear = 2016
test_startyear=2017
test_endyear=2018

tres = 6      # this is based on the raw data. the tfrecord data is already in this timeresolution!
batch_size = 6
n_t_input = 2  # the "t" in weyn et al DO NOT CHANGE, THE CODE IS NOT FLEXIBLE ENOGUH!
# specify variables and levels. level can be a single number, or a list (not tuple!) of numbers
var_and_lev_list = [['geopotential', [1000, 700, 500, 300]], ['temperature', [850]],
                    ['toa_incident_solar_radiation', 'single']]
# for filenames
var_lev_str = ''.join(str(e) for a in var_and_lev_list for e in a).replace(' ', ''). \
    replace('[', '').replace(']', '_').replace(',', '_')
leadstep = int(leadtime / tres)

datapath = f'/proj/bolinc/users/x_sebsc/nn_reanalysis/era5_benchmarkdata/tfrecord_{resolution}'

modelpath = '/proj/bolinc/users/x_sebsc/nn_reanalysis/trained_models/'


outpath=f'/proj/bolinc/users/x_sebsc/nn_reanalysis/eval/{nettype}_{resolution}/'
os.system(f'mkdir -p {outpath}')

paramstr = f'{nettype}_{var_lev_str}_{startyear}-{endyear}_{resolution}.deg_tres{tres}_itrain{i_train}'

ifiles = [file for year in range(test_startyear, test_endyear + 1) for file in sorted(
    glob.glob(f'{datapath}/{var_lev_str}_{year}_*_{resolution}deg_tres{tres}.tfr'))]

if len(ifiles) == 0:
    raise FileNotFoundError(f'no input files were found!, datapath:{datapath}, filestring:{var_lev_str}_{year}_*_{resolution}deg_tres{tres}.tfr')
shapefile = f'{datapath}/shape_{var_lev_str}_{resolution}deg_tres{tres}.pkl'

shape_raw = list(pickle.load(open(shapefile, 'rb')))
shape_out = [shape_raw[1], shape_raw[2], shape_raw[0]]
nlat, nlon, nlev_data = shape_out
# we treat toa_radiation as a constant for the length of the forecast, therefore
# the actual "unfixed" data levels are 1 less. we can do this because it is the last level in the data before the
# real constants
constans_in_data = 3 # toa_radiation,orography,lsm
constants_additional = 4 # doy_sin, doy_cos, hour_sin, hour_cos
nlev_data = nlev_data-constans_in_data
# and instead we add it to the additional vars
add_vars_in = constans_in_data + constants_additional
nlev_in = n_t_input * nlev_data + add_vars_in
nlev_out = n_t_input * nlev_data
shape_in = (nlat, nlon, nlev_in)

coords = pickle.load(open(f'data/coords_{resolution}.pkl','rb'))

# (approximate) estimation of samples
valid_fraction = 1 / 40
nsamples = (endyear - startyear + 1) * 365 * tres
n_valid = int(nsamples * valid_fraction)
n_train = nsamples - n_valid
# load data normalization weights
# these are per level
norm_mean = xr.open_dataarray(
    f'{datapath}/norm_mean_geopotential_level{var_lev_str}_{startyear}-{endyear}_{resolution}.deg_tres{tres}.nc')
norm_std = xr.open_dataarray(
    f'{datapath}/norm_std_geopotential_level{var_lev_str}_{startyear}-{endyear}_{resolution}.deg_tres{tres}.nc')
norm_mean = tf.convert_to_tensor(norm_mean.values)
norm_std = tf.convert_to_tensor(norm_std.values)
assert (len(norm_mean) == len(norm_std))
assert (len(norm_mean) == nlev_data+constans_in_data)


def decode(seralized_example):
    data_all = tf.io.parse_single_example(seralized_example,
                                          features={
                                              'data': tf.io.FixedLenFeature(shape_raw, tf.float32),
                                              'doy': tf.io.FixedLenFeature(1, tf.float32),
                                              'hour': tf.io.FixedLenFeature(1, tf.float32)
                                          })
    data = data_all['data']
    doy = data_all['doy']
    hour = data_all['hour']
    # make level dimension last dimension
    data = tf.transpose(data, [1, 2, 0])
    # data normalization
    data = (data - norm_mean) / norm_std

    # circular normalization
    doy_1d_sin = tf.sin(doy * 2 * np.pi / 365)
    doy_1d_cos = tf.cos(doy * 2 * np.pi / 365)
    # expand doy to same size as input grid
    doy_sin = tf.ones((nlat, nlon, 1)) * doy_1d_sin
    doy_cos = tf.ones((nlat, nlon, 1)) * doy_1d_cos

    # expand hour to same size as input grid and convert hour to local hour
    # create grid with longitude [0,24[
    local_hour_grid = tf.tile(tf.expand_dims(tf.range(nlon, dtype='float32'),0),[nlat,1])/nlon*24
    # add channel dimension
    local_hour_grid = tf.expand_dims(local_hour_grid,-1)
    # shift the longitude hour grid to represent the actual hour in the input
    local_hour = (local_hour_grid + hour)%24
    # local hour is now in [0,24]
    # convert to sin and cos (circular normalization)
    local_hour_sin = tf.sin(local_hour * 2 * np.pi / 24)
    local_hour_cos = tf.cos(local_hour * 2 * np.pi / 24)

    ## stack doy
    data = tf.concat([data, doy_sin, doy_cos, local_hour_sin, local_hour_cos], axis=-1)
    return data

dataset = tf.data.TFRecordDataset(filenames=ifiles)
dataset = dataset.map(decode, num_parallel_calls=tf.data.experimental.AUTOTUNE)

# here we have to differ form the trainign script, webause we only need the input
# thus windowsize is not 6, but 2
windowsize = 2
dataset = dataset.window(windowsize, shift=leadstep, stride=1, drop_remainder=True)
#dataset = dataset.flat_map(lambda window: window.batch(windowsize))
dataset = dataset.flat_map(lambda window: window.batch(windowsize, drop_remainder=True))

dataset = dataset.map(lambda x : tf.concat(tf.unstack(x,axis=0), axis=-1))
# now we have in the channel dimension in  x: [t1,add_vars,t2,add_vars]
# so now we remove the redundant first occurence of add_vars
dataset = dataset.map(lambda x: tf.gather(x, list(range(nlev_data))+list(range(nlev_data+add_vars_in, n_t_input*(nlev_data+add_vars_in))), axis=-1))



modelfile = modelpath+'/'+paramstr+'_final.h5'

def rmse_z500(y_true, y_pred):
    ilev=2 +5 # z500 is 3rd out of 5 variables, and we have two output steps
    y_true = y_true[...,ilev]
    y_pred = y_pred[...,ilev]
    mse = tf.keras.metrics.mean_squared_error(y_true, y_pred)
    mse = mse * norm_std[ilev%5]**2
    return tf.sqrt(mse)
def rmse_t850(y_true, y_pred):
    ilev=4+5
    y_true = y_true[...,ilev]
    y_pred = y_pred[...,ilev]
    mse = tf.keras.metrics.mean_squared_error(y_true, y_pred)
    mse = mse * norm_std[ilev%5]**2
    return tf.sqrt(mse)


## load trained models

# there is a bug in SphericalHemConv2D and SphericalHemConv2D_shared that is only relevant when
# loading trained models. in the init function, kernel is passe dtow times when loading the model (
# one time in the parent class SphericalResample, and one time in SphericalHemConv2D itself). the problem
# is that the kernel is saved two times. if we would change the original layer we would need to retrain.
# therefore, here we use a workaround shere the init function of SphericalHemConv2D is overloaded before
# loading the saved model.
class SphericalHemConv2D_fixed(SphericalHemConv2D):

    def __init__(self, filters, stride=1, conv_args={}, **kwargs):

        super(SphericalHemConv2D, self).__init__(stride=stride, **kwargs)

        self.filters = filters
        self.conv_args = conv_args
        self.conv1 = tf.keras.layers.Conv2D(self.filters, kernel_size=(1, self.nkernel), **conv_args)
        self.conv2 = tf.keras.layers.Conv2D(self.filters, kernel_size=(1, self.nkernel), **conv_args)


class SphericalHemConv2D_shared_fixed(SphericalHemConv2D_shared):

    def __init__(self, filters, stride=1, conv_args={}, **kwargs):

        super(SphericalHemConv2D_shared, self).__init__(stride=stride, **kwargs)

        self.filters = filters
        self.conv_args = conv_args
        self.conv1 = tf.keras.layers.Conv2D(self.filters, kernel_size=(1, self.nkernel), **conv_args)

net = tf.keras.models.load_model(modelfile, custom_objects={
    'SphericalConv2D':SphericalConv2D,
    'PolarPaddingConv2d':PolarPaddingConv2d,
    'SphericalHemConv2D_shared':SphericalHemConv2D_shared_fixed,
    'SphericalHemConv2D':SphericalHemConv2D_fixed,
    'HemishpereConv2D':HemishpereConv2D,
    'rmse_z500':rmse_z500,
    'rmse_t850':rmse_t850,
    })


init_dates = pd.date_range(f'{test_startyear}0101', f'{test_endyear}12311800', freq=f'{tres}h')

fc_steps=10

nlev,nlat,nlon = shape_raw
nlev=nlev-constans_in_data
for i, data in enumerate(dataset):
    t_init = init_dates[i]
    # add empty batch dim
    data = tf.expand_dims(data,0)
    fixed_inputs = data[..., -add_vars_in:]
    x = data
    fc = np.zeros((fc_steps+1, nlat,nlon,nlev))
    # as initial value we use the second timestep of the input
    fc[0] = x[:,:,:,nlev:nlev*2]
    for j in range(fc_steps):
        x = net(x)
        # extract only last 2 timesteps (needed for initialization of the next step)
        x = x[...,-nlev_out:]
        # store only the last timesteps
        fc[j+1] = x[...,-nlev:]
        # we make 24 hour steps, therefore we dont need to update hour of the day.
        # in principle we could update doyofyear, but our forecasts are so short compared
        # to the length of a year that we dont do this here.
        # therefore, we simply persist the additonal inputs
        x = tf.concat([x,fixed_inputs],-1)


    # revere normalization
    fc = (fc * norm_std[:-constans_in_data]) + norm_mean[:-constans_in_data]
    # compute the dates. we start with 2 timesteps, therefore we have to add +6h to t_init
    t_init_adjusted = t_init + pd.to_timedelta('6h')
    t = pd.date_range(t_init_adjusted, t_init_adjusted+pd.to_timedelta(fc_steps,unit='d'))

    fc_xr = xr.DataArray(fc, dims=['fcday','lat', 'lon', 'lev'],
                         coords={'fcday':np.arange(fc_steps+1), 'lat':coords['lat'], 'lon':coords['lon'],
                                 # we dont have radiation in output
                                 'lev':[e[0]+str(l)  for e in var_and_lev_list[:-1] for l in e[1]]})

    fc_xr['time_init'] = t_init_adjusted

    fc_xr['time_valid'] = xr.Variable('fcday',t)

    fc_xr.to_netcdf(f'{outpath}/fc_{nettype}_{resolution}_itrain{i_train}_{t_init_adjusted.strftime("%Y%m%d%H")}.nc')
