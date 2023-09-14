"""
for experiments with latitude and longitude added as additional constant
feature channels

changes to era5_train_batch.py:
function decode was adapted to compute and include lat and lon grids on the fly
"add_latlon" was added to paramstr
"""
import os
import glob
import pickle
import sys
import numpy as np
import tensorflow as tf
import xarray as xr

from custom_layers import PolarPaddingConv2d, SphericalConv2D, SphericalHemConv2D_shared, SphericalHemConv2D, \
    HemishpereConv2D



nettype = sys.argv[1]
resolution = sys.argv[2]
i_train = int(sys.argv[3])

n_epochs_no_earlystop = 100
n_epochs_additional = 50

leadtime = 6  # h

startyear = 1979
endyear = 2016
tres = 6  # this is based on the raw data. the tfrecord data is already in this timeresolution!
batch_size = 6
n_t_input = 2  # the "t" in weyn et al DO NOT CHANGE, THE CODE IS NOT FLEXIBLE ENOGUH!
# specify variables and levels. level can be a single number, or a list (not tuple!) of numbers
var_and_lev_list = [['geopotential', [1000, 700, 500, 300]], ['temperature', [850]],
                    ['toa_incident_solar_radiation', 'single']]
# for filenames
var_lev_str = ''.join(str(e) for a in var_and_lev_list for e in a).replace(' ', ''). \
    replace('[', '').replace(']', '_').replace(',', '_')

leadstep = int(leadtime / tres)

# on tetralith we copy the files to the local scratch space of the node.
# the copying is done in the bash script that is submitted, so here
# we have to use the local path

datapath = f'/scratch/local/tfrecord_{resolution}/'

modelpath = '/proj/bolinc/users/x_sebsc/nn_reanalysis/trained_models/'
os.system(f'mkdir -p {modelpath}')

histpath = 'train_hists'
os.system(f'mkdir -p {histpath}')

paramstr = f'{nettype}_latlon_{var_lev_str}_{startyear}-{endyear}_{resolution}.deg_tres{tres}_itrain{i_train}'

ifiles = [file for year in range(startyear, endyear + 1) for file in sorted(
    glob.glob(f'{datapath}/{var_lev_str}_{year}_*_{resolution}deg_tres{tres}.tfr'))]

if len(ifiles) == 0:
    raise FileNotFoundError('no input files were found!')
shapefile = f'{datapath}/shape_{var_lev_str}_{resolution}deg_tres{tres}.pkl'

shape_raw = list(pickle.load(open(shapefile, 'rb')))
shape_out = [shape_raw[1], shape_raw[2], shape_raw[0]]
nlat, nlon, nlev_data = shape_out
# we treat toa_radiation as a constant for the length of the forecast, therefore
# the actual "unfixed" data levels are 1 less. we can do this because it is the last level in the data before the
# real constants
constans_in_data = 3 # toa_radiation,orography,lsm
constants_additional = 7 # doy_sin, doy_cos, hour_sin, hour_cos, lat, lon_sin, lon_cos
nlev_data = nlev_data-constans_in_data
# and instead we add it to the additional vars
add_vars_in = constans_in_data + constants_additional
nlev_in = n_t_input * nlev_data + add_vars_in
nlev_out = n_t_input * nlev_data
shape_in = (nlat, nlon, nlev_in)

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

    # add latitude, normalized to [0,1] 
    lat = tf.range(0, nlat, dtype='float32') / nlat + 1/nlat*0.5
    # add longitude as sin and cos
    lon = tf.range(0, nlon, dtype='float32') * 360 / nlon + 360/nlon*0.5
    lon_sin = tf.sin(lon * 2* np.pi/360)
    lon_cos = tf.cos(lon * 2* np.pi/360)
    # expand to whole grid
    lon_sin_grid = tf.tile(tf.expand_dims(lon_sin,0), [nlat, 1])
    lon_cos_grid = tf.tile(tf.expand_dims(lon_cos,0), [nlat, 1])
    lat_grid = tf.tile(tf.expand_dims(lat, 1), [1, nlon])
    # add channel dimension
    lat_grid = tf.expand_dims(lat_grid,-1)
    lon_sin_grid = tf.expand_dims(lon_sin_grid,-1)
    lon_cos_grid = tf.expand_dims(lon_cos_grid,-1)

    ## stack doy
    data = tf.concat([data, doy_sin, doy_cos, local_hour_sin, local_hour_cos, lat_grid, lon_sin_grid, lon_cos_grid], axis=-1)
    return data


dataset = tf.data.TFRecordDataset(filenames=ifiles)
dataset = dataset.map(decode, num_parallel_calls=tf.data.experimental.AUTOTUNE)

# now make the next timestep the target
# https://stackoverflow.com/questions/55429307/how-to-use-windows-created-by-the-dataset-window-method-in-tensorflow-2-0
windowsize = 6  # 2 timesteps input (0,1), then we make a 1-step prediction, and we thus
# land at step 2,3 and then we repeat the thing, and we land at 4,5)
dataset = dataset.window(windowsize, shift=leadstep, stride=1, drop_remainder=True)
# dataset = dataset.flat_map(lambda window: window.batch(windowsize))
dataset = dataset.flat_map(lambda window: window.batch(windowsize, drop_remainder=True))
# now split into input and target. the input are simply the first 2 timesteps.
# the target are steps 4,5,8,9 (step 3 to end)
dataset = dataset.map(lambda window: (window[:2], window[2:]))
# remove the additional input vars in y
dataset = dataset.map(lambda x, y: (x, y[..., :-add_vars_in]))
# this is now the typical format for recursive networks, with an explicit time dimension
# here, however, we want to have the time dimension added to the last dimension (so for n_t_input=2, we
# have twice the number of channels).
dataset = dataset.map(
    lambda x, y: (tf.concat(tf.unstack(x, axis=0), axis=-1), tf.concat(tf.unstack(y, axis=0), axis=-1)))
# now we have in the channel dimension in  x: [t1,add_vars,t2,add_vars]
# so now we remove the redundant first occurence of add_vars
dataset = dataset.map(lambda x, y: (tf.gather(x, list(range(nlev_data))+list(range(nlev_data+add_vars_in, n_t_input*(nlev_data+add_vars_in))), axis=-1), y))

dataset = dataset.shuffle(10)

dataset = dataset.batch(batch_size, drop_remainder=True)

dataset_valid = dataset.take(n_valid)
dataset_train = dataset.skip(n_valid)

dataset_train = dataset_train.repeat()
dataset_valid = dataset_valid.repeat()


# metrics for the first timestep
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




# only needed for spherical convolution
def full_kernel(n):
    nh = np.floor(n / 2)
    kernel = np.array(np.meshgrid(np.mgrid[-nh:nh + 1], np.mgrid[-nh:nh + 1])).transpose((1, 2, 0))
    kernel = kernel.reshape((-1, 2))
    kernel = kernel.astype(int)
    assert (len(kernel == n ** 2))
    return kernel


kernel = full_kernel(3)


# now the right convolution block for the nettype


def conv_block(x, filters):
    kernel_size = 3  # must be 3!!!

    if nettype == 'unet_base':
        conv = PolarPaddingConv2d(filters, kernel_size=kernel_size)
    elif nettype == 'unet_sphereconv':
        conv = SphericalConv2D(filters, kernel=kernel)
    elif nettype == 'unet_sphereconv_hemconv':
        conv = SphericalHemConv2D(filters)
    elif nettype == 'unet_sphereconv_hemconv_shared':
        conv = SphericalHemConv2D_shared(filters)
    elif nettype == 'unet_hemconv_sharedweights':
        conv = HemishpereConv2D(filters, kernel_size=kernel_size, share_weights=True)
    elif nettype == 'unet_hemconv_shared_nonflipped':
        conv = HemishpereConv2D(filters, kernel_size=kernel_size, share_weights=True, flip_shared_weights=False)
    elif nettype == 'unet_hemconv':
        conv = HemishpereConv2D(filters, kernel_size=kernel_size, share_weights=False)
    elif nettype == 'unet_hemconv_halfsize':
        conv = HemishpereConv2D(filters//2, kernel_size=kernel_size, share_weights=False)

    x = conv(x)
    x = tf.keras.layers.ReLU(negative_slope=0.1, max_value=10)(x)
    return x


inp = tf.keras.layers.Input(shape=shape_in)
fixed_inputs = tf.keras.layers.Lambda(lambda x: x[..., -add_vars_in:], name='fixed_inputs')(inp)

unfixed_inputs = tf.keras.layers.Lambda(lambda x: x[..., :-add_vars_in], name='unfixed_inputs')(inp)
x1 = inp
x2 = conv_block(x1, 32)
x3 = conv_block(x2, 32)
x4 = tf.keras.layers.AveragePooling2D(2)(x3)
x5 = conv_block(x4, 64)
x6 = conv_block(x5, 64)
x7 = tf.keras.layers.AveragePooling2D(2)(x6)
x8 = conv_block(x7, 128)
x9 = conv_block(x8, 64)
x10 = tf.keras.layers.UpSampling2D(2)(x9)
x11 = tf.keras.layers.concatenate([x10, x6])
x12 = conv_block(x11, 64)
x13 = conv_block(x12, 32)
x14 = tf.keras.layers.UpSampling2D(2)(x13)
x15 = tf.keras.layers.concatenate([x14, x3])
x16 = conv_block(x15, 32)
x17 = conv_block(x16, 32)
x18 = tf.keras.layers.Convolution2D(nlev_out, kernel_size=1, activation='linear')(x17)
x19 = tf.keras.layers.add([unfixed_inputs, x18])
net_onestep = tf.keras.Model(inp, x19)
output_step1 = x19

# add additional vars again as input for step 2
input_step2 = tf.keras.layers.concatenate([output_step1, fixed_inputs], name='add_additional_vars_in_step2')
# make second step with the network
output_step2 = net_onestep(input_step2)
# concatanaet the two output steps along the channel dim
out_combined = tf.keras.layers.concatenate([output_step1, output_step2], name='out_combined')
net = tf.keras.Model(inp, out_combined)

optimizer = tf.keras.optimizers.Adam(lr=1e-3)




net.compile(loss='mse', optimizer=optimizer, metrics=[rmse_z500, rmse_t850])
net.build(input_shape=shape_in)
net.summary()
# tf.keras.utils.plot_model(net, 'model.png', expand_nested=False)
# tf.keras.utils.plot_model(net, 'model_expanded.png', expand_nested=True)





# train epochs without early stopping
hist1 = net.fit(dataset_train, validation_data=dataset_valid,
              steps_per_epoch=n_train//batch_size,
              validation_steps=n_valid // batch_size, epochs=n_epochs_no_earlystop,
              callbacks=[
                  tf.keras.callbacks.ModelCheckpoint(modelpath + '/' + paramstr + '{epoch:03d}_{val_loss:.4f}.h5'),
                  ])

# train more with early stopping
hist2 = net.fit(dataset_train, validation_data=dataset_valid,

               steps_per_epoch=n_train//batch_size,
               validation_steps=n_valid // batch_size, epochs=n_epochs_additional,
               callbacks=[
                   tf.keras.callbacks.ModelCheckpoint(modelpath + '/' + paramstr + '{epoch:03d}_{val_loss:.4f}.h5'),
                   tf.keras.callbacks.EarlyStopping(
                       monitor='val_loss',
                       min_delta=0,
                       patience=10,
                       verbose=1,
                       mode='auto',
                       restore_best_weights=True
                   )
                   ])

net.save(modelpath + '/' + paramstr + '_final.h5', save_format='h5')

pickle.dump([hist1.history,hist2.history], open(f'{histpath}/{paramstr}.hist.pkl', 'wb'))
