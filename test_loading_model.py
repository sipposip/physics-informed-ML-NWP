


import os
import glob
import pickle
import sys
import numpy as np
import tensorflow as tf
import xarray as xr

from custom_layers import PolarPaddingConv2d, SphericalConv2D, SphericalHemConv2D_shared, SphericalHemConv2D, \
    HemishpereConv2D

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


# START HACK (uncommented norm_std)
def rmse_z500(y_true, y_pred):
    ilev=2 +5 # z500 is 3rd out of 5 variables, and we have two output steps
    y_true = y_true[...,ilev]
    y_pred = y_pred[...,ilev]
    mse = tf.keras.metrics.mean_squared_error(y_true, y_pred)
    mse = mse # norm_std[ilev%5]**2
    return tf.sqrt(mse)
def rmse_t850(y_true, y_pred):
    ilev=4+5
    y_true = y_true[...,ilev]
    y_pred = y_pred[...,ilev]
    mse = tf.keras.metrics.mean_squared_error(y_true, y_pred)
    mse = mse #* norm_std[ilev%5]**2
    return tf.sqrt(mse)
# END HACK

constans_in_data = 3 # toa_radiation,orography,lsm
constants_additional = 4 # doy_sin, doy_cos, hour_sin, hour_cos
# and instead we add it to the additional vars
add_vars_in = constans_in_data + constants_additional


modelfile='/proj/bolinc/users/x_sebsc/nn_reanalysis/trained_models/unet_sphereconv_geopotential1000_700_500_300_temperature850_toa_incident_solar_radiationsingle_1979-2016_1.40625.deg_tres6_itrain3040_0.0062.h5'


net_loaded = tf.keras.models.load_model(modelfile, custom_objects={
    'SphericalConv2D':SphericalConv2D,
    'PolarPaddingConv2d':PolarPaddingConv2d,
    'SphericalHemConv2D_shared':SphericalHemConv2D_shared_fixed,
    'SphericalHemConv2D':SphericalHemConv2D_fixed,
    'HemishpereConv2D':HemishpereConv2D,
    'rmse_z500':rmse_z500,
    'rmse_t850':rmse_t850,
    })

