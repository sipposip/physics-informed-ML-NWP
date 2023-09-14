import os
import glob
import pickle
import sys
import numpy as np
import tensorflow as tf


from custom_layers import SphericalConv2D
from custom_layers import  PeriodicPadding, PolarPaddingConv2d

class HemishpereConv2D(tf.keras.layers.Layer):


    def __init__(self, filters, kernel_size,share_weights=True, **kwargs):
        """
            convolution on 2 hemispheres.
            the input is padded with zeros at the poles, and "wrapped" around
            in the longitude direction.
            is share_weights=True, the same convolution weights are used
            for both hemispheres. in this case, the second hemisphere is flipped
            along the lat dimension before the convolution and flipped back afterwards.
            if share_weights=False, separate weights are used for the hemispheres.
            input: (batch,lat,lon,lev)
            lat must be even
        """

        super(HemishpereConv2D, self).__init__()

        if kernel_size !=3:
            raise NotImplementedError('kernel size !=3 not implemented')
        self.filters=filters
        self.kernel_size = kernel_size
        self.share_weights = share_weights
        self.conv1 = tf.keras.layers.Conv2D(self.filters, kernel_size=kernel_size, **kwargs)
        if not share_weights:
            self.conv2 = tf.keras.layers.Conv2D(self.filters, kernel_size=kernel_size, **kwargs)

    def build(self, input_shape):

        self.nlat = input_shape[1]
        self.nlon = input_shape[2]
        if self.nlat %2:
            raise ValueError('nlat must be even!')


    def get_config(self):
        config = {
            'filters': self.filters,
            'kernel_size':self.kernel_size,
            'share_weights': self.share_weights,
        }
        base_config = super(HemishpereConv2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def call(self, x):
        # zero pad the poles
        x = tf.keras.backend.spatial_2d_padding(x,padding=((1,1),(0,0)))
        # pad longitude with wrapping
        left = x[:,:,0:1]
        right = x[:,:,-1:]
        middle = x
        x = tf.concat([right, middle, left], axis=2)
        # split input into NH and SH, with one row padding into the other hemisphere
        # since nlat is without the padding, we have to add 1 more
        H1 = x[:,:self.nlat//2+2]
        H2 = x[:,-self.nlat//2-2:]
        H1out = self.conv1(H1)
        if self.share_weights:
            # in case of shared weights, we flip the data to account for the other hemisphere
            H2 = H2[:,::-1]
            H2out = self.conv1(H2)
            # flip back
            H2out = H2out[:,::-1]
        else:
            H2out = self.conv2(H2)
        # combine along lat dimension
        out = tf.concat([H1out,H2out],axis=1)

        return out

    def compute_output_shape(self, input_shape):
        output_shape = input_shape[:3] + [self.filters]
        return output_shape

# single layer

# whole networks

shape_in = [128,256,12]
add_vars_in=4
nlev_out = 8
def build_base_net():
    def periodic_conv2d(x, filters):
        kernel_size = 3  # must be 3!!!
        conv = PolarPaddingConv2d(filters, kernel_size=kernel_size)
        x = conv(x)
        x = tf.keras.layers.ReLU(negative_slope=0.1, max_value=10)(x)
        return x

    inp = tf.keras.layers.Input(shape=shape_in)
    fixed_inputs = tf.keras.layers.Lambda(lambda x: x[...,-add_vars_in:], name='fixed_inputs')(inp)

    unfixed_inputs = tf.keras.layers.Lambda(lambda x: x[...,:-add_vars_in], name='unfixed_inputs')(inp)
    x1 = inp
    x2 = periodic_conv2d(x1, 32)
    x3 = periodic_conv2d(x2, 32)
    x4 = tf.keras.layers.AveragePooling2D(2)(x3)
    x5 = periodic_conv2d(x4, 64)
    x6 = periodic_conv2d(x5, 64)
    x7 = tf.keras.layers.AveragePooling2D(2)(x6)
    x8 = periodic_conv2d(x7, 128)
    x9 = periodic_conv2d(x8, 64)
    x10 = tf.keras.layers.UpSampling2D(2)(x9)
    x11 = tf.keras.layers.concatenate([x10, x6])
    x12 = periodic_conv2d(x11, 64)
    x13 = periodic_conv2d(x12, 32)
    x14 = tf.keras.layers.UpSampling2D(2)(x13)
    x15 = tf.keras.layers.concatenate([x14, x3])
    x16 = periodic_conv2d(x15, 32)
    x17 = periodic_conv2d(x16, 32)
    x18 = tf.keras.layers.Convolution2D(nlev_out, kernel_size=1, activation='linear')(x17)
    x19 = tf.keras.layers.add([unfixed_inputs,x18])
    net_onestep = tf.keras.Model(inp, x19)
    output_step1 = x19

    # add additional vars again as input for step 2
    input_step2 = tf.keras.layers.concatenate([output_step1,fixed_inputs], name='add_additional_vars_in_step2')
    # make second step with the network
    output_step2 = net_onestep(input_step2)
    # concatanaet the two output steps along the channel dim
    out_combined = tf.keras.layers.concatenate([output_step1, output_step2], name='out_combined')
    net = tf.keras.Model(inp, out_combined)

    optimizer = tf.keras.optimizers.Adam(lr=1e-3)
    net.compile(loss='mse', optimizer=optimizer)
    net.build(input_shape=shape_in)

    return net


def build_spherconv_net():
    # we should get a smoother
    def full_kernel(n):
        nh = np.floor(n / 2)
        kernel = np.array(np.meshgrid(np.mgrid[-nh:nh + 1], np.mgrid[-nh:nh + 1])).transpose((1, 2, 0))
        kernel = kernel.reshape((-1, 2))
        kernel = kernel.astype(int)
        assert (len(kernel == n ** 2))
        return kernel

    kernel = full_kernel(3)

    def sphereconv(x, filters):
        x = SphericalConv2D(filters=filters, kernel=kernel)(x)
        x = tf.keras.layers.ReLU(negative_slope=0.1, max_value=10)(x)
        return x

    inp = tf.keras.layers.Input(shape=shape_in)
    fixed_inputs = tf.keras.layers.Lambda(lambda x: x[..., -add_vars_in:], name='fixed_inputs')(inp)

    unfixed_inputs = tf.keras.layers.Lambda(lambda x: x[..., :-add_vars_in], name='unfixed_inputs')(inp)
    x1 = inp
    x2 = sphereconv(x1, 32)
    x3 = sphereconv(x2, 32)
    x4 = tf.keras.layers.AveragePooling2D(2)(x3)
    x5 = sphereconv(x4, 64)
    x6 = sphereconv(x5, 64)
    x7 = tf.keras.layers.AveragePooling2D(2)(x6)
    x8 = sphereconv(x7, 128)
    x9 = sphereconv(x8, 64)
    x10 = tf.keras.layers.UpSampling2D(2)(x9)
    x11 = tf.keras.layers.concatenate([x10, x6])
    x12 = sphereconv(x11, 64)
    x13 = sphereconv(x12, 32)
    x14 = tf.keras.layers.UpSampling2D(2)(x13)
    x15 = tf.keras.layers.concatenate([x14, x3])
    x16 = sphereconv(x15, 32)
    x17 = sphereconv(x16, 32)
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
    net.compile(loss='mse', optimizer=optimizer)
    net.build(input_shape=shape_in)
    return net


def build_hemconv(share_weights):
    def conv_block(x, filters):
        x = HemishpereConv2D(filters, kernel_size=3, share_weights=share_weights)(x)
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
    net.compile(loss='mse', optimizer=optimizer)
    net.build(input_shape=shape_in)
    return net

basenet = build_base_net()
sphereconvnet = build_spherconv_net()
hemconvnet = build_hemconv(share_weights=False)
hemconvnet_shared = build_hemconv(share_weights=True)
x = np.random.random([6] + shape_in).astype('float32')

basenet(x)
sphereconvnet(x)
hemconvnet(x)
hemconvnet_shared(x)
# only works in ipython or jupyter
# %timeit basenet(x)
# %timeit sphereconvnet(x)
# %timeit hemconvnet(x)
# %timeit hemconvnet_shared(x)

