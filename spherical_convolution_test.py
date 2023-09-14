
"""
NoteL requires tf >=2.0

"""


import numpy as np
import tensorflow as tf
from tensorflow import keras
from pylab import plt
from custom_layers import SphericalResample, SphericalConv2D

plt.ion()
n_circ_kernel = 8
radius = 1
def circular_kernel(radius, n):
    x = radius * np.cos(np.linspace(0,2*np.pi*(1-1/n),n))
    y = radius * np.sin(np.linspace(0,2*np.pi*(1-1/n),n))
    kernel = list(zip(x,y))
    return kernel


Nlat=720
Nlon=1440
Nlat=40
Nlon=60
nchannel=5

test_data = np.sin(np.indices((Nlat,Nlon))[1]).astype('float32')
test_data = test_data[np.newaxis,:,:,np.newaxis]
test_data = np.tile(test_data, (30,1,1,nchannel))

kernel = [[-1,0],[0,0], [1,0]]

net = keras.Sequential([keras.layers.InputLayer(input_shape=(Nlat,Nlon,nchannel)),
    SphericalConv2D(filters=5,kernel=kernel,stride=1)])

net.compile(optimizer='adam', loss='mean_absolute_error')
net.fit([test_data], [test_data])  # very slow the first time it is called, but then fast

l = SphericalResample(kernel)
l2 = SphericalConv2D(4,kernel)


# some tests
test_data = np.sin(np.indices((Nlat,Nlon))[1]).astype('float32')
test_data = test_data[np.newaxis,:,:,np.newaxis]
test_data = np.tile(test_data, (30,1,1,nchannel))
x = test_data
# with kernel [0,0], the expanded data needs to be the same as the input, except flattened
# along Nlat,Nlon
l = SphericalResample(kernel=[[0,0]])
expanded = np.array(l(x))
reshaped = np.reshape(expanded,(x.shape))
assert(np.array_equal(x,reshaped))

# test convolution]
# if we initialize with constant values 1/len(kernel), then
# we should get a smoother
def full_kernel(n):
    nh = np.floor(n / 2)
    kernel = np.array(np.meshgrid(np.mgrid[-nh:nh + 1], np.mgrid[-nh:nh + 1])).transpose((1, 2, 0))
    kernel = kernel.reshape((-1, 2))
    kernel = kernel.astype(int)
    assert (len(kernel == n ** 2))
    return kernel


kernel = full_kernel(3)
sc = SphericalConv2D(filters=5,kernel=kernel,stride=1,
                     conv_args=dict(kernel_initializer=tf.constant_initializer(1/len(kernel))))

Nlat=41
Nlon=61
nchannel=1

test_data = np.sin(np.indices((Nlat,Nlon))[1]).astype('float32')
test_data = test_data[np.newaxis,:,:,np.newaxis]
x = np.tile(test_data, (1,1,1,nchannel))
a = np.array(sc(x))
plt.figure()
plt.subplot(211)
plt.imshow(x[0,:,:,0])
plt.colorbar()
plt.subplot(212)
plt.imshow(a[0,:,:,0])
plt.colorbar()

## even better check:
## if we select one target gridpoint, and compute the gradient
## w.r.t. to the input field, we get the interpolation weights (as long
## as the convolution kernel is set to constant initialization)



for ix,iy in ((20,20),(10,10), (5,10),
              (2,30), (35,40)):
    xt = tf.convert_to_tensor(x)
    with tf.GradientTape() as t:

        t.watch(xt)
        target = sc(xt)[0, ix, iy, 0]

    sens = t.gradient(target, xt)
    plt.figure()
    plt.imshow(np.array(sens).squeeze(), interpolation='nearest')
    plt.colorbar()
    plt.title(f"{ix},{iy}")


