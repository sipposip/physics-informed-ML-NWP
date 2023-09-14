

import numpy as np
import tensorflow as tf
from tqdm import tqdm



class PeriodicPadding(tf.keras.layers.Layer):
    def __init__(self, axis, padding, **kwargs):
        """
        layer with periodic padding for specified axis
        tensor: input tensor
        axis: on or multiple axis to pad along
        padding: number of cells to pad
        """

        super(PeriodicPadding, self).__init__(**kwargs)

        if isinstance(axis, int):
            axis = (axis,)
        if isinstance(padding, int):
            padding = (padding,)

        self.axis = axis
        self.padding = padding

    def build(self, input_shape):
        super(PeriodicPadding, self).build(input_shape)

    # in order to be able to load the saved model we need to define
    # get_config
    def get_config(self):
        config = {
            'axis': self.axis,
            'padding': self.padding,

        }
        base_config = super(PeriodicPadding, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def call(self, input):

        tensor = input
        for ax, p in zip(self.axis, self.padding):
            # create a slice object that selects everything form all axes,
            # except only 0:p for the specified for right, and -p: for left
            ndim = len(tensor.shape)
            ind_right = [slice(-p, None) if i == ax else slice(None) for i in range(ndim)]
            ind_left = [slice(0, p) if i == ax else slice(None) for i in range(ndim)]
            right = tensor[ind_right]
            left = tensor[ind_left]
            middle = tensor
            tensor = tf.concat([right, middle, left], axis=ax)
        return tensor

    def compute_output_shape(self, input_shape):
        output_shape = list(input_shape)
        for ax, p in zip(self.axis, self.padding):
            output_shape[ax] += 2 * p
        return tuple(output_shape)


class PolarPaddingConv2d(tf.keras.layers.Layer):


    def __init__(self, filters, kernel_size, **kwargs):
        """
        convlution that wraps around longitude, and padds the poles correctly (padded with
        the last latitude band, but mirrored and shifted by 180 degrees
        """

        super(PolarPaddingConv2d, self).__init__()

        if kernel_size !=3:
            raise NotImplementedError('kernel size !=3 not implemented')
        self.filters=filters
        self.kernel_size = kernel_size
        self.conv1 = tf.keras.layers.Conv2D(self.filters, kernel_size=kernel_size, **kwargs)


    def build(self, input_shape):

        self.nlat = input_shape[1]
        self.nlon = input_shape[2]
        if self.nlat %2:
            raise ValueError('nlat must be even!')


    def get_config(self):
        config = {
            'filters': self.filters,
            'kernel_size':self.kernel_size,
        }
        base_config = super(PolarPaddingConv2d, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def call(self, x):
        # pad longitude with wrapping
        left = x[:,:,0:1]
        right = x[:,:,-1:]
        middle = x
        x = tf.concat([right, middle, left], axis=2)

        # pad poles with mirroring and shifting the first and last row
        middle = x
        upper = x[:,-1:,:]
        lower = x[:,0:1,:]
        # roll by 180 defrees (=(nlat+1)/2 pixels) and mirror
        # nlat+1 because we have already padded lon
        upper = tf.roll(upper[:,:,::-1], (self.nlat+1)//2,axis=2)
        lower = tf.roll(lower[:,:,::-1], (self.nlat+1)//2,axis=2)
        x = tf.concat([lower, middle, upper], axis=1)
        # convolution
        x = self.conv1(x)
        return x

    def compute_output_shape(self, input_shape):
        output_shape = input_shape[:3] + [self.filters]
        return output_shape


class HemishpereConv2D(tf.keras.layers.Layer):


    def __init__(self, filters, kernel_size,share_weights=True, flip_shared_weights=True, **kwargs):
        """
            convolution on 2 hemispheres.
            the input is  "wrapped" around in the longitude direction, and wrapped and mirrored at the pole.
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
        self.flip_shared_weights = flip_shared_weights
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
        # pad longitude with wrapping
        left = x[:,:,0:1]
        right = x[:,:,-1:]
        middle = x
        x = tf.concat([right, middle, left], axis=2)
        # pad poles with mirroring and shifting the first and last row
        middle = x
        upper = x[:,-1:,:]
        lower = x[:,0:1,:]
        # roll by 180 defrees (=(nlat+1)/2 pixels) and mirror
        # nlat+1 because we have already padded lon
        upper = tf.roll(upper[:,:,::-1], (self.nlat+1)//2,axis=2)
        lower = tf.roll(lower[:,:,::-1], (self.nlat+1)//2,axis=2)
        x = tf.concat([lower, middle, upper], axis=1)
        # split input into NH and SH, with one row padding into the other hemisphere
        # since nlat is without the padding, we have to add 1 more
        H1 = x[:,:self.nlat//2+2]
        H2 = x[:,-self.nlat//2-2:]
        H1out = self.conv1(H1)
        if self.share_weights:
            if self.flip_shared_weights:
                # flip the data to account for the other hemisphere
                H2 = H2[:,::-1]
                H2out = self.conv1(H2)
                # flip back
                H2out = H2out[:,::-1]
            else:
                H2out = self.conv1(H2)
        else:
            H2out = self.conv2(H2)
        # combine along lat dimension
        out = tf.concat([H1out,H2out],axis=1)

        return out

    def compute_output_shape(self, input_shape):
        output_shape = input_shape[:3] + [self.filters]
        return output_shape


class SphericalResample(tf.keras.layers.Layer):

    def _gen_grid(self,kernel, Nlat, Nlon, stride):
        """generate the grid-coordinates for expanding the input tensor
        returns a grid with shape (Nlat*Nlon,len(kernel),2)
        each slice of the len(kernel) dimension conatains the coordinates for this particular point of the kernel
        """


        nkernel = len(kernel)
        # set up grid and loop over all target points.
        # the theta values start half a gridpoint off +-90degree
        grid = np.zeros((Nlat//stride, Nlon//stride, nkernel, 2))
        phi_arr = np.linspace(-np.pi/2*(1-1/Nlat), np.pi / 2*(1-1/Nlat), Nlat//stride) # radians
        theta_arr = np.linspace(0, 2*np.pi*(1-1/Nlon), Nlon//stride) # radians
        delta_phi_0 = np.pi/Nlat
        delta_theta_0 = 2*np.pi/Nlon
        print('computing interpolation coordinates')
        for ii in tqdm(range(0,Nlat,stride)):
            for jj in range(0,Nlon,stride):
                # center of the tantent plaen in radians
                phi_P = phi_arr[ii]
                theta_P = theta_arr[jj]
                for ikernel in  range(len(kernel)):
                    ix, iy = kernel[ikernel]
                    # convert to radians
                    x = ix*delta_theta_0
                    y = iy*delta_phi_0
                    rho = np.sqrt(x**2+y**2)
                    nu = np.arctan(rho)
                    # compute kernel point in lat and lon in radians ( eq. (11) in Coors2018)
                    # this only makes sense if not ix,iy = (0,0)
                    # if ii==15 and jj==10:
                    #     raise Exception()
                    if ix==0 and iy ==0:
                        lonidx = jj
                        latidx  = ii
                    else:
                        phi = np.arcsin(np.cos(nu)*np.sin(phi_P)+y*np.sin(nu)*np.cos(phi_P)/rho)
                        theta = theta_P + np.arctan(x*np.sin(nu)/(rho*np.cos(phi_P)*np.cos(nu)-y*np.sin(phi_P)*np.sin(nu)))

                        # now convert phi and theta to latidx and lonidx (these are floates, because
                        # they are interpolated indices!)
                        lonidx = theta * (Nlon)/(2*np.pi)
                        latidx = phi *Nlat/np.pi + (Nlat-1)/2
                        # for lon, everything that is > Nlon-1 is set to the same minus Nlon.
                        lonidx = lonidx % (Nlon)
                        latidx = latidx % (Nlat)

                        if latidx > Nlat - 1 :
                            latidx = 2*(Nlat-1) - latidx
                            # change lon by 180 degrees. since we work with interpolation anyway, we can shift
                            # for exactly 180 degrees, even when Nlon is not a multiple of 2
                            lonidx = (lonidx + Nlon/2) % (Nlon)
                        if latidx < 0:
                            latidx = -latidx
                            # change lon by 180 degrees
                            lonidx = (lonidx + Nlon / 2) % (Nlon)
                    grid[ii,jj,ikernel,0] = lonidx
                    grid[ii,jj,ikernel,1] = latidx

                    assert(~np.isnan(lonidx))
                    assert(~np.isnan(latidx))


        assert (grid.shape == (Nlat//stride, Nlon//stride, nkernel, 2))
        # flatten spatial dimension
        grid = grid.reshape((Nlat//stride * Nlon//stride, nkernel, 2))
        return grid

    def _to_flat_idx(self,lat, lon):
        return lon + lat * self.Nlon

    def _from_flat_idx(self,idx):
        lat = int(np.floor(idx / self.Nlon))
        lon = int(idx % self.Nlon)
        return lat, lon

    def gen_interp_matrix(self, grid):
        """
            generate sparse interpolation matrix that maps from the input Nlat*Nlon data to all points in 'grid'.
            returns: sparse tensor with shape (nkernel,Nlat*Nlon,n_interpolated_points)

        """
        # grid contains Nlat*Nlon tuples (x,y) for each filter

        sparse_idcs = []
        sparse_values = []
        n_target = grid.shape[0]
        # n_target is the number of flattened output points
        print('setting up interpolation matrix')
        for ikernel in tqdm(range(self.nkernel)):
            # get lons and lots of all target points for this kernel
            lats = grid[:, ikernel, 1]
            lons = grid[:, ikernel, 0]
            # loop over target points
            for ii in tqdm(range(n_target)):
                lat, lon = lats[ii], lons[ii]
                # in general each target point is a weighted average of 4 points, except
                # in the cases where the target point lies exactly on one gridpoint
                wlon1 = (np.ceil(lon) - lon)
                wlon2 = 1 - wlon1
                wlat1 = (np.ceil(lat)-lat)
                wlat2 = 1-wlat1

                w1 = (wlat1+wlon1)/4
                w2 = (wlat1+wlon2)/4
                w3 = (wlat2+wlon1)/4
                w4 = (wlat2+wlon2)/4

                np.testing.assert_allclose(w1+w2+w3+w4,1.0)

                lonidx1 = int(np.floor(lon))
                lonidx2 = int(np.ceil(lon))
                latidx1 = int(np.floor(lat))
                latidx2 = int(np.ceil(lat))

                # since we use ceil, it can be that
                # lonidx is the same as Nlon. in that case we set it to
                # 0 (wrapping around)
                if lonidx1 == self.Nlon:
                    lonidx1 = 0
                if lonidx2 == self.Nlon:
                    lonidx2 = 0
                assert(lonidx1<self.Nlon)
                assert(lonidx2<self.Nlon)

                # these are now the indices for lon on the 2d grid.
                # now we convert them to flat coordinates
                flat_idx1 = self._to_flat_idx(latidx1, lonidx1)
                flat_idx2 = self._to_flat_idx(latidx1, lonidx2)
                flat_idx3 = self._to_flat_idx(latidx2, lonidx1)
                flat_idx4 = self._to_flat_idx(latidx2, lonidx2)




                sparse_idcs.append([ikernel, flat_idx1, ii])
                sparse_values.append(w1)
                sparse_idcs.append([ikernel, flat_idx2, ii])
                sparse_values.append(w2)
                sparse_idcs.append([ikernel, flat_idx3, ii])
                sparse_values.append(w3)
                sparse_idcs.append([ikernel, flat_idx4, ii])
                sparse_values.append(w4)


        assert (np.max(sparse_idcs) < self.Nlat * self.Nlon)

        interp_matrix = tf.SparseTensor(indices=sparse_idcs, values=np.array(sparse_values, 'float32'),
                                        dense_shape=(self.nkernel, self.Nlat * self.Nlon, n_target),
                                        )
        # most ops on sprase tensors need a special order, this can be achieved with sparse.reorder
        interp_matrix = tf.sparse.reorder(interp_matrix)

        return interp_matrix

    def __init__(self, kernel,stride=1, **kwargs):
        """
            spherical expansion layer
            The transformation coefficients for the spherical expansion are computed only once (when building the layer)
            and then reused in every layer call.
            The image is expanded from Nlat,Nlon to kernel_size*Nlat*Nlon,  thus every gridpoint getes assigned
            a len(kernel) array. This array is the local projection of the data.
            kernel: list of (x,y) pairs in gridpointpcoordindates (can be float) on can any float, but lats must be whole number at the moment
                    in ordoer to allow also arbitray lats, gen_interp_matrix needs to be adapted. _gen_grid already works with
                    fractional lats
            conv_args: kwargs passed on to Convolution2D
        """

        super(SphericalResample, self).__init__(**kwargs)

        ys = [e[1] for e in kernel]
        if any([isinstance(e, float) for e in ys]):
            raise NotImplementedError('non-whole lat points in kernel not implemented!')
        self.kernel = kernel
        self.nkernel = len(kernel)
        self.stride=stride
        self.grid = None
        self.interp_matrix = None
        self.Nlat=None
        self.Nlon=None
        self.nchannel=None


    def build(self, input_shape):
        super(SphericalResample, self).build(input_shape)

        self.Nlat = input_shape[1]
        self.Nlon = input_shape[2]
        self.nchannel = input_shape[3]
        self.n_flatout = self.Nlat//self.stride * self.Nlon//self.stride

        if self.Nlat % self.stride !=0 or self.Nlon % self.stride !=0:
            raise ValueError('Nlat and Nlon must be multiples of stride')

        grid = self._gen_grid(self.kernel, self.Nlat, self.Nlon, self.stride)
        grid = grid.astype(np.float32)
        self.grid = grid

        self.interp_matrix = self.gen_interp_matrix(grid)

        super(SphericalResample, self).build(input_shape)

    # in order to be able to load the saved model we need to define
    # get_config
    def get_config(self):
        config = {
            'kernel': self.kernel,
            'stride': self.stride,
        }
        base_config = super(SphericalResample, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def call(self, x):

        # we have to extend the grid to the batch size
        # because the batch_size is dynamic, we cannot use tf.repeat_elements,
        # but we can use tf.tile
        _batch_size = tf.shape(x)[0]
        # interp_matrix = tf.sparse.to_dense(self.interp_matrix)
        interp_matrix = self.interp_matrix
        flat_x = tf.reshape(x, (_batch_size,self.Nlat*self.Nlon,self.nchannel))
        # now channel ist last dimension, but for the matrix multiplication we need the latlon dimension
        #to be the last one
        _x = tf.transpose(flat_x, (0,2,1))

        # to incoroporate the different kernel points, we reshaped the interp matrix
        # to incorporate batch_size and channels, we can reshape x
        # in the end we want to multipl (n_batch*n_kernel,Nlat*Nlon) x (Nlat*Nlon,nflatout*nkernel)

        m = tf.sparse.reshape(tf.sparse.transpose(interp_matrix, (1,2,0)),
                                                  (self.Nlat*self.Nlon,self.n_flatout*self.nkernel))
        # _x has shape (n_batch,n_channel,nflat)
        _x = tf.reshape (_x, (_batch_size*self.nchannel,self.Nlat*self.Nlon))
        # sparse_dense_matmul only supports sparse x dense, not dense x sparse.
        # therefore we have to do m x _x, and transppose both inputs with the adjoint keywords
        expanded = tf.sparse.sparse_dense_matmul(m,_x, adjoint_a=True, adjoint_b=True)
        # this now has shape (n_flatout*kkernel, batch_size*nchannel)
        expanded = tf.transpose(expanded)
        # reshape again to explicit batch and channel dim
        expanded = tf.reshape(expanded, (_batch_size, self.nchannel, self.n_flatout,self.nkernel))
        expanded = tf.transpose(expanded, (0,2,3,1))
        # assert(expanded.shape == (_batch_size, self.n_flatout,self.nkernel,self.nchannel))

        return expanded

    def compute_output_shape(self, input_shape):
        output_shape = (input_shape[0], self.Nlat//self.stride * self.Nlon//self.stride,self.nkerlen, self.nchannel )
        return tuple(output_shape)


class SphericalConv2D(SphericalResample):

    def __init__(self, filters, kernel,stride=1, conv_args={}, **kwargs):
        """
            sphereical convolution layer
            combines SphericalResample with following convolution. (Note: the same can be done via stacking
            a SphericalResample layer and a convolution layer with kernel_size=(1,nkernel)
            The transformation coefficients for the spherical expansion are computed only once (when building the layer)
            and then reused in every layer call.
            The image is expanded from Nlat,Nlon to kernel_size*Nlat*Nlon,  thus every gridpoint getes assigned
            a len(kernel) array. This array is the local projection of the data. Then convolution is done on
            this expanded data
            kernel: list of (x,y) pairs in gridpointpcoordindates. lon can any float, but lats must be whole number at the moment
                    in ordoer to allow also arbitray lats, gen_interp_matrix needs to be adapted. _gen_grid already works with
                    fractional lats
            conv_args: kwargs passed on to Convolution2D
        """

        super(SphericalConv2D, self).__init__(kernel, stride,**kwargs)

        self.filters=filters
        self.conv_args = conv_args
        self.conv1 = tf.keras.layers.Conv2D(self.filters, kernel_size=(1, self.nkernel), **conv_args)

    def build(self, input_shape):
        super(SphericalConv2D, self).build(input_shape)
        self._trainable_weights = self.conv1.trainable_weights

    def get_config(self):
        config = {
            'filters': self.filters,
            'conv_args':self.conv_args
        }
        base_config = super(SphericalConv2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def call(self, x):
        _batch_size = tf.shape(x)[0]
        expanded = super(SphericalConv2D, self).call(x)
        conv = self.conv1(expanded)
        # reshape to output shape
        reshaped = tf.reshape(conv,(_batch_size,self.Nlat//self.stride,self.Nlon//self.stride,self.filters))
        return reshaped

    def compute_output_shape(self, input_shape):
        output_shape = (input_shape[0], self.Nlat//self.stride, self.Nlon//self.stride,self.filters )
        return tuple(output_shape)


class SphericalHemConv2D(SphericalResample):

    def __init__(self, filters, stride=1, conv_args={}, **kwargs):
        """
        spherical convolution, seperate for NH and SH
        implemented with fixed 3x3 kernel

        """
        # first third of kernel is northern points, then points with dlat=0, then third with southern points
        # kernel points are (dlon,dlat)
        self.kernel = [[-1, 1], [0, 1], [1, 1],
                       [-1, 0], [0, 0], [1, 0],
                       [-1, -1], [0, -1], [1, -1]]

        super(SphericalHemConv2D, self).__init__(self.kernel, stride, **kwargs)

        self.filters = filters
        self.conv_args = conv_args
        self.conv1 = tf.keras.layers.Conv2D(self.filters, kernel_size=(1, self.nkernel), **conv_args)
        self.conv2 = tf.keras.layers.Conv2D(self.filters, kernel_size=(1, self.nkernel), **conv_args)

    def build(self, input_shape):
        super(SphericalHemConv2D, self).build(input_shape)
        self.nlat = input_shape[1]
        self.nlon = input_shape[2]
        if self.nlat % 2:
            raise ValueError('nlat must be even!')

    def get_config(self):
        config = {
            'filters': self.filters,
            'conv_args': self.conv_args,
        }
        base_config = super(SphericalHemConv2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def call(self, x):
        _batch_size = tf.shape(x)[0]
        expanded = super(SphericalHemConv2D, self).call(x)
        # split expanded input into NH and SH, with one row padding into the other hemisphere
        # the expanded data is on a flattened grid. the first half of the flattened dim is one hemisphere,
        # the second half is the other hemisphere
        nflat = self.nlat * self.nlon // self.stride ** 2
        H1 = expanded[:, :nflat // 2]
        H2 = expanded[:, -nflat // 2:]
        conv1 = self.conv1(H1)
        conv2 = self.conv2(H2)
        # combine
        combined = tf.concat([conv1, conv2], axis=1)
        # reshape to output shape
        reshaped = tf.reshape(combined, (_batch_size, self.Nlat // self.stride, self.Nlon // self.stride, self.filters))
        return reshaped

    def compute_output_shape(self, input_shape):
        output_shape = (input_shape[0], self.Nlat // self.stride, self.Nlon // self.stride, self.filters)
        return tuple(output_shape)


class SphericalHemConv2D_shared(SphericalResample):

    def __init__(self, filters, stride=1, conv_args={}, **kwargs):
        """
        spherical convolution, with same weights, but "flipped" for NH and SH
        implemented with fixed 3x3 kernel

        """
        # first third of kernel is northern points, then points with dlat=0, then third with southern points
        # kernel points are (dlon,dlat)
        self.kernel = [[-1, 1], [0, 1], [1, 1],
                       [-1, 0], [0, 0], [1, 0],
                       [-1, -1], [0, -1], [1, -1]]

        super(SphericalHemConv2D_shared, self).__init__(self.kernel, stride, **kwargs)

        self.filters = filters
        self.conv_args = conv_args
        self.conv1 = tf.keras.layers.Conv2D(self.filters, kernel_size=(1, self.nkernel), **conv_args)

    def build(self, input_shape):
        super(SphericalHemConv2D_shared, self).build(input_shape)
        self.nlat = input_shape[1]
        self.nlon = input_shape[2]
        if self.nlat % 2:
            raise ValueError('nlat must be even!')

    def get_config(self):
        config = {
            'filters': self.filters,
            'conv_args': self.conv_args,
        }
        base_config = super(SphericalHemConv2D_shared, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def call(self, x):
        _batch_size = tf.shape(x)[0]
        expanded = super(SphericalHemConv2D_shared, self).call(x)
        # split expanded input into NH and SH, with one row padding into the other hemisphere
        # the expanded data is on a flattened grid. the first half of the flattened dim is one hemisphere,
        # the second half is the other hemisphere
        nflat = self.nlat * self.nlon // self.stride ** 2
        H1 = expanded[:, :nflat // 2]
        H2 = expanded[:, -nflat // 2:]
        conv1 = self.conv1(H1)
        # we need to "flip" the data of the second hemisphere along the lat dimension. because our spatial dimension
        # is flattened, this is a bit tricky. we take advantage of the fixed kernel in the implementation.
        # our kernel has 9 points. so if we swap the first 3 with the last 3 of our kernel dimension in the data, this
        # amounts to flipping along the lat dimension. the order of dlon is -1,0,1 in boh thirds, so lon is not swapped
        H2_swapped = tf.gather(H2, [6, 7, 8, 3, 4, 5, 0, 1, 2], axis=2)
        conv2 = self.conv1(H2_swapped)
        # combine
        combined = tf.concat([conv1, conv2], axis=1)
        # reshape to output shape
        reshaped = tf.reshape(combined, (_batch_size, self.Nlat // self.stride, self.Nlon // self.stride, self.filters))
        return reshaped

    def compute_output_shape(self, input_shape):
        output_shape = (input_shape[0], self.Nlat // self.stride, self.Nlon // self.stride, self.filters)
        return tuple(output_shape)






