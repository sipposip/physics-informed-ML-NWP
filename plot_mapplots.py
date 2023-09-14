import os
import numpy as np
import pandas as pd
import cartopy.crs as ccrs
import xarray as xr
from pylab import plt
import seaborn as sns


#resolution = '2.8125'
resolution='1.40625'

#Note: to save spaec in HOME, the input data for this script is not in .data, but in /data/nn_reanalysis/data
base = pd.read_pickle(f'data/fc_res_unet_base_{resolution}_itrain0.pkl')
vars = ['geopotential500', 'temperature850']

vmax_per_var = {
    'geopotential500':1500,
    'geopotential850':1500,
    'temperature500':5,
    'temperature850':5,
}

diffmax_per_var = {
    'geopotential500':700,
    'geopotential850':700,
    'temperature500':1,
    'temperature850':1.5,
}

labels = {
    'geopotential500':'$[m^2/s^2]$',
    'geopotential850':'$[m^2/s^2]$',
    'temperature500':'$[K]$',
    'temperature850':'$[K]$',
}

sns.set(style=None, rc=None ,font_scale=2)

n_train = 4

for fcday in np.arange(11):
    for varname in vars:

        plt.figure()
        da_base = []
        for i_train in range(n_train):
            ifile_base = f'data/fc_res_2d_unet_base_{resolution}_{varname}_itrain{i_train}_day{fcday}.nc'
            da = xr.open_dataarray(ifile_base)
            da_base.append(da)
        da_base = xr.concat(da_base,'mem')
        # compute mean over members
        da_base = da_base.mean('mem')

        da_sphereconv = []
        for i_train in range(n_train):
            ifile_sphereconv = f'data/fc_res_2d_unet_sphereconv_{resolution}_{varname}_itrain{i_train}_day{fcday}.nc'
            da = xr.open_dataarray(ifile_sphereconv)
            da_sphereconv.append(da)
        da_sphereconv = xr.concat(da_sphereconv,'mem')
        # compute mean over members
        da_sphereconv = da_sphereconv.mean('mem')

        #vmax = np.max([da_base.max(), da_sphereconv.max()])
        #vmax = 1000
        vmax = vmax_per_var[varname]
        vmin=0
        plt.subplot(3,1,1)
        da_base.plot(vmax=vmax, vmin=vmin,  cmap=plt.cm.magma_r, extend='max')
        plt.title('base')

        plt.subplot(3, 1, 2)
        da_sphereconv.plot(vmax=vmax, vmin=vmin, cmap=plt.cm.magma_r, extend='max')
        plt.title('sphereconv')

        plt.subplot(3, 1, 3)
        (da_sphereconv - da_base).plot(vmin=-diffmax_per_var[varname], vmax=diffmax_per_var[varname],
                                       extend='both',
                                       cmap=plt.cm.RdBu_r)
        plt.title('sphereconv - base')
        plt.suptitle(f'day {fcday} {varname}')

        plt.savefig(f'plots/error_mapplots_{resolution}_{varname}_day{fcday}.png')


        plt.close('all')


# all leadtimes (except 0 ) in one plot
for nettype in ('base', 'sphereconv', 'sphereconv_hemconv_shared',
                'hemconv_sharedweights'):
    for varname in vars:
        base = []
        ref = []
        # omit day zero (because rmse is 0 anyway), and go up to day 9 (1-9 is easy to plot,
        # so we omit day 10)
        for fcday in np.arange(1,10):

            da_base = []
            for i_train in range(n_train):
                ifile_base = f'data/fc_res_2d_unet_base_{resolution}_{varname}_itrain{i_train}_day{fcday}.nc'
                da = xr.open_dataarray(ifile_base)
                da_base.append(da)
            da_base = xr.concat(da_base, 'mem')
            # compute mean over members
            da_base = da_base.mean('mem')
            da_base = da_base.assign_coords(fcday=fcday)
            base.append(da_base)

            da_ref = []
            for i_train in range(n_train):
                ifile_ref = f'data/fc_res_2d_unet_{nettype}_{resolution}_{varname}_itrain{i_train}_day{fcday}.nc'
                da = xr.open_dataarray(ifile_ref)
                da_ref.append(da)
            da_ref = xr.concat(da_ref, 'mem')
            # compute mean over members
            da_ref = da_ref.mean('mem')
            da_ref = da_ref.assign_coords(fcday=fcday)
            ref.append(da_ref)

        base= xr.concat(base,'fcday')
        ref= xr.concat(ref,'fcday')

        diff = ref - base

        plt.rcParams.update({'font.size': 18})
        plt.figure(figsize=(6,3))
        p = base.plot(col='fcday', cmap=plt.cm.magma_r, col_wrap=3,
                subplot_kws=dict(projection=ccrs.PlateCarree()),
                    aspect=1.8, cbar_kwargs={'label': labels[varname]})
        for ax in p.axes.flat:
            ax.coastlines()
            ax.gridlines(linestyle='--', color='grey', alpha=0.5)
        plt.subplots_adjust(hspace=0.0, wspace=0.01, right=0.77)
        plt.suptitle(f'base {varname}', fontsize=25)
        plt.savefig(f'plots/fcdaygridplot_base_{varname}_{resolution}.png', bbox_inches='tight')

        plt.figure()
        # plot, but ommit fcday 0
        p = ref.plot(col='fcday',cmap=plt.cm.magma_r, col_wrap=3,
                subplot_kws=dict(projection=ccrs.PlateCarree()),
                    aspect=1.8, cbar_kwargs={'label': labels[varname]})
        for ax in p.axes.flat:
            ax.coastlines()
            ax.gridlines(linestyle='--', color='grey', alpha=0.5)
        plt.subplots_adjust(hspace=0.0, wspace=0.01, right=0.77)
        plt.suptitle(f'{nettype} {varname}', fontsize=25)
        plt.savefig(f'plots/fcdaygridplot_sphereconv_{nettype}_{varname}_{resolution}.png', bbox_inches='tight')

        plt.figure(figsize=(10,5))
        # plot, but ommit fcday 0
        p = diff.plot(col='fcday', col_wrap=3, cmap=plt.cm.RdBu_r,
                    vmin=-diffmax_per_var[varname], vmax=diffmax_per_var[varname],
                subplot_kws=dict(projection=ccrs.PlateCarree()),
                    aspect=1.8, cbar_kwargs={'label': labels[varname]})
        for ax in p.axes.flat:
            ax.coastlines()
            ax.gridlines(linestyle='--', color='grey', alpha=0.5)
        plt.subplots_adjust(hspace=0.0, wspace=0.01, right=0.77)
        plt.suptitle(f'{nettype}-base {varname}', fontsize=25)
        plt.savefig(f'plots/fcdaygridplot_diff_{nettype}_{varname}_{resolution}.png', bbox_inches='tight')

        plt.close('all')



# difference between 'sphereconv_hemconv_shared' and 'hemconv_sharedweights'
for varname in vars:
    base = []
    ref = []
    # omit day zero (because rmse is 0 anyway), and go up to day 9 (1-9 is easy to plot,
    # so we omit day 10)
    for fcday in np.arange(1,10):

        da_base = []
        for i_train in range(n_train):
            ifile_base = f'data/fc_res_2d_unet_hemconv_sharedweights_{resolution}_{varname}_itrain{i_train}_day{fcday}.nc'

            da = xr.open_dataarray(ifile_base)
            da_base.append(da)
        da_base = xr.concat(da_base, 'mem')
        # compute mean over members
        da_base = da_base.mean('mem')
        da_base = da_base.assign_coords(fcday=fcday)
        base.append(da_base)

        da_ref = []
        for i_train in range(n_train):
            ifile_ref = f'data/fc_res_2d_unet_sphereconv_hemconv_shared_{resolution}_{varname}_itrain{i_train}_day{fcday}.nc'
            da = xr.open_dataarray(ifile_ref)
            da_ref.append(da)
        da_ref = xr.concat(da_ref, 'mem')
        # compute mean over members
        da_ref = da_ref.mean('mem')
        da_ref = da_ref.assign_coords(fcday=fcday)
        ref.append(da_ref)

    base= xr.concat(base,'fcday')
    ref= xr.concat(ref,'fcday')

    diff = ref - base
    plt.figure(figsize=(10,5))
    # plot, but ommit fcday 0
    p = diff.plot(col='fcday', col_wrap=3, cmap=plt.cm.RdBu_r,
                vmin=-diffmax_per_var[varname], vmax=diffmax_per_var[varname],
            subplot_kws=dict(projection=ccrs.PlateCarree()),
                aspect=1.8, cbar_kwargs={'label': labels[varname]})
    for ax in p.axes.flat:
        ax.coastlines()
        ax.gridlines(linestyle='--', color='grey', alpha=0.5)
    plt.subplots_adjust(hspace=0.0, wspace=0.01, right=0.77)
    plt.suptitle(f'sphereconv_hemconv_shared - hemconv_shared {varname}', fontsize=25)
    plt.savefig(f'plots/fcdaygridplot_diff_sphereconv_hemconv_shared-_hemconv_shared{varname}_{resolution}.png',
    bbox_inches='tight')