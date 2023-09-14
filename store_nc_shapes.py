
import pickle
import os
import xarray as xr

os.system('mkdir -p data')
inpath = '/climstorage/sebastian/nn_reanalysis/era5_benchmarkdata/'

resolution = '2.8125'
#resolution='1.40625'
ifile = f'{inpath}/geopotential_1979_{resolution}deg.nc'

data = xr.open_dataarray(ifile)

pickle.dump(data.coords, open(f'data/coords_{resolution}.pkl','wb'))