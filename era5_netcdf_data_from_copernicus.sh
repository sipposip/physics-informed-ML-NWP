# originally all data was downloaded from the weatherbench repository.
# then it was converted to tfrecords, and after the first submission the netcdf files
# were deleted to free storage space.
# now we need the netcdf data again to compute the bust anomalies.
# therefore, the netcdf data for 500hpa geopotential for 2017-2018 is downloaded from 
# copernicus via the webform, and then regridded to our highres resolution

cdo remapbil,r256x128 era5_geopotential_500hpa_2017-1028_fullres.nc era5_geopotential_500hpa_2017-1028_highres.nc  