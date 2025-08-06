# Step 1: Import necessary libraries
import xarray as xr
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor
import pandas as pd
import matplotlib.pyplot as plt
import netCDF4
import nc_time_axis

# Step 2: Load the netCDF datasets
vo_data_19511970 = xr.open_dataset('data/vo_Omon_GISS-E2-1-G_historical_r101i1p1f1_gn_195101-197012.nc').fillna(0)
vo_data_19711990 = xr.open_dataset('data/vo_Omon_GISS-E2-1-G_historical_r101i1p1f1_gn_197101-199012.nc').fillna(0)
vo_data_19912010 = xr.open_dataset('data/vo_Omon_GISS-E2-1-G_historical_r101i1p1f1_gn_199101-201012.nc').fillna(0)
vo_data_20112014 = xr.open_dataset('data/vo_Omon_GISS-E2-1-G_historical_r101i1p1f1_gn_201101-201412.nc').fillna(0)

uo_data_19511970 = xr.open_dataset('data/uo_Omon_GISS-E2-1-G_historical_r101i1p1f1_gn_195101-197012.nc').fillna(0)
uo_data_19711990 = xr.open_dataset('data/uo_Omon_GISS-E2-1-G_historical_r101i1p1f1_gn_197101-199012.nc').fillna(0)
uo_data_19912010 = xr.open_dataset('data/uo_Omon_GISS-E2-1-G_historical_r101i1p1f1_gn_199101-201012.nc').fillna(0)
uo_data_20112014 = xr.open_dataset('data/uo_Omon_GISS-E2-1-G_historical_r101i1p1f1_gn_201101-201412.nc').fillna(0)

tos_data_19512000 = xr.open_dataset('data/tos_Omon_GISS-E2-1-G_historical_r101i1p1f1_gn_195101-200012.nc').fillna(0)
tos_data_20012014 = xr.open_dataset('data/tos_Omon_GISS-E2-1-G_historical_r101i1p1f1_gn_200101-201412.nc').fillna(0) 

psl_data_19512000 = xr.open_dataset('data/psl_Amon_GISS-E2-1-G_historical_r101i1p1f1_gn_195101-200012.nc').fillna(0)
psl_data_20012014 = xr.open_dataset('data/psl_Amon_GISS-E2-1-G_historical_r101i1p1f1_gn_200101-201412.nc').fillna(0)

# Merge the temperature datasets
temperature_data = xr.merge([tos_data_19512000['tos'], tos_data_20012014['tos']])

# Merge the ocean current datasets
ocean_current_data_vo = xr.merge([vo_data_19511970['vo'],vo_data_19711990['vo'], vo_data_19912010['vo'], vo_data_20112014['vo']])
ocean_current_data_uo = xr.merge([vo_data_19511970['vo'],uo_data_19711990['uo'], uo_data_19912010['uo'], uo_data_20112014['uo']])

# Merge the sea level pressure datasets
psl_data = xr.merge([psl_data_19512000['psl'], psl_data_20012014['psl']])

ocean_current_data_vo_surface = ocean_current_data_vo.isel(lev=0)
ocean_current_data_uo_surface = ocean_current_data_uo.isel(lev=0)

temperature_data.to_netcdf('_data/_temperature_data.nc')
ocean_current_data_vo_surface.to_netcdf('_data/_ocean_current_data_vo_surface.nc')
ocean_current_data_uo_surface.to_netcdf('_data/_ocean_current_data_uo_surface.nc')
psl_data.to_netcdf('_data/_psl_data.nc')