#Code written by Mike Follum to try and evaluate the mean flow from GEOGLOWS datasets.
#GEOGLOWS data can be downloaded from http://geoglows-v2.s3-website-us-west-2.amazonaws.com/

# built-in imports
import sys
import os

# third-party imports
import netCDF4   #conda install netCDF4
import numpy as np
import xarray as xr
import pandas as pd
from scipy.io import netcdf

def GetMeanFlowValues(NetCDF_Directory):
    # create a list of all files in the NetCDF directory
    file_list = os.listdir(NetCDF_Directory)
    all_mean_Qout_dfs = []
    for f in file_list:
        if f.endswith(".nc"):
            qout_file_path = os.path.join(NetCDF_Directory, f)
            qout_ds = xr.open_dataset(qout_file_path)
            
            # Compute the mean and Qout value over the 'time' dimension for each rivid
            mean_Qout_all_rivids = qout_ds['Qout'].mean(dim='time')

            # Trigger the computation if using Dask (although not necessary here since the dataset is 335MB)
            mean_Qout_all_rivids_values = mean_Qout_all_rivids.compute()

            # Convert the xarray DataArray to a pandas DataFrame
            mean_Qout_df = mean_Qout_all_rivids.to_dataframe(name='qout_mean').reset_index()
            
            all_mean_Qout_dfs.append(mean_Qout_df)
            
    # Concatenate all DataFrames into a single DataFrame
    if all_mean_Qout_dfs:
        all_mean_Qout_df = pd.concat(all_mean_Qout_dfs, ignore_index=True).round(3)
    else:
        print("No valid data found in the NetCDF files.")
        return None
    
    # Compute overall average by rivid
    overall_mean_Qout = all_mean_Qout_df.groupby('rivid')['qout_mean'].mean().round(3)
        
    return (overall_mean_Qout)

def GetMedianFlowValues(NetCDF_Directory):
    # Create a list of all files in the NetCDF directory
    file_list = os.listdir(NetCDF_Directory)
    all_median_Qout_dfs = []
    
    for f in file_list:
        if f.endswith(".nc"):
            qout_file_path = os.path.join(NetCDF_Directory, f)
            qout_ds = xr.open_dataset(qout_file_path)
            
            # Compute the median Qout value over the 'time' dimension for each rivid
            median_Qout_all_rivids = qout_ds['Qout'].median(dim='time')

            # Trigger the computation if using Dask (although not necessary here since the dataset is 335MB)
            median_Qout_all_rivids_values = median_Qout_all_rivids.compute()

            # Convert the xarray DataArray to a pandas DataFrame
            median_Qout_df = median_Qout_all_rivids.to_dataframe(name='qout_median').reset_index()
            
            all_median_Qout_dfs.append(median_Qout_df)
            
    # Concatenate all DataFrames into a single DataFrame
    if all_median_Qout_dfs:
        all_median_Qout_df = pd.concat(all_median_Qout_dfs, ignore_index=True).round(3)
    else:
        print("No valid data found in the NetCDF files.")
        return None
    
    # Compute overall median by rivid
    overall_median_Qout = all_median_Qout_df.groupby('rivid')['qout_median'].median().round(3)
        
    return overall_median_Qout

def GetMaxFlowValues(NetCDF_Directory):
    # create a list of all files in the NetCDF directory
    file_list = os.listdir(NetCDF_Directory)
    all_max_Qout_dfs = []
    for f in file_list:
        if f.endswith(".nc"):
            qout_file_path = os.path.join(NetCDF_Directory, f)
            qout_ds = xr.open_dataset(qout_file_path)
            
            # Compute the max Qout value over the 'time' dimension for each rivid
            max_Qout_all_rivids = qout_ds['Qout'].max(dim='time')

            # Trigger the computation if using Dask (although not necessary here since the dataset is 335MB)
            max_Qout_all_rivids_values = max_Qout_all_rivids.compute()

            # Convert the xarray DataArray to a pandas DataFrame
            max_Qout_df = max_Qout_all_rivids.to_dataframe(name='qout_max').reset_index()
            
            all_max_Qout_dfs.append(max_Qout_df)
            
    # Concatenate all DataFrames into a single DataFrame
    if all_max_Qout_dfs:
        all_max_Qout_df = pd.concat(all_max_Qout_dfs, ignore_index=True).round(3)
    else:
        print("No valid data found in the NetCDF files.")
        return None
    
    # Compute overall average by rivid
    overall_max_Qout = all_max_Qout_df.groupby('rivid')['qout_max'].max().round(3)
        
    return (overall_max_Qout)

def GetReturnPeriodFlowValues(NetCDF_Directory):
    # create a list of all files in the NetCDF directory
    file_list = os.listdir(NetCDF_Directory)
    all_max_Qout_dfs = []
    for f in file_list:
        if f.endswith(".nc"):
            qout_file_path = os.path.join(NetCDF_Directory, f)
            qout_ds = xr.open_dataset(qout_file_path)
            
    # Convert xarray Dataset to pandas DataFrame
    qout_df = qout_ds.to_dataframe()
            
    return (qout_df)

def Create_ARC_Streamflow_Input(NetCDF_RecurrenceInterval_Folder, NetCDF_Historical_Folder, Outfile_file_path):
    overall_median_Qout = GetMedianFlowValues(NetCDF_Historical_Folder)
    overall_median_Qout = abs(overall_median_Qout)
    overall_mean_Qout = GetMeanFlowValues(NetCDF_Historical_Folder)
    combined_df = GetReturnPeriodFlowValues(NetCDF_RecurrenceInterval_Folder)
    
    # Append Series to DataFrame using .loc indexer
    combined_df.loc[:, overall_mean_Qout.name] = overall_mean_Qout
    combined_df.loc[:, overall_median_Qout.name] = overall_median_Qout
    
    combined_df['COMID'] = combined_df.index
    
    # Define custom order of columns
    custom_order = ['COMID','qout_mean','qout_median','qout_max','rp2','rp5','rp10','rp25','rp50','rp100']
    
    # Sort columns in custom order
    combined_df = combined_df[custom_order]
    
    # Output the combined Dataframe as a CSV
    combined_df.to_csv(Outfile_file_path,index=False)
    
    return (combined_df)
   

if __name__ == "__main__":
    
    WatershedID = '714'
    NetCDF_RecurrenceInterval_Folder = '714_ReturnPeriods'
    NetCDF_Historical_Folder = '714_HistoricFlows'
    Outfile_file_path = 'GeoGLoWS_Flow_Data_' + str(WatershedID)  + 'JLG.csv'
    combined_df = Create_ARC_Streamflow_Input(NetCDF_RecurrenceInterval_Folder, NetCDF_Historical_Folder, Outfile_file_path)

  