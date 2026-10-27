"""Streamflow preprocessing helpers.

This module contains optional utilities for producing ARC-compatible flow tables
from GEOGLOWS ECMWF Streamflow Service NetCDF inputs (historical retrospectives
and recurrence interval products).

The ARC core model only requires a reach ID column (Flow_File_ID) and a
maximum-flow column (Flow_File_QMax); a baseflow/bankfull column
(Flow_File_BF), which sets the bathymetry's depth, is optional. Richer flow
tables can still be used as long as the MIF points ARC to the correct column
names.
"""

# Code written by Mike Follum to try and evaluate the mean flow from GEOGLOWS datasets.
# GEOGLOWS data can be downloaded from http://geoglows-v2.s3-website-us-west-2.amazonaws.com/

# built-in imports
import os

import pandas as pd
import s3fs
import xarray as xr

# third-party imports
from osgeo import gdal
from shapely.geometry import box


def GetMeanFlowValues(NetCDF_Directory):
    """
    Estimates the mean streamflow for all stream reaches by cycling through a directory of yearly retrospective GEOGLOWS ECMWF Streamflow Service NetCDF files, 
    estimating the yearly mean, and then estimating a mean of those yearly means

    Parameters
    ----------
    NetCDF_Directory: str
        The file path to a directory containing yearly retrospective GEOGLOWS ECMWF Streamflow Service NetCDF files
    
    Returns
    -------
    overall_mean_Qout: Pandas series
        A Pandas series of mean streamflow values with the streams unique identifier used as the index

    """
    # create a list of all files in the NetCDF directory
    file_list = os.listdir(NetCDF_Directory)
    all_mean_Qout_dfs = []
    for f in file_list:
        if f.endswith(".nc"):
            qout_file_path = os.path.join(NetCDF_Directory, f)
            qout_ds = xr.open_dataset(qout_file_path, engine='netcdf4')
            
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
    """
    Estimates the median streamflow for all stream reaches by cycling through a directory of yearly retrospective GEOGLOWS ECMWF Streamflow Service NetCDF files, 
    estimating the yearly median, and then estimating a median of those yearly medians

    Parameters
    ----------
    NetCDF_Directory: str
        The file path to a directory containing yearly retrospective GEOGLOWS ECMWF Streamflow Service NetCDF files
    
    Returns
    -------
    overall_median_Qout: Pandas series
        A Pandas series of median streamflow values with the streams unique identifier used as the index

    """
    # Create a list of all files in the NetCDF directory
    file_list = os.listdir(NetCDF_Directory)
    all_median_Qout_dfs = []
    
    for f in file_list:
        if f.endswith(".nc"):
            qout_file_path = os.path.join(NetCDF_Directory, f)
            qout_ds = xr.open_dataset(qout_file_path, engine='netcdf4')
            
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

def GetReturnPeriodFlowValues(NetCDF_File_Path):
    """
    Reads a recurrence interval streamflow NetCDF file from the GEOGLOWS ECMWF Streamflow Service into a dataframe,
    as it is: a row for each stream reach and a column for each of the file's variables (such as qout_max and rp2 to
    rp100). Nothing is estimated.

    Parameters
    ----------
    NetCDF_File_Path: str
        The file path and file name of a NetCDF of recurrence interval streamflow file from the GEOGLOWS ECMWF Streamflow Service
    
    Returns
    -------
    qout_df: Pandas dataframe
        A Pandas dataframe of the values contained in the recurrence interval streamflow file from the GEOGLOWS ECMWF
        Streamflow Service, indexed by the file's stream reach IDs

    """
    # Open the NetCDF with xarray
    qout_ds = xr.open_dataset(NetCDF_File_Path, engine='netcdf4')
            
    # Convert xarray Dataset to pandas DataFrame
    qout_df = qout_ds.to_dataframe()
            
    return (qout_df)

def Create_ARC_Streamflow_Input(NetCDF_RecurrenceInterval_File_Path, NetCDF_Historical_Folder, Outfile_file_path):
    """
    Creates a streamflow input file that can be used by the Automated Rating Curve (ARC) tool.
    The Outfile_file_path will be a csv file that contains:

    - COMID or the stream reach's unique identifier (the recurrence interval file's)
    - average flow, cubic meters per second (qout_mean): the mean of each retrospective file's mean
    - median flow, cubic meters per second (qout_median): the median of each retrospective file's median, made positive
    - maximum flow, cubic meters per second (qout_max), copied, like the return period flows below, from the
      recurrence interval file
    - 2-year return period streamflow, cubic meters per second (rp2)
    - 5-year return period streamflow, cubic meters per second (rp5) 
    - 10-year return period streamflow, cubic meters per second (rp10)
    - 25-year return period streamflow, cubic meters per second (rp25)
    - 50-year return period streamflow, cubic meters per second (rp50)
    - 100-year return period streamflow, cubic meters per second (rp100)

    Parameters
    ----------
    NetCDF_RecurrenceInterval_File_Path: str
        The file path and file name of a NetCDF of recurrence interval streamflow file from the GEOGLOWS ECMWF Streamflow Service
    NetCDF_Historical_Folder: str
        The file path to a directory containing yearly retrospective GEOGLOWS ECMWF Streamflow Service NetCDF files
    Outfile_file_path: str
        The file path and file name of the file that will store the resulting streamflow inputs for ARC
    
    Returns
    -------
    combined_df: Pandas dataframe
        A Pandas dataframe of the columns written to Outfile_file_path: the mean and median streamflow values from the
        retrospective files, and the maximum, 2-year recurrence interval, 5-year recurrence interval, 10-year recurrence
        interval, 25-year recurrence interval, 50-year recurrence interval, and 100-year recurrence interval streamflow
        values contained in the recurrence interval streamflow file from the GEOGLOWS ECMWF Streamflow Service

    """
    print("Processing Median Values...\n")
    overall_median_Qout = GetMedianFlowValues(NetCDF_Historical_Folder)
    overall_median_Qout = abs(overall_median_Qout)
    print("Processing Mean Values...\n")
    overall_mean_Qout = GetMeanFlowValues(NetCDF_Historical_Folder)
    print("Processing Return Period Values...\n")
    combined_df = GetReturnPeriodFlowValues(NetCDF_RecurrenceInterval_File_Path)
    
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

def Process_and_Write_Retrospective_Data_for_DEM_Tile(StrmShp_gdf, rivid_field, DEM_Tile, CSV_File_Name, OutShp_File_Name):
    """Compute retrospective summary flows for a single DEM tile.

    This helper filters the stream network to features lying wholly within the
    DEM tile bounds, writes them to OutShp_File_Name, and then writes a flow
    summary table for that subset to CSV_File_Name: from the GEOGLOWS
    retrospective data on S3, each reach's median flow (qout_median, exceeded
    50% of the time), maximum flow (qout_max, exceeded 0% of the time) and
    return period flows (rp2 and so on), and qout_max_premium and rp100_premium,
    1.5 times qout_max and rp100.

    Returns (CSV_File_Name, OutShp_File_Name, the reach IDs as integers, the
    filtered stream network), or four Nones if the tile has no stream features
    or no flows.
    """

    # Load the raster tile and get its bounds using gdal
    raster_dataset = gdal.Open(DEM_Tile)
    gt = raster_dataset.GetGeoTransform()

    # Get the bounds of the raster (xmin, ymin, xmax, ymax), whichever way up it is
    xmin, xmax = sorted((gt[0], gt[0] + gt[1] * raster_dataset.RasterXSize))
    ymin, ymax = sorted((gt[3], gt[3] + gt[5] * raster_dataset.RasterYSize))

    # Create a bounding box
    raster_bbox = box(xmin, ymin, xmax, ymax)

    # Use GeoPandas spatial index to quickly find geometries within the bounding box
    sindex = StrmShp_gdf.sindex
    possible_matches_index = list(sindex.intersection(raster_bbox.bounds))
    possible_matches = StrmShp_gdf.iloc[possible_matches_index]

    # Collect IDs of polyline features within the raster tile boundary
    StrmShp_filtered_gdf = possible_matches[possible_matches.geometry.within(raster_bbox)]

    # Check if StrmShp_filtered_gdf is empty
    if StrmShp_filtered_gdf.empty:
        print(f"Skipping processing for {DEM_Tile} because StrmShp_filtered_gdf is empty.")
        CSV_File_Name = None
        OutShp_File_Name = None
        rivids_int = None
        StrmShp_filtered_gdf = None
        return (CSV_File_Name, OutShp_File_Name, rivids_int, StrmShp_filtered_gdf)
    
    StrmShp_filtered_gdf.to_file(OutShp_File_Name)
    StrmShp_filtered_gdf[rivid_field] = StrmShp_filtered_gdf[rivid_field].astype(int)

    # create a list of river IDs to throw to AWS
    rivids_str = StrmShp_filtered_gdf[rivid_field].astype(str).to_list()
    rivids_int = StrmShp_filtered_gdf[rivid_field].astype(int).to_list()

    # Set up the S3 connection
    ODP_S3_BUCKET_REGION = 'us-west-2'
    s3 = s3fs.S3FileSystem(anon=True, client_kwargs=dict(region_name=ODP_S3_BUCKET_REGION))

    # Open the flow duration curve (FDC) data on S3, for the median (p_exceed 50) and maximum (p_exceed 0) flows
    fdc_s3_uri = 's3://geoglows-v2-retrospective/fdc.zarr'
    fdc_s3store = s3fs.S3Map(root=fdc_s3_uri, s3=s3, check=False)
    p_exceedance = [50.0, 0.0]
    fdc_ds = xr.open_zarr(fdc_s3store).sel(p_exceed=p_exceedance, river_id=rivids_str)
    # Convert Xarray to a pandas DataFrame
    fdc_df = fdc_ds.to_dataframe().reset_index()

    # Check if fdc_df is empty
    if fdc_df.empty:
        print(f"Skipping processing for {DEM_Tile} because fdc_df is empty.")
        CSV_File_Name = None
        OutShp_File_Name = None
        rivids_int = None
        StrmShp_filtered_gdf = None
        return (CSV_File_Name, OutShp_File_Name, rivids_int, StrmShp_filtered_gdf)

    # Create 'qout_median' column where 'p_exceed' is 50.0
    fdc_df.loc[fdc_df['p_exceed'] == 50.0, 'qout_median'] = fdc_df['fdc']
    # Create 'qout_max' column where 'p_exceed' is 0.0 (the flow exceeded 0% of the time)
    fdc_df.loc[fdc_df['p_exceed'] == 0.0, 'qout_max'] = fdc_df['fdc']
    # Group by 'river_id' and aggregate 'qout_median' and 'qout_max' by taking the non-null value
    fdc_df = fdc_df.groupby('river_id').agg({
        'qout_median': 'max',  # each has one non-null value, so 'first' would work too
        'qout_max': 'max'
    }).reset_index()

    # making our index for this dataframe match the recurrence interval index 
    fdc_df['rivid'] = fdc_df['river_id'].astype(int)
    # Drop the river_id column from the DataFrame
    fdc_df = fdc_df.drop(['river_id'], axis=1)
    fdc_df = fdc_df.set_index('rivid')

    # round the values
    fdc_df['qout_median'] = fdc_df['qout_median'].round(3)
    fdc_df['qout_max'] = fdc_df['qout_max'].round(3)
    
    # Open the return periods data on S3
    rp_s3_uri = 's3://geoglows-v2-retrospective/return-periods.zarr'
    rp_s3store = s3fs.S3Map(root=rp_s3_uri, s3=s3, check=False)
    rp_ds = xr.open_zarr(rp_s3store).sel(rivid=rivids_int)
    
    # Convert Xarray to a pandas DataFrame, to pivot below
    rp_df = rp_ds.to_dataframe().reset_index()

    # Check if rp_df is empty
    if rp_df.empty:
        print(f"Skipping processing for {DEM_Tile} because rp_df is empty.")
        CSV_File_Name = None
        OutShp_File_Name = None
        rivids_int = None
        StrmShp_filtered_gdf = None
        return (CSV_File_Name, OutShp_File_Name, rivids_int, StrmShp_filtered_gdf)

    # Convert 'return_period' to category dtype
    rp_df['return_period'] = rp_df['return_period'].astype('category')
    
    # Pivot the table
    rp_pivot_df = rp_df.pivot_table(index='rivid', columns='return_period', values='return_period_flow', aggfunc='mean')

    # Rename columns to indicate return periods
    rp_pivot_df = rp_pivot_df.rename(columns={col: f'rp{int(col)}' for col in rp_pivot_df.columns})

    # Combine the results from retrospective and return periods data
    # final_df = pd.concat([combined_df, rp_pivot_df], axis=1)
    final_df = pd.concat([fdc_df, rp_pivot_df], axis=1)
    final_df['COMID'] = final_df.index

    # Column to move to the front
    target_column = 'COMID'

    # Reorder the DataFrame
    columns = [target_column] + [col for col in final_df.columns if col != target_column]
    final_df = final_df[columns]

    # Add a safety factor to the columns we could use as ARC's maximum flow: qout_max_premium and rp100_premium,
    # 1.5 times them
    for col in final_df.columns:
        if col in ['qout_max','rp100']:
            final_df[f'{col}_premium'] = round(final_df[col]*1.5, 3)
    
    # Write the final pandas DataFrame to CSV
    final_df.to_csv(CSV_File_Name, index=False)
    
    # Return the file names, the reach IDs and the tile's stream network
    
    return (CSV_File_Name, OutShp_File_Name, rivids_int, StrmShp_filtered_gdf)
