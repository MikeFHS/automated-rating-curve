#Code written by Mike Follum to try and evaluate the mean flow from GEOGLOWS datasets.
#GEOGLOWS data can be downloaded from http://geoglows-v2.s3-website-us-west-2.amazonaws.com/

# built-in imports
import gc
import os
import sys

# third-party imports
import dask.array as da
import dask.dataframe as dd
from dask.diagnostics import ProgressBar
# import geoglows       #pip install geoglows -q     #conda install pip      #https://gist.github.com/rileyhales/873896e426a5bd1c4e68120b286bc029
import geopandas as gpd
#import netCDF4   #conda install netCDF4
import numpy as np
from osgeo import gdal, osr
import pandas as pd
from scipy.io import netcdf
from shapely.geometry import box
import s3fs
import xarray as xr



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

def GetMaxFlowValues(NetCDF_Directory):
    """
    Estimates the maximum streamflow for all stream reaches by cycling through a directory of yearly retrospective GEOGLOWS ECMWF Streamflow Service NetCDF files, 
    estimating the yearly maximum, and then estimating a maximum of those yearly maximums

    Parameters
    ----------
    NetCDF_Directory: str
        The file path to a directory containing yearly retrospective GEOGLOWS ECMWF Streamflow Service NetCDF files
    
    Returns
    -------
    overall_median_Qout: Pandas series
        A Pandas series of maximum streamflow values with the streams unique identifier used as the index

    """
    # create a list of all files in the NetCDF directory
    file_list = os.listdir(NetCDF_Directory)
    all_max_Qout_dfs = []
    for f in file_list:
        if f.endswith(".nc"):
            qout_file_path = os.path.join(NetCDF_Directory, f)
            qout_ds = xr.open_dataset(qout_file_path, engine='netcdf4')
            
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

def GetReturnPeriodFlowValues(NetCDF_File_Path):
    """
    Estimates the maximum streamflow for all stream reaches by cycling through a directory of yearly retrospective GEOGLOWS ECMWF Streamflow Service NetCDF files, 
    estimating the yearly maximum, and then estimating a maximum of those yearly maximums

    Parameters
    ----------
    NetCDF_File_Path: str
        The file path and file name of a NetCDF of recurrence interval streamflow file from the GEOGLOWS ECMWF Streamflow Service
    
    Returns
    -------
    qout_df: Pandas dataframe
        A Pandas dataframe of the recurrence interval values contained in the recurrence interval streamflow file from the GEOGLOWS ECMWF Streamflow Service

    """
    # Open the NetCDF with xarray
    qout_ds = xr.open_dataset(NetCDF_File_Path, engine='netcdf4')
            
    # Convert xarray Dataset to pandas DataFrame
    qout_df = qout_ds.to_dataframe()
            
    return (qout_df)

def Create_ARC_Streamflow_Input(NetCDF_RecurrenceInterval_File_Path, NetCDF_Historical_Folder, Outfile_file_path):
    """
    Creates a streamflow input file that can be used by the Automated Rating Curve (ARC) tool

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
        A Pandas dataframe of the mean, median, maximum, 2-year recurrence interval, 5-year recurrence interval, 10-year recurrence interval, 25-year recurrence interval,
        50-year recurrence interval, and 100-year recurrence interval streamflow values contained in the recurrence interval streamflow file 
        from the GEOGLOWS ECMWF Streamflow Service

    """
    overall_median_Qout = GetMedianFlowValues(NetCDF_Historical_Folder)
    overall_median_Qout = abs(overall_median_Qout)
    overall_mean_Qout = GetMeanFlowValues(NetCDF_Historical_Folder)
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

def Process_and_Write_Retrospective_Data(StrmShp_gdf, rivid_field, CSV_File_Name):
    rivids = StrmShp_gdf[rivid_field].astype(int).values

    # Set up the S3 connection
    ODP_S3_BUCKET_REGION = 'us-west-2'
    s3 = s3fs.S3FileSystem(anon=True, client_kwargs=dict(region_name=ODP_S3_BUCKET_REGION))

    # Enable Dask progress bar
    with ProgressBar():
    
        # Load retrospective data from S3 using Dask
        retro_s3_uri = 's3://geoglows-v2-retrospective/retrospective.zarr'
        retro_s3store = s3fs.S3Map(root=retro_s3_uri, s3=s3, check=False)
        retro_ds = xr.open_zarr(retro_s3store, chunks='auto').sel(rivid=rivids)
        
        # Convert Xarray to Dask DataFrame
        retro_ddf = retro_ds.to_dask_dataframe().reset_index()

        # Perform groupby operations in Dask for mean, median, and max
        mean_ddf = retro_ddf.groupby('rivid').Qout.mean().rename('qout_mean').reset_index()
        median_ddf = retro_ddf.groupby('rivid').Qout.median().rename('qout_median').reset_index()
        max_ddf = retro_ddf.groupby('rivid').Qout.max().rename('qout_max').reset_index()

        # Set the index for alignment and repartition
        mean_ddf = mean_ddf.set_index('rivid')
        median_ddf = median_ddf.set_index('rivid')
        max_ddf = max_ddf.set_index('rivid')

        # Repartition to align the partitions
        mean_ddf = mean_ddf.repartition(npartitions=10)
        median_ddf = median_ddf.repartition(npartitions=10)
        max_ddf = max_ddf.repartition(npartitions=10)

        # Align partitions
        combined_ddf = dd.concat([
            mean_ddf,
            median_ddf,
            max_ddf
        ], axis=1)

    # Clean up memory
    del retro_ds, retro_ddf, mean_ddf, median_ddf, max_ddf
    gc.collect()

    # Enable Dask progress bar
    with ProgressBar():
    
        # Load return periods data from S3 using Dask
        rp_s3_uri = 's3://geoglows-v2-retrospective/return-periods.zarr'
        rp_s3store = s3fs.S3Map(root=rp_s3_uri, s3=s3, check=False)
        rp_ds = xr.open_zarr(rp_s3store, chunks='auto').sel(rivid=rivids)
        
        # Convert Xarray to Dask DataFrame and pivot
        rp_ddf = rp_ds.to_dask_dataframe().reset_index()

        # Convert 'return_period' to category dtype
        rp_ddf['return_period'] = rp_ddf['return_period'].astype('category')

        # Ensure the categories are known
        rp_ddf['return_period'] = rp_ddf['return_period'].cat.as_known()
        
        # Pivot the table
        rp_pivot_ddf = rp_ddf.pivot_table(index='rivid', columns='return_period', values='return_period_flow', aggfunc='mean')

        # Rename columns to indicate return periods
        rp_pivot_ddf = rp_pivot_ddf.rename(columns={col: f'rp{int(col)}' for col in rp_pivot_ddf.columns})

        # Set the index for rp_pivot_ddf and ensure known divisions
        rp_pivot_ddf = rp_pivot_ddf.reset_index().set_index('rivid').repartition(npartitions=rp_pivot_ddf.npartitions)
        rp_pivot_ddf = rp_pivot_ddf.set_index('rivid', sorted=True)

    # Clean up memory
    del rp_ds, rp_ddf
    gc.collect()
    
    # # Align partitions
    # aligned_dfs, divisions, result = dd.multi.align_partitions(combined_ddf, rp_pivot_ddf)

    # # Extract aligned DataFrames
    # aligned_combined_ddf = aligned_dfs[0]
    # aligned_rp_pivot_ddf = aligned_dfs[1]

    # Repartition to align the partitions
    aligned_combined_ddf = combined_ddf.repartition(npartitions=10)
    aligned_rp_pivot_ddf = rp_pivot_ddf.repartition(npartitions=10)

    # Combine the results from retrospective and return periods data
    final_ddf = dd.concat([aligned_combined_ddf, aligned_rp_pivot_ddf], axis=1)

    # Write the final Dask DataFrame to CSV
    final_ddf.to_csv(CSV_File_Name, single_file=True, index=False)

    # Clean up memory
    del rp_pivot_ddf, combined_ddf, final_ddf
    gc.collect()
    
    # Return the combined DataFrame as a Dask DataFrame
    return

def Process_and_Write_Retrospective_Data_for_DEM_Tile(StrmShp_gdf, rivid_field, DEM_Tile, CSV_File_Name, OutShp_File_Name):

    # Load the raster tile and get its bounds using gdal
    raster_dataset = gdal.Open(DEM_Tile)
    gt = raster_dataset.GetGeoTransform()

    # Get the bounds of the raster (xmin, ymin, xmax, ymax)
    xmin = gt[0]
    xmax = xmin + gt[1] * raster_dataset.RasterXSize
    ymin = gt[3] + gt[5] * raster_dataset.RasterYSize
    ymax = gt[3]

    raster_bounds = (xmin, ymin, xmax, ymax)
    raster_bbox = box(*raster_bounds)  # Create a shapely box from raster bounds

    # Check which polyline features are within the raster tile boundary
    StrmShp_gdf['within_raster'] = StrmShp_gdf.geometry.apply(lambda geom: geom.within(raster_bbox))

    # Collect IDs of polyline features within the raster tile boundary
    DEM_StrmShp_gdf = StrmShp_gdf.loc[StrmShp_gdf['within_raster']==True]
    DEM_StrmShp_gdf.to_file(OutShp_File_Name)

    rivids = DEM_StrmShp_gdf[rivid_field].values


    # Set this back to false for the next go round with another DEM tile
    StrmShp_gdf['within_raster'] = False

    # Set up the S3 connection
    ODP_S3_BUCKET_REGION = 'us-west-2'
    s3 = s3fs.S3FileSystem(anon=True, client_kwargs=dict(region_name=ODP_S3_BUCKET_REGION))

    # Load retrospective data from S3 using Dask
    retro_s3_uri = 's3://geoglows-v2-retrospective/retrospective.zarr'
    retro_s3store = s3fs.S3Map(root=retro_s3_uri, s3=s3, check=False)
    retro_ds = xr.open_zarr(retro_s3store).sel(rivid=rivids)

    # Convert Xarray to Dask DataFrame
    retro_df = retro_ds.to_dataframe().reset_index()

    # Perform groupby operations in Dask for mean, median, and max
    mean_df = retro_df.groupby('rivid').Qout.mean().round(3).rename('qout_mean').reset_index()
    median_df = retro_df.groupby('rivid').Qout.median().round(3).rename('qout_median').reset_index()
    max_df = retro_df.groupby('rivid').Qout.max().round(3).rename('qout_max').reset_index()

    # Set the index for alignment and repartition
    mean_df = mean_df.set_index('rivid')
    median_df = median_df.set_index('rivid')
    max_df = max_df.set_index('rivid')

    # Align partitions
    combined_df = pd.concat([ mean_df,
                               median_df,
                               max_df
                            ], axis=1)

    # Clean up memory
    del retro_ds, retro_df, mean_df, median_df, max_df
    gc.collect()

    # Enable Dask progress bar
    with ProgressBar():
    
        # Load return periods data from S3 using Dask
        rp_s3_uri = 's3://geoglows-v2-retrospective/return-periods.zarr'
        rp_s3store = s3fs.S3Map(root=rp_s3_uri, s3=s3, check=False)
        rp_ds = xr.open_zarr(rp_s3store).sel(rivid=rivids)
        
        # Convert Xarray to Dask DataFrame and pivot
        rp_df = rp_ds.to_dataframe().reset_index()

        # Convert 'return_period' to category dtype
        rp_df['return_period'] = rp_df['return_period'].astype('category')
        
        # Pivot the table
        rp_pivot_df = rp_df.pivot_table(index='rivid', columns='return_period', values='return_period_flow', aggfunc='mean')

        # Rename columns to indicate return periods
        rp_pivot_df = rp_pivot_df.rename(columns={col: f'rp{int(col)}' for col in rp_pivot_df.columns})

    # Clean up memory
    del rp_ds, rp_df
    gc.collect()

    # Combine the results from retrospective and return periods data
    final_df = pd.concat([combined_df, rp_pivot_df], axis=1)

    final_df['COMID'] = final_df.index

    # Column to move to the front
    target_column = 'COMID'

    # Reorder the DataFrame
    columns = [target_column] + [col for col in final_df.columns if col != target_column]
    final_df = final_df[columns]

    # Write the final Dask DataFrame to CSV
    final_df.to_csv(CSV_File_Name, index=False)

    # Clean up memory
    del rp_pivot_df, combined_df, final_df
    gc.collect()
    
    # Return the combined DataFrame as a Dask DataFrame
    return (CSV_File_Name, OutShp_File_Name, rivids, DEM_StrmShp_gdf)


if __name__ == "__main__":
    
    # WatershedID = '714'
    # NetCDF_RecurrenceInterval_Folder = '714_ReturnPeriods'
    # NetCDF_Historical_Folder = '714_HistoricFlows'
    # Outfile_file_path = 'GeoGLoWS_Flow_Data_' + str(WatershedID)  + 'JLG.csv'
    # combined_df = Create_ARC_Streamflow_Input(NetCDF_RecurrenceInterval_Folder, NetCDF_Historical_Folder, Outfile_file_path)

    StrmGDB = r"E:\2023_MultiModelFloodMapping\Global_Forecast\StrmShp\geoglows-v2-map-optimized.parquet"
    rivid_field = "LINKNO"
    #CSV_File_Name = r"E:\2023_MultiModelFloodMapping\Global_Forecast\HistoricFlow\GEOGLOWS_retrospective.csv"
    
    # Specify the layer you want to access
    layer_name = "geoglowsv2"
    # Read the layer from the geodatabase
    StrmShp_gdf = gpd.read_file(StrmGDB, layer=layer_name)
    
    # list all of the DEM tiles to create a flow file and stream shapefile for
    DEM_Dir = r"E:\2023_MultiModelFloodMapping\Global_Forecast\DEM"
    DEM_Tiles = os.listdir(DEM_Dir)
    
    # Directory where all the flow data will be stored
    FLOW_Dir = r"E:\2023_MultiModelFloodMapping\Global_Forecast\FLOW"
    StrmShp_Dir = r"E:\2023_MultiModelFloodMapping\Global_Forecast\StrmShp"
    
    
    for DEM_Tile in DEM_Tiles:
        if DEM_Tile.endswith(".tif"):
            DEM_Path = os.path.join(DEM_Dir, DEM_Tile)
            
            # Load the DEM file and get its CRS using gdal
            dem_dataset = gdal.Open(DEM_Path)
            dem_proj = dem_dataset.GetProjection()  # Get the projection as a WKT string
            dem_spatial_ref = osr.SpatialReference()
            dem_spatial_ref.ImportFromWkt(dem_proj)
            dem_crs = dem_spatial_ref.ExportToProj4()  # Export CRS to a Proj4 string (or other formats if needed)
	   
            # Check if the CRS of the shapefile matches the DEM's CRS
            if StrmShp_gdf.crs != dem_crs:
                # Reproject the shapefile to match the DEM's CRS
                StrmShp_gdf = StrmShp_gdf.to_crs(dem_crs)
        
            
            CSV_File_Name = os.path.join(FLOW_Dir,f"{DEM_Tile[:-4]}_Reanalysis.csv")
            OutShp_File_Name = os.path.join(StrmShp_Dir,f"{DEM_Tile[:-4]}_StrmShp.shp")
            
            Process_and_Write_Retrospective_Data_for_DEM_Tile(StrmShp_gdf, rivid_field, DEM_Path, CSV_File_Name, OutShp_File_Name)