# Key Concepts
The following concepts are useful for understanding how ARC works:

## ARC, under the hood
ARC is a Python tool that generates rating-curve-like hydraulic relationships for each stream cell in a raster domain. Given a DEM, a stream-ID raster, land-cover, and a flow table, ARC will do the following for each stream cell in your domain of interest:

1. Identify the slope and direction of the stream at that cell, and sample a cross-section perpendicular to the stream.
    1. If `Low_Spot_Range` is greater than 0, ARC will look left and right of the stream centerline for a lower spot to re-center the cross-section around. This is done to try to get a more accurate cross-section in cases where the stream raster does not perfectly identify the centerline of the stream.
    2. If `Degree_Manip` is greater than 0, ARC will rotate the cross-section in increments of `Degree_Interval` up to a maximum of `Degree_Manip` in either direction from perpendicular to the stream direction, and select the orientation which yields the smallest water surface top-width. This is done to try to get a more accurate cross-section in cases where the stream raster does not perfectly identify the direction of the stream.
2. Estimates bathymetry (optional)  
    1. If `Bathy_Use_Banks` is false, ARC will try to find the banks based on the DEM or land cover information, and cut out just that area between the banks to fit the given baseflow.
    2. If `Bathy_Use_Banks` is true, ARC assumes that the baseflow value is a channel-forming discharge (bankfull flow), and is free to recreate the bathymetry in the channel.
3. Attempts to adjust the slope to achieve a more realistic maximum water surface elevation. 
4. Computes water-surface elevation (WSE), depth, velocity, and top width across discharge increments
4. Writes one or more output datasets (VDT database, curve file, bathymetry raster, etc.) 

## Additional Details
- Raster padding: ARC will pad each input raster by a minimum of 1 cell around all edges, and more if `Gen_Dir_Dist` or `Gen_Slope_Dist` are greater than 1. This is done to avoid out-of-bounds errors when ARC looks around stream cells to calculate stream direction and slope. The output files will report the rows and columns of the original un-padded rasters, so the padding is effectively invisible to the user.
- Manning's n clipping. ARC will clip the Manning's n values. Values over 10 are clamped to 0.035, and values under or equal to 0 are clamped to 0.005. 
- Data types: to reduce memory usage and increase performance where possible, the land cover raster is read as an unsigned 8-bit integer, the stream raster is read as a signed 64-bit integer, and the DEM is read as a 32-bit float. If your input rasters are in a different format, they will be converted to these formats when they are read in. The output bathymetry raster is written as a 32-bit float, and the remaining output files are rounded to 3 or 8 decimal places, depending on the column.