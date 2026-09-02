# Key Concepts
The following concepts are useful for understanding how ARC works:

## ARC, under the hood
ARC is a Python tool that generates rating-curve-like hydraulic relationships for each stream cell in a raster domain. Given a DEM, a stream-ID raster, land-cover, and a flow table, ARC now runs the cross-section workflow in ordered stages:

1. Sample every stream-cell cross section.
    1. ARC identifies the slope and direction of the stream at each cell and samples a cross-section perpendicular to the stream.
    2. If `Low_Spot_Range` is greater than 0, ARC looks left and right of the stream centerline for a lower spot, re-centers the cross-section on that low point, and resamples the section.
    3. If `Degree_Manip` is greater than 0, ARC rotates the cross-section in increments of `Degree_Interval` up to a maximum of `Degree_Manip` in either direction from perpendicular to the stream direction, and selects the orientation that yields the smallest water-surface top width.
2. Evaluate reach-scale INFLECT and bank-finding across the cached sections.
    1. ARC computes an INFLECT curve for every sampled cross section.
    2. ARC averages those curves by reach to define reach-scale INFLECT bank and terrace indices.
    3. ARC then applies the bank hierarchy for every cached section: land cover first if enabled, then reach-scale INFLECT, then the local DEM fallback methods.
    4. Once the local bank indices exist, ARC orders the cross sections from upstream to downstream within each reach, smooths the bank height above the thalweg, reconstructs a bank-elevation line that follows the exact reach thalweg slope, and converts that smoothed elevation back into a local top width for each section.
3. Estimate bathymetry (optional).
    1. If `Bathy_Use_Banks` is false, ARC uses the precomputed bank locations with the WSE-style bathymetry workflow. `Flow_File_BF` drives a downstream-to-upstream reach-network friction-slope solve, while a valid drainage-area power-law depth takes precedence.
    2. If `Bathy_Use_Banks` is true, ARC uses the smoothed bank elevation as the network energy reference and treats the baseflow as a channel-forming discharge. Positive-flow outlets are initialized with Manning normal depth; the optional power-law depth can instead provide the initial bankfull depth.
    3. In either mode, ARC filters initial depths marked for application that are finite, positive, and below 25 m to the inclusive 25th-75th percentile interval for each reach, assigns the retained median to every section on that reach, and raises downstream reach medians where needed so depth does not decrease downstream.
4. Attempt to adjust the slope to achieve a more realistic maximum water surface elevation.
5. Compute water-surface elevation (WSE), depth, velocity, and top width across discharge increments.
6. Write one or more output datasets (VDT database, curve file, bathymetry raster, representative cross sections, etc.).

## Additional Details
- Raster padding: ARC will pad each input raster by a minimum of 1 cell around all edges, and more if `Gen_Dir_Dist` or `Gen_Slope_Dist` are greater than 1. This is done to avoid out-of-bounds errors when ARC looks around stream cells to calculate stream direction and slope. The output files will report the rows and columns of the original un-padded rasters, so the padding is effectively invisible to the user.
- Manning's n clipping. ARC will clip the Manning's n values. Values over 10 are clamped to 0.035, and values under or equal to 0 are clamped to 0.005. 
- Data types: to reduce memory usage and increase performance where possible, the land cover raster is read as an unsigned 8-bit integer, the stream raster is read as a signed 64-bit integer, and the DEM is read as a 32-bit float. If your input rasters are in a different format, they will be converted to these formats when they are read in. The output bathymetry raster is written as a 32-bit float, and the remaining output files are rounded to 3 or 8 decimal places, depending on the column.
