# Outputs
ARC can write several outputs depending on which output paths are provided in the MIF. If an output path is blank, ARC skips generating that output.

## VDT database
The VDT output is a per-stream-cell table of hydraulic variables by increment (e.g., discharge, velocity, top width, WSE). It is commonly used downstream for inundation mapping workflows. It may be saved as a CSV or as a parquet. The following table details the columns in the VDT database.

| Column Name | Data Type | Description |
| --- | --- | --- |
| COMID | String | The unique identifier for the simulated row. In standard raster-sampled runs this is the stream/reach ID. In manual-cross-section runs it is the manual cross-section ID from `Flow_File_ID` (for example `XS_ID`). |
| Row | Integer | The row in the DEM where the stream cell is located. |
| Col | Integer | The column in the DEM where the stream cell is located. |
| Elev | Float | The elevation of the input DEM (before bathymetry estimation) where the stream cell is located. |
| QBaseflow | Float | The baseflow value for the stream cell, taken from the flow file. |
| Slope | Float | The slope of the stream at the stream cell. |
| XS_Angle | Float | The angle of the cross-section sampled for the stream cell, in degrees from east (positive x-axis), clockwise. |
| BaseElev | Float | The elevation of the channel bottom after bathymetry has been estimated where the stream cell is located. |
| q_* | Float | The discharge increment for the row. For example, q_1 is the first discharge increment, q_2 is the second discharge increment, etc. |
| v_* | Float | The velocity for the discharge increment. |
| t_* | Float | The top width for the discharge increment. |
| wse_* | Float | The water surface elevation for the discharge increment. |

All floating point columns are rounded to 3 decimal places, except for slope, which is rounded to 8.

## Curve file
The following table details the columns in the curve file, which may be saved as a CSV or as a parquet. 

| Column Name | Data Type | Description |
| --- | --- | --- |
| COMID | String | The unique identifier for the simulated row. In standard raster-sampled runs this is the stream/reach ID. In manual-cross-section runs it is the manual cross-section ID from `Flow_File_ID`. |
| Row | Integer | The row in the DEM where the stream cell is located. |
| Col | Integer | The column in the DEM where the stream cell is located.  |
| BaseElev | Float | The elevation of the channel bottom after bathymetry has been estimated where the stream cell is located.  |
| DEM_Elev | Float | The elevation of the input DEM (before bathymetry estimation) where the stream cell is located. |
| QMax | Float | The maximum flow used to generate the rating curves for the stream cell. This should be nearly equal to the value for the stream reach in the - - max_flow_field. The value will not likely be identical to the value in the max_flow_field as ARC iteratively solves Manning's equation and does so by - iteratively increasing the streamflow. |
| Slope | Float | The slope of the stream at the stream cell. |
| XS_Angle | Float | The angle of the cross-section sampled for the stream cell, in degrees from east (positive x-axis), clockwise. |
| depth_a | Float | In the formula $depth = a * streamflow^b$, this value represent $a$. The value estimates a water depth at the stream channel thalweg for the stream cell if `Reach_Average_Curve_File` is `False`, otherwise it represents the average value for the entire reach. |
| depth_b | Float | In the formula $depth = a * streamflow^b$, this value represent $b$. The value estimates a water depth at the stream channel thalweg for the stream cell if `Reach_Average_Curve_File` is `False`, otherwise it represents the average value for the entire reach. |
| tw_a | Float | In the formula $top-width = a * streamflow^b$, this value represent $a$. The value estimates a top-width of the flow for the stream cell if `Reach_Average_Curve_File` is `False`, otherwise it represents the average value for the entire reach. |
| tw_b | Float | In the formula $top-width = a * streamflow^b$, this value represent $b$. The value estimates a top-width of the flow for the stream cell if `Reach_Average_Curve_File` is `False`, otherwise it represents the average value for the entire reach. |
| vel_a | Float | In the formula $velocity = a * streamflow^b$, this value represent $a$. The value estimates a cross-section average velocity of the flow for the stream cell if `Reach_Average_Curve_File` is `False`, otherwise it represents the average value for the entire reach. |
| vel_b | Float | In the formula $velocity = a * streamflow^b$, this value represent $b$. The value estimates a cross-section average velocity of the flow for the stream cell if `Reach_Average_Curve_File` is `False`, otherwise it represents the average value for the entire reach. |

All floating point columns are rounded to 3 decimal places, except for slope, which is rounded to 8.

## Area/Perimeter (AP) database
The AP database stores discharge with derived cross-sectional area (from `a = q / v`) and wetted perimeter by increment. This is useful for workflows that need geometry rather than curve coefficients.

The following table details the columns in the AP database, which may be saved as a CSV or as a parquet.

| Column Name | Data Type | Description |
| --- | --- | --- |
| COMID | String | The unique identifier for the simulated row. In standard raster-sampled runs this is the stream/reach ID. In manual-cross-section runs it is the manual cross-section ID from `Flow_File_ID`. |
| Row | Integer | The row in the DEM where the stream cell is located. |
| Col | Integer | The column in the DEM where the stream cell is located.  |
| q_* | Float | The discharge increment for the row. For example, q_1 is the first discharge increment, q_2 is the second discharge increment, etc. |
| a_* | Float | The cross-sectional area for the discharge increment, derived from `a = q / v`. |
| p_* | Float | The wetted perimeter for the discharge increment, derived from the cross-sectional geometry. |

All floating point columns are rounded to 3 decimal places, except for slope, which is rounded to 8.

## Bathymetry raster
If bathymetry outputs are enabled, ARC writes a raster with estimated channel-bed elevations (based on the configured bathymetry method). The default value for cells not considered as bathymetry is NaN.

## Cross section export
If the cross section output is enabled, ARC writes a tab-delimited text file containing sampled cross-section profiles and the associated metadata used during computation.

If `Manual_Cross_Sections_File` is provided, ARC uses the supplied cross-section table as input instead of sampling a new cross section from the raster stack. The exported `XS_Out_File`, if requested, still reflects the cross sections ARC actually used during the run.

The following table details the columns in the cross section export file:

| Column Name | Data Type | Description |
| --- | --- | --- |
| COMID | String | The unique identifier for the simulated row. In standard raster-sampled runs this is the stream/reach ID. In manual-cross-section runs it is the manual cross-section ID from `Flow_File_ID`. |
| Row | Integer | The row in the DEM where the stream cell is located. |
| Col | Integer | The column in the DEM where the stream cell is located. |
| XS1_Profile | String | A string representation of one half of the cross-section profile. It is a list of elevation values, rounded to 6 decimal places. |
| Ordinate_Dist | Float | The distance between each elevation value in the cross-section profile, in meters. |
| Manning_N_Raster1 | String | A string representation of the Manning's n values for the land cover types corresponding to each elevation value in the cross-section profile. It is a list of values, rounded to 6 decimal places. |
| XS2_Profile | String | A string representation of the other half of the cross-section profile. It is a list of elevation values, rounded to 6 decimal places. |
| Manning_N_Raster2 | String | A string representation of the Manning's n values for the land cover types corresponding to each elevation value in the cross-section profile. It is a list of values, rounded to 6 decimal places. |
| r1 | Integer | The row representing the farthest point in the first side of the cross-section. |
| c1 | Integer | The column representing the farthest point in the first side of the cross-section. |
| r2 | Integer | The row representing the farthest point in the second side of the cross-section. |
| c2 | Integer | The column representing the farthest point in the second side of the cross-section. |
