# Model Input File (MIF)

ARC can be configured using a file called the Model Input File (MIF). The MIF contains all the necessary information for ARC to run, including paths to input datasets, output locations, and various parameters that control how ARC processes the data. The MIF is a simple tab-separated text OR YAML file with key-value pairs, where each key corresponds to a specific input or parameter that ARC uses. The following shows an example MIF in YAML format:

```yaml
#ARC_Inputs
DEM_File: /path/to/DEM.tif
Stream_File: /path/to/Stream_Raster.tif
LU_Raster_SameRes: /path/to/Land_Cover.tif
LU_Manning_n: /path/to/Mannings_n.txt
Flow_File: /path/to/Flow_File.csv
Flow_File_ID: COMID
Flow_File_BF: p_exceed_50
Flow_File_QMax: rp100_premium
Spatial_Units: deg
X_Section_Dist: 5000
Degree_Manip: 6.1
Degree_Interval: 1.5
Low_Spot_Range: 2
Str_Limit_Val: None
Gen_Dir_Dist: 10
Gen_Slope_Dist: 10
Stream_Slope_Method: local_average_corrected

#VDT_Output_File_and_CurveFile
VDT_Database_NumIterations: 30
Print_VDT_Database: /path/to/Output_VDT_Database.csv
Reach_Average_Curve_File: False

#Bathymetry_Information
Bathy_Trap_H: 0.2
Bathy_Use_Banks: False
FindBanksBasedOnLandCover: True
AROutBATHY: /path/to/Output_ARC_Bathy.tif
BATHY_Out_File: /path/to/Output_Bathy.tif
```

## Argument Descriptions

### Input Files
| Key | Default Value | Data Type | Description |
| --- | --- | --- | --- |
| `DEM_File` | --- | str | Path to the Digital Elevation Model (DEM) raster file. All subsequent raster files are assumed to have the same resolution, extent, and projection. |
| `Flow_File` | --- | str | Path to the flow file containing streamflow data. |
| `Flow_File_ID` | --- | str | Column name in the flow file that contains unique identifiers for each reach. |
| `Flow_File_BF` | --- | str | Column name in the flow file that contains baseflow values. |
| `Flow_File_QMax` | --- | str | Column name in the flow file that contains maximum discharge values to simulate. |
| `LU_Manning_n` | --- | str | Path to the text file containing Manning's n values for different land cover types. |
| `LU_Raster_SameRes` | --- | str | Path to the land cover raster file. |
| `Stream_File` | --- | str | Path to the stream raster file. |
| `StrmShp_File` | --- | str | Path to the stream shapefile. Required if `Stream_Slope_Method` is set to 'end_points' or 'reach_average'. |

### Parameters
| Key | Default Value | Data Type | Description |
| --- | --- | --- | --- |
| `Degree_Manip` | 1.1 | float | The maximum angle, in degrees, that the cross-section may be rotated in either direction from perpendicular to the stream direction to find the orientation which yields the smallest water surface top-width. |
| `Degree_Interval` | 1.0 | float | The interval, in degrees, at which the cross-section is rotated to find the orientation which yields the smallest water surface top-width. |
| `Gen_Dir_Dist` | 10 | int | The number of DEM cells to look around (left, right, up, down) any given stream cell to use in calculating the direction of the stream. |
| `Gen_Slope_Dist` | 0 | int | The number of DEM cells to look around (left, right, up, down) any given stream cell to use in calculating the slope of the stream. |
| `Low_Spot_Range` | 0 | int | The number of DEM cells to look left and right of the stream centerline to find the lowest spot. If a spot with an elevation lower than the cell identified as the stream centerline by the stream raster is found, the cross-section is re-centered around that spot. |
| `Reach_Average_Curve_File` | --- | bool | Flag indicating whether to average the values of the curve file across reaches. |
| `Stream_Slope_Method` | local_average_corrected | str | The method to use for calculating stream slope. Options include 'local_average_corrected', 'local_average', 'local_average_corrected', 'reach_average', and 'end_points'. See [**Stream Slope Methods**](stream_slope_methods.md) for more details. |
| `VDT_Database_NumIterations` | 15 | int | The number of iterations to run when creating the VDT database. |
| `X_Section_Dist` | 5000 | float | Width of the cross-section for each stream cell in meters. |

### Output Files
See [**Outputs**](outputs.md) documentation for details on the output datasets that ARC can generate.

| Key | Default Value | Data Type | Description |
| --- | --- | --- | --- |
| `AROutBATHY` | --- | str | Path to the output bathymetry raster file. |
| `BATHY_Out_File` | --- | str | The same as `AROutBATHY`, which takes precedence. |
| `Print_AP_Database` | --- | str | Path to the output Area/Perimeter (AP) database file. |
| `Print_Curve_File` | --- | str | Path to the output curve file. |
| `Print_VDT_Database` | --- | str | Path to the output VDT database file. |
| `XS_Out_File` | --- | str | Path to the output cross-section export file. |

### Bathymetry Information
| Key | Default Value | Data Type | Description |
| --- | --- | --- | --- |
| `Bathy_Trap_H` | 0.2 | float | A value from the range 0-1, representing how much of a trapezoidal bathymetry is sloping on one side. For example, a value of 0.2 indicates that, given a stream bathymetry with a width of 100 meters, the bathymetry is sloping on one side by 20 meters, for a total of 40 meters of sloping. |
| `Bathy_Use_Banks` | False | bool | When false, ARC assumes that the DEM is representative of the water surface (typically not in a flood stage, and often much less than bankfull), and that the baseflow value is less than bankfull. It will try to find the banks based on the DEM or land cover information, and cut out just that area between the banks to fit the given baseflow. When true, ARC assumes that the baseflow value is a channel-forming discharge (bankfull flow), and is free to recreate the bathymetry in the channel. See [**Bathymetry**](bathymetry.md) for more details. |
| `FindBanksBasedOnLandCover` | False | bool | If true, ARC will first attempt to find the banks based on the land cover, by identifying the cells that are classified as water. Requires `LC_Water_Value` to be defined. See [**Bathymetry**](bathymetry.md) for more details. |
| `LC_Water_Value` | 80 | int | The value in the land cover raster that corresponds to water. Required if `FindBanksBasedOnLandCover` is true. Defaults to 80, which is the value for water in the ESA Land Cover dataset. |
