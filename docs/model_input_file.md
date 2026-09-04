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
Flow_File_QMax: rp100_premium
StrmShp_File: /path/to/stream_network.gpkg
reach_id: COMID
downstream_reach_id: ToCOMID
drainage_area_field: TotDASqKm
coefficient_depth: 0.12
exponent_depth: 0.42
coefficient_width: 1.75
exponent_width: 0.55
Manual_Cross_Sections_File: /path/to/ARC_Manual_Cross_Sections.tsv
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
XS_Out_File: /path/to/Output_Cross_Sections.txt
Build_Representative_Cross_Section: True
Representative_Cross_Section_File: /path/to/Representative_Cross_Sections.csv

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
| `Flow_File` | --- | str | Path to the flow file containing streamflow data. In representative-cross-section mode this is optional unless baseflow-driven bathymetry is requested. |
| `Flow_File_ID` | --- | str | Column name in the flow file that contains unique identifiers for each reach or manual cross section. When `Manual_Cross_Sections_File` is used, this same field name must also exist in the manual cross-section table. In representative mode, supplying this field together with `Flow_File` and `Flow_File_BF` selects baseflow-driven bathymetry. |
| `Flow_File_BF` | --- | str | Column name containing baseflow or bankfull discharge values used by the downstream-to-upstream reach-network friction-slope solver. In representative mode, `Flow_File`, `Flow_File_ID`, and `Flow_File_BF` must be supplied together to select baseflow-driven bathymetry; otherwise ARC falls back to a complete drainage-area power-law configuration. |
| `Flow_File_QMax` | --- | str | Column name containing the maximum discharge used by standard rating-curve, VDT, and curve-file processing. It is not required or used when `Build_Representative_Cross_Section` is `True`. |
| `LU_Manning_n` | --- | str | Path to the text file containing Manning's n values for different land cover types. |
| `Manual_Cross_Sections_File` | --- | str | Optional tabular cross-section input file. When provided, ARC skips raster-based cross-section sampling and instead uses the supplied profiles, row/column arrays, and land-cover arrays for each `Flow_File_ID`. |
| `LU_Raster_SameRes` | --- | str | Path to the land cover raster file. |
| `Stream_File` | --- | str | Path to the stream raster file. |
| `StrmShp_File` | --- | str | Path to the stream shapefile or vector stream network. Required if `Stream_Slope_Method` is set to `end_points`, and also required when the optional drainage-area bathymetry parameters are used because ARC reads `drainage_area_field` from this dataset. |
| `reach_id` | --- | str | Field name in `StrmShp_File` containing the stream-network reach identifier used by the network-based bank-elevation smoother. When bathymetry output is requested, this parameter is required and ARC uses it instead of `Flow_File_ID` to build the directed reach graph. |
| `downstream_reach_id` | --- | str | Field name in `StrmShp_File` containing the immediate downstream reach identifier for each reach. When bathymetry output is requested, this parameter is required so ARC can build the directed reach network with `networkx` and estimate smoothed bank elevations for all sampled cross sections. Reach geometry lengths are converted to meters from the vector layer CRS, including ellipsoidal measurement for geographic flowlines. Headwater surfaces are initialized between their highest filtered raw bank and their minimum outlet bank. Outlet surfaces are initialized between the lowest incoming predecessor minimum and the outlet's own lowest filtered bank. A stream with neither neighbor uses its filtered maximum and minimum to infer flow direction and endpoint slope. Lower per-cell banks are subsequently applied as anchors. |

### Parameters
| Key | Default Value | Data Type | Description |
| --- | --- | --- | --- |
| `Degree_Manip` | 1.1 | float | The maximum angle, in degrees, that the cross-section may be rotated in either direction from perpendicular to the stream direction to find the orientation which yields the smallest water surface top-width. |
| `Degree_Interval` | 1.0 | float | The interval, in degrees, at which the cross-section is rotated to find the orientation which yields the smallest water surface top-width. |
| `Build_Representative_Cross_Section` | False | bool | When true, ARC builds a representative cross section for each positive reach ID in `Stream_File`. ARC samples and preprocesses each stream-cell cross section, applies bathymetry only when a bathymetry output path is configured, and recomputes area, wetted perimeter, velocity, discharge, top width, and hydraulic radius every 0.10 m above each local thalweg. Each reach is capped at 25 m, or at the last successful stage before any hydraulic value becomes non-finite. At each stage, ARC removes cross sections whose area or hydraulic radius falls outside two standard deviations of the reach mean, then stores reach-mean hydraulics and representative dimensions derived from monotonic staged mean top width and area. `Representative_Cross_Section_File` is required; `Flow_File_QMax` is not. |
| `Gen_Dir_Dist` | 10 | int | The number of DEM cells to look around (left, right, up, down) any given stream cell to use in calculating the direction of the stream. |
| `Gen_Slope_Dist` | 0 | int | The number of DEM cells to look around (left, right, up, down) any given stream cell to use in calculating the slope of the stream. |
| `Low_Spot_Range` | 0 | int | The number of DEM cells to look left and right of the stream centerline to find the lowest spot. If a spot with an elevation lower than the cell identified as the stream centerline by the stream raster is found, the cross-section is re-centered around that spot and resampled before ARC performs any reach-scale INFLECT, bank-finding, or bathymetry steps. |
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
| `Representative_Cross_Section_File` | --- | str | Path to the output representative cross-section CSV file. This file is only written when `Build_Representative_Cross_Section` is `True`. It stores one row per reach and successful 0.10 m stage, up to 25 m or the first stage that produces a non-finite area, wetted perimeter, velocity, discharge, or top width. Each row includes `Stream_Slope`, the reach-mean positive stream slope from the sampled cross sections. |
| `XS_Out_File` | --- | str | Path to the output cross-section export file. |

## Manual Cross-Section Input Schema

When `Manual_Cross_Sections_File` is provided, ARC expects the following columns:

| Column Name | Data Type | Description |
| --- | --- | --- |
| `Flow_File_ID` | Integer or string-like integer | The cross-section identifier used to join the manual table to the flow file. In the gap-crossing export this is `XS_ID`. |
| `Row` | Integer | DEM row of the center stream cell. |
| `Col` | Integer | DEM column of the center stream cell. |
| `Ordinate_Dist` | Float | Distance between adjacent cross-section ordinates, in meters. |
| `XS1_Profile` | JSON array string | Elevations from the center cell outward along side 1. |
| `XS2_Profile` | JSON array string | Elevations from the center cell outward along side 2. |
| `LC1_Profile` | JSON array string | Land-cover values for side 1, aligned to `XS1_Profile`. |
| `LC2_Profile` | JSON array string | Land-cover values for side 2, aligned to `XS2_Profile`. |
| `XS1_Row` | JSON array string | DEM row indices for the side-1 ordinates. |
| `XS1_Col` | JSON array string | DEM column indices for the side-1 ordinates. |
| `XS2_Row` | JSON array string | DEM row indices for the side-2 ordinates. |
| `XS2_Col` | JSON array string | DEM column indices for the side-2 ordinates. |

The gap-crossing seasonal export writes this schema directly. Additional metadata columns, such as `Route_ID`, `Source_Stream_ID`, or `XS_Angle`, are allowed and are preserved for reference but are not required by ARC.

### Bathymetry Information
| Key | Default Value | Data Type | Description |
| --- | --- | --- | --- |
| `Bathy_Trap_H` | 0.2 | float | A value from the range 0-1, representing how much of a trapezoidal bathymetry is sloping on one side. For example, a value of 0.2 indicates that, given a stream bathymetry with a width of 100 meters, the bathymetry is sloping on one side by 20 meters, for a total of 40 meters of sloping. |
| `Bathy_Use_Banks` | False | bool | When false, ARC assumes that the DEM is representative of the water surface (typically not in a flood stage, and often much less than bankfull), and bathymetry depth is estimated within the detected banks. When true, ARC treats detected bank elevations as bankfull controls and recreates the channel below the smoothed bank surface. In both cases ARC samples every cross section first, evaluates the land-cover/INFLECT/local-DEM bank hierarchy, filters reach-scale top-width outliers, and fills invalid bank indices with the reach-median top width. It then smooths bank elevation and grade along the network. For `Flow_File_BF`, the smoothed bank elevation becomes the vertical reference for a downstream-to-upstream friction-slope solve; positive-flow outlets use Manning normal depth, and a valid drainage-area depth takes precedence. Initial depths marked for application that are finite, positive, and below `25` m are then reduced to inclusive interquartile-filtered reach medians and constrained to stay equal or increase downstream before being burned into the sections. See [**Bathymetry**](bathymetry.md) for details and boundary fallbacks. |
| `drainage_area_field` | --- | str | Optional field name in `StrmShp_File` containing drainage area values. Required when either the depth or width power-law pair below is supplied. |
| `coefficient_depth` | --- | float | Optional coefficient in the power-law relationship used to estimate bankfull depth: `depth = coefficient_depth * drainage_area ^ exponent_depth`. Must be supplied with `exponent_depth`. |
| `exponent_depth` | --- | float | Optional exponent in the power-law relationship used to estimate bankfull depth. Must be supplied with `coefficient_depth`. |
| `coefficient_width` | --- | float | Optional coefficient in the power-law relationship used to estimate bankfull width: `width = coefficient_width * drainage_area ^ exponent_width`. Must be supplied with `exponent_width`. When a width prior is available, ARC uses it in the first bank-search gate to decide whether a sampled section is narrow enough to accept the explicit one-cell triangular bathymetry fallback, including representative runs that also provide complete flow-based bathymetry inputs. |
| `exponent_width` | --- | float | Optional exponent in the power-law relationship used to estimate bankfull width. Must be supplied with `coefficient_width`. Together with `coefficient_width`, this value controls the expected-width test that limits the one-cell bathymetry path to channels no wider than three raster cells. |
| `FindBanksBasedOnLandCover` | False | bool | If true, ARC will first attempt to find the banks based on the land cover, by identifying the cells that are classified as water. If that does not produce usable banks, ARC next tries a reach-average INFLECT maximum-curvature bank estimate before falling back to the older local DEM methods. All of those checks occur after ARC has sampled and cached the full set of stream-cell cross sections. Requires `LC_Water_Value` to be defined. See [**Bathymetry**](bathymetry.md) for more details. |
| `LC_Water_Value` | 80 | int | The value in the land cover raster that corresponds to water. Required if `FindBanksBasedOnLandCover` is true. Defaults to 80, which is the value for water in the ESA Land Cover dataset. |

`Flow_File_QMax` remains required for standard rating-curve runs even when `Flow_File_BF` is omitted. It is not required in representative-cross-section mode because that workflow builds its own 0.10 m hydraulic stages instead of the QMax-based VDT increments.
