# Model Input File (MIF)

ARC can be configured using a file called the Model Input File (MIF). The MIF contains all the necessary information for ARC to run, including paths to input datasets, output locations, and various parameters that control how ARC processes the data. The MIF is a simple tab-separated text OR YAML file with key-value pairs, where each key corresponds to a specific input or parameter that ARC uses. The following shows an example MIF in YAML format:

```yaml
#ARC_Inputs
DEM_File: /path/to/DEM.tif
Stream_File: /path/to/Stream_Raster.tif
LU_Raster_SameRes: /path/to/Land_Cover.tif
LU_Manning_n: /path/to/Mannings_n.txt
slope_adjustment_factor: 1.0
k_decay: 6.0
shallow_factor: 2.0
deep_factor: 1.0
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

### Depth-dependent Manning's roughness

| Parameter | Default | Valid values |
| --- | --- | --- |
| `k_decay` | `6.0` | Finite and positive (inverse meters). |
| `shallow_factor` | `2.0` | Finite and >= 1. |
| `deep_factor` | `1.0` | Finite and in (0, 1]. |

```text
n_adjusted = n * (deep_factor + (shallow_factor - deep_factor)
                  * exp(-k_decay * max(depth_m, 0)))
```

The factors multiply the baseline Manning's n at zero wet depth and in the
deep-water limit, respectively. Both factors equal to 1 disable the adjustment.
Dry depths are clamped to zero only for roughness scaling; signed depths remain
available for geometry and bank-intersection calculations.

The parameters reach the geometry helper through WSE searches, VDT staging,
and representative-cross-section exports. Existing rounding is unchanged.

Python API:

```python
Arc(args={**inputs, "k_decay": 6.0, "shallow_factor": 2.0, "deep_factor": 1.0}).run()
```

YAML MIF:

```yaml
slope_adjustment_factor: 1.0
k_decay: 6.0
shallow_factor: 2.0
deep_factor: 1.0
```

Text MIFs use the same keys separated from their values by a tab.
`alpha_low`, `alpha_boost`, and `f_min` are no longer accepted. To reproduce
the previous shallow-boost formula, use `shallow_factor = 1 + old_alpha_low`
and `deep_factor = 1`. The new defaults reproduce the old default boost.

### Slope adjustment

`slope_adjustment_factor` defaults to `1.0` and must be finite and positive.
It multiplies `sqrt_slope` in the discharge calculation:

```text
adjusted_sqrt_slope = sqrt_slope * slope_adjustment_factor
```

This is equivalent to multiplying the underlying slope by the factor squared.
The multiplier is applied once, before discharge rounding, in both WSE searches
and staged hydraulics (including representative exports). Stored slope values
are unchanged. At fixed geometry and roughness, a factor of 2 doubles discharge
and velocity; the default factor preserves existing results.

Pass `"slope_adjustment_factor": 1.0` in `Arc(args=...)`, or use the same key
in a YAML or tab-delimited input file. The analysis tuning script exposes
`--slope-adjustment-factor-range 0.5 2.0` and saves the optimized factor with
the depth-roughness parameters.

### Channel and floodplain conveyance

When valid inferred bank stations are available, ARC divides each section into the main channel and the two overbanks. Each subdivision uses its own area, bed wetted perimeter, and depth-adjusted composite roughness. ARC sums `Q_i = A_i * (A_i / P_i)**(2/3) / n_i * sqrt(slope) * slope_adjustment_factor` to obtain total discharge. The two channel halves form one hydraulic subsection; vertical interfaces at the banks do not add wetted perimeter. Floodplain conveyance starts as the water surface overtops the sampled bank. No bankfull discharge is required.

WSE searches, flood increments, and representative-section calculations use this same subdivision. Per-cell cross-section exports include `Bank_Index1` and `Bank_Index2`. Missing or invalid banks retain the original whole-section calculation. Integration follows connected wet profile segments and stops at the first dry barrier. The method assumes a common friction slope and does not explicitly model channel/floodplain momentum exchange. VDT velocity remains total discharge divided by total wetted area. In representative-section exports, `Mean_Velocity` and `Representative_Velocity` are the filtered arithmetic mean of per-section main-channel discharge divided by main-channel area. The existing two-standard-deviation filtering is applied using this channel velocity, total area, and total width; discharge and geometry remain whole-section quantities. Sections without valid banks use whole-section velocity as a fallback. The reported composite roughness for a subdivided section is an equivalent diagnostic value reproducing the summed conveyance, not a roughness used to recompute discharge. Existing numeric output precision is retained.

### Initial channel roughness

Before bank-elevation smoothing in `_finalize_cross_section_records`, ARC sets the baseline Manning's n at raster cells from each section's center through both valid bank indices (inclusive) to the water-class value from `LU_Manning_n`, selected using `LC_Water_Value` (default 80). Cells outside the union of these in-bank samples keep their existing roughness. This updates the shared roughness raster, so subsequent section resampling and representative output use the assignment. Land-cover classes are unchanged, and depth-dependent roughness adjustments still apply afterward. Sections with missing or invalid banks are skipped. A missing or nonfinite water-class table value raises an error; numeric bounds corrections match the existing Manning-table loader.
