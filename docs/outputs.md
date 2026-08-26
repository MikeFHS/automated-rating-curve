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
| QBaseflow | Float | The bathymetry discharge value for the stream cell, taken from `Flow_File_BF` when that field is supplied. When bathymetry is estimated from drainage-area power laws instead, this column remains `0.0`. |
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
If bathymetry outputs are enabled, ARC writes a raster with estimated channel-bed elevations (based on the configured bathymetry method). The default value for cells not considered as bathymetry is NaN. In the staged smoothed-bank workflow, every stream cell that has a sampled cross section receives a reach-smoothed bank-elevation attempt before ARC decides whether bathymetry can be written. Before that elevation smoothing step, ARC also uses the reach-median top width to replace width outliers and to assign bank indices to sampled sections whose local bank search remained invalid. If that smoothed section still reduces to a one-cell-wide channel, ARC uses the single-cell triangular fallback.

The depth written to the raster is a reach-level value, not necessarily the initial depth calculated at that cell. ARC first obtains a network friction-slope or drainage-area power-law depth, filters each reach's values marked for application that are finite, positive, and below 25 m to the inclusive 25th-75th percentile interval, and assigns the retained median to the reach. Positive-flow network outlets use Manning normal depth, and failed interior friction-slope solves fall back to normal depth before the final 0.5 m fallback. ARC then raises downstream reach medians where necessary so depth stays equal or increases downstream. The constrained depth is finally subtracted or reconstructed according to the selected bank/WSE bathymetry method.


## Cross section export
If the cross section output is enabled, ARC writes a tab-delimited text file containing the cross-section profiles and the associated metadata used during computation.

ARC now builds these sections in stages. It first samples every stream-cell cross section, applies the low-spot recentering and any optional angle-based resampling, evaluates reach-scale INFLECT and the remaining bank-finding hierarchy across the cached sections, and then applies bathymetry if requested. The exported `XS_Out_File`, if requested, reflects the final section ARC actually used during the run.

If `Manual_Cross_Sections_File` is provided, ARC uses the supplied cross-section table as input instead of sampling a new cross section from the raster stack. The same staged bank-search and optional bathymetry workflow still applies to those supplied sections.

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
| Inflect_D2W_Dy2 | String | A string representation of the raw sampled cross section's INFLECT `d2W/dy^2` curve. This curve is computed before bathymetry is applied and is the diagnostic signal ARC uses to build reach-average INFLECT bank and terrace indices. |

## Representative cross section export
If `Build_Representative_Cross_Section` is `True`, ARC writes a comma-separated CSV file that summarizes the sampled cross sections belonging to each positive reach ID in `Stream_File`. Representative grouping does not depend on `Flow_File_ID` or `Flow_File_QMax`.

ARC first samples and caches every stream-cell cross section, including its final profile, Manning's *n* arrays, slope, and thalweg. If `AROutBATHY` or `BATHY_Out_File` is configured, ARC completes bank finding and bathymetry preprocessing before building the representative output. Complete `Flow_File`, `Flow_File_ID`, and `Flow_File_BF` inputs select baseflow-driven bathymetry in representative mode. Otherwise, ARC uses the complete drainage-area power-law configuration when one is supplied. If neither bathymetry output path is configured, bathymetry estimation and excavation are bypassed and the representative hydraulics use the unexcavated sampled profiles.

For each reach, ARC evaluates every contributing cross section at 0.10 m increments above its local thalweg, up to a maximum depth of 25 m. At every stage it recomputes area, wetted perimeter, velocity, discharge, and top width with Manning's equation. A finite result with positive area and top width contributes to the stage medians; sections with unusable finite geometry are skipped for that stage. If any calculation produces a non-finite area, wetted perimeter, velocity, discharge, or top width, ARC stops staging that reach immediately and retains only the previously successful rows. The representative export stores the reach-median positive stream slope and the median discharge, depth, velocity, top width, area, and WSE from the contributing sections at each retained stage.

The representative cross-section dimensions are then derived from the staged median top width and staged median area. Starting from a thalweg stage with zero width and zero area, ARC solves a trapezoidal area equation between successive stages to recover each representative depth increment and cumulative representative depth. Those stages are finally written as symmetric left/right stations around the reach-median thalweg elevation.

Because the staged medians are computed independently at each 0.10 m depth increment, small non-monotonic artifacts can occur. ARC therefore applies a cumulative-maximum adjustment to representative area and top width before deriving the representative dimensions so the exported envelope remains physically ordered.

The following table details the columns in the representative cross-section export file:

| Column Name | Data Type | Description |
| --- | --- | --- |
| COMID | Integer | The positive reach identifier from `Stream_File`. |
| Cross_Section_Count | Integer | Number of sampled stream cells combined for the reach. |
| Hydraulic_Sample_Count | Integer | Number of sampled stream cells that contributed valid hydraulic values at this 0.10 m stage. |
| Depth_Stage_Index | Integer | Index of the 0.10 m evaluation stage, starting at `1` for `0.10` m above each local thalweg. |
| Depth_Stage_Meters | Float | Depth stage above the local thalweg used to evaluate the sampled cross sections. |
| Stream_Slope | Float | Reach-median positive stream slope from the sampled stream-cell cross sections. This is the slope used by ARC when recomputing representative hydraulics for the reach. |
| Reach_Inflect_Terrace_Depth | Float | Legacy column name containing the last successful evaluation depth retained for the reach. The value is capped at 25 m and may be lower when a non-finite hydraulic result terminates staging. It is no longer derived from the reach-average INFLECT curve. |
| Representative_Thalweg_Elevation | Float | Reach-median thalweg elevation across the contributing sampled cross sections. |
| Median_Discharge | Float | Median Manning discharge across the valid stream cells in the reach for this 0.10 m stage. |
| Median_Depth | Float | Median hydraulic depth across the valid stream cells in the reach for this stage. In the current workflow this matches `Depth_Stage_Meters` because ARC evaluates all cross sections at a common depth increment above their local thalweg. |
| Median_Velocity | Float | Median hydraulic velocity across the valid stream cells in the reach for this stage. |
| Median_Top_Width | Float | Median hydraulic top width across the valid stream cells in the reach for this stage. |
| Median_Cross_Sectional_Area | Float | Median hydraulic cross-sectional area across the valid stream cells in the reach for this stage. |
| Median_WSE | Float | Median water-surface elevation across the valid stream cells in the reach for this stage. |
| Representative_Cross_Sectional_Area | Float | Cross-sectional area used in the exported representative geometry after enforcing non-decreasing staged area by depth increment. |
| Representative_Depth_Increment | Float | Depth added between the previous representative stage and this stage when solving the trapezoidal width-area relationship for the representative section. |
| Representative_Depth | Float | Cumulative depth used in the exported representative geometry after deriving the stage increments from representative area and representative top width. |
| Representative_Top_Width | Float | Top width used in the exported representative geometry after enforcing non-decreasing staged width by depth increment. |
| Representative_Stage_Elevation | Float | Elevation used for the representative stage point, computed as `Representative_Thalweg_Elevation + Representative_Depth`. |
| Representative_Left_Station | Float | Left station for the representative stage point, computed as `-Representative_Top_Width / 2`. |
| Representative_Right_Station | Float | Right station for the representative stage point, computed as `Representative_Top_Width / 2`. |
