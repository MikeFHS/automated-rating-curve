# Model Input File (MIF)

The model input file tells ARC where its inputs are, how to build the cross sections, and which outputs to write. It
can be YAML:

```yaml
# Inputs
DEM_File: /path/to/dem.tif
Stream_File: /path/to/streams.tif
LU_Raster_SameRes: /path/to/land_cover.tif
LU_Manning_n: /path/to/mannings_n.txt
Flow_File: /path/to/flows.csv
Flow_File_ID: LINKNO
Flow_File_BF: qbase
Flow_File_QMax: qmax
StrmShp_File: /path/to/streams.gpkg
reach_id: LINKNO
downstream_reach_id: DSLINKNO

# Slopes and hydraulics
Stream_Slope_Method: local_average_corrected
Standard_Step: True

# Bathymetry
Bathy_Use_Banks: False
BathyWaterMask: /path/to/water_mask.tif
BATHY_Out_File: /path/to/bathymetry.tif

# Outputs
VDT_Database_NumIterations: 30
Print_VDT_Database: /path/to/vdt.csv
Print_Curve_File: /path/to/curves.csv
XS_Out_File: /path/to/cross_sections.txt
```

or text, with a key, a tab (or spaces) and its value on each line. Lines starting with `#` are comments.

```text
DEM_File	/path/to/dem.tif
Stream_File	/path/to/streams.tif
```

Keys aren't case-sensitive. Keys that ARC does not know stops the run with an error that suggests the closest key, so a typo
can't go unnoticed. Leave an output out to skip it.

## What a run needs

| To make | Give |
| --- | --- |
| Rating curves | `DEM_File`, `Stream_File`, `LU_Raster_SameRes`, `LU_Manning_n`, `Flow_File` with `Flow_File_ID` and `Flow_File_QMax`, and an output such as `Print_VDT_Database` or `Print_Curve_File`. |
| Bathymetry too | `BATHY_Out_File`; `Flow_File_BF`, or `drainage_area_field` with `coefficient_depth` and `exponent_depth`; and `StrmShp_File` with `reach_id` and `downstream_reach_id`. |
| Representative cross sections only | The rasters and the Manning's n table, `Build_Representative_Cross_Section` and `Representative_Cross_Section_File`, and for bathymetry as above (the flow file only for the baseflow). |
| `end_points` slopes | `StrmShp_File`, with a field named as `Flow_File_ID` is (or as `reach_id` is, in a run that builds only representative cross sections) holding each line's reach ID. |
| `Standard_Step` | `StrmShp_File` with `reach_id` and `downstream_reach_id`. |

## Recommended settings

The defaults keep older input files running as they always have, and some of them aren't what we'd choose for a new
run. For a new run, with a stream network and a flow file, we recommend the settings below. The flood-map numbers
come from 68 sites where the USGS mapped real floods (406 maps), scored by the Matthews correlation coefficient (MCC),
a higher MCC being a better map: differences of a few thousandths are within the noise between sites.

| Setting | Recommended | Why |
| --- | --- | --- |
| `StrmShp_File`, `reach_id`, `downstream_reach_id` | Give them | The bathymetry and `Standard_Step` need the stream network, and with it a reach whose slope can't be measured takes its neighbours'. |
| `Standard_Step` | True | It ties each cross section's water to the water below it, as a river's is, instead of putting each at its own normal depth. The water surface's swings from one cell to the next fell from 0.28 m to 0.05 m, and the flood maps' mean MCC rose by 0.01 to 0.02. See [Standard Step](standard_step.md). |
| `Standard_Step_Eddy_Length` | 300 (the default) | 0 charges every step the change in velocity head from the cross section below, which on cross sections a cell apart adds tenths of a metre of water for nothing (0.46 m on a river whose width varies by a fifth from cell to cell). 0 did raise the flood maps' mean MCC by 0.006, but by raising the water everywhere, which maps that were mostly too small reward whatever raises it. |
| `BathyWaterMask` | Give one if you have it | Its flat water gives the water level beside a stream cell burned below it, and its edge stops the DEM's water running across a floodplain at the water's level. See [Find the banks](bathymetry.md#1-find-the-banks). |
| `FindBanksBasedOnLandCover` | True, without a water mask | Its water does the water mask's job; its widths were a median 1.00 times the water mask's. |
| `Bathy_Use_Banks` with `Flow_File_BF` | **Match the flow to the level** | `Flow_File_BF` must be the flow that fills the channel to the level `Bathy_Use_Banks` names: a low flow for the DEM's water surface (False), the flow at the bank tops for the bank elevation (True), with a water mask or without. A mismatched pair carves the bed metres too deep or too shallow. See [Water surface or banks](bathymetry.md#water-surface-or-banks). |
| `drainage_area_field`, `coefficient_width`, `exponent_width` | A width power law, such as nencarta's: 2.44 and 0.34, with drainage areas in km² | It sets the width of the channels too narrow for the water mask, the land cover or the DEM to show, which would otherwise take a few spurious wide banks' median: a creek in Salt Lake City was carved 262 m wide without it. The flood maps' mean MCC rose by 0.007. |
| `Bathy_Smooth_Banks` | True (the default) | Turned off, the flood maps' mean MCC fell by 0.043. |
| `Stream_Slope_Method` | `local_average_corrected` (the default), or `local_fit` | See [Which method to use](stream_slope_methods.md#which-method-to-use). |
| `Slope_Low_Percentile`, `Slope_High_Percentile` | 5 and 95 (the defaults) | The old defaults, 25 and 75, cap a reach's slopes at its upper quartile; with the standard step, any upper percentile of 75 or less lowered the flood maps' mean MCC by about 0.006, while 0 and 100, 1 and 99, 5 and 95 or 10 and 90 all scored within 0.001 of each other. |
| `Gen_Slope_Dist` | 5 to 10 | 30 and 100 cells scored 0.001 and 0.002 lower than 5. |
| `Gen_Dir_Dist` | 10 (the default) | On a 30 m DEM, 10 rated the most cells (96.6%) and turned the cross sections steadily from cell to cell; 2 made 7% of neighbouring cross sections cross each other, and above about 40 they turned off the channel at bends (15° at 122) and, along a canal, ran down it. |
| `Degree_Manip`, `Degree_Interval` | No search (`Degree_Manip` under twice `Degree_Interval`, as the defaults are) | Turning each cross section to where the valley is narrowest made neighbouring cross sections cross more than twice as often (41% against 19%) and left the flood maps within 0.003. |
| `Low_Spot_Range` | 0 if the stream raster follows the DEM's channel; 1 or 2 if it sits beside it | Moving a cross section onto lower ground beside the stream cell helps only where the stream lines miss the channel. |
| `Depth_Varying_N` | False (the default), unless you calibrate it | Measured n changes with depth both ways: of 53 USGS gauges, it fell as the water deepened at 20, rose at 22 and was unclear at 11, and ARC's form can only fall. Pooled, its default factors fitted the gauges slightly worse than a constant n. The bathymetry carves each channel for the n the rating curves use, so with it channels are carved about a tenth deeper. If you do use it, keep `deep_factor` 1 (it only scales n, as `slope_adjustment_factor` does) and tune `shallow_factor` and `k_decay` together: what matters is (`shallow_factor` − 1) / `k_decay`, a depth in metres around which n changes, about the stream's own depth. |
| `VDT_Database_NumIterations` | 30 | Twice the default's detail in each rating curve, for 5 to 13% more run time. |
| `Subcell_Channels` | False (the default), or True where the narrowest channels are a few cells across and their water level matters | On a synthetic constriction it cut the backwater's error from 0.30 m to 0.004 m on 10 m cells and from 1.05 m to 0.07 m on 20 m. On the 65 FIM sites (30 m DEMs) it changed the flood maps' mean MCC by +0.0016, within the noise between sites, and it costs about half a millisecond for each narrow cross section. See [Limitations](limitations.md#narrow-channels-and-the-dems-cells). |
| `Valley_Sections` | True for meandering rivers with wide floodplains, otherwise False | On synthetic meanders it cut ARC's error over the floodplain from 0.15 m to 0.012 m. On the 68 sites it changed the flood maps' mean MCC by nothing to speak of, and it adds about a quarter to the run time. See [Valley Cross Sections](valley_sections.md). |

## Input files

| Key | Description |
| --- | --- |
| `DEM_File` | The DEM, in metres. The stream and land cover rasters, and `BathyWaterMask`, must be on its grid: the same rows and columns, north-up or south-up (not rotated), with coordinates in metres or degrees. Cells without data act as walls. |
| `Stream_File` | The stream raster: each stream cell holds its reach's ID, and every other cell 0. |
| `LU_Raster_SameRes` | The land cover raster. |
| `LU_Manning_n` | A tab-separated table (or Parquet) of Manning's n for each land cover class, with a header row: the class in the first column and its n in the third (as `Create_BaseLine_Manning_n_File` writes it). It must include the water class, `LC_Water_Value`. An n above 10 is read as 0.035, and one at or below 0 as 0.005. |
| `Flow_File` | A table of flows for each reach, as CSV or Parquet. ARC works on the stream cells whose reach ID is in it. |
| `StrmShp_File` | The stream lines (shapefile, GeoPackage and so on). The bathymetry, the drainage-area power laws and the `end_points` slope method need it. With `reach_id` and `downstream_reach_id`, it also gives a reach whose slope can't be measured its neighbours' slope, whatever the slope method, and lets the local averages near a reach's ends take in its main-stem neighbours' cells (see [Stream Slope Methods](stream_slope_methods.md)). |
| `Manual_Cross_Sections_File` | Cross sections to use instead of sampling them from the DEM (see [Manual cross sections](#manual-cross-sections)). |

## Fields

| Key | Default | Description |
| --- | --- | --- |
| `Flow_File_ID` | | The flow file's column of reach IDs, the same IDs as in the stream raster. |
| `Flow_File_QMax` | | The flow file's column of maximum flows in m³/s, the top of each rating curve. |
| `Flow_File_BF` | | The flow file's column of the flows, in m³/s, that fill each channel to the level `Bathy_Use_Banks` names, which sets the depth of the bathymetry: without it, the flow when the DEM was made (a low flow, the baseflow); with it, the flow at the bank elevation (bankfull, or a channel-forming flow). See [Bathymetry](bathymetry.md#water-surface-or-banks). |
| `reach_id` | Reach_ID | The stream lines' field of reach IDs. |
| `downstream_reach_id` | Downstream_Reach_ID | The stream lines' field holding the ID of the reach each reach flows into. |
| `LC_Water_Value` | 80 | The water class in the land cover: 80 in ESA WorldCover, 11 in NLCD. ARC makes the stream cells it works on water, and gives the channel between its banks the water's Manning's n. |

## Cross sections and slopes

| Key | Default | Description |
| --- | --- | --- |
| `Gen_Dir_Dist` | 10 | How many cells each way around a stream cell to fit the stream's direction to, among its own reach's cells. |
| `Degree_Manip` | 1.1 | The range of angles, in degrees, to turn the cross section through to find where the valley is narrowest: up to half of it each way from square to the stream. 0 turns the search off, and so does anything under twice `Degree_Interval`, as the defaults are. |
| `Degree_Interval` | 1.0 | The step between those angles, in degrees. |
| `Low_Spot_Range` | 0 | How many points either side of the stream cell to look along the cross section for lower ground. If there is some, the cross section moves to the cell of the lowest point. 0 turns this off. |
| `Stream_Slope_Method` | local_average_corrected | `local_average`, `local_average_corrected`, `reach_average`, `end_points` or `local_fit` (see [Stream Slope Methods](stream_slope_methods.md)). |
| `Gen_Slope_Dist` | 10 | How many cells each way around a stream cell the slope methods use. At least 1, except for `end_points`, which doesn't use it. |
| `Slope_Low_Percentile`, `Slope_High_Percentile` | 5, 95 | The percentiles of a reach's slopes that `reach_average` and `local_average_corrected` work between. |
| `Valley_Sections` | False | Take each cross section's overbanks from lines out from its banks square to the valley, at the valley's slope, and its channel from its own (see [Valley Cross Sections](valley_sections.md)): over a meandering river's floodplain, the water runs down the valley, not along the channel. |
| `Valley_Dir_Widths` | 16 | How far along the stream, in channel widths either way, the valley's direction and the stream's sinuosity are taken over, for `Valley_Sections`: about a meander's length, which is 10 to 14 widths (see [Valley Cross Sections](valley_sections.md#the-window)). A reach's width is the median distance between its cross sections' banks. 0 uses `Valley_Dir_Dist` for every reach. |
| `Subcell_Channels` | False | Reconstruct the ground of each channel under 10 cells across below the cell scale (the polyline whose cells' means are the DEM's and whose slope changes least, kept between each cell's and its neighbours' means), instead of joining the cells with straight lines, which turns a narrow channel's corners into ramps and leaves it too little area. A carved channel keeps its carve between its banks. See [Limitations](limitations.md#narrow-channels-and-the-dems-cells). |
| `Valley_Dir_Dist` | 40 | How far along the stream, in cells either way, the valley's direction and the stream's sinuosity are taken over, for `Valley_Sections`, where a reach has no banks to give its width, or with `Valley_Dir_Widths` 0. At least 2. |
| `Drop_Edge_Sections` | False | Drop the rating curve of each cross section whose water, at the top of its rating curve, reaches the raster's edge (or comes within `Edge_Margin` of it). The edge cuts such a cross section off, so its water depends on where the raster ends, and tiles whose rasters overlap give the same cell different water, which shows as a seam where their maps meet. A cell without a rating curve isn't mapped from. |
| `Edge_Margin` | 0 | How close (m, on the map) the water's edge may come to the raster's edge before `Drop_Edge_Sections` drops the cross section. 0 drops only those whose water reaches the edge. |

## Roughness and discharge

| Key | Default | Description |
| --- | --- | --- |
| `Depth_Varying_N` | False | Make Manning's n change with the depth of the water. False keeps each land cover's n at every depth. |
| `k_decay` | 6.0 | How quickly n falls to its deep-water value as the water deepens, per metre. Above 0. |
| `shallow_factor` | 2.0 | n at the water's edge, as a multiple of the table's n. At least 1. |
| `deep_factor` | 1.0 | n in deep water, as a multiple of the table's n. Above 0 and at most 1. |
| `slope_adjustment_factor` | 1.0 | Multiplies the square root of the slope in Manning's equation, and so every discharge: 2 doubles the discharge at each water level, as four times the slope would. Above 0. |
| `Standard_Step` | False | Work the water surfaces out along the stream network by the standard-step method, instead of at each cross section's own normal depth (see [Standard Step](standard_step.md)). Where the bathymetry took a channel's depth from the baseflow, the water at flows up to the baseflow is held at most at the level the channel was carved to carry the baseflow at (see [Keeping the baseflow in its banks](standard_step.md#keeping-the-baseflow-in-its-banks)). Needs `StrmShp_File` with `reach_id` and `downstream_reach_id`. |
| `Standard_Step_Eddy_Length` | 300 | How far downstream, in metres, the standard step reckons each step's eddy loss over (see [The eddy loss](standard_step.md#the-eddy-loss)). 0 charges each step the change in velocity head from the cross section below, as HEC-RAS does with cross sections hundreds of metres apart. |
| `Outlet_WSE_File` | None | A table (CSV or Parquet) of the water surface to start reaches that leave the raster from, instead of normal depth: columns `reach_id`, `x` and `y` (a point, in the DEM's coordinates), `flow` and `wse`, a row for each flow at each point, such as a neighbouring tile's rating curves where its raster overlaps this one. An outlet takes its reach's point nearest its last cross section, if one is within 3 cells, its water at the flows within the point's, and beyond them normal depth as far above or below as at the nearest of those. Normal depth at an outlet is often metres off, and the error backs up tens of kilometres; where two tiles' maps meet, it shows as a seam. Needs `Standard_Step`. |

With `Depth_Varying_N`, ground under a depth of water $h$ (in metres) has

$$
n(h) = n \left( \text{deep\_factor} + \frac{\text{shallow\_factor} - \text{deep\_factor}}{1 + \text{k\_decay} \, h} \right)
$$

where $n$ is the table's. Both factors at 1 leave $n$ as it is.

## Bathymetry

See [Bathymetry](bathymetry.md) for how these are used.

!!! danger "Match `Flow_File_BF` to `Bathy_Use_Banks`"
    `Flow_File_BF` must be the flow that fills each channel exactly to the level `Bathy_Use_Banks` names: the DEM's
    water surface without it (a low flow, the baseflow), the bank elevation with it (bankfull). A water mask doesn't
    change that: with `Bathy_Use_Banks` the banks are the bank tops, mask or not. A mismatched pair
    carves the bed metres too deep, or fills the channel in metres too high, and ARC can't tell: there is no error or
    warning. Check the pair before every run with bathymetry. See
    [Water surface or banks](bathymetry.md#water-surface-or-banks).

| Key | Default | Description |
| --- | --- | --- |
| `BATHY_Out_File` | | Where to write the bathymetry raster. Giving it turns the bathymetry on, which needs `Flow_File_BF` or the drainage-area depth law (`coefficient_depth`, `exponent_depth`) for the depth (without either, ARC warns and writes no bathymetry: the width law alone gives no depth), and `StrmShp_File` with `reach_id` and `downstream_reach_id`. `AROutBATHY` is another name for it, and wins if both are given. |
| `Bathy_Use_Banks` | False | False: the DEM at the stream cell is the water's surface at `Flow_File_BF`, and the channel is carved below it, between the DEM's water's edges; the rating curves divide the carved cross section at the bank tops all the same. True: the banks are the bank tops, mask or not, the bank elevation is the level `Flow_File_BF` fills the channel to, and the channel is rebuilt below it. See [Find the banks](bathymetry.md#1-find-the-banks). |
| `Bathy_Trap_H` | 0.2 | The share of the channel's top width that each sloping side of its trapezoid takes, from 0 (a rectangle) to 0.5 (a triangle). |
| `Bathy_Smooth_Banks` | True | Smooth each reach's channel widths and bank elevations. False keeps each cross section's own. |
| `Bathy_Smooth_Bed` | True, or False with `Standard_Step` or `Bathy_Use_Banks` | Smooth each reach's channel bed with a running median. False keeps each cross section's own bed (missing depths are still filled in), and turns `Bathy_Bed_Cap` off. With `Standard_Step` or `Bathy_Use_Banks` it is off unless given: each channel then carries the baseflow at its banks, where a smoothed bed leaves narrow channels too shallow and wide ones too deep. The standard step's water rides above the dips that leaves; without it, each rating curve's water lands metres above or below its banks (see [Bathymetry](bathymetry.md#4-smooth-the-bed-along-each-reach)). |
| `Bathy_Bed_Cap` | True | Keep the channel bed from rising or falling more than 0.5 m per metre along the stream. |
| `Bathy_Bankless_Max_Baseflow` | 20 | With `Bathy_Use_Banks`: a cross section without banks of its own (none were found, or it is a single cell wide), and whose baseflow (m³/s) is below this, carries the baseflow at its stream's level instead of at its reach's bank elevation (see [Bathymetry](bathymetry.md#2-smooth-the-banks-along-each-reach)). 0 turns it off. |
| `BathyWaterMask` | | A raster on the DEM's grid whose non-zero cells are water (it takes the place of the land cover's water). Without `Bathy_Use_Banks`, its flat water beside a burned stream cell gives the water level, and its edge stops the DEM's water running more than two cells past it; with it, it gives the water level the bank tops are found above, and its edge is the banks where no bank tops are found. Without bathymetry, as with `Bathy_Use_Banks`: the banks are the bank tops, and the mask's edge only where none are found. Curve2Flood burns ARC's bathymetry raster as it is, inside the mask or not, and reads the mask only with `Flood_Bathy_Mask`, to flood its water. |
| `FindBanksBasedOnLandCover` | False | Without `BathyWaterMask`, use the land cover's water class as the water mask. |
| `drainage_area_field` | | The stream lines' field of drainage areas, for the power laws below. |
| `coefficient_depth`, `exponent_depth` | | Bankfull depth = `coefficient_depth` × drainage area ^ `exponent_depth`, in metres. Sets the channel's depth instead of the baseflow. |
| `coefficient_width`, `exponent_width` | | Bankfull width = `coefficient_width` × drainage area ^ `exponent_width`, in metres. Sets the width of channels whose banks the water mask, the land cover and the DEM don't show: a single cell that wide where it's no wider than a cell, and otherwise the width of a reach none of whose banks are found (see [Bathymetry](bathymetry.md#1-find-the-banks)). |

## Outputs

See [Outputs](outputs.md) for what each one contains. A table whose path ends in `.parquet` is written as Parquet,
and otherwise as CSV, except the cross-section file, which is otherwise tab-separated.

| Key | Default | Description |
| --- | --- | --- |
| `Print_VDT_Database` | | The VDT database: each stream cell's flows, velocities, top widths and water surface elevations. |
| `Print_Curve_File` | | The curve file: power laws for depth, top width and velocity against flow. |
| `Reach_Average_Curve_File` | False | Fit the curve file's power laws to each reach's rating curves together, instead of to each stream cell's. |
| `Print_AP_Database` | | The flows, areas and wetted perimeters of each rating curve. |
| `VDT_Database_NumIterations` | 15 | How many flows each rating curve has. |
| `XS_Out_File` | | The cross sections ARC used. |
| `Build_Representative_Cross_Section` | False | Build a representative cross section for each reach. |
| `Representative_Cross_Section_File` | | Where to write the representative cross sections. Needed with `Build_Representative_Cross_Section`. |
| `Print_Decision_Log` | | The decision log: a row for each stream cell, with what each step of the run decided for it and its quality flags (see [Outputs](outputs.md#decision-log)). |
| `Log_File` | | A text file of the run's messages at the info level, whatever the console shows, after the ARC version, the inputs and what was worked out from them. |
| `Compression` | ZSTD | The output rasters' compression: ZSTD, DEFLATE, LZW, LZMA or NONE. |

### Representative cross sections only

A run with `Build_Representative_Cross_Section` that asks for none of the VDT database, the curve file and the AP
database builds only the representative cross sections. It needs no `Flow_File_QMax`, and without a flow file it works
on every reach in the stream raster. Its bathymetry takes its depth from the baseflow if `Flow_File`, `Flow_File_ID`
and `Flow_File_BF` are all given, and otherwise from the drainage-area depth law; with neither, ARC makes no
bathymetry.

## Manual cross sections

`Manual_Cross_Sections_File` gives ARC cross sections to use as they are, instead of sampling them from the DEM. It's
a table (CSV, tab-separated if its name ends in `.tsv` or `.txt`, or Parquet) with a row for each cross section:

| Column | Description |
| --- | --- |
| `Flow_File_ID`'s column | The cross section's ID, as in the flow file. A run that builds only representative cross sections uses `reach_id`'s column instead. |
| `Row`, `Col` | The DEM cell at the centre of the cross section. |
| `Ordinate_Dist` | The distance between the points along the cross section, in metres. |
| `XS1_Profile`, `XS2_Profile` | The elevations along each half of the cross section, from the centre cell outwards, as lists such as `[263.1, 263.4, 264.0]`. |
| `LC1_Profile`, `LC2_Profile` | The land cover class at each point. |
| `XS1_Row`, `XS1_Col`, `XS2_Row`, `XS2_Col` | The DEM row and column of each point. |
| `XS_Angle` | Optional: the cross section's angle (see [Directions and sides](miscellaneous.md#directions-and-sides)), which ARC only reports. |
| `Source_Stream_ID` | Optional: the reach the cross section is on, if not its ID. |

Other columns are allowed and ignored.
