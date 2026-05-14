ARC estimates channel bathymetry at each stream cell by fitting a trapezoidal cross-section whose geometry is constrained by detected bank locations and a target discharge. Two primary approaches are available, controlled by the `Bathy_Use_Banks` flag. The logic below applies to each stream cell in the modeled domain. 

# **Bank or Water Surface Elevation**

## **Water Surface Elevations and Baseflow** (Bathy_Use_Banks = False)
This method assumes that the DEM represents a water surface elevation (WSE) below bankfull conditions. Bathymetry is inferred by estimating a channel depth that conveys a specified baseflow. For instance, we've used the 50th percentile of the GEOGLOWS ECWMF Streamflow Service's flow duration curve as an approximate baseflow in the past. The figure below of a cross-section of the channel from the DEM describes this scenario.

![image](https://github.com/user-attachments/assets/f5352c93-d9a5-4e87-b2d2-40ba64517eea)

ARC identifies bank locations using a tiered approach:

1. Primary method
    - Land cover (if enabled via `FindBanksBasedOnLandCover` = True): Banks are defined where water-class pixels transition to non-water.
    - Flat-water assumption (default): Banks are identified where elevation rises above the local WSE.
2. Fallback 1 – Width-to-Depth Ratio
    - Uses the cross-section shape to find an inflection point where width-to-depth behavior changes.
3. Fallback 2 – Elevation Inflection Point
    - Smooths the profile and detects where slope decreases laterally.

If all methods fail, ARC defaults to a minimal channel (1 cell wide, no bathymetry).

Once banks are found:

1. A trapezoidal channel is constructed:
    - Top width = bank-to-bank distance
    - Bottom width = top width minus side slopes
    - Side slope width ≈ `Bathy_Trap_H` * total_width
2. Depth is solved iteratively using Manning’s equation to match:
    - Input discharge
    - Local slope
    - Fixed roughness

If the computed depth is unrealistic (≥ 25 m):

1. Retry using:
    - Width-to-depth method
    - Inflection-point method
2. If still invalid:
    - Bathymetry is discarded (depth = 0)

## **Bank Elevations and a Channel Forming Discharge** (Bathy_Use_Banks = True)
This method assumes that detected bank elevations represent bankfull conditions, and the channel must convey a channel-forming discharge. For instance, we've used the 2-year annual recurrence interval discharge from the GEOGLOWS ECWMF Streamflow Service as an approximate channel forming discharge in the past. The figure below of a cross-section of the channel from the DEM describes this scenario. 

![image](https://github.com/user-attachments/assets/af7fa153-e0c1-4c4d-a0e3-b6910644cc0d)

Bank detection follows the same sequence as above:

1. Land cover or flat-water
2. Width-to-depth ratio
3. Inflection point

However, this method additionally:

1. Extracts bank elevations explicitly
2. Computes a bankfull elevation from both sides

Once banks are found:

1. Compute:
    - Bankfull elevation
    - Cross-section geometry (distances, widths)
2. Solve for depth using Manning’s equation:
    - Ensures flow matches d_q_baseflow
    - Constrains depth relative to bankfull elevation

Same quality control as the first method applies: if depth is unrealistic, fallback methods are attempted before discarding bathymetry.

# [**Discovering the Waterfront**](https://music.youtube.com/watch?v=7vOclgxKkic)
The following describe the three methods ARC uses to find the banks of the stream cell cross-section.

## **1.a. Land Cover**
If the stream cell was within land cover dataset's designated water land use class, starting at the stream cell and working outward in the cross-section, ARC looks laterally in the cross-section and determines where the continuous water land use classification ends. The cell where the water land use ends is determined to be the location of the banks. The image below depicts the location of a stream (green line) flowing through a land cover designated as water. In this case, the location where the water ends will be used to define the banks of the stream.

![image](https://github.com/user-attachments/assets/b1e28ac8-f103-4837-8292-f897fc08d1d0)

The arguments 'LC_Water_Value' that specifies the value that represents water in the land use raster and setting 'FindBanksBasedOnLandCover' equal to True are necessary in your ARC input file to use this approach to find the banks of the stream. 

## **1.b. Flat Water Surface**
Alternative to using land cover, you can attempt to find the banks of a stream by assuming that the water surface in the original DEM is flat. Making this assumption, ARC will start at the stream cell and iteratively work outward in the cross-section until it encounters an elevation value that is 0.1 m or greater above the stream cells elevation. The figure below illustrates how the bank location is found using the flat water surface approach. 

![ChatGPT Image May 2, 2025, 01_22_11 PM](https://github.com/user-attachments/assets/fb10c5c9-d8d2-4c74-bfc6-ce96b27f9e08)

In order to use this approach to finding the banks of the stream, the argument 'FindBanksBasedOnLandCover' in the ARC input file must be equal to False or left out of the ARC input file, as ARC defaults to using the flat water surface to find the banks of streams.

## **2. Width-to-Depth Ratio**
If using the land cover (1.a.) or flat surface (1.b.) methodology proves unsuccessful, ARC then utilizes the DEM-derived cross-section elevations to incrementally assess width-to-depth ratio. At a stream’s cross-section, this method assumes that as depth of water in the cross-section increases, the width-to-depth ratio will typically decrease until water spills out of the banks and into the floodplain (Knighton, 1984; Copeland et al., 2000). The point of inflection of the width-to-depth ratio can be considered the location of the banks and this is the secondary method ARC utilizes to find the banks of the stream cell cross-section. The figure below, taken from Copeland et al. (2000), illustrates this process.

![image](https://github.com/user-attachments/assets/857afd82-7aca-4e25-a85f-4a52740d52fb)

## **3. Changes in Elevation**
If the width-to-depth ratio fails to determine the banks of the cross-section, ARC assesses change in elevation moving laterally from the stream cell outward in the cross-section on the left and right side of the cross-section. ARC smoothes the cross-section elevations using a Savitzky-Golay filter (Savitzky and Golay, 1964; SciPy, 2024) and then determines where in the cross-section the change in elevation decreases and uses this point in the cross-section as the stream banks.

In order to remove potential outliers, if any of the above options generate a thalweg depth that was >= 25 m, the same order of operations to find the banks is applied. If ARC reaches a point where either a bathymetry could not be estimated or the bathymetry creates a thalweg depth >= 25 m, no bathymetry is estimated for the stream cell.  The resulting bathymetry is then burned into the cross-section, using the elevation of the stream cell or the minimum stream bank height and the depth of the estimated bathymetry. 

# **Test the Functionality Yourself** 
All that being said, hopefully you're still with us and not drooling on your keyboard! In order to use either the WSE or bank elevations in ARC, we've added the argument **Bathy_Use_Banks** to the ARC code. To use the bank elevations (the second bathymetry method), make sure that you specify **Bathy_Use_Banks** as **True** in your ARC input file. ARC will default to this value being False if it is not specified and will use the WSE methodology instead. In either case, you will need to specify either a baseflow or channel forming discharge when you estimate bathymetry. These two values are typically not the same. This method assumes that you are using the land-cover as your first step in determining the locations and elevations of the banks.

[example_step2.py](https://github.com/MikeFHS/automated-rating-curve/blob/main/examples/example_step2.py) includes a True/False Boolean argument that you can use to manipulate which functionality your using to estimate bathymetry. 

# **References**
Copeland, R. R., Biedenharn, D. S., & Fischenich, J. C. (2000). Channel-Forming Discharge. https://erdc-library.erdc.dren.mil/server/api/core/bitstreams/81b728f8-6ea7-4ef8-e053-411ac80adeb3/content

Knighton, D. (1984). Fluvial Forms and Processes. Edward Arnold.

Savitzky, A., & Golay, M. J. E. (1964). Smoothing and Differentiation of Data by Simplified Least Squares Procedures. Analytical Chemistry, 36(8), 1627–1639. https://doi.org/10.1021/ac60214a047

SciPy. (2024). savgol_filter. https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.savgol_filter.html 
