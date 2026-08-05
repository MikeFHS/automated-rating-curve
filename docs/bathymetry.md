ARC estimates channel bathymetry at each stream cell by fitting a triangular or trapezoidal cross section whose geometry is constrained by detected bank locations and a target depth. ARC can obtain the initial target depth in one of two ways:

1. From `Flow_File_BF`, by marching downstream-to-upstream through the directed reach network and matching the local friction slope to an effective energy slope.
2. From a drainage-area power law, by combining `drainage_area_field`, `coefficient_depth`, and `exponent_depth`.

After either initial estimate, ARC filters and aggregates the depths by reach and enforces a downstream non-decreasing depth constraint. Consequently, the depth finally burned into a stream cell can differ from its initial network or power-law estimate.

`Flow_File_QMax` is still required either way because ARC still needs a maximum discharge to build the VDT database and curve outputs.

Before any bathymetry is burned into the DEM-derived section, ARC now performs a staged preprocessing workflow:

1. Sample every stream-cell cross section from the DEM or the manual cross-section table.
2. If `Low_Spot_Range` is greater than `0`, shift each section to the lowest nearby sampled point and resample it.
3. If angle testing is enabled, rotate and resample each section to the narrowest trial orientation.
    - During that angle search, ARC now also screens candidate orientations for sampled stream cells away from the thalweg center and prefers angles that avoid following the stream network.
4. After all stream cells have a sampled section, compute the reach-scale INFLECT curves and evaluate the bank-finding hierarchy for every cached section.
    - ARC now first checks whether a one-cell triangular channel is explicitly supported by the sampled geometry. That one-cell path is only accepted when the first sampled cell on both sides of the thalweg is above the thalweg and the drainage-area width prior indicates an expected bankfull width no greater than three raster cells.
5. Within each reach, order the sampled sections from upstream to downstream using the stream-direction information and evaluate the locally detected bank-to-bank top width at each sampled stream cell.
    - If a sampled top width falls outside the 25th-75th percentile band for that reach, ARC replaces its bank indices with bank locations that match the reach-median top width as closely as the sampled cross-section spacing allows.
    - If a sampled section still does not have a valid local bank result after the local bank-search hierarchy runs, ARC now also uses that same reach-median top width to assign replacement bank indices on the sampled profile.
6. After the width screen, ARC tracks the minimum sampled bank elevation found within each reach and then builds a downstream reach network from `StrmShp_File` using the required `reach_id` and `downstream_reach_id` fields.
    - ARC uses that `networkx` digraph to propagate a monotone downstream bank-elevation trend reach by reach.
    - Equal reach controls and flat segments use the numerical `MIN_SLOPE` grade (`1e-8`) instead of the former synthetic `0.001` grade.
    - After establishing each reach's outlet control and initial grade, ARC walks its sampled stream cells from upstream to downstream. If the observed cell minimum bank elevation is below the active interpolation, that observation becomes a new interpolation anchor. ARC recalculates the slope from the nearest upstream anchor to the new low point and applies the revised slope across that interval. It then resets the outgoing slope from the new anchor toward the fixed reach outlet, preventing a steep approaching slope from being extrapolated through the rest of the reach.
    - An observed anchor is capped when necessary so it cannot create a downstream rise. The first cell of a connected reach also cannot exceed the incoming upstream control. Every consecutive pair of cells must fall by at least `MIN_SLOPE` times their along-reach distance.
    - The piecewise outgoing grade calculated at each cell replaces the raster- or flowline-derived slope in the later bathymetry-depth and hydraulic calculations.
    - A `downstream_reach_id` that points outside the available reach set is treated as an external outlet connection rather than an internal smoothing error.
    - ARC keeps the filtered or reach-filled bank indices and top widths from the DEM-based bank-search hierarchy. The network-smoothed reach-scale profile is used only to define the vertical bathymetry elevation target.
    - ARC no longer falls back to local reach minima when the reach graph or its smoothed elevations cannot be built; it now stops with an error instead.
7. Only after those filtered and smoothed bank controls are known does ARC estimate and burn bathymetry.
    - `CrossSection.extract_scalar_hydraulic_geometry()` reduces each staged section to a triangle or trapezoid and uses `smoothed_bank_elevation` as its vertical energy reference. If that elevation is unavailable, the sampled center ordinate is used. The scalar record also contains the cell baseflow and a fixed bathymetry Manning roughness of `0.03`.
    - For baseflow-driven bathymetry, ARC marches through the reach graph from downstream to upstream. At an interior node, it calculates the raw energy gradient as `(smoothed_bank_elevation - downstream_energy_head) / connection_length`; connection lengths shorter than `1` m are treated as `1` m. The effective slope is the larger of that energy gradient and the node's slope control. A missing slope defaults to `0.001`, and the slope control cannot be less than `1e-4`.
    - Trial depth determines cross-sectional area, velocity, hydraulic radius, Manning friction slope, and the velocity-head gradient. Brent's method searches from `0.001` to `25` m for the depth where `friction_slope - velocity_head_gradient` equals the effective slope.
    - A positive-flow outlet is initialized with Manning normal depth using its slope control. A node whose baseflow is zero or negative receives the fixed `0.5` m depth. Outlet velocity remains `0.0`, its initialized friction slope is `1e-4`, and its WSE equals the smoothed bank elevation unless a scalar `default_tailwater_wse` is explicitly supplied by a caller.
    - If an interior friction-slope solve cannot bracket a root, ARC solves Manning normal depth using the effective slope. If that normal-depth solve also cannot bracket a root between `0.001` and `25` m, the final fallback is `0.5` m.
    - A valid drainage-area power-law depth takes precedence over the network result for that stream cell.
    - ARC then groups every staged depth that is marked for application, finite, positive, and below `25` m by source reach. It retains values within the inclusive 25th-75th percentile interval and calculates their median; if interpolation of the quartiles leaves a small sample empty, all valid values for that reach are retained for the median.
    - Finally, ARC moves downstream through the reach graph and raises a downstream reach median when necessary so depth stays equal or increases downstream. At a confluence, the deepest valid incoming branch controls. The constrained reach median replaces the initial depth on every sampled cross section in that reach, including sections whose initial value came from the drainage-area power law.
8. Before writing the bathymetry GeoTIFF, ARC performs one synchronous gap-fill pass. A NaN cell is filled when at least four of its eight surrounding cells contain bathymetry, and it receives the arithmetic mean of only those non-NaN values. Newly filled values are not used to fill any other cell.

# **Bank or Water Surface Elevation**

## **Water Surface Elevations and Baseflow** (`Bathy_Use_Banks = False`)
This method assumes that the DEM represents a water surface elevation (WSE) below bankfull conditions. Bathymetry is inferred by estimating a channel depth that either conveys a specified baseflow or is supplied directly by the drainage-area power-law relationship. The figure below shows the conceptual cross section ARC uses in this mode.

![image](https://github.com/user-attachments/assets/f5352c93-d9a5-4e87-b2d2-40ba64517eea)

ARC identifies bank locations using a tiered approach:

1. Primary method
    - Single-cell feasibility gate: if both adjacent sampled cells are above the thalweg and the expected bankfull width from the drainage-area power law is no more than three cross-section sample spacings, ARC accepts a one-cell triangular channel immediately.
    - Land cover (if enabled via `FindBanksBasedOnLandCover = True`): banks are defined where water-class pixels transition to non-water.
    - Reach-scale INFLECT maximum curvature: ARC samples the INFLECT `d2W/dy^2` curve at every stream cell in the reach, averages those curves by reach, finds the maximum of that mean curve, and converts the corresponding depth back into local bank locations on each sampled cross section.
    - Flat-water assumption (default fallback after the reach-scale INFLECT method): banks are identified where elevation rises above the local WSE.
2. Fallback 1
    - Width-to-depth ratio: ARC uses the cross-section shape to find an inflection point where width-to-depth behavior changes.
3. Fallback 2
    - Elevation inflection point: ARC smooths the profile and detects where lateral slope decreases.
If all methods fail, ARC defaults to a minimal channel. In the staged smoothed-bank workflow, sampled sections that still collapse to a one-cell width can continue into the single-cell triangular bathymetry fallback instead of being discarded immediately.

Once banks are found:

1. ARC evaluates the locally detected bank-to-bank top widths within each reach and replaces any cross section whose width falls outside the reach 25th-75th percentile band with bank indices matching the reach-median top width.
2. If a cross section still does not have a valid bank result after that screen, ARC uses the same reach-median top width to assign bank indices directly from the sampled profile spacing.
3. ARC converts the resulting bank elevations into reach-scale controls by ordering the sections from upstream to downstream, tracking each reach's minimum bank elevation, and using the `reach_id` and `downstream_reach_id` fields from `StrmShp_File` to build a directed reach network with `networkx`.
    - Along each headwater-to-outlet path, ARC uses reach minima to establish outlet controls and an initial grade no smaller than `MIN_SLOPE`. Graph reach lengths are measured in meters: geographic flowlines use ellipsoidal geodesic length, while projected flowlines are converted from their declared CRS linear units. Headwaters receive a separate upstream-to-downstream initialization: the highest filtered raw bank is placed at the upstream endpoint, the reach minimum remains at the outlet, and their difference over graph reach length defines a linear, monotonically decreasing surface. Outlets use the lowest incoming predecessor minimum as their upstream endpoint and their own lowest filtered bank as the missing downstream endpoint. A reach with neither an upstream nor downstream neighbor is handled once as an isolated stream: its filtered maximum and minimum determine the downhill cell order, endpoint elevations, and initial grade. For every terminal type, a filtered bank below the initial line becomes a new anchor and refits the approaching segment before the slope is reset toward the outlet.
    - Within a reach, a cell observation below the active interpolated elevation becomes a new anchor. ARC refits the interval from the nearest upstream anchor to that low point, then recalculates the outgoing grade toward the fixed reach outlet. Anchors are limited to elevations that can reach that outlet while retaining the required downstream fall.
4. ARC keeps the filtered or reach-filled bank indices and the corresponding bank-to-bank top width on each sampled cross section.
    - The smoothed reach-scale elevation is retained for staged diagnostics and for workflows that use bank elevation as the vertical bathymetry control; it does not replace the locally detected width geometry.
5. A trapezoidal channel is constructed:
    - Top width = bank-to-bank distance
    - Bottom width = top width minus side slopes
    - Side-slope width approximately equals `Bathy_Trap_H * total_width`
6. An initial depth is assigned in one of two ways:
    - If `Flow_File_BF` is provided, the supplied discharge drives the downstream-to-upstream reach-network friction-slope solution described above.
    - If the drainage-area power-law parameters provide a valid target, that target takes precedence and is calculated as `coefficient_depth * drainage_area ^ exponent_depth`.
7. ARC replaces the initial cell values with inclusive interquartile-filtered reach medians, then raises downstream medians as needed to prevent depth from decreasing downstream.

If the computed depth is unrealistic (greater than or equal to 25 m), ARC does not rerun bank finding during the bathymetry burn step. Instead, it skips bathymetry for that stream cell and retains the precomputed staged bank diagnostics.

## **Bank Elevations and a Channel Forming Discharge** (`Bathy_Use_Banks = True`)
This method assumes that detected bank elevations represent bankfull conditions, and the channel depth is defined either by a channel-forming discharge or by the optional drainage-area power-law depth relationship. The figure below shows the conceptual cross section ARC uses in this mode.

![image](https://github.com/user-attachments/assets/af7fa153-e0c1-4c4d-a0e3-b6910644cc0d)

Bank detection follows the same sequence as above:

1. Single-cell feasibility gate
2. Land cover banks
3. Reach-scale INFLECT maximum curvature
4. Local bank-elevation search with `_find_bank`
5. Width-to-depth ratio
6. Local elevation inflection point

Once banks are found, ARC:

1. Evaluates the locally detected bank-to-bank top widths within each reach and replaces any cross section whose width falls outside the reach 25th-75th percentile band with bank indices matching the reach-median top width.
2. If a cross section still lacks a valid bank result, uses that same reach-median top width to assign replacement bank indices on the sampled profile.
3. Orders the cross sections from upstream to downstream within each reach, tracks each reach's minimum bank elevation, and uses the downstream reach network to establish outlet controls and initial reach grades. For each headwater, ARC creates its initial line between the highest filtered raw bank at the upstream endpoint and the minimum bank at the outlet. For each outlet, the line connects the lowest incoming predecessor minimum to the outlet's lowest filtered bank. For an isolated stream, ARC explicitly orients the filtered maximum upstream of the filtered minimum and connects those endpoints over the graph reach length. Lower filtered banks encountered along any terminal line become piecewise anchors while the downstream endpoint remains fixed.
    - ARC then interpolates through the ordered cells and promotes an observed cell minimum to a new anchor whenever it is lower than the active interpolation. The interval from the nearest upstream anchor is refitted to the new low point, after which the outgoing grade is reset toward the fixed reach outlet. An outlet-feasibility floor prevents an extreme observation from driving the extrapolated surface beneath that control.
    - Every cell-to-cell segment retains at least `MIN_SLOPE`; an observation that would create a rise is capped by that monotonic constraint.
4. Keeps the filtered or reach-filled bank indices and bank-to-bank top width for each sampled cross section.
5. Uses the smoothed bank elevation as the vertical bathymetry control to compute bankfull elevation while preserving the local width geometry.
6. Estimates depth using one of two paths:
    - Use `Flow_File_BF` in the downstream-to-upstream reach-network friction-slope solve
    - Or, when the drainage-area parameters provide a valid target, use `coefficient_depth * drainage_area ^ exponent_depth` with precedence over the network result
7. Filters the initial depths marked for application that are finite, positive, and below `25` m to the inclusive 25th-75th percentile interval within each reach, assigns the retained median to the reach, and enforces equal-or-increasing depth downstream.
8. Constrains that depth relative to the bankfull elevation before burning the bathymetry into the cross section.
    - If the smoothed bank width still collapses to a single cell, ARC now uses the existing triangular one-cell bathymetry formulation rather than skipping that sampled stream cell.

The same quality control applies here: if the resulting depth is unrealistic, ARC does not rerun bank finding during the bathymetry burn step and instead skips bathymetry for that stream cell.

# **Discovering the Waterfront**
The following sections describe the bank-finding methods in more detail.

## **1.a. Land Cover**
If the stream cell is within the land-cover dataset's designated water class, ARC starts at the stream cell and walks outward along the cross section until the continuous water classification ends. The cell where water ends is treated as the bank location.

![image](https://github.com/user-attachments/assets/b1e28ac8-f103-4837-8292-f897fc08d1d0)

To use this approach, set `FindBanksBasedOnLandCover = True` and provide `LC_Water_Value` in the ARC input file.

## **1.b. Flat Water Surface**
If land cover is not being used or does not produce usable banks, ARC now next tries a reach-scale INFLECT bank estimate before falling back to the older flat-water assumption. The flat-water method moves laterally away from the stream cell until it encounters an elevation that is at least 0.1 m above the stream-cell elevation.

![ChatGPT Image May 2, 2025, 01_22_11 PM](https://github.com/user-attachments/assets/fb10c5c9-d8d2-4c74-bfc6-ce96b27f9e08)

Use this approach by setting `FindBanksBasedOnLandCover = False`, or by omitting that parameter.

## **1.c. Reach-Scale INFLECT Maximum Curvature**
ARC now computes an INFLECT `d2W/dy^2` curve for every sampled cross section in the reach *before bathymetry is applied*, averages those curves by reach, and finds the maximum of the reach-average curve. ARC interprets that shared depth index as a representative bank depth for the reach, converts the depth to a water-surface elevation on each sampled cross section, and then measures the side-specific top widths at that elevation to place the bank locations.

This method is attempted ahead of `_find_bank` in both bathymetry workflows so ARC can use a reach-consistent hydraulic bank indicator before falling back to purely local DEM cues.

## **2. Width-to-Depth Ratio**
If the direct bank search options fail, ARC uses the DEM-derived cross-section elevations to assess width-to-depth behavior. As water depth increases, the width-to-depth ratio typically decreases until flow spills from the channel into the floodplain. ARC uses that inflection to approximate the bank location.

![image](https://github.com/user-attachments/assets/857afd82-7aca-4e25-a85f-4a52740d52fb)

## **3. Changes in Elevation**
If the width-to-depth ratio also fails, ARC smooths the cross-section elevations with a Savitzky-Golay filter and identifies where the lateral change in elevation decreases. That point becomes the bank location.

If the staged bank result produces a bathymetry depth greater than or equal to 25 m, ARC does not re-enter the bank-search hierarchy from the modified cross section. Instead, no bathymetry is estimated for that stream cell.

# **Test the Functionality Yourself**
To use the bank-elevation workflow, set `Bathy_Use_Banks = True` in your ARC input file. If it is omitted, ARC defaults to the WSE-based workflow.

For bathymetry depth, you now have two supported options:

1. Provide `Flow_File_BF` and let ARC match friction and effective energy slopes over the reach network to obtain an initial depth.
2. Omit `Flow_File_BF` and instead provide `drainage_area_field`, `coefficient_depth`, `exponent_depth`, `coefficient_width`, and `exponent_width` together.

Both paths feed the same reach-median filtering and downstream non-decreasing depth constraint before bathymetry is applied.

In the current staged workflow, those drainage-area width parameters are used first to decide whether a sampled section is narrow enough to accept the explicit one-cell triangular fallback. If that gate does not pass, bank placement still comes from land cover, reach-scale INFLECT, and the local DEM fallback methods.

[example_step2.py](https://github.com/MikeFHS/automated-rating-curve/blob/main/examples/example_step2.py) includes the helper used to create a starter ARC input file.

# **References**
Copeland, R. R., Biedenharn, D. S., & Fischenich, J. C. (2000). Channel-Forming Discharge. https://erdc-library.erdc.dren.mil/server/api/core/bitstreams/81b728f8-6ea7-4ef8-e053-411ac80adeb3/content

Knighton, D. (1984). Fluvial Forms and Processes. Edward Arnold.

Savitzky, A., & Golay, M. J. E. (1964). Smoothing and Differentiation of Data by Simplified Least Squares Procedures. Analytical Chemistry, 36(8), 1627-1639. https://doi.org/10.1021/ac60214a047

SciPy. (2024). `savgol_filter`. https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.savgol_filter.html
