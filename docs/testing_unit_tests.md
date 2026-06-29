# Synthetic Unit Tests in `testing/`

ARC includes a small family of synthetic test scripts in `automated-rating-curve/testing/`. These files are useful for two related jobs:

1. Automated regression testing with `pytest`
2. Manual diagnostic runs that generate ARC inputs, outputs, summary tables, and plots

All five scripts build idealized channels with known hydraulic behavior so ARC output can be compared against a Manning-based reference solution. They are intentionally simple enough to isolate one modeling variable at a time, such as cross-section shape, coordinate system, stream angle, or slope method.

## Common Pattern

Each script follows the same broad workflow:

1. Build a synthetic DEM and stream raster
2. Write the ARC inputs needed for one or more synthetic cases
3. Run ARC directly through the Python API
4. Read the resulting VDT output
5. Compare ARC output against a simple Manning reference
6. Either assert regression conditions (`pytest`) or generate plots and reusable artifacts (`python`)

In manual mode, the scripts typically write some or all of the following:

- `dem.tif`
- `stream.tif`
- `land_cover.tif`
- `vdt.csv`
- `vdt_points.geojson`
- `cross_sections.txt`
- `cross_sections.gpkg`
- `stream_network.gpkg`

These outputs are especially useful when debugging cross-section sampling, slope estimation, angle recovery, or endpoint behavior.

## Execution Modes

Each file is designed to support both automated and manual use:

- Automated mode:
  `pytest automated-rating-curve/testing/<script_name>.py`
- Manual mode:
  `python automated-rating-curve/testing/<script_name>.py`

Automated mode writes temporary outputs and runs assertions without opening figures. Manual mode writes persistent artifacts under `testing/` and opens diagnostic plots.

## Test Summary

| Script | Main purpose | Best used for |
| --- | --- | --- |
| `unit_test_with_shapes.py` | Compare trapezoid, rectangle, and triangle channels in projected coordinates | General hydraulic regression testing and shape-to-shape comparisons |
| `unit_test_with_geographic.py` | Repeat the shape test in a geographic CRS | Verifying ARC behavior when raster cell sizes must be converted from degrees to meters |
| `unit_test_with_angle.py` | Compare the three channel shapes for a user-specified `XS_Angle` | Diagnosing stream-direction recovery and cross-section sampling on rotated reaches |
| `unit_test_with_slope.py` | Hold geometry fixed and compare `local_average`, `reach_average`, and `end_points` slope methods | Isolating slope-source effects without angle complications |
| `unit_test_with_slope_and_angle.py` | Compare the three slope methods on one angled triangle reach | Stress-testing the interaction between rotated geometry and slope estimation |

## `unit_test_with_shapes.py`

This is the baseline synthetic regression test. It creates three channel shapes:

- trapezoid
- rectangle
- triangle

All three share the same discharge, roughness, bed slope, reach length, and raster spacing. Only the cross-section geometry changes.

### Functionality

- Builds a projected-coordinate synthetic DEM
- Writes shape-specific ARC input files
- Runs ARC for each shape
- Solves a Manning reference depth for each idealized shape
- Compares ARC WSE, top width, and velocity against the Manning reference
- Plots one row per shape in manual mode

### Utility

Use this file when you want the simplest synthetic comparison for:

- cross-section geometry sensitivity
- basic ARC regression checks
- confirming that refactors do not break normal projected-coordinate runs

This is usually the first synthetic test to run after changing core hydraulic logic.

## `unit_test_with_geographic.py`

This file mirrors `unit_test_with_shapes.py`, but the rasters are written in a geographic CRS (`EPSG:4269`, NAD83) instead of a projected CRS.

### Functionality

- Uses the same three synthetic shapes as the projected baseline test
- Converts a target physical cell size into degree increments
- Writes geographic rasters whose physical spacing remains approximately 1 meter
- Runs ARC through its geographic-coordinate code path
- Compares ARC output against the same Manning reference logic used in the projected test

### Utility

Use this file when you need to verify:

- ARC detection of geographic rasters
- cell-size conversion from degrees to meters
- whether geographic georeferencing changes sampled hydraulics or cross sections

This test is especially useful after modifying code that depends on geotransforms, CRS handling, or slope/distance calculations.

## `unit_test_with_angle.py`

This file keeps the same three synthetic channel shapes, but rotates the reach in plan view so the stream is no longer aligned with raster rows or columns.

The user-facing configuration is the requested ARC `XS_Angle`, stored in radians so it can be compared directly to the `XS_Angle` values ARC writes into the VDT output.

### Functionality

- Rebuilds the synthetic raster domain for the requested angle
- Creates an angled thalweg and stream raster
- Runs ARC for trapezoid, rectangle, and triangle channels
- Exports `stream_network.gpkg` so ARC can use `StrmShp_File`
- Compares ARC `XS_Angle`, WSE, top width, and velocity against the expected synthetic setup
- Plots a fifth diagnostic column for `XS_Angle`

### Utility

Use this file when you need to diagnose:

- stream-direction estimation
- cross-section orientation recovery
- raster sampling artifacts on rotated channels
- differences between theoretically uniform channels and rasterized angled geometry

This is the most useful shape-based test for debugging plan-view orientation issues.

## `unit_test_with_slope.py`

This file removes shape variability and instead fixes the channel to one triangular cross section while comparing ARC's three primary stream-slope methods:

- `local_average`
- `reach_average`
- `end_points`

### Functionality

- Builds one projected-coordinate triangle channel
- Runs ARC once for each slope method
- Uses a shared Manning reference across all three runs
- Exports vector stream inputs for the `end_points` method
- Plots one row per slope method

### Utility

Use this file when you want to isolate differences caused by slope source rather than by cross-section shape.

It is the best test for checking:

- whether `local_average` is noisy
- whether `reach_average` is smoothing too aggressively
- whether `end_points` is sampling the DEM or stream vector correctly

If the three methods diverge unexpectedly in this test, the problem is usually in slope estimation rather than cross-section geometry.

## `unit_test_with_slope_and_angle.py`

This file combines the two more specialized ideas:

- one fixed triangle geometry
- one rotated reach
- three slope methods

It is effectively the most demanding synthetic slope-method comparison in the repository.

### Functionality

- Builds one angled triangular channel
- Runs `local_average`, `reach_average`, and `end_points`
- Compares ARC output against Manning reference values
- Plots WSE, top width, velocity, slope, and `XS_Angle`
- Writes persistent outputs for detailed post-run inspection

### Utility

Use this file when you need to debug interactions between:

- slope estimation
- rotated stream geometry
- cross-section recovery
- raster discretization effects

This is the best synthetic test for reproducing subtle issues that only appear once the stream is both angled and sensitive to the selected slope method.

## Which Test Should You Start With?

Use the following rule of thumb:

- Start with `unit_test_with_shapes.py` for general projected-coordinate regression checks
- Move to `unit_test_with_geographic.py` if the issue may be CRS or geotransform related
- Use `unit_test_with_angle.py` if the issue appears only on angled reaches
- Use `unit_test_with_slope.py` if the issue appears tied to the selected stream-slope method
- Use `unit_test_with_slope_and_angle.py` if the issue depends on both angle and slope-method logic

## Practical Debugging Value

These scripts are more than pass/fail tests. They are compact synthetic experiments that let you:

- inspect exported cross sections directly
- compare VDT rows to known synthetic geometry
- track endpoint artifacts
- test `StrmShp_File`-based slope workflows
- verify `XS_Angle` behavior
- separate hydraulic discrepancies caused by geometry from those caused by slope estimation

That makes the `testing/` folder one of the fastest places to reproduce and understand ARC behavior before moving to larger real-world datasets.
