# Making Inputs for FIST

FIST maps flood inundation from ARC's rating curves. It needs the water surface elevation at each stream cell for the
flows you want to map, and SEED points, which mark the upstream ends of the stream network. `arc.Create_GeoJSON` makes
both from the curve file (or the VDT database), a table of flows, the DEM and the stream lines.

You'll need a CSV flow table with one flow for each reach: for example, the mean of the latest
[GEOGLOWS ECMWF forecast](http://geoglows-v2-forecasts.s3-website-us-west-2.amazonaws.com/). Its columns are taken by
position, whatever their names: the first is the reach ID, the same IDs as the curve file's `COMID` column (the stream
raster's values), and the second the flow, in m³/s. Any other columns aren't used.

## From the curve file

`Run_Main_Curve_to_GEOJSON_Program_Stream_Vector` works out each stream cell's water surface elevation from its
power laws at the reach's flow, and writes them as points in a GeoJSON file. Run it from Python, as
[example_step4.py](https://github.com/MikeFHS/automated-rating-curve/blob/main/examples/example_step4.py) does, or from
the command line with `curve-to-geojson-stream-vector` (add `-h` for its options).

::: arc.Create_GeoJSON.Run_Main_Curve_to_GEOJSON_Program_Stream_Vector
    options:
      show_root_heading: true

The SEED points are the upstream ends of the headwater reaches, those that no other reach flows into. A reach's
upstream end is the end farther from the reach it flows into, or, where that reach isn't in the stream lines, the
higher end on the DEM. They're written to `SEED_Output_File`, and read from it instead when it already exists.

The GeoJSON marks with `"SEED": "1"` the stream cells where the reaches with flows start:

1. For each SEED point, the closest stream cell of its own reach. A SEED point whose reach has no stream cells in the
   GeoJSON (it has no flow in the table, or thinning left none of its cells) marks nothing.
2. For each reach with stream cells whose inflowing reaches have none, its stream cell closest to those reaches: its
   upstream end.

With a flow for every reach, these are just the headwater reaches' upstream ends.

With `Thin_Output`, ARC leaves out stream cells that are likely outliers or repeats:

1. Cells at least three times as deep as their reach's median depth, and every cell of a reach whose median depth is
   0.
2. Cells within 50 m of another cell whose water surface elevation is within 0.05% of theirs.

FIST then maps the flood from the remaining cells' water surface elevations (`WaterSurfaceElev_m`):

![A FIST flood map](https://github.com/user-attachments/assets/942d8d66-6c7e-41ff-a98e-a3c23b62161b)

## From the VDT database

`Run_Main_VDT_to_GEOJSON_Program_Stream_Vector` does the same from the VDT database, interpolating each stream cell's
water surface elevation linearly between its flows, and extrapolating beyond them. With or without `Thin_Output`, it
then drops the cells whose water surface elevation is more than two standard deviations from their reach's mean, and
the cell of a reach with only one. Its thinning skips the depth check.

::: arc.Create_GeoJSON.Run_Main_VDT_to_GEOJSON_Program_Stream_Vector
    options:
      show_root_heading: true

## Try it

For the Shields River test case, an example flow table (`COMID_Q_File`) is in the `714_ExampleForecast` folder of
this [shared folder](https://drive.google.com/drive/folders/1xJkZoY3xoUAcLjODHUINJcS4pk7ljyCr?usp=drive_link). Run
[example_step4.py](https://github.com/MikeFHS/automated-rating-curve/blob/main/examples/example_step4.py) with your
paths and the files you made in the earlier steps.
