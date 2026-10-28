# Produce Geospatial Inputs

ARC needs three rasters on the same grid: a DEM, a land cover raster, and a stream raster whose cells hold their
reach's ID. A fourth, a water mask whose non-zero cells are water (`BathyWaterMask`), is optional: if you give one,
ARC takes the water's level from it, and its edge where it finds no banks (see
[Find the banks](bathymetry.md#1-find-the-banks)). You will usually start from:

1. A DEM, such as the [USGS 3DEP 1/3 arc-second DEM](https://apps.nationalmap.gov/downloader/).
2. A land cover raster, such as the [National Land Cover Database (NLCD)](https://www.mrlc.gov/data).
3. Stream lines, the same ones as your flows, such as the
   [GEOGLOWS streams](http://geoglows-v2.s3-website-us-west-2.amazonaws.com/#streams/).

`arc.process_geospatial_data.Process_ARC_Geospatial_Data` turns these into ARC's inputs. It cuts the land cover to
the DEM's grid, burns the stream lines into a stream raster on the same grid, writes a Manning's n table for the NLCD
classes, and writes a starter input file, `ARC_InputFiles/ARC_Input_File.txt`. Put the DEM, land cover and stream
lines in the same coordinate system first.

::: arc.process_geospatial_data.Process_ARC_Geospatial_Data
    options:
      show_root_heading: true

The stream raster's values, the stream lines' `reach_id` field and the flow file's ID column must hold the same IDs,
though the field and the column can have different names. `id_field` names the stream lines' field: it is burned into
the stream raster, and the input file gives it as `reach_id`. `flow_id_field` names the flow file's column, which the
input file gives as `Flow_File_ID`. The GEOGLOWS streams hold their IDs in `LINKNO`, and
[Produce Streamflow Inputs](produce_streamflow_inputs.md) writes the same IDs in a column named `COMID`, which is
`flow_id_field`'s default.

The starter input file asks for bathymetry when you give a baseflow field or the drainage-area depth law. The
bathymetry follows the stream network, so also give `downstream_id_field`, the stream lines' field holding the ID of
the reach downstream of each reach (`DSLINKNO` in the GEOGLOWS streams). Check the input file before running ARC, and
see [Model Input File](model_input_file.md) for its keys. Besides your files and fields, it sets `Low_Spot_Range` to 10,
and `Degree_Manip` to 6.1 with `Degree_Interval` 1.5, which turns on the angle search; the
[recommended settings](model_input_file.md#recommended-settings) are 0 to 2 and no search.

## Try it

The Shields River test case's `DEM`, `LandCover` and `StrmShp` inputs are in this
[shared folder](https://drive.google.com/drive/folders/1xJkZoY3xoUAcLjODHUINJcS4pk7ljyCr?usp=sharing). Run
[example_step2.py](https://github.com/MikeFHS/automated-rating-curve/blob/main/examples/example_step2.py) with your
paths, then go on to [Running ARC](running_arc.md).
