# Produce Streamflow Inputs

ARC needs a table of flows for the stream reaches in your area: a maximum flow for the top of each rating curve, and
usually a baseflow for the bathymetry. Large-scale streamflow models such as the
[National Water Model (NWM)](https://water.noaa.gov/about/nwm) and the
[GEOGLOWS ECMWF Streamflow Service](https://geoglows.ecmwf.int/) provide them. These steps use GEOGLOWS.

1. Download the [retrospective](http://geoglows-v2-retrospective.s3-website-us-west-2.amazonaws.com/#retrospective/)
   and [return period](http://geoglows-v2-retrospective.s3-website-us-west-2.amazonaws.com/#return-periods/) NetCDF
   files for your area.
2. Pass them to `arc.streamflow_processing.Create_ARC_Streamflow_Input`, which writes a CSV file with a row for each
   reach: its ID (`COMID`), its mean and median flow (`qout_mean`, the mean of each retrospective file's means, and
   `qout_median`, the median of their medians), and its maximum flow (`qout_max`) and 2- to 100-year flows (`rp2` to
   `rp100`), copied from the return period file.

::: arc.streamflow_processing.Create_ARC_Streamflow_Input
    options:
      show_root_heading: true

In the input file, `Flow_File_ID` names the column of reach IDs, `Flow_File_QMax` the maximum flow (for example the
100-year flow) and `Flow_File_BF` the baseflow (for example `qout_median`). If you estimate the bathymetry from drainage
area instead, you can leave out `Flow_File_BF` (see [Model Input File](model_input_file.md#bathymetry)). The IDs must
be the stream raster's values. [Produce Geospatial Inputs](produce_geospatial_inputs.md) sets `Flow_File_ID` to
`COMID`, this file's column, unless you give it another (`flow_id_field`).

## Try it

Test data for the Shields River in Montana are in this
[shared folder](https://drive.google.com/drive/folders/1xJkZoY3xoUAcLjODHUINJcS4pk7ljyCr?usp=sharing):
`NetCDF_RecurrenceInterval_File_Path` is `returnperiods_714.nc` in the `714_ReturnPeriods` folder, and
`NetCDF_Historical_Folder` is the `714_HistoricFlows` folder. Run
[example_step1.py](https://github.com/MikeFHS/automated-rating-curve/blob/main/examples/example_step1.py) with your
paths, then go on to [Produce Geospatial Inputs](produce_geospatial_inputs.md).
