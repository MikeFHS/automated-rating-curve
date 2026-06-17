Once you have ARC output, mainly your `Curvefile.csv` and/or `VDT_Database.txt`, we can use these to create flood inundation maps. One option is to use the FIST software.

For the same stream network you used to create your `CurveFile.csv`, you'll also need to retrieve the streamflow's you'll want to use for your FIST simulation and a means to convert those into inputs for your simulation. For instance, you may want to download the latest [GEOGLOWS ECMWF forecast](http://geoglows-v2-forecasts.s3-website-us-west-2.amazonaws.com/) and simulate the mean of the this forecast ensemble. For FIST, the streamflow input will need to be in the same format you generated in [Produce Streamflow Inputs](https://automated-rating-curve.readthedocs.io/en/latest/produce_streamflow_inputs/) but have just one streamflow value for each stream reach. 

To create the inputs for FIST, the Python functions in `Create_GeoJSON.py` were created. These functions do two things. 

1. Builds the SEED dataset which determines the upstream extents of the model domain. The SEED locations are the uppermost coordinates of those uppermost stream reaches.

2. Creates a GeoJSON file that will be used by FIST to simulate flood inundation. 

`Create_GeoJSON.py` determines SEED locations using a stream vector based methodology.

The function `Run_Main_Curve_to_GEOJSON_Program_Stream_Vector()` will quickly find the SEED locations, using the stream vector network described in [Produce Geospatial Inputs](produce_geospatial_inputs.md), and will identify the SEED locations in the resulting GeoJSON file. `Run_Main_Curve_to_GEOJSON_Program_Stream_Vector()` relies on the Python library [Networkx](https://networkx.org/documentation/stable/index.html) to identify all uppermost streams in your domain. The code then finds the SEED locations.

More specifically, the helper function `find_SEED_locations()` in `Create_GeoJSON.py` first removes duplicate rows by stream identifier so downstream lookups remain stable, then identifies source reaches as those whose `Stream_ID_Field` never appears in any other row's `Downstream_ID_Field`. Only those source reaches are eligible to create SEED points.

For each source reach, `find_SEED_locations()` extracts the terminal endpoints from the reach geometry without assuming that the stored coordinate order already runs from upstream to downstream. For `LineString` reaches this means taking the two boundary endpoints. For `MultiLineString` reaches, the function evaluates each component line, collects its boundary endpoints, and removes duplicates so shared internal coordinates are not treated as separate SEED candidates.

If the downstream reach geometry exists in the local vector network, the function measures each candidate terminal point against that downstream geometry. The terminal point or points with the minimum distance are treated as the downstream connection, and the remaining terminal points are treated as upstream SEED candidates. For multipart reaches, `prune_downstream_points_within_multiline()` then builds a small endpoint graph with NetworkX and removes any candidate point that is still topologically downstream of another candidate point inside the same composite reach. This allows a branched headwater reach to keep multiple real upstream seed points while dropping terminals that are farther downstream within the same geometry.

When a reach looks like a local source reach because its downstream identifier is missing from the local vector network, the workflow switches to a DEM-backed inference step. Before sampling the DEM, `filter_terminal_points_adjacent_to_composite_ordinates()` removes terminal candidates that are adjacent to two or more ordinates in the composite geometry, because those points behave like internal junction nodes rather than exposed termini. The remaining terminal points are sampled against the DEM, the lowest-elevation terminal is treated as the outlet, and the higher-elevation terminal points are preserved as SEED candidates. If every sampled terminal ties at the same minimum elevation, the code falls back to keeping the highest sampled terminal so the reach still produces one SEED point instead of none.

`Run_Main_Curve_to_GEOJSON_Program_Stream_Vector()`can be called using an independent Python script, like in [example_step4.py](https://github.com/MikeFHS/automated-rating-curve/blob/main/examples/example_step4.py), or via command line. If you choose to use the command line, type `curve-to-geojson-stream-vector -h` into the command line for additional instructions.

::: arc.Create_GeoJSON.Run_Main_Curve_to_GEOJSON_Program_Stream_Vector 
    options:
      show_root_heading: true 

The output GeoJSON from either the stream raster or stream vector workflow contains point features that look like the image below. These points are stream cell locations from the CurveParam_File where the a and b parameters have been paired with a streamflow to estimate the water surface elevation. This is not a very pretty file, but the unnecessary spacing has been removed from the GeoJSON to reduce the file size.

![image](https://github.com/user-attachments/assets/244864e6-be89-4d85-82ef-5a53cbd928e9)

The points in the SEED dataset will be marked as "SEED": "1" in the GeoJSON file. FIST utilizes the water surface elevation ("WaterSurfaceElev_m" in the GeoJSON) and the location of the stream cell from the GeoJSON to produce a flood inundation map. 

Also, by marking `Thin_Output = True`, the stream cells have been filtered in two ways that attempt to reduce the likelihood of outliers and reduce redundancy, these are:

1. Removes stream cells that have depths that are greater than 3 times the average depth for the stream reach
2. Removes stream cells that are within 50 meters of other streams and have water surface elevations that are within 0.05% of one another.

The remaining stream cells are fed into FIST producing a flood inundation map like the image below:

![image](https://github.com/user-attachments/assets/942d8d66-6c7e-41ff-a98e-a3c23b62161b)

If you prefer to use the `VDT_Database.txt` to create your GeoJSON input, `Run_Main_VDT_to_GEOJSON_Program_Stream_Vector()` can be ran just like `Run_Main_Curve_to_GEOJSON_Program_Stream_Vector()` are but pointed to the `VDT_Database.txt` instead of the `CurveFile.csv`. It uses the same DEM-backed SEED fallback, so the lowest terminal point is treated as the outlet whenever the downstream reach geometry is missing from the local vector network.

::: arc.Create_GeoJSON.Run_Main_VDT_to_GEOJSON_Program_Stream_Vector 
    options:
      show_root_heading: true 

If you're following along with the Shields River test case, an example "COMID_Q_File" containing a streamflow forecast can be found [here](https://drive.google.com/drive/folders/1xJkZoY3xoUAcLjODHUINJcS4pk7ljyCr?usp=drive_link) in the "714_ExampleForecast" folder.

Try running this test case on your own using [example_step4.py](https://github.com/MikeFHS/automated-rating-curve/blob/main/examples/example_step4.py) on your local machine using the test case data provided above and the data you produced using [Produce Streamflow Inputs](produce_streamflow_inputs.md) and [Produce Geospatial Inputs](produce_geospatial_inputs.md).








