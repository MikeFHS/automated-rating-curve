Once you have ARC output, mainly your `Curvefile.csv` and/or `VDT_Database.txt`, we can use these to create flood inundation maps. One option is to use the FIST software.

For the same stream network you used to create your `CurveFile.csv`, you'll also need to retrieve the streamflow's you'll want to use for your FIST simulation and a means to convert those into inputs for your simulation. For instance, you may want to download the latest [GEOGLOWS ECMWF forecast](http://geoglows-v2-forecasts.s3-website-us-west-2.amazonaws.com/) and simulate the mean of the this forecast ensemble. For FIST, the streamflow input will need to be in the same format you generated in [Step 1](https://github.com/MikeFHS/automated-rating-curve/wiki/Formatting-Streamflow-Data-for-ARC) but have just one streamflow value for each stream reach. 

To create the inputs for FIST, the Python script `Create_GeoJSON.py` was created. The script does two things. 

1. Builds the SEED dataset which determines the upstream extents of the model domain. The SEED locations are the uppermost coordinates of those uppermost stream reaches.

2. Creates a GeoJSON file that will be used by FIST to simulate flood inundation. 

`Create_GeoJSON.py` determines SEED locations using a stream vector based methodology.

The function `Run_Main_Curve_to_GEOJSON_Program_Stream_Vector()` will quickly find the SEED locations, using the stream vector network described in [Step 2](https://github.com/MikeFHS/automated-rating-curve/wiki/Processing-ARC-Geospatial-Inputs), and will identify the SEED locations in the resulting GeoJSON file. `Run_Main_Curve_to_GEOJSON_Program_Stream_Vector()` relies on the Python library [Networkx](https://networkx.org/documentation/stable/index.html) (thank you ChatGPT) to identify all uppermost streams in your domain. The code then finds the SEED locations.

`Run_Main_Curve_to_GEOJSON_Program_Stream_Vector()`can be called using an independent Python script, like in [example_step4.py](https://github.com/MikeFHS/automated-rating-curve/blob/main/examples/example_step4.py), or via command line. If you choose to use the command line, type `curve-to-geojson-stream-vector -h` into the command line for additional instructions.

`Run_Main_Curve_to_GEOJSON_Program_Stream_Vector()` requires the following inputs:

1. CurveParam_File (str): The CurveFile.csv you created in [Step 3](https://github.com/MikeFHS/automated-rating-curve/wiki/Running-ARC-and-Looking-at-ARC-Outputs).
2. COMID_Q_File (str): The file containing the unique ID and streamflow for your area of interest that you intend to simulate flood inundation for using FIST.
3. STRM_Raster_File (str): The path to the stream raster that you created as input for ARC in [Step 2](https://github.com/MikeFHS/automated-rating-curve/wiki/Processing-ARC-Geospatial-Inputs).
4. OutGeoJSON_File (str): The path to the GeoJSON file that will be used as input into the FIST model. An empty FIST directory was set up in [Step 2](https://github.com/MikeFHS/automated-rating-curve/wiki/Processing-ARC-Geospatial-Inputs) as a place to stash this file.
5. OutProjection (str): The string describing the output coordinate system of of the GeoJSON file (e.g., "EPSG:4269").
6. StrmShp (str): The file path and file name of the vector shapefile of flowlines you used in [Step 2](https://github.com/MikeFHS/automated-rating-curve/wiki/Processing-ARC-Geospatial-Inputs).
7. Stream_ID_Field (str): The field in the StrmShp that is the streams unique identifier.
8. Downstream_ID_Field (str): The field in the StrmShp that is used to identify the stream downstream of the stream.
9. SEED_Output_File (str): The file path and file name of the output shapefile that contains the SEED locations and the unique ID of the stream each represents. An empty FIST directory was set up in [Step 2](https://github.com/MikeFHS/automated-rating-curve/wiki/Processing-ARC-Geospatial-Inputs) as a place to stash this file.
10. Thin_Output (bool): True/False of whether or not to filter the output GeoJSON.

The output GeoJSON from either the stream raster or stream vector workflow contains point features that look like the image below. These points are stream cell locations from the CurveParam_File where the a and b parameters have been paired with a streamflow to estimate the water surface elevation. This is not a very pretty file, but the unnecessary spacing has been removed from the GeoJSON to reduce the file size.

![image](https://github.com/user-attachments/assets/244864e6-be89-4d85-82ef-5a53cbd928e9)

The points in the SEED dataset will be marked as "SEED": "1" in the GeoJSON file. FIST utilizes the water surface elevation ("WaterSurfaceElev_m" in the GeoJSON) and the location of the stream cell from the GeoJSON to produce a flood inundation map. 

Also, by marking `Thin_Output = True`, the stream cells have been filtered in two ways that attempt to reduce the likelihood of outliers and reduce redundancy, these are:

1. Removes stream cells that have depths that are greater than 3 times the average depth for the stream reach
2. Removes stream cells that are within 50 meters of other streams and have water surface elevations that are within 0.05% of one another.

The remaining stream cells are fed into FIST producing a flood inundation map like the image below:

![image](https://github.com/user-attachments/assets/942d8d66-6c7e-41ff-a98e-a3c23b62161b)

If you prefer to use the `VDT_Database.txt` to create your GeoJSON input, `Run_Main_VDT_to_GEOJSON_Program_Stream_Vector()` can be ran just like `Run_Main_Curve_to_GEOJSON_Program_Stream_Vector()` are but pointed to the `VDT_Database.txt` instead of the `CurveFile.csv`.

If you're following along with the Shields River test case, an example "COMID_Q_File" containing a streamflow forecast can be found [here](https://drive.google.com/drive/folders/1xJkZoY3xoUAcLjODHUINJcS4pk7ljyCr?usp=drive_link) in the "714_ExampleForecast" folder.

Try running this test case on your own using [example_step4.py](https://github.com/MikeFHS/automated-rating-curve/blob/main/examples/example_step4.py) on your local machine using the test case data provided above and the data you produced in [Step 2](https://github.com/MikeFHS/automated-rating-curve/wiki/Processing-ARC-Geospatial-Inputs) and [Step 3](https://github.com/MikeFHS/automated-rating-curve/wiki/Running-ARC-and-Looking-at-ARC-Outputs). The [example_step4.py](https://github.com/MikeFHS/automated-rating-curve/blob/main/examples/example_step4.py) assumes we're using the vector methodology we discussed above. 








