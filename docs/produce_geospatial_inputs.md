Once we have a streamflow dataset handy, we need to get to producing ARC's geospatial inputs and it's input file. 

Please ensure that the geospatial data you're using is in the same coordinate system and that the coordinate system is a geographic coordinate system such as the North American Datum of 1983 (NAD83) or World Geodetic System 1984 (WGS84). 

It's important to remember that while we are inputting vector data, such our flowline shapefile, ARC requires all inputs to be raster and to share a common grid structure. 

The geospatial inputs that ARC requires are:

  1. Digital Elevation Model (DEM) such as the [1/3 Arc Second Data from the U. S. Geological Survey (USGS) 3DEP Program](https://apps.nationalmap.gov/downloader/)
  2. Land Cover/Land Use (LU/LC) such as the [2011 National Landcover Database](https://www.mrlc.gov/data/nlcd-2011-land-cover-conus)
  3. Streamlines such as the [TDX-Hydro flowlines used by the GEOGLOWS ECMWF Streamflow Service](http://geoglows-v2.s3-website-us-west-2.amazonaws.com/#streams/). 

To achieve the needed consistence of a common grid, the `process_geospatial_data.Process_ARC_Geospatial_Data()` function in the `process_geospatial_data.py` script within this repository functions to convert all input geospatial data into a raster that is the same grid as the DEM your utilizing.  

Here are the arguments for `process_geospatial_data.Process_ARC_Geospatial_Data()`:

- Main_Directory (str): Path to the directory where ARC inputs are stored and where outputs will also be stored.
- id_field (str): Name of the ID field containing the unique identifier for your stream shapefile.
- max_flow_field (str): Name of the field containing the maximum streamflow input into ARC that is within the flow file you generated in [Step 1](https://github.com/MikeFHS/automated-rating-curve/wiki/Formatting-Streamflow-Data-for-ARC). 
- baseflow_bankfull_field (str): Name of the field containing the baseflow or bankfull/channel forming discharge input into ARC that is within the flow file you generated in [Step 1](https://github.com/MikeFHS/automated-rating-curve/wiki/Formatting-Streamflow-Data-for-ARC). 
- flow_file_path (str): Path to the flow file you generated in [Step 1](https://github.com/MikeFHS/automated-rating-curve/wiki/Formatting-Streamflow-Data-for-ARC).
- bathy_use_banks (bool): True/False argument on that describes if you want to run ARC bathymetry estimation using the bank elevations (True) or water surface elevation (False).
- use_land_cover_to_find_banks (bool): True/False argument on whether to use land cover to find banks (True) or to use the flat water surface in the DEM (False)

`process_geospatial_data.Process_ARC_Geospatial_Data()` also assumes that you've stored your inputs in a specific way. In the same directory where you're planning to store your inputs should be:

1. A "DEM" directory with a GeoTIFF named "DEM.tif" representing your DEM input.
2. A "LandCover" directory with a GeoTIFF named "LandCover.tif" representing your LU/LC input.
3. A "StrmShp" directory with a shapefile named "StreamShapefile.shp" representing your streamline input. 

The streamline data you use in your ARC simulation will need to be same that you used to create your flow inputs in [Step 1](https://github.com/MikeFHS/automated-rating-curve/wiki/Formatting-Streamflow-Data-for-ARC). 

Once the script is ran, you should be ready to run your ARC simulation.

A test case for the Shields River watershed in Montana, with the "DEM", "LandCover", and "StrmShp" inputs can be accessed [here](https://drive.google.com/drive/folders/1xJkZoY3xoUAcLjODHUINJcS4pk7ljyCr?usp=sharing). 

Try running this test case on your own using [example_step2.py](https://github.com/MikeFHS/automated-rating-curve/blob/main/examples/example_step2.py) on your local machine using the test case data provided above. 

Once you've successfully ran the test case on your own. Let's proceed to Step 3: [Running ARC and Looking at ARC Outputs](https://github.com/MikeFHS/automated-rating-curve/wiki/Running-ARC-and-Looking-at-ARC-Outputs)



