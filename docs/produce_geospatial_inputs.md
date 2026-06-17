Once we have a streamflow dataset handy, we need to get to producing ARC's geospatial inputs and it's input file. 

Please ensure that the geospatial data you're using is in the same coordinate system and that the coordinate system is a geographic coordinate system such as the North American Datum of 1983 (NAD83) or World Geodetic System 1984 (WGS84). 

It's important to remember that while we are inputting vector data, such our flowline shapefile, ARC requires all inputs to be raster and to share a common grid structure. 

The geospatial inputs that ARC requires are:

  1. Digital Elevation Model (DEM) such as the [1/3 Arc Second Data from the U. S. Geological Survey (USGS) 3DEP Program](https://apps.nationalmap.gov/downloader/)
  2. Land Cover/Land Use (LU/LC) such as the [2011 National Landcover Database](https://www.mrlc.gov/data/nlcd-2011-land-cover-conus)
  3. Streamlines such as the [TDX-Hydro flowlines used by the GEOGLOWS ECMWF Streamflow Service](http://geoglows-v2.s3-website-us-west-2.amazonaws.com/#streams/). 

To achieve the needed consistence of a common grid, the `arc.process_geospatial_data.Process_ARC_Geospatial_Data()` functions to convert all input geospatial data into a raster that is the same grid as the DEM your utilizing.  

::: arc.process_geospatial_data.Process_ARC_Geospatial_Data
    options:
      show_root_heading: true 

The streamline data you use in your ARC simulation will need to be same that you used to create your flow inputs in [Produce Streamflow Inputs](produce_streamflow_inputs.md). 

Once the script is ran, you should be ready to run your ARC simulation.

A test case for the Shields River watershed in Montana, with the "DEM", "LandCover", and "StrmShp" inputs can be accessed [here](https://drive.google.com/drive/folders/1xJkZoY3xoUAcLjODHUINJcS4pk7ljyCr?usp=sharing). 

Try running this test case on your own using [example_step2.py](https://github.com/MikeFHS/automated-rating-curve/blob/main/examples/example_step2.py) on your local machine using the test case data provided above. 

Once you've successfully ran the test case on your own. Let's proceed to [Running ARC](running_arc.md)



