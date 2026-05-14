A streamflow dataset is one of the primary inputs into ARC. The inputs are based upon a vector network of stream reaches that have a streamflow associated with them. Two of the more famous of these datasets are the [National Water Model (NWM)](https://water.noaa.gov/about/nwm) and the [Group on Earth Observation (GEO) Global Water Sustainability (GEOGlOWS) European Centre for Medium Range Weather Forecasts (ECMWF) Streamflow Service](https://geoglows.ecmwf.int/). We will focus our instructions on using the GEOGLOWS ECMWF Streamflow Service. 

We will first need to download the streamflow data needed for ARC to function. Historic and return period interval streamflow data from the GEOGLOWS ECMWF Streamflow Service that can be accessed [here](https://data.geoglows.org/available-data). You will need to download the [retrospective](http://geoglows-v2-retrospective.s3-website-us-west-2.amazonaws.com/#retrospective/) and [return period](http://geoglows-v2-retrospective.s3-website-us-west-2.amazonaws.com/#return-periods/) datasets as netCDF files for your domain of interest to produce your ARC input file on your machine. 

Once the retrospective and return period datasets are downloaded, the directories hosting these datasets and the location you would like to save your ARC streamflow input file are passed to the `streamflow_processing.Create_ARC_Streamflow_Input()` function in the `streamflow_processing.py` script in this repository. `streamflow_processing.Create_ARC_Streamflow_Input()` requires the following inputs:

1. NetCDF_RecurrenceInterval_File_Path (str): The file path and filename of a NetCDF of GEOGLOWS ECWMF streamflow service recurrence intervals in your domain of interest.
2. NetCDF_Historical_Folder (str): The path to the directory containing NetCDF's of GEOGLOWS retrospective streamflow estimates for the stream reaches in your domain of interest.
3. Outfile_file_path (str): The path and filename of the file containing the ARC formatted streamflow data (see the image below).

The output of `streamflow_processing.Create_ARC_Streamflow_Input()` will be a csv file that looks like this: 

![image](https://github.com/user-attachments/assets/e43ab48b-cd56-42b1-8577-0365a4b789c6)

and contains: 

- COMID or the stream reach's unique identifier
- average flow, cubic meters per second (qout_mean)
- median flow, cubic meters per second (qout_median)
- maximum flow, cubic meters per second (qout_max) 
- 2-year return period streamflow, cubic meters per second (rp2)
- 5-year return period streamflow, cubic meters per second (rp5) 
- 10-year return period streamflow, cubic meters per second (rp10)
- 25-year return period streamflow, cubic meters per second (rp25)
- 50-year return period streamflow, cubic meters per second (rp50)
- 100-year return period streamflow, cubic meters per second (rp100)

The resulting streamflow is used by ARC user to define a baseflow, for bathymetry estimation, and a maximum flow, to used as the highest streamflow used in estimating an ARC synthetic rating curve. 

[Here](https://drive.google.com/drive/folders/1xJkZoY3xoUAcLjODHUINJcS4pk7ljyCr?usp=sharing) is a shared folder that contains test data for the Shields River in Montana. The "NetCDF_RecurrenceInterval_File_Path" corresponds to "returnperiods_714.nc" in the "714_ReturnPeriods" shared folder. The "NetCDF_Historical_Folder" corresponds to "714_HistoricFlows" in the shared folder. These data were downloaded from the [GEOGLOWS ECMWF Streamflow Service](https://geoglows.ecmwf.int/). 

Try running this test case on your own using [example_step1.py](https://github.com/MikeFHS/automated-rating-curve/blob/main/examples/example_step1.py) on your local machine using the test case data provided above. 

Once you've got an output csv (the "Outfile_file_path" file), lets proceed to Step 2: [Processing ARC Geospatial Inputs](https://github.com/MikeFHS/automated-rating-curve/wiki/Processing-ARC-Geospatial-Inputs)

