In October 2016, Hurricane Matthew brought devastating floods to the Carolinas. The Neuse River near Goldsboro, North Carolina was one such location that experienced tremendous flooding.

Using streamflow from the [retrospective simulations of the National Water Model (NWM) version 3.0](https://registry.opendata.aws/nwm-archive/), ARC, and Curve2Flood, we can provide an estimate of flooding that this community experienced. 

We can then take remotely sensed flood inundation maps and see how our simulations performed.

Here are the steps we will take to get our flood inundation maps and compare them to observations.

# Set Things Up

1. In [this Google Drive](https://drive.google.com/drive/folders/1hPwoo1jKDoBKTCwTTN5J7eEDxfHJCnvX?usp=sharing), you can find the inputs necessary to run this simulation. Please download these.  

2. Download and [install Miniconda for your local machine](https://www.anaconda.com/download/). Then open a terminal or command prompt where Miniconda is active and issue this command to create your conda environment (**NOTE: This will take a long time, so be prepared to wait!**):

```bash
conda create -c conda-forge -n arc_n_curve2flood_py310 python=3.10 numba=0.60 pillow=9.0.1 gdal geopandas pandas netcdf4 dask fiona s3fs xarray zarr geojson progress tqdm pygeos rasterio pyarrow memory_profiler
```

3. Now let's activate your Miniconda environment with the command:

```bash
conda activate arc_n_curve2flood_py310
```

4. Use the Conda environment you set up in Step 2 and activated in Step 3 by [downloading and extracting ARC](https://github.com/MikeFHS/automated-rating-curve/archive/refs/heads/main.zip), navigating into the `automated-rating-curve-main` directory using your command prompt (i.e., `cd path\to\automated-rating-curve-main`), and run the command:

```bash
pip install .
```

4. Use the Conda environment you set up in Step 2 and activated in Step 3 by [downloading and extracting Curve2Flood](https://github.com/MikeFHS/curve2flood/archive/refs/heads/main.zip), navigating into the `curve2flood-main` directory using your command prompt (i.e., `cd path\to\curve2flood-main`), and run the command:

```bash
pip install .
```

6. In the files you downloaded from Google Drive, a folder labeled "ARC_InputFiles" will contain two text files. These text files will be used by ARC and Curve2Flood to perform their simulations. Most of these inputs are the files you downloaded from Google Drive. Replace the paths in the text files with those on your local machine. **NOTE: Make sure a tab is maintained between a label like "DEM_File" and the path you add. You may need to download and install [TextPad](https://www.textpad.com/download) in order to make sure a proper tab remains.**

7. Open or navigate to the command window you are using to run Miniconda, activate your Conda environment.

# Running ARC

Open up the "path\to\your\ARC_InputFiles\ARC_Input_File_Bathy.txt" file that is in the Google Drive data you downloaded. Let's use [TextPad](https://www.textpad.com/home) to open it since tabs in this file are important. Replace the filepaths in this file with those are on your machine. NOTE: The space between an argument and the path must have a tab between it!

Now let's first run our ARC simulation by typing the command below into your prompt:

```bash
arc "path\to\your\ARC_InputFiles\ARC_Input_File_Bathy.txt"
```

In this simulation, you're using the "qout_median" and "qout_max_premium" streamflow in the "nwm30_reanalysis_streamflow.csv" file. These are the median and 1.5-times-maximum from the 1979-2023 simulation of the NWM version 3.0 retrospective simulation.  

If ARC begins to run successfully, you should see a progress bar beginning to fill. The outputs of this are simulation will be in your Bathymetry and VDT folder. 

In your Bathymetry folder, the outputs are your ARC_Bathy.tif file, which can be opened in the GIS of your choice and represent the two-dimensional profile for bathymetry that ARC was able to estimate for our modeling domain. A description of how ARC estimates bathymetry can be found [here](https://github.com/MikeFHS/automated-rating-curve/wiki/Estimating-Bathymetry-in-ARC). 

In your VDT folder, the outputs are the VDT_Database.txt and ARC_Curvefile.csv, which are described [here](https://github.com/MikeFHS/automated-rating-curve/wiki/Running-ARC-and-Looking-at-ARC-Outputs). 

# Running Curve2Flood, The First Time
Once your ARC simulation ceases, we are now going to run Curve2Flood to "burn" or "smooth" the ARC bathymetry into the source digital elevation model (DEM). This can be done by running the command below in the same terminal where you ran ARC.

```bash
curve2flood "path\to\your\ARC_InputFiles\ARC_Input_File_Bathy.txt"
```
Once the simulation ceases, you will now see a new file in your bathymetry folder named "FS_Bathy.tif". This is the DEM with the ARC bathymetry added to it. We will now use this and Curve2Flood to map different days of flooding during 2016 floods. 

For more details on this Curve2Flood bathymetry smoothing process check out [this page](https://github.com/MikeFHS/curve2flood/wiki/Creating-a-Topobathymetric-Surface-with-Curve2Flood).

# Running Curve2Flood, The Second Time
Now, finally, we can create some flood inundation maps!

Open up the "path\to\your\ARC_InputFiles\ARC_Input_File_Forecast.txt" file that is in the Google Drive data you downloaded. Let's use [TextPad](https://www.textpad.com/home) to open it since tabs in this file are important. Replace the filepaths in this file with those are on your machine. NOTE: The space between an argument and the path must have a tab between it!

1. In "ARC_InputFiles\ARC_Input_File_Forecast.txt", lets point the "Comid_Flow_File" argument to the "flood_event_hydrographs\oct13_1pm.csv" streamflow and name "OutFLD" the "oct13_1pm.tif". The file "oct13_1pm.csv" contains NWM Version 3.0 streamflow for the October 13, 2016 at 13:00 UTC for our study area. 

2. Now let's simulate the flood inundation with Curve2Flood by issuing the following command in your command prompt:

```bash
curve2flood "path\to\your\ARC_InputFiles\ARC_Input_File_Forecast.txt"
```
For more details on this Curve2Flood flood inundation mapping process check out [this page](https://github.com/MikeFHS/curve2flood/wiki/Creating-a-Flood-Inundation-Map-with-Curve2Flood).

# Compare Results to Observations
Remember this famous quote, ["All models are wrong, but some are useful"](https://en.wikipedia.org/wiki/All_models_are_wrong)?

If you're a nerd, maybe. But that quote summarizes how our computer models are never going to totally reflect reality. 

However, let's try to see how well our flood maps reflect reality.

To do that, we are going to take some observed flood inundation maps that were captured by satellites and refined by [The University of Alabama Surface Dynamics Modeling Lab](https://sdml.ua.edu/) to see how well our estimates of flood inundation compared to what the satellites captured for the 2016 floods. Remotely sensed flood inundation maps are not perfect themselves but this one will give us an idea of how well our model performed.

To do this, were going to use a simple but popular and effective metric for evaluating estimated flood inundation maps called the Critical Success Index (CSI). CSI is:
$$
CSI =  \frac{TP}{TP+FP+FN} 
$$
To calculate CSI, we will use a defined study area, the "Analysis_Boundaries\Neuse_shp.shp" you have in the folder you downloaded from the Google Drive. The study area will define which raster cells we are analyzing. In the CSI formula, TP or true positive, is the number of raster cells in our observed flood inundation raster and our estimated flood inundation raster that are both flooded. FP or false positive, is the number of raster cells where our observed flood inundation raster is not flooded but the estimated flood inundation raster is flooded. FN or false negative, is the number of raster cells where where observed flood inundation raster is flooded but the estimated flood inundation raster is not flooded.   

In your downloaded folder, you have a Python script named "calculate_csi.py". In the command prompt where you are running ARC and Curve2Flood, type:

```bash
cd path\to\where\your\downloaded\file\is
```

Now open "calculate_csi.py" in a text editor of your choice. Scroll to the bottom of the script and update the paths for observed_flood_raster,     estimated_flood_raster, and study_area_boundary_shapefile.

Then go back to your command prompt and issue this command:

```bash
python calculate_csi.py
```

When the processing ceases, the command prompt will issue a decimal value. This is your CSI for the flood map you created using ARC and Curve2Flood.




