Now let's run ARC. [Here](https://github.com/MikeFHS/automated-rating-curve/blob/main/examples/example_step3.py) is a script that illustrates how ARC runs. ARC can also be called directly from the command line by entering `arc "path\to\input.txt"` into the command line window.

::: arc.Arc

The only argument that needs to pass to the `Arc.run()` function is a path to your ARC input file. The file will be the "ARC_InputFiles\ARC_Input_File.txt" that you created in [Produce Geospatial Inputs](produce_geospatial_inputs.md). 

If you are running ARC from gap-crossing seasonal exports, add `Manual_Cross_Sections_File` to the MIF and set `Flow_File_ID` to the cross-section identifier field written by gap-crossing, typically `XS_ID`. ARC will then use the supplied manual profiles instead of sampling new cross sections from the stream raster.

If you also want a reach-level geometry summary, set `Build_Representative_Cross_Section: True` and provide `Representative_Cross_Section_File`. ARC groups sampled stream cells by the positive reach IDs stored in `Stream_File`; it does not require `Flow_File_QMax` for this output. The run samples, low-spot-adjusts, and resamples every stream-cell cross section, evaluates the reach-scale bank-finding workflow, and optionally applies bathymetry when `AROutBATHY` or `BATHY_Out_File` is configured. In representative mode, complete `Flow_File`, `Flow_File_ID`, and `Flow_File_BF` inputs select baseflow-driven bathymetry. If that trio is not complete, ARC uses the full drainage-area power-law configuration when available. If no bathymetry output path is configured, bathymetry estimation and excavation are skipped.

After cross-section preprocessing, ARC evaluates each contributing section every 0.10 m above its local thalweg. At each stage it recomputes area, wetted perimeter, velocity, discharge, and top width with Manning's equation, then stores the reach medians. Staging is capped at 25 m and stops earlier for a reach as soon as any of those hydraulic calculations becomes non-finite. ARC enforces non-decreasing median area and top width before deriving the representative geometry from the staged width-area relationship.

When the ARC simulation completes, you should find a `CurveFile.csv` and `VDT_Database.txt` in your "VDT" folder. If you do, congrats! You've completed a successful ARC simulation!

[Here](outputs.md) is a breakdown of what these values represent, each row in the CurveFile or VDT Database is a stream cell in your domain of interest:

You may wonder, "What has ARC done?", and that's a good question! 

ARC has used [Manning's equation](https://www.weather.gov/aprfc/NormalDepthCalc) to solve for the various hydraulic outputs contained in the `CurveFile.csv` for each stream cell in your domain of interest. ARC solved Manning's equation once for each stream cell, using the max_flow_field value we supplied to it. ARC then divided the resulting water surface elevation into 15 increments, reducing the original maximum water surface elevation by 1/15 each step of the way. For each of these water surface elevation increments, ARC again solved Manning's equation for each stream cell. The result is that for each stream cell, we have 15 separate sets of hydraulic outputs that Manning's equation produced. 

From these 15 sets of solutions, ARC then fits a power function (e.g., $depth = a * streamflow^b$) to the data to produce the rating curves described by the a and b variables above in the `CurveFile.csv`. 

The `VDT_Database.txt` file stores those 15 sets of solutions and looks like this:

![image](https://github.com/user-attachments/assets/efc2e991-c37e-47a1-8c8b-88fd7bb04587)

The q_* values represent the discharge in cubic meters per second, the v_* values represent the cross-sectional average velocity in meters per second, t_* values represent the top-width of the stream in meters, and the wse_* values represent the water surface elevation of the stream, in meters above the datum of your digital elevation model (DEM). 

If you're using the Shields Test Case, once you have a `CurveFile.csv` then let's proceed to [Making Inputs for FIST](making_inputs_for_fist.md).
