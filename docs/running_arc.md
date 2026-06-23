Now let's run ARC. [Here](https://github.com/MikeFHS/automated-rating-curve/blob/main/examples/example_step3.py) is a script that illustrates how ARC runs. ARC can also be called directly from the command line by entering `arc "path\to\input.txt"` into the command line window.

::: arc.Arc

The only argument that needs to pass to the `Arc.run()` function is a path to your ARC input file. The file will be the "ARC_InputFiles\ARC_Input_File.txt" that you created in [Produce Geospatial Inputs](produce_geospatial_inputs.md). 

If you are running ARC from gap-crossing seasonal exports, add `Manual_Cross_Sections_File` to the MIF and set `Flow_File_ID` to the cross-section identifier field written by gap-crossing, typically `XS_ID`. ARC will then use the supplied manual profiles instead of sampling new cross sections from the stream raster.

When the ARC simulation completes, you should find a `CurveFile.csv` and `VDT_Database.txt` in your "VDT" folder. If you do, congrats! You've completed a successful ARC simulation!

[Here](outputs.md) is a breakdown of what these values represent, each row in the CurveFile or VDT Database is a stream cell in your domain of interest:

You may wonder, "What has ARC done?", and that's a good question! 

ARC has used [Manning's equation](https://www.weather.gov/aprfc/NormalDepthCalc) to solve for the various hydraulic outputs contained in the `CurveFile.csv` for each stream cell in your domain of interest. ARC solved Manning's equation once for each stream cell, using the max_flow_field value we supplied to it. ARC then divided the resulting water surface elevation into 15 increments, reducing the original maximum water surface elevation by 1/15 each step of the way. For each of these water surface elevation increments, ARC again solved Manning's equation for each stream cell. The result is that for each stream cell, we have 15 separate sets of hydraulic outputs that Manning's equation produced. 

From these 15 sets of solutions, ARC then fits a power function (e.g., $depth = a * streamflow^b$) to the data to produce the rating curves described by the a and b variables above in the `CurveFile.csv`. 

The `VDT_Database.txt` file stores those 15 sets of solutions and looks like this:

![image](https://github.com/user-attachments/assets/efc2e991-c37e-47a1-8c8b-88fd7bb04587)

The q_* values represent the discharge in cubic meters per second, the v_* values represent the cross-sectional average velocity in meters per second, t_* values represent the top-width of the stream in meters, and the wse_* values represent the water surface elevation of the stream, in meters above the datum of your digital elevation model (DEM). 

If you're using the Shields Test Case, once you have a `CurveFile.csv` then let's proceed to [Making Inputs for FIST](making_inputs_for_fist.md).
