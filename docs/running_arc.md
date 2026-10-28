# Running ARC

With the input file from [Produce Geospatial Inputs](produce_geospatial_inputs.md) (`ARC_InputFiles/ARC_Input_File.txt`),
run ARC from the command line:

```bash
arc path/to/ARC_InputFiles/ARC_Input_File.txt
```

or from Python, as [example_step3.py](https://github.com/MikeFHS/automated-rating-curve/blob/main/examples/example_step3.py)
does:

```python
from arc import Arc

Arc("path/to/ARC_InputFiles/ARC_Input_File.txt").run()
```

A progress bar shows ARC working through the stream cells. When it finishes, the `VDT` folder holds
`VDT_Database.txt` and `CurveFile.csv`, and the `Bathymetry` folder `ARC_Bathy.tif`.

## What ARC did

For each stream cell, ARC cut a cross section across the stream and used
[Manning's equation](https://www.weather.gov/aprfc/NormalDepthCalc) to find the water surface that carries the
maximum flow from your flow file. It then split the depth up to that water surface into equal steps (15 by default,
`VDT_Database_NumIterations`) and worked out the discharge, velocity, top width and water surface elevation at each.

The VDT database holds those steps, a row per stream cell with a complete rating curve:

```text
COMID,Row,Col,Elev,QBaseflow,Slope,XS_Angle,q_1,v_1,t_1,wse_1,q_2,v_2,t_2,wse_2,...
710045613,412,1033,99.412,12.0,0.001,2.0943951,1.334,0.239,29.652,99.619,4.903,0.398,35.304,99.827,...
```

`q_*` is the discharge in m³/s, `v_*` the mean velocity in m/s, `t_*` the top width in m, and `wse_*` the water
surface elevation in m, in the DEM's datum. `Elev` is the bed the steps start from, so `wse_* − Elev` is the water's
depth.

The curve file fits power laws such as $depth = a \cdot Q^b$ to each stream cell's steps:

![Depth, top width and velocity against discharge: the VDT database's steps and the curve file's power laws](images/rating_curve.svg)

See [Outputs](outputs.md) for every column.

## Manual cross sections

To use cross sections you already have (for example, exported by gap-crossing), add `Manual_Cross_Sections_File` to
the input file, and set `Flow_File_ID` to the field holding their IDs, typically `XS_ID`. ARC then uses them instead of
sampling cross sections from the DEM. See [Model Input File](model_input_file.md#manual-cross-sections).

## Representative cross sections

To get one representative cross section for each reach as well, set `Build_Representative_Cross_Section: True` and
`Representative_Cross_Section_File`. See [Representative Cross Section](representative_cross_section.md).

If you're following the Shields River test case, go on to [Making Inputs for FIST](making_inputs_for_fist.md).
