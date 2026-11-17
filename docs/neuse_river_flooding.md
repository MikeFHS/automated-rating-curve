# Neuse River Flooding

In October 2016, Hurricane Matthew brought devastating floods to the Carolinas, among them along the Neuse River near
Goldsboro, North Carolina. This example maps that flood with flows from the
[National Water Model (NWM) version 3.0 retrospective simulation](https://registry.opendata.aws/nwm-archive/), ARC
and Curve2Flood, and compares the map with flooding seen from satellites.

## Set things up

1. Download the inputs from [this Google Drive folder](https://drive.google.com/drive/folders/1hPwoo1jKDoBKTCwTTN5J7eEDxfHJCnvX?usp=sharing).
2. Install ARC as in [Installation](installation.md), and activate its environment:
   ```bash
   conda activate arc
   ```
3. In the same environment, install [Curve2Flood](https://github.com/MikeFHS/curve2flood): download and unzip it, go
   into its folder (`cd path/to/curve2flood-main`), and run:
   ```bash
   pip install .
   ```
4. The `ARC_InputFiles` folder holds two input files, which ARC and Curve2Flood share. Change the paths in them to
   where you saved the inputs. Keep a tab or spaces between each key, such as `DEM_File`, and its value: both programs
   take the first word on a line as its key and the rest as its value, so a path may contain spaces.

## Run ARC

```bash
arc path/to/ARC_InputFiles/ARC_Input_File_Bathy.txt
```

This run uses the `qout_median` and `qout_max_premium` flows in `nwm30_reanalysis_streamflow.csv`: the median, and
1.5 times the maximum, of the NWM 3.0 retrospective simulation from 1979 to 2023. A progress bar shows ARC working.

When it finishes, the `Bathymetry` folder holds `ARC_Bathy.tif`, the channel bathymetry ARC estimated (see
[Bathymetry](bathymetry.md)), and the `VDT` folder holds the VDT database and the curve file (see
[Outputs](outputs.md)).

## Run Curve2Flood, the first time

Curve2Flood first burns ARC's bathymetry into the DEM:

```bash
curve2flood path/to/ARC_InputFiles/ARC_Input_File_Bathy.txt
```

This writes `FS_Bathy.tif` in the `Bathymetry` folder: the DEM with the channel added. See
[Creating a Topobathymetric Surface with Curve2Flood](https://github.com/MikeFHS/curve2flood/wiki/Creating-a-Topobathymetric-Surface-with-Curve2Flood).

## Run Curve2Flood, the second time

Now map the flood:

1. In `ARC_InputFiles/ARC_Input_File_Forecast.txt`, point `Comid_Flow_File` to
   `flood_event_hydrographs/oct13_1pm.csv`, the NWM 3.0 flows for 13 October 2016 at 13:00 UTC, and set `OutFLD` to
   `oct13_1pm.tif`.
2. Run:
   ```bash
   curve2flood path/to/ARC_InputFiles/ARC_Input_File_Forecast.txt
   ```

See [Creating a Flood Inundation Map with Curve2Flood](https://github.com/MikeFHS/curve2flood/wiki/Creating-a-Flood-Inundation-Map-with-Curve2Flood).

## Compare with what was observed

"All models are wrong, but some are useful." To see how useful this one is, compare its flood map with one mapped
from satellite images by [The University of Alabama Surface Dynamics Modeling Lab](https://sdml.ua.edu/). Satellite
maps aren't perfect either, but they show how well the model did.

The Critical Success Index (CSI) is a simple, popular measure of how well two flood maps agree:

$$
CSI = \frac{TP}{TP + FP + FN}
$$

Counting the cells inside the study area (`Analysis_Boundaries/Neuse_shp.shp`), TP (true positives) are flooded in
both maps, FP (false positives) only in the model's, and FN (false negatives) only in the observed one. A CSI of 1 is
a perfect match.

To work it out, open `calculate_csi.py` from the downloaded folder, set `observed_flood_raster`,
`estimated_flood_raster` and `study_area_boundary_shapefile` at its end, and run it:

```bash
python calculate_csi.py
```

It prints the CSI of your flood map.
