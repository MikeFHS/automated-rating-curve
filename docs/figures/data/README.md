# Real rivers for the documentation's figures

`make_site_figures.py` draws the standard step and the bank methods on real rivers from these files alone.
`extract_site_data.py` made them from the full inputs, which aren't in the repository: run it again to change what the
figures show.

## The standard step (`standard_step_sites.svg`)

Three USGS flood-mapping sites, each a folder of ARC's inputs as nencarta made them, like Sweetwater Creek's in
`testing/data`, clipped to the stream cells and the water of the flood drawn, without and with the standard step, of
the reach drawn and the reaches below it, and 10 cells around. The DEM is rounded to multiples of 1/256 m (2 mm at
most), which compress better. Against the full sites, the standard step's water along the reaches drawn moves by a
median of about 1 cm (4 cm at most), and each cross section's own water by a median of 2 to 6 cm (up to 0.4 m at a
few): the clip leaves some neighbouring reaches' stream cells out of the slopes' local averages, and the beds move by
the DEM's rounding.

| File | What it is | Source |
| --- | --- | --- |
| `dem.tif` | The DEM, conditioned for the stream network | FABDEM v1-2 |
| `streams.tif`, `streams.parquet` | The stream raster and lines (`LINKNO`, `DSLINKNO`) | GEOGLOWS v2 (TDX-Hydro) |
| `land_cover.tif`, `water_mask.tif` | Land cover classes, and the water mask the banks come from (made by nencarta from the land cover and the DEM) | ESA WorldCover |
| `mannings_n.txt` | Manning's n for each land cover class | |
| `flows.csv` | Each reach's baseflow and maximum flow, and the flow of the flood drawn (`flood`) | GEOGLOWS v2 retrospective; the flood's from the USGS Flood Inundation Mapper's library |
| `inputs.yaml` | ARC's input file, with the site's settings and paths relative to its folder | |

| Folder | Site | Flood drawn |
| --- | --- | --- |
| `wabash` | Wabash River at New Harmony, Indiana (2016) | stage 18 |
| `flatrock` | Flatrock River at Columbus, Indiana (2012) | stage 624.7 |
| `leaf` | Leaf River at Hattiesburg, Mississippi (2013) | stage 3 |

## The bank methods (`bank_methods_sites.svg`)

`bank_sections.csv`: four cross sections as ARC samples them to find their banks (without carving), with nencarta's
Bathy input files for the Karnali (Nepal) and the N40W112 tile (Salt Lake City), by tile and stream cell (`row`, `col`
on the tile's rasters). Each ordinate's `distance` (m) from the stream cell, its `elevation` (m; 9999 off the raster),
and whether the water mask (`water_mask`) and the land cover (`land_cover_water`) have water there (1) or not (0),
blank off the raster. The stream cells count as water in both. Each is cut down to what the figure shows, or as much
more as keeps each method's banks where they are on the whole cross section.

## Licences

The data keep their own licences, which this repository's licence doesn't change:

- FABDEM: Hawker, L., Uhe, P., Paulo, L., Sosa, J., Savage, J., Sampson, C., Neal, J. (2022). A 30 m global map of
  elevation with forests and buildings removed. *Environmental Research Letters*, 17, 024016.
  [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/): the DEMs and elevations here are clipped,
  conditioned copies, under the same licence, and aren't for commercial use.
- ESA WorldCover (© ESA WorldCover project, contains modified Copernicus Sentinel data):
  [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
- GEOGLOWS v2 streams and flows: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
- The USGS's flood flows are in the public domain.
