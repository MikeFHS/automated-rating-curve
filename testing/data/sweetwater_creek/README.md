# Sweetwater Creek near Austell, Georgia

A small real site for the tests: ARC's inputs for one of the USGS flood-mapping sites, clipped to 379 × 295 cells
(1 arc second, about 11 × 8 km) around the nine reaches in `flows.csv`. Sweetwater Creek's main stem carries up to
963 m³/s; two tributaries join it, with 6 and 31 m³/s.

| File | What it is | Source |
| --- | --- | --- |
| `dem.tif` | The DEM, conditioned for the stream network | FABDEM v1-2 |
| `streams.tif`, `streams.gpkg` | The stream raster and lines (`LINKNO`, `DSLINKNO`, `DrainageArea_km2` from `DSContArea`) | GEOGLOWS v2 (TDX-Hydro) |
| `land_cover.tif`, `water_mask.tif` | Land cover classes, and the water mask the banks come from (made by nencarta from the land cover and the DEM) | ESA WorldCover |
| `mannings_n.txt` | Manning's n for each land cover class | |
| `flows.csv` | Each reach's baseflow and maximum flow | GEOGLOWS v2 retrospective |
| `inputs.yaml` | ARC's input file, with paths relative to this folder | |

The data keep their own licences, which this repository's licence doesn't change:

- FABDEM: Hawker, L., Uhe, P., Paulo, L., Sosa, J., Savage, J., Sampson, C., Neal, J. (2022). A 30 m global map of
  elevation with forests and buildings removed. *Environmental Research Letters*, 17, 024016.
  [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/): the DEM here is a clipped, conditioned
  copy, under the same licence, and isn't for commercial use.
- ESA WorldCover (© ESA WorldCover project, contains modified Copernicus Sentinel data):
  [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
- GEOGLOWS v2 streams and flows: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
