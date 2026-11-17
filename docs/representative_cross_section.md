# Representative Cross Section

ARC can build one representative cross section for each reach, for routing in hydrologic models that need a single
channel shape per reach.

The data for this example are on [Google Drive](https://drive.google.com/drive/folders/1ZJVpZwK71TbdteDmJDophFu1SIad0jhz?usp=sharing).
The [example script](https://github.com/MikeFHS/automated-rating-curve/blob/main/examples/example_representative_cross_section.py)
builds representative cross sections for a site in southern Africa, from the
[FABDEM](https://research-information.bris.ac.uk/en/datasets/fabdem-v1-2/) DEM and the
[ESA WorldCover 2021](https://esa-worldcover.org/en) land cover.

Set `Build_Representative_Cross_Section: True` and give `Representative_Cross_Section_File`, and the stream lines
(`StrmShp_File`) with `Stream_Length_m`, their field of each reach's full length in metres:

```yaml
Build_Representative_Cross_Section: true
Representative_Cross_Section_File: representative.csv
StrmShp_File: streams.gpkg
Stream_Length_m: LengthGeodesicMeters
```

If the input file asks for no rating curve output (the VDT database, the curve file or the AP database), the run builds
only the representative cross sections and needs no maximum flow. The example takes its channel depths from a
drainage-area power law, so it needs no flow file at all. A file name ending in `.parquet` writes Parquet, and any other
CSV.

For each reach, ARC raises the water 0.10 m at a time over each of its cross sections, averages their area, top width,
discharge and velocity at each stage, and builds a cross section with those averages. With `Standard_Step`, the
discharges are the rating curves' rather than normal depth's, so the representative cross sections and the VDT
database tell the same story. See [Outputs](outputs.md#representative-cross-sections) for how, and for the columns of
the file.

Beside it, ARC writes each reach's travel time K for Muskingum routing as a power law of its discharge,
K = A Q<sup>p</sup>, fitted to the reach's length over five thirds of its velocity at each stage: in
`representative_exponents.csv` beside `representative.csv` (see [Outputs](outputs.md#representative-cross-sections)).
The lengths are the stream lines' own, as `Stream_Length_m` names them: ARC doesn't measure them or convert their units,
and stops with an error if a reach's is missing, not above 0, or given twice.
