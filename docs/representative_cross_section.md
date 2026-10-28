# Representative Cross Section

ARC can build one representative cross section for each reach, for routing in hydrologic models that need a single
channel shape per reach.

The data for this example are on [Google Drive](https://drive.google.com/drive/folders/1ZJVpZwK71TbdteDmJDophFu1SIad0jhz?usp=sharing).
The [example script](https://github.com/MikeFHS/automated-rating-curve/blob/main/examples/example_representative_cross_section.py)
builds representative cross sections for a site in southern Africa, from the
[FABDEM](https://research-information.bris.ac.uk/en/datasets/fabdem-v1-2/) DEM and the
[ESA WorldCover 2021](https://esa-worldcover.org/en) land cover.

Set `Build_Representative_Cross_Section: True` and give `Representative_Cross_Section_File`. If the input file asks for
no rating curve output (the VDT database, the curve file or the AP database), the run builds only the representative
cross sections and needs no maximum flow. The example takes its channel depths from a drainage-area power law, so it
needs no flow file at all.

For each reach, ARC raises the water 0.10 m at a time over each of its cross sections, averages their area, top width,
discharge and velocity at each stage, and builds a cross section with those averages. With `Standard_Step`, the
discharges are the rating curves' rather than normal depth's, so the representative cross sections and the VDT
database tell the same story. See [Outputs](outputs.md#representative-cross-sections) for how, and for the columns of
the file.
