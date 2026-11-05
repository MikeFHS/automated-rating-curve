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

Representative mode requires `StrmShp_File` and `Stream_Length_m`. The latter is the name of a stream attribute column containing reach lengths in meters, for example `LengthGeodesicMeters`. ARC matches lengths to COMIDs and fits `K = A Q^p` independently for each reach using its estimated discharge and celerity-based K. The exported `Celerity_K_Power_Law_A` and `Celerity_K_Power_Law_p` values are written once per COMID in a separate companion CSV, so downstream plotting and routing can use the stored coefficients directly. Its filename adds `_exponents` before the original extension: `representative.csv` produces `representative_exponents.csv`. The representative hydraulic-stage CSV contains no coefficient columns.

The representative cross-section CSV includes one row per reach and successful 0.10 m depth stage. At each stage, ARC filters out sampled cross sections whose area or hydraulic radius falls outside two standard deviations of the reach mean, then writes the mean hydraulics from the retained sections and derives a monotonic representative geometry from the staged mean area and top width. Each row includes `Stream_Slope`, the reach-mean positive stream slope from the sampled stream-cell cross sections. See [Outputs](outputs.md#representative-cross-section-export) for the full column definitions.
