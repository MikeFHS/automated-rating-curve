ARC can be used to create a representative cross-section for routing in hydrologic modleing systems. Here, we demonstrate how this works.

Data for this example can be found on this [Google Drive](https://drive.google.com/drive/folders/1ZJVpZwK71TbdteDmJDophFu1SIad0jhz?usp=sharing). If you're following along, please download this data.

Once the above data is downloaded, look in the ARC directory to find this [example implementation script](https://github.com/MikeFHS/automated-rating-curve/blob/bathy_changes/examples/example_representative_cross_section.py)

In this example script, ARC is called to produce a representative cross-section dataset for Southern Africa. The DEM used in this example is [FABDEM](https://research-information.bris.ac.uk/en/datasets/fabdem-v1-2/). Land cover is from the [European Space Agency (ESA) World Cover 2021 dataset](https://esa-worldcover.org/en).

The representative cross-section CSV includes one row per reach and successful 0.10 m depth stage. Along with the staged hydraulic medians and representative geometry, each row includes `Stream_Slope`, the reach-median positive stream slope from the sampled stream-cell cross sections. See [Outputs](outputs.md#representative-cross-section-export) for the full column definitions.
