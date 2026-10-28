# Installation

ARC runs in a conda environment, which installs GDAL and the other libraries it needs.

1. Install [Miniforge](https://conda-forge.org/download/) or [Miniconda](https://docs.anaconda.com/miniconda/) if you
   don't have conda yet.
2. Get the code, with [Git](https://git-scm.com/downloads) or by downloading it as a zip file from GitHub:
   ```bash
   git clone https://github.com/MikeFHS/automated-rating-curve
   cd automated-rating-curve
   ```
3. Create the environment from `environment.yaml` and activate it. This takes a few minutes.
   ```bash
   conda env create -f environment.yaml
   conda activate arc
   ```
   To use an environment you already have, install the libraries listed in
   [environment.yaml](https://github.com/MikeFHS/automated-rating-curve/blob/main/environment.yaml) into it instead.
   Only the flow file tool, `Create_ARC_Streamflow_Input`, needs netCDF4, xarray, zarr and s3fs: without it you can
   leave them out, and `pip install ".[streamflow]"` adds them later.
4. Install ARC into the environment:
   ```bash
   pip install .
   ```
   If you will change ARC's code, install it with `pip install -e .` instead, so your changes take effect without
   reinstalling.
5. Check that it works:
   ```bash
   arc -h
   ```

To run the tests as well, see [Running the Tests](testing_unit_tests.md).
