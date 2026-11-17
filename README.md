# Automated Rating Curve (ARC) Generator

ARC is a Python tool that carries on much of the work of the hydraulic modeling software AutoRoute. From a DEM, a
stream raster, a land cover raster and a table of flows, it builds a rating curve for each stream cell of the reaches
in the flow file: how the water surface elevation, top width and velocity change with the flow. Flood mapping tools
such as Curve2Flood and FIST turn the rating curves into flood inundation maps.

See the [documentation](https://automated-rating-curve.readthedocs.io/en/latest/) for more details.

## Project status
This project is under active development.

## Getting started
1. Install [Miniforge](https://conda-forge.org/download/) or [Miniconda](https://docs.anaconda.com/miniconda/) if you
   don't have conda yet.
2. Get the code:
```bash
git clone https://github.com/MikeFHS/automated-rating-curve
cd automated-rating-curve
```
3. Create the conda environment from [environment.yaml](environment.yaml), activate it, and install ARC into it:
```bash
conda env create -f environment.yaml
conda activate arc
pip install .
```
4. Run `arc -h` to check that it works, and `arc path/to/ARC_Input_File.yaml` to run ARC.

## Running ARC simulations
The [documentation](https://automated-rating-curve.readthedocs.io/en/latest/) walks through preparing the inputs,
running ARC and using its outputs, and describes every input and output.

## Tests
```bash
pytest testing
```
runs the unit tests, and `pytest testing/unit_test_with_*.py` the end-to-end tests on synthetic channels. GitHub runs
both on every pull request.

## Authors and acknowledgment
Mike Follum has been the lead for this project since the code was in its AutoRoute days. Other contributors include Ahmad Tavakoly, Alan Snow, Joseph Gutenson, Drew Loney, and Ricky Rosas.

Follum Hydrologic Solutions, LLC (FHS) has open-sourced the ARC and Curve2Flood tools to support early flood warning and preparedness in riverine areas lacking adequate flood data or alert systems. These are first-order tools, designed to encourage the development of more advanced modeling systems. We are grateful to all who share in this mission.

The owners of FHS also believe that those with the ability to help others have a responsibility to do so. This belief is rooted in our faith and supported by the following scriptures:

**Hebrews 13:16** - But to do good and to communicate forget not: for with such sacrifices God is well pleased.

**1 Timothy 6:18** - That they do good, that they be rich in good works, ready to distribute, willing to communicate.

**James 4:17** - Therefore to him that knoweth to do good, and doeth it not, to him it is sin.

**Galatians 6:10** - As we have therefore opportunity, let us do good unto all men, especially unto them who are of the household of faith.


## License
This software has been released under a [GNU General Public Licence](https://github.com/MikeFHS/automated-rating-curve/blob/main/license.txt). 


