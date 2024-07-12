# Automated Rating Curve (ARC) Generator
ARC is an Python-based code that encompasses much of the functionality of the hydraulic modeling software AutoRoute. ARC inputs a series of raster datasets and outputs rating curves that describe the relationship between streamflow and water surface elevation.

The resulting rating curves can be paired with a flood inundation mapping software to produce flood inundation maps.

## Project status
This project is under active development.

## Getting started
Below are step-by-step instructions for setting up the ARC tool.

1. Clone this repository.
   - In the upper right hand corner of this screen, left-click the green <> Code button and copy the text.
2. Download [Miniconda](https://docs.anaconda.com/miniconda/miniconda-install/).
3. Open the "Anaconda Prompt" and create you Conda environment.
   - To activate the "ARC" environment open an Anaconda Command Prompt, navigate to where you've clone the ARC repository on your local machine, and type "conda env create -f environment_ARC.yml"
4. Your ARC environment should be ready to roll!

## Running ARC simulations
Step-by-step instructions for running ARC simulations can be found in the Wiki page at the top of this page.

1. Inputs
  - Example Input Datasets
     1. [DEM from the 1/3 Arc Second National Elevation Dataset](https://apps.nationalmap.gov/downloader/)
     2. [Land Cover from the National Land Cover Database 2011](https://www.mrlc.gov/data/nlcd-2011-land-cover-conus)
     3. [Streamlines from GeoGLoWS](http://geoglows-v2.s3-website-us-west-2.amazonaws.com/#streams/)
     4. [Return period flow rates from GeoGLoWS](http://geoglows-v2-retrospective.s3-website-us-west-2.amazonaws.com/#return-periods/)
     5. [GeoGLoWS Streamflow Data in General](https://data.geoglows.org/available-data)

## Using ARC output with flood iundation mapping software
- Instructions for using ARC to create inputs into the Flood Inundation Surface Topology (FIST) model can be found here. 
- Instructions for using ARC to create inputs into the FloodSpreader model can be found here. 

## Authors and acknowledgment
Mike Follum has been the lead for this project since the code was in it's AutoRoute days. Other contributors include Joseph Gutenson and Drew Loney.

## License
This software has been released under a [GNU General Public Licence](https://github.com/MikeFHS/automated-rating-curve/blob/main/license.txt). 


