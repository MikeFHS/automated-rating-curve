# Automated Rating Curve (ARC) Generator

MOST RECENT UPDATE: 
```
Merge pull request #6 from MikeFHS/fixing_geojson_generation
```

ARC is an Python-based code that encompasses much of the functionality of the hydraulic modeling software AutoRoute. ARC inputs a series of raster datasets and outputs rating curves that describe the relationship between streamflow and water surface elevation.

The resulting rating curves can be paired with a flood inundation mapping software to produce flood inundation maps.

## Project status
This project is under active development.

## Getting started
Below are step-by-step instructions for setting up the ARC tool.

1. Clone this repository.
   - In the upper right hand corner of this screen, left-click the green <> Code button and copy the text.
2. Download [Miniconda](https://docs.anaconda.com/miniconda/miniconda-install/).
3. Open the "Anaconda Prompt" on PC or your terminal in Mac or Linux, and create you Conda environment.
   - To activate the "ARC" environment open an Anaconda Command Prompt, navigate to where you've clone the ARC repository on your local machine, and type "conda env create -f environment_ARC.yml"
4. Your ARC environment should be ready to roll!

Our [Wiki](https://github.com/MikeFHS/automated-rating-curve/wiki) provides an in-depth, step-by-step guide to running ARC. 

## Running ARC simulations
Step-by-step instructions for running ARC simulations can be found in the Wiki page at the top of this page.

## Authors and acknowledgment
Mike Follum has been the lead for this project since the code was in it's AutoRoute days. Other contributors include Ahmad Tavakoly, Alan Snow, Joseph Gutenson and Drew Loney.

## License
This software has been released under a [GNU General Public Licence](https://github.com/MikeFHS/automated-rating-curve/blob/main/license.txt). 


