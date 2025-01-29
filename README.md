# Automated Rating Curve (ARC) Generator

ARC is an Python-based code that encompasses much of the functionality of the hydraulic modeling software AutoRoute. ARC inputs a series of raster datasets and outputs rating curves that describe the relationship between streamflow and water surface elevation.

The resulting rating curves can be paired with a flood inundation mapping software to produce flood inundation maps.

## Project status
This project is under active development.

## Getting started
Below are step-by-step instructions for setting up the ARC tool.

1. If you haven't already, Download [Miniconda](https://docs.anaconda.com/miniconda/miniconda-install/).
2. Clone this repository.
   - In the upper right hand corner of this screen, left-click the green <> Code button and copy the text.
   - If your on a PC, open Git Bash on your local machine or if you have a Linux or Mac open a terminal with Git installed within it.
   - Run the following command in Git Bash or Terminal:
```bash
git clone https://github.com/MikeFHS/automated-rating-curve
```
3. Open the "Anaconda Prompt" on PC or remain in your terminal in Mac or Linux, and create you Conda environment.

4. Navigate to the location where you cloned the ARC repository.
```bash
cd automated-rating-curve
```
5. Create the Conda environment using the following command. Alternatively, if you have an existing conda environment you would like integrate ARC within, take a look at it [dependencies](https://github.com/MikeFHS/automated-rating-curve/blob/main/environment.yaml) and make sure you environment has these libraries.
```bash
conda env create -f environment.yaml
```
6. Activate the Conda environment using the following command.
```bash
conda activate arc
```
7. Install ARC within your Conda environment by navigating to the local ARC instance on your machine from within "Anaconda Prompt" on PC or terminal in Mac or Linux machine, and run the following command.
```bash
pip install .
```
8. Your ARC environment should be ready to roll!

You may run `arc -h` to get help using ARC.

## Running ARC simulations
Our [Wiki](https://github.com/MikeFHS/automated-rating-curve/wiki) provides an in-depth, step-by-step guide to running ARC and provides some context for how ARC works. 

## Authors and acknowledgment
Mike Follum has been the lead for this project since the code was in it's AutoRoute days. Other contributors include Ahmad Tavakoly, Alan Snow, Joseph Gutenson, Drew Loney, and Ricky Rosas.

## License
This software has been released under a [GNU General Public Licence](https://github.com/MikeFHS/automated-rating-curve/blob/main/license.txt). 


