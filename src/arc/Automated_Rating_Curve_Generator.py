
"""
Written initially by Mike Follum with Follum Hydrologic Solutions, LLC.
Program simply creates depth, velocity, and top-width information for each stream cell in a domain.

"""

import sys
import os
import math
import warnings

import tqdm
import numpy as np
import pandas as pd
from datetime import datetime
from scipy.optimize import curve_fit, OptimizeWarning, brentq
from scipy.signal import savgol_filter
from osgeo import gdal
from numba import njit
from numba.core.errors import TypingError

from arc import LOG

warnings.filterwarnings("ignore", category=OptimizeWarning)
gdal.UseExceptions()

@njit(cache=True)
def safe_signs_differ(fa, fb, tol=1e-10):

    safe_signs = False

    # Rounds small floating point noise and checks for real sign difference
    fa = np.round(fa, 5)
    fb = np.round(fb, 5)

    if fa == 0 or fb == 0:
        safe_signs = False
    elif fa * fb < 0:
        safe_signs = True
    else:
        safe_signs = False


    return safe_signs

def format_array(da_array: np.ndarray, s_format: str):
    """
    Formats a string. Helper function to allow multidimensional formats

    Parameters
    ----------
    da_array: np.array
        Array to be formatted
    s_format: str
        Specifies the format of the output

    Returns
    -------
    s_formatted_output: str
        Formatted output as a string

    """

    # Format the output
    s_formatted_output = "[" + ",".join(s_format.format(x) for x in da_array) + "]"

    # Return to the calling function
    return s_formatted_output


def array_to_string(da_array: np.ndarray, i_decimal_places: int = 6):
    """
    Convert a NumPy array to a formatted string with consistent spacing and no new lines.

    Parameters
    ----------
    da_array: np.ndarray
        Array to conver to a string
    i_decimal_places: int
        Number oof decimal places to return in the string

    Returns
    -------

    """

    # Define the format string
    s_format = f"{{:.{i_decimal_places}f}}"

    # Format the string based on the dimensionality of the array
    if da_array.ndim == 1:
        # Array is one-dimensional
        s_output = format_array(da_array, s_format)

    elif da_array.ndim == 2:
        # Array is two-dimensional
        s_output = "[" + ",".join(format_array(row, s_format) for row in da_array) + "]"

    else:
        # Array is ill formated. Throw an error.
        raise ValueError("Only 1D and 2D arrays are supported")

    # Return to the calling function
    return s_output


# Power function equation
@njit(cache=True)
def power_func(d_value: np.ndarray, d_coefficient: float, d_power: float):
    """
    Define a general power function that can be used for fitting

    Parameters
    ----------
    d_value: float
        Current x value
    d_coefficient: float
        Coefficient at the lead of the power function
    d_power: float
        Power value

    Returns
    -------
    d_power_value: float
        Calculated value

    """

    # Calculate the power
    d_power_value = d_coefficient * (d_value ** d_power)

    # Return to the calling function
    return d_power_value


def linear_regression_power_function(da_x_input: np.ndarray, da_y_input: np.ndarray, init_guess: list = [1.0, 1.0]):
    """
    Performs a curve fit to a power function

    Parameters
    ----------
    da_x_input: np.ndarray
        X values input to the fit
    da_y_input: np.ndarray
        Y values input to the fit

    Returns
    -------
    d_coefficient: float
         Coeffient of the fit
    d_power: float
        Power of the fit
    d_R2: float
        Goodness of fit

    """
    # Default values in case of failure
    d_coefficient, d_power, d_R2 = -9999.9, -9999.9, -9999.9

    # Attempt to calculate the fit
    try:
        (d_coefficient, d_power), dm_pcov = curve_fit(power_func, da_x_input, da_y_input, p0=init_guess)
        # Calculate R², this is never used so don't bother
        # da_y_pred = power_func(da_x_input, d_coefficient, d_power)
        # mean_y = np.mean(da_y_input)
        # ss_tot = np.dot(da_y_input - mean_y, da_y_input - mean_y)
        # ss_res = np.dot(da_y_input - da_y_pred, da_y_input - da_y_pred)
        # d_R2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else -9999.9
    except TypingError as e:
        LOG.error(e)
    except RuntimeError as e:
        pass

    # Return to the calling function
    return d_coefficient, d_power, d_R2


def write_output_raster(s_output_filename: str, dm_raster_data: np.ndarray, i_number_of_columns: int, i_number_of_rows: int, l_dem_geotransform: list, s_dem_projection: str,
                        s_file_format: str, s_output_type: str):
    """
    Writes dataset to the output raster file specified

    Parameters
    ----------
    s_output_filename: str
        Output filename
    dm_raster_data: ndarray
        Data to be written to disk
    i_number_of_columns: int
        Number of columns in the dataset
    i_number_of_rows: int
        Number of rows in the dataset
    l_dem_geotransform: list
        The geotransform information for the file
    s_dem_projection: str
        The projection of the file
    s_file_format: str
        Output format for the file
    s_output_type: str
        Output data type

    Returns
    -------
    None. Outputs are written to disk

    """

    # Set the filename to write to
    o_driver = gdal.GetDriverByName(s_file_format)  # Typically will be a GeoTIFF "GTiff"
    
    # Construct the file with the appropriate data shape
    # o_output_file = o_driver.Create(s_output_filename, xsize=i_number_of_columns, ysize=i_number_of_rows, bands=1, eType=s_output_type)
    o_output_file = o_driver.Create(s_output_filename, xsize=i_number_of_columns, ysize=i_number_of_rows, bands=1, eType=s_output_type, options=['COMPRESS=LZW', "PREDICTOR=2"])

    # Set the geotransform
    o_output_file.SetGeoTransform(l_dem_geotransform)
    
    # Set the spatial reference
    o_output_file.SetProjection(s_dem_projection)
    
    # Write the data to the file
    o_output_file.GetRasterBand(1).WriteArray(dm_raster_data)
    
    # Once we're done, close properly the dataset
    o_output_file = None


def read_raster_gdal(s_input_filename: str):
    """
    Reads a raster file from disk into memory

    Parameters
    ----------
    s_input_filename: str
        Input filename

    Returns
    -------
    dm_raster_array: ndarray
        Contains the data matrix from the input file
    i_number_of_columns: int
        Number of columns in the dataset
    i_number_of_rows: int
        Number of rows in the dataset
    d_cell_size: float
        Size of each cell
    d_y_lower_left: float
        y coordinate of the lower left corner of the dataset
    d_y_upper_right: float
        y coordinate of the upper right corner of the dataset
    d_x_lower_left: float
        x coordinate of the lower left corner of the dataset
    d_x_upper_right: float
        x coordinate of the upper right corner of the dataset
    d_latitude: float
        Latitude of the dataset
    l_geotransform: list
        Geotransform information from the dataset
    s_raster_projection: str
        Projection information from the dataset

    """

    # Check that the file exists to open
    if os.path.isfile(s_input_filename) == False:
        LOG.info('Cannot Find Raster ' + s_input_filename)

    # Attempt to open the dataset
    try:
        o_dataset = gdal.Open(s_input_filename, gdal.GA_ReadOnly)
    except RuntimeError:
        sys.exit(" ERROR: Field Raster File cannot be read!")

    # Retrieve dimensions of cell size and cell count then close DEM dataset
    l_geotransform = o_dataset.GetGeoTransform()

    # Continue importing geospatial information
    o_band = o_dataset.GetRasterBand(1)
    dm_raster_array = o_band.ReadAsArray()

    # Read the size of the band object
    i_number_of_columns = o_band.XSize
    i_number_of_rows = o_band.YSize

    # Close the band object
    o_band = None

    # Extract information from the geotransform
    d_cell_size = l_geotransform[1]

    d_y_lower_left = l_geotransform[3] - i_number_of_rows * np.fabs(l_geotransform[5])
    d_y_upper_right = l_geotransform[3]
    d_x_lower_left = l_geotransform[0]
    d_x_upper_right = d_x_lower_left + i_number_of_columns * l_geotransform[1]

    d_latitude = np.fabs((d_y_lower_left + d_y_upper_right) / 2.0)
    s_raster_projection = o_dataset.GetProjectionRef()

    # Close the dataset
    o_dataset = None

    # Write metdata information to the console
    LOG.info('Spatial Data for Raster File:')
    LOG.info('   ncols = ' + str(i_number_of_columns))
    LOG.info('   nrows = ' + str(i_number_of_rows))
    LOG.info('   cellsize = ' + str(d_cell_size))
    LOG.info('   yll = ' + str(d_y_lower_left))
    LOG.info('   yur = ' + str(d_y_upper_right))
    LOG.info('   xll = ' + str(d_x_lower_left))
    LOG.info('   xur = ' + str(d_x_upper_right))

    # Return dataset information to the calling function
    return dm_raster_array, i_number_of_columns, i_number_of_rows, d_cell_size, d_y_lower_left, d_y_upper_right, d_x_lower_left, d_x_upper_right, d_latitude, l_geotransform, s_raster_projection


def get_parameter_name(sl_lines: list[str], s_target: str):
    """
    Gets parameter values from a list of strings, assuming that the file is tab delimited and the first characters are the target string.
    The second column is returned as the target value.

    Parameters
    ----------
    sl_lines: list
        Lines to test for target string
    s_target: str
        Target string to match at the start for each line

    Returns
    -------
    d_return_value: float
        Returned value. This may be other variable types but is assumed to be a double for typing.

    """

    # Set the default value of the target
    d_return_value = ''

    # Loop over entries in the list
    for line in sl_lines:
        # Split the line and strip special characters
        ls = line.strip().split('\t')

        # Check if the first entry is the target string
        if ls[0] == s_target:
            # Override the initial the default value
            d_return_value = 1

            # String is found. Process the rest of the line
            if len(ls) > 1 and len(ls[1]) > 0 :
                # More information is available to parse
                d_return_value = ls[1]

    # Log the value to the console
    if d_return_value != '':
        LOG.info(f'  {s_target} is set to {d_return_value}')

    else:
       LOG.info(f'  Could not find {s_target}')

    # Return value to the calling function
    return d_return_value


def read_main_input_file(s_mif_name: str, args: dict):
    """

    Parameters
    ----------
    s_mif_name: str
        Path to the input file
    args: dict
        Dictionary of arguments passed to the function. This is used to set the global variables in the function

    Returns
    -------
    None. Global values are used to set values outside of the function for the full program

    """

    ### Open and read the input file ###
    # Open the file
    if s_mif_name:
        with open(s_mif_name, 'r') as o_input_file:
            sl_lines = o_input_file.readlines()
    else:
        # Convert arg dict to a list of lines
        sl_lines = []
        for key, value in args.items():
            sl_lines.append(f"{key}\t{value}\n")


    ### Process the parameters ###
    # Find the path to the DEM file
    global s_input_dem_path
    s_input_dem_path = get_parameter_name(sl_lines,  'DEM_File')

    # Find the path to the stream file
    global s_input_stream_path
    s_input_stream_path = get_parameter_name(sl_lines,  'Stream_File')

    # Find the path to the land use raster file
    global s_input_land_use_path
    s_input_land_use_path = get_parameter_name(sl_lines,  'LU_Raster_SameRes')

    # Find the path to the mannings n file
    global s_input_mannings_path
    s_input_mannings_path = get_parameter_name(sl_lines,  'LU_Manning_n')

    # Find the path to the input flow file
    global s_input_flow_file_path
    s_input_flow_file_path = get_parameter_name(sl_lines,  'Flow_File')

    # Find the method used to number the streams
    global s_flow_file_id
    s_flow_file_id = get_parameter_name(sl_lines,  'Flow_File_ID')

    # Find the flow method
    global s_flow_file_baseflow
    s_flow_file_baseflow = get_parameter_name(sl_lines,  'Flow_File_BF')

    # Find the field that determines the maximum flow
    global s_flow_file_qmax
    s_flow_file_qmax = get_parameter_name(sl_lines,  'Flow_File_QMax')

    # Find the spatial units
    global s_spatial_units
    s_spatial_units = get_parameter_name(sl_lines,  'Spatial_Units')

    if s_spatial_units == '':
        # Assume degree if not specified in the input efile
        s_spatial_units = 'deg'

    # Find the x section distance
    global d_x_section_distance
    d_x_section_distance = get_parameter_name(sl_lines,  'X_Section_Dist')

    if d_x_section_distance == '':
        # Value does not occur in the input file. Assume a reasonable value
        d_x_section_distance = 5000.0

    d_x_section_distance = float(d_x_section_distance)

    # Find the path to the output velocity, depth, and top width file
    global s_output_vdt_database
    s_output_vdt_database = get_parameter_name(sl_lines,  'Print_VDT_Database')

    # Find the path to the output area and wetted-perimeter file
    global s_output_ap_database
    s_output_ap_database = get_parameter_name(sl_lines,  'Print_AP_Database')
    
    global s_output_curve_file
    s_output_curve_file = get_parameter_name(sl_lines,  'Print_Curve_File')
    
    # Find the path to the output metdata file
    global s_output_meta_file
    s_output_meta_file = get_parameter_name(sl_lines,  'Meta_File')

    # Find degree manipulation attribute
    global d_degree_manipulation
    d_degree_manipulation = get_parameter_name(sl_lines,  'Degree_Manip')

    if d_degree_manipulation == '':
        # Value does not occur in the input file. Assume a reasonable value
        d_degree_manipulation = 1.1

    d_degree_manipulation = float(d_degree_manipulation)

    # Find the degree interval attribute
    global d_degree_interval
    d_degree_interval = get_parameter_name(sl_lines,  'Degree_Interval')

    if d_degree_interval == '':
        # Value does not occur in the input file. Assume a reasonable value
        d_degree_interval = 1.0

    d_degree_interval = float(d_degree_interval)

    # Find the low spot range attribute
    global i_low_spot_range
    i_low_spot_range = get_parameter_name(sl_lines,  'Low_Spot_Range')

    if i_low_spot_range == '':
        # Value does not occur in the input file. Assume a reasonable value
        i_low_spot_range = 0

    i_low_spot_range = int(i_low_spot_range)

    # Find the general direction distance attribute
    global i_general_direction_distance
    i_general_direction_distance= get_parameter_name(sl_lines,  'Gen_Dir_Dist')

    if i_general_direction_distance == '':
        # Value does not occur in the input file. Assume a reasonable value
        i_general_direction_distance = 10

    i_general_direction_distance = int(i_general_direction_distance)

    # Find the general slope distance attribute
    global i_general_slope_distance
    i_general_slope_distance = get_parameter_name(sl_lines,  'Gen_Slope_Dist')

    if i_general_slope_distance == '':
        # Value does not occur in the input file. Assume a reasonable value
        i_general_slope_distance = 0

    i_general_slope_distance = int(i_general_slope_distance)

    # Find the bathymetry trapezoid height attribute
    global d_bathymetry_trapzoid_height
    d_bathymetry_trapzoid_height = get_parameter_name(sl_lines,  'Bathy_Trap_H')

    if d_bathymetry_trapzoid_height == '':
        # Value does not occur in the input file. Assume a reasonable value
        d_bathymetry_trapzoid_height = 0.2

    d_bathymetry_trapzoid_height = float(d_bathymetry_trapzoid_height)

    # Find the True/False variable to use the bank elevations to calculate the depth of the bathymetry estimate
    global b_bathy_use_banks
    b_bathy_use_banks = get_parameter_name(sl_lines,  'Bathy_Use_Banks')
    if "True" in b_bathy_use_banks:
        b_bathy_use_banks = True
    elif "False" in b_bathy_use_banks or b_bathy_use_banks == '':
        b_bathy_use_banks = False

    # Find the path to the output bathymetry file
    global s_output_bathymetry_path
    s_output_bathymetry_path = get_parameter_name(sl_lines,  'AROutBATHY')

    if s_output_bathymetry_path == '':
        # Value does not occur in the input file. Assume a reasonable value
        s_output_bathymetry_path = get_parameter_name(sl_lines,  'BATHY_Out_File')

    # Find the path to the output depth file
    global s_output_depth
    s_output_depth = get_parameter_name(sl_lines,  'AROutDEPTH')

    # Find the path to the output flood file
    global s_output_flood
    s_output_flood = get_parameter_name(sl_lines,  'AROutFLOOD')

    # Find the path to the output cross-section file (JLG added this to recalculate top-width and velocity)
    global s_xs_output_file
    s_xs_output_file = get_parameter_name(sl_lines,  'XS_Out_File')
    
    global i_lc_water_value
    i_lc_water_value = get_parameter_name(sl_lines,  'LC_Water_Value')
    if i_lc_water_value =='': 
        #Value is defaulted to the water value in the ESA land cover dataset
        i_lc_water_value = 80

    # These are the number of increments of water surface elevation that we will use to construct the VDT database and the curve file
    global i_number_of_increments
    i_number_of_increments = get_parameter_name(sl_lines,  'VDT_Database_NumIterations')
    if i_number_of_increments=='':
        i_number_of_increments = 15
    i_number_of_increments  = int(i_number_of_increments)
    
    #Default is to find the banks of the river based on flat water in the DEM.  However, you can also find the banks using the water surface (please also set i_lc_water_value)
    global b_FindBanksBasedOnLandCover
    b_FindBanksBasedOnLandCover = get_parameter_name(sl_lines,  'FindBanksBasedOnLandCover')
    if "True" in b_FindBanksBasedOnLandCover:
        b_FindBanksBasedOnLandCover = True
    elif "False" in b_FindBanksBasedOnLandCover or b_FindBanksBasedOnLandCover == '':
        b_FindBanksBasedOnLandCover = False
    
    # Find the True/False variable to use the bank elevations to calculate the depth of the bathymetry estimate
    global b_reach_average_curve_file
    b_reach_average_curve_file = get_parameter_name(sl_lines,  'Reach_Average_Curve_File')
    if "True" in b_reach_average_curve_file:
        b_reach_average_curve_file = True
    elif "False" in b_reach_average_curve_file or b_reach_average_curve_file == '':
        b_reach_average_curve_file = False
    if len(s_output_curve_file)<1:
        b_reach_average_curve_file = False   #Has to be false because there is no curve file to be used.

def convert_cell_size(d_dem_cell_size: float, d_dem_lower_left: float, d_dem_upper_right: float):
    """
    Determines the x and y cell sizes based on the geographic location

    Parameters
    ----------
    d_dem_cell_size: float
        Size of the dem cell
    d_dem_lower_left: float
        Lower left corner value
    d_dem_upper_right: float
        Upper right corner value

    Returns
    -------
    d_x_cell_size: float
        Resolution of the cells in the x direction
    d_y_cell_size: float
        Resolution of the cells in teh y direction
    d_projection_conversion_factor: float
        Factor to convert to the projection

    """

    ### Set default values ###
    d_x_cell_size = d_dem_cell_size
    d_y_cell_size = d_dem_cell_size
    d_projection_conversion_factor = 1

    ### Get the cell size ###
    d_lat = np.fabs((d_dem_lower_left + d_dem_upper_right) / 2)

    ### Determine if conversion is needed
    if d_dem_cell_size > 0.5:
        # This indicates that the DEM is projected, so no need to convert from geographic into projected.
        d_x_cell_size = d_dem_cell_size
        d_y_cell_size = d_dem_cell_size
        d_projection_conversion_factor = 1

    else:
        # Reprojection from geographic coordinates is needed
        assert d_lat > 1e-16, "Please use lat and long values greater than or equal to 0."

        # Determine the latitude range for the model
        if d_lat >= 0 and d_lat <= 10:
            d_lat_up = 110.61
            d_lat_down = 110.57
            d_lon_up = 109.64
            d_lon_down = 111.32
            d_lat_base = 0.0

        elif d_lat > 10 and d_lat <= 20:
            d_lat_up = 110.7
            d_lat_down = 110.61
            d_lon_up = 104.64
            d_lon_down = 109.64
            d_lat_base = 10.0

        elif d_lat > 20 and d_lat <= 30:
            d_lat_up = 110.85
            d_lat_down = 110.7
            d_lon_up = 96.49
            d_lon_down = 104.65
            d_lat_base = 20.0

        elif d_lat > 30 and d_lat <= 40:
            d_lat_up = 111.03
            d_lat_down = 110.85
            d_lon_up = 85.39
            d_lon_down = 96.49
            d_lat_base = 30.0

        elif d_lat > 40 and d_lat <= 50:
            d_lat_up = 111.23
            d_lat_down = 111.03
            d_lon_up = 71.70
            d_lon_down = 85.39
            d_lat_base = 40.0

        elif d_lat > 50 and d_lat <= 60:
            d_lat_up = 111.41
            d_lat_down = 111.23
            d_lon_up = 55.80
            d_lon_down = 71.70
            d_lat_base = 50.0

        elif d_lat > 60 and d_lat <= 70:
            d_lat_up = 111.56
            d_lat_down = 111.41
            d_lon_up = 38.19
            d_lon_down = 55.80
            d_lat_base = 60.0

        elif d_lat > 70 and d_lat <= 80:
            d_lat_up = 111.66
            d_lat_down = 111.56
            d_lon_up = 19.39
            d_lon_down = 38.19
            d_lat_base = 70.0

        elif d_lat > 80 and d_lat <= 90:
            d_lat_up = 111.69
            d_lat_down = 111.66
            d_lon_up = 0.0
            d_lon_down = 19.39
            d_lat_base = 80.0

        else:
            raise AttributeError('Please use legitimate (0-90) lat and long values.')

        ## Convert the latitude ##
        d_lat_conv = d_lat_down + (d_lat_up - d_lat_down) * (d_lat - d_lat_base) / 10
        d_y_cell_size = d_dem_cell_size * d_lat_conv * 1000.0  # Converts from degrees to m

        ## Longitude Conversion ##
        d_lon_conv = d_lon_down + (d_lon_up - d_lon_down) * (d_lat - d_lat_base) / 10
        d_x_cell_size = d_dem_cell_size * d_lon_conv * 1000.0  # Converts from degrees to m

        ## Make sure the values are in bounds ##
        if d_lat_conv < d_lat_down or d_lat_conv > d_lat_up or d_lon_conv < d_lon_up or d_lon_conv > d_lon_down:
            raise ArithmeticError("Problem in conversion from geographic to projected coordinates")

        ## Calculate the conversion factor ##
        d_projection_conversion_factor = 1000.0 * (d_lat_conv + d_lon_conv) / 2.0

    # Return to the calling function
    return d_x_cell_size, d_y_cell_size, d_projection_conversion_factor


def read_flow_file(s_flow_file_name: str, s_flow_id: str, s_flow_baseflow: str, s_flow_qmax: str):
    """

    Parameters
    ----------
    s_flow_file_name
    s_flow_id
    s_flow_baseflow
    s_flow_qmax

    Returns
    -------

    """
    df = pd.read_csv(s_flow_file_name)
    da_comid = df[s_flow_id].values
    da_base_flow = df[s_flow_baseflow].values
    da_flow_maximum = df[s_flow_qmax].values

    return da_comid, da_base_flow, da_flow_maximum

@njit(cache=True)
def get_stream_slope_information(i_row: int, i_column: int, dm_dem: np.ndarray, im_streams: np.ndarray, d_dx: float, d_dy: float):
    """
    Calculates the stream slope using the following process:

        1.) Find all stream cells within the Gen_Slope_Dist that have the same stream id value
        2.) Look at the slope of each of the stream cells.
        3.) Average the slopes to get the overall slope we use in the model.

    Guaranteed to be >= 0.0002

    Parameters
    ----------
    i_row: int
        Target cell row index
    i_column: int
        Target cell column index
    dm_dem: ndarray
        Elevation raster
    im_streams: ndarray
        Stream raster
    d_dx: float
        Cell resolution in the x direction
    d_dy: float
        Cell resolution in the y direction

    Returns
    -------
    d_stream_slope: float
        Average slope from the stream cells in the specified search box

    """

    # Initialize a default stream flow
    d_stream_slope = 0.0

    # Get the elevation of the cell
    d_cell_of_interest = dm_dem[i_row, i_column]

    # Get the stream id of the cell
    i_cell_value = im_streams[i_row, i_column]

    # Slice a box around both the stream and elevations
    im_stream_box = im_streams[i_row - i_general_slope_distance:i_row + i_general_slope_distance, i_column - i_general_slope_distance:i_column + i_general_slope_distance]
    dm_elevation_box = dm_dem[i_row - i_general_slope_distance:i_row + i_general_slope_distance, i_column - i_general_slope_distance:i_column + i_general_slope_distance]

    # Get the indices of all locations of the stream id within the box
    ia_matching_row_indices, ia_matching_column_indices = np.where(im_stream_box == i_cell_value)

    # Find the slope if there are stream cells
    if len(ia_matching_row_indices) > 0:
        # da_matching_elevations = dm_elevation_box[ia_matching_row_indices, ia_matching_column_indices]
        da_matching_elevations = dm_elevation_box.ravel()[ia_matching_row_indices * dm_elevation_box.shape[1] + ia_matching_column_indices]
        # The Gen_Slope_Dist is the row/col for the cell of interest within the subsample box
        # Distance between the cell of interest and every cell with a similar stream id
        dz_list = np.sqrt(np.square((ia_matching_row_indices - i_general_slope_distance) * d_dy) + np.square((ia_matching_column_indices - i_general_slope_distance) * d_dx))

        for x in range(len(ia_matching_row_indices)):
            if dz_list[x] > 0.0:
                d_stream_slope = d_stream_slope + abs(d_cell_of_interest-da_matching_elevations[x]) / dz_list[x]

        # Average across the cells
        if len(ia_matching_row_indices)>1:
            d_stream_slope = d_stream_slope / (len(ia_matching_row_indices)-1)  #Add the minus one because the cell of interest was in the list
        
        
        #if ia_matching_row_indices has less than 2 values then the slope will be set to the default value

    # if slope is less than the threshold, reset it
    if d_stream_slope < 0.0002:
        d_stream_slope = 0.0002

    # Return the slope to the calling function
    return d_stream_slope

@njit(cache=True)
def linear_regression_plus_angle_njit(x, y):
    """
    Perform linear regression to find the slope and intercept of the best-fit line.

    Args:
        x (np.ndarray): Independent variable.
        y (np.ndarray): Dependent variable.

    Returns:
        tuple: (slope, intercept) of the best-fit line.
    """
    n = len(x)
    
    # Compute means of x and y
    x_mean = np.sum(x) / n
    y_mean = np.sum(y) / n

    # Compute the numerator and denominator for the slope
    numerator = 0.0
    denominator = 0.0
    for i in range(n):
        numerator += (x[i] - x_mean) * (y[i] - y_mean)
        denominator += (x[i] - x_mean) ** 2

    #If this occurs it means the line is straight up
    if denominator<=0.000001:
        return -1, -1, np.pi
    #If this occurs it means the line is flat
    if abs(numerator)<=0.000001:
        return -1, -1, 0.0

    # Calculate slope (m) and intercept (b)
    slope = numerator / denominator
    intercept = y_mean - slope * x_mean

    # Convert slope to angle in radians (normalized to be between 0 and 2pi)
    d_stream_direction = np.arctan(slope) % (2 * np.pi)

    return slope, intercept, d_stream_direction

@njit(cache=True)
def polyfit_linear_plus_angle(x, y):
    """
    Perform linear regression (degree 1 polynomial fitting) with Numba.
    
    Args:
        x (np.ndarray): Array of x values.
        y (np.ndarray): Array of y values.
        
    Returns:
        (float, float): Slope and intercept of the best-fit line.
    """
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    
    # Compute slope (m) and intercept (b)
    numerator = np.sum((x - x_mean) * (y - y_mean))
    denominator = np.sum((x - x_mean) ** 2)

    
    #If this occurs it means the line is straight up
    if denominator<=0.000001 or abs(numerator)<=0.000001:
        DX = np.max(x) - np.min(x)
        DY = np.max(y) - np.min(y)
        # Even though the regression cant find the slope, it is dominated in the X direction, meaning angle of zero
        if DX>DY:
            return -1, -1, 0
        #The change in Y direction is dominant, meaning a stream angle of pi
        else:
            return -1, -1, np.pi/2.0

    slope = numerator / denominator
    intercept = y_mean - slope * x_mean

    # Convert slope to angle in radians (normalized to be between 0 and 2pi)
    d_stream_direction = np.arctan(slope) % (2 * np.pi)
    
    return slope, intercept, d_stream_direction

@njit(cache=True)
def get_stream_direction_information(i_row: int, i_column: int, im_streams: np.ndarray, d_dx: float, d_dy: float):
    """
    Finds the general direction of the stream following the process:

        1.) Find all stream cells within the general_direction_distance that have the same stream id value
        2.) Assume there are 4 quadrants:
                Q3 | Q4      r<0 c<0  |  r<0 c>0
                Q2 | Q1      r>0 c<0  |  r>0 c>0
        3.) Calculate the distance from the cell of interest to each of the stream cells idendified.
        4.) Create a weight that provides a higher weight to the cells that are farther away
        5.) Calculate the Stream Direction based on the Unit Circle inverted around the x axis (this is done because rows increase downward)
        6.) The stream direction needs to be betweeen 0 and pi, so adjust directions between pi and 2pi to be between 0 and pi

    Parameters
    ----------
    i_row: int
        Row cell index
    i_column: int
        Column cell index
    im_streams: ndarray
        Stream raster
    d_dx: float
        Cell resolution in the x direction
    d_dy: float
        Cell resolution in the y direction

    Returns
    -------
    d_stream_direction: float
        Direction of the stream
    d_xs_direction float
        Direction of the cross section

    """

    # Initialize default values
    d_stream_direction = 0.0
    d_xs_direction = 0.0

    # Get the values from the stream raster
    i_cell_value = im_streams[i_row,i_column]

    # Slice the search box from the stream raster
    im_stream_box = im_streams[i_row - i_general_direction_distance:i_row + i_general_direction_distance, i_column - i_general_direction_distance:i_column + i_general_direction_distance]
    
    # The Gen_Dir_Dist is the row/col for the cell of interest within the subsample box
    ia_matching_row_indices, ia_matching_column_indices = np.where(im_stream_box == i_cell_value)

    # Find the direction if there are stream cells
    if len(ia_matching_row_indices) > 1:  #Need at least a few cells to determine direction

        # METHOD 1 - Calculate the angle based on all of the stream cells in the search box and do distance weighting
        '''
        # Adjust the cell indices with the general direction distance
        ia_matching_row_indices = ia_matching_row_indices - i_general_direction_distance
        ia_matching_column_indices = ia_matching_column_indices - i_general_direction_distance

        # Calculate the distance between the cell of interest and every cell with a similar stream id
        da_dz_list = np.sqrt(np.square((ia_matching_row_indices) * d_dy) + np.square((ia_matching_column_indices) * d_dx))
        da_dz_list = da_dz_list / max(da_dz_list)
        
        # Calculate the angle to the cells within the search box
        da_atanvals = np.arctan2(ia_matching_row_indices, ia_matching_column_indices)
        
        
        # Account for the angle sign when aggregating the distance
        for x in range(len(ia_matching_row_indices)):
            if da_dz_list[x] > 0.0:
                if da_atanvals[x] > math.pi:
                    d_stream_direction = d_stream_direction + (da_atanvals[x] - math.pi) * da_dz_list[x]
                elif da_atanvals[x] < 0.0:
                    d_stream_direction = d_stream_direction + (da_atanvals[x] + math.pi) * da_dz_list[x]
                else:
                    d_stream_direction = d_stream_direction + da_atanvals[x] * da_dz_list[x]
        d_stream_direction = d_stream_direction / sum(da_dz_list)
        '''
        
        # METHOD 2 - Calculate the angle based on the streamcells that are the furthest in the search box
        '''
        # Adjust the cell indices with the general direction distance
        ia_matching_row_indices = ia_matching_row_indices - i_general_direction_distance
        ia_matching_column_indices = ia_matching_column_indices - i_general_direction_distance

        # Account for the angle sign when aggregating the distance
        # Calculate the distance between the cell of interest and every cell with a similar stream id
        da_dz_list = np.sqrt(np.square((ia_matching_row_indices) * d_dy) + np.square((ia_matching_column_indices) * d_dx))
        
        # Calculate the angle to the cells within the search box
        #da_atanvals = np.arctan2( np.multiply(ia_matching_row_indices, d_dy), np.multiply(ia_matching_column_indices, d_dx) )
        #da_atanvals = np.arctan2(ia_matching_row_indices, ia_matching_column_indices)
        
        #Finds the cell within the scan box that is the farthest away the stream cell of interest.
        x = np.argmax(da_dz_list)
        #print(x)
        
        da_atanvals_single = np.arctan2(ia_matching_row_indices[x], ia_matching_column_indices[x])
        if da_atanvals_single >= math.pi:
            d_stream_direction = (da_atanvals_single - math.pi)
        elif da_atanvals_single < 0.0:
            d_stream_direction = (da_atanvals_single + math.pi)
        else:
            d_stream_direction = da_atanvals_single
        '''
        
        
        
        # METHOD 3 - Calculate the angle to each stream cell around and then take the median
        '''
        # Adjust the cell indices with the general direction distance
        ia_matching_row_indices = ia_matching_row_indices - i_general_direction_distance
        ia_matching_column_indices = ia_matching_column_indices - i_general_direction_distance

        # Calculate the angle to the cells within the search box
        #da_atanvals = np.arctan2(ia_matching_row_indices, ia_matching_column_indices)
        da_atanvals = np.arctan2( np.multiply(ia_matching_row_indices, d_dy), np.multiply(ia_matching_column_indices, d_dx) )
        
        # Calculate the distance between the cell of interest and every cell with a similar stream id
        da_dz_list = np.sqrt(np.square((ia_matching_row_indices) * d_dy) + np.square((ia_matching_column_indices) * d_dx))
        zone = np.zeros(len(da_dz_list))
        
        for x in range(len(ia_matching_row_indices)):
            if da_dz_list[x] <= 0.0:
                da_atanvals[x] = np.nan
            elif da_atanvals[x]>math.pi:
                da_atanvals[x] = da_atanvals[x] - math.pi
            elif da_atanvals[x]<0:
                da_atanvals[x] = da_atanvals[x] + math.pi
            
            if da_atanvals[x] >= (3*math.pi/4):
                zone[x]=4
            elif da_atanvals[x] >= (2*math.pi/4):
                zone[x]=3
            elif da_atanvals[x] >= (1*math.pi/4):
                zone[x]=2
            else:
                zone[x]=1
        
        n1 = int((zone==1).sum())
        n2 = int((zone==2).sum())
        n3 = int((zone==3).sum())
        n4 = int((zone==4).sum())
        max_n = max(n1, n2, n3, n4)
        
        a1=np.nan
        a2=np.nan
        a3=np.nan
        a4=np.nan
        da_atanvals_single = 0.0
        
        if n4 == max_n:
            a4 = np.nanmean(da_atanvals[zone==4])
            da_atanvals_single = a4
            #if n1>0:
            #    a1 = np.nanmean(da_atanvals[zone==1])
            #    a1 = a1 + math.pi
            #    da_atanvals_single = (a4*n4 + a1*n1) / (n4+n1)
        elif n3 == max_n:
            a3 = np.nanmean(da_atanvals[zone==3])
            da_atanvals_single = a3
            #if n2>0:
            #    a2 = np.nanmean(da_atanvals[zone==2])
            #    da_atanvals_single = (a3*n3 + a2*n2) / (n3+n2)
        elif n2 == max_n:
            a2 = np.nanmean(da_atanvals[zone==2])
            da_atanvals_single = a2
            #if n3>0:
            #    a3 = np.nanmean(da_atanvals[zone==3])
            #    da_atanvals_single = (a3*n3 + a2*n2) / (n3+n2)
        elif n1 == max_n:
            a1 = np.nanmean(da_atanvals[zone==1])
            da_atanvals_single = a1
            #if n4>0:
            #    a4 = np.nanmean(da_atanvals[zone==4])
            #    a4 = a4 - math.pi
            #    da_atanvals_single = (a4*n4 + a1*n1) / (n4+n1)
        
        if da_atanvals_single<0.0:
            da_atanvals_single = da_atanvals_single + math.pi
        if da_atanvals_single!=np.nan and da_atanvals_single>=0.0 and da_atanvals_single<=math.pi:
            da_atanvals_single = da_atanvals_single
        else:
            da_atanvals_single = 0.12345
        
        d_stream_direction = da_atanvals_single
        '''



        # METHOD 4 - Using precalculated angles, find which one best serves the data points in the box

        # Use numpy.polyfit for linear regression, but does not work with njit
        #    Because the ia_matching_row_indices came from a np.where() function, no need to multiply by -1 due to rows increasing downward, it is a mute point due to the np.where()
        #slope, intercept = np.polyfit(ia_matching_column_indices, ia_matching_row_indices, 1)  # Degree 1 for linear
        #d_stream_direction = np.arctan(slope) % (2 * np.pi)   # Convert slope to angle in radians (normalized to be between 0 and 2pi)

        # Uses njit compatable functions
        #    Because rows increase in the downward direction, readjust so the rows to be positive in the upward direction
        #slope, intercept, d_stream_direction = linear_regression_plus_angle_njit(ia_matching_column_indices, ia_matching_row_indices)
        slope, intercept, d_stream_direction = polyfit_linear_plus_angle(ia_matching_column_indices, ia_matching_row_indices)
        
        
        
        '''
        # Account for the angle sign when aggregating the distance
        for x in range(len(ia_matching_row_indices)):
            if da_dz_list[x] > 0.0:
                if da_atanvals[x] > math.pi:
                    da_atanvals[x] = da_atanvals[x] - math.pi
                elif da_atanvals[x] < 0.0:
                    da_atanvals[x] = da_atanvals[x] + math.pi
                
                danglediff = abs(da_atanvals_single-da_atanvals[x])
                if danglediff > (math.pi/2):
                    da_atanvals[x] = da_atanvals[x] - math.pi
                    if da_atanvals[x]<0.0:
                        da_atanvals[x] = da_atanvals[x] + 2*math.pi
        #print(da_atanvals)
        d_stream_direction = np.nanmedian(da_atanvals)
        #print(d_stream_direction)
        '''
        
        
        # Cross-Section Direction is just perpendicular to the Stream Direction
        d_xs_direction = d_stream_direction - math.pi / 2.0

        if d_xs_direction < 0.0:
            # Check that the cross section direction is reasonable
            d_xs_direction = d_xs_direction + math.pi
        
        #print('r=' + str(ia_matching_row_indices[x]) + '  c=' + str(ia_matching_column_indices[x]) + '  a=' + str(d_stream_direction*180.0/math.pi))
       
    #if int(i_cell_value) == 760748000:
        #print(da_atanvals)
    #    print(str(d_stream_direction) + '  ' + str(d_xs_direction))
    
    # Return to the calling function
    return d_stream_direction, d_xs_direction

@njit(cache=True)
def get_xs_index_values_precalculated(ia_xc_dr_index_main: np.ndarray, ia_xc_dc_index_main: np.ndarray, ia_xc_dr_index_second: np.ndarray, ia_xc_dc_index_second: np.ndarray, da_xc_main_fract: np.ndarray,
                        da_xc_second_fract: np.ndarray, d_xs_direction: np.ndarray, i_centerpoint: int, d_dx: float, d_dy: float):
    """
    ia_xc_dr_index_main, ia_xc_dc_index_main, ia_xc_dr_index_second, ia_xc_dc_index_second, da_xc_main_fract, da_xc_main_fract_int, da_xc_second_fract, da_xc_second_fract_int, d_xs_direction, i_row_cell,
                                               i_column_cell, i_center_point, dx, dy
    Calculates the distance of the stream cross section

    Parameters
    ----------
    ia_xc_dr_index_main: ndarray
        Indices of the first cross section index
    ia_xc_dc_index_main: ndarray
        Index offsets of the first cross section index
    ia_xc_dr_index_second: ndarray
        Indices of the second cross section index
    ia_xc_dc_index_second: ndarray
        Index offsets of the second cross section index
    da_xc_main_fract: ndarray: ndarray
        # todo: add
    da_xc_second_fract: ndarray
        # todo: add
    d_xs_direction: float
        Orientation of the cross section
    i_centerpoint: int
        Distance from the cell to search
    d_dx: float
        Cell resolution in the x direction
    d_dy: float
        Cell resolution in the y direction

    Returns
    -------
    d_distance_z: float
        Distance along the cross section direction

    """
    
    
    '''
    Assume there are 4 quadrants:
            Q3 | Q4      r<0 c<0  |  r<0 c>0
            Q2 | Q1      r>0 c<0  |  r>0 c>0
    
    These quadrants are inversed about the x-axis due to rows being positive in the downward direction
    '''
    
    
    # Determine the best direction to perform calcualtions
    #  Row-Dominated
    if d_xs_direction >= (math.pi / 4) and d_xs_direction <= (3 * math.pi / 4):
        # Calculate the distance in the x direction
        da_distance_x = np.arange(i_centerpoint) * d_dy * math.cos(d_xs_direction)

        # Convert the distance to a number of indices
        ia_x_index_offset: int = da_distance_x // d_dx

        ia_xc_dr_index_main[0:i_centerpoint] = np.arange(i_centerpoint)
        ia_xc_dc_index_main[0:i_centerpoint] = ia_x_index_offset

        # Calculate the sign of the angle
        ia_sign = np.ones(i_centerpoint)
        ia_sign[da_distance_x < 0] = -1

        # Round using the angle direction
        ia_x_index_offset = np.round((da_distance_x / d_dx) + 0.5 * ia_sign, 0)

        # Set the values in as index locations
        ia_xc_dr_index_second[0:i_centerpoint] = np.arange(i_centerpoint)
        ia_xc_dc_index_second[0:i_centerpoint] = ia_x_index_offset

        # ddx is the distance from the main cell to the location where the line passes through.  Do 1-ddx to get the weight
        da_ddx = np.fabs((da_distance_x / d_dx) - ia_x_index_offset)
        da_xc_main_fract[0:i_centerpoint] = 1.0 - da_ddx
        da_xc_second_fract[0:i_centerpoint] = da_ddx
        
        # da_xc_main_fract_int = np.rint(da_xc_main_fract).astype(int)
        da_xc_main_fract_int = np.empty(da_xc_main_fract.shape, dtype=np.int64)
        for i in range(da_xc_main_fract.size):
            da_xc_main_fract_int[i] = int(np.round(da_xc_main_fract[i]))

        # da_xc_second_fract_int = np.subtract(1,da_xc_main_fract_int, dtype=int)
        da_xc_second_fract_int = 1 - da_xc_main_fract_int

        # Distance between each increment
        d_distance_z = math.sqrt((d_dy * math.cos(d_xs_direction)) * (d_dy * math.cos(d_xs_direction)) + d_dy * d_dy)

    # Col-Dominated
    else:
        # Calculate based on the column being the dominate direction
        # Calculate the distance in the y direction
        da_distance_y = np.arange(i_centerpoint) * d_dx * math.sin(d_xs_direction)

        # Convert the distance to a number of indices
        ia_y_index_offset: int = da_distance_y // d_dy
        
        column_pos_or_neg = 1 
        if d_xs_direction >= (math.pi / 2): 
            column_pos_or_neg = -1

        ia_xc_dr_index_main[0:i_centerpoint] = ia_y_index_offset
        ia_xc_dc_index_main[0:i_centerpoint] = np.arange(i_centerpoint) * column_pos_or_neg

        # Calculate the sign of the angle
        ia_sign = np.ones(i_centerpoint)   #I think this can always just be positive one
        #ia_sign[da_distance_y < 0] = -1
        #ia_sign[da_distance_y > 0] = -1
        #ia_sign = ia_sign * -1

        # Round using the angle direction
        ia_y_index_offset = np.round((da_distance_y / d_dy) + 0.5 * ia_sign, 0)

        # Set the values in as index locations
        ia_xc_dr_index_second[0:i_centerpoint] = ia_y_index_offset
        ia_xc_dc_index_second[0:i_centerpoint] = np.arange(i_centerpoint) * column_pos_or_neg

        # ddy is the distance from the main cell to the location where the line passes through.  Do 1-ddx to get the weight
        da_ddy = np.fabs((da_distance_y / d_dy) - ia_y_index_offset)
        da_xc_main_fract[0:i_centerpoint] = 1.0 - da_ddy
        da_xc_second_fract[0:i_centerpoint] = da_ddy
        
        # da_xc_main_fract_int = np.round(da_xc_main_fract).astype(int)
        da_xc_main_fract_int = np.empty(da_xc_main_fract.shape, dtype=np.int64)
        for i in range(da_xc_main_fract.size):
            da_xc_main_fract_int[i] = int(np.round(da_xc_main_fract[i]))
        # da_xc_second_fract_int = np.subtract(1,da_xc_main_fract_int, dtype=int)
        da_xc_second_fract_int = 1 - da_xc_main_fract_int

        # Distance between each increment
        d_distance_z = math.sqrt((d_dx * math.sin(d_xs_direction)) * (d_dx * math.sin(d_xs_direction)) + d_dx * d_dx)

    # Return to the calling function
    return d_distance_z, da_xc_main_fract_int, da_xc_second_fract_int

@njit(cache=True)
def get_xs_index_values(i_entry_cell: int, ia_xc_dr_index_main: np.ndarray, ia_xc_dc_index_main: np.ndarray, ia_xc_dr_index_second: np.ndarray, ia_xc_dc_index_second: np.ndarray, da_xc_main_fract: np.ndarray,
                        da_xc_second_fract: np.ndarray, d_xs_direction: np.ndarray, i_r_start: int, i_c_start: int, i_centerpoint: int, d_dx: float, d_dy: float):
    """
    i_entry_cell, ia_xc_dr_index_main, ia_xc_dc_index_main, ia_xc_dr_index_second, ia_xc_dc_index_second, da_xc_main_fract, da_xc_main_fract_int, da_xc_second_fract, da_xc_second_fract_int, d_xs_direction, i_row_cell,
                                               i_column_cell, i_center_point, dx, dy
    Calculates the distance of the stream cross section

    Parameters
    ----------
    ia_xc_dr_index_main: ndarray
        Indices of the first cross section index
    ia_xc_dc_index_main: ndarray
        Index offsets of the first cross section index
    ia_xc_dr_index_second: ndarray
        Indices of the second cross section index
    ia_xc_dc_index_second: ndarray
        Index offsets of the second cross section index
    da_xc_main_fract: ndarray: ndarray
        # todo: add
    da_xc_second_fract: ndarray
        # todo: add
    d_xs_direction: float
        Orientation of the cross section
    i_r_start: int
        Starting row index of the search
    i_c_start: int
        Starting column index of the search
    i_centerpoint: int
        Distance from the cell to search
    d_dx: float
        Cell resolution in the x direction
    d_dy: float
        Cell resolution in the y direction

    Returns
    -------
    d_distance_z: float
        Distance along the cross section direction

    """
    
    
    '''
    Assume there are 4 quadrants:
            Q3 | Q4      r<0 c<0  |  r<0 c>0
            Q2 | Q1      r>0 c<0  |  r>0 c>0
    
    These quadrants are inversed about the x-axis due to rows being positive in the downward direction
    '''
    
    
    # Determine the best direction to perform calcualtions
    #  Row-Dominated
    if d_xs_direction >= (math.pi / 4) and d_xs_direction <= (3 * math.pi / 4):
        # Calculate the distance in the x direction
        da_distance_x = np.arange(i_centerpoint) * d_dy * math.cos(d_xs_direction)

        # Convert the distance to a number of indices
        ia_x_index_offset: int = da_distance_x // d_dx

        ia_xc_dr_index_main[0:i_centerpoint] = np.arange(i_centerpoint)
        ia_xc_dc_index_main[0:i_centerpoint] = ia_x_index_offset

        # Calculate the sign of the angle
        ia_sign = np.ones(i_centerpoint)
        ia_sign[da_distance_x < 0] = -1

        # Round using the angle direction
        ia_x_index_offset = np.round((da_distance_x / d_dx) + 0.5 * ia_sign, 0)

        # Set the values in as index locations
        ia_xc_dr_index_second[0:i_centerpoint] = np.arange(i_centerpoint)
        ia_xc_dc_index_second[0:i_centerpoint] = ia_x_index_offset

        # ddx is the distance from the main cell to the location where the line passes through.  Do 1-ddx to get the weight
        da_ddx = np.fabs((da_distance_x / d_dx) - ia_x_index_offset)
        da_xc_main_fract[0:i_centerpoint] = 1.0 - da_ddx
        da_xc_second_fract[0:i_centerpoint] = da_ddx
        
        # da_xc_main_fract_int = np.rint(da_xc_main_fract).astype(int)
        da_xc_main_fract_int = np.empty(da_xc_main_fract.shape, dtype=np.int64)
        for i in range(da_xc_main_fract.size):
            da_xc_main_fract_int[i] = int(np.round(da_xc_main_fract[i]))

        # da_xc_second_fract_int = np.subtract(1,da_xc_main_fract_int, dtype=int)
        da_xc_second_fract_int = 1 - da_xc_main_fract_int

        # Distance between each increment
        d_distance_z = math.sqrt((d_dy * math.cos(d_xs_direction)) * (d_dy * math.cos(d_xs_direction)) + d_dy * d_dy)

    # Col-Dominated
    else:
        # Calculate based on the column being the dominate direction
        # Calculate the distance in the y direction
        da_distance_y = np.arange(i_centerpoint) * d_dx * math.sin(d_xs_direction)

        # Convert the distance to a number of indices
        ia_y_index_offset: int = da_distance_y // d_dy
        
        column_pos_or_neg = 1 
        if d_xs_direction >= (math.pi / 2): 
            column_pos_or_neg = -1

        ia_xc_dr_index_main[0:i_centerpoint] = ia_y_index_offset
        ia_xc_dc_index_main[0:i_centerpoint] = np.arange(i_centerpoint) * column_pos_or_neg

        # Calculate the sign of the angle
        ia_sign = np.ones(i_centerpoint)   #I think this can always just be positive one
        #ia_sign[da_distance_y < 0] = -1
        #ia_sign[da_distance_y > 0] = -1
        #ia_sign = ia_sign * -1

        # Round using the angle direction
        ia_y_index_offset = np.round((da_distance_y / d_dy) + 0.5 * ia_sign, 0)

        # Set the values in as index locations
        ia_xc_dr_index_second[0:i_centerpoint] = ia_y_index_offset
        ia_xc_dc_index_second[0:i_centerpoint] = np.arange(i_centerpoint) * column_pos_or_neg

        # ddy is the distance from the main cell to the location where the line passes through.  Do 1-ddx to get the weight
        da_ddy = np.fabs((da_distance_y / d_dy) - ia_y_index_offset)
        da_xc_main_fract[0:i_centerpoint] = 1.0 - da_ddy
        da_xc_second_fract[0:i_centerpoint] = da_ddy
        
        # da_xc_main_fract_int = np.round(da_xc_main_fract).astype(int)
        da_xc_main_fract_int = np.empty(da_xc_main_fract.shape, dtype=np.int64)
        for i in range(da_xc_main_fract.size):
            da_xc_main_fract_int[i] = int(np.round(da_xc_main_fract[i]))
        # da_xc_second_fract_int = np.subtract(1,da_xc_main_fract_int, dtype=int)
        da_xc_second_fract_int = 1 - da_xc_main_fract_int

        # Distance between each increment
        d_distance_z = math.sqrt((d_dx * math.sin(d_xs_direction)) * (d_dx * math.sin(d_xs_direction)) + d_dx * d_dx)

    # Return to the calling function
    return d_distance_z, da_xc_main_fract_int, da_xc_second_fract_int

@njit(cache=True)
def sample_cross_section_from_dem(i_entry_cell: int, da_xs_profile: np.ndarray, dm_elevation: np.ndarray, i_center_point: int, ia_xc_row_index_main: np.ndarray,
                                  ia_xc_column_index_main: np.ndarray, ia_xc_row_index_second: np.ndarray, ia_xc_column_index_second: np.ndarray, da_xc_main_fract: np.ndarray, da_xc_main_fract_int: np.ndarray,
                                  da_xc_second_fract: np.ndarray, da_xc_second_fract_int: np.ndarray, i_row_bottom: int, i_row_top: int, i_column_bottom: int, i_column_top: int, ia_lc_xs: np.ndarray, dm_land_use: np.ndarray = None):
    """

    Parameters
    ----------
    da_xs_profile: ndarray
        Elevations of the cross section
    i_row: int
        Starting row index
    i_column: int
        Starting column index
    dm_elevation: ndarray
        Elevation raster
    i_center_point: int
        Starting centerpoint distance
    ia_xc_row_index_main: ndarray
        Indices of the first cross section row index
    ia_xc_column_index_main: ndarray
        Indices of the first cross section column index
    ia_xc_row_index_second: ndarray
        Indices of the second cross section row index
    ia_xc_column_index_second: ndarray
        Indices of the second cross section column index
    da_xc_main_fract: ndarray
        # todo: add
    da_xc_second_fract: ndarray
        # todo: add
    i_row_bottom: int
        Bottom row of the search window
    i_row_top: int
        Top row of the search window
    i_column_bottom: int
        Left column of the search window
    i_column_top: int
        Right column of the search window

    Returns
    -------
    i_center_point: int
        Updated center point value

    """
    
    # Get the limits of the cross-section index
    a = np.where(ia_xc_row_index_main == i_row_bottom)
    if len(a[0]) > 0 and int(a[0][0]) < i_center_point:
        i_center_point = int(a[0][0])

    a = np.where(ia_xc_row_index_second == i_row_bottom)
    if len(a[0]) > 0 and int(a[0][0]) < i_center_point:
        i_center_point = int(a[0][0])

    a = np.where(ia_xc_row_index_main >= i_row_top)
    if len(a[0]) > 0 and int(a[0][0]) < i_center_point:
        i_center_point = int(a[0][0])

    a = np.where(ia_xc_row_index_second >= i_row_top)
    if len(a[0]) > 0 and int(a[0][0]) < i_center_point:
        i_center_point = int(a[0][0])

    a = np.where(ia_xc_column_index_main == i_column_bottom)
    if len(a[0]) > 0 and int(a[0][0]) < i_center_point:
        i_center_point = int(a[0][0])

    a = np.where(ia_xc_column_index_second == i_column_bottom)
    if len(a[0]) > 0 and int(a[0][0]) < i_center_point:
        i_center_point = int(a[0][0])

    a = np.where(ia_xc_column_index_main >= i_column_top)
    if len(a[0]) > 0 and int(a[0][0]) < i_center_point:
        i_center_point = int(a[0][0])

    a = np.where(ia_xc_column_index_second >= i_column_top)
    if len(a[0]) > 0 and int(a[0][0]) < i_center_point:
        i_center_point = int(a[0][0])

    # Set a default value for the profile at the center
    da_xs_profile[i_center_point] = 99999.9

    # Extract the indices and set into the profile
    try:
        for i in range(i_center_point):
            row_main = ia_xc_row_index_main[i]
            col_main = ia_xc_column_index_main[i]
            row_second = ia_xc_row_index_second[i]
            col_second = ia_xc_column_index_second[i]
            
            # Calculate the profile value based on the indexed values and fractions
            da_xs_profile[i] = (
                dm_elevation[row_main, col_main] * da_xc_main_fract[i] +
                dm_elevation[row_second, col_second] * da_xc_second_fract[i]
            )
        
        # Iterate through each index up to i_center_point to avoid advanced indexing
        # Also, we don't need to fill land use raster here for finding best stream angle
        if dm_land_use is not None:
            for i in range(i_center_point):
                row_main = ia_xc_row_index_main[i]
                col_main = ia_xc_column_index_main[i]
                row_second = ia_xc_row_index_second[i]
                col_second = ia_xc_column_index_second[i]
                
                # Calculate the land use value for each element and convert to int
                ia_lc_xs[i] = int(
                    dm_land_use[row_main, col_main] * da_xc_main_fract_int[i] +
                    dm_land_use[row_second, col_second] * da_xc_second_fract_int[i]
                )

    except:
        print('Error on Cell ' + str(i_entry_cell))

    # Return the center point to the calling function
    return i_center_point

@njit(cache=True)
def find_bank(da_xs_profile: np.ndarray, i_cross_section_number: int, d_z_target: float, wse: bool = False):
    """
    Finds the cell containing the bank of the cross section. Subtract 1 to get WSE elevation

    Parameters
    ----------
    da_xs_profile: ndarray
        Elevations of the stream cross section
    i_cross_section_number: int
        Index of the cross section cell
    d_z_target: float
        Target elevation that defines the bank

    Returns
    -------
    i_cross_section_number: int
        Updated cell index that defines the bank

    """

    # Loop on the cells of the cross section
    for entry in range(1, i_cross_section_number):
        # Check if the profile elevation matches the target elevation
        if da_xs_profile[entry] >= d_z_target:
                return entry - 1 if wse else entry

    # Return to the calling function
    return i_cross_section_number

@njit(cache=True)
def find_wse_and_banks_by_lc(da_xs_profile1, ia_lc_xs1, xs1_n, da_xs_profile2, ia_lc_xs2, xs2_n, d_z_target, i_lc_water_value):
    
    #Initially set the bank info to zeros
    i_bank_1_index = 0
    i_bank_2_index = 0
    
    bank_elev_1 = da_xs_profile1[0]
    bank_elev_2 = da_xs_profile2[0]
    for i in range(1, xs1_n):
        if ia_lc_xs1[i] == i_lc_water_value:
            if da_xs_profile1[i] < bank_elev_1:
                bank_elev_1 = da_xs_profile1[i]
        else:
            i_bank_1_index = i
            break

    for i in range(1, xs2_n):
        if ia_lc_xs2[i] == i_lc_water_value:
            if da_xs_profile2[i] < bank_elev_2:
                bank_elev_2 = da_xs_profile2[i]
        else:
            i_bank_2_index = i
            break
    
    if bank_elev_1>da_xs_profile1[0]:
        if bank_elev_2>da_xs_profile1[0]:
            d_wse_from_dem = min(bank_elev_1, bank_elev_2)
        else:
            d_wse_from_dem = bank_elev_1
    elif bank_elev_2>da_xs_profile1[0]:
        d_wse_from_dem = bank_elev_2
    else:
        d_wse_from_dem = d_z_target
    
    return d_wse_from_dem, i_bank_1_index, i_bank_2_index

@njit(cache=True)
def find_depth_of_bathymetry(d_baseflow: float, d_bottom_width: float, d_top_width: float, d_slope: float, d_mannings_n: float):
    """
    Estimates the depth iteratively by comparing the calculated flow to the baseflow

    Parameters
    ----------
    d_baseflow: float
        Baseflow input for flow convergence calculation
    d_bottom_width: float
        Bottom width of the stream
    d_top_width: float
        Top width of the stream
    d_slope: float
        Slope of the stream
    d_mannings_n: float
        Manning's roughness of the stream

    Returns
    -------
    d_working_depth: float
        Estimated depth of the stream

    """

    # Calculate the average width of the stream
    d_average_width = (d_top_width - d_bottom_width) * 0.5

    # Assign a starting depth
    d_depth_start = 0.0

    # Set the incremental convergence targets
    l_dy_list = [1.0, 0.5, 0.1, 0.01]
    
    # Loop over each convergence target
    for d_dy in l_dy_list:
        # Set the initial value
        d_flow_calculated = 0.0
        d_working_depth = d_depth_start

        # This will prevent infinite loops
        d_max_depth = d_depth_start + 25

        # Converge until the calculate flow is above the baseflow
        while d_flow_calculated <= d_baseflow and d_working_depth < d_max_depth:
            d_working_depth = d_working_depth + d_dy
            d_area = d_working_depth * (d_bottom_width + d_top_width) / 2.0
            d_perimeter = d_bottom_width + 2.0 * math.sqrt(d_average_width * d_average_width + d_working_depth * d_working_depth)
            d_hydraulic_radius = d_area / d_perimeter
            d_flow_calculated = (1.0 / d_mannings_n) * d_area * d_hydraulic_radius**(2 / 3) * d_slope**0.5

        # Update the starting depth
        d_depth_start = d_working_depth - d_dy

    # Update the calculated depth
    d_working_depth = d_working_depth - d_dy

    # Debugging variables
    # A = y * (B + TW) / 2.0
    # P = B + 2.0*math.sqrt(H*H + y*y)
    # R = A / P
    # Qcalc = (1.0/n)*A*math.pow(R,(2/3)) * pow(slope,0.5)
    # print(str(d_top_width) + ' ' + str(d_working_depth) + '  ' + str(d_flow_calculated) + ' vs ' + str(d_baseflow))

    return d_working_depth

@njit(cache=True)
def adjust_profile_for_bathymetry(da_xs_profile: np.ndarray, i_bank_index: int, d_total_bank_dist: float, d_trap_base: float, d_distance_z: float, d_distance_h: float, d_y_bathy: float,
                                  d_y_depth: float, dm_output_bathymetry: np.ndarray, ia_xc_r_index_main: np.ndarray, ia_xc_c_index_main: np.ndarray, 
                                  dm_land_use: np.ndarray, d_side_dist: float, dm_elevation: np.ndarray, b_bathy_use_banks: bool):
    """
    Adjusts the profile for the estimated bathymetry

    Parameters
    ----------
    da_xs_profile: ndarray
        Elevations of the stream cross section
    i_bank_index: int
        Distance in index space from the stream to the bank
    d_total_bank_dist: float
        Distance to the bank estimated in unit space
    d_trap_base: float
        Bottom distance of the stream cross section
    d_distance_z: float
        Incremental distance per cell parallel to the orientation of the cross section
    d_distance_h: float
        Distance of the slope section of the trapezoidal channel.  Typically d_distance_h = 0.2* TW of Trapezoid
    d_y_bathy: float
        Bathymetry elevation of the bottom
    d_y_depth: float
        Depth.  Basically water surface elevation (WSE) minus d_y_bathy
    dm_output_bathymetry: ndarray
        Output bathymetry matrix
    ia_xc_r_index_main: ndarray
        Row indices for the stream cross section
    ia_xc_c_index_main: ndarray
        Column indices for the stream cross section

    Returns
    -------
    None. Values are updated in the output bathymetry matrix

    """

    # If banks are calculated, make an adjustment to the trapezoidal bathymetry
    if i_bank_index > 0:
        # Loop over the bank width offset indices
        for x in range(min(i_bank_index + 1, len(ia_xc_r_index_main))):
            # Calculate the distance to the bank
            d_dist_cell_to_bank = (i_bank_index - x) * d_distance_z + d_side_dist   #d_side_dist should be zero if using Flat WSE or LC method.
            # lc_grid_val = int(dm_land_use[ia_xc_r_index_main[x], ia_xc_c_index_main[x]])

            # if lc_grid_val<0 or (i_lc_water_value>0 and lc_grid_val!=i_lc_water_value):
            #     return

            # # Joseph added this because it looks like we aren't getting a bathymetry output for the first cell in the cross-section
            # if x == 0:
            #     # If the cell is the first cell, then set it to the bottom elevation of the trapezoid.
            #     da_xs_profile[x] = d_y_bathy
            #     dm_output_bathymetry[ia_xc_r_index_main[x], ia_xc_c_index_main[x]] = da_xs_profile[x]

            # If the cell is in the flat part of the trapezoidal cross-section, set it to the bottom elevation of the trapezoid.
            if d_dist_cell_to_bank > d_distance_h:
                if b_bathy_use_banks == False and d_y_bathy < dm_elevation[ia_xc_r_index_main[x], ia_xc_c_index_main[x]]:
                    da_xs_profile[x] = d_y_bathy
                    dm_output_bathymetry[ia_xc_r_index_main[x], ia_xc_c_index_main[x]] = da_xs_profile[x]
                elif b_bathy_use_banks == True:
                    da_xs_profile[x] = d_y_bathy
                    dm_output_bathymetry[ia_xc_r_index_main[x], ia_xc_c_index_main[x]] = da_xs_profile[x]

            # If the cell is in the slope part of the trapezoid you need to find the elevation based on the slope of the trapezoid side.
            elif d_dist_cell_to_bank <= d_distance_h and d_dist_cell_to_bank < d_trap_base + d_distance_h:
                if b_bathy_use_banks == False and (d_y_bathy + d_y_depth * (1.0 - (d_dist_cell_to_bank / d_distance_h))) < dm_elevation[ia_xc_r_index_main[x], ia_xc_c_index_main[x]]:
                    da_xs_profile[x] = d_y_bathy + d_y_depth * (1.0 - (d_dist_cell_to_bank / d_distance_h))
                    dm_output_bathymetry[ia_xc_r_index_main[x], ia_xc_c_index_main[x]] = da_xs_profile[x]
                elif b_bathy_use_banks == True:
                    da_xs_profile[x] = d_y_bathy + d_y_depth * (1.0 - (d_dist_cell_to_bank / d_distance_h))
                    dm_output_bathymetry[ia_xc_r_index_main[x], ia_xc_c_index_main[x]] = da_xs_profile[x]

            # Similar to above, but on the far-side slope of the trapezoid.  You need to find the elevation based on the slope of the trapezoid side.
            elif d_dist_cell_to_bank >= d_trap_base + d_distance_h:
                d_dist_cell_to_bank_other_side = d_total_bank_dist - d_dist_cell_to_bank
                if b_bathy_use_banks == False and d_dist_cell_to_bank_other_side>0.0 and (d_y_bathy + d_y_depth * (1.0 - (d_dist_cell_to_bank_other_side / d_distance_h))) < dm_elevation[ia_xc_r_index_main[x], ia_xc_c_index_main[x]]:
                    da_xs_profile[x] = d_y_bathy + d_y_depth * (1.0 - (d_dist_cell_to_bank_other_side / d_distance_h))
                    dm_output_bathymetry[ia_xc_r_index_main[x], ia_xc_c_index_main[x]] = da_xs_profile[x]
                elif b_bathy_use_banks == True:
                    da_xs_profile[x] = d_y_bathy + d_y_depth * (1.0 - (d_dist_cell_to_bank_other_side / d_distance_h))
                    dm_output_bathymetry[ia_xc_r_index_main[x], ia_xc_c_index_main[x]] = da_xs_profile[x]
                #if (d_y_bathy + d_y_depth * (d_dist_cell_to_bank - (d_trap_base + d_distance_h)) / d_distance_h) < dm_elevation[ia_xc_r_index_main[x], ia_xc_c_index_main[x]]:
                #    da_xs_profile[x] = d_y_bathy + d_y_depth * (d_dist_cell_to_bank - (d_trap_base + d_distance_h)) / d_distance_h
                #    dm_output_bathymetry[ia_xc_r_index_main[x], ia_xc_c_index_main[x]] = da_xs_profile[x]

            # If the cell is outside of the banks, then just ignore this cell (set it to it's same elevation).  No need to update the output bathymetry raster.
            elif d_dist_cell_to_bank <= 0 or d_dist_cell_to_bank >= d_total_bank_dist:
                return


            
            #JUST FOR TESTING
            #da_xs_profile[x] = d_y_bathy
            #dm_output_bathymetry[ia_xc_r_index_main[x], ia_xc_c_index_main[x]] = da_xs_profile[x]

    return

@njit(cache=True)
def calculate_hypotnuse(d_side_one: float, d_side_two: float):
    """
    Calculates the hypotenuse distance of a right triangle

    Parameters
    ----------
    d_side_one: float
        Length of the first right triangle side
    d_side_two: float
        Length of the second right triangle side

    Returns
    -------
    d_distance: float
        Length of the hypotenuse

    """

    # Calculate the distance
    d_distance = (d_side_one ** 2 + d_side_two ** 2)**(1/2)

    # Return to the calling function
    return d_distance

@njit(cache=True)
def calculate_stream_geometry(da_xs_profile: np.ndarray, 
                              d_wse: float, 
                              d_distance_z: float, 
                              da_n_profile: np.ndarray,) -> tuple[float, ...]:
    """
    Estimates the stream geometry

    Uses a composite Manning's n as given by:
    Composite Manning N based on https://www.hec.usace.army.mil/confluence/rasdocs/ras1dtechref/6.5/theoretical-basis-for-one-dimensional-and-two-dimensional-hydrodynamic-calculations/1d-steady-flow-water-surface-profiles/composite-manning-s-n-for-the-main-channel

    Parameters
    ----------
    da_xs_profile: ndarray
        Elevations of the stream cross section
    d_wse: float
        Water surface elevation
    d_distance_z: float
        Incremental distance per cell parallel to the orientation of the cross section
    da_n_profile: float
        Input initial Manning's n for the stream

    Returns
    -------
    d_area, d_perimeter, d_hydraulic_radius, d_composite_n, d_top_width

    """
    # Initial output
    d_area, d_perimeter, d_hydraulic_radius, d_composite_n, d_top_width = 0.0, 0.0, 0.0, 0.0, 0.0

    # Estimate the depth of the stream
    da_y_depth = d_wse - da_xs_profile

    # Return if the depth is not valid.
    if da_y_depth.shape[0] <= 0 or da_y_depth[0] <= 1e-16:
        return 0, 0, 0, 0, 0

    # Take action if there are values < 0
    lt_0_in_depths = False
    for i_target_index, value in enumerate(da_y_depth[1:]):
        if value <= 0:
            lt_0_in_depths = True
            break
    
    if lt_0_in_depths:
        # A value < 0 exists. Calculate up to that value then break for the rest of hte values.
        # Get the index of the first bad vadlue
        i_target_index += 1

        # Calculate the distance to use
        d_dist_use = d_distance_z * da_y_depth[i_target_index - 1] / (np.abs(da_y_depth[i_target_index - 1]) + np.abs(da_y_depth[i_target_index]))

        # Calculate the geometric variables
        d_area = np.sum(d_distance_z * 0.5 * (da_y_depth[1:i_target_index] + da_y_depth[:i_target_index-1])) + 0.5 * d_dist_use * da_y_depth[i_target_index-1]

        d_perimeter_i = calculate_hypotnuse(d_dist_use, da_y_depth[i_target_index - 1])
        perim_array = calculate_hypotnuse(d_distance_z, (da_y_depth[1:i_target_index] - da_y_depth[:i_target_index-1]))

        d_perimeter = np.sum(perim_array) + d_perimeter_i
        
        d_hydraulic_radius = np.inf if d_perimeter == 0 else d_area / d_perimeter

        # Calculate the composite n
        d_composite_n = np.sum(perim_array[:i_target_index-1] * da_n_profile[1:i_target_index]**1.5) + d_perimeter_i * da_n_profile[i_target_index - 1]**1.5

        # Update the top width
        d_top_width = d_distance_z * (i_target_index - 1) + d_dist_use

    else:
        # All values are positive, so include them all.

        # Calculate the geometric values
        d_area = np.sum(d_distance_z * 0.5 * (da_y_depth[2:] + da_y_depth[1:-1]))

        perim_array = calculate_hypotnuse(d_distance_z, da_y_depth[1:] - da_y_depth[:-1])

        d_perimeter = np.sum(perim_array[1:])

        d_hydraulic_radius = np.inf if d_perimeter == 0 else d_area / d_perimeter

        d_composite_n = np.sum(perim_array * da_n_profile[1:]**1.5)

        d_top_width = d_distance_z * (da_y_depth.shape[0] - 1)

    # Return to the calling function
    return d_area, d_perimeter, d_hydraulic_radius, d_composite_n, d_top_width

@njit(cache=True)
def find_bank_using_width_to_depth_ratio(da_xs_profile1, da_xs_profile2, xs1_n, xs2_n, d_distance_z):
    """
    da_xs_profile1: ndarray
        Elevations of the stream cross section on one side
    da_xs_profile2: ndarray
        Elevations of the stream cross section on the other side
    xs1_n: int
        Index of the cross section cells on one of the cross section
    xs2_n: int
        Index of the cross section cells on the other side of the cross section
    d_distance_z: float
        Incremental distance per cell parallel to the orientation of the cross section

    """
    # Precompute sliced arrays
    da_xs_profile1_sliced = da_xs_profile1[0:xs1_n]
    da_xs_profile2_sliced = da_xs_profile2[0:xs2_n]
    
    # We don't use mannings n in this func, so these are just dummys (they are generated really quickly)
    fake_mannings_1 = np.empty_like(da_xs_profile1_sliced, dtype=np.float32)
    fake_mannings_2 = np.empty_like(da_xs_profile2_sliced, dtype=np.float32)

    d_bottom_elevation = da_xs_profile1[0]
    d_depth = 0
    d_new_width_to_depth_ratio = 0
    d_width_to_depth_ratio = np.inf  # Start with a large value

    prev_t1 = 0.
    prev_t2 = 0.

    # we will assume that if we get to a depth of 25 meters, something has gone wrong
    while d_new_width_to_depth_ratio <= d_width_to_depth_ratio and d_depth <= 25:
        d_depth += 0.01
        d_wse = d_bottom_elevation + d_depth
        
        # Calculate stream geometry for both sides
        A1, P1, R1, np1, T1 = calculate_stream_geometry(da_xs_profile1_sliced, d_wse, d_distance_z, fake_mannings_1)
        A2, P2, R2, np2, T2 = calculate_stream_geometry(da_xs_profile2_sliced, d_wse, d_distance_z, fake_mannings_2)
        
        TW = T1 + T2
        d_new_width_to_depth_ratio = TW / d_depth

        if d_new_width_to_depth_ratio > d_width_to_depth_ratio:
            # Recalculate the last valid depth
            d_depth -= 0.01
            T1 = prev_t1
            T2 = prev_t2
            break

        d_width_to_depth_ratio = d_new_width_to_depth_ratio
        prev_t1 = T1
        prev_t2 = T2

    if d_depth < 25:
        i_bank_1_index = int(T1 / d_distance_z)
        i_bank_2_index = int(T2 / d_distance_z)
    # if we have made it to 25 on d_depth, something is wrong and the banks will be set at the stream cell
    elif d_depth >= 25:
        i_bank_1_index = 0
        i_bank_2_index = 0

    return (i_bank_1_index, i_bank_2_index)

def find_bank_inflection_point(da_xs_profile: np.ndarray, i_cross_section_number: int, d_distance_z: float, window_length: int = 11, polyorder: int = 3):
    """
    Finds the cell containing the bank of the cross section, with smoothing applied.

    Parameters
    ----------
    da_xs_profile: ndarray
        Elevations of the stream cross section
    i_cross_section_number: int
        Index of the cross section cell
    d_distance_z: float
        Incremental distance per cell parallel to the orientation of the cross section
    window_length: int, optional
        The length of the filter window for smoothing (must be an odd number, default is 11)
    polyorder: int, optional
        The order of the polynomial used to fit the samples for smoothing (default is 3)

    Returns
    -------
    i_cross_section_number: int
        Updated cell index that defines the bank
    """
    # Apply smoothing to the cross-section data
    # da_xs_smooth = da_xs_profile
    try:
        da_xs_smooth = savgol_filter(da_xs_profile, window_length=window_length, polyorder=polyorder)
    except np.linalg.LinAlgError:
        # If the rare case smoothing fails, just use original profile
        da_xs_smooth = da_xs_profile
    
    return find_bank_inflection_point_helper(da_xs_smooth, i_cross_section_number, d_distance_z)

@njit(cache=True)
def find_bank_inflection_point_helper(da_xs_smooth: np.ndarray, i_cross_section_number: int, d_distance_z: float) -> int:
    # Loop on the smoothed cross-section cells
    entry = 0
    previous_delta_elevation = 0.0
    total_width = 0.0
    while entry < i_cross_section_number:
        elevation_0 = da_xs_smooth[entry]
        elevation_1 = da_xs_smooth[entry + 1]

        current_delta_elevation = elevation_1 - elevation_0

        if current_delta_elevation >= previous_delta_elevation:
            previous_delta_elevation = current_delta_elevation
            total_width += d_distance_z
            entry += 1  # move forward
        else:
            # Found the bank – go back one if needed
            return entry  # or return entry - 1 if you want the previous one

    # Return to the calling function
    return 0

def read_manning_table(s_manning_path: str, da_input_mannings: np.ndarray):
    """
    Reads the Manning's n information from the input file

    Parameters
    ----------
    s_manning_path: str
        Path to the Manning's n input table
    da_input_mannings: ndarray
        Array holding the mannings estimates

    Returns
    -------
    da_input_mannings: ndarray
        Array holding the mannings estimates

    """

    # Open and read the input file
    df = pd.read_csv(s_manning_path, sep='\t')

    # Create a lookup array for the Manning's n values
    # This is the fastest way to reclassify the values in the input array
    idx = df.iloc[:, 0].astype(int).values
    lookup_array = np.zeros(idx.max() + 1)
    lookup_array[idx] = df.iloc[:, 2].values
    da_input_mannings = lookup_array[da_input_mannings.astype(int)]

    # Return to the calling function
    return da_input_mannings

@njit(cache=True)
def adjust_cross_section_to_lowest_point(i_low_point_index, d_dem_low_point_elev, da_xs_profile_one, da_xs_profile_two, ia_xc_r1_index_main, ia_xc_r2_index_main, ia_xc_c1_index_main, ia_xc_c2_index_main, da_xs1_mannings, da_xs2_mannings,
                                         i_center_point, i_low_spot_range):
    """
    Reorients the cross section through the lowest point of the stream. Cross-section needs to be re-sampled if the low spot in the cross-section changes location.

    Parameters
    ----------
    i_low_point_index: int
        Offset index along the cross section of the lowest point
    d_dem_low_point_elev: float
        Elevation of the lowest point
    da_xs_profile_one: ndarray
        Cross section elevations of the first cross section
    da_xs_profile_two: ndarray
        Cross section elevations of the second cross section
    ia_xc_r1_index_main: ndarray
        Row indices of the first cross section
    ia_xc_r2_index_main: ndarray
        Row indices of the second cross section
    ia_xc_c1_index_main: ndarray
        Column indices of the first cross section
    ia_xc_c2_index_main: ndarray
        Column indicies of the second cross section
    da_xs1_mannings: ndarray
        Manning's roughness of the first cross section
    da_xs2_mannings: ndarray
        Manning's roughness of the second cross section
    i_center_point: int
        Center point index
    i_low_spot_range: int
        The number of cells on each side of the cross-section we're looking at moving to. 

    Returns
    -------
    i_low_point_index: int
        Index of the low point in the cross section array
    """
    # Loop on the search range for the low point
    for i_entry in range(i_low_spot_range):
        # Look in the first profile
        if da_xs_profile_one[i_entry] > 0.0 and da_xs_profile_one[i_entry] < d_dem_low_point_elev:
            # New low point was found. Update the index.
            d_dem_low_point_elev = da_xs_profile_one[i_entry]
            i_low_point_index = i_entry

        # Look in the second profile
        if da_xs_profile_two[i_entry] > 0.0 and da_xs_profile_two[i_entry] < d_dem_low_point_elev:
            # New low point was found. Update the index.
            d_dem_low_point_elev = da_xs_profile_two[i_entry]
            i_low_point_index = i_entry * -1

    # Process based on if the low point is in the first or second profile
    if i_low_point_index > 0:
        # Low point is in the first profile. Update the cross section and mannings.
        da_xs_profile_two[i_low_point_index:i_center_point] = da_xs_profile_two[0:i_center_point - i_low_point_index]
        da_xs_profile_two[0:i_low_point_index + 1] = np.flip(da_xs_profile_one[0:i_low_point_index + 1])
        da_xs_profile_one[0:i_center_point - i_low_point_index] = da_xs_profile_one[i_low_point_index:i_center_point]
        da_xs1_mannings = da_xs1_mannings - i_low_point_index
        da_xs2_mannings = da_xs2_mannings + i_low_point_index
        da_xs_profile_one[da_xs1_mannings] = 99999.9

        # Update the row indices
        ia_xc_r2_index_main[i_low_point_index:i_center_point] = ia_xc_r2_index_main[0:i_center_point - i_low_point_index]
        ia_xc_r2_index_main[0:i_low_point_index + 1] = np.flip(ia_xc_r1_index_main[0:i_low_point_index + 1])
        ia_xc_r1_index_main[0:i_center_point - i_low_point_index] = ia_xc_r1_index_main[i_low_point_index:i_center_point]

        # Update the column indices
        ia_xc_c2_index_main[i_low_point_index:i_center_point] = ia_xc_c2_index_main[0:i_center_point - i_low_point_index]
        ia_xc_c2_index_main[0:i_low_point_index + 1] = np.flip(ia_xc_c1_index_main[0:i_low_point_index + 1])
        ia_xc_c1_index_main[0:i_center_point - i_low_point_index] = ia_xc_c1_index_main[i_low_point_index:i_center_point]

    elif i_low_point_index < 0:
        # Low point is in the second profile Update the cross section and mannings.
        i_low_point_index = i_low_point_index * -1
        da_xs_profile_one[i_low_point_index:i_center_point] = da_xs_profile_one[0:i_center_point - i_low_point_index]
        da_xs_profile_one[0:i_low_point_index + 1] = np.flip(da_xs_profile_two[0:i_low_point_index + 1])
        da_xs_profile_two[0:i_center_point - i_low_point_index] = da_xs_profile_two[i_low_point_index:i_center_point]
        da_xs2_mannings = da_xs2_mannings - i_low_point_index
        da_xs1_mannings = da_xs1_mannings + i_low_point_index
        da_xs_profile_two[da_xs2_mannings] = 99999.9

        # Update the row indices
        ia_xc_r1_index_main[i_low_point_index:i_center_point] = ia_xc_r1_index_main[0:i_center_point - i_low_point_index]
        ia_xc_r1_index_main[0:i_low_point_index + 1] = np.flip(ia_xc_r2_index_main[0:i_low_point_index + 1])
        ia_xc_r2_index_main[0:i_center_point - i_low_point_index] = ia_xc_r2_index_main[i_low_point_index:i_center_point]

        # Update the column indices
        ia_xc_c1_index_main[i_low_point_index:i_center_point] = ia_xc_c1_index_main[0:i_center_point - i_low_point_index]
        ia_xc_c1_index_main[0:i_low_point_index + 1] = np.flip(ia_xc_c2_index_main[0:i_low_point_index + 1])
        ia_xc_c2_index_main[0:i_center_point - i_low_point_index] = ia_xc_c2_index_main[i_low_point_index:i_center_point]
    
    #Set the index values to be within the confines of the raster
    # ia_xc_r1_index_main = np.clip(ia_xc_r1_index_main,0,nrows+2*i_boundary_number-2)
    # ia_xc_r2_index_main = np.clip(ia_xc_r2_index_main,0,nrows+2*i_boundary_number-2)
    # ia_xc_c1_index_main = np.clip(ia_xc_c1_index_main,0,ncols+2*i_boundary_number-2)
    # ia_xc_c2_index_main = np.clip(ia_xc_c2_index_main,0,ncols+2*i_boundary_number-2)

    # Return to the calling function
    return i_low_point_index

def Calculate_Bathymetry_Based_on_WSE_or_LC(da_xs_profile1, xs1_n, da_xs_profile2, xs2_n, ia_lc_xs1, ia_lc_xs2, dm_land_use, d_dem_low_point_elev, d_distance_z, d_slope_use, 
                                                           ia_xc_r1_index_main, ia_xc_c1_index_main, ia_xc_r2_index_main, ia_xc_c2_index_main, d_q_baseflow, dm_output_bathymetry, i_lc_water_value,
                                                           dm_elevation, b_FindBanksBasedOnLandCover, b_bathy_use_banks, d_bathymetry_trapzoid_height):
    """
    Calculate bathymetry based on water surface elevations.
    """


    # set the function used to none before we start running things
    function_used = None
    
    # First find the bank information
    if b_FindBanksBasedOnLandCover:   
        (d_wse_from_dem, i_bank_1_index, i_bank_2_index) = find_wse_and_banks_by_lc(da_xs_profile1, ia_lc_xs1, xs1_n, da_xs_profile2, ia_lc_xs2, xs2_n, d_dem_low_point_elev + 0.1, i_lc_water_value)
        i_total_bank_cells = i_bank_1_index + i_bank_2_index - 1
        if i_total_bank_cells > 1:
            function_used = "find_wse_and_banks_by_lc"
    else:
        i_bank_1_index = find_bank(da_xs_profile1, xs1_n, d_dem_low_point_elev + 0.1, wse=True)
        i_bank_2_index = find_bank(da_xs_profile2, xs2_n, d_dem_low_point_elev + 0.1, wse=True)
        i_total_bank_cells = i_bank_1_index + i_bank_2_index - 1
        if i_total_bank_cells > 1:
            function_used = "find_wse_and_banks_by_flat_water"

    if i_total_bank_cells <= 1:
        (i_bank_1_index, i_bank_2_index) = find_bank_using_width_to_depth_ratio(da_xs_profile1, da_xs_profile2, xs1_n, xs2_n, d_distance_z)
        i_total_bank_cells = i_bank_1_index + i_bank_2_index - 1
        if i_total_bank_cells > 1:
            function_used = "find_bank_using_width_to_depth_ratio"

    if i_total_bank_cells <= 1:
        i_bank_1_index = find_bank_inflection_point(da_xs_profile1, xs1_n, d_distance_z)
        i_bank_2_index = find_bank_inflection_point(da_xs_profile2, xs2_n, d_distance_z)
        i_total_bank_cells = i_bank_1_index + i_bank_2_index - 1
        if i_total_bank_cells > 1:
            function_used = "find_bank_inflection_point"

    if i_total_bank_cells <= 1:
        i_total_bank_cells = 1

    #Trapezoid Shape
    #      d_total_bank_dist 
    #   -----------------------
    #    -                   -
    #     -                 -
    #      -               -
    #       ---------------
    #         d_trap_base
    #  |    | <-d_h_dist->|    |
    #                     |    |<--d_h_dist = d_bathymetry_trapzoid_height * d_total_bank_dist
    # d_bathymetry_trapzoid_height is the fraction of d_total_bank_dist that is for the sloped part (see Follum et al., 2023).
    #        Basically, it assumes ~40% of the total top-width of the trapezoid is part of the sloping part
    #        Typically, d_bathymetry_trapzoid_height is set to 0.2
    
    d_total_bank_dist = i_total_bank_cells * d_distance_z
    d_h_dist = d_bathymetry_trapzoid_height * d_total_bank_dist
    d_trap_base = d_total_bank_dist - 2.0 * d_h_dist

    d_y_bathy = 0.0  # Initialize d_y_bathy to avoid UnboundLocalError

    if d_q_baseflow > 0.0 and function_used != None:
        d_y_depth = find_depth_of_bathymetry(d_q_baseflow, d_trap_base, d_total_bank_dist, d_slope_use, 0.03)
        if d_y_depth >= 25:
            if i_total_bank_cells <= 1:
                (i_bank_1_index, i_bank_2_index) = find_bank_using_width_to_depth_ratio(da_xs_profile1, da_xs_profile2, xs1_n, xs2_n, d_distance_z)
                i_total_bank_cells = i_bank_1_index + i_bank_2_index - 1
                d_total_bank_dist = i_total_bank_cells * d_distance_z
                d_h_dist = d_bathymetry_trapzoid_height * d_total_bank_dist
                d_trap_base = d_total_bank_dist - 2.0 * d_h_dist
                d_y_depth = find_depth_of_bathymetry(d_q_baseflow, d_trap_base, d_total_bank_dist, d_slope_use, 0.03)
                function_used = "find_bank_using_width_to_depth_ratio"

            if d_y_depth >= 25 and function_used == "find_bank_using_width_to_depth_ratio":
                i_bank_1_index = find_bank_inflection_point(da_xs_profile1, xs1_n, d_distance_z)
                i_bank_2_index = find_bank_inflection_point(da_xs_profile2, xs2_n, d_distance_z)
                i_total_bank_cells = i_bank_1_index + i_bank_2_index - 1
                d_total_bank_dist = i_total_bank_cells * d_distance_z
                d_h_dist = d_bathymetry_trapzoid_height * d_total_bank_dist
                d_trap_base = d_total_bank_dist - 2.0 * d_h_dist
                d_y_depth = find_depth_of_bathymetry(d_q_baseflow, d_trap_base, d_total_bank_dist, d_slope_use, 0.03)
                function_used = "find_bank_inflection_point"

            if d_y_depth >= 25:
                d_y_depth = 0.0
                d_y_bathy = da_xs_profile1[0] - d_y_depth
                i_bank_1_index = 0
                i_bank_2_index = 0
                i_total_bank_cells = 1
        if i_total_bank_cells > 1:
            d_y_bathy = da_xs_profile1[0] - d_y_depth
            adjust_profile_for_bathymetry(da_xs_profile1, i_bank_1_index, d_total_bank_dist, 
                                          d_trap_base, d_distance_z, d_h_dist, d_y_bathy, d_y_depth, 
                                          dm_output_bathymetry, ia_xc_r1_index_main, ia_xc_c1_index_main, 
                                          dm_land_use, 0.0, dm_elevation, b_bathy_use_banks)
            adjust_profile_for_bathymetry(da_xs_profile2, i_bank_2_index, d_total_bank_dist, 
                                          d_trap_base, d_distance_z, d_h_dist, d_y_bathy, d_y_depth, 
                                          dm_output_bathymetry, ia_xc_r2_index_main, ia_xc_c2_index_main, 
                                          dm_land_use, 0.0, dm_elevation, b_bathy_use_banks)

    else:
        d_y_depth = 0.0

    return i_bank_1_index, i_bank_2_index, i_total_bank_cells, d_y_depth, d_y_bathy

def Calculate_Bathymetry_Based_on_RiverBank_Elevations(da_xs_profile1, xs1_n, da_xs_profile2,
                                                       xs2_n, ia_lc_xs1, ia_lc_xs2, dm_land_use, d_dem_low_point_elev,
                                                       d_distance_z, d_slope_use, ia_xc_r1_index_main,
                                                       ia_xc_c1_index_main, ia_xc_r2_index_main, ia_xc_c2_index_main,
                                                       d_q_baseflow, dm_output_bathymetry,
                                                       i_lc_water_value, dm_elevation, b_FindBanksBasedOnLandCover,
                                                       i_landcover_for_bathy, b_bathy_use_banks, d_bathymetry_trapzoid_height):
    """
    Calculate the bathymetry (water depth and thalweg elevation) based on river bank elevations.
    """

    # --- Local helper functions --- #

    def calc_side_distance(profile, bank_index, bankfull_elev, d_distance_z):
        """Compute the horizontal distance along a side based on elevation difference."""
        try:
            d_d_elev = profile[bank_index + 1] - profile[bank_index]
            if d_d_elev > 0:
                side_dist = d_distance_z * (bankfull_elev - profile[bank_index]) / d_d_elev
                if side_dist < 0.0 or side_dist > d_distance_z:
                    return 0.5 * d_distance_z
                return side_dist
            else:
                return 0.0
        except Exception:
            return 0.5 * d_distance_z

    def compute_depth(i_total_bank_cells, d_distance_z, da_xs_profile1, da_xs_profile2, i_bank_1_index, i_bank_2_index, d_bankfull_elevation,
                      d_bathymetry_trapzoid_height, d_q_baseflow, d_slope_use):
        
        """Compute trapezoid dimensions and the corresponding water depth."""
        d_side1_dist = calc_side_distance(da_xs_profile1, i_bank_1_index, d_bankfull_elevation, d_distance_z)
        d_side2_dist = calc_side_distance(da_xs_profile2, i_bank_2_index, d_bankfull_elevation, d_distance_z)
        d_total_bank_dist = i_total_bank_cells * d_distance_z + d_side1_dist + d_side2_dist
        d_h_dist = d_bathymetry_trapzoid_height * d_total_bank_dist
        d_trap_base = d_total_bank_dist - 2.0 * d_h_dist
        d_y_depth = find_depth_of_bathymetry(d_q_baseflow, d_trap_base, d_total_bank_dist, d_slope_use, 0.03)
        return d_side1_dist, d_side2_dist, d_total_bank_dist, d_h_dist, d_trap_base, d_y_depth

    def is_valid_number(elev):
        """Check if elev is a valid number (not None, NaN, or non-numeric)."""
        return isinstance(elev, (int, float)) and not np.isnan(elev)

    def calc_bankfull_elevation(base_elev, bank_elev_1, bank_elev_2): 
        """
        Determine the bankfull elevation based on the two bank elevation values.
        It collects all valid bank elevations that are at least base_elev.
        If both are valid, it picks the minimum one.
        If neither is valid, it defaults to base_elev.
        """
        valid_banks = [elev for elev in (bank_elev_1, bank_elev_2) if is_valid_number(elev) and elev >= base_elev]
        return min(valid_banks, default=base_elev)

    # --- End of helper functions --- #

    # Initialize variables
    function_used = None
    
    # Initially set the bank info to zeros and bank elevations to the current water surface elevation
    i_bank_1_index = 0
    i_bank_2_index = 0
    bank_elev_1 = da_xs_profile1[0]
    bank_elev_2 = da_xs_profile2[0]
    d_y_depth = 0.0

    # === First: find the bank information === #
    if b_FindBanksBasedOnLandCover:
        # Use land cover data to find the banks of the stream
        if xs1_n >= 1 and i_landcover_for_bathy == i_lc_water_value:
            bank_elev_1 = da_xs_profile1[0]
            for i in range(1, xs1_n):
                if ia_lc_xs1[i] != i_lc_water_value:
                    bank_elev_1 = da_xs_profile1[i]
                    i_bank_1_index = i - 1
                    break
        if xs2_n >= 1 and i_landcover_for_bathy == i_lc_water_value:
            bank_elev_2 = da_xs_profile2[0]
            for i in range(1, xs2_n):
                if ia_lc_xs2[i] != i_lc_water_value:
                    bank_elev_2 = da_xs_profile2[i]
                    i_bank_2_index = i - 1
                    break
        i_total_bank_cells = i_bank_1_index + i_bank_2_index  - 1
        if i_total_bank_cells <= 1:
            i_total_bank_cells = 1
        else:
            function_used = "find_wse_and_banks_by_lc"
    else:
        #Default is to determine bank locations based on the flat water within the DEM
        i_bank_1_index = find_bank(da_xs_profile1, xs1_n, d_dem_low_point_elev + 0.1)
        i_bank_2_index = find_bank(da_xs_profile2, xs2_n, d_dem_low_point_elev + 0.1)
        # set the bank elevations
        bank_elev_1 = da_xs_profile1[i_bank_1_index]
        bank_elev_2 = da_xs_profile2[i_bank_2_index]
        i_total_bank_cells = i_bank_1_index + i_bank_2_index - 1
        if i_total_bank_cells > 1:
            function_used = "find_wse_and_banks_by_flat_water"

    # Try the width-to-depth ratio method if the banks are not found
    if i_total_bank_cells <= 1:
        (i_bank_1_index, i_bank_2_index) = find_bank_using_width_to_depth_ratio(da_xs_profile1, da_xs_profile2, xs1_n, xs2_n, d_distance_z)
        bank_elev_1 = da_xs_profile1[i_bank_1_index]
        bank_elev_2 = da_xs_profile2[i_bank_2_index]
        i_total_bank_cells = i_bank_1_index + i_bank_2_index - 1
        if i_total_bank_cells <= 1:
            i_total_bank_cells = 1
        else:
            function_used = "find_bank_using_width_to_depth_ratio"

    # If still not found, try the inflection point method
    if i_total_bank_cells <= 1:
        i_bank_1_index = find_bank_inflection_point(da_xs_profile1, xs1_n, d_distance_z)
        bank_elev_1 = da_xs_profile1[i_bank_1_index]
        i_bank_2_index = find_bank_inflection_point(da_xs_profile2, xs2_n, d_distance_z)
        bank_elev_2 = da_xs_profile2[i_bank_2_index]
        i_total_bank_cells = i_bank_1_index + i_bank_2_index
        if i_total_bank_cells <= 1:
            i_total_bank_cells = 1
        else:
            function_used = "find_bank_inflection_point"

    # Calculate bankfull elevation using the base water surface elevation (first point of profile1)
    base_elev = da_xs_profile1[0]
    d_bankfull_elevation = calc_bankfull_elevation(base_elev, bank_elev_1, bank_elev_2)

    # === Estimate bathymetry depth === #
    if d_q_baseflow > 0.0 and function_used is not None:
        # Calculate trapezoid dimensions and initial depth estimate
        (d_side1_dist, d_side2_dist, d_total_bank_dist, d_h_dist,
         d_trap_base, d_y_depth) = compute_depth(
                                                 i_total_bank_cells, d_distance_z, da_xs_profile1, da_xs_profile2,
                                                 i_bank_1_index, i_bank_2_index, d_bankfull_elevation,
                                                 d_bathymetry_trapzoid_height, d_q_baseflow, d_slope_use
                                                )
        # calculate the elevation of the bathy depth and re-calculate if higher than the bankfull elevation
        d_y_bathy = d_bankfull_elevation - d_y_depth
        # If the estimated depth is an outlier, try alternate approaches
        if d_y_depth >= 25 or d_y_bathy > d_bankfull_elevation and (function_used == "find_wse_and_banks_by_lc" or
                                function_used == "find_wse_and_banks_by_flat_water"):
            # Recalculate using width-to-depth ratio
            (i_bank_1_index, i_bank_2_index) = find_bank_using_width_to_depth_ratio(da_xs_profile1, da_xs_profile2, xs1_n, xs2_n, d_distance_z)
            i_total_bank_cells = i_bank_1_index + i_bank_2_index -1
            if i_total_bank_cells <= 1:
                i_total_bank_cells = 1
            else:
                function_used = "find_bank_using_width_to_depth_ratio"
            
            # find the elevation of the banks
            bank_elev_1 = da_xs_profile1[i_bank_1_index]
            bank_elev_2 = da_xs_profile2[i_bank_2_index]
            d_bankfull_elevation = calc_bankfull_elevation(base_elev, bank_elev_1, bank_elev_2)
            (d_side1_dist, d_side2_dist, d_total_bank_dist, d_h_dist,
             d_trap_base, d_y_depth) = compute_depth(
                i_total_bank_cells, d_distance_z, da_xs_profile1, da_xs_profile2,
                i_bank_1_index, i_bank_2_index, d_bankfull_elevation,
                d_bathymetry_trapzoid_height, d_q_baseflow, d_slope_use
            )
            # calculate the elevation of the bathy depth and re-calculate if higher than the bankfull elevation
            d_y_bathy = d_bankfull_elevation - d_y_depth
            if d_y_depth >= 25 or d_y_bathy > d_bankfull_elevation:
                # Try using the inflection point method
                i_bank_1_index = find_bank_inflection_point(da_xs_profile1, xs1_n, d_distance_z)
                bank_elev_1 = da_xs_profile1[i_bank_1_index]
                i_bank_2_index = find_bank_inflection_point(da_xs_profile2, xs2_n, d_distance_z)
                bank_elev_2 = da_xs_profile2[i_bank_2_index]
                i_total_bank_cells = i_bank_1_index + i_bank_2_index - 1
                if i_total_bank_cells <= 1:
                    i_total_bank_cells = 1
                else:
                    function_used = "find_bank_inflection_point"
                d_bankfull_elevation = calc_bankfull_elevation(base_elev, bank_elev_1, bank_elev_2)
                (d_side1_dist, d_side2_dist, d_total_bank_dist, d_h_dist,
                 d_trap_base, d_y_depth) = compute_depth(
                    i_total_bank_cells, d_distance_z, da_xs_profile1, da_xs_profile2,
                    i_bank_1_index, i_bank_2_index, d_bankfull_elevation,
                    d_bathymetry_trapzoid_height, d_q_baseflow, d_slope_use
                )
                # calculate the elevation of the bathy depth and re-calculate if higher than the bankfull elevation
                d_y_bathy = d_bankfull_elevation - d_y_depth
                if d_y_depth >= 25 or d_y_bathy > d_bankfull_elevation or i_total_bank_cells <= 1:
                    d_y_depth = 0
                    d_y_bathy = da_xs_profile1[0]
                    i_bank_1_index = 0
                    i_bank_2_index = 0
                    i_total_bank_cells = 1

        elif d_y_depth >= 25 or d_y_bathy > d_bankfull_elevation and function_used == "find_bank_using_width_to_depth_ratio":
            # Use the inflection point method directly
            i_bank_1_index = find_bank_inflection_point(da_xs_profile1, xs1_n, d_distance_z)
            bank_elev_1 = da_xs_profile1[i_bank_1_index]
            i_bank_2_index = find_bank_inflection_point(da_xs_profile2, xs2_n, d_distance_z)
            bank_elev_2 = da_xs_profile2[i_bank_2_index]
            i_total_bank_cells = i_bank_1_index + i_bank_2_index -1
            if i_total_bank_cells <= 1:
                i_total_bank_cells = 1
            else:
                function_used = "find_bank_inflection_point"
            d_bankfull_elevation = calc_bankfull_elevation(base_elev, bank_elev_1, bank_elev_2)
            (d_side1_dist, d_side2_dist, d_total_bank_dist, d_h_dist,
             d_trap_base, d_y_depth) = compute_depth(
                i_total_bank_cells, d_distance_z, da_xs_profile1, da_xs_profile2,
                i_bank_1_index, i_bank_2_index, d_bankfull_elevation,
                d_bathymetry_trapzoid_height, d_q_baseflow, d_slope_use
            )
            # calculate the elevation of the bathy depth and re-calculate if higher than the bankfull elevation
            d_y_bathy = d_bankfull_elevation - d_y_depth
            if d_y_depth >= 25 or d_y_bathy > d_bankfull_elevation or i_total_bank_cells <= 1:
                d_y_depth = 0
                d_y_bathy = da_xs_profile1[0]
                i_bank_1_index = 0
                i_bank_2_index = 0
                i_total_bank_cells = 1

        elif d_y_depth >= 25 or d_y_bathy > d_bankfull_elevation and function_used == "find_bank_inflection_point":
            d_y_depth = 0
            d_y_bathy = da_xs_profile1[0]
            i_bank_1_index = 0
            i_bank_2_index = 0
            i_total_bank_cells = 1
            function_used = None

    else:
        # No valid baseflow or method; set defaults.
        d_y_depth = 0.0
        d_y_bathy = da_xs_profile1[0]
        i_bank_1_index = 0
        i_bank_2_index = 0
        i_total_bank_cells = 1
    
    # if function_used == "find_wse_and_banks_by_flat_water":
    #     i_bank_1_index = 0
    #     i_bank_2_index = 0
    #     i_total_bank_cells = 0
    #     d_y_depth = 0
    #     d_y_bathy = 0


    # --- Adjust bathymetry on both profiles if valid banks were found --- #
    if i_total_bank_cells > 1:
        # Add 1 to the bank index to get to the actual bank cell.
        adjust_profile_for_bathymetry(
            da_xs_profile1, i_bank_1_index + 1, d_total_bank_dist,
            d_trap_base, d_distance_z, d_h_dist, d_y_bathy, d_y_depth,
            dm_output_bathymetry, ia_xc_r1_index_main, ia_xc_c1_index_main,
            dm_land_use, d_side1_dist, dm_elevation,
            b_bathy_use_banks
        )
        adjust_profile_for_bathymetry(
            da_xs_profile2, i_bank_2_index + 1, d_total_bank_dist,
            d_trap_base, d_distance_z, d_h_dist, d_y_bathy, d_y_depth,
            dm_output_bathymetry, ia_xc_r2_index_main, ia_xc_c2_index_main,
            dm_land_use, d_side2_dist, dm_elevation,
            b_bathy_use_banks
        )

    return i_bank_1_index, i_bank_2_index, i_total_bank_cells, d_y_depth, d_y_bathy

# @njit(cache=True)
# def discharge_at_wse(wse, d_q_maximum, 
#                      da_xs_profile1, da_xs_profile2, xs1_n, xs2_n, 
#                      d_distance_z, n_x_section_1, n_x_section_2, d_slope_use):
#     # Compute geometry for both cross-sections.
#     A1, P1, R1, comp_n1, T1 = calculate_stream_geometry(da_xs_profile1[:xs1_n], wse, d_distance_z, n_x_section_1)
#     A2, P2, R2, comp_n2, T2 = calculate_stream_geometry(da_xs_profile2[:xs2_n], wse, d_distance_z, n_x_section_2)

#     # Aggregate the geometric properties.
#     d_a_sum = A1 + A2
#     d_p_sum = P1 + P2
#     d_t_sum = T1 + T2

#     # Compute the discharge using Manning's equation if the geometry is valid.
#     if d_a_sum > 0.0 and d_p_sum > 0.0 and d_t_sum > 0.0:
#         d_composite_n = ((comp_n1 + comp_n2) / d_p_sum) ** (2.0 / 3.0)
#         d_q = (1.0 / d_composite_n) * d_a_sum * ((d_a_sum / d_p_sum) ** (2.0 / 3.0)) * (d_slope_use ** 0.5)
#     else:
#         d_q = 0.0

#     return d_q

# @njit(cache=True)
# def find_wse(initial_wse, tolerance, max_iterations, d_q_maximum, 
#                     da_xs_profile1, da_xs_profile2, xs1_n, xs2_n, 
#                     d_distance_z, n_x_section_1, n_x_section_2, d_slope_use):
#     wse = initial_wse
#     epsilon = 1e-6  # small increment for finite difference
#     for i in range(max_iterations):
#         # Compute discharge at the current water-surface elevation
#         q = discharge_at_wse(wse, d_q_maximum, 
#                              da_xs_profile1, da_xs_profile2, xs1_n, xs2_n, 
#                              d_distance_z, n_x_section_1, n_x_section_2, d_slope_use)
#         # f(wse) = discharge(wse) - d_q_maximum
#         f_value = q - d_q_maximum

#         # Check if the solution is within the desired tolerance
#         if abs(f_value) < tolerance:
#             return wse, d_q_maximum

#         # Estimate the derivative using a finite difference
#         q_eps = discharge_at_wse(wse + epsilon, d_q_maximum, 
#                                  da_xs_profile1, da_xs_profile2, xs1_n, xs2_n, 
#                                  d_distance_z, n_x_section_1, n_x_section_2, d_slope_use)
#         f_deriv = (q_eps - q) / epsilon

#         # Protect against division by zero
#         if abs(f_deriv) < 1e-12:
#             break

#         # Update the water-surface elevation using Newton's method
#         wse = wse - f_value / f_deriv

#     # If convergence is not reached within max_iterations, return the current estimate.
#     return wse, discharge_at_wse(wse, d_q_maximum, 
#                                  da_xs_profile1, da_xs_profile2, xs1_n, xs2_n, 
#                                  d_distance_z, n_x_section_1, n_x_section_2, d_slope_use)


@njit(cache=True)
def find_wse(range_end, start_wse, increment, d_q_maximum, da_xs_profile1, da_xs_profile2, xs1_n, xs2_n, d_distance_z, ia_xc_r1_index_main, ia_xc_c1_index_main, ia_xc_r2_index_main, ia_xc_c2_index_main, n_x_section_1, n_x_section_2, d_slope_use):
    d_wse, d_q_sum = 0.0, 0.0
    prev_wse, prev_q = 0.0, 0.0  # Store previous iteration values for interpolation

    # Let us try the maximum depth increment first. If it cannot give us an answer, return
    d_wse = start_wse + range_end * increment
    A1, P1, R1, np1, T1 = calculate_stream_geometry(da_xs_profile1[:xs1_n], d_wse, d_distance_z, n_x_section_1)
    A2, P2, R2, np2, T2 = calculate_stream_geometry(da_xs_profile2[:xs2_n], d_wse, d_distance_z, n_x_section_2)

    # Aggregate the geometric properties
    d_a_sum = A1 + A2
    d_p_sum = P1 + P2
    d_t_sum = T1 + T2
    d_q_sum = 0.0

    # Estimate Manning's n and flow
    if d_a_sum > 0.0 and d_p_sum > 0.0 and d_t_sum > 0.0:
        d_composite_n = np.round(((np1 + np2) / d_p_sum)**(2 / 3), 4)
        d_q_sum = (1 / d_composite_n) * d_a_sum * (d_a_sum / d_p_sum)**(2 / 3) * d_slope_use**0.5


    if d_q_sum < d_q_maximum:
        # Even the greatest depth increment is not enough to reach the target discharge
        return d_wse, d_q_sum

    # We will try 10 increments to narrow down the search space, reducing time spent
    range_start = 1
    step = range_end // 10
    for i in range(range_end - step, 0, -step):
        d_wse = start_wse + i * increment

        A1, P1, R1, np1, T1 = calculate_stream_geometry(da_xs_profile1[:xs1_n], d_wse, d_distance_z, n_x_section_1)
        A2, P2, R2, np2, T2 = calculate_stream_geometry(da_xs_profile2[:xs2_n], d_wse, d_distance_z, n_x_section_2)

        d_a_sum = A1 + A2
        d_p_sum = P1 + P2
        d_t_sum = T1 + T2
        d_q_sum = 0.0

        if d_a_sum > 0.0 and d_p_sum > 0.0 and d_t_sum > 0.0:
            d_composite_n = round(((np1 + np2) / d_p_sum)**(2 / 3), 4)
            d_q_sum = (1 / d_composite_n) * d_a_sum * (d_a_sum / d_p_sum)**(2 / 3) * d_slope_use**0.5

        if d_q_sum > d_q_maximum:
            # The WSE is too high, keep going
            pass
        else:
            # We can start searching here
            range_start = i
            break

    for i_depthincrement in range(range_start, range_end):
        d_wse = start_wse + i_depthincrement * increment
        A1, P1, R1, np1, T1 = calculate_stream_geometry(da_xs_profile1[:xs1_n], d_wse, d_distance_z, n_x_section_1)
        A2, P2, R2, np2, T2 = calculate_stream_geometry(da_xs_profile2[:xs2_n], d_wse, d_distance_z, n_x_section_2)

        # Aggregate the geometric properties
        d_a_sum = A1 + A2
        d_p_sum = P1 + P2
        d_t_sum = T1 + T2
        d_q_sum = 0.0

        # Estimate Manning's n and flow
        if d_a_sum > 0.0 and d_p_sum > 0.0 and d_t_sum > 0.0:
            d_composite_n = np.round(((np1 + np2) / d_p_sum)**(2 / 3), 4)
            d_q_sum = (1 / d_composite_n) * d_a_sum * (d_a_sum / d_p_sum)**(2 / 3) * d_slope_use**0.5

        
        # Check for overshoot in discharge
        if d_q_sum == d_q_maximum:
            break
        elif d_q_sum > d_q_maximum:
            # If overshoot occurs at the very first increment, interpolation cannot be done
            if i_depthincrement == 1:
                break
            else:
                # Linear interpolation between previous and current values:
                # interp_wse = prev_wse + (target_q - prev_q) * (d_wse - prev_wse) / (d_q_sum - prev_q)
                interp_wse = prev_wse + (d_q_maximum - prev_q) * (d_wse - prev_wse) / (d_q_sum - prev_q)
                # Recalculate geometry and discharge at the interpolated water surface elevation
                A1, P1, R1, np1, T1 = calculate_stream_geometry(da_xs_profile1[:xs1_n], interp_wse, d_distance_z, n_x_section_1)
                A2, P2, R2, np2, T2 = calculate_stream_geometry(da_xs_profile2[:xs2_n], interp_wse, d_distance_z, n_x_section_2)
                d_a_sum = A1 + A2
                d_p_sum = P1 + P2
                d_t_sum = T1 + T2
                if d_a_sum > 0.0 and d_p_sum > 0.0 and d_t_sum > 0.0:
                    d_composite_n = np.round(((np1 + np2) / d_p_sum)**(2 / 3), 4)
                    # d_composite_n = round(d_composite_n, 3)
                    d_q_sum = (1 / d_composite_n) * d_a_sum * (d_a_sum / d_p_sum)**(2 / 3) * d_slope_use**0.5
                d_wse = interp_wse
            break

        # Save current values for the next iteration
        prev_wse = d_wse
        prev_q = d_q_sum

    return d_wse, d_q_sum


# @njit(cache=True)
# def find_wse(range_end, start_wse, increment, d_q_maximum, da_xs_profile1, da_xs_profile2, xs1_n, xs2_n, d_distance_z, ia_xc_r1_index_main, ia_xc_c1_index_main, ia_xc_r2_index_main, ia_xc_c2_index_main, n_x_section_1, n_x_section_2, d_slope_use):
#     d_wse, d_q_sum = 0.0, 0.0
#     for i_depthincrement in range(1, range_end):
#         d_wse = start_wse + i_depthincrement * increment
#         A1, P1, R1, np1, T1 = calculate_stream_geometry(da_xs_profile1[0:xs1_n], d_wse, d_distance_z, n_x_section_1)
#         A2, P2, R2, np2, T2 = calculate_stream_geometry(da_xs_profile2[0:xs2_n], d_wse, d_distance_z, n_x_section_2)

#         # Aggregate the geometric properties
#         d_a_sum = A1 + A2
#         d_p_sum = P1 + P2
#         d_t_sum = T1 + T2
#         d_q_sum = 0.0

#         # Estimate mannings n and flow
#         if d_a_sum > 0.0 and d_p_sum > 0.0 and d_t_sum > 0.0:
#             d_composite_n = math.pow(((np1 + np2) / d_p_sum), (2 / 3))
#             d_q_sum = (1 / d_composite_n) * d_a_sum * math.pow((d_a_sum / d_p_sum), (2 / 3)) * math.pow(d_slope_use, 0.5)
        
#         # Perform check on the maximum flow
#         if d_q_sum > d_q_maximum:
#             break

#     return d_wse, d_q_sum

@njit(cache=True)
def flood_increments(i_number_of_increments, d_inc_y, da_xs_profile1, da_xs_profile2, xs1_n, xs2_n,
                             d_ordinate_dist, n_x_section_1, n_x_section_2, d_slope_use, da_total_t, da_total_a,
                             da_total_p, da_total_v, da_total_q, da_total_wse):

    i_start_elevation_index, i_last_elevation_index = 0, 0

    # Initialize previous values
    prev_t = 0.0
    prev_a = 0.0
    prev_p = 0.0
    prev_q = 0.0
    prev_v = 0.0
    prev_wse = 0.0

    for i_entry_elevation in range(i_number_of_increments):
        d_wse = da_xs_profile1[0] + d_inc_y * i_entry_elevation
        d_wse = np.round(d_wse, 3)

        # Calculate the geometry          
        A1, P1, R1, np1, T1 = calculate_stream_geometry(da_xs_profile1[:xs1_n], d_wse, d_ordinate_dist, n_x_section_1)
        A2, P2, R2, np2, T2 = calculate_stream_geometry(da_xs_profile2[:xs2_n], d_wse, d_ordinate_dist, n_x_section_2)

        T = T1 + T2
        A = A1 + A2
        P = P1 + P2

        if T > 0 and A > 0 and P > 0:

            # Estimate mannings n
            d_composite_n = np.round(((np1 + np2) / P)**(2 / 3), 4)

            # use Manning's equation to estimate the flow
            Q = (1 / d_composite_n) * A * (A / P)**(2 / 3) * d_slope_use**0.5
            V = Q / A

            if Q < prev_q:
                # increase d_wse by 1 cm to try to make sure Q is greater than prev_q
                d_wse_lower_bound = d_wse + 0.01
                # set the upper bound for the water surface elevation to the next increment
                d_wse_upper_bound = da_xs_profile1[0] + d_inc_y * i_entry_elevation+1
                d_wse_upper_bound = np.round(d_wse_upper_bound, 3)
                while d_wse_lower_bound < d_wse_upper_bound:
                    # Calculate the geometry          
                    A1, P1, R1, np1, T1 = calculate_stream_geometry(da_xs_profile1[:xs1_n], d_wse_lower_bound, d_ordinate_dist, n_x_section_1)
                    A2, P2, R2, np2, T2 = calculate_stream_geometry(da_xs_profile2[:xs2_n], d_wse_lower_bound, d_ordinate_dist, n_x_section_2)

                    T = T1 + T2
                    A = A1 + A2
                    P = P1 + P2

                    # Estimate mannings n
                    d_composite_n = np.round(((np1 + np2) / P)**(2 / 3), 4)

                    # use Manning's equation to estimate the flow
                    Q = (1 / d_composite_n) * A * (A / P)**(2 / 3) * d_slope_use**0.5
                    V = Q / A

                    if A > prev_a and P > prev_p and Q > prev_q:
                        d_wse = d_wse_lower_bound
                        break
                    else:
                        d_wse_lower_bound += 0.01
                        
                # if we reach the upper bound without finding a solution, let's skip this increment
                if Q < prev_q:
                    da_total_wse[i_entry_elevation] = prev_wse
                    da_total_t[i_entry_elevation] = prev_t
                    da_total_a[i_entry_elevation] = prev_a
                    da_total_p[i_entry_elevation] = prev_p
                    da_total_q[i_entry_elevation] = prev_q
                    da_total_v[i_entry_elevation] = prev_v
                    continue

            # Save the values
            da_total_wse[i_entry_elevation] = d_wse
            da_total_t[i_entry_elevation] = T
            da_total_a[i_entry_elevation] = A
            da_total_p[i_entry_elevation] = P
            da_total_q[i_entry_elevation] = Q
            da_total_v[i_entry_elevation] = V

            # Update previous values
            prev_t = T
            prev_a = A
            prev_p = P
            prev_q = Q
            prev_v = V
            prev_wse = d_wse


            i_last_elevation_index = i_entry_elevation
        else:
            # Invalid geometry case
            da_total_t[i_entry_elevation] = 0.0
            da_total_a[i_entry_elevation] = 0.0
            da_total_p[i_entry_elevation] = 0.0
            da_total_q[i_entry_elevation] = 0.0
            da_total_v[i_entry_elevation] = 0.0
            i_start_elevation_index = i_entry_elevation

    return i_start_elevation_index, i_last_elevation_index

def modify_array(arr, b_modified_dem):
    """
    Checks and modifies the DEM if there are negative elevations in it by adding 100 to all elevations.
    """
    # Check if the array contains any negative value
    if np.any(arr < 0) and not b_modified_dem:
        # Add 100 to the entire array
        arr += 100
        b_modified_dem = True

    return arr, b_modified_dem

@njit(cache=True)
def compute_gaussian_kernel(window_size, sigma):
    """
    Compute a 2D Gaussian kernel.

    Parameters:
        window_size (int): Size of the kernel (must be odd).
        sigma (float): Standard deviation of the Gaussian.

    Returns:
        numpy.ndarray: Normalized 2D Gaussian kernel.
    """
    half_window = window_size // 2
    kernel = np.zeros((window_size, window_size))
    for i in range(-half_window, half_window + 1):
        for j in range(-half_window, half_window + 1):
            kernel[i + half_window, j + half_window] = np.exp(-(i**2 + j**2) / (2 * sigma**2))
    kernel /= np.sum(kernel)
    return kernel


@njit(cache=True)
def smooth_bathymetry_gaussian_numba(dm_output_bathymetry, window_size=7, sigma=2, n_pass=3):
    """
    Smooth a 2D array using a Gaussian filter with multiple passes, excluding NaN values.
    NaN values in the original array are preserved.

    Parameters:
        dm_output_bathymetry (numpy.ndarray): Input 2D array with float and NaN values.
        window_size (int): Size of the Gaussian kernel (must be odd).
        sigma (float): Standard deviation of the Gaussian.
        n_pass (int): Number of smoothing iterations to apply.

    Returns:
        numpy.ndarray: Smoothed 2D array with NaN values preserved.
    """
    rows, cols = dm_output_bathymetry.shape
    half_window = window_size // 2
    kernel = compute_gaussian_kernel(window_size, sigma)
    output = dm_output_bathymetry.copy()

    for _ in range(n_pass):
        temp_output = output.copy()
        for r in range(rows):
            for c in range(cols):
                if np.isnan(output[r, c]):
                    # Preserve NaN values
                    continue

                # Apply the Gaussian kernel
                weighted_sum = 0.0
                weight_total = 0.0
                for i in range(-half_window, half_window + 1):
                    for j in range(-half_window, half_window + 1):
                        nr, nc = r + i, c + j
                        if 0 <= nr < rows and 0 <= nc < cols and not np.isnan(output[nr, nc]):
                            weight = kernel[i + half_window, j + half_window]
                            weighted_sum += output[nr, nc] * weight
                            weight_total += weight

                # Normalize by total weight
                if weight_total > 0:
                    temp_output[r, c] = weighted_sum / weight_total
                else:
                    temp_output[r, c] = np.nan  # Retain NaN if no valid neighbors

        output = temp_output  # Update for the next iteration

    return output

@njit(cache=True)
def get_best_xsection_angle(l_angles_to_test, d_xs_direction, d_precompute_angles,
                          i_row_cell, i_column_cell, i_entry_cell,
                          ia_xc_dr_index_main, ia_xc_dc_index_main, ia_xc_dr_index_second, ia_xc_dc_index_second,
                          da_xs_profile1, dm_elevation, i_center_point,
                          da_xc_main_fract, da_xc_main_fract_int, da_xc_second_fract, da_xc_second_fract_int, 
                          i_row_bottom, i_row_top, i_column_bottom, i_column_top, ia_lc_xs1, ia_lc_xs2,
                          d_distance_z, da_xs_profile2)-> float:
    d_test_depth = 0.5
    d_shortest_tw_angle = 0.0
    d_t_test = np.inf

    # Some dummy values since we don't care about the Manning's n in this function
    fake_mannings = np.empty_like(da_xs_profile1, dtype=np.float32)

    # Loop through the angles to test
    for d_entry_angle_adjustment in l_angles_to_test:
        # Ensure angle is between 0 and pi
        d_xs_angle_use = (d_xs_direction + d_entry_angle_adjustment) % np.pi
    
        #We now precompute the cross-section ordinates
        i_precompute_angle_closest = int(round(d_xs_angle_use / d_precompute_angles))

        # Pull the cross-section again
        ia_xc_r1_index_main = i_row_cell + ia_xc_dr_index_main[i_precompute_angle_closest]
        ia_xc_r2_index_main = i_row_cell - ia_xc_dr_index_main[i_precompute_angle_closest]
        ia_xc_c1_index_main = i_column_cell + ia_xc_dc_index_main[i_precompute_angle_closest]
        ia_xc_c2_index_main = i_column_cell - ia_xc_dc_index_main[i_precompute_angle_closest]

        ia_xc_r1_index_second = i_row_cell + ia_xc_dr_index_second[i_precompute_angle_closest]
        ia_xc_r2_index_second = i_row_cell - ia_xc_dr_index_second[i_precompute_angle_closest]
        ia_xc_c1_index_second = i_column_cell + ia_xc_dc_index_second[i_precompute_angle_closest]
        ia_xc_c2_index_second = i_column_cell - ia_xc_dc_index_second[i_precompute_angle_closest]

        xs1_n = sample_cross_section_from_dem(i_entry_cell, da_xs_profile1, dm_elevation, i_center_point, ia_xc_r1_index_main, ia_xc_c1_index_main, ia_xc_r1_index_second,
                                            ia_xc_c1_index_second, da_xc_main_fract[i_precompute_angle_closest], da_xc_main_fract_int[i_precompute_angle_closest], da_xc_second_fract[i_precompute_angle_closest], da_xc_second_fract_int[i_precompute_angle_closest], i_row_bottom, i_row_top, i_column_bottom, i_column_top, ia_lc_xs1)
        
        d_wse = da_xs_profile1[0] + d_test_depth
        A1, P1, R1, np1, T1 = calculate_stream_geometry(da_xs_profile1[:xs1_n], d_wse, d_distance_z[i_precompute_angle_closest], fake_mannings[:xs1_n])
        if T1 > d_t_test:
            continue

        xs2_n = sample_cross_section_from_dem(i_entry_cell, da_xs_profile2, dm_elevation, i_center_point, ia_xc_r2_index_main, ia_xc_c2_index_main, ia_xc_r2_index_second,
                                            ia_xc_c2_index_second, da_xc_main_fract[i_precompute_angle_closest], da_xc_main_fract_int[i_precompute_angle_closest], da_xc_second_fract[i_precompute_angle_closest], da_xc_second_fract_int[i_precompute_angle_closest], i_row_bottom, i_row_top, i_column_bottom, i_column_top, ia_lc_xs2)
        
        A2, P2, R2, np2, T2 = calculate_stream_geometry(da_xs_profile2[:xs2_n], d_wse, d_distance_z[i_precompute_angle_closest], fake_mannings[:xs2_n])

        if (T1 + T2) < d_t_test:
            d_t_test = T1 + T2
            d_shortest_tw_angle = d_xs_angle_use

    return d_shortest_tw_angle


@profile
def main(MIF_Name: str, args: dict, quiet: bool):
    starttime = datetime.now()  
    ### Read Main Input File ###
    read_main_input_file(MIF_Name, args)
    
    ### Read the Flow Information ###
    COMID, QBaseFlow, QMax = read_flow_file(s_input_flow_file_path, s_flow_file_id, s_flow_file_baseflow, s_flow_file_qmax)

    ### Read Raster Data ###
    dm_elevation, dncols, dnrows, dcellsize, dyll, dyur, dxll, dxur, dlat, dem_geotransform, dem_projection = read_raster_gdal(s_input_dem_path)
    dm_stream, sncols, snrows, scellsize, syll, syur, sxll, sxur, slat, strm_geotransform, strm_projection = read_raster_gdal(s_input_stream_path)
    dm_land_use, lncols, lnrows, lcellsize, lyll, lyur, lxll, lxur, llat, land_geotransform, land_projection = read_raster_gdal(s_input_land_use_path)



    # if the DEM contains negative values, add 100 m to the height to get rid of the negatives, we'll subtract it back out later
    b_modified_dem = False
    dm_elevation, b_modified_dem = modify_array(dm_elevation, b_modified_dem)


    if dnrows != snrows or dnrows != lnrows:
        LOG.error('Rows do not Match!')
        return
    else:
        nrows = dnrows

    if dncols != sncols or dncols != lncols:
        LOG.error('Cols do not Match!')
        return
    else:
        ncols = dncols

    ### If you're outputting a cross-section file start the list here
    if s_xs_output_file != '':
        o_xs_file = open(s_xs_output_file, 'w')

    ### Imbed the Stream and DEM data within a larger Raster to help with the boundary issues. ###
    i_boundary_number = max(1, i_general_slope_distance, i_general_direction_distance)

    dm_stream = np.pad(dm_stream, i_boundary_number, mode='constant', constant_values=0).astype(np.int64)

    dm_elevation = np.pad(dm_elevation, i_boundary_number, mode='constant', constant_values=0)

    dm_land_use = np.pad(dm_land_use, i_boundary_number, mode='constant', constant_values=0).astype(float)
    

    ##### Begin Calculations #####
    # Create working matrices
    da_total_t = np.zeros(i_number_of_increments + 1, dtype=float)
    da_total_a = np.zeros(i_number_of_increments + 1, dtype=float)
    da_total_p = np.zeros(i_number_of_increments + 1, dtype=float)
    da_total_v = np.zeros(i_number_of_increments + 1, dtype=float)
    da_total_q = np.zeros(i_number_of_increments + 1, dtype=float)
    da_total_wse = np.zeros(i_number_of_increments + 1, dtype=float)

    # Create output rasters
    # Create an array with NaN values instead of zeros
    dm_output_bathymetry = np.full(
        (nrows + i_boundary_number * 2, ncols + i_boundary_number * 2), 
        np.nan, 
        dtype=np.float32  # Optional: Specify dtype if needed
    )
    
    # This is used for debugging purposes with stream and cross-section angles.
    #dm_output_streamangles = np.zeros((nrows + i_boundary_number * 2, ncols + i_boundary_number * 2))
    
    
    if len(s_output_flood) > 1:
        dm_out_flood = np.zeros((nrows + i_boundary_number * 2, ncols + i_boundary_number * 2)).astype(int)

    # Get the list of stream locations
    ia_valued_row_indices, ia_valued_column_indices = np.where(np.isin(dm_stream, COMID, kind='table'))

    i_number_of_stream_cells = len(ia_valued_row_indices)

    # Get the original landcover value before we start changing it
    dm_land_use_before_streams = dm_land_use
    
    # Make all Land Cover that is a stream look like water
    dm_land_use[ia_valued_row_indices,ia_valued_column_indices] = i_lc_water_value
    
    
    #Assing Manning n Values
    ### Read in the Manning Table ###
    dm_manning_n_raster = np.copy(dm_land_use)
    dm_manning_n_raster = read_manning_table(s_input_mannings_path, dm_manning_n_raster)

    # Correct the mannings values here
    dm_manning_n_raster[dm_manning_n_raster > 0.3] = 0.035
    dm_manning_n_raster[dm_manning_n_raster < 0.005] = 0.005
    

    # Get the cell dx and dy coordinates
    dx, dy, dproject = convert_cell_size(dcellsize, dyll, dyur)
    LOG.info('Cellsize X = ' + str(dx))
    LOG.info('Cellsize Y = ' + str(dy))

    i_precompute_angles = 30
    d_precompute_angles = np.pi / i_precompute_angles
    
    # Pull cross sections
    i_center_point = int((d_x_section_distance / (sum([dx, dy]) * 0.5)) / 2.0) + 1
    da_xs_profile1 = np.zeros(i_center_point + 1)
    da_xs_profile2 = np.zeros(i_center_point + 1)
    ia_lc_xs1 = np.zeros(i_center_point + 1)
    ia_lc_xs2 = np.zeros(i_center_point + 1)
    ia_xc_dr_index_main = np.zeros((i_precompute_angles + 1, i_center_point + 1), dtype=int)  # Only need to go to center point, because the other side of xs we can just use *-1
    ia_xc_dc_index_main = np.zeros((i_precompute_angles + 1, i_center_point + 1), dtype=int)  # Only need to go to center point, because the other side of xs we can just use *-1
    ia_xc_dr_index_second = np.zeros((i_precompute_angles + 1, i_center_point + 1), dtype=int)  # Only need to go to center point, because the other side of xs we can just use *-1
    ia_xc_dc_index_second = np.zeros((i_precompute_angles + 1, i_center_point + 1), dtype=int)  # Only need to go to center point, because the other side of xs we can just use *-1
    d_distance_z = np.zeros(i_precompute_angles + 1, dtype=int)
    da_xc_main_fract = np.zeros((i_precompute_angles + 1, i_center_point + 1))
    da_xc_second_fract = np.zeros((i_precompute_angles + 1, i_center_point + 1))
    da_xc_main_fract_int = np.zeros((i_precompute_angles + 1, i_center_point + 1), dtype=np.int64)
    da_xc_second_fract_int = np.zeros((i_precompute_angles + 1, i_center_point + 1), dtype=np.int64)

    for i in range(i_precompute_angles+1):
        d_xs_direction = d_precompute_angles * i
        # Get the Cross-Section Ordinates
        (d_distance_z[i], da_xc_main_fract_int[i], da_xc_second_fract_int[i]) = get_xs_index_values_precalculated(ia_xc_dr_index_main[i], ia_xc_dc_index_main[i], ia_xc_dr_index_second[i], ia_xc_dc_index_second[i], da_xc_main_fract[i], da_xc_second_fract[i], d_xs_direction,
                                                                                           i_center_point, dx, dy)

    # Find all the different angle increments to test
    l_angles_to_test = [0.0]
    d_increments = 0

    if d_degree_manipulation > 0.0 and d_degree_interval > 0.0:
        # Calculate the increment
        d_increments = int(d_degree_manipulation / (2.0 * d_degree_interval))

        # Test if the increment should be considered
        if d_increments > 0:
            for d in range(1, d_increments + 1):
                for s in range(-1, 2, 2):
                    l_angles_to_test.append(s * d * d_degree_interval)

    LOG.info('With Degree_Manip=' + str(d_degree_manipulation) + '  and  Degree_Interval=' + str(d_degree_interval) + '\n  Angles to evaluate= ' + str(l_angles_to_test))
    l_angles_to_test = np.multiply(l_angles_to_test, math.pi / 180.0)
    LOG.info('  Angles (radians) to evaluate= ' + str(l_angles_to_test))


    # Get the extents of the boundaries
    i_row_bottom = i_boundary_number
    i_row_top = nrows + i_boundary_number-1
    i_column_bottom = i_boundary_number
    i_column_top = ncols + i_boundary_number-1

    # This is now a model input
    # These are the number of increments of water surface elevation that we will use to construct the VDT database and the 
    #i_number_of_increments = 15

    # Here we will capture a list of all stream cell values that will be used if we build a reach average curve file
    if s_output_curve_file and b_reach_average_curve_file:
        All_COMID_curve_list = []
        All_Row_curve_list = []
        All_Col_curve_list = []
        All_BaseElev_curve_list = []
        All_DEM_Elev_curve_list = []
        All_QMax_curve_list = []
    
    # Create the lists that will be used to create our VDT database
    vdt_list: list = []
    q_list = []
    v_list = []
    t_list = []
    wse_list = []

    # Create the dictionary and lists that will be used to create our ATW database
    if s_output_ap_database:
        o_ap_file_dict: dict[str, list] = {}
        o_ap_file_dict['COMID'] = []
        o_ap_file_dict['Row'] = []
        o_ap_file_dict['Col'] = []
        comid_ap_dict_list = o_ap_file_dict['COMID']
        row_ap_dict_list = o_ap_file_dict['Row']
        col_ap_dict_list = o_ap_file_dict['Col']
        for i in range(1, i_number_of_increments+1):
            o_ap_file_dict[f'q_{i}'] = []
            o_ap_file_dict[f'a_{i}'] = []
            o_ap_file_dict[f'p_{i}'] = []
    
    #Create the list that we will use to generate the output Curve file
    if s_output_curve_file:
        COMID_curve_list = []
        Row_curve_list = []
        Col_curve_list = []
        BaseElev_curve_list = []
        DEM_Elev_curve_list = []
        QMax_curve_list = []
        depth_a_curve_list = []
        depth_b_curve_list = []
        tw_a_curve_list = []
        tw_b_curve_list = []
        vel_a_curve_list = []
        vel_b_curve_list = []

    # Write the percentiles into the files
    LOG.info('Looking at ' + str(i_number_of_stream_cells) + ' stream cells')

    # Creating strings takes a lot of time. Let's compute the AWP database column names here
    if len(s_output_ap_database) > 0:
        ap_column_names = []
        for i in range(1, i_number_of_increments+1):
            ap_column_names.append((f'q_{i}', f'a_{i}', f'p_{i}'))

    # Solve using the volume fill approach
    i_volume_fill_approach = 1

    d_depth_increment_big = 0.5
    d_depth_increment_med = 0.05
    d_depth_increment_small = 0.01

    ### Begin the stream cell solution loop ###
    pbar = tqdm.tqdm(range(i_number_of_stream_cells), total=i_number_of_stream_cells, disable=quiet)
    for i_entry_cell in pbar:
        # i_entry_cell = 3184282


        # pbar.disable = True
        
        # Get the metadata for the loop
        i_row_cell = ia_valued_row_indices[i_entry_cell]
        i_column_cell = ia_valued_column_indices[i_entry_cell]
        i_cell_comid = dm_stream[i_row_cell,i_column_cell]


        # Get the Flow Rates Associated with the Stream Cell
        try:
            im_flow_index = np.where(COMID == i_cell_comid)[0][0]
            # im_flow_index = np.where(COMID == int(dm_stream[i_row_cell, i_column_cell]))
            # print("This is the flow index: ", im_flow_index)
            d_q_baseflow = QBaseFlow[im_flow_index]
            d_q_maximum = QMax[im_flow_index]
        except:
            # print("I cant find the flow index")
            continue

        # Get the Stream Direction of each Stream Cell.  Direction is between 0 and pi.  Also get the cross-section direction (also between 0 and pi)
        d_stream_direction, d_xs_direction = get_stream_direction_information(i_row_cell, i_column_cell, dm_stream, dx, dy)
        #dm_output_streamangles[i_row_cell,i_column_cell] = d_xs_direction * 180.0 / math.pi

        # Get the Slope of each Stream Cell. Slope should be in m/m
        d_slope_use = get_stream_slope_information(i_row_cell, i_column_cell, dm_elevation, dm_stream, dx, dy)

        #We now precompute the cross-section ordinates
        if d_xs_direction > np.pi:
            i_precompute_angle_closest = int(round((d_xs_direction-np.pi) / d_precompute_angles))
        else:
            i_precompute_angle_closest = int(round(d_xs_direction / d_precompute_angles))

        # Now Pull a Cross-Section
        ia_xc_r1_index_main = i_row_cell + ia_xc_dr_index_main[i_precompute_angle_closest]
        ia_xc_r2_index_main = i_row_cell + ia_xc_dr_index_main[i_precompute_angle_closest] * -1
        ia_xc_c1_index_main = i_column_cell + ia_xc_dc_index_main[i_precompute_angle_closest]
        ia_xc_c2_index_main = i_column_cell + ia_xc_dc_index_main[i_precompute_angle_closest] * -1

        # todo: These appear to be resetting the center point only?
        xs1_n = sample_cross_section_from_dem(i_entry_cell, da_xs_profile1, dm_elevation, i_center_point, ia_xc_r1_index_main, ia_xc_c1_index_main, i_row_cell + ia_xc_dr_index_second[i_precompute_angle_closest],
                                            i_column_cell + ia_xc_dc_index_second[i_precompute_angle_closest], da_xc_main_fract[i_precompute_angle_closest], da_xc_main_fract_int[i_precompute_angle_closest], da_xc_second_fract[i_precompute_angle_closest], da_xc_second_fract_int[i_precompute_angle_closest], i_row_bottom, i_row_top, i_column_bottom, i_column_top, ia_lc_xs1, dm_land_use)
        xs2_n = sample_cross_section_from_dem(i_entry_cell, da_xs_profile2, dm_elevation, i_center_point, ia_xc_r2_index_main, ia_xc_c2_index_main, i_row_cell + ia_xc_dr_index_second[i_precompute_angle_closest] * -1,
                                            i_column_cell + ia_xc_dc_index_second[i_precompute_angle_closest] * -1, da_xc_main_fract[i_precompute_angle_closest], da_xc_main_fract_int[i_precompute_angle_closest], da_xc_second_fract[i_precompute_angle_closest], da_xc_second_fract_int[i_precompute_angle_closest], i_row_bottom, i_row_top, i_column_bottom, i_column_top, ia_lc_xs2, dm_land_use)

        # Adjust to the lowest-point in the Cross-Section
        i_lowest_point_index_offset=0
        d_dem_low_point_elev = da_xs_profile1[0]
        if i_low_spot_range > 0:
            i_lowest_point_index_offset = adjust_cross_section_to_lowest_point(i_lowest_point_index_offset, d_dem_low_point_elev, da_xs_profile1, da_xs_profile2, ia_xc_r1_index_main,
                                                                            ia_xc_r2_index_main, ia_xc_c1_index_main, ia_xc_c2_index_main, xs1_n, xs2_n, i_center_point, i_low_spot_range)


        # The r and c for the stream cell is adjusted because it may have moved
        i_row_cell = ia_xc_r1_index_main[0]
        i_column_cell = ia_xc_c1_index_main[0]

        # re-sample the cross-section to make sure all of the low-spot data has the same values through interpolation
        if abs(i_lowest_point_index_offset) > 0:
            xs1_n = sample_cross_section_from_dem(i_entry_cell, da_xs_profile1, dm_elevation, i_center_point, ia_xc_r1_index_main, ia_xc_c1_index_main, i_row_cell + ia_xc_dr_index_second[i_precompute_angle_closest],
                                                    i_column_cell + ia_xc_dc_index_second[i_precompute_angle_closest], da_xc_main_fract[i_precompute_angle_closest], da_xc_main_fract_int[i_precompute_angle_closest], da_xc_second_fract[i_precompute_angle_closest], da_xc_second_fract_int[i_precompute_angle_closest], i_row_bottom, i_row_top, i_column_bottom, i_column_top, ia_lc_xs1, dm_land_use)
            xs2_n = sample_cross_section_from_dem(i_entry_cell, da_xs_profile2,  dm_elevation, i_center_point, ia_xc_r2_index_main, ia_xc_c2_index_main, i_row_cell + ia_xc_dr_index_second[i_precompute_angle_closest] * -1,
                                                    i_column_cell + ia_xc_dc_index_second[i_precompute_angle_closest] * -1, da_xc_main_fract[i_precompute_angle_closest], da_xc_main_fract_int[i_precompute_angle_closest], da_xc_second_fract[i_precompute_angle_closest], da_xc_second_fract_int[i_precompute_angle_closest], i_row_bottom, i_row_top, i_column_bottom, i_column_top, ia_lc_xs2, dm_land_use)
            # set the low-spot value to the new low spot in the cross-section 
            d_dem_low_point_elev = da_xs_profile1[0]
        
        # Adjust cross-section angle to ensure shortest top-width at a specified depth
        if d_increments > 0:
            d_xs_direction = get_best_xsection_angle(l_angles_to_test, d_xs_direction, d_precompute_angles,
                          i_row_cell, i_column_cell, i_entry_cell,
                          ia_xc_dr_index_main, ia_xc_dc_index_main, ia_xc_dr_index_second, ia_xc_dc_index_second,
                          da_xs_profile1, dm_elevation, i_center_point,
                          da_xc_main_fract, da_xc_main_fract_int, da_xc_second_fract, da_xc_second_fract_int, 
                          i_row_bottom, i_row_top, i_column_bottom, i_column_top, ia_lc_xs1, ia_lc_xs2,
                          d_distance_z, da_xs_profile2)

            # Now rerun everything with the shortest top-width angle
            #(d_distance_z, da_xc_main_fract_int, da_xc_second_fract_int) = get_xs_index_values(i_entry_cell, ia_xc_dr_index_main, ia_xc_dc_index_main, ia_xc_dr_index_second, ia_xc_dc_index_second, da_xc_main_fract, da_xc_second_fract, d_xs_direction, #THIS USES THE UPDATED CROSS-SECTION ANGLE!!!!!!!!!
            #                                                                                   i_row_cell, i_column_cell, i_center_point, dx, dy)
            #We now precompute the cross-section ordinates
            if d_xs_direction > np.pi:
                i_precompute_angle_closest = int(round((d_xs_direction-np.pi) / d_precompute_angles))
            else:
                i_precompute_angle_closest = int(round(d_xs_direction / d_precompute_angles))

            ia_xc_r1_index_main = i_row_cell + ia_xc_dr_index_main[i_precompute_angle_closest]
            ia_xc_r2_index_main = i_row_cell + ia_xc_dr_index_main[i_precompute_angle_closest] * -1
            ia_xc_c1_index_main = i_column_cell + ia_xc_dc_index_main[i_precompute_angle_closest]
            ia_xc_c2_index_main = i_column_cell + ia_xc_dc_index_main[i_precompute_angle_closest] * -1
            
            #Set the index values to be within the confines of the raster
            ia_xc_r1_index_main = np.clip(ia_xc_r1_index_main,0,nrows+2*i_boundary_number-1)
            ia_xc_r2_index_main = np.clip(ia_xc_r2_index_main,0,nrows+2*i_boundary_number-1)
            ia_xc_c1_index_main = np.clip(ia_xc_c1_index_main,0,ncols+2*i_boundary_number-1)
            ia_xc_c2_index_main = np.clip(ia_xc_c2_index_main,0,ncols+2*i_boundary_number-1)

            xs1_n = sample_cross_section_from_dem(i_entry_cell, da_xs_profile1, dm_elevation, i_center_point, ia_xc_r1_index_main, ia_xc_c1_index_main, i_row_cell + ia_xc_dr_index_second[i_precompute_angle_closest],
                                                  i_column_cell + ia_xc_dc_index_second[i_precompute_angle_closest], da_xc_main_fract[i_precompute_angle_closest], da_xc_main_fract_int[i_precompute_angle_closest], da_xc_second_fract[i_precompute_angle_closest], da_xc_second_fract_int[i_precompute_angle_closest], i_row_bottom, i_row_top, i_column_bottom, i_column_top, ia_lc_xs1, dm_land_use)
            xs2_n = sample_cross_section_from_dem(i_entry_cell, da_xs_profile2, dm_elevation, i_center_point, ia_xc_r2_index_main, ia_xc_c2_index_main, i_row_cell + ia_xc_dr_index_second[i_precompute_angle_closest] * -1,
                                              i_column_cell + ia_xc_dc_index_second[i_precompute_angle_closest] * -1, da_xc_main_fract[i_precompute_angle_closest], da_xc_main_fract_int[i_precompute_angle_closest], da_xc_second_fract[i_precompute_angle_closest], da_xc_second_fract_int[i_precompute_angle_closest], i_row_bottom, i_row_top, i_column_bottom, i_column_top, ia_lc_xs2, dm_land_use)
        
        # Burn bathymetry profile into cross-section profile
        # "Be the banks for your river" - Needtobreathe
                
        # If you don't have a cross-section, skip it or fill in empty values for the reach average processing
        if xs1_n<=0 and xs2_n<=0:
            # print("I'm skipping because xs1_n<=0 and xs2_n<=0")
            if b_reach_average_curve_file:
                All_COMID_curve_list.append(i_cell_comid)
                All_Row_curve_list.append(i_row_cell)
                All_Col_curve_list.append(i_column_cell)
                All_BaseElev_curve_list.append(round(dm_elevation[i_row_cell,i_column_cell], 3))
                All_DEM_Elev_curve_list.append(round(dm_elevation[i_row_cell,i_column_cell], 3))
                All_QMax_curve_list.append(d_q_maximum)
            continue

        # pull the landcover data prior to making the streams cells all water 
        xs1_n = sample_cross_section_from_dem(i_entry_cell, da_xs_profile1, dm_elevation, i_center_point, ia_xc_r1_index_main, ia_xc_c1_index_main, i_row_cell + ia_xc_dr_index_second[i_precompute_angle_closest],
                                                i_column_cell + ia_xc_dc_index_second[i_precompute_angle_closest], da_xc_main_fract[i_precompute_angle_closest], da_xc_main_fract_int[i_precompute_angle_closest], da_xc_second_fract[i_precompute_angle_closest], da_xc_second_fract_int[i_precompute_angle_closest], i_row_bottom, i_row_top, i_column_bottom, i_column_top, ia_lc_xs1, dm_land_use_before_streams)

        # now switch the values back
        xs1_n = sample_cross_section_from_dem(i_entry_cell, da_xs_profile1, dm_elevation, i_center_point, ia_xc_r1_index_main, ia_xc_c1_index_main, i_row_cell + ia_xc_dr_index_second[i_precompute_angle_closest],
                                                i_column_cell + ia_xc_dc_index_second[i_precompute_angle_closest], da_xc_main_fract[i_precompute_angle_closest], da_xc_main_fract_int[i_precompute_angle_closest], da_xc_second_fract[i_precompute_angle_closest], da_xc_second_fract_int[i_precompute_angle_closest], i_row_bottom, i_row_top, i_column_bottom, i_column_top, ia_lc_xs1, dm_land_use)
        
        #BATHYMETRY CALCULATION
        #This method calculates bathymetry based on the water surface elevation or LandCover ("FindBanksBasedOnLandCover" and "LC_Water_Value").
        if not b_bathy_use_banks and s_output_bathymetry_path != '':
            (i_bank_1_index, i_bank_2_index, i_total_bank_cells, d_y_depth, d_y_bathy) = Calculate_Bathymetry_Based_on_WSE_or_LC(da_xs_profile1, xs1_n, da_xs_profile2, xs2_n, ia_lc_xs1, ia_lc_xs2, dm_land_use, d_dem_low_point_elev, d_distance_z[i_precompute_angle_closest], d_slope_use,
                                                                                                                                                ia_xc_r1_index_main, ia_xc_c1_index_main, ia_xc_r2_index_main, ia_xc_c2_index_main, d_q_baseflow, dm_output_bathymetry, i_lc_water_value,
                                                                                                                                                dm_elevation, b_FindBanksBasedOnLandCover, b_bathy_use_banks, d_bathymetry_trapzoid_height)
        #This method calculates the banks based on the Riverbank
        elif b_bathy_use_banks and s_output_bathymetry_path != '':
            (i_bank_1_index, i_bank_2_index, i_total_bank_cells, d_y_depth, d_y_bathy) = Calculate_Bathymetry_Based_on_RiverBank_Elevations(da_xs_profile1, xs1_n, da_xs_profile2, xs2_n, ia_lc_xs1, ia_lc_xs2, dm_land_use, d_dem_low_point_elev, d_distance_z[i_precompute_angle_closest], d_slope_use, 
                                                                                                                                 ia_xc_r1_index_main, ia_xc_c1_index_main, ia_xc_r2_index_main, ia_xc_c2_index_main, d_q_baseflow, dm_output_bathymetry, i_lc_water_value, dm_elevation, 
                                                                                                                                 b_FindBanksBasedOnLandCover, ia_lc_xs1[0], b_bathy_use_banks, d_bathymetry_trapzoid_height)


        # Get a list of elevations within the cross-section profile that we need to evaluate
        if i_volume_fill_approach==2:
            da_elevation_list_mm = np.unique(np.concatenate((da_xs_profile1[0:xs1_n] * 1000, da_xs_profile2[0:xs2_n] * 1000)).astype(int))
            da_elevation_list_mm = da_elevation_list_mm[np.logical_and(da_elevation_list_mm[:] > 0, da_elevation_list_mm[:] < 99999900)]
            da_elevation_list_mm = np.sort(da_elevation_list_mm)
            
            da_elevation_list_mm = Create_List_of_Elevations_within_CrossSection((da_xs_profile1[0:xs1_n]*100).astype(int), xs1_n, (da_xs_profile2[0:xs2_n]*100).astype(int), xs2_n)
        
            i_number_of_elevations = len(da_elevation_list_mm)
            if i_number_of_elevations <= 0:
                # print("I'm skipping because i_number_of_elevations <= 0")
                if b_reach_average_curve_file:
                    All_COMID_curve_list.append(i_cell_comid)
                    All_Row_curve_list.append(i_row_cell)
                    All_Col_curve_list.append(i_column_cell)
                    All_BaseElev_curve_list.append(round(dm_elevation[i_row_cell,i_column_cell], 3))
                    All_DEM_Elev_curve_list.append(round(dm_elevation[i_row_cell,i_column_cell], 3))
                    All_QMax_curve_list.append(d_q_maximum)
                    continue
                else:
                    continue
    
            if i_number_of_elevations >= ep:
                LOG.error('ERROR, HAVE TOO MANY ELEVATIONS TO EVALUATE')
                # print("I'm skipping because i_number_of_elevations >= ep")
                if b_reach_average_curve_file:
                    All_COMID_curve_list.append(i_cell_comid)
                    All_Row_curve_list.append(i_row_cell)
                    All_Col_curve_list.append(i_column_cell)
                    All_BaseElev_curve_list.append(round(dm_elevation[i_row_cell,i_column_cell], 3))
                    All_DEM_Elev_curve_list.append(round(dm_elevation[i_row_cell,i_column_cell], 3))
                    All_QMax_curve_list.append(d_q_maximum)
                    continue
                else:
                    continue
        
        # Calculate the volumes
        # VolumeFillApproach 1 is to find the height within ElevList_mm that corresponds to the Qmax flow.  THen increment depths to have a standard number of depths to get to Qmax.  
        # This is preferred for VDTDatabase method.
        
        # VolumeFillApproach 2 just looks at the different elevation points wtihin ElevList_mm.  It also adds some in if the gaps between depths is too large.
        
        
        #This is the Stream Cell Location
        if s_output_flood:
            dm_out_flood[i_row_cell,i_column_cell] = 3
        

        # Set output arrays to zero
        da_total_t *= 0
        da_total_a *= 0
        da_total_p *= 0
        da_total_v *= 0
        da_total_q *= 0
        da_total_wse *= 0
        
        # This just tells the curve file whether to print out a result or not.  If no realistic depths were calculated, no reason to output results.
        b_outprint_yes = False
        
        # This is the first and last indice of elevations we'll need for the Curve Fitting for this cell
        i_start_elevation_index = -1
        i_last_elevation_index = 0
        
        
        if i_volume_fill_approach==1:
            # Here are the n values for each side of the cross-section
            n_x_section_1 = dm_manning_n_raster[ia_xc_r1_index_main[0:xs1_n], ia_xc_c1_index_main[0:xs1_n]]
            n_x_section_2 = dm_manning_n_raster[ia_xc_r2_index_main[0:xs2_n], ia_xc_c2_index_main[0:xs2_n]]

            # space between ordinates in the cross-section
            d_ordinate_dist = d_distance_z[i_precompute_angle_closest]

            # we'll assume the results are acceptable until we think otherwise
            acceptable = True

            # This is the bottom of the channel
            d_maxflow_wse_initial = da_xs_profile1[0]

            # set this as the default in case we don't find a better one
            d_maxflow_wse_final = -999.0

            # Define an objective function: the difference between the calculated max flow and d_q_maximum.
            def objective_with_wse(trial_wse):

                trial_wse = np.round(trial_wse, 3)
                
                # Calculate the geometry
                A1, P1, R1, np1, T1 = calculate_stream_geometry(da_xs_profile1[0:xs1_n], trial_wse, d_ordinate_dist, n_x_section_1)
                A2, P2, R2, np2, T2 = calculate_stream_geometry(da_xs_profile2[0:xs2_n], trial_wse, d_ordinate_dist, n_x_section_2)

                # Aggregate the geometric properties
                d_a_sum = A1 + A2
                d_p_sum = max(P1 + P2, 1e-6)  # Avoid division by zero

                d_composite_n = np.round(((np1 + np2) / d_p_sum)**(2 / 3), 4)

                # Check that the mannings n is physically realistic
                if d_composite_n < 0.0001:
                    d_composite_n = 0.035

                trial_d_q_sum = (1 / d_composite_n) * d_a_sum * (d_a_sum / d_p_sum)**(2 / 3) * d_slope_use**0.5


                # trial_d_q_sum = round(trial_d_q_sum, 3)
                difference = trial_d_q_sum - d_q_maximum

                # The objective is zero when trial_d_q_sum equals d_q_maximum.
                return difference
            
            # Define an objective function: the difference between the calculated max flow and d_q_maximum.
            def objective_with_slope(trial_slope):
                # find_wse returns a tuple: (d_maxflow_wse_final, d_q_sum)
                _, trial_d_q_sum = find_wse(
                    2501, 
                    d_maxflow_wse_initial, 
                    d_depth_increment_small, 
                    d_q_maximum, 
                    da_xs_profile1, 
                    da_xs_profile2,
                    xs1_n,
                    xs2_n,
                    d_ordinate_dist, 
                    ia_xc_r1_index_main, 
                    ia_xc_c1_index_main, 
                    ia_xc_r2_index_main, 
                    ia_xc_c2_index_main, 
                    n_x_section_1, 
                    n_x_section_2, 
                    trial_slope
                )
                # The objective is zero when trial_d_q_sum equals d_q_maximum.
                return trial_d_q_sum - d_q_maximum

            wse_lower = d_maxflow_wse_initial + 0.01
            wse_upper = d_maxflow_wse_initial + 24.99

            # Check if the objective function changes sign between the bounds.
            f_lower = objective_with_wse(wse_lower)
            f_upper = objective_with_wse(wse_upper)

            if safe_signs_differ(f_lower, f_upper):
                # if i_entry_cell == 3184282:
                #     print(f"F_lower = {str(f_lower)}")
                #     print(f"F_upper = {str(f_upper)}")
                # The signs differ, so we have a valid bracket.
                # For 3 decimal places, xtol only needs to be 0.001
                d_maxflow_wse_final = np.round(brentq(objective_with_wse, wse_lower, wse_upper, xtol=0.001), 3)

                A1, P1, R1, np1, T1 = calculate_stream_geometry(da_xs_profile1[:xs1_n], d_maxflow_wse_final, d_ordinate_dist, n_x_section_1)
                A2, P2, R2, np2, T2 = calculate_stream_geometry(da_xs_profile2[:xs2_n], d_maxflow_wse_final, d_ordinate_dist, n_x_section_2)

                # Aggregate the geometric properties
                d_a_sum = A1 + A2
                d_p_sum = P1 + P2

                d_composite_n = np.round(((np1 + np2) / d_p_sum)**(2 / 3), 4)

                # Check that the mannings n is physically realistic
                if d_composite_n < 0.0001:
                    d_composite_n = 0.035

                d_q_sum = (1 / d_composite_n) * d_a_sum * (d_a_sum / d_p_sum)**(2 / 3) * d_slope_use**0.5 

            # if the f_lower or f_upper is equal to zero, it's probably close enough to be the WSE we are looking for, so we'll use it
            elif np.round(f_lower, 5) == 0 or np.round(f_upper, 5) == 0:          
                d_maxflow_wse_final = np.round(wse_lower, 3) if np.round(f_lower, 5) == 0 else np.round(wse_upper, 3)
                A1, P1, _, np1, T1 = calculate_stream_geometry(da_xs_profile1[:xs1_n], d_maxflow_wse_final, d_ordinate_dist, n_x_section_1)
                A2, P2, _, np2, T2 = calculate_stream_geometry(da_xs_profile2[:xs2_n], d_maxflow_wse_final, d_ordinate_dist, n_x_section_2)

                # Aggregate the geometric properties
                d_a_sum = A1 + A2
                d_p_sum = P1 + P2

                d_composite_n = np.round(((np1 + np2) / d_p_sum)**(2 / 3), 4)

                # Check that the mannings n is physically realistic
                if d_composite_n < 0.0001:
                    d_composite_n = 0.035

                d_q_sum = (1 / d_composite_n) * d_a_sum * (d_a_sum / d_p_sum)**(2 / 3) * d_slope_use**0.5 


            # Let's see if the volume-fill approach gave us a better answer and use that if it did
            # To find the depth / wse where the maximum flow occurs we use two sets of incremental depths.  The first is 0.5m followed by 0.05m


            d_maxflow_wse_initial, d_q_sum_test = find_wse(101, d_maxflow_wse_initial, d_depth_increment_big, d_q_maximum, da_xs_profile1, da_xs_profile2, xs1_n, xs2_n, d_ordinate_dist, ia_xc_r1_index_main, ia_xc_c1_index_main, ia_xc_r2_index_main, ia_xc_c2_index_main, n_x_section_1, n_x_section_2, d_slope_use)
            
            # Based on using depth increments of 0.5, now lets fine-tune the wse using depth increments of 0.05
            d_maxflow_wse_initial = max(d_maxflow_wse_initial - 0.5, da_xs_profile1[0])
            d_maxflow_wse_med = d_maxflow_wse_initial
            d_maxflow_wse_med, d_q_sum_test = find_wse(101, d_maxflow_wse_med, d_depth_increment_med, d_q_maximum, da_xs_profile1, da_xs_profile2, xs1_n, xs2_n, d_ordinate_dist, ia_xc_r1_index_main, ia_xc_c1_index_main, ia_xc_r2_index_main, ia_xc_c2_index_main, n_x_section_1, n_x_section_2, d_slope_use)

            # Based on using depth increments of 0.05, now lets fine-tune the wse even more using depth increments of 0.01
            d_maxflow_wse_med = max(d_maxflow_wse_med - 0.05, da_xs_profile1[0])
            d_maxflow_wse_final_test = d_maxflow_wse_med
            d_maxflow_wse_final_test, d_q_sum_test = find_wse(2501, d_maxflow_wse_med, d_depth_increment_small, d_q_maximum, da_xs_profile1, da_xs_profile2, xs1_n, xs2_n, d_ordinate_dist, ia_xc_r1_index_main, ia_xc_c1_index_main, ia_xc_r2_index_main, ia_xc_c2_index_main, n_x_section_1, n_x_section_2, d_slope_use)

            # let's see if the iterative method gave use a better result and use that if it did
            if abs(d_q_sum_test - d_q_maximum) < abs(d_q_sum-d_q_maximum):
                d_maxflow_wse_final = d_maxflow_wse_final_test
                d_q_sum = d_q_sum_test

            # here we will see if we can get a better answer with a revised slope
            # from our Missouri study, relative DEM error was around 0.70, so dividing that by our d_ordinate_dist gives us a round about
            # idea of potential error in slope.  We'll use this to adjust the slope and see if we can get a fit.
            potential_slope_error = 0.6 / d_ordinate_dist
            
            # Set lower and upper bounds for the slope search.
            slope_lower = max(d_slope_use - potential_slope_error, 1e-8) # Avoids domain error, taking sqrt of negative number, in find wse
            slope_upper = d_slope_use + potential_slope_error

            # Check if the objective function changes sign between the bounds.
            f_lower = objective_with_slope(slope_lower)
            f_upper = objective_with_slope(slope_upper)
            if safe_signs_differ(f_lower, f_upper):
                # The signs differ, so we have a valid bracket.
                # Needs xtol of 0.0001 to get to 3 decimal places
                trial_slope_use = brentq(objective_with_slope, slope_lower, slope_upper, xtol=0.0001)
                trial_slope_use = np.round(trial_slope_use, 3)
                # Optionally, recompute d_maxflow_wse_final and d_q_sum with the new slope:
                d_maxflow_wse_final_test, d_q_sum_test = find_wse(
                    2501, 
                    d_maxflow_wse_initial, 
                    d_depth_increment_small, 
                    d_q_maximum, 
                    da_xs_profile1, 
                    da_xs_profile2, 
                    xs1_n, 
                    xs2_n,
                    d_ordinate_dist, 
                    ia_xc_r1_index_main, 
                    ia_xc_c1_index_main, 
                    ia_xc_r2_index_main, 
                    ia_xc_c2_index_main, 
                    n_x_section_1, 
                    n_x_section_2, 
                    trial_slope_use
                )
                # Check if d_q_sum is within acceptable bounds
                if abs(d_q_sum_test - d_q_maximum) < abs(d_q_sum-d_q_maximum):
                    # Optionally update d_slope_use to the accepted value:
                    d_slope_use = trial_slope_use
                    d_maxflow_wse_final = d_maxflow_wse_final_test
                    d_q_sum = d_q_sum_test
                else:
                    pass
            # if the f_lower or f_upper is equal to zero, it's probably close enough to be the WSE we are looking for, so we'll use it
            elif np.round(f_lower, 5) == 0 or np.round(f_upper, 5) == 0:          
                trial_slope_use = np.round(slope_lower, 3) if np.round(f_lower, 5) == 0 else np.round(slope_upper, 3)
                # Optionally, recompute d_maxflow_wse_final and d_q_sum with the new slope:
                d_maxflow_wse_final_test, d_q_sum_test = find_wse(
                    2501, 
                    d_maxflow_wse_initial, 
                    d_depth_increment_small, 
                    d_q_maximum, 
                    da_xs_profile1, 
                    da_xs_profile2, 
                    xs1_n, 
                    xs2_n,
                    d_ordinate_dist, 
                    ia_xc_r1_index_main, 
                    ia_xc_c1_index_main, 
                    ia_xc_r2_index_main, 
                    ia_xc_c2_index_main, 
                    n_x_section_1, 
                    n_x_section_2, 
                    trial_slope_use
                )
                # Check if d_q_sum is within acceptable bounds
                if abs(d_q_sum_test - d_q_maximum) < abs(d_q_sum-d_q_maximum):
                    # Optionally update d_slope_use to the accepted value:
                    d_slope_use = trial_slope_use
                    d_maxflow_wse_final = d_maxflow_wse_final_test
                    d_q_sum = d_q_sum_test
                else:
                    pass


            #If the max flow calculated from the cross-section is 50% high or low, let's try changing the slope
            if d_q_sum > d_q_maximum * 1.5 or d_q_sum < d_q_maximum * 0.5:

                # print("I'm here because d_q_sum > d_q_maximum * 1.5 or d_q_sum < d_q_maximum * 0.5")
                # something isn't good with our results
                acceptable = False

                # here we will see if we can get a better answer with a revised slope
                # from our Missouri study, relative DEM error was around 0.70, so dividing that by our d_ordinate_dist gives us a round about
                # idea of potential error in slope.  We'll use this to adjust the slope and see if we can get a fit.
                potential_slope_error = 0.6 / d_ordinate_dist

                # Set lower and upper bounds for the slope search.
                slope_lower = max(d_slope_use - potential_slope_error, 1e-8) # Avoids domain error, taking sqrt of negative number, in find wse
                slope_upper = d_slope_use + potential_slope_error

                # Check if the objective function changes sign between the bounds.
                f_lower = objective_with_slope(slope_lower)
                f_upper = objective_with_slope(slope_upper)


                if safe_signs_differ(f_lower, f_upper):
                    # The signs differ, so we have a valid bracket.
                    new_slope = brentq(objective_with_slope, slope_lower, slope_upper, xtol=0.0001)
                    trial_slope_use = new_slope
                    # Optionally, recompute d_maxflow_wse_final and d_q_sum with the new slope:
                    d_maxflow_wse_final_test, d_q_sum_test = find_wse(
                        2501, 
                        d_maxflow_wse_initial, 
                        d_depth_increment_small, 
                        d_q_maximum, 
                        da_xs_profile1, 
                        da_xs_profile2, 
                        xs1_n, 
                        xs2_n,
                        d_ordinate_dist, 
                        ia_xc_r1_index_main, 
                        ia_xc_c1_index_main, 
                        ia_xc_r2_index_main, 
                        ia_xc_c2_index_main, 
                        n_x_section_1, 
                        n_x_section_2, 
                        trial_slope_use
                    )
                    # Check if d_q_sum is within acceptable bounds
                    if d_q_maximum * 0.5 <= d_q_sum_test <= d_q_maximum * 1.5:
                        acceptable = True
                        d_slope_use = trial_slope_use
                        d_maxflow_wse_final = d_maxflow_wse_final_test
                        d_q_sum = d_q_sum_test
                        continue

                # if the f_lower is equal to zero, it's probably close enough to be the WSE we are looking for, so we'll use it
                elif np.round(f_lower, 5) == 0:          
                    trial_slope_use = np.round(slope_lower, 3)
                    # Optionally, recompute d_maxflow_wse_final and d_q_sum with the new slope:
                    d_maxflow_wse_final_test, d_q_sum_test = find_wse(
                        2501, 
                        d_maxflow_wse_initial, 
                        d_depth_increment_small, 
                        d_q_maximum, 
                        da_xs_profile1, 
                        da_xs_profile2, 
                        xs1_n, 
                        xs2_n,
                        d_ordinate_dist, 
                        ia_xc_r1_index_main, 
                        ia_xc_c1_index_main, 
                        ia_xc_r2_index_main, 
                        ia_xc_c2_index_main, 
                        n_x_section_1, 
                        n_x_section_2, 
                        trial_slope_use
                    )
                    # Check if d_q_sum is within acceptable bounds
                    if abs(d_q_sum_test - d_q_maximum) < abs(d_q_sum-d_q_maximum):
                        # Optionally update d_slope_use to the accepted value:
                        d_slope_use = trial_slope_use
                        d_maxflow_wse_final = d_maxflow_wse_final_test
                        d_q_sum = d_q_sum_test
                    else:
                        pass

                # if the f_upper is equal to zero, it's probably close enough to be the WSE we are looking for, so we'll use it
                elif np.round(f_upper, 5) == 0:          
                    trial_slope_use = np.round(slope_upper, 3)
                    # Optionally, recompute d_maxflow_wse_final and d_q_sum with the new slope:
                    d_maxflow_wse_final_test, d_q_sum_test = find_wse(
                        2501, 
                        d_maxflow_wse_initial, 
                        d_depth_increment_small, 
                        d_q_maximum, 
                        da_xs_profile1, 
                        da_xs_profile2, 
                        xs1_n, 
                        xs2_n,
                        d_ordinate_dist, 
                        ia_xc_r1_index_main, 
                        ia_xc_c1_index_main, 
                        ia_xc_r2_index_main, 
                        ia_xc_c2_index_main, 
                        n_x_section_1, 
                        n_x_section_2, 
                        trial_slope_use
                    )
                    # Check if d_q_sum is within acceptable bounds
                    if abs(d_q_sum_test - d_q_maximum) < abs(d_q_sum-d_q_maximum):
                        # Optionally update d_slope_use to the accepted value:
                        d_slope_use = trial_slope_use
                        d_maxflow_wse_final = d_maxflow_wse_final_test
                        d_q_sum = d_q_sum_test
                    else:
                        pass
                
                else:
                    pass

            #This prevents the way-over simulated cells.  These are outliers.
            if d_q_baseflow>0.0 and da_total_q[i_start_elevation_index+1] >= 3.0 * d_q_baseflow:

                # something isn't good with our results
                acceptable = False

                # here we will see if we can get a better answer with a revised slope
                # from our Missouri study, relative DEM error was around 0.70, so dividing that by our d_distance_z[i_precompute_angle_closest] gives us a round about
                # idea of potential error in slope.  We'll use this to adjust the slope and see if we can get a fit.
                potential_slope_error = 0.6 / d_ordinate_dist

                # Set lower and upper bounds for the slope search.
                slope_lower = max(d_slope_use - potential_slope_error, 1e-8) # Avoids domain error, taking sqrt of negative number, in find wse
                slope_upper = d_slope_use + potential_slope_error

                # Check if the objective function changes sign between the bounds.
                f_lower = objective_with_slope(slope_lower)
                f_upper = objective_with_slope(slope_upper)
                
                
                if safe_signs_differ(f_lower, f_upper):
                    
                    # The signs differ, so we have a valid bracket.
                    trial_slope_use = brentq(objective_with_slope, slope_lower, slope_upper, xtol=0.0001)
                
                    # Optionally, recompute d_maxflow_wse_final and d_q_sum with the new slope:
                    d_maxflow_wse_final_test, d_q_sum_test = find_wse(
                        2501, 
                        d_maxflow_wse_initial, 
                        d_depth_increment_small, 
                        d_q_maximum, 
                        da_xs_profile1, 
                        da_xs_profile2, 
                        xs1_n, 
                        xs2_n,
                        d_ordinate_dist, 
                        ia_xc_r1_index_main, 
                        ia_xc_c1_index_main, 
                        ia_xc_r2_index_main, 
                        ia_xc_c2_index_main, 
                        n_x_section_1, 
                        n_x_section_2, 
                        trial_slope_use
                    )
                    # Check if d_q_sum is within acceptable bounds
                    if d_q_baseflow>0.0 and da_total_q[i_start_elevation_index+1] >= 3.0 * d_q_baseflow and d_maxflow_wse_final_test > 0.0:
                        acceptable = True
                        d_slope_use = trial_slope_use
                        d_maxflow_wse_final = d_maxflow_wse_final_test
                        d_q_sum = d_q_sum_test
                        
                # if the f_lower is equal to zero, it's probably close enough to be the WSE we are looking for, so we'll use it
                elif np.round(f_lower, 5) == 0:          
                    trial_slope_use = np.round(slope_lower, 3)
                    # Optionally, recompute d_maxflow_wse_final and d_q_sum with the new slope:
                    d_maxflow_wse_final_test, d_q_sum_test = find_wse(
                        2501, 
                        d_maxflow_wse_initial, 
                        d_depth_increment_small, 
                        d_q_maximum, 
                        da_xs_profile1, 
                        da_xs_profile2, 
                        xs1_n, 
                        xs2_n,
                        d_ordinate_dist, 
                        ia_xc_r1_index_main, 
                        ia_xc_c1_index_main, 
                        ia_xc_r2_index_main, 
                        ia_xc_c2_index_main, 
                        n_x_section_1, 
                        n_x_section_2, 
                        trial_slope_use
                    )
                    # Check if d_q_sum is within acceptable bounds
                    if abs(d_q_sum_test - d_q_maximum) < abs(d_q_sum-d_q_maximum):
                        # Optionally update d_slope_use to the accepted value:
                        d_slope_use = trial_slope_use
                        d_maxflow_wse_final = d_maxflow_wse_final_test
                        d_q_sum = d_q_sum_test
                    else:
                        pass

                # if the f_upper is equal to zero, it's probably close enough to be the WSE we are looking for, so we'll use it
                elif np.round(f_upper, 5) == 0:          
                    trial_slope_use = np.round(slope_upper, 3)
                    # Optionally, recompute d_maxflow_wse_final and d_q_sum with the new slope:
                    d_maxflow_wse_final_test, d_q_sum_test = find_wse(
                        2501, 
                        d_maxflow_wse_initial, 
                        d_depth_increment_small, 
                        d_q_maximum, 
                        da_xs_profile1, 
                        da_xs_profile2, 
                        xs1_n, 
                        xs2_n,
                        d_ordinate_dist, 
                        ia_xc_r1_index_main, 
                        ia_xc_c1_index_main, 
                        ia_xc_r2_index_main, 
                        ia_xc_c2_index_main, 
                        n_x_section_1, 
                        n_x_section_2, 
                        trial_slope_use
                    )
                    # Check if d_q_sum is within acceptable bounds
                    if abs(d_q_sum_test - d_q_maximum) < abs(d_q_sum-d_q_maximum):
                        # Optionally update d_slope_use to the accepted value:
                        d_slope_use = trial_slope_use
                        d_maxflow_wse_final = d_maxflow_wse_final_test
                        d_q_sum = d_q_sum_test
                    else:
                        pass
                
                else:
                    pass

            if not acceptable:
                # print("I'm skipping because d_q_sum is outside acceptable range after varying d_slope_use")
                if b_reach_average_curve_file:
                    All_COMID_curve_list.append(i_cell_comid)
                    All_Row_curve_list.append(i_row_cell)
                    All_Col_curve_list.append(i_column_cell)
                    All_BaseElev_curve_list.append(dm_elevation[i_row_cell, i_column_cell])
                    All_DEM_Elev_curve_list.append(dm_elevation[i_row_cell, i_column_cell])
                    All_QMax_curve_list.append(d_q_maximum)
                    continue
                else:
                    continue
            
            # if we have a usable value for d_maxflow_wse_final, lets get rest of the VDT data
            if acceptable and d_maxflow_wse_final > 0.0:
            
                # Now lets get a set number of increments between the low elevation and the elevation where Qmax hits
                d_inc_y = (d_maxflow_wse_final - da_xs_profile1[0]) / i_number_of_increments
                i_number_of_elevations = i_number_of_increments + 1
                i_start_elevation_index, i_last_elevation_index = flood_increments(i_number_of_increments + 1, 
                                                                                d_inc_y, 
                                                                                da_xs_profile1, da_xs_profile2, xs1_n, xs2_n,
                                                                                d_ordinate_dist, 
                                                                                n_x_section_1, n_x_section_2, d_slope_use, da_total_t, 
                                                                                da_total_a, da_total_p, da_total_v, da_total_q, 
                                                                                da_total_wse)

                if d_q_baseflow>0.001 and da_total_q[i_start_elevation_index+1] >= d_q_baseflow:
                    da_total_q[i_start_elevation_index+1] = d_q_baseflow-0.001
                    

                # Process each of the elevations to the output file if feasbile values were produced
                da_total_q_half_sum = sum(da_total_q[0 : int(i_number_of_elevations / 2.0)])
                if da_total_q_half_sum > 1e-16 and i_row_cell >= 0 and i_column_cell >= 0 and dm_elevation[i_row_cell, i_column_cell] > 1e-16:
                    if s_output_ap_database:
                        comid_ap_dict_list.append(i_cell_comid)
                        row_ap_dict_list.append(i_row_cell - i_boundary_number)
                        col_ap_dict_list.append(i_column_cell - i_boundary_number)
                    
                    vdt_row = [i_cell_comid, i_row_cell - i_boundary_number, i_column_cell - i_boundary_number]
                    vdt_row.append(dm_elevation[i_row_cell, i_column_cell] - 100 if b_modified_dem else dm_elevation[i_row_cell, i_column_cell])
                    vdt_row.append(d_q_baseflow)
                    vdt_list.append(vdt_row)

                    # Loop backward through the elevations
                    if s_output_vdt_database:
                        if b_modified_dem:
                            da_total_wse -= 100

                        q_list.append(da_total_q[1:])
                        v_list.append(da_total_v[1:])
                        t_list.append(da_total_t[1:])
                        wse_list.append(da_total_wse[1:])

                        if b_modified_dem:
                            da_total_wse += 100
                    
                    if s_output_ap_database:
                        for i, (q_name, a_name, p_name) in enumerate(ap_column_names[:i_number_of_elevations - 1], start=1):
                            o_ap_file_dict[q_name].append(da_total_q[i])
                            o_ap_file_dict[a_name].append(da_total_a[i])
                            o_ap_file_dict[p_name].append(da_total_p[i])


                if i_number_of_elevations > 0:
                    b_outprint_yes = True
        
        elif i_volume_fill_approach == 2:
            # This was trying to set a max elevation difference between the ordinates
            l_add_list = []
            i_add_level = 250

            # Check that the difference between elevation increments exceeds the target to process to set the contours
            for i_entry_elevation in range(1, i_number_of_elevations):
                if da_elevation_list_mm[i_entry_elevation] - da_elevation_list_mm[i_entry_elevation - 1] > i_add_level:
                    l_add_list.append(da_elevation_list_mm[i_entry_elevation] + i_add_level)

            # Set one above the current value to ensure all values get processed
            if len(l_add_list) > 0:
                da_elevation_list_mm = np.append(da_elevation_list_mm, l_add_list)
                da_elevation_list_mm = np.sort(da_elevation_list_mm)
                i_number_of_elevations = len(da_elevation_list_mm)
            
            
            # Check that the number of elevations is reasonable
            if i_number_of_elevations >= ep:
                LOG.error('ERROR, HAVE TOO MANY ELEVATIONS TO EVALUATE')

            for i_entry_elevation in range(1, i_number_of_elevations):
                # Calculate the geometry
                d_wse = da_elevation_list_mm[i_entry_elevation] / 1000.0
                A1, P1, R1, np1, T1 = calculate_stream_geometry(da_xs_profile1[:xs1_n], d_wse, d_ordinate_dist, dm_manning_n_raster[ia_xc_r1_index_main[0:xs1_n], ia_xc_c1_index_main[0:xs1_n]])
                A2, P2, R2, np2, T2 = calculate_stream_geometry(da_xs_profile2[:xs2_n], d_wse, d_ordinate_dist, dm_manning_n_raster[ia_xc_r2_index_main[0:xs2_n], ia_xc_c2_index_main[0:xs2_n]])

                # Aggregate the geometric properties
                da_total_t[i_entry_elevation] = T1 + T2
                da_total_a[i_entry_elevation] = A1 + A2
                da_total_p[i_entry_elevation] = P1 + P2

                # Check the properties are physically realistic. If so, estimate the flow with them.
                if da_total_t[i_entry_elevation] <= 0.0 or da_total_a[i_entry_elevation] <= 0.0 or da_total_p[i_entry_elevation] <= 0.0:
                    da_total_t[i_entry_elevation] = 0.0
                    da_total_a[i_entry_elevation] = 0.0
                    da_total_p[i_entry_elevation] = 0.0
                    i_start_elevation_index = i_entry_elevation

                else:
                    # Estimate mannings n
                    d_composite_n = math.pow(((np1 + np2) / da_total_p[i_entry_elevation]), (2 / 3))

                    # Check that the mannings n is physically realistic
                    if d_composite_n < 0.0001:
                        d_composite_n = 0.035

                    # Estimate total flows
                    da_total_q[i_entry_elevation] = ((1 / d_composite_n) * da_total_a[i_entry_elevation] * (da_total_a[i_entry_elevation] / da_total_p[i_entry_elevation])**(2 / 3) * d_slope_use**0.5)
                    da_total_v[i_entry_elevation] = da_total_q[i_entry_elevation] / da_total_a[i_entry_elevation]
                    da_total_wse[i_entry_elevation] = d_wse
                    i_last_elevation_index = i_entry_elevation

                # Perform check on the maximum flow
                if da_total_q[i_entry_elevation] > d_q_maximum:
                    i_number_of_elevations = i_entry_elevation + 1   # Do the +1 because of the printing below
                    break
                '''
                #Just for checking, add the boundaries to the output flood raster for ARC.
                if i_entry_elevation==i_number_of_elevations-1:
                    #This is the Edge of the Flood.  Value is 2 because that is the flood method used.
                    dm_out_flood[ia_xc_r1_index_main[np1], ia_xc_c1_index_main[np1]] = 2
                    dm_out_flood[ia_xc_r2_index_main[np2], ia_xc_c2_index_main[np2]] = 2
                '''

            # Process each of the elevations to the output file if feasbile values were produced
            da_total_q_half_sum = sum(da_total_q[0 : int(i_number_of_elevations / 2.0)])
            if da_total_q_half_sum > 1e-16 and i_row_cell >= 0 and i_column_cell >= 0 and dm_elevation[i_row_cell, i_column_cell] > 1e-16:
                comid_dict_list.append(i_cell_comid)
                row_dict_list.append(i_row_cell - i_boundary_number)
                col_dict_list.append(i_column_cell - i_boundary_number)
                if len(s_output_ap_database) > 0:
                    comid_ap_dict_list.append(i_cell_comid)
                    row_ap_dict_list.append(i_row_cell - i_boundary_number)
                    col_ap_dict_list.append(i_column_cell - i_boundary_number)
                if b_modified_dem:
                    elev_dict_list.append(dm_elevation[i_row_cell, i_column_cell]-100)
                else:
                    elev_dict_list.append(dm_elevation[i_row_cell, i_column_cell])
                qbaseflow_dict_list.append(d_q_baseflow)

                # Loop backward through the elevations
                if s_output_vdt_database:
                    if b_modified_dem:
                        da_total_wse -= 100
                    for i, (q_name, v_name, t_name, wse_name) in enumerate(vdt_column_names[:i_number_of_elevations - 1], start=1):
                        o_out_file_dict[q_name].append(da_total_q[i])
                        o_out_file_dict[v_name].append(da_total_v[i])
                        o_out_file_dict[t_name].append(da_total_t[i])
                        o_out_file_dict[wse_name].append(da_total_wse[i])
                    if b_modified_dem:
                        da_total_wse += 100
        
            if i_number_of_elevations > 0:
                i_outprint_yes = 1

        # Gather up all the values for the stream cell if we are going to build a reach average curve file
        if b_reach_average_curve_file:
            All_COMID_curve_list.append(int(i_cell_comid))
            All_Row_curve_list.append(int(i_row_cell - i_boundary_number))
            All_Col_curve_list.append(int(i_column_cell - i_boundary_number))
            if b_modified_dem:
                All_BaseElev_curve_list.append(np.round(da_xs_profile1[0], 3)-100)
                All_DEM_Elev_curve_list.append(np.round(d_dem_low_point_elev, 3)-100)
            else:
                All_BaseElev_curve_list.append(np.round(da_xs_profile1[0], 3))
                All_DEM_Elev_curve_list.append(np.round(d_dem_low_point_elev, 3))
            All_QMax_curve_list.append(np.round(d_q_maximum, 3))

        # Work on the Regression Equations File
        if b_outprint_yes and s_output_curve_file and i_start_elevation_index>=0 and i_last_elevation_index>(i_start_elevation_index+1):
            # Not needed here, but [::-1] basically reverses the order of the array
            (d_t_a, d_t_b, d_t_R2) = linear_regression_power_function(da_total_q[i_start_elevation_index:i_last_elevation_index + 1][1:], da_total_t[i_start_elevation_index:i_last_elevation_index + 1][1:], [12, 0.3])
            (d_v_a, d_v_b, d_v_R2) = linear_regression_power_function(da_total_q[i_start_elevation_index:i_last_elevation_index + 1][1:], da_total_v[i_start_elevation_index:i_last_elevation_index + 1][1:], [1, 0.3])
            da_total_depth = da_total_wse - da_xs_profile1[0]
            (d_d_a, d_d_b, d_d_R2) = linear_regression_power_function(da_total_q[i_start_elevation_index:i_last_elevation_index + 1][1:], da_total_depth[i_start_elevation_index:i_last_elevation_index + 1][1:], [0.2, 0.5])
            COMID_curve_list.append(int(i_cell_comid))
            Row_curve_list.append(int(i_row_cell - i_boundary_number))
            Col_curve_list.append(int(i_column_cell - i_boundary_number))
            if b_modified_dem:
                BaseElev_curve_list.append(np.round(da_xs_profile1[0], 3)-100)
                DEM_Elev_curve_list.append(np.round(d_dem_low_point_elev, 3)-100)
            else:
                BaseElev_curve_list.append(np.round(da_xs_profile1[0], 3))
                DEM_Elev_curve_list.append(np.round(d_dem_low_point_elev, 3))
            QMax_curve_list.append(np.round(da_total_q[i_last_elevation_index], 3))
            depth_a_curve_list.append(np.round(d_d_a, 3))
            depth_b_curve_list.append(np.round(d_d_b, 3))
            tw_a_curve_list.append(np.round(d_t_a, 3))
            tw_b_curve_list.append(np.round(d_t_b, 3))
            vel_a_curve_list.append(np.round(d_v_a, 3))
            vel_b_curve_list.append(np.round(d_v_b, 3))

        # Output the XS information, if you've chosen to do so
        if s_xs_output_file:
            if b_modified_dem:
                da_xs_profile1_str = array_to_string(da_xs_profile1[0:xs1_n]-100)
                da_xs_profile2_str = array_to_string(da_xs_profile2[0:xs2_n]-100) 
            else:
                da_xs_profile1_str = array_to_string(da_xs_profile1[0:xs1_n])
                da_xs_profile2_str = array_to_string(da_xs_profile2[0:xs2_n]) 
            dm_manning_n_raster1_str = array_to_string(dm_manning_n_raster[ia_xc_r1_index_main[0:xs1_n], ia_xc_c1_index_main[0:xs1_n]]) 
            dm_manning_n_raster2_str = array_to_string(dm_manning_n_raster[ia_xc_r2_index_main[0:xs2_n], ia_xc_c2_index_main[0:xs2_n]])
            o_xs_file.write(f"{i_cell_comid}\t{i_row_cell - i_boundary_number}\t{i_column_cell - i_boundary_number}\t{da_xs_profile1_str}\t{d_wse}\t{d_ordinate_dist}\t{dm_manning_n_raster1_str}\t{da_xs_profile2_str}\t{d_wse}\t{d_ordinate_dist}\t{dm_manning_n_raster2_str}\n")

   
    # Create the output VDT Database file - datatypes are figured out automatically
    colorder = ['COMID', 'Row', 'Col', 'Elev', 'QBaseflow'] + [f"{prefix}_{i}" for i in range(1, i_number_of_increments + 1) for prefix in ['q', 'v', 't', 'wse']]
    vdt_df = (pd.concat([pd.DataFrame(vdt_list, columns=['COMID', 'Row', 'Col', 'Elev', 'QBaseflow']), 
                        pd.DataFrame(q_list, columns=[f'q_{i}' for i in range(1, len(q_list[0]) + 1)]),
                        pd.DataFrame(v_list, columns=[f'v_{i}' for i in range(1, len(v_list[0]) + 1)]),
                        pd.DataFrame(t_list, columns=[f't_{i}' for i in range(1, len(t_list[0]) + 1)]),
                        pd.DataFrame(wse_list, columns=[f'wse_{i}' for i in range(1, len(wse_list[0]) + 1)])], axis=1)
                        .round(3)
                        [colorder] # Reorder columns to match the way Mike had it
                        )
    
    # Remove rows with NaN values
    vdt_df = vdt_df.dropna()
    # # Remove rows where any column has a negative value except wse or elevation
    # Select columns NOT starting with 'wse' or 'Elev'
    cols_to_check = [col for col in vdt_df.columns if (col.startswith('q') or col.startswith('t') or col.startswith('v'))]
    # Remove rows where any of the selected columns have a negative value
    vdt_df = vdt_df.loc[~(vdt_df[cols_to_check] < 0).any(axis=1)]
    if s_output_vdt_database.endswith('.parquet'):
        vdt_df.to_parquet(s_output_vdt_database, compression='brotli', index=False) # Brotli does very well with VDT data
    else:
        vdt_df.to_csv(s_output_vdt_database, index=False)    
    LOG.info('Finished writing ' + str(s_output_vdt_database))
    
    # Output the area and wetted perimeter database file if requested
    if len(s_output_ap_database) > 0:
        # Write the output VDT Database file
        dtypes = {
                    "COMID": 'int64',
                    "Row": 'int64',
                    "Col": 'int64',
        }
        for i in range(1, i_number_of_increments + 1):
            o_ap_file_dict[f'q_{i}'] = np.round(o_ap_file_dict[f'q_{i}'], 3)
            o_ap_file_dict[f'a_{i}'] = np.round(o_ap_file_dict[f'a_{i}'], 3)
            o_ap_file_dict[f'p_{i}'] = np.round(o_ap_file_dict[f'p_{i}'], 3)

        o_ap_file_df = pd.DataFrame(o_ap_file_dict).astype(dtypes)
        # Remove rows with NaN values
        o_ap_file_df = o_ap_file_df.dropna()
        # # Remove rows where any column has a negative value except wse or elevation
        # Select columns NOT starting with 'wse' or 'Elev'
        cols_to_check = [col for col in o_ap_file_df.columns if (col.startswith('q') or col.startswith('a') or col.startswith('p'))]
        # Remove rows where any of the selected columns have a negative value
        o_ap_file_df = o_ap_file_df.loc[~(o_ap_file_df[cols_to_check] < 0).any(axis=1)]
        if s_output_ap_database.endswith('.parquet'):
            o_ap_file_df.to_parquet(s_output_ap_database, compression='brotli', index=False)
        else:
            o_ap_file_df.to_csv(s_output_ap_database, index=False)
        LOG.info('Finished writing ' + str(s_output_ap_database))

    # Here we'll generate reach-based coefficients for all stream cells, if the flag is triggered
    if b_reach_average_curve_file:
        # Creating a dictionary to map column names to the lists
        data = {
            "COMID": All_COMID_curve_list,
            "Row": All_Row_curve_list,
            "Col": All_Col_curve_list,
            "BaseElev":  [np.round(num, 3) for num in All_BaseElev_curve_list],
            "DEM_Elev": [np.round(num, 3) for num in All_DEM_Elev_curve_list],
            "QMax": All_QMax_curve_list,
        }

        # Creating the DataFrame
        reach_average_curvefile_df = pd.DataFrame(data)

        # Dynamically select columns, starting with prefixes
        q_prefixes = [col for col in vdt_df.columns if col.startswith("q_")]
        t_prefixes = [col for col in vdt_df.columns if col.startswith("t_")]
        v_prefixes = [col for col in vdt_df.columns if col.startswith("v_")]
        wse_prefixes = [col for col in vdt_df.columns if col.startswith("wse_")]

        # Initialize lists to store regression coefficients
        comid_list = []
        d_t_a_list, d_t_b_list = [], []
        d_v_a_list, d_v_b_list = [], []
        d_d_a_list, d_d_b_list = [], []

        # Extract all unique COMID values
        unique_comids = vdt_df["COMID"].unique()

        # Process each unique COMID
        for comid in unique_comids:
            group = vdt_df[vdt_df["COMID"] == comid]
            
            # Create a MultiIndex from the current group's Row and Col for precise matching
            group_index = pd.MultiIndex.from_arrays([group["Row"].values, group["Col"].values], names=["Row", "Col"])

            # Filter reach_average_curvefile_df using COMID and matching Row-Col pairs
            matching_reach = reach_average_curvefile_df[
                (reach_average_curvefile_df["COMID"] == comid) &
                (pd.MultiIndex.from_frame(reach_average_curvefile_df[["Row", "Col"]]).isin(group_index))
            ]

            matching_reach = matching_reach.drop_duplicates(subset=["Row", "Col", "COMID"])

            if matching_reach.empty:
                LOG.warning(f"No matching BaseElev values found for COMID {comid}. Skipping...")
                continue

            # Get the BaseElev values for subtraction
            base_elev_values = matching_reach.set_index(["Row", "Col"])["BaseElev"]

            # Combine WSE_ values and subtract BaseElev
            depth_combined_values_list = []
            for prefix in wse_prefixes:
                # Match rows using Row and Col from the group
                wse_values = group.set_index(["Row", "Col"])[prefix]
                depth_values = wse_values - base_elev_values
                depth_combined_values_list.extend(depth_values.values)
            d_combined_values = np.array(depth_combined_values_list)

            # Combine Q_ values
            q_combined_values_list = []
            for prefix in q_prefixes:
                q_combined_values_list.extend(group[prefix].values)
            q_combined_values = np.array(q_combined_values_list)

            # Combine T_ values
            t_combined_values_list = []
            for prefix in t_prefixes:
                t_combined_values_list.extend(group[prefix].values)
            t_combined_values = np.array(t_combined_values_list)

            # Combine V_ values
            v_combined_values_list = []
            for prefix in v_prefixes:
                v_combined_values_list.extend(group[prefix].values)
            v_combined_values = np.array(v_combined_values_list)

            # Calculate regression coefficients
            try:
                (d_t_a, d_t_b, d_t_R2) = linear_regression_power_function(q_combined_values, t_combined_values, [12, 0.3])
                (d_v_a, d_v_b, d_v_R2) = linear_regression_power_function(q_combined_values, v_combined_values, [1, 0.3])
                (d_d_a, d_d_b, d_d_R2) = linear_regression_power_function(q_combined_values, d_combined_values, [0.2, 0.5])
            except Exception as e:
                # Handle cases where regression fails (e.g., insufficient data)
                LOG.warning(f"Regression failed for COMID {comid}: {e}")
                d_t_a, d_t_b, d_v_a, d_v_b, d_d_a, d_d_b = np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

            # Append results to lists
            comid_list.append(comid)
            d_t_a_list.append(np.round(d_t_a, 3) if not np.isnan(d_t_a) else np.nan)
            d_t_b_list.append(np.round(d_t_b, 3) if not np.isnan(d_t_b) else np.nan)
            d_v_a_list.append(np.round(d_v_a, 3) if not np.isnan(d_v_a) else np.nan)
            d_v_b_list.append(np.round(d_v_b, 3) if not np.isnan(d_v_b) else np.nan)
            d_d_a_list.append(np.round(d_d_a, 3) if not np.isnan(d_d_a) else np.nan)
            d_d_b_list.append(np.round(d_d_b, 3) if not np.isnan(d_d_b) else np.nan)

        # Create a DataFrame with regression coefficients
        regression_df = pd.DataFrame({
            "COMID": comid_list,
            "depth_a": d_d_a_list,
            "depth_b": d_d_b_list,
            "tw_a": d_t_a_list,
            "tw_b": d_t_b_list,
            "vel_a": d_v_a_list,
            "vel_b": d_v_b_list,
        })

        # Merge the regression_df into reach_average_curvefile_df based on COMID
        reach_average_curvefile_df = reach_average_curvefile_df.merge(regression_df, on="COMID", how="left")

        # Drop all rows with any NaN values
        reach_average_curvefile_df = reach_average_curvefile_df.dropna()

        # Write the output file
        if s_output_curve_file.endswith('.parquet'):
            reach_average_curvefile_df.to_parquet(s_output_curve_file, compression='brotli', index=False)
        else:
            reach_average_curvefile_df.to_csv(s_output_curve_file, index=False)        
        LOG.info('Finished writing ' + str(s_output_curve_file))

    else:
    
        # Write the output Curve file
        if len(s_output_curve_file)>0:
            o_curve_file_dict = {'COMID': COMID_curve_list,
                                'Row': Row_curve_list,
                                'Col': Col_curve_list,
                                'BaseElev': BaseElev_curve_list,
                                'DEM_Elev': DEM_Elev_curve_list,
                                'QMax': QMax_curve_list,
                                'depth_a': depth_a_curve_list,
                                'depth_b': depth_b_curve_list,
                                'tw_a': tw_a_curve_list,
                                'tw_b': tw_b_curve_list,
                                'vel_a': vel_a_curve_list,
                                'vel_b': vel_b_curve_list,}
            o_curve_file_df = pd.DataFrame(o_curve_file_dict)
            # Remove rows with NaN values
            o_curve_file_df = o_curve_file_df.dropna()
            # # Remove rows where any column has negative a coefficient value
            o_curve_file_df = o_curve_file_df.loc[(o_curve_file_df['depth_a'] > 0) & (o_curve_file_df['tw_a'] > 0) & (o_curve_file_df['vel_a'] > 0)]
            if s_output_curve_file.endswith('.parquet'):
                o_curve_file_df.to_parquet(s_output_curve_file, compression='brotli', index=False)
            else:
                o_curve_file_df.to_csv(s_output_curve_file, index=False)            
            LOG.info('Finished writing ' + str(s_output_curve_file))
    
    

    # Write the output rasters
    if len(s_output_bathymetry_path) > 1:
        #Make sure all the bathymetry points are above the DEM elevation
        if not b_bathy_use_banks:
            dm_output_bathymetry = np.where(dm_output_bathymetry>dm_elevation, np.nan, dm_output_bathymetry)
        # remove the increase in elevation, if negative elevations were present
        if b_modified_dem:
            # Subtract 100 only for cells that are not NaN
            dm_output_bathymetry[~np.isnan(dm_output_bathymetry)] -= 100
        # # Joseph was testing a simple smoothing algorithm here to attempt to reduce variation in the bank based bathmetry (functions but doesn't provide better results)
        # if b_bathy_use_banks:
        #     dm_output_bathymetry = smooth_bathymetry_gaussian_numba(dm_output_bathymetry)
        write_output_raster(s_output_bathymetry_path, dm_output_bathymetry[i_boundary_number:nrows + i_boundary_number, i_boundary_number:ncols + i_boundary_number], ncols, nrows, dem_geotransform, dem_projection, "GTiff", gdal.GDT_Float32)

    if len(s_output_flood) > 1:
        write_output_raster(s_output_flood, dm_out_flood[i_boundary_number:nrows + i_boundary_number, i_boundary_number:ncols + i_boundary_number], ncols, nrows, dem_geotransform, dem_projection, "GTiff", gdal.GDT_Int32)

    # Log the compute time
    d_sim_time = datetime.now() - starttime
    i_sim_time_s = int(d_sim_time.seconds)

    if i_sim_time_s < 60:
        LOG.info('Simulation Took ' + str(i_sim_time_s) + ' seconds')
    else:
        LOG.info('Simulation Took ' + str(int(i_sim_time_s / 60)) + ' minutes and ' + str(i_sim_time_s - (int(i_sim_time_s / 60) * 60)) + ' seconds')
        
if __name__ == "__main__":
    LOG.info('Inputs to the Program is a Main Input File')
    LOG.info('\nFor Example:')
    LOG.info('  python Automated_Rating_Curve_Generator.py ARC_InputFiles/ARC_Input_File.txt')
    
    ### User-Defined Main Input File ###
    if len(sys.argv) > 1:
        MIF_Name = sys.argv[1]
        LOG.info('Main Input File Given: ' + MIF_Name)
    else:
        #Read Main Input File
        MIF_Name = '/Users/ricky/Documents/data_dir/mifns/USGS_1_n40w111_20240130_buff__mifn.txt'
        LOG.warning('Moving forward with Default MIF Name: ' + MIF_Name)
        
    main(MIF_Name, False)
