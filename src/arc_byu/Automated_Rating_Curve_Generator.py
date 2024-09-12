
"""
Written initially by Mike Follum with Follum Hydrologic Solutions, LLC.
Program simply creates depth, velocity, and top-width information for each stream cell in a domain.

"""

import sys
import os
import math, time
import logging

import tqdm
import numpy as np
import pandas as pd
from datetime import datetime
from scipy.optimize import curve_fit
try:
    import gdal 

except: 
    from osgeo import gdal


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
        s_output = None

    # Return to the calling function
    return s_output


# Power function equation
def power_func(d_value: float, d_coefficient: float, d_power: float):
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
    d_power_value = d_coefficient * np.power(d_value, d_power)

    # Return to the calling function
    return d_power_value


def linear_regression_power_function(da_x_input: np.ndarray, da_y_input: np.ndarray):
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

    # Attempt to calculate the fit
    try:
        (d_coefficient, d_power), dm_pcov = curve_fit(power_func, da_x_input, da_y_input)
        dm_corr_matrix = np.corrcoef(da_y_input, power_func(da_x_input, d_coefficient, d_power))
        d_corr = dm_corr_matrix[0, 1]
        d_R2 = d_corr ** 2

    except:
        # Fit was not successful. Put in flag values to indicate failure
        d_coefficient=-9999.9
        d_power=-9999.9
        d_R2 = -9999.9

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
    o_output_file = o_driver.Create(s_output_filename, xsize=i_number_of_columns, ysize=i_number_of_rows, bands=1, eType=s_output_type)
    
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
        logging.info('Cannot Find Raster ' + s_input_filename)

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
    logging.info('Spatial Data for Raster File:')
    logging.info('   ncols = ' + str(i_number_of_columns))
    logging.info('   nrows = ' + str(i_number_of_rows))
    logging.info('   cellsize = ' + str(d_cell_size))
    logging.info('   yll = ' + str(d_y_lower_left))
    logging.info('   yur = ' + str(d_y_upper_right))
    logging.info('   xll = ' + str(d_x_lower_left))
    logging.info('   xur = ' + str(d_x_upper_right))

    # Return dataset information to the calling function
    return dm_raster_array, i_number_of_columns, i_number_of_rows, d_cell_size, d_y_lower_left, d_y_upper_right, d_x_lower_left, d_x_upper_right, d_latitude, l_geotransform, s_raster_projection


def get_parameter_name(sl_lines, i_number_of_lines, s_target):
    """
    Gets parameter values from a list of strings, assuming that the file is tab delimited and the first characters are the target string.
    The second column is returned as the target value.

    Parameters
    ----------
    sl_lines: list
        Lines to test for target string
    i_number_of_lines: int
        Number of lines to test from the list in order
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
    for entry in range(i_number_of_lines):
        # Split the line and strip special characters
        ls = sl_lines[entry].strip().split('\t')

        # Check if the first entry is the target string
        if ls[0] == s_target:
            # Override the initial the default value
            d_return_value = 1

            # String is found. Process the rest of the line
            if len(ls[1]) > 0:
                # More information is available to parse
                d_return_value = ls[1]

    # Log the value to the console
    if d_return_value != '':
        logging.info('  ' + s_target + ' is set to ' + d_return_value)

    else:
       logging.warning('  Could not find ' + s_target)

    # Return value to the calling function
    return d_return_value


def read_main_input_file(s_mif_name: str):
    """

    Parameters
    ----------
    s_mif_name: str
        Path to the input file

    Returns
    -------
    None. Global values are used to set values outside of the function for the full program

    """

    ### Open and read the input file ###
    # Open the file
    o_input_file = open(s_mif_name, 'r')
    sl_lines = o_input_file.readlines()
    o_input_file.close()

    # Count the number of lines in the file
    i_number_of_lines = len(sl_lines)

    ### Process the parameters ###
    # Find the path to the DEM file
    global s_input_dem_path
    s_input_dem_path = get_parameter_name(sl_lines, i_number_of_lines, 'DEM_File')

    # Find the path to the stream file
    global s_input_stream_path
    s_input_stream_path = get_parameter_name(sl_lines, i_number_of_lines, 'Stream_File')

    # Find the path to the land use raster file
    global s_input_land_use_path
    s_input_land_use_path = get_parameter_name(sl_lines, i_number_of_lines, 'LU_Raster_SameRes')

    # Find the path to the mannings n file
    global s_input_mannings_path
    s_input_mannings_path = get_parameter_name(sl_lines, i_number_of_lines, 'LU_Manning_n')

    # Find the path to the input flow file
    global s_input_flow_file_path
    s_input_flow_file_path = get_parameter_name(sl_lines, i_number_of_lines, 'Flow_File')

    # Find the method used to number the streams
    global s_flow_file_id
    s_flow_file_id = get_parameter_name(sl_lines, i_number_of_lines, 'Flow_File_ID')

    # Find the flow method
    global s_flow_file_baseflow
    s_flow_file_baseflow = get_parameter_name(sl_lines, i_number_of_lines, 'Flow_File_BF')

    # Find the field that determines the maximum flow
    global s_flow_file_qmax
    s_flow_file_qmax = get_parameter_name(sl_lines, i_number_of_lines, 'Flow_File_QMax')

    # Find the spatial units
    global s_spatial_units
    s_spatial_units = get_parameter_name(sl_lines, i_number_of_lines, 'Spatial_Units')

    if s_spatial_units == '':
        # Assume degree if not specified in the input efile
        s_spatial_units = 'deg'

    # Find the x section distance
    global d_x_section_distance
    d_x_section_distance = get_parameter_name(sl_lines, i_number_of_lines, 'X_Section_Dist')

    if d_x_section_distance == '':
        # Value does not occur in the input file. Assume a reasonable value
        d_x_section_distance = 5000.0

    d_x_section_distance = float(d_x_section_distance)

    # Find the path to the output velocity, depth, and top width file
    global s_output_vdt_database
    s_output_vdt_database = get_parameter_name(sl_lines, i_number_of_lines, 'Print_VDT_Database')
    
    global s_output_curve_file
    s_output_curve_file = get_parameter_name(sl_lines, i_number_of_lines, 'Print_Curve_File')
    
    # Find the path to the output metdata file
    global s_output_meta_file
    s_output_meta_file = get_parameter_name(sl_lines, i_number_of_lines, 'Meta_File')

    # Find degree manipulation attribute
    global d_degree_manipulation
    d_degree_manipulation = get_parameter_name(sl_lines, i_number_of_lines, 'Degree_Manip')

    if d_degree_manipulation == '':
        # Value does not occur in the input file. Assume a reasonable value
        d_degree_manipulation = 1.1

    d_degree_manipulation = float(d_degree_manipulation)

    # Find the degree interval attribute
    global d_degree_interval
    d_degree_interval = get_parameter_name(sl_lines, i_number_of_lines, 'Degree_Interval')

    if d_degree_interval == '':
        # Value does not occur in the input file. Assume a reasonable value
        d_degree_interval = 1.0

    d_degree_interval = float(d_degree_interval)

    # Find the low spot range attribute
    global i_low_spot_range
    i_low_spot_range = get_parameter_name(sl_lines, i_number_of_lines, 'Low_Spot_Range')

    if i_low_spot_range == '':
        # Value does not occur in the input file. Assume a reasonable value
        i_low_spot_range = 0

    i_low_spot_range = int(i_low_spot_range)

    # Find the general direction distance attribute
    global i_general_direction_distance
    i_general_direction_distance= get_parameter_name(sl_lines, i_number_of_lines, 'Gen_Dir_Dist')

    if i_general_direction_distance == '':
        # Value does not occur in the input file. Assume a reasonable value
        i_general_direction_distance = 10

    i_general_direction_distance = int(i_general_direction_distance)

    # Find the general slope distance attribute
    global i_general_slope_distance
    i_general_slope_distance = get_parameter_name(sl_lines, i_number_of_lines, 'Gen_Slope_Dist')

    if i_general_slope_distance == '':
        # Value does not occur in the input file. Assume a reasonable value
        i_general_slope_distance = 0

    i_general_slope_distance = int(i_general_slope_distance)

    # Find the bathymetry trapezoid height attribute
    global d_bathymetry_trapzoid_height
    d_bathymetry_trapzoid_height = get_parameter_name(sl_lines, i_number_of_lines, 'Bathy_Trap_H')

    if d_bathymetry_trapzoid_height == '':
        # Value does not occur in the input file. Assume a reasonable value
        d_bathymetry_trapzoid_height = 0.2

    d_bathymetry_trapzoid_height = float(d_bathymetry_trapzoid_height)

    # Find the path to the output bathymetry file
    global s_output_bathymetry_path
    s_output_bathymetry_path = get_parameter_name(sl_lines, i_number_of_lines, 'AROutBATHY')

    if s_output_bathymetry_path == '':
        # Value does not occur in the input file. Assume a reasonable value
        s_output_bathymetry_path = get_parameter_name(sl_lines, i_number_of_lines, 'BATHY_Out_File')

    # Find the path to the output depth file
    global s_output_depth
    s_output_depth = get_parameter_name(sl_lines, i_number_of_lines, 'AROutDEPTH')

    # Find the path to the output flood file
    global s_output_flood
    s_output_flood = get_parameter_name(sl_lines, i_number_of_lines, 'AROutFLOOD')

    # Find the path to the output cross-section file (JLG added this to recalculate top-width and velocity)
    global s_xs_output_file
    s_xs_output_file = get_parameter_name(sl_lines, i_number_of_lines, 'XS_Out_File')


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


def read_flow_file(s_flow_file_name: str, s_flow_id: str, s_flow_baseflow: str, s_flow_qmax):
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
    # Check if file exists
    if not os.path.exists(s_flow_file_name):
        logging.error('Cannot Find Flow File ' + s_flow_file_name)
        exit()

    df = pd.read_csv(s_flow_file_name)

    # Extract the required columns into numpy arrays
    da_comid = df[s_flow_id].to_numpy(dtype=int)
    da_base_flow = df[s_flow_baseflow].to_numpy(dtype=float)
    da_flow_maximum = df[s_flow_qmax].to_numpy(dtype=float)

    # Return to the calling function
    return da_comid, da_base_flow, da_flow_maximum


def get_stream_slope_information(i_row: int, i_column: int, dm_dem: np.ndarray, im_streams: np.ndarray, d_dx: float, d_dy: float):
    """
    Calculates the stream slope using the following process:

        1.) Find all stream cells within the Gen_Slope_Dist that have the same stream id value
        2.) Look at the slope of each of the stream cells.
        3.) Average the slopes to get the overall slope we use in the model.

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
        da_matching_elevations = dm_elevation_box[ia_matching_row_indices, ia_matching_column_indices]
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

    # Return the slope to the calling function
    return d_stream_slope


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
        
        # Cross-Section Direction is just perpendicular to the Stream Direction
        d_xs_direction = d_stream_direction - math.pi / 2.0

        if d_xs_direction < 0.0:
            # Check that the cross section direction is reasonable
            d_xs_direction = d_xs_direction + math.pi

    # Return to the calling function
    return d_stream_direction, d_xs_direction


def get_xs_index_values(i_entry_cell: int, ia_xc_dr_index_main: np.ndarray, ia_xc_dc_index_main: np.ndarray, ia_xc_dr_index_second: np.ndarray, ia_xc_dc_index_second: np.ndarray, da_xc_main_fract: np.ndarray,
                        da_xc_second_fract: np.ndarray, d_xs_direction: np.ndarray, i_r_start: int, i_c_start: int, i_centerpoint: int, d_dx: float, d_dy: float):
    """
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
    
    # Determine the best direction to perform calcualtions
    if d_xs_direction >= math.pi / 4 and d_xs_direction <= 3 * math.pi / 4:
        # Calculate the distance in the x direction
        da_distance_x = np.arange(i_centerpoint) * d_dy * math.cos(d_xs_direction)

        # Convert the distance to a number of indices
        ia_x_index_offset = da_distance_x / d_dx
        ia_x_index_offset = ia_x_index_offset.astype(int)

        ia_xc_dr_index_main[np.arange(i_centerpoint)] = np.arange(i_centerpoint)
        ia_xc_dc_index_main[np.arange(i_centerpoint)] = ia_x_index_offset

        # Calculate the sign of the angle
        ia_sign = np.ones(i_centerpoint)
        ia_sign[da_distance_x < 0] = -1

        # Round using the angle direction
        ia_x_index_offset = np.round((da_distance_x / d_dx) + 0.5 * ia_sign, 0)

        # Set the values in as index locations
        ia_xc_dr_index_second[np.arange(i_centerpoint)] = np.arange(i_centerpoint)
        ia_xc_dc_index_second[np.arange(i_centerpoint)] = ia_x_index_offset

        # ddx is the distance from the main cell to the location where the line passes through.  Do 1-ddx to get the weight
        da_ddx = np.fabs((da_distance_x / d_dx) - ia_x_index_offset)
        da_xc_main_fract[np.arange(i_centerpoint)] = 1.0 - da_ddx
        da_xc_second_fract[np.arange(i_centerpoint)] = da_ddx

        # Distance between each increment
        d_distance_z = math.sqrt((d_dy * math.cos(d_xs_direction)) * (d_dy * math.cos(d_xs_direction)) + d_dy * d_dy)


    else:
        # Calculate based on the column being the dominate direction
        # Calculate the distance in the y direction
        da_distance_y = np.arange(i_centerpoint) * d_dx * math.sin(d_xs_direction)

        # Convert the distance to a number of indices
        ia_y_index_offset = da_distance_y / d_dy
        ia_y_index_offset = ia_y_index_offset.astype(int)

        ia_xc_dr_index_main[np.arange(i_centerpoint)] = ia_y_index_offset
        ia_xc_dc_index_main[np.arange(i_centerpoint)] = np.arange(i_centerpoint)

        # Calculate the sign of the angle
        ia_sign = np.ones(i_centerpoint)
        ia_sign[da_distance_y < 0] = -1

        # Round using the angle direction
        ia_y_index_offset = np.round((da_distance_y / d_dy) + 0.5 * ia_sign, 0)

        # Set the values in as index locations
        ia_xc_dr_index_second[np.arange(i_centerpoint)] = ia_y_index_offset
        ia_xc_dc_index_second[np.arange(i_centerpoint)] = np.arange(i_centerpoint)

        # ddy is the distance from the main cell to the location where the line passes through.  Do 1-ddx to get the weight
        da_ddy = np.fabs((da_distance_y / d_dy) - ia_y_index_offset)
        da_xc_main_fract[np.arange(i_centerpoint)] = 1.0 - da_ddy
        da_xc_second_fract[np.arange(i_centerpoint)] = da_ddy

        # Distance between each increment
        d_distance_z = math.sqrt((d_dx * math.sin(d_xs_direction)) * (d_dx * math.sin(d_xs_direction)) + d_dx * d_dx)

    # Return to the calling function
    return d_distance_z


def sample_cross_section_from_dem(i_entry_cell: int, da_xs_profile: np.ndarray, i_row: int, i_column: int, dm_elevation: np.ndarray, i_center_point: int, ia_xc_row_index_main: np.ndarray,
                                  ia_xc_column_index_main: np.ndarray, ia_xc_row_index_second: np.ndarray, ia_xc_column_index_second: np.ndarray, da_xc_main_fract: np.ndarray,
                                  da_xc_second_fract: np.ndarray, i_row_bottom: int, i_row_top: int, i_column_bottom: int, i_column_top: int):
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
        da_xs_profile[0:i_center_point] = dm_elevation[ia_xc_row_index_main[0:i_center_point], ia_xc_column_index_main[0:i_center_point]] * da_xc_main_fract[0:i_center_point] + dm_elevation[ia_xc_row_index_second[0:i_center_point],
                                                   ia_xc_column_index_second[0:i_center_point]] * da_xc_second_fract[0:i_center_point]
    except:
        logging.error('Error on Cell ' + str(i_entry_cell))

    # Return the center point to the calling function
    return i_center_point


def find_bank(da_xs_profile: np.ndarray, i_cross_section_number: int, d_z_target: float):
    """
    Finds the cell containing the bank of the cross section

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
    for entry in range(i_cross_section_number):
        # Check if the profile elevation matches the target elevation
        if da_xs_profile[entry] >= d_z_target:
            return entry - 1
            

    # Return to the calling function
    return i_cross_section_number


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

        # Converge until the calculate flow is above the baseflow
        while d_flow_calculated <= d_baseflow:
            d_working_depth = d_working_depth + d_dy
            d_area = d_working_depth * (d_bottom_width + d_top_width) / 2.0
            d_perimeter = d_bottom_width + 2.0 * math.sqrt(d_average_width * d_average_width + d_working_depth * d_working_depth)
            d_hydraulic_radius = d_area / d_perimeter
            d_flow_calculated = (1.0 / d_mannings_n) * d_area * math.pow(d_hydraulic_radius, (2 / 3)) * pow(d_slope, 0.5)

        # Update the starting depth
        d_depth_start = d_working_depth - d_dy

    # Update the calculated depth
    d_working_depth = d_working_depth - d_dy

    # Debugging variables
    # A = y * (B + TW) / 2.0
    # P = B + 2.0*math.sqrt(H*H + y*y)
    # R = A / P
    # Qcalc = (1.0/n)*A*math.pow(R,(2/3)) * pow(slope,0.5)

    return d_working_depth


def adjust_profile_for_bathymetry(i_entry_cell: int, da_xs_profile: np.ndarray, i_bank_index: int, d_total_bank_dist: float, d_trap_base: float, d_distance_z: float, d_distance_h: float, d_y_bathy: float,
                                  d_y_depth: float, dm_output_bathymetry: np.ndarray, ia_xc_r_index_main: np.ndarray, ia_xc_c_index_main: np.ndarray, nrows: int, ncols: int):
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
        Distance of the slope section of the trapezoidal channel.  Typically d_distance_h = 0.2* Depth of Trapezoid
    d_y_bathy: float
        Bathymetry elevation of the bottom
    d_y_depth: float
        Depth.  Basically water surface elevation (WSE) minus d_y_bathy
    dm_output_bathymetry: ndarray
        Output bathymetry matrix
    ia_xc_r_index_main: ndarray
        Row indices for the stream cross section
    ia_xc_c_index_main: ndarray
        Column indices for the stream corss section

    Returns
    -------
    None. Values are updated in the output bathymetry matrix

    """

    # If banks are calculated, make an adjustment to the trapezoidal bathymetry
    if i_bank_index > 0:
        # Loop over the bank width offset indices
        for x in range(i_bank_index + 1):
            # Calculate the distance to the bank
            d_dist_cell_to_bank = (i_bank_index - x) * d_distance_z

            # If the cell is outside of the banks, then just ignore this cell (set it to it's same elevation).  No need to update the output bathymetry raster.
            if d_dist_cell_to_bank < 0 or d_dist_cell_to_bank > d_total_bank_dist:
                da_xs_profile[x] = da_xs_profile[x]

            # If the cell is in the flat part of the trapezoidal cross-section, set it to the bottom elevation of the trapezoid.
            elif d_dist_cell_to_bank >= d_distance_h and d_dist_cell_to_bank <= (d_trap_base + d_distance_h):
                da_xs_profile[x] = d_y_bathy
                dm_output_bathymetry[ia_xc_r_index_main[x], ia_xc_c_index_main[x]] = da_xs_profile[x]

            # If the cell is in the slope part of the trapezoid you need to find the elevaiton based on the slope of the trapezoid side.
            elif d_dist_cell_to_bank <= d_distance_h:
                da_xs_profile[x] = d_y_bathy + d_y_depth * (1.0 - (d_dist_cell_to_bank / d_distance_h))
                dm_output_bathymetry[ia_xc_r1_index_main[x], ia_xc_c_index_main[x]] = da_xs_profile[x]

            # Similar to above, but on the far-side slope of the trapezoid.  You need to find the elevaiton based on the slope of the trapezoid side.
            elif d_dist_cell_to_bank >= d_trap_base + d_distance_h:
                da_xs_profile[x] = d_y_bathy + d_y_depth * (d_dist_cell_to_bank - (d_trap_base + d_distance_h)) / d_distance_h
                dm_output_bathymetry[ia_xc_r_index_main[x], ia_xc_c_index_main[x]] = da_xs_profile[x]


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
    d_distance = np.sqrt(d_side_one ** 2 + d_side_two ** 2)

    # Return to the calling function
    return d_distance


def calculate_stream_geometry(da_xs_profile: np.ndarray, d_wse: float, d_distance_z: float, da_n_profile: np.ndarray):
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


    # Set initial values for the variables
    d_area = 0.0
    d_perimeter = 0.0
    d_hydraulic_radius = 0.0
    d_top_width = 0.0
    d_composite_n = 0.0

    # Estimate the depth of the stream
    da_y_depth = d_wse - da_xs_profile

    # Estimate geometry if the depth is valid
    if da_y_depth.shape[0] > 0 and da_y_depth[0] > 1e-16:
        # Calculate the good values for later use
        da_area_good = d_distance_z * 0.5 * (da_y_depth[1:] + da_y_depth[:-1])
        da_perimeter_i_good = calculate_hypotnuse(d_distance_z, (da_y_depth[1:] - da_y_depth[:-1]))

        # Correct the Mannings values if too high or low
        da_n_profile_copy = np.copy(da_n_profile)
        da_n_profile_copy[da_n_profile_copy > 0.3] = 0.035
        da_n_profile_copy[da_n_profile_copy < 0.005] = 0.005

        # Calculate the Mannings n for later use
        da_composite_n_good = da_perimeter_i_good * np.power(da_n_profile_copy[1:], 1.5)

        # Take action if there are bad values
        if np.any(da_y_depth[1:] <= 0):
            # A bad value exists. Calculate up to that value then break for the rest of hte values.
            # Get the index of the first bad vadlue
            i_target_index = int(np.argwhere(da_y_depth[1:] <= 0)[0]) + 1

            # Calculate the distance to use
            d_dist_use = d_distance_z * da_y_depth[i_target_index - 1] / (abs(da_y_depth[i_target_index - 1]) + abs(da_y_depth[i_target_index]))

            # Calculate the geometric variables
            d_area = np.sum(da_area_good[:i_target_index - 1]) + 0.5 * d_dist_use * da_y_depth[i_target_index-1]
            d_perimeter_i = calculate_hypotnuse(d_dist_use, da_y_depth[i_target_index - 1])
            d_perimeter = np.sum(da_perimeter_i_good[:i_target_index - 1]) + d_perimeter_i
            d_hydraulic_radius = d_area / d_perimeter

            # Calculate the composite n
            d_composite_n = np.sum(da_composite_n_good[:i_target_index - 1]) + d_perimeter_i * np.power(da_n_profile_copy[i_target_index - 1], 1.5)

            # Update the top width
            d_top_width = d_distance_z * (i_target_index - 1) + d_dist_use

        else:
            # All values are good, so include them all.
            # Calculate teh geometric values
            d_area = np.sum(da_area_good[da_y_depth[1:] > 0])
            d_perimeter = np.sum(da_perimeter_i_good[da_y_depth[1:] > 0])
            d_hydraulic_radius = d_area / d_perimeter

            # Calculate the composite Mannings N
            d_composite_n = np.sum(da_composite_n_good)

            # Update the top width
            d_top_width = d_distance_z * np.sum(da_y_depth[1:] > 0)

    # Return to the calling function
    return d_area, d_perimeter, d_hydraulic_radius, d_composite_n, d_top_width


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
    o_input_file = open(s_manning_path, 'r')
    sl_lines = o_input_file.readlines()
    o_input_file.close()

    # Calculate the number of lines in the file
    i_number_of_lines = len(sl_lines)

    # Extract the roughness from the file
    for i_entry in range(1, i_number_of_lines):
        # Split the line
        sl_line_split = sl_lines[i_entry].strip().split('\t')

        # Store the information into the list
        da_input_mannings[da_input_mannings == int(sl_line_split[0])] = float(sl_line_split[2])

    # Return to the calling function
    return da_input_mannings


def adjust_cross_section_to_lowest_point(i_low_point_index, d_dem_low_point_elev, da_xs_profile_one, da_xs_profile_two, ia_xc_r1_index_main, ia_xc_r2_index_main, ia_xc_c1_index_main, ia_xc_c2_index_main, da_xs1_mannings, da_xs2_mannings,
                                         i_center_point, nrows, ncols, i_boundary_number):
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
    return i_low_point_index, d_dem_low_point_elev

    
    
def main(MIF_Name: str):
    ### Read Main Input File ###
    read_main_input_file(MIF_Name)
    
    ### Read the Flow Information ###
    COMID, QBaseFlow, QMax = read_flow_file(s_input_flow_file_path, s_flow_file_id, s_flow_file_baseflow, s_flow_file_qmax)

    ### Read Raster Data ###
    DEM, dncols, dnrows, dcellsize, dyll, dyur, dxll, dxur, dlat, dem_geotransform, dem_projection = read_raster_gdal(s_input_dem_path)
    STRM, sncols, snrows, scellsize, syll, syur, sxll, sxur, slat, strm_geotransform, strm_projection = read_raster_gdal(s_input_stream_path)
    LC, lncols, lnrows, lcellsize, lyll, lyur, lxll, lxur, llat, land_geotransform, land_projection = read_raster_gdal(s_input_land_use_path)

    if dnrows != snrows or dnrows != lnrows:
        logging.warning('Rows do not Match!')
    else:
        nrows = dnrows

    if dncols != sncols or dncols != lncols:
        logging.warning('Cols do not Match!')
    else:
        ncols = dncols

    ### If you're outputting a cross-section file start the list here
    if s_xs_output_file != '':
        o_xs_file = open(s_xs_output_file, 'w')

    ### Imbed the Stream and DEM data within a larger Raster to help with the boundary issues. ###
    i_boundary_number = max(1, i_general_slope_distance, i_general_direction_distance)

    dm_stream = np.zeros((nrows + i_boundary_number * 2, ncols + i_boundary_number * 2))
    dm_stream[i_boundary_number:(nrows + i_boundary_number), i_boundary_number:(ncols + i_boundary_number)] = STRM

    dm_elevation = np.zeros((nrows + i_boundary_number * 2, ncols + i_boundary_number * 2))
    dm_elevation[i_boundary_number:(nrows + i_boundary_number), i_boundary_number:(ncols + i_boundary_number)] = DEM

    dm_land_use = np.zeros((nrows + i_boundary_number * 2, ncols + i_boundary_number * 2))
    dm_land_use[i_boundary_number:(nrows + i_boundary_number), i_boundary_number:(ncols + i_boundary_number)] = LC
    
    ### Read in the Manning Table ###
    dm_manning_n_raster = read_manning_table(s_input_mannings_path, dm_land_use)


    ##### Begin Calculations #####
    # Create working matrices
    ep = 1000
    da_total_t = np.zeros(ep, dtype=float)
    da_total_a = np.zeros(ep, dtype=float)
    da_total_p = np.zeros(ep, dtype=float)
    da_total_v = np.zeros(ep, dtype=float)
    da_total_q = np.zeros(ep, dtype=float)
    da_total_wse = np.zeros(ep, dtype=float)

    # Create output rasters
    dm_output_bathymetry = np.zeros((nrows + i_boundary_number * 2, ncols + i_boundary_number * 2))
    dm_out_flood = np.zeros((nrows + i_boundary_number * 2, ncols + i_boundary_number * 2)).astype(int)

    # Get the list of stream locations
    ia_valued_row_indices, ia_valued_column_indices = dm_stream.nonzero()
    i_number_of_stream_cells = len(ia_valued_row_indices)

    # Get the cell dx and dy coordinates
    dx, dy, dproject = convert_cell_size(dcellsize, dyll, dyur)
    logging.info('Cellsize X = ' + str(dx))
    logging.info('Cellsize Y = ' + str(dy))
    
    # Pull cross sections
    i_center_point = int((d_x_section_distance / (sum([dx, dy]) * 0.5)) / 2.0) + 1
    da_xs_profile1 = np.zeros(i_center_point + 1)
    da_xs_profile2 = np.zeros(i_center_point + 1)
    ia_xc_dr_index_main = np.zeros(i_center_point + 1, dtype=int)  # Only need to go to center point, because the other side of xs we can just use *-1
    ia_xc_dc_index_main = np.zeros(i_center_point + 1, dtype=int)  # Only need to go to center point, because the other side of xs we can just use *-1
    ia_xc_dr_index_second = np.zeros(i_center_point + 1, dtype=int)  # Only need to go to center point, because the other side of xs we can just use *-1
    ia_xc_dc_index_second = np.zeros(i_center_point + 1, dtype=int)  # Only need to go to center point, because the other side of xs we can just use *-1
    da_xc_main_fract = np.zeros(i_center_point + 1)
    da_xc_second_fract = np.zeros(i_center_point + 1)

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

    logging.info('With Degree_Manip=' + str(d_degree_manipulation) + '  and  Degree_Interval=' + str(d_degree_interval) + '\n  Angles to evaluate= ' + str(l_angles_to_test))
    
    # Get the extents of the boundaries
    i_row_bottom = i_boundary_number
    i_row_top = nrows + i_boundary_number-1
    i_column_bottom = i_boundary_number
    i_column_top = ncols + i_boundary_number-1
    
    # Open the output file and write the initial metadata
    o_out_file = open(s_output_vdt_database, 'w')
    o_out_file.write('COMID,Row,Col,Elev,QBaseflow,Q,V,T,WSE')
    
    #Open the output Curve file and write the initial metadata
    if len(s_output_curve_file)>0:
        o_curve_file = open(s_output_curve_file, 'w')
        o_curve_file.write('COMID,Row,Col,BaseElev,DEM_Elev,QMax,depth_a,depth_b,tw_a,tw_b,vel_a,vel_b')

    # Write the percentiles into the files
    logging.info('Looking at ' + str(i_number_of_stream_cells) + ' stream cells')
    da_percent_cells = np.array([1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 99, 100, 110])
    da_percent_cells = (da_percent_cells * i_number_of_stream_cells / 100).astype(int)

    ### Begin the stream cell solution loop ###
    for i_entry_cell in tqdm.tqdm(range(i_number_of_stream_cells), total=i_number_of_stream_cells):
        # Get the metadata for the loop
        i_row_cell = ia_valued_row_indices[i_entry_cell]
        i_column_cell = ia_valued_column_indices[i_entry_cell]
        i_cell_comid = int(dm_stream[i_row_cell,i_column_cell])
        
        # Get the Stream Direction of each Stream Cell.  Direction is between 0 and pi.  Also get the cross-section direction (also between 0 and pi)
        d_stream_direction, d_xs_direction = get_stream_direction_information(i_row_cell, i_column_cell, dm_stream, dx, dy)
        
        # Get the Slope of each Stream Cell. Slope should be in m/m
        d_stream_slope = get_stream_slope_information(i_row_cell, i_column_cell, dm_elevation, dm_stream, dx, dy)

        # Set a minimum threshold for the slope
        d_slope_use = d_stream_slope

        # if slope is less than the threshold, reset it
        if d_slope_use < 0.0002:
            d_slope_use = 0.0002
        
        # Get the Flow Rates Associated with the Stream Cell
        try:
            im_flow_index = np.where(COMID == int(dm_stream[i_row_cell, i_column_cell]))
            im_flow_index = int(im_flow_index[0][0])
            d_q_baseflow = QBaseFlow[im_flow_index]
            d_q_maximum = QMax[im_flow_index]
        except:
            continue
    
        # Get the Cross-Section Ordinates
        d_distance_z = get_xs_index_values(i_entry_cell, ia_xc_dr_index_main, ia_xc_dc_index_main, ia_xc_dr_index_second, ia_xc_dc_index_second, da_xc_main_fract, da_xc_second_fract, d_xs_direction, i_row_cell,
                                           i_column_cell, i_center_point, dx, dy)
        
        # Now Pull a Cross-Section
        ia_xc_r1_index_main = i_row_cell + ia_xc_dr_index_main
        ia_xc_r2_index_main = i_row_cell + ia_xc_dr_index_main * -1
        ia_xc_c1_index_main = i_column_cell + ia_xc_dc_index_main
        ia_xc_c2_index_main = i_column_cell + ia_xc_dc_index_main * -1

        # todo: These appear to be resetting the center point only?
        xs1_n = sample_cross_section_from_dem(i_entry_cell, da_xs_profile1, i_row_cell, i_column_cell, dm_elevation, i_center_point, ia_xc_r1_index_main, ia_xc_c1_index_main, i_row_cell + ia_xc_dr_index_second,
                                              i_column_cell + ia_xc_dc_index_second, da_xc_main_fract, da_xc_second_fract, i_row_bottom, i_row_top, i_column_bottom, i_column_top)
        xs2_n = sample_cross_section_from_dem(i_entry_cell, da_xs_profile2, i_row_cell, i_column_cell, dm_elevation, i_center_point, ia_xc_r2_index_main, ia_xc_c2_index_main, i_row_cell + ia_xc_dr_index_second * -1,
                                              i_column_cell + ia_xc_dc_index_second * -1, da_xc_main_fract, da_xc_second_fract, i_row_bottom, i_row_top, i_column_bottom, i_column_top)

        # Adjust to the lowest-point in the Cross-Section
        i_lowest_point_index_offset=0
        d_dem_low_point_elev = da_xs_profile1[0]
        if i_low_spot_range > 0:
            i_lowest_point_index_offset = adjust_cross_section_to_lowest_point(i_lowest_point_index_offset, d_dem_low_point_elev, da_xs_profile1, da_xs_profile2, ia_xc_r1_index_main,
                                                                               ia_xc_r2_index_main, ia_xc_c1_index_main, ia_xc_c2_index_main, xs1_n, xs2_n, i_center_point, nrows, ncols,
                                                                               i_boundary_number)

   
        # The r and c for the stream cell is adjusted because it may have moved
        i_row_cell = ia_xc_r1_index_main[0]
        i_column_cell = ia_xc_c1_index_main[0]

        # re-sample the cross-section to make sure all of the low-spot data has the same values through interpolation
        if abs(i_lowest_point_index_offset) > 0:
            xs1_n = sample_cross_section_from_dem(i_entry_cell, da_xs_profile1, i_row_cell, i_column_cell, dm_elevation, i_center_point, ia_xc_r1_index_main, ia_xc_c1_index_main, i_row_cell + ia_xc_dr_index_second,
                                                    i_column_cell + ia_xc_dc_index_second, da_xc_main_fract, da_xc_second_fract, i_row_bottom, i_row_top, i_column_bottom, i_column_top)
            xs2_n = sample_cross_section_from_dem(i_entry_cell, da_xs_profile2, i_row_cell, i_column_cell, dm_elevation, i_center_point, ia_xc_r2_index_main, ia_xc_c2_index_main, i_row_cell + ia_xc_dr_index_second * -1,
                                                    i_column_cell + ia_xc_dc_index_second * -1, da_xc_main_fract, da_xc_second_fract, i_row_bottom, i_row_top, i_column_bottom, i_column_top)
            # set the low-spot value to the new low spot in the cross-section 
            d_dem_low_point_elev = da_xs_profile1[0]

        # Adjust cross-section angle to ensure shortest top-width at a specified depth
        d_t_test = d_x_section_distance * 3.0
        d_shortest_tw_angle = d_xs_direction
        d_test_depth = 0.5

        if d_increments > 0:
            for d_entry_angle_adjustment in l_angles_to_test:
                d_xs_angle_use = d_xs_direction + d_entry_angle_adjustment
                if d_xs_angle_use > math.pi:
                    d_xs_angle_use = d_xs_angle_use - math.pi
                if d_xs_angle_use < 0.0:
                    d_xs_angle_use = d_xs_angle_use + math.pi
                
                # Get XS ordinates... again
                d_distance_z = get_xs_index_values(i_entry_cell, ia_xc_dr_index_main, ia_xc_dc_index_main, ia_xc_dr_index_second, ia_xc_dc_index_second, da_xc_main_fract, da_xc_second_fract, d_xs_angle_use,
                                                   i_row_cell, i_column_cell, i_center_point, dx, dy)
                
                # Pull the cross-section again
                ia_xc_r1_index_main = i_row_cell + ia_xc_dr_index_main
                ia_xc_r2_index_main = i_row_cell + ia_xc_dr_index_main * -1
                ia_xc_c1_index_main = i_column_cell + ia_xc_dc_index_main
                ia_xc_c2_index_main = i_column_cell + ia_xc_dc_index_main * -1

                xs1_n = sample_cross_section_from_dem(i_entry_cell, da_xs_profile1, i_row_cell, i_column_cell, dm_elevation, i_center_point, ia_xc_r1_index_main, ia_xc_c1_index_main, i_row_cell + ia_xc_dr_index_second,
                                                      i_column_cell + ia_xc_dc_index_second, da_xc_main_fract, da_xc_second_fract, i_row_bottom, i_row_top, i_column_bottom, i_column_top)
                xs2_n = sample_cross_section_from_dem(i_entry_cell, da_xs_profile2, i_row_cell, i_column_cell, dm_elevation, i_center_point, ia_xc_r2_index_main, ia_xc_c2_index_main, i_row_cell + ia_xc_dr_index_second * -1,
                                                      i_column_cell + ia_xc_dc_index_second * -1, da_xc_main_fract, da_xc_second_fract, i_row_bottom, i_row_top, i_column_bottom, i_column_top)
                
                d_wse = da_xs_profile1[0] + d_test_depth
                A1, P1, R1, np1, T1 = calculate_stream_geometry(da_xs_profile1[0:xs1_n], d_wse, d_distance_z, dm_manning_n_raster[ia_xc_r1_index_main[0:xs1_n], ia_xc_c1_index_main[0:xs1_n]])
                A2, P2, R2, np2, T2 = calculate_stream_geometry(da_xs_profile2[0:xs2_n], d_wse, d_distance_z, dm_manning_n_raster[ia_xc_r2_index_main[0:xs2_n], ia_xc_c2_index_main[0:xs2_n]])

                if T1 + T2 < d_t_test:
                    d_t_test = T1 + T2
                    d_shortest_tw_angle = d_xs_angle_use

            # Now rerun everything with the shortest top-width angle
            d_xs_direction = d_shortest_tw_angle
            d_distance_z = get_xs_index_values(i_entry_cell, ia_xc_dr_index_main, ia_xc_dc_index_main, ia_xc_dr_index_second, ia_xc_dc_index_second, da_xc_main_fract, da_xc_second_fract, d_xs_direction, i_row_cell,
                                               i_column_cell, i_center_point, dx, dy)

            ia_xc_r1_index_main = i_row_cell + ia_xc_dr_index_main
            ia_xc_r2_index_main = i_row_cell + ia_xc_dr_index_main * -1
            ia_xc_c1_index_main = i_column_cell + ia_xc_dc_index_main
            ia_xc_c2_index_main = i_column_cell + ia_xc_dc_index_main * -1
            
            #Set the index values to be within the confines of the raster
            ia_xc_r1_index_main = np.clip(ia_xc_r1_index_main,0,nrows+2*i_boundary_number-1)
            ia_xc_r2_index_main = np.clip(ia_xc_r2_index_main,0,nrows+2*i_boundary_number-1)
            ia_xc_c1_index_main = np.clip(ia_xc_c1_index_main,0,ncols+2*i_boundary_number-1)
            ia_xc_c2_index_main = np.clip(ia_xc_c2_index_main,0,ncols+2*i_boundary_number-1)

            xs1_n = sample_cross_section_from_dem(i_entry_cell, da_xs_profile1, i_row_cell, i_column_cell, dm_elevation, i_center_point, ia_xc_r1_index_main, ia_xc_c1_index_main, i_row_cell + ia_xc_dr_index_second,
                                                  i_column_cell + ia_xc_dc_index_second, da_xc_main_fract, da_xc_second_fract, i_row_bottom, i_row_top, i_column_bottom, i_column_top)
            xs2_n = sample_cross_section_from_dem(i_entry_cell, da_xs_profile2, i_row_cell, i_column_cell, dm_elevation, i_center_point, ia_xc_r2_index_main, ia_xc_c2_index_main, i_row_cell + ia_xc_dr_index_second * -1,
                                                  i_column_cell + ia_xc_dc_index_second * -1, da_xc_main_fract, da_xc_second_fract, i_row_bottom, i_row_top, i_column_bottom, i_column_top)
        
        # Burn bathymetry profile into cross-section profile
        i_bank_1_index = find_bank(da_xs_profile1, xs1_n, d_dem_low_point_elev + 0.1)
        i_bank_2_index = find_bank(da_xs_profile2, xs2_n, d_dem_low_point_elev + 0.1)
        i_total_bank_cells = i_bank_1_index + i_bank_2_index - 1

        # Set a minimum for the number of bank cells
        if i_total_bank_cells < 1:
            i_total_bank_cells = 1

        # Calculate the trapezoid dimensions
        d_total_bank_dist = i_total_bank_cells * d_distance_z
        d_h_dist = d_bathymetry_trapzoid_height * d_total_bank_dist
        d_trap_base = d_total_bank_dist - 2.0 * d_h_dist

        if d_q_baseflow > 0.0:
            d_y_depth = find_depth_of_bathymetry(d_q_baseflow, d_trap_base, d_total_bank_dist, d_slope_use, 0.03)
            d_y_bathy = da_xs_profile1[0] - d_y_depth
            da_xs_profile1[0] = d_y_bathy
            da_xs_profile2[0] = d_y_bathy

            if i_total_bank_cells > 1:
                adjust_profile_for_bathymetry(i_entry_cell, da_xs_profile1, i_bank_1_index, d_total_bank_dist, d_trap_base, d_distance_z, d_h_dist, d_y_bathy, d_y_depth, dm_output_bathymetry, ia_xc_r1_index_main, ia_xc_c1_index_main, nrows, ncols)
                adjust_profile_for_bathymetry(i_entry_cell, da_xs_profile2, i_bank_2_index, d_total_bank_dist, d_trap_base, d_distance_z, d_h_dist, d_y_bathy, d_y_depth, dm_output_bathymetry, ia_xc_r2_index_main, ia_xc_c2_index_main, nrows, ncols)

        else:
            d_y_depth = 0.0
            d_y_bathy = da_xs_profile1[0] - d_y_depth

        # Get a list of elevations within the cross-section profile that we need to evaluate
        da_elevation_list_mm = np.unique(np.concatenate((da_xs_profile1[0:xs1_n] * 1000, da_xs_profile1[0:xs2_n] * 1000)).astype(int))
        da_elevation_list_mm = da_elevation_list_mm[np.logical_and(da_elevation_list_mm[:] > 0, da_elevation_list_mm[:] < 99999900)]
        da_elevation_list_mm = np.sort(da_elevation_list_mm)

        i_number_of_elevations = len(da_elevation_list_mm)
        if i_number_of_elevations <= 0:
            continue

        if i_number_of_elevations >= ep:
            logging.error('ERROR, HAVE TOO MANY ELEVATIONS TO EVALUATE')
            continue
        
        # Calculate the volumes
        """
        VolumeFillApproach 1 is to find the height within ElevList_mm that corresponds to the Qmax flow.  THen increment depths to have a standard number of depths to get to Qmax.  
        This is preferred for VDTDatabase method.
        
        VolumeFillApproach 2 just looks at the different elevation points wtihin ElevList_mm.  It also adds some in if the gaps between depths is too large.
        """
        
        
        #This is the Stream Cell Location
        dm_out_flood[int(i_row_cell),int(i_column_cell)] = 3
        

        # Initialize default values
        da_total_t = da_total_t * 0
        da_total_a = da_total_a * 0
        da_total_p = da_total_p * 0
        da_total_v = da_total_v * 0
        da_total_q = da_total_q * 0
        da_total_wse = da_total_wse * 0

        # Solve using the volume fill approach
        i_volume_fill_approach = 1
        
        # This just tells the curve file whether to print out a result or not.  If no realistic depths were calculated, no reason to output results.
        i_outprint_yes = 0
        
        # This is the first and last indice of elevations we'll need for the Curve Fitting for this cell
        i_start_elevation_index = -1
        i_last_elevation_index = 0
        
        # To find the depth / wse where the maximum flow occurs we use two sets of incremental depths.  The first is 0.5m followed by 0.05m
        d_depth_increment_big = 0.5
        d_depth_increment_med = 0.05
        d_depth_increment_small = 0.01

        if i_volume_fill_approach==1:
            # Find elevation where maximum flow is hit
            i_ordinate_for_Qmax = 0
            
            '''
            for i_entry_elevation in range(1, i_number_of_elevations):
                # Calculate the geometry
                d_wse = da_elevation_list_mm[i_entry_elevation] / 1000.0
                A1, P1, R1, np1, T1 = calculate_stream_geometry(da_xs_profile1[0:xs1_n], d_wse, d_distance_z, dm_manning_n_raster[ia_xc_r1_index_main[0:xs1_n], ia_xc_c1_index_main[0:xs1_n]])
                A2, P2, R2, np2, T2 = calculate_stream_geometry(da_xs_profile2[0:xs2_n], d_wse, d_distance_z, dm_manning_n_raster[ia_xc_r2_index_main[0:xs2_n], ia_xc_c2_index_main[0:xs2_n]])

                # Aggregate the geometric properties
                d_a_sum = A1 + A2
                d_p_sum = P1 + P2
                d_t_sum = T1 + T2
                d_q_sum = 0.0

                # Estimate mannings n and flow
                if d_a_sum > 0.0 and d_p_sum > 0.0 and d_t_sum > 0.0:
                    d_composite_n = math.pow(((np1 + np2) / d_p_sum), (2 / 3))
                    d_q_sum = (1 / d_composite_n) * d_a_sum * math.pow((d_a_sum / d_p_sum), (2 / 3)) * math.pow(d_slope_use, 0.5)

                # Perform check on the maximum flow
                if d_q_sum > d_q_maximum:
                    i_ordinate_for_Qmax = i_entry_elevation
                    break
            i_number_of_increments = 15
            d_inc_y = ((da_elevation_list_mm[i_ordinate_for_Qmax] - da_elevation_list_mm[0]) / 1000.0) / i_number_of_increments
            i_number_of_elevations = i_number_of_increments + 1
            '''
            
            # Find the wse associated with the Maximum flow using 0.5m increments
            d_maxflow_wse_initial = da_xs_profile1[0]
            for i_depthincrement in range(1,101):
                # Calculate the geometry
                d_wse = da_xs_profile1[0] + i_depthincrement * d_depth_increment_big
                A1, P1, R1, np1, T1 = calculate_stream_geometry(da_xs_profile1[0:xs1_n], d_wse, d_distance_z, dm_manning_n_raster[ia_xc_r1_index_main[0:xs1_n], ia_xc_c1_index_main[0:xs1_n]])
                A2, P2, R2, np2, T2 = calculate_stream_geometry(da_xs_profile2[0:xs2_n], d_wse, d_distance_z, dm_manning_n_raster[ia_xc_r2_index_main[0:xs2_n], ia_xc_c2_index_main[0:xs2_n]])

                # Aggregate the geometric properties
                d_a_sum = A1 + A2
                d_p_sum = P1 + P2
                d_t_sum = T1 + T2
                d_q_sum = 0.0

                # Estimate mannings n and flow
                if d_a_sum > 0.0 and d_p_sum > 0.0 and d_t_sum > 0.0:
                    d_composite_n = math.pow(((np1 + np2) / d_p_sum), (2 / 3))
                    d_q_sum = (1 / d_composite_n) * d_a_sum * math.pow((d_a_sum / d_p_sum), (2 / 3)) * math.pow(d_slope_use, 0.5)
                
                d_maxflow_wse_initial = d_wse
                
                # Perform check on the maximum flow
                if d_q_sum > d_q_maximum:
                    break
            
            # Based on using depth increments of 0.5, now lets fine-tune the wse using depth increments of 0.05
            d_maxflow_wse_initial = d_maxflow_wse_initial - 0.5
            if d_maxflow_wse_initial < da_xs_profile1[0]:
                d_maxflow_wse_initial = da_xs_profile1[0]
            d_maxflow_wse_med = d_maxflow_wse_initial
            
            for i_depthincrement in range(1,101):
                # Calculate the geometry
                d_wse = d_maxflow_wse_initial + i_depthincrement * d_depth_increment_med
                A1, P1, R1, np1, T1 = calculate_stream_geometry(da_xs_profile1[0:xs1_n], d_wse, d_distance_z, dm_manning_n_raster[ia_xc_r1_index_main[0:xs1_n], ia_xc_c1_index_main[0:xs1_n]])
                A2, P2, R2, np2, T2 = calculate_stream_geometry(da_xs_profile2[0:xs2_n], d_wse, d_distance_z, dm_manning_n_raster[ia_xc_r2_index_main[0:xs2_n], ia_xc_c2_index_main[0:xs2_n]])

                # Aggregate the geometric properties
                d_a_sum = A1 + A2
                d_p_sum = P1 + P2
                d_t_sum = T1 + T2
                d_q_sum = 0.0

                # Estimate mannings n and flow
                if d_a_sum > 0.0 and d_p_sum > 0.0 and d_t_sum > 0.0:
                    d_composite_n = math.pow(((np1 + np2) / d_p_sum), (2 / 3))
                    d_q_sum = (1 / d_composite_n) * d_a_sum * math.pow((d_a_sum / d_p_sum), (2 / 3)) * math.pow(d_slope_use, 0.5)
                
                d_maxflow_wse_med = d_wse
                
                # Perform check on the maximum flow
                if d_q_sum > d_q_maximum:
                    break
            
            # Based on using depth increments of 0.05, now lets fine-tune the wse even more using depth increments of 0.01
            d_maxflow_wse_med = d_maxflow_wse_med - 0.05
            if d_maxflow_wse_med < da_xs_profile1[0]:
                d_maxflow_wse_med = da_xs_profile1[0]
            d_maxflow_wse_final = d_maxflow_wse_med
            
            for i_depthincrement in range(1,51):
                # Calculate the geometry
                d_wse = d_maxflow_wse_med + i_depthincrement * d_depth_increment_small
                A1, P1, R1, np1, T1 = calculate_stream_geometry(da_xs_profile1[0:xs1_n], d_wse, d_distance_z, dm_manning_n_raster[ia_xc_r1_index_main[0:xs1_n], ia_xc_c1_index_main[0:xs1_n]])
                A2, P2, R2, np2, T2 = calculate_stream_geometry(da_xs_profile2[0:xs2_n], d_wse, d_distance_z, dm_manning_n_raster[ia_xc_r2_index_main[0:xs2_n], ia_xc_c2_index_main[0:xs2_n]])

                # Aggregate the geometric properties
                d_a_sum = A1 + A2
                d_p_sum = P1 + P2
                d_t_sum = T1 + T2
                d_q_sum = 0.0

                # Estimate mannings n and flow
                if d_a_sum > 0.0 and d_p_sum > 0.0 and d_t_sum > 0.0:
                    d_composite_n = math.pow(((np1 + np2) / d_p_sum), (2 / 3))
                    d_q_sum = (1 / d_composite_n) * d_a_sum * math.pow((d_a_sum / d_p_sum), (2 / 3)) * math.pow(d_slope_use, 0.5)
                
                d_maxflow_wse_final = d_wse
                
                # Perform check on the maximum flow
                if d_q_sum > d_q_maximum:
                    break
                '''
                #This is the Edge of the Flood.  Value is 1 because that is the flood method used.
                dm_out_flood[ia_xc_r1_index_main[np1], ia_xc_c1_index_main[np1]] = 1
                dm_out_flood[ia_xc_r2_index_main[np2], ia_xc_c2_index_main[np2]] = 1
                '''
            
            #If the max flow calculated from the cross-section is 20% high or low, just skip this cell
            if d_q_sum > d_q_maximum * 1.5 or d_q_sum < d_q_maximum * 0.5:
                continue
            
            # Now lets get a set number of increments between the low elevation and the elevation where Qmax hits
            i_number_of_increments = 15
            d_inc_y = (d_maxflow_wse_final - da_xs_profile1[0]) / i_number_of_increments
            i_number_of_elevations = i_number_of_increments + 1

            for i_entry_elevation in range(i_number_of_increments + 1):
                # Calculate the geometry
                d_wse = da_xs_profile1[0] + d_inc_y * i_entry_elevation

                    
                A1, P1, R1, np1, T1 = calculate_stream_geometry(da_xs_profile1[0:xs1_n], d_wse, d_distance_z, dm_manning_n_raster[ia_xc_r1_index_main[0:xs1_n], ia_xc_c1_index_main[0:xs1_n]])
                A2, P2, R2, np2, T2 = calculate_stream_geometry(da_xs_profile2[0:xs2_n], d_wse, d_distance_z, dm_manning_n_raster[ia_xc_r2_index_main[0:xs2_n], ia_xc_c2_index_main[0:xs2_n]])

                # Aggregate the geometric properties
                da_total_t[i_entry_elevation] = T1 + T2
                da_total_a[i_entry_elevation] = A1 + A2
                da_total_p[i_entry_elevation] = P1 + P2

                # Check the properties are physically realistic. If so, estimate the flow with them.
                if da_total_t[i_entry_elevation] <= 0.0 or da_total_a[i_entry_elevation] <= 0.0 or da_total_p[i_entry_elevation] <= 0.0:
                    da_total_t[i_entry_elevation] = 0.0
                    da_total_a[i_entry_elevation] = 0.0
                    da_total_p[i_entry_elevation] = 0.0
                    # this is the channel bottom elevation that we will use as depth = 0 when building the power functions
                    da_total_wse[i_entry_elevation] = d_wse
                    i_start_elevation_index = i_entry_elevation

                else:
                    # Estimate mannings n
                    d_composite_n = math.pow(((np1 + np2) / da_total_p[i_entry_elevation]), (2 / 3))

                    # Check that the mannings n is physically realistic
                    if d_composite_n < 0.0001:
                        d_composite_n = 0.035

                    # Estimate total flows
                    da_total_q[i_entry_elevation] = ((1 / d_composite_n) * da_total_a[i_entry_elevation] * math.pow((da_total_a[i_entry_elevation] / da_total_p[i_entry_elevation]), (2 / 3)) *
                                                    math.pow(d_slope_use, 0.5))
                    da_total_v[i_entry_elevation] = da_total_q[i_entry_elevation] / da_total_a[i_entry_elevation]
                    da_total_wse[i_entry_elevation] = d_wse
                    i_last_elevation_index = i_entry_elevation
            
            # Process each of the elevations to the output file if feasbile values were produced
            for i_entry_elevation in range(i_number_of_elevations - 1, 0, -1):
                if i_entry_elevation == i_number_of_elevations-1:
                    if sum(da_total_q[0:int(i_number_of_elevations / 2.0)]) <= 1e-16 or i_row_cell < 0 or i_column_cell < 0 or dm_elevation[i_row_cell, i_column_cell] <= 1e-16:  # Need at least half the increments filled in order to report results.
                        break
                    else:
                        o_out_file.write('\n' + str(i_cell_comid) + ',' + str(int(i_row_cell - i_boundary_number)) + ',' + str(int(i_column_cell - i_boundary_number)) + ",{:.3f}".format(dm_elevation[i_row_cell,i_column_cell]) + ",{:.3f}".format(d_q_baseflow))
                        # o_out_file.write('\n' + str(i_cell_comid) + ',' + str(int(i_row_cell - i_boundary_number)) + ',' + str(int(i_column_cell - i_boundary_number)) + ",{:.3f}".format(da_xs_profile1[0]) + ",{:.3f}".format(d_q_baseflow))

                o_out_file.write(",{:.3f}".format(da_total_q[i_entry_elevation]) + ",{:.3f}".format(da_total_v[i_entry_elevation]) + ",{:.3f}".format(da_total_t[i_entry_elevation]) + ",{:.3f}".format(da_total_wse[i_entry_elevation]))
                i_outprint_yes = 1
        
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
                logging.error('ERROR, HAVE TOO MANY ELEVATIONS TO EVALUATE')

            for i_entry_elevation in range(1, i_number_of_elevations):
                # Calculate the geometry
                d_wse = da_elevation_list_mm[i_entry_elevation] / 1000.0
                A1, P1, R1, np1, T1 = calculate_stream_geometry(da_xs_profile1[0:xs1_n], d_wse, d_distance_z, dm_manning_n_raster[ia_xc_r1_index_main[0:xs1_n], ia_xc_c1_index_main[0:xs1_n]])
                A2, P2, R2, np2, T2 = calculate_stream_geometry(da_xs_profile2[0:xs2_n], d_wse, d_distance_z, dm_manning_n_raster[ia_xc_r2_index_main[0:xs2_n], ia_xc_c2_index_main[0:xs2_n]])

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
                    da_total_q[i_entry_elevation] = ((1 / d_composite_n) * da_total_a[i_entry_elevation] * math.pow((da_total_a[i_entry_elevation] / da_total_p[i_entry_elevation]), (2 / 3)) * math.pow(d_slope_use, 0.5))
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
            for i_entry_elevation in range(i_number_of_elevations - 1, 0, -1):
                if i_entry_elevation == i_number_of_elevations - 1:
                    if da_total_q[i_entry_elevation] < d_q_maximum or da_total_q[i_entry_elevation] > d_q_maximum * 3:
                        break
                    else:
                        o_out_file.write('\n' + str(i_cell_comid) + ',' + str(int(i_row_cell - i_boundary_number)) + ',' + str(int(i_column_cell - i_boundary_number)) + ",{:.3f}".format(dm_elevation[i_row_cell,i_column_cell]) + ",{:.3f}".format(d_q_baseflow))

                o_out_file.write(",{:.3f}".format(da_total_q[i_entry_elevation]) + ",{:.3f}".format(da_total_v[i_entry_elevation]) + ",{:.3f}".format(da_total_t[i_entry_elevation]) + ",{:.3f}".format(da_total_wse[i_entry_elevation]))
                i_outprint_yes = 1

        # Work on the Regression Equations File
        if i_outprint_yes == 1 and len(s_output_curve_file)>0 and i_start_elevation_index>=0 and i_last_elevation_index>(i_start_elevation_index+1):
            # Not needed here, but [::-1] basically reverses the order of the array
            (d_t_a, d_t_b, d_t_R2) = linear_regression_power_function(da_total_q[i_start_elevation_index:i_last_elevation_index + 1], da_total_t[i_start_elevation_index:i_last_elevation_index + 1])
            (d_v_a, d_v_b, d_v_R2) = linear_regression_power_function(da_total_q[i_start_elevation_index:i_last_elevation_index + 1], da_total_v[i_start_elevation_index:i_last_elevation_index + 1])
            da_total_depth = da_total_wse - da_xs_profile1[0]
            (d_d_a, d_d_b, d_d_R2) = linear_regression_power_function(da_total_q[i_start_elevation_index:i_last_elevation_index + 1], da_total_depth[i_start_elevation_index:i_last_elevation_index + 1])
            # COMID,Row,Col,BaseElev,DEM_Elev,QMax,depth_a,depth_b,tw_a,tw_b,vel_a,vel_b
            o_curve_file.write('\n' + str(i_cell_comid) + ',' + str(int(i_row_cell - i_boundary_number)) + ',' + str(int(i_column_cell - i_boundary_number)) + ",{:.3f}".format(da_xs_profile1[0]) + ",{:.3f}".format(d_dem_low_point_elev) + ",{:.3f}".format(da_total_q[i_last_elevation_index]))
            o_curve_file.write(",{:.3f}".format(d_d_a) + ",{:.3f}".format(d_d_b) + ",{:.3f}".format(d_t_a) + ",{:.3f}".format(d_t_b) + ",{:.3f}".format(d_v_a) + ",{:.3f}".format(d_v_b) )

        # Output the XS information, if you've chosen to do so
        da_xs_profile1_str = array_to_string(da_xs_profile1[0:xs1_n])
        dm_manning_n_raster1_str = array_to_string(dm_manning_n_raster[ia_xc_r1_index_main[0:xs1_n], ia_xc_c1_index_main[0:xs1_n]]) 
        da_xs_profile2_str = array_to_string(da_xs_profile2[0:xs2_n]) 
        dm_manning_n_raster2 = array_to_string(dm_manning_n_raster[ia_xc_r2_index_main[0:xs2_n], ia_xc_c2_index_main[0:xs2_n]])
        if s_xs_output_file != '':
            o_xs_file.write(f"{i_cell_comid}\t{i_row_cell - i_boundary_number}\t{i_column_cell - i_boundary_number}\t{da_xs_profile1_str}\t{d_wse}\t{d_distance_z}\t{dm_manning_n_raster1_str}\t{da_xs_profile2_str}\t{d_wse}\t{d_distance_z}\t{dm_manning_n_raster2}\n")

   
    # Close the output file
    o_out_file.close()
    logging.info('Finished writing ' + str(s_output_vdt_database))
    
    # Close the output file
    if len(s_output_curve_file)>0:
        o_curve_file.close()
        logging.info('Finished writing ' + str(s_output_curve_file))
    
    
    # Write the output rasters
    if len(s_output_bathymetry_path) > 1:
        write_output_raster(s_output_bathymetry_path, dm_output_bathymetry[i_boundary_number:nrows + i_boundary_number, i_boundary_number:ncols + i_boundary_number], ncols, nrows, dem_geotransform, dem_projection, "GTiff", gdal.GDT_Float32)

    if len(s_output_flood) > 1:
        write_output_raster(s_output_flood, dm_out_flood[i_boundary_number:nrows + i_boundary_number, i_boundary_number:ncols + i_boundary_number], ncols, nrows, dem_geotransform, dem_projection, "GTiff", gdal.GDT_Int32)

    # Log the compute time
    d_sim_time = datetime.now() - starttime
    i_sim_time_s = int(d_sim_time.seconds)

    if i_sim_time_s < 60:
        logging.info('Simulation Took ' + str(i_sim_time_s) + ' seconds')
    else:
        logging.info('Simulation Took ' + str(int(i_sim_time_s / 60)) + ' minutes and ' + str(i_sim_time_s - (int(i_sim_time_s / 60) * 60)) + ' seconds')
        
if __name__ == "__main__":
    starttime = datetime.now()
    
    logging.info('Inputs to the Program is a Main Input File')
    logging.info('\nFor Example:')
    logging.info('  python Automated_Rating_Curve_Generator.py ARC_InputFiles/ARC_Input_File.txt')
    
    ### User-Defined Main Input File ###
    if len(sys.argv) > 1:
        MIF_Name = sys.argv[1]
        logging.info('Main Input File Given: ' + MIF_Name)
    else:
        #Read Main Input File
        MIF_Name = 'ARC_InputFiles/ARC_Input_File.txt'
        logging.warning('Moving forward with Default MIF Name: ' + MIF_Name)
        
    main(MIF_Name)