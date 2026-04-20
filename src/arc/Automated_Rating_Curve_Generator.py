
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
import geopandas as gpd
from scipy.optimize import curve_fit, OptimizeWarning, brentq
from scipy.signal import savgol_filter
from shapely.geometry import LineString, MultiLineString
from osgeo import gdal
from pyproj import CRS, Geod
from numba import njit, vectorize
from numba.core.errors import TypingError

from arc import LOG
from arc.cross_section import CrossSection, calculate_discharge_from_wse
from arc.hydraulic_data import HydraulicData

warnings.filterwarnings("ignore", category=OptimizeWarning)
gdal.UseExceptions()

# def geodesic_length_m(line_geom, dem_projection):
#     """Return geodesic length of a LineString in meters."""
#     crs = CRS.from_wkt(dem_projection)
#     geod = Geod(ellps=crs.name)
#     coords = list(line_geom.coords)
#     lons, lats = zip(*coords)
#     length = geod.line_length(lons, lats)  # meters
#     return length

def sample_line_for_valid_z(line: LineString, dm_elevation: np.ndarray, xy_to_rowcol, length_m, step_fraction=0.02):
    """
    Walk along a line until a valid DEM value is found.
    Returns elevation and distance along the line (meters).
    """
    nsteps = int(1 / step_fraction) + 1

    for i in range(nsteps):
        frac = i * step_fraction
        pt = line.interpolate(frac, normalized=True)
        rc = xy_to_rowcol(pt.x, pt.y)

        if rc is None:
            continue

        r, c = rc
        z = dm_elevation[r, c]

        if z > -9999:
            dist = frac * length_m
            return z, dist

    return np.nan, np.nan

def line_slope_from_dem(line_geom: LineString, dm_elevation: np.ndarray, dem_geotransform, length_m):
    """
    Compute slope along a line using a DEM that was read with read_raster_gdal.

    Parameters
    ----------
    line_geom : shapely.geometry.LineString or MultiLineString
        Stream/reach geometry in the same (lon/lat) CRS as the DEM.
    dm_elevation : np.ndarray
        DEM values from read_raster_gdal (2D array: [rows, cols]).
    dem_geotransform : tuple or list
        GDAL geotransform from read_raster_gdal.
    length_m : float
        Length of the line_geom in meters.

    Returns
    -------
    slope_pct : float
    slope_deg : float
    z_start   : float
    z_end     : float
    length_m  : float
    """

    # Handle None/empty
    if line_geom is None or line_geom.is_empty:
        return np.nan, np.nan, np.nan, np.nan, 0.0

    # Handle MultiLineString by choosing longest part
    if isinstance(line_geom, MultiLineString):
        if len(line_geom.geoms) == 0:
            return np.nan, np.nan, np.nan, np.nan, 0.0
        line_geom = max(line_geom.geoms, key=lambda g: g.length)

    if not isinstance(line_geom, LineString):
        try:
            line_geom = LineString(line_geom)
        except Exception:
            return np.nan, np.nan, np.nan, np.nan, 0.0

    coords = list(line_geom.coords)
    if len(coords) < 2:
        return np.nan, np.nan, np.nan, np.nan, 0.0

    # Start/end coordinates (lon, lat)
    coord_1 = coords[0]
    coord_2 = coords[-1]

    # --- helper: convert lon/lat → row/col using GDAL geotransform ---
    gt0, gt1, gt2, gt3, gt4, gt5 = dem_geotransform
    nrows, ncols = dm_elevation.shape

    def xy_to_rowcol(x, y):
        """
        Convert map coordinates (x,y) to DEM row/col indices.
        Assumes a north-up grid (gt2 == gt4 == 0).
        """
        # column: straightforward with positive pixel width
        col = int((x - gt0) / gt1)

        # row: geotransform[5] is typically negative for north-up rasters
        if gt5 < 0:
            row = int((gt3 - y) / abs(gt5))
        else:
            row = int((y - gt3) / gt5)

        # clip to DEM bounds; if outside, return None
        if row < 0 or row >= nrows or col < 0 or col >= ncols:
            return None
        return row, col

    rc1 = xy_to_rowcol(coord_1[0], coord_1[1])
    rc2 = xy_to_rowcol(coord_2[0], coord_2[1])

    if rc1 is None or rc2 is None:
        if rc1 is None and rc2 is None:
            return np.nan, np.nan, np.nan, np.nan, length_m

        z_start, dist_start = sample_line_for_valid_z(
            line_geom,
            dm_elevation,
            xy_to_rowcol
        )

        z_end, dist_from_end = sample_line_for_valid_z(
            LineString(list(line_geom.coords)[::-1]),
            dm_elevation,
            xy_to_rowcol
        )

        if np.isnan(z_start) or np.isnan(z_end):
            return np.nan, np.nan, z_start, z_end, length_m

        dist_end = length_m - dist_from_end
        length_m = abs(dist_end - dist_start)
    else:
        r1, c1 = rc1
        r2, c2 = rc2
        z_start = float(dm_elevation[r1, c1])
        z_end   = float(dm_elevation[r2, c2])

    if length_m == 0:
        return np.nan, np.nan, z_start, z_end, length_m

    rise = abs(z_end - z_start)  # meters
    slope_fraction = rise / length_m
    slope_pct = slope_fraction * 100.0
    slope_deg = math.degrees(math.atan(slope_fraction))

    return slope_pct, slope_deg, z_start, z_end, length_m


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
    o_dataset = gdal.Open(s_input_filename, gdal.GA_ReadOnly)
    if o_dataset is None:
        LOG.info('Cannot Open Raster ' + s_input_filename)
        raise FileNotFoundError(f"Cannot open raster {s_input_filename}")
    

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

    # Normalize south-up rasters (pixel height > 0) to north-up arrays.
    if l_geotransform[5] > 0:
        LOG.warning('Raster appears south-up (positive pixel height); flipping to north-up: ' + str(s_input_filename))
        dm_raster_array = np.flipud(dm_raster_array)
        geotransform = (
            l_geotransform[0],
            l_geotransform[1],
            l_geotransform[2],
            l_geotransform[3] + l_geotransform[5] * i_number_of_rows,
            l_geotransform[4],
            -l_geotransform[5],
        )

    # Extract information from the geotransform
    d_cell_size = l_geotransform[1]

    d_y_lower_left = l_geotransform[3] - i_number_of_rows * np.fabs(l_geotransform[5])
    d_y_upper_right = l_geotransform[3]
    d_x_lower_left = l_geotransform[0]
    d_x_upper_right = d_x_lower_left + i_number_of_columns * l_geotransform[1]
    dy = l_geotransform[5]
    maxx = d_x_lower_left + d_cell_size * i_number_of_columns
    miny = d_y_upper_right + dy * i_number_of_rows

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
    return dm_raster_array, i_number_of_columns, i_number_of_rows, d_cell_size, d_y_lower_left, d_y_upper_right, d_x_lower_left, d_x_upper_right, d_latitude, l_geotransform, s_raster_projection, maxx, miny, dy


def get_parameter_name(sl_lines: list[str], s_target: str, default_value: str = ''):
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
    d_return_value = default_value

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

def to_bool(val):
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        return val.strip().lower() in {"true", "1", "yes", "y"}
    return bool(val)

def read_main_input_file(s_mif_name: str, args: dict):
    """

    Parameters
    ----------
    s_mif_name: str
        Path to the input file
    args: dict
        Dictionary of arguments passed to the function.

    Returns
    -------
    dict        Dictionary of parameters to be used in the model

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

    s_stream_slope_method = get_parameter_name(sl_lines,  'Stream_Slope_Method')
    # path to the stream shapefile
    s_strmshp_path = get_parameter_name(sl_lines,  'StrmShp_File')
    if s_stream_slope_method == '':
        # Assume degree if not specified in the input efile
        s_stream_slope_method = 'local_average'
    if s_stream_slope_method == 'end_points' and s_strmshp_path == '':
            raise AttributeError('You need to specify the shapefile of stream lines if you plan to use the end_points slope method.')
        
    b_bathy_use_banks = to_bool(get_parameter_name(sl_lines, 'Bathy_Use_Banks', False))

    #Default is to find the banks of the river based on flat water in the DEM.  However, you can also find the banks using the water surface (please also set i_lc_water_value)
    b_FindBanksBasedOnLandCover = to_bool(
        get_parameter_name(sl_lines, 'FindBanksBasedOnLandCover', False)
    )

    # Find the True/False variable to use the bank elevations to calculate the depth of the bathymetry estimate. Has to be false if there is no curve file to be used.
    curve_file = get_parameter_name(sl_lines, 'Print_Curve_File')
    b_reach_average_curve_file = to_bool(
        get_parameter_name(sl_lines, 'Reach_Average_Curve_File', False)
    ) and curve_file

    params = {
        's_input_dem_path': get_parameter_name(sl_lines,  'DEM_File'), # Find the path to the DEM file
        's_stream_slope_method': s_stream_slope_method,
        's_strmshp_path'    : s_strmshp_path,
        's_input_stream_path': get_parameter_name(sl_lines,  'Stream_File'), # Find the path to the stream file
        's_input_land_use_path': get_parameter_name(sl_lines,  'LU_Raster_SameRes'), # Find the path to the land use raster file
        's_input_mannings_path': get_parameter_name(sl_lines,  'LU_Manning_n'), # Find the path to the mannings n file
        's_input_flow_file_path': get_parameter_name(sl_lines,  'Flow_File'), # Find the path to the flow file
        's_flow_file_id': get_parameter_name(sl_lines,  'Flow_File_ID'), # Find the column name 
        's_flow_file_baseflow': get_parameter_name(sl_lines,  'Flow_File_BF'), # Find the baseflow column name
        's_flow_file_qmax': get_parameter_name(sl_lines,  'Flow_File_QMax'), # Find the column name for the maximum flow
        'd_x_section_distance': float(get_parameter_name(sl_lines,  'X_Section_Dist', 5000.0)), # Find the x section distance
        's_output_vdt_database': get_parameter_name(sl_lines,  'Print_VDT_Database'), # Find the path to the output velocity, depth, and top width file
        's_output_ap_database': get_parameter_name(sl_lines,  'Print_AP_Database'), # Find the path to the output area and wetted perimeter file
        's_output_curve_file': curve_file, # Find the path to the output curve file
        'd_degree_manipulation': float(get_parameter_name(sl_lines,  'Degree_Manip', 1.1)), # Find the degree manipulation parameter
        'd_degree_interval': float(get_parameter_name(sl_lines,  'Degree_Interval', 1.0)), # Find the degree interval parameter
        'i_low_spot_range': int(get_parameter_name(sl_lines,  'Low_Spot_Range', 0)), # Find the low spot range parameter
        'i_general_direction_distance': int(get_parameter_name(sl_lines,  'Gen_Dir_Dist', 10)), # Find the general direction distance parameter
        'i_general_slope_distance': int(get_parameter_name(sl_lines,  'Gen_Slope_Dist', 0)), # Find the general slope distance parameter
        'd_bathymetry_trapzoid_height': float(get_parameter_name(sl_lines,  'Bathy_Trap_H', 0.2)), # Find the bathymetry trapezoid height parameter,
        'b_bathy_use_banks': b_bathy_use_banks, # Find the true/false variable to use the bank elevations to calculate the depth of the bathymetry estimate
        's_output_bathymetry_path': get_parameter_name(sl_lines,  'AROutBATHY', get_parameter_name(sl_lines,  'BATHY_Out_File')), # Find the path to the output bathymetry file
        's_output_flood': get_parameter_name(sl_lines,  'AROutFLOOD'), # Find the path to the output flood file
        's_xs_output_file': get_parameter_name(sl_lines,  'XS_Out_File'), # Find the path to the output cross-section file (JLG added this to recalculate top-width and velocity)
        'i_lc_water_value': int(get_parameter_name(sl_lines,  'LC_Water_Value', 80)), # Find the value in the land cover dataset that corresponds to water. This is used to find the banks of the river if b_FindBanksBasedOnLandCover is set to True
        'i_number_of_increments': int(get_parameter_name(sl_lines,  'VDT_Database_NumIterations', 15)), # Find the number of increments to use in the velocity, depth, and top width database
        'b_FindBanksBasedOnLandCover': b_FindBanksBasedOnLandCover, # Find the true/false variable to find the banks of the river based on the land cover dataset instead of the DEM
        'b_reach_average_curve_file': b_reach_average_curve_file # Find the true/false variable to use a reach-average curve file
    }

    return params

def convert_cell_size(
    d_dem_cell_size_x: float,
    d_dem_cell_size_y: float,
    d_dem_lower_left: float,
    d_dem_upper_right: float,
    s_dem_projection: str
):
    """
    Converts DEM cell size to x/y resolution in meters.

    For geographic rasters (degrees), this uses pyproj geodesic distances
    on the DEM ellipsoid. For projected rasters, it returns the original
    map-unit cell size for x and y.

    Parameters
    ----------
    d_dem_cell_size_x: float
        DEM x cell size (degrees for geographic rasters; map units otherwise)
    d_dem_cell_size_y: float
        DEM y cell size (degrees for geographic rasters; map units otherwise)
    d_dem_lower_left: float
        Lower-left y value (latitude for geographic rasters)
    d_dem_upper_right: float
        Upper-right y value (latitude for geographic rasters)
    s_dem_projection: str
        DEM projection WKT/CRS definition

    Returns
    -------
    d_x_cell_size: float
        Resolution of the cells in x direction (meters for geographic rasters)
    d_y_cell_size: float
        Resolution of the cells in y direction (meters for geographic rasters)
    d_projection_conversion_factor: float
        Mean meters-per-degree factor used for conversion

    """

    # Default output for projected/non-geographic rasters
    d_dem_cell_size_x = np.fabs(d_dem_cell_size_x)
    d_dem_cell_size_y = np.fabs(d_dem_cell_size_y)
    d_x_cell_size = d_dem_cell_size_x
    d_y_cell_size = d_dem_cell_size_y
    d_projection_conversion_factor = 1

    # Parse DEM CRS and use geodesic conversion for geographic grids.
    try:
        o_crs = CRS.from_user_input(s_dem_projection)
    except Exception as e:
        raise ValueError("Unable to parse DEM projection for cell-size conversion.") from e

    if o_crs.is_geographic:
        d_lat = (d_dem_lower_left + d_dem_upper_right) / 2.0
        d_lon = 0.0  # Geodesic spacing at a reference longitude

        # Build a geodesic calculator from the DEM ellipsoid.
        o_ellps = o_crs.ellipsoid
        if o_ellps is not None and o_ellps.semi_major_metre and o_ellps.inverse_flattening:
            o_geod = Geod(a=o_ellps.semi_major_metre, rf=o_ellps.inverse_flattening)
        else:
            o_geod = Geod(ellps="WGS84")

        # North-south cell spacing (meters)
        _, _, d_y_cell_size = o_geod.inv(d_lon, d_lat, d_lon, d_lat + d_dem_cell_size_y)
        # East-west cell spacing (meters) at midpoint latitude
        _, _, d_x_cell_size = o_geod.inv(d_lon, d_lat, d_lon + d_dem_cell_size_x, d_lat)

        d_x_cell_size = np.fabs(d_x_cell_size)
        d_y_cell_size = np.fabs(d_y_cell_size)
        d_projection_conversion_factor = 0.5 * (
            (d_x_cell_size / max(d_dem_cell_size_x, 1e-12))
            + (d_y_cell_size / max(d_dem_cell_size_y, 1e-12))
        )
    # if the raster is projected, we assume the cell size is already in meters and use it directly
    elif o_crs.is_projected:
        # For projected rasters, x/y map units are already meters based on CRS checks in main().
        d_x_cell_size = d_dem_cell_size_x
        d_y_cell_size = d_dem_cell_size_y
        d_projection_conversion_factor = 1.0


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

@vectorize(target='cpu', cache=True)
def round_sig(x, sig=3):
    if x == 0.0:
        return 0.0
    if not np.isfinite(x):
        return x
    exp = int(math.floor(math.log10(abs(x))))
    factor = 10.0 ** (sig - 1 - exp)
    return math.floor(x * factor + 0.5) / factor

@njit(cache=True)
def get_reach_median_stream_slope_information(stream_id: int, dm_dem: np.ndarray, im_streams: np.ndarray, d_dx: float, d_dy: float, i_general_slope_distance: int):
    """
    Calculates the stream slope for each stream cell using the following process:

        1.) Find all stream cells that have the same stream id value
        2.) Look at the slope of each of the stream cells.
        3.) Average the slopes to get the overall slope we use in the model.

    Guaranteed to be >= 0.0002 and <= 0.03

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
    i_general_slope_distance: int
        Distance in number of cells to look for slope calculations.

    Returns
    -------
    d_stream_slope: float
        Average slope from the stream cells in the specified search box
    d_stream_slope_25: float
        25th percentile slope from the stream cells in the specified search box
    d_stream_slope_75: float
        75th percentile slope from the stream cells in the specified search box

    """

    # Initialize a default stream flow
    d_stream_slope = 0.0

    # All cells in this reach (global indices)
    reach_rows, reach_cols = np.where(im_streams == stream_id)
    n = len(reach_rows)


    d_stream_slope = 0.0002
    lower_bound = 0.0002
    upper_bound = 0.0002

    if n < 2:
        # Not enough cells to define a slope
        return d_stream_slope, lower_bound, upper_bound

    total_slope = 0.0
    count = 0

    slope_list = []

    # Loop over all unique pairs (a, b), a < b
    for a in range(n):
        ra = reach_rows[a]
        ca = reach_cols[a]
        za = dm_dem[ra, ca]

        for b in range(a + 1, n):
            rb = reach_rows[b]
            cb = reach_cols[b]

            # Check if within the "box" in row/col space
            dr = rb - ra
            dc = cb - ca

            if (dr >= -i_general_slope_distance and dr <= i_general_slope_distance and
                dc >= -i_general_slope_distance and dc <= i_general_slope_distance):

                zb = dm_dem[rb, cb]

                # Horizontal distance
                dx = dc * d_dx
                dy = dr * d_dy
                dist = math.sqrt(dx * dx + dy * dy)

                if dist > 0.0:
                    slope = round(abs(za - zb) / dist, 8)
                    if slope > 0.0:
                        total_slope += slope
                        count += 1
                        slope_list.append(slope)

    # remove any outliers using quartiles
    if len(slope_list) > 0:
        slope_arr = np.array(slope_list)
        slope_arr = round_sig(slope_arr, 8)   
        Q1 = round(np.percentile(slope_arr, 25), 8)
        Q3 = round(np.percentile(slope_arr, 75), 8)
        IQR = Q3 - Q1
        lower_bound = Q1
        upper_bound = Q3
        slope_list = [x for x in slope_list if lower_bound <= x <= upper_bound]

    # Compute median slope
    if len(slope_list) > 0:
        d_stream_slope = np.median(np.array(slope_list))


    return d_stream_slope, lower_bound, upper_bound

@njit(cache=True)
def get_local_average_stream_slope_information(i_row: int, i_column: int, dm_dem: np.ndarray, im_streams: np.ndarray, d_dx: float, d_dy: float, i_general_slope_distance: int):
    """
    Calculates the stream slope using the following process:

        1.) Find all stream cells within the Gen_Slope_Dist that have the same stream id value
        2.) Look at the slope of each of the stream cells.
        3.) Average the slopes to get the overall slope we use in the model.

    Guaranteed to be >= 0.0002 and <= 0.03

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
    i_general_slope_distance: int
        Distance in number of cells to look for slope calculations.
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
    
    # # if slope is less than the threshold, reset it
    # if d_stream_slope < 0.0002:
    #     d_stream_slope = 0.0002
    # # if slope is greater than the threshold, reset it
    # elif d_stream_slope > 0.03:
    #     d_stream_slope = 0.03

    # Return the slope to the calling function
    return d_stream_slope

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
def get_stream_direction_information(i_row: int, i_column: int, im_streams: np.ndarray, i_general_direction_distance: int):
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
    i_general_direction_distance: int
        Distance to search for stream cells

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

def find_wse(range_end, start_wse, increment, d_q_maximum, x_section: CrossSection, d_slope_use):
    d_q_sum = 0.0
    sqrt_slope = d_slope_use**0.5

    low = 0
    high = range_end

    # Let us try the maximum depth increment first. If it cannot give us an answer, return
    wse_high = start_wse + high * increment
    d_q_sum_high = calculate_discharge_from_wse(wse_high, sqrt_slope, *x_section.get_calculate_discharge_from_wse_args())

    if d_q_sum_high < d_q_maximum:
        return wse_high, d_q_sum_high
    
    # Use bisection algorithm to find the water surface elevation that corresponds to the target discharge
    while high - low > 1:
        mid = (low + high) // 2
        wse = start_wse + mid * increment
        d_q_sum = calculate_discharge_from_wse(wse, sqrt_slope, *x_section.get_calculate_discharge_from_wse_args())

        if d_q_sum < d_q_maximum:
            low = mid
        else:
            high = mid

    d_wse = 0.0
    prev_wse = 0.0
    prev_q = 0.0
    can_interpolate = False
    for i_depthincrement in range(low, high + 1):
        d_wse = start_wse + i_depthincrement * increment
        d_q_sum = calculate_discharge_from_wse(d_wse, sqrt_slope, *x_section.get_calculate_discharge_from_wse_args())

        # Check for overshoot in discharge
        if d_q_sum == d_q_maximum:
            break
        elif d_q_sum > d_q_maximum:
            # If overshoot occurs at the very first increment, interpolation cannot be done
            if can_interpolate:
                # Linear interpolation between previous and current values:
                # interp_wse = prev_wse + (target_q - prev_q) * (d_wse - prev_wse) / (d_q_sum - prev_q)
                interp_wse = prev_wse + (d_q_maximum - prev_q) * (d_wse - prev_wse) / (d_q_sum - prev_q)
                # Recalculate geometry and discharge at the interpolated water surface elevation
                d_q_sum = calculate_discharge_from_wse(interp_wse, sqrt_slope, *x_section.get_calculate_discharge_from_wse_args())
                d_wse = interp_wse
            break

        # Save current values for the next iteration
        prev_wse = d_wse
        prev_q = d_q_sum
        can_interpolate = True

    return d_wse, d_q_sum

def flood_increments(i_number_of_increments, d_inc_y, x_section: CrossSection, d_slope_use, d_q_sum, hydraulic_data: HydraulicData):

    i_start_elevation_index, i_last_elevation_index = 0, 0

    # Initialize previous values
    prev_t = 0.0
    prev_a = 0.0
    prev_p = 0.0
    prev_q = 0.0
    prev_v = 0.0
    prev_wse = 0.0

    for i_entry_elevation in range(i_number_of_increments):
        d_wse = x_section.get_thalweg() + d_inc_y * i_entry_elevation
        d_wse = np.round(d_wse, 3)

        # Calculate the geometry          
        A1, P1, np1, T1 = x_section.calculate_stream_geometry_and_topwidth_side_1(d_wse)
        A2, P2, np2, T2 = x_section.calculate_stream_geometry_and_topwidth_side_2(d_wse)

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
                d_wse_upper_bound = x_section.get_thalweg() + d_inc_y * (i_entry_elevation + 1)
                d_wse_upper_bound = np.round(d_wse_upper_bound, 3)
                while d_wse_lower_bound < d_wse_upper_bound:
                    # Calculate the geometry          
                    A1, P1, np1, T1 = x_section.calculate_stream_geometry_and_topwidth_side_1(d_wse_lower_bound)
                    A2, P2, np2, T2 = x_section.calculate_stream_geometry_and_topwidth_side_2(d_wse_lower_bound)

                    T = T1 + T2
                    A = A1 + A2
                    P = P1 + P2

                    # Estimate mannings n
                    d_composite_n = np.round(((np1 + np2) / P)**(2 / 3), 4)

                    # use a local candidate to avoid reusing stale Q in the while condition
                    Q_cand = (1.0 / d_composite_n) * A * (A / P) ** (2.0 / 3.0) * d_slope_use ** 0.5
                    V_cand = Q_cand / A

                    # accept only if it improves AND respects the cap
                    if (A > prev_a) and (P > prev_p) and (Q_cand > prev_q) and (Q_cand <= d_q_sum):
                        d_wse = d_wse_lower_bound
                        Q = Q_cand
                        V = V_cand
                        break

                    d_wse_lower_bound += 0.01
                        
            # if we reach the upper bound without a valid candidate, or we overshot, revert
            # also add a top‑level guard before saving the initial (non‑refined) Q
            # right after computing the first Q/V for this increment:
            if (Q <= prev_q) or (Q > d_q_sum) or Q > d_q_sum:
                hydraulic_data.add_hydraulic_data(i_entry_elevation, prev_wse, prev_t, prev_a, prev_p, prev_q, prev_v)
                continue

            # Save the values
            hydraulic_data.add_hydraulic_data(i_entry_elevation, d_wse, T, A, P, Q, V)

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
            i_start_elevation_index = i_entry_elevation

    return i_start_elevation_index, i_last_elevation_index

def add_100_if_elevation_less_than_0(arr):
    """
    Checks and modifies the DEM if there are negative elevations in it by adding 100 to all elevations.
    """
    # Check if the array contains any negative value
    b_modified_dem = False
    if np.any(arr < 0):
        # Add 100 to the entire array
        arr += 100
        b_modified_dem = True

    return arr, b_modified_dem

def create_reach_average_slope_dicts(dm_stream, dm_elevation, dx, dy, quiet, i_general_slope_distance):
    # create a list of unique stream IDs to loop through
    unique_stream_ids = np.unique(dm_stream)
    unique_stream_ids = unique_stream_ids[unique_stream_ids > 0]
    pbar_slopes = tqdm.tqdm(unique_stream_ids, disable=quiet)
    dict_stream_slopes = {}
    dict_stream_slopes_25th = {}
    dict_stream_slopes_75th = {}
    for stream_id in pbar_slopes:
        reach_slope, reach_slope_25th, reach_slope_75th = get_reach_median_stream_slope_information(stream_id, dm_elevation, dm_stream, dx, dy, i_general_slope_distance)
        dict_stream_slopes[stream_id] = reach_slope
        dict_stream_slopes_25th[stream_id] = reach_slope_25th
        dict_stream_slopes_75th[stream_id] = reach_slope_75th

    return dict_stream_slopes, dict_stream_slopes_25th, dict_stream_slopes_75th

def dict_stream_slopes_from_endpoints(dm_stream, dm_elevation, dem_geotransform, dem_projection, s_strmshp_path, s_flow_file_id, quiet):
    # create a list of unique stream IDs to loop through
    unique_stream_ids = np.unique(dm_stream)
    unique_stream_ids = unique_stream_ids[unique_stream_ids > 0]
    # Load line shapefile
    gdf_StrmSHP = gpd.read_file(s_strmshp_path)
    pbar_slopes = tqdm.tqdm(unique_stream_ids, disable=quiet)
    dict_stream_slopes = {}
    for stream_id in pbar_slopes:
        gdf_StrmSHP_filtered: gpd.GeoDataFrame = gdf_StrmSHP[gdf_StrmSHP[s_flow_file_id]==stream_id]
        utm_crs = gdf_StrmSHP_filtered.estimate_utm_crs()
        gdf_utm = gdf_StrmSHP_filtered.to_crs(utm_crs)
        StrmSHP_geom = gdf_StrmSHP_filtered.to_crs(dem_projection).geometry
        length_m = float(gdf_utm.length.iloc[0])
        slope_pct, slope_deg, z_start, z_end, length_m = line_slope_from_dem(StrmSHP_geom.iloc[0], dm_elevation, dem_geotransform, length_m)
        dict_stream_slopes[stream_id] = round(slope_pct/100, 8)

    return dict_stream_slopes

def objective_with_wse(trial_wse: float, x_section: CrossSection, slope_squared: float,
                       d_q_maximum: float) -> float:
    # Define an objective function: the difference between the calculated max flow and d_q_maximum.
    trial_wse = np.round(trial_wse, 3)

    trial_d_q_sum = calculate_discharge_from_wse(trial_wse, slope_squared, *x_section.get_calculate_discharge_from_wse_args())

    # trial_d_q_sum = round(trial_d_q_sum, 3)
    difference = trial_d_q_sum - d_q_maximum

    # The objective is zero when trial_d_q_sum equals d_q_maximum.
    return difference


# Define an objective function: the difference between the calculated max flow and d_q_maximum.
# @njit(cache=True)
def objective_with_slope(trial_slope: float,
                         d_maxflow_wse_initial: float, d_depth_increment_small: float, d_q_maximum: float,
                         x_section: CrossSection) -> float:
    # find_wse returns a tuple: (d_maxflow_wse_final, d_q_sum)
    _, trial_d_q_sum = find_wse(
        2501, 
        d_maxflow_wse_initial, 
        d_depth_increment_small, 
        d_q_maximum, 
        x_section,
        trial_slope
    )
    # The objective is zero when trial_d_q_sum equals d_q_maximum.
    return trial_d_q_sum - d_q_maximum

# @profile
def main(MIF_Name: str, args: dict, quiet: bool):
    starttime = datetime.now()  
    ### Read Main Input File ###
    params = read_main_input_file(MIF_Name, args)
    
    ### Read the Flow Information ###
    COMID, QBaseFlow, QMax = read_flow_file(params['s_input_flow_file_path'], params['s_flow_file_id'], params['s_flow_file_baseflow'], params['s_flow_file_qmax'])

    ### Read Raster Data ###
    dm_elevation, dncols, dnrows, dcellsize, dyll, dyur, dxll, dxur, dlat, dem_geotransform, dem_projection, dem_maxx, dem_miny, dem_dy = read_raster_gdal(params['s_input_dem_path'])
    dm_stream, sncols, snrows, scellsize, syll, syur, sxll, sxur, slat, strm_geotransform, strm_projection, maxx, miny, dy = read_raster_gdal(params['s_input_stream_path'])
    dm_land_use, lncols, lnrows, lcellsize, lyll, lyur, lxll, lxur, llat, land_geotransform, land_projection, maxx, miny, dy = read_raster_gdal(params['s_input_land_use_path'])

    ### Determine if the rasters are in a projected coordinate system (units in meters) or geographic coordinate system (units in degrees)
    if 'PROJCS' in dem_projection:
        LOG.info('Rasters are in a projected coordinate system with units in meters.')
        # set a flag to indicate that the rasters are in a projected coordinate system
        b_projected = True
    elif 'GEOGCS' in dem_projection:
        LOG.info('Rasters are in a geographic coordinate system with units in degrees.')
        # set a flag to indicate that the rasters are in a geographic coordinate system
        b_projected = False

    ### if the DEM contains negative values, add 100 m to the height to get rid of the negatives, we'll subtract it back out later
    dm_elevation, b_modified_dem = add_100_if_elevation_less_than_0(dm_elevation)

    ### make sure the rasters are all the same size and aligned and if not, end with log an error message and stop processing
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

    ### check the coordinate system of the rasters and if they are not in meters or degrees, end with log an error message and stop processing
    unit_aliases = {
        'meter': 'meter',
        'meters': 'meter',
        'metre': 'meter',
        'metres': 'meter',
        'degree': 'degree',
        'degrees': 'degree'
    }
    raster_projections = {
        'DEM': dem_projection,
        'STREAM': strm_projection,
        'LAND_USE': land_projection
    }
    for raster_name, raster_projection in raster_projections.items():
        try:
            raster_crs = CRS.from_wkt(raster_projection)
        except Exception as ex:
            LOG.error(f'Unable to parse CRS for {raster_name} raster: {ex}')
            return

        axis_units = {(axis.unit_name or '').strip().lower() for axis in raster_crs.axis_info if axis is not None}
        axis_units.discard('')
        if not axis_units:
            LOG.error(f'Unable to determine CRS units for {raster_name} raster.')
            return

        invalid_units = [u for u in sorted(axis_units) if unit_aliases.get(u) not in {'meter', 'degree'}]
        if invalid_units:
            LOG.error(f'{raster_name} raster CRS units are not meters or degrees: {", ".join(invalid_units)}')
            return


    ### Imbed the Stream and DEM data within a larger Raster to help with the boundary issues. ###
    i_general_direction_distance = params['i_general_direction_distance']
    i_general_slope_distance = params['i_general_slope_distance']
    i_boundary_number = max(1, i_general_slope_distance, i_general_direction_distance)

    dm_stream = np.pad(dm_stream, i_boundary_number, mode='constant', constant_values=0).astype(np.int64)

    dm_elevation = np.pad(dm_elevation, i_boundary_number, mode='constant', constant_values=0)

    dm_land_use = np.pad(dm_land_use, i_boundary_number, mode='constant', constant_values=0).astype(float)
    

    ##### Begin Calculations #####
    # Create working matrices
    i_number_of_increments = params['i_number_of_increments']
    hydraulic_data = HydraulicData(params, b_modified_dem)

    # Create output rasters
    # Create an array with NaN values instead of zeros
    dm_output_bathymetry = np.full(
        (nrows + i_boundary_number * 2, ncols + i_boundary_number * 2), 
        np.nan, 
        dtype=np.float32
    )
    
    s_output_flood = params['s_output_flood']
    if s_output_flood:
        dm_out_flood = np.zeros((nrows + i_boundary_number * 2, ncols + i_boundary_number * 2)).astype(int)

    # Get the list of stream locations
    ia_valued_row_indices, ia_valued_column_indices = np.where(np.isin(dm_stream, COMID, kind='table'))

    i_number_of_stream_cells = len(ia_valued_row_indices)
    
    # Make all Land Cover that is a stream look like water
    i_lc_water_value = params['i_lc_water_value']
    dm_land_use[ia_valued_row_indices,ia_valued_column_indices] = i_lc_water_value
    
    
    #Assing Manning n Values
    ### Read in the Manning Table ###
    dm_manning_n_raster = np.copy(dm_land_use)
    dm_manning_n_raster = read_manning_table(params['s_input_mannings_path'], dm_manning_n_raster)

    # Correct the mannings values here
    dm_manning_n_raster[dm_manning_n_raster > 10] = 0.035
    dm_manning_n_raster[dm_manning_n_raster <= 0.0] = 0.005
    

    # Get the cell dx and dy coordinates
    dx, dy, dproject = convert_cell_size(dcellsize, dem_dy, dyll, dyur, dem_projection)
    LOG.info('Cellsize X = ' + str(dx))
    LOG.info('Cellsize Y = ' + str(dy))

    # Create cross section, with precomputed angles
    i_precompute_angles = 30
    d_precompute_angles = np.pi / i_precompute_angles
    x_section = CrossSection(dx, dy, i_precompute_angles, d_precompute_angles, dm_elevation, dm_land_use, params["d_x_section_distance"], params["b_FindBanksBasedOnLandCover"], params["i_lc_water_value"], params["d_bathymetry_trapzoid_height"], params["b_bathy_use_banks"])
    hydraulic_data.associate_with_cross_section(x_section)

    # Find all the different angle increments to test
    l_angles_to_test = [0.0]
    d_increments = 0
    d_degree_manipulation = params['d_degree_manipulation']
    d_degree_interval = params['d_degree_interval']
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
    x_section.set_boundary_extents(i_boundary_number, nrows, ncols)

    # Write the percentiles into the files
    LOG.info('Looking at ' + str(i_number_of_stream_cells) + ' stream cells')

    # create a reach average slope before we go stream cell by stream cell
    s_stream_slope_method = params['s_stream_slope_method']
    if s_stream_slope_method == 'reach_average' or s_stream_slope_method == 'local_average_corrected':
        dict_stream_slopes, dict_stream_slopes_25th, dict_stream_slopes_75th = create_reach_average_slope_dicts(dm_stream, dm_elevation, dx, dy, quiet, i_general_slope_distance)
    elif s_stream_slope_method == 'end_points':
        dict_stream_slopes = dict_stream_slopes_from_endpoints(dm_stream, dm_elevation, dem_geotransform, dem_projection, params['s_strmshp_path'], params['s_flow_file_id'], quiet)

    # Extract some parameters
    b_bathy_use_banks = params['b_bathy_use_banks']
    s_output_bathymetry_path = params['s_output_bathymetry_path']
    b_reach_average_curve_file = params['b_reach_average_curve_file']

    d_depth_increment_big = 0.5
    d_depth_increment_med = 0.05
    d_depth_increment_small = 0.01
    ### Begin the stream cell solution loop ###
    pbar = tqdm.tqdm(range(i_number_of_stream_cells), total=i_number_of_stream_cells, disable=quiet)
    for i_entry_cell in pbar:

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
        d_stream_direction, d_xs_direction = get_stream_direction_information(i_row_cell, i_column_cell, dm_stream, i_general_direction_distance)

        # Get the Slope of each Stream Cell. Slope should be in m/m
        if s_stream_slope_method == 'local_average':
            d_slope_use = get_local_average_stream_slope_information(i_row_cell, i_column_cell, dm_elevation, dm_stream, dx, dy, i_general_slope_distance)
        elif s_stream_slope_method =='reach_average' or s_stream_slope_method == 'end_points':
            d_slope_use = dict_stream_slopes[i_cell_comid]
        elif s_stream_slope_method == 'local_average_corrected':
            d_slope_use = get_local_average_stream_slope_information(i_row_cell, i_column_cell, dm_elevation, dm_stream, dx, dy, i_general_slope_distance)
            d_slope_25th = dict_stream_slopes_25th[i_cell_comid]
            d_slope_75th = dict_stream_slopes_75th[i_cell_comid]
            # if the corrected slope is less than the streams 25th percentile slope, use the 25th percentile slope
            if d_slope_use < d_slope_25th:
                d_slope_use = d_slope_25th
            # if the corrected slope is greater than the streams 75th percentile slope, use the 75th percentile slope
            elif d_slope_use > d_slope_75th:
                d_slope_use = d_slope_75th  
        else: 
            #Default to using the 'local_average' method
            d_slope_use = get_local_average_stream_slope_information(i_row_cell, i_column_cell, dm_elevation, dm_stream, dx, dy, i_general_slope_distance)

        # Now Pull the Cross-Section again with the new angle
        if d_xs_direction > np.pi:
            i_precompute_angle_closest = int(round((d_xs_direction-np.pi) / d_precompute_angles))
        else:
            i_precompute_angle_closest = int(round(d_xs_direction / d_precompute_angles))

        x_section.set_cross_section(i_row_cell, i_column_cell, i_precompute_angle_closest, d_xs_direction)
        
        # Adjust to the lowest-point in the Cross-Section
        i_low_spot_range = params['i_low_spot_range']
        if i_low_spot_range > 0:
            x_section.adjust_cross_section_to_lowest_point(i_low_spot_range)
            # The r and c for the stream cell is adjusted because it may have moved
            i_row_cell, i_column_cell = x_section.get_row_col()
        
        d_dem_low_point_elev = x_section.get_thalweg()

        # Adjust cross-section angle to ensure shortest top-width at a specified depth
        if d_increments > 0:
            d_xs_direction = x_section.get_best_xsection_angle(d_precompute_angles, l_angles_to_test)

            # Now Pull the Cross-Section again with the new angle
            if d_xs_direction > np.pi:
                i_precompute_angle_closest = int(round((d_xs_direction-np.pi) / d_precompute_angles))
            else:
                i_precompute_angle_closest = int(round(d_xs_direction / d_precompute_angles))

            x_section.set_cross_section(i_row_cell, i_column_cell, i_precompute_angle_closest, d_xs_direction)

        # Burn bathymetry profile into cross-section profile
        # "Be the banks for your river" - Needtobreathe
                
        # If you don't have a cross-section, skip it and fill in empty values for the reach average processing
        if not x_section.is_valid():
            hydraulic_data.add_empty_x_section_for_curve_file(i_cell_comid, d_q_maximum, d_slope_use)
            continue

        #BATHYMETRY CALCULATION
        #This method calculates bathymetry based on the water surface elevation or LandCover ("FindBanksBasedOnLandCover" and "LC_Water_Value").
        if not b_bathy_use_banks and s_output_bathymetry_path != '':
            x_section.Calculate_Bathymetry_Based_on_WSE_or_LC(d_q_baseflow, d_slope_use, dm_output_bathymetry)
        #This method calculates the banks based on the Riverbank
        elif b_bathy_use_banks and s_output_bathymetry_path != '':
            x_section.Calculate_Bathymetry_Based_on_RiverBank_Elevations(d_q_baseflow, d_slope_use, dm_output_bathymetry)

        # Calculate the volumes
        # VolumeFillApproach 1 is to find the height within ElevList_mm that corresponds to the Qmax flow.  THen increment depths to have a standard number of depths to get to Qmax.  
        # This is preferred for VDTDatabase method.
        
        #This is the Stream Cell Location
        if s_output_flood:
            dm_out_flood[i_row_cell,i_column_cell] = 3
        
        # Set output arrays to zero
        hydraulic_data.reset_hydraulic_data()
        
        # This just tells the curve file whether to print out a result or not.  If no realistic depths were calculated, no reason to output results.
        b_outprint_yes = False
        
        # This is the first and last indice of elevations we'll need for the Curve Fitting for this cell
        i_start_elevation_index = -1
        i_last_elevation_index = 0
        
        
        # Here are the n values for each side of the cross-section
        x_section.set_mannings_n_values(dm_manning_n_raster)

        # space between ordinates in the cross-section
        d_ordinate_dist = x_section.d_ordinate_dist

        # we'll assume the results are acceptable until we think otherwise
        acceptable = True

        # This is the bottom of the channel
        d_maxflow_wse_initial = x_section.get_thalweg()

        # set this as the default in case we don't find a better one
        d_maxflow_wse_final = -999.0

        # initialize some variables
        d_q_sum = 0.0
        slope_use_squared = d_slope_use ** 0.5

        wse_lower = d_maxflow_wse_initial + 0.01
        wse_upper = d_maxflow_wse_initial + 24.99
        wse_obj_args = (x_section, slope_use_squared, d_q_maximum)

        # Check if the objective function changes sign between the bounds.
        f_lower = objective_with_wse(wse_lower, *wse_obj_args)
        f_upper = objective_with_wse(wse_upper, *wse_obj_args)

        if safe_signs_differ(f_lower, f_upper):
            # The signs differ, so we have a valid bracket.
            # For 3 decimal places, xtol only needs to be 0.001
            d_maxflow_wse_final = np.round(brentq(objective_with_wse, wse_lower, wse_upper, xtol=0.001, args=wse_obj_args), 3)
            d_q_sum = calculate_discharge_from_wse(d_maxflow_wse_final, slope_use_squared, *x_section.get_calculate_discharge_from_wse_args())
        elif np.round(f_lower, 5) == 0 or np.round(f_upper, 5) == 0:          
            # if the f_lower or f_upper is equal to zero, it's probably close enough to be the WSE we are looking for, so we'll use it
            d_maxflow_wse_final = np.round(wse_lower, 3) if np.round(f_lower, 5) == 0 else np.round(wse_upper, 3)
            d_q_sum = calculate_discharge_from_wse(d_maxflow_wse_final, slope_use_squared, *x_section.get_calculate_discharge_from_wse_args())

        # Let's see if the volume-fill approach gave us a better answer and use that if it did
        # To find the depth / wse where the maximum flow occurs we use two sets of incremental depths.  The first is 0.5m followed by 0.05m
        d_maxflow_wse_initial, d_q_sum_test = find_wse(101, d_maxflow_wse_initial, d_depth_increment_big, d_q_maximum, x_section, d_slope_use)


        # Based on using depth increments of 0.5, now lets fine-tune the wse using depth increments of 0.05
        d_maxflow_wse_initial = max(d_maxflow_wse_initial - 0.5, x_section.get_thalweg())
        d_maxflow_wse_med = d_maxflow_wse_initial
        d_maxflow_wse_med, d_q_sum_test = find_wse(101, d_maxflow_wse_med, d_depth_increment_med, d_q_maximum, x_section, d_slope_use)

        # Based on using depth increments of 0.05, now lets fine-tune the wse even more using depth increments of 0.01
        d_maxflow_wse_med = max(d_maxflow_wse_med - 0.05, x_section.get_thalweg())
        d_maxflow_wse_final_test = d_maxflow_wse_med
        d_maxflow_wse_final_test, d_q_sum_test = find_wse(2501, d_maxflow_wse_med, d_depth_increment_small, d_q_maximum, x_section, d_slope_use)

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

        # if slope is greater than the threshold, let's change it to the threshold
        if slope_upper > 0.03:
            slope_upper = 0.03

        slope_obj_args = (d_maxflow_wse_initial, d_depth_increment_small, d_q_maximum, x_section)
        # Check if the objective function changes sign between the bounds.
        f_lower = objective_with_slope(slope_lower, *slope_obj_args)
        f_upper = objective_with_slope(slope_upper, *slope_obj_args)
        if safe_signs_differ(f_lower, f_upper):
            # The signs differ, so we have a valid bracket.
            # Needs xtol of 0.0001 to get to 3 decimal places
            trial_slope_use = brentq(objective_with_slope, slope_lower, slope_upper, xtol=0.0001, args=slope_obj_args)
            trial_slope_use = np.round(trial_slope_use, 3)
            # Optionally, recompute d_maxflow_wse_final and d_q_sum with the new slope:
            d_maxflow_wse_final_test, d_q_sum_test = find_wse(
                2501, 
                d_maxflow_wse_initial, 
                d_depth_increment_small, 
                d_q_maximum, 
                x_section,
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
                x_section,
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

            # if slope is greater than the threshold, let's change it to the threshold
            if slope_upper > 0.03:
                slope_upper = 0.03

            # Check if the objective function changes sign between the bounds.
            f_lower = objective_with_slope(slope_lower, *slope_obj_args)
            f_upper = objective_with_slope(slope_upper, *slope_obj_args)


            if safe_signs_differ(f_lower, f_upper):
                # The signs differ, so we have a valid bracket.
                new_slope = brentq(objective_with_slope, slope_lower, slope_upper, xtol=0.0001, args=slope_obj_args)
                trial_slope_use = new_slope
                # Optionally, recompute d_maxflow_wse_final and d_q_sum with the new slope:
                d_maxflow_wse_final_test, d_q_sum_test = find_wse(
                    2501, 
                    d_maxflow_wse_initial, 
                    d_depth_increment_small, 
                    d_q_maximum, 
                    x_section,
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
                    x_section,
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
                    x_section,
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
        # 20250808 Joseph changeed this
        if d_q_sum > d_q_maximum * 1.5 or d_q_sum < d_q_maximum * 0.5:

            # something isn't good with our results
            acceptable = False

            # here we will see if we can get a better answer with a revised slope
            # from our Missouri study, relative DEM error was around 0.70, so dividing that by our d_distance_z[i_precompute_angle_closest] gives us a round about
            # idea of potential error in slope.  We'll use this to adjust the slope and see if we can get a fit.
            potential_slope_error = 0.6 / d_ordinate_dist

            # Set lower and upper bounds for the slope search.
            slope_lower = max(d_slope_use - potential_slope_error, 1e-8) # Avoids domain error, taking sqrt of negative number, in find wse
            slope_upper = d_slope_use + potential_slope_error

            # if slope is greater than the threshold, let's change it to the threshold
            if slope_upper > 0.03:
                slope_upper = 0.03

            # Check if the objective function changes sign between the bounds.
            f_lower = objective_with_slope(slope_lower, *slope_obj_args)
            f_upper = objective_with_slope(slope_upper, *slope_obj_args)
            
            
            if safe_signs_differ(f_lower, f_upper):
                
                # The signs differ, so we have a valid bracket.
                trial_slope_use = brentq(objective_with_slope, slope_lower, slope_upper, xtol=0.0001, args=slope_obj_args)
            
                # Optionally, recompute d_maxflow_wse_final and d_q_sum with the new slope:
                d_maxflow_wse_final_test, d_q_sum_test = find_wse(
                    2501, 
                    d_maxflow_wse_initial, 
                    d_depth_increment_small, 
                    d_q_maximum, 
                    x_section,
                    trial_slope_use
                )
                # Check if d_q_sum is within acceptable bounds
                # 20250808 Joseph changed this
                if d_q_sum < d_q_maximum * 1.5 or d_q_sum > d_q_maximum * 0.5:
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
                    x_section,
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
                    x_section,
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
        
        # one more check of outliers to make sure we don't have any
        if d_q_sum > d_q_maximum * 1.5 or d_q_sum < d_q_maximum * 0.5:
            acceptable = False

        if not acceptable:
            hydraulic_data.add_empty_x_section_for_curve_file(i_cell_comid, d_q_maximum, d_slope_use)
            continue
        
        # if we have a usable value for d_maxflow_wse_final, lets get rest of the VDT data
        if acceptable and d_maxflow_wse_final > 0.0:
            # Now lets get a set number of increments between the low elevation and the elevation where Qmax hits
            d_inc_y = (d_maxflow_wse_final - x_section.get_thalweg()) / i_number_of_increments
            i_number_of_elevations = i_number_of_increments + 1
            i_start_elevation_index, i_last_elevation_index = flood_increments(i_number_of_increments + 1, 
                                                                            d_inc_y, 
                                                                            x_section, d_slope_use, 
                                                                            d_q_sum, hydraulic_data)

            if d_q_baseflow > 0.001 and hydraulic_data.is_start_q_greater_than_baseflow(i_start_elevation_index, d_q_baseflow):
                hydraulic_data.set_q_at_index(i_start_elevation_index + 1, d_q_baseflow - 0.001)
                
            # Process each of the elevations to the output file if feasbile values were produced
            da_total_q_half_sum = sum(hydraulic_data.da_total_q[0 : int(i_number_of_elevations / 2.0)])
            if da_total_q_half_sum > 1e-16 and i_row_cell >= 0 and i_column_cell >= 0 and dm_elevation[i_row_cell, i_column_cell] > 1e-16:
                hydraulic_data.set_vdt_data(i_cell_comid, d_q_baseflow, d_slope_use, i_number_of_elevations)

            if i_number_of_elevations > 0:
                b_outprint_yes = True

        # Gather up all the values for the stream cell if we are going to build a reach average curve file
        hydraulic_data.set_non_vdt_data(b_outprint_yes, i_start_elevation_index, i_last_elevation_index, i_cell_comid, i_row_cell, i_column_cell,
                                        d_slope_use, d_dem_low_point_elev, d_q_maximum)
   
    # Create the output VDT Database file - datatypes are figured out automatically
    if not hydraulic_data.vdt_data_exists():
        LOG.warning('No VDT data was generated, so no output VDT database file will be created.')
        return
    
    hydraulic_data.save_files()
    
    #write_output_raster('StreamAngles.tif', dm_output_streamangles[i_boundary_number:nrows + i_boundary_number, i_boundary_number:ncols + i_boundary_number], ncols, nrows, dem_geotransform, dem_projection, "GTiff", gdal.GDT_Float32)
    

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
