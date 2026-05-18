"""
Automated Rating Curve (ARC) generator.

This module implements the core ARC workflow:

1. Read geospatial rasters (DEM, stream IDs, land cover) and a flow table.
2. For each stream raster cell, sample and adjust a cross-section.
3. Estimate bathymetry (optional).
4. Compute hydraulic relationships (WSE, depth, velocity, top width) for a set
   of discharge increments and write requested outputs.

ARC can be run from Python via :class:`arc.arc.Arc` or from the command line via
the ``arc`` console script.

Notes
-----
ARC's configuration is controlled by a "model input file" (MIF) and/or an
override ``args`` dictionary. Input parameter strings are documented on the ARC
wiki (see the repository's GitHub Wiki).
"""

import sys
import os
import math
import warnings
from typing import Literal

import tqdm
import yaml
import numpy as np
import pandas as pd
from datetime import datetime
import geopandas as gpd
from scipy.optimize import OptimizeWarning, brentq
from shapely.geometry import LineString, MultiLineString
from osgeo import gdal
from pyproj import CRS, Geod
from numba import njit, vectorize
from multiprocessing import Pool, shared_memory

from arc import LOG
from arc.cross_section import CrossSection, calculate_discharge_from_wse, _calculate_all
from arc.hydraulic_data import HydraulicData, add_hydraulic_data

warnings.filterwarnings("ignore", category=OptimizeWarning)
gdal.UseExceptions()

_DEM: np.ndarray = None
_STREAMS: np.ndarray = None
_BATHYMETRY: np.ndarray = None
_MANNINGS_N: np.ndarray = None
_LAND_COVER: np.ndarray = None
_OUTPUT_DATA_ARRAY: np.ndarray = None
_OUT_FLOOD: np.ndarray = None
_PARAMS: dict | None = None
_SHARED_MEMORYS: dict[str, shared_memory.SharedMemory] = {}
_CROSS_SECTION: CrossSection = None
_HYDRAULIC_DATA: HydraulicData = None
_INDEX_ARRAYS: np.ndarray = None
_Z_DISTANCE_ARRAY: np.ndarray = None
_INDEX_FRACT_ARRAYS: np.ndarray = None
_CELL_ROWS: np.ndarray = None
_CELL_COLS: np.ndarray = None
_CELL_COMIDS: np.ndarray = None
_CELL_QBASE: np.ndarray = None
_CELL_QMAX: np.ndarray = None
_CELL_REACH_SLOPE: np.ndarray = None
_CELL_SLOPE_25: np.ndarray = None
_CELL_SLOPE_75: np.ndarray = None

ARRAY_NAMES = [
    '_DEM',
    '_STREAMS',
    '_BATHYMETRY',
    '_MANNINGS_N',
    '_LAND_COVER',
    '_OUTPUT_DATA_ARRAY',
    '_OUT_FLOOD',
    '_INDEX_ARRAYS',
    '_Z_DISTANCE_ARRAY',
    '_INDEX_FRACT_ARRAYS',
    '_CELL_ROWS',
    '_CELL_COLS',
    '_CELL_COMIDS',
    '_CELL_QBASE',
    '_CELL_QMAX',
    '_CELL_REACH_SLOPE',
    '_CELL_SLOPE_25',
    '_CELL_SLOPE_75',
]

def get_cross_section(*args):
    global _CROSS_SECTION, _INDEX_ARRAYS, _Z_DISTANCE_ARRAY, _INDEX_FRACT_ARRAYS
    if _CROSS_SECTION is None and args:
        _CROSS_SECTION = CrossSection(*args)
        _CROSS_SECTION.associate_with_precomputed_index_arrays(_INDEX_ARRAYS, _Z_DISTANCE_ARRAY, _INDEX_FRACT_ARRAYS)
    return _CROSS_SECTION

def get_hydraulic_data(*args):
    global _HYDRAULIC_DATA
    if _HYDRAULIC_DATA is None:
        _HYDRAULIC_DATA = HydraulicData(*args)
        _HYDRAULIC_DATA.associate_with_cross_section(get_cross_section())
        _HYDRAULIC_DATA.associate_with_output_data(_OUTPUT_DATA_ARRAY)
    return _HYDRAULIC_DATA

def _set_shared(name: str, shm: shared_memory.SharedMemory):
    """
    We need the shared memory objects to persist somewhere; otherwise, the memory is freed and the numpy arrays point to invalid memory.
    These must last the lifetime of the program! Reason being, bathymetry is the last thing written, and we need the shared memory to persist until then.
    """
    global _SHARED_MEMORYS
    _SHARED_MEMORYS[name] = shm

def reset_globals():
    for name in ARRAY_NAMES + ['_CROSS_SECTION', '_HYDRAULIC_DATA']:
        globals()[name] = None

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
            xy_to_rowcol,
            length_m,
        )

        z_end, dist_from_end = sample_line_for_valid_z(
            LineString(list(line_geom.coords)[::-1]),
            dm_elevation,
            xy_to_rowcol,
            length_m,
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

def read_and_pad_and_maybe_make_shared(s_input_filename: str, processes: int, pad_distance: int, dtype: np.dtype, array_name: str):
    """
    Read a raster into memory, pad it, and optionally place it in shared memory.

    Parameters
    ----------
    s_input_filename : str
        Path to the input raster (GDAL-readable).
    processes : int
        Number of worker processes. If ``processes > 1``, ARC places the padded
        raster in :mod:`multiprocessing.shared_memory` so workers can access it
        without per-process copies.
    pad_distance : int
        Number of cells to pad on each edge (used to avoid boundary issues for
        neighborhood operations like slope/direction/cross-section sampling).
    dtype : numpy.dtype
        Dtype to cast the raster values to after reading.
    array_name : str
        Name of the global array variable to assign when shared memory is used
        (e.g., ``"_DEM"``).

    Returns
    -------
    dm_raster_array : numpy.ndarray
        Padded raster array (possibly shared-memory backed).
    l_geotransform : tuple
        GDAL geotransform.
    s_raster_projection : str
        Raster projection (WKT).

    """

    # Check that the file exists to open
    if os.path.isfile(s_input_filename) == False:
        LOG.info('Cannot Find Raster ' + s_input_filename)

    # Attempt to open the dataset
    o_dataset: gdal.Dataset = gdal.Open(s_input_filename, gdal.GA_ReadOnly)
    if o_dataset is None:
        LOG.info('Cannot Open Raster ' + s_input_filename)
        raise FileNotFoundError(f"Cannot open raster {s_input_filename}")

    # Retrieve dimensions of cell size and cell count then close DEM dataset
    l_geotransform = o_dataset.GetGeoTransform()

    # Read the size of the band object
    o_band: gdal.Band = o_dataset.GetRasterBand(1)
    i_number_of_columns = o_band.XSize
    i_number_of_rows = o_band.YSize
    shape = (i_number_of_rows + 2 * pad_distance, i_number_of_columns + 2 * pad_distance)

    # Use this function, which handles both the single-process and multi-process cases, to create the array and shared memory if needed
    dm_raster_array = create_array(array_name, processes, shape, dtype, fill_value=0)

    # Read raster into preallocated array, leaving a border of zeros around the edge based on the pad distance
    dm_raster_array[pad_distance:-pad_distance, pad_distance:-pad_distance] = o_band.ReadAsArray()

    # Close the band object
    o_band = None

    # Normalize south-up rasters (pixel height > 0) to north-up arrays.
    if l_geotransform[5] > 0:
        LOG.warning('Raster appears south-up (positive pixel height); flipping to north-up: ' + str(s_input_filename))
        dm_raster_array[:] = np.flipud(dm_raster_array)

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
    Parse an ARC model input file (MIF) and apply overrides.

    Parameters
    ----------
    s_mif_name : str
        Path to the MIF text file. The file is tab-delimited with one parameter
        per line (``<ParameterString>\\t<Value>``), or it is a YAML file. If empty, ARC builds an
        in-memory "file" from ``args``.
    args : dict
        Parameter overrides. Keys correspond to the input-file parameter
        strings (e.g., ``"DEM_File"``, ``"Stream_File"``, ``"Print_VDT_Database"``).
        Values in this dict override values in the MIF.

    Returns
    -------
    dict
        Normalized parameter dictionary used by the simulation.

    """

    ### Open and read the input file ###
    # Open the file
    if s_mif_name:
        if s_mif_name.lower().endswith(('.yaml', '.yml')):
            # If it's a YAML file, parse it with PyYAML and convert to the expected list of lines format
            data = yaml.safe_load(open(s_mif_name))
            sl_lines = [f"{key}\t{value}\n" for key, value in data.items()]
        else:
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

    # check for baseflow parameters for bathymetry estimation. If not provided, disable bathymetry estimation.
    s_flow_file_baseflow = get_parameter_name(sl_lines,  'Flow_File_BF')
    s_flow_file_qmax = get_parameter_name(sl_lines,  'Flow_File_QMax')
    s_output_bathymetry_path = get_parameter_name(sl_lines,  'AROutBATHY', get_parameter_name(sl_lines,  'BATHY_Out_File'))
    if s_flow_file_baseflow == '' and len(s_output_bathymetry_path) > 1:
        LOG.warning('Flow_File_BF was not provided; disabling bathymetry estimation.')
        s_output_bathymetry_path = ''

    params = {
        's_input_dem_path': get_parameter_name(sl_lines,  'DEM_File'), # Find the path to the DEM file
        's_stream_slope_method': s_stream_slope_method,
        's_strmshp_path'    : s_strmshp_path,
        's_input_stream_path': get_parameter_name(sl_lines,  'Stream_File'), # Find the path to the stream file
        's_input_land_use_path': get_parameter_name(sl_lines,  'LU_Raster_SameRes'), # Find the path to the land use raster file
        's_input_mannings_path': get_parameter_name(sl_lines,  'LU_Manning_n'), # Find the path to the mannings n file
        's_input_flow_file_path': get_parameter_name(sl_lines,  'Flow_File'), # Find the path to the flow file
        's_flow_file_id': get_parameter_name(sl_lines,  'Flow_File_ID'), # Find the column name 
        's_flow_file_baseflow': s_flow_file_baseflow, # Find the baseflow column name
        's_flow_file_qmax': s_flow_file_qmax, # Find the column name for the maximum flow
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
        's_output_bathymetry_path': s_output_bathymetry_path, # Find the path to the output bathymetry file
        's_xs_output_file': get_parameter_name(sl_lines,  'XS_Out_File'), # Find the path to the output cross-section file (JLG added this to recalculate top-width and velocity)
        'i_lc_water_value': int(get_parameter_name(sl_lines,  'LC_Water_Value', 80)), # Find the value in the land cover dataset that corresponds to water. This is used to find the banks of the river if b_FindBanksBasedOnLandCover is set to True
        'i_number_of_increments': int(get_parameter_name(sl_lines,  'VDT_Database_NumIterations', 15)), # Find the number of increments to use in the velocity, depth, and top width database
        'b_FindBanksBasedOnLandCover': b_FindBanksBasedOnLandCover, # Find the true/false variable to find the banks of the river based on the land cover dataset instead of the DEM
        'b_reach_average_curve_file': b_reach_average_curve_file, # Find the true/false variable to use a reach-average curve file
        's_output_flood': get_parameter_name(sl_lines,  'AROutFLOOD'), # Find the path to the output flood file

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
    Read streamflow information for ARC.

    Parameters
    ----------
    s_flow_file_name : str
        Path to a CSV containing per-reach flow information.
    s_flow_id : str
        Column name containing the stream/reach identifier (typically COMID).
    s_flow_baseflow : str
        Column name containing the baseflow discharge (used for bathymetry and
        metadata).
    s_flow_qmax : str
        Column name containing the maximum discharge used to build rating-curve
        increments.

    Returns
    -------
    dict
        Mapping ``reach_id -> {flow_column: value, ...}``. If ``s_flow_baseflow``
        is blank, only the qmax column is loaded.

    """
    if s_flow_file_name.endswith('.parquet'):
        df = pd.read_parquet(s_flow_file_name)
    else:
        df = pd.read_csv(s_flow_file_name)

    flow_columns = [s_flow_qmax] if s_flow_baseflow == '' else [s_flow_baseflow, s_flow_qmax]
    return df.set_index(s_flow_id)[flow_columns].to_dict(orient='index')

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
def get_reach_median_stream_slope_information(dm_dem: np.ndarray, im_streams: np.ndarray, stream_id: int, d_dx: float, d_dy: float, i_general_slope_distance: int):
    """
    Calculates the stream slope for each stream cell using the following process:

        1.) Find all stream cells that have the same stream id value
        2.) Look at the slope of each of the stream cells.
        3.) Average the slopes to get the overall slope we use in the model.

    Guaranteed to be >= 0.0002 and <= 0.03

    Parameters
    ----------
    dm_dem: ndarray
        Elevation raster
    im_streams: ndarray
        Stream raster
    stream_id: int
        ID of the stream for which to calculate slope
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
                    slope = np.round(abs(za - zb) / dist, 8)
                    if slope > 0.0:
                        total_slope += slope
                        count += 1
                        slope_list.append(slope)

    # remove any outliers using quartiles
    if len(slope_list) > 0:
        slope_arr = np.array(slope_list)
        slope_arr = round_sig(slope_arr, 8)   
        Q1 = np.round(np.percentile(slope_arr, 25), 8)
        Q3 = np.round(np.percentile(slope_arr, 75), 8)
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

    # Get the indices of all locations of the stream id within a box around the cell of interest
    row_min = i_row - i_general_slope_distance
    row_max = i_row + i_general_slope_distance
    col_min = i_column - i_general_slope_distance
    col_max = i_column + i_general_slope_distance

    total = 0.0
    count = 0
    # Find the slope if there are stream cells
    for r in range(row_min, row_max):
        for c in range(col_min, col_max):
            if im_streams[r, c] != i_cell_value:
                continue
            
            dr = r - i_row
            dc = c - i_column

            if dr == 0 and dc == 0:
                continue

            # Distance between the cell of interest and a cell with a similar stream id
            dx = dc * d_dx
            dy = dr * d_dy
            dist = math.sqrt(dx*dx + dy*dy)

            if dist > 0.0:
                total += abs(d_cell_of_interest - dm_dem[r, c]) / dist
                count += 1

    # Average across the cells
    if count > 0:
        d_stream_slope = total / count

    return d_stream_slope

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
    # Get the COMID from the stream raster
    stream_id = im_streams[i_row, i_column]

    # Define the search box around the cell of interest
    row_min = i_row - i_general_direction_distance
    row_max = i_row + i_general_direction_distance
    col_min = i_column - i_general_direction_distance
    col_max = i_column + i_general_direction_distance

    # Regression accumulators
    n = 0
    sum_x = 0.0
    sum_y = 0.0
    sum_xy = 0.0
    sum_x2 = 0.0

    # Search for stream cells within the box and accumulate values for linear regression to find the dominant direction of the stream
    for r in range(row_min, row_max):
        for c in range(col_min, col_max):
            if im_streams[r, c] != stream_id:
                continue

            # local coordinates centered at target cell
            x = c - i_column
            y = r - i_row

            sum_x += x
            sum_y += y
            sum_xy += x * y
            sum_x2 += x * x
            n += 1

    if n <= 1:
        return 0.0, 0.0

    denom = n * sum_x2 - sum_x * sum_x
    numer = n * sum_xy - sum_x * sum_y

    #If this occurs it means the line is straight up
    if denom <= 1e-6 or abs(numer) <= 1e-6:
        dx = 0.0
        dy = 0.0

        for r in range(row_min, row_max):
            for c in range(col_min, col_max):
                if im_streams[r, c] == stream_id:
                    dx = max(dx, abs(c - i_column))
                    dy = max(dy, abs(r - i_row))

        # Even though the regression cant find the slope, it is dominated in the X direction, meaning angle of zero
        if dx > dy:
            d_stream_direction = 0.0
        else:
            #The change in Y direction is dominant, meaning a stream angle of pi/2
            d_stream_direction = np.pi / 2.0
    else:
        slope = numer / denom
        # Convert slope to angle in radians (normalized to be between 0 and 2pi)
        d_stream_direction = np.arctan(slope) % (2 * np.pi)

    d_xs_direction = d_stream_direction - np.pi / 2.0
    if d_xs_direction < 0.0:
        d_xs_direction += np.pi

    return d_stream_direction, d_xs_direction

def read_manning_table(s_manning_path: str, land_cover_array: np.ndarray, processes: int):
    """
    Reclassify a land-cover raster into Manning's *n* values.

    Parameters
    ----------
    s_manning_path : str
        Path to a tab-delimited table mapping land-cover codes to Manning's
        roughness values.
    land_cover_array : numpy.ndarray
        Land-cover raster (integer codes).
    processes : int
        Number of worker processes. If ``processes > 1``, the returned array may
        be allocated in shared memory for worker access.

    Returns
    -------
    numpy.ndarray
        Manning's *n* raster aligned to ``land_cover_array``.

    """

    # Open and read the input file
    if s_manning_path.endswith('.parquet'):
        df = pd.read_parquet(s_manning_path)
    else:
        df = pd.read_csv(s_manning_path, sep='\t')

    # Create a lookup array for the Manning's n values
    # This is the fastest way to reclassify the values in the input array
    idx = df.iloc[:, 0].astype(np.uint8).values
    lookup_array = np.zeros(256, dtype=np.float32)
    lookup_array[idx] = df.iloc[:, 2].values

    # Create the output array and fill it with the Manning's n values based on the land cover array
    output_raster = create_array("_MANNINGS_N", processes, land_cover_array.shape, np.float32, fill_value=0.0)
    output_raster[:] = lookup_array[land_cover_array]
    
    # Correct the mannings values here
    output_raster[output_raster > 10] = 0.035
    output_raster[output_raster <= 0.0] = 0.005
    

@njit(cache=True)
def find_wse(range_end, start_wse, increment, d_q_maximum, x_sect_args, d_slope_use):
    d_q_sum = 0.0
    sqrt_slope = d_slope_use**0.5

    low = 0
    high = range_end
    
    # Use bisection algorithm to find the water surface elevation that corresponds to the target discharge
    while high - low > 1:
        mid = (low + high) // 2
        wse = start_wse + mid * increment
        d_q_sum = calculate_discharge_from_wse(wse, sqrt_slope, *x_sect_args)

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
        d_q_sum = calculate_discharge_from_wse(d_wse, sqrt_slope, *x_sect_args)

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
                d_q_sum = calculate_discharge_from_wse(interp_wse, sqrt_slope, *x_sect_args)
                d_wse = interp_wse
            break

        # Save current values for the next iteration
        prev_wse = d_wse
        prev_q = d_q_sum
        can_interpolate = True

    return d_wse, d_q_sum

@njit(cache=True)
def flood_increments(i_number_of_increments: int, d_inc_y: float, flood_increments_args: tuple, thalweg: float, d_slope_use: float, d_q_sum: float, output_data: np.ndarray, i_entry_cell: int, b_modified_dem: bool):
    i_start_elevation_index, i_last_elevation_index = 0, 0

    # Initialize previous values
    prev_t = 0.0
    prev_a = 0.0
    prev_p = 0.0
    prev_q = 0.0
    prev_v = 0.0
    prev_wse = 0.0
    sqrt_slope = d_slope_use**0.5

    for i_entry_elevation in range(i_number_of_increments):
        d_wse = thalweg + d_inc_y * i_entry_elevation

        # Calculate the geometry          
        A, P, V, Q, T = _calculate_all(*flood_increments_args, d_wse, sqrt_slope)

        if T > 0 and A > 0 and P > 0:
            if Q < prev_q:
                # increase d_wse by 1 cm to try to make sure Q is greater than prev_q
                d_wse_lower_bound = d_wse + 0.01
                # set the upper bound for the water surface elevation to the next increment
                d_wse_upper_bound = thalweg + d_inc_y * (i_entry_elevation + 1)
                d_wse_upper_bound = np.round(d_wse_upper_bound, 3)
                while d_wse_lower_bound < d_wse_upper_bound:
                    # Calculate the geometry       
                    A, P, V_cand, Q_cand, T = _calculate_all(*flood_increments_args, d_wse_lower_bound, sqrt_slope)   

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
                add_hydraulic_data(output_data, i_entry_elevation, prev_wse, prev_t, prev_p, prev_q, prev_v, i_entry_cell, b_modified_dem)
                continue

            # Save the values
            add_hydraulic_data(output_data, i_entry_elevation, d_wse, T, P, Q, V, i_entry_cell, b_modified_dem)

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
            add_hydraulic_data(output_data, i_entry_elevation, 0, 0, 0, 0, 0, i_entry_cell, b_modified_dem)

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

    return b_modified_dem

def get_reach_median_stream_slope_information_wrapper(args):
    return get_reach_median_stream_slope_information(_DEM, _STREAMS, *args)

def create_reach_average_slope_dicts(dm_stream, dx, dy, quiet, i_general_slope_distance, processes):
    # create a list of unique stream IDs to loop through
    unique_stream_ids = np.unique(dm_stream)
    unique_stream_ids = unique_stream_ids[unique_stream_ids > 0]
    pbar_slopes = tqdm.tqdm(unique_stream_ids, disable=quiet)
    dict_stream_slopes = {}
    dict_stream_slopes_25th = {}
    dict_stream_slopes_75th = {}
    if processes == 1:
        for stream_id in pbar_slopes:
            reach_slope, reach_slope_25th, reach_slope_75th = get_reach_median_stream_slope_information(_DEM, dm_stream, stream_id, dx, dy, i_general_slope_distance)
            dict_stream_slopes[stream_id] = reach_slope
            dict_stream_slopes_25th[stream_id] = reach_slope_25th
            dict_stream_slopes_75th[stream_id] = reach_slope_75th
    else:
        args = get_init_parallel_args(["_DEM", "_STREAMS"])
        with Pool(processes, initializer=init_parallel, initargs=args) as pool:
            chunksize = min(10, len(unique_stream_ids) // (processes * 4) + 1)  # Adjust chunksize based on the number of processes and total tasks. I found 10 to be the most we should go
            for stream_id, (reach_slope, reach_slope_25th, reach_slope_75th) in zip(pbar_slopes, pool.imap(get_reach_median_stream_slope_information_wrapper, [(stream_id, dx, dy, i_general_slope_distance) for stream_id in unique_stream_ids], chunksize=chunksize)):
                dict_stream_slopes[stream_id] = reach_slope
                dict_stream_slopes_25th[stream_id] = reach_slope_25th
                dict_stream_slopes_75th[stream_id] = reach_slope_75th


    return dict_stream_slopes, dict_stream_slopes_25th, dict_stream_slopes_75th

def dict_stream_slopes_from_endpoints(dm_stream, dem_geotransform, dem_projection, s_strmshp_path, s_flow_file_id, quiet):
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
        slope_pct, slope_deg, z_start, z_end, length_m = line_slope_from_dem(StrmSHP_geom.iloc[0], _DEM, dem_geotransform, length_m)
        dict_stream_slopes[stream_id] = round(slope_pct/100, 8)

    return dict_stream_slopes

@njit(cache=True)
def objective_with_wse(trial_wse: float, slope_squared: float,
                       d_q_maximum: float, x_sect_args: tuple) -> float:
    # Define an objective function: the difference between the calculated max flow and d_q_maximum.
    trial_wse = np.round(trial_wse, 3)

    trial_d_q_sum = calculate_discharge_from_wse(trial_wse, slope_squared, *x_sect_args)

    # trial_d_q_sum = round(trial_d_q_sum, 3)
    difference = trial_d_q_sum - d_q_maximum

    # The objective is zero when trial_d_q_sum equals d_q_maximum.
    return difference


# Define an objective function: the difference between the calculated max flow and d_q_maximum.
@njit(cache=True)
def objective_with_slope(trial_slope: float,
                         d_maxflow_wse_initial: float, d_depth_increment_small: float, d_q_maximum: float,
                         x_sect_args) -> float:
    # find_wse returns a tuple: (d_maxflow_wse_final, d_q_sum)
    _, trial_d_q_sum = find_wse(
        2501, 
        d_maxflow_wse_initial, 
        d_depth_increment_small, 
        d_q_maximum, 
        x_sect_args,
        trial_slope
    )
    # The objective is zero when trial_d_q_sum equals d_q_maximum.
    return trial_d_q_sum - d_q_maximum

def initialize_stream_slope_dictionaries(params: dict, dx, dy, dem_geotransform, dem_projection, quiet, processes):
    s_stream_slope_method = params['s_stream_slope_method']
    if s_stream_slope_method == 'reach_average' or s_stream_slope_method == 'local_average_corrected':
        dict_stream_slopes, dict_stream_slopes_25th, dict_stream_slopes_75th = create_reach_average_slope_dicts(_STREAMS, dx, dy, quiet, params['i_general_slope_distance'], processes)
        return (dict_stream_slopes, dict_stream_slopes_25th, dict_stream_slopes_75th)
    elif s_stream_slope_method == 'end_points':
        dict_stream_slopes = dict_stream_slopes_from_endpoints(_STREAMS, dem_geotransform, dem_projection, params['s_strmshp_path'], params['s_flow_file_id'], quiet)
        return (dict_stream_slopes, None, None)
    
    return (None, None, None)

def calculate_hydraulic_data_for_cell(i_entry_cell: int):
    """
    Compute bathymetry and hydraulic increments for a single stream cell.

    This function is the core per-cell kernel. It reads per-cell metadata
    (row/col, COMID, baseflow, qmax) from shared/global arrays, samples a
    cross-section, optionally estimates bathymetry, then fills the shared output
    array with hydraulic results.

    Parameters
    ----------
    i_entry_cell : int
        Index into the per-cell arrays (rows/cols/COMIDs/flows). This is *not*
        a raster index; it is the index of the extracted stream-cell list.

    Returns
    -------
    tuple or None
        If cross-section output is enabled, returns a tuple containing the
        per-cell cross-section export fields. Otherwise returns ``None``.
    """
    i_row_cell = _CELL_ROWS[i_entry_cell]
    i_column_cell = _CELL_COLS[i_entry_cell]
    i_cell_comid = _CELL_COMIDS[i_entry_cell]
    d_q_baseflow = _CELL_QBASE[i_entry_cell]
    d_q_maximum = _CELL_QMAX[i_entry_cell]
    i_number_of_increments = _PARAMS['i_number_of_increments']
    i_general_direction_distance = _PARAMS['i_general_direction_distance']
    i_general_slope_distance = _PARAMS['i_general_slope_distance']

    
    d_depth_increment_big = 0.5
    d_depth_increment_med = 0.05
    d_depth_increment_small = 0.01


    # Get the Slope of each Stream Cell. Slope should be in m/m
    s_stream_slope_method = _PARAMS['s_stream_slope_method']
    dx = _PARAMS['dx']
    dy = _PARAMS['dy']
    if s_stream_slope_method == 'local_average':
        d_slope_use = get_local_average_stream_slope_information(i_row_cell, i_column_cell, _DEM, _STREAMS, dx, dy, i_general_slope_distance)
    elif s_stream_slope_method =='reach_average' or s_stream_slope_method == 'end_points':
        d_slope_use = _CELL_REACH_SLOPE[i_entry_cell]
    elif s_stream_slope_method == 'local_average_corrected':
        d_slope_use = get_local_average_stream_slope_information(i_row_cell, i_column_cell, _DEM, _STREAMS, dx, dy, i_general_slope_distance)
        d_slope_25th = _CELL_SLOPE_25[i_entry_cell]
        d_slope_75th = _CELL_SLOPE_75[i_entry_cell]
        # if the corrected slope is less than the streams 25th percentile slope, use the 25th percentile slope
        if d_slope_use < d_slope_25th:
            d_slope_use = d_slope_25th
        # if the corrected slope is greater than the streams 75th percentile slope, use the 75th percentile slope
        elif d_slope_use > d_slope_75th:
            d_slope_use = d_slope_75th  
    else: 
        #Default to using the 'local_average' method
        d_slope_use = get_local_average_stream_slope_information(i_row_cell, i_column_cell, _DEM, _STREAMS, dx, dy, i_general_slope_distance)

    # Get the Stream Direction of each Stream Cell.  Direction is between 0 and pi.  Also get the cross-section direction (also between 0 and pi)
    d_stream_direction, d_xs_direction = get_stream_direction_information(i_row_cell, i_column_cell, _STREAMS, i_general_direction_distance)

    # Now Pull the Cross-Section again with the new angle
    x_section = get_cross_section(dx, dy, _DEM, _LAND_COVER, _PARAMS)
    if d_xs_direction > np.pi:
        i_precompute_angle_closest = int(round((d_xs_direction-np.pi) / x_section.d_precompute_angles))
    else:
        i_precompute_angle_closest = int(round(d_xs_direction / x_section.d_precompute_angles))

    x_section.set_cross_section(i_row_cell, i_column_cell, i_precompute_angle_closest, d_xs_direction)
    
    # Adjust to the lowest-point in the Cross-Section
    i_low_spot_range = _PARAMS['i_low_spot_range']
    if i_low_spot_range > 0:
        x_section.adjust_cross_section_to_lowest_point(i_low_spot_range)
        # The r and c for the stream cell is adjusted because it may have moved
        i_row_cell, i_column_cell = x_section.get_row_col()
    
    d_dem_low_point_elev = x_section.get_thalweg()

    # Adjust cross-section angle to ensure shortest top-width at a specified depth
    if x_section.has_angles_to_test():
        x_section.test_angles_and_reset_cross_section(i_row_cell, i_column_cell)

    # Burn bathymetry profile into cross-section profile
    # "Be the banks for your river" - Needtobreathe
            
    # If you don't have a cross-section, skip it and fill in empty values for the reach average processing
    hydraulic_data = get_hydraulic_data(_PARAMS)
    if not x_section.is_valid():
        hydraulic_data.add_empty_x_section_for_curve_file(i_cell_comid, d_slope_use, i_entry_cell)
        return

    #BATHYMETRY CALCULATION
    #This method calculates bathymetry based on the water surface elevation or LandCover ("FindBanksBasedOnLandCover" and "LC_Water_Value").
    b_bathy_use_banks = _PARAMS['b_bathy_use_banks']
    s_output_bathymetry_path = _PARAMS['s_output_bathymetry_path']
    if not b_bathy_use_banks and s_output_bathymetry_path != '':
        x_section.Calculate_Bathymetry_Based_on_WSE_or_LC(d_q_baseflow, d_slope_use, _BATHYMETRY)
    #This method calculates the banks based on the Riverbank
    elif b_bathy_use_banks and s_output_bathymetry_path != '':
        x_section.Calculate_Bathymetry_Based_on_RiverBank_Elevations(d_q_baseflow, d_slope_use, _BATHYMETRY)

    # Calculate the volumes
    # VolumeFillApproach 1 is to find the height within ElevList_mm that corresponds to the Qmax flow.  THen increment depths to have a standard number of depths to get to Qmax.  
    # This is preferred for VDTDatabase method.
    if _PARAMS['s_output_flood']:
        _OUT_FLOOD[i_row_cell, i_column_cell] = 3
    
    # Here are the n values for each side of the cross-section
    x_section.set_mannings_n_values(_MANNINGS_N)

    # space between ordinates in the cross-section
    d_ordinate_dist = x_section.d_ordinate_dist

    # we'll assume the results are acceptable until we think otherwise
    acceptable = True

    # This is the bottom of the channel
    thalweg = x_section.get_thalweg()
    d_maxflow_wse_initial = thalweg

    # set this as the default in case we don't find a better one
    d_maxflow_wse_final = -999.0

    # initialize some variables
    d_q_sum = 0.0
    slope_use_squared = d_slope_use ** 0.5

    wse_lower = d_maxflow_wse_initial + 0.01
    wse_upper = d_maxflow_wse_initial + 24.99
    x_sect_args = x_section.get_calculate_discharge_from_wse_args()
    wse_obj_args = (slope_use_squared, d_q_maximum, x_sect_args)

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
    d_maxflow_wse_initial, d_q_sum_test = find_wse(101, d_maxflow_wse_initial, d_depth_increment_big, d_q_maximum, x_sect_args, d_slope_use)


    # Based on using depth increments of 0.5, now lets fine-tune the wse using depth increments of 0.05
    d_maxflow_wse_initial = max(d_maxflow_wse_initial - 0.5, thalweg)
    d_maxflow_wse_med = d_maxflow_wse_initial
    d_maxflow_wse_med, d_q_sum_test = find_wse(101, d_maxflow_wse_med, d_depth_increment_med, d_q_maximum, x_sect_args, d_slope_use)

    # Based on using depth increments of 0.05, now lets fine-tune the wse even more using depth increments of 0.01
    d_maxflow_wse_med = max(d_maxflow_wse_med - 0.05, thalweg)
    d_maxflow_wse_final_test = d_maxflow_wse_med
    d_maxflow_wse_final_test, d_q_sum_test = find_wse(2501, d_maxflow_wse_med, d_depth_increment_small, d_q_maximum, x_sect_args, d_slope_use)

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

    slope_obj_args = (d_maxflow_wse_initial, d_depth_increment_small, d_q_maximum, x_sect_args)
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
            x_sect_args,
            trial_slope_use
        )
        # Check if d_q_sum is within acceptable bounds
        if abs(d_q_sum_test - d_q_maximum) < abs(d_q_sum-d_q_maximum):
            # Optionally update d_slope_use to the accepted value:
            d_slope_use = trial_slope_use
            d_maxflow_wse_final = d_maxflow_wse_final_test
            d_q_sum = d_q_sum_test
    # if the f_lower or f_upper is equal to zero, it's probably close enough to be the WSE we are looking for, so we'll use it
    elif np.round(f_lower, 5) == 0 or np.round(f_upper, 5) == 0:          
        trial_slope_use = np.round(slope_lower, 3) if np.round(f_lower, 5) == 0 else np.round(slope_upper, 3)
        # Optionally, recompute d_maxflow_wse_final and d_q_sum with the new slope:
        d_maxflow_wse_final_test, d_q_sum_test = find_wse(
            2501, 
            d_maxflow_wse_initial, 
            d_depth_increment_small, 
            d_q_maximum, 
            x_sect_args,
            trial_slope_use
        )
        # Check if d_q_sum is within acceptable bounds
        if abs(d_q_sum_test - d_q_maximum) < abs(d_q_sum-d_q_maximum):
            # Optionally update d_slope_use to the accepted value:
            d_slope_use = trial_slope_use
            d_maxflow_wse_final = d_maxflow_wse_final_test
            d_q_sum = d_q_sum_test

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
                x_sect_args,
                trial_slope_use
            )
            # Check if d_q_sum is within acceptable bounds
            if d_q_maximum * 0.5 <= d_q_sum_test <= d_q_maximum * 1.5:
                acceptable = True
                d_slope_use = trial_slope_use
                d_maxflow_wse_final = d_maxflow_wse_final_test
                d_q_sum = d_q_sum_test
                return # Why is there a return here? This seems wrong, but I am leaving it assuming Joseph or Mike know why we should exit early if we find an acceptable solution here.

        # if the f_lower is equal to zero, it's probably close enough to be the WSE we are looking for, so we'll use it
        elif np.round(f_lower, 5) == 0:          
            trial_slope_use = np.round(slope_lower, 3)
            # Optionally, recompute d_maxflow_wse_final and d_q_sum with the new slope:
            d_maxflow_wse_final_test, d_q_sum_test = find_wse(
                2501, 
                d_maxflow_wse_initial, 
                d_depth_increment_small, 
                d_q_maximum, 
                x_sect_args,
                trial_slope_use
            )
            # Check if d_q_sum is within acceptable bounds
            if abs(d_q_sum_test - d_q_maximum) < abs(d_q_sum-d_q_maximum):
                # Optionally update d_slope_use to the accepted value:
                d_slope_use = trial_slope_use
                d_maxflow_wse_final = d_maxflow_wse_final_test
                d_q_sum = d_q_sum_test

        # if the f_upper is equal to zero, it's probably close enough to be the WSE we are looking for, so we'll use it
        elif np.round(f_upper, 5) == 0:          
            trial_slope_use = np.round(slope_upper, 3)
            # Optionally, recompute d_maxflow_wse_final and d_q_sum with the new slope:
            d_maxflow_wse_final_test, d_q_sum_test = find_wse(
                2501, 
                d_maxflow_wse_initial, 
                d_depth_increment_small, 
                d_q_maximum, 
                x_sect_args,
                trial_slope_use
            )
            # Check if d_q_sum is within acceptable bounds
            if abs(d_q_sum_test - d_q_maximum) < abs(d_q_sum-d_q_maximum):
                # Optionally update d_slope_use to the accepted value:
                d_slope_use = trial_slope_use
                d_maxflow_wse_final = d_maxflow_wse_final_test
                d_q_sum = d_q_sum_test

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
                x_sect_args,
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
                x_sect_args,
                trial_slope_use
            )
            # Check if d_q_sum is within acceptable bounds
            if abs(d_q_sum_test - d_q_maximum) < abs(d_q_sum-d_q_maximum):
                # Optionally update d_slope_use to the accepted value:
                d_slope_use = trial_slope_use
                d_maxflow_wse_final = d_maxflow_wse_final_test
                d_q_sum = d_q_sum_test

        # if the f_upper is equal to zero, it's probably close enough to be the WSE we are looking for, so we'll use it
        elif np.round(f_upper, 5) == 0:          
            trial_slope_use = np.round(slope_upper, 3)
            # Optionally, recompute d_maxflow_wse_final and d_q_sum with the new slope:
            d_maxflow_wse_final_test, d_q_sum_test = find_wse(
                2501, 
                d_maxflow_wse_initial, 
                d_depth_increment_small, 
                d_q_maximum, 
                x_sect_args,
                trial_slope_use
            )
            # Check if d_q_sum is within acceptable bounds
            if abs(d_q_sum_test - d_q_maximum) < abs(d_q_sum-d_q_maximum):
                # Optionally update d_slope_use to the accepted value:
                d_slope_use = trial_slope_use
                d_maxflow_wse_final = d_maxflow_wse_final_test
                d_q_sum = d_q_sum_test
    
    # one more check of outliers to make sure we don't have any
    if d_q_sum > d_q_maximum * 1.5 or d_q_sum < d_q_maximum * 0.5:
        acceptable = False

    if not acceptable:
        hydraulic_data.add_empty_x_section_for_curve_file(i_cell_comid, d_slope_use, i_entry_cell)
        return
    
    # This just tells the curve file whether to print out a result or not.  If no realistic depths were calculated, no reason to output results.
    add_curve_file_data = False

    # This is the first and last indice of elevations we'll need for the Curve Fitting for this cell
    i_start_elevation_index = -1
    i_last_elevation_index = 0

    # if we have a usable value for d_maxflow_wse_final, lets get rest of the VDT data
    if acceptable and d_maxflow_wse_final > 0.0:
        # Now lets get a set number of increments between the low elevation and the elevation where Qmax hits
        d_inc_y = (d_maxflow_wse_final - thalweg) / i_number_of_increments
        i_number_of_elevations = i_number_of_increments + 1
        flood_increments_args = x_section.get_flood_increment_args()
        i_start_elevation_index, i_last_elevation_index = flood_increments(i_number_of_increments + 1, 
                                                                        d_inc_y, 
                                                                        flood_increments_args, thalweg, d_slope_use, 
                                                                        d_q_sum, _OUTPUT_DATA_ARRAY, i_entry_cell, hydraulic_data.b_modified_dem)
        
        if i_last_elevation_index > i_start_elevation_index:
            if d_q_baseflow > 0.001 and hydraulic_data.is_start_q_greater_than_baseflow(i_start_elevation_index, d_q_baseflow, i_entry_cell):
                hydraulic_data.set_q_at_index(i_start_elevation_index + 1, d_q_baseflow - 0.001, i_entry_cell)
                
            # Process each of the elevations to the output file if feasbile values were produced
            hydraulic_data.set_vdt_data(i_cell_comid, d_q_baseflow, d_slope_use, i_entry_cell, i_number_of_elevations)

        add_curve_file_data = i_number_of_elevations > 0

    # Gather up all the values for the stream cell if we are going to build a reach average curve file
    hydraulic_data.set_non_vdt_data(add_curve_file_data, i_start_elevation_index, i_last_elevation_index, i_cell_comid, i_row_cell, i_column_cell,
                                    d_slope_use, d_dem_low_point_elev, i_entry_cell)
    
    if hydraulic_data.s_xs_output_file:
        return hydraulic_data.get_cross_section_data(i_cell_comid, i_row_cell, i_column_cell)
    
def close_shared_arrays(names: list[str] = None):
    """
    Close and unlink shared-memory arrays created by ARC.

    Parameters
    ----------
    names : list of str, optional
        Names of shared memory blocks to close. If omitted, all shared blocks
        tracked in the internal registry are closed and unlinked.

    Notes
    -----
    This should be called once ARC is done with shared arrays. Unlinking makes
    the shared memory segment eligible for deletion once all handles are closed.
    """
    global _SHARED_MEMORYS
    if names is None:
        names = list(_SHARED_MEMORYS.keys())

    for name in names:
        shm = _SHARED_MEMORYS.get(name)
        if shm is None:
            continue
        shm.close()
        shm.unlink()
        del _SHARED_MEMORYS[name]

def get_init_parallel_args(global_array_names: list[str]):
    """
    Build metadata needed to attach shared arrays in worker processes.

    Parameters
    ----------
    global_array_names : list of str
        Names of globals (and shared memory segments) to attach.

    Returns
    -------
    list[str]
        Shared memory names (same as global names).
    list[tuple]
        Array shapes.
    list[numpy.dtype]
        Array dtypes.
    """
    names = []
    shapes = []
    dtypes = []
    for name in global_array_names:
        arr = globals()[name]
        if arr is None:
            continue

        names.append(name)
        shapes.append(arr.shape)
        dtypes.append(arr.dtype)

    return names, shapes, dtypes

def init_parallel(
    names: list[str],
    shapes: list[tuple],
    dtypes: list[np.dtype],
    params: dict | None = None,
):
    """
    Worker initializer for multiprocessing.

    Attaches shared memory segments into NumPy arrays and stores them into
    module-level globals so the per-cell worker function can run without
    pickling large arrays.

    Parameters
    ----------
    names, shapes, dtypes
        Metadata produced by :func:`get_init_parallel_args`.
    params : dict, optional
        Simulation parameters to store in a module-level global.
    """
    shms = [shared_memory.SharedMemory(name=name) for name in names]

    for shm, name, shape, dtype in zip(shms, names, shapes, dtypes):
        _set_shared(name, shm)
        globals()[name] = np.ndarray(shape, dtype=dtype, buffer=shm.buf)

    global _PARAMS
    if params is not None:
        _PARAMS = params

def _build_flow_arrays(id_flow_dict: dict,  baseflow_key: str, qmax_key: str, processes: int) -> tuple[np.ndarray, np.ndarray]:
    if baseflow_key == '':
        create_array("_CELL_QBASE", processes, (_CELL_COMIDS.size,), np.float64, fill_value=0.0)
    else:
        create_array("_CELL_QBASE", processes, (_CELL_COMIDS.size,), np.float64)[:] = np.fromiter((id_flow_dict[cid][baseflow_key] for cid in _CELL_COMIDS), dtype=np.float64, count=len(_CELL_COMIDS))
    create_array("_CELL_QMAX", processes, (_CELL_COMIDS.size,), np.float64)[:] = np.fromiter((id_flow_dict[cid][qmax_key] for cid in _CELL_COMIDS), dtype=np.float64, count=len(_CELL_COMIDS))

def _build_reach_slope_arrays(stream_slope_dicts: tuple[dict], params: dict, processes: int) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    method = params['s_stream_slope_method']
    if method in {'reach_average', 'end_points'}:
        slope_dict = stream_slope_dicts[0]
        create_array("_CELL_REACH_SLOPE", processes, (_CELL_COMIDS.size,), np.float64)[:] = np.fromiter((slope_dict[cid] for cid in _CELL_COMIDS), dtype=np.float64, count=len(_CELL_COMIDS))
    if method == 'local_average_corrected':
        slope25_dict = stream_slope_dicts[1]
        slope75_dict = stream_slope_dicts[2]
        create_array("_CELL_SLOPE_25", processes, (_CELL_COMIDS.size,), np.float64)[:] = np.fromiter((slope25_dict[cid] for cid in _CELL_COMIDS), dtype=np.float64, count=len(_CELL_COMIDS))
        create_array("_CELL_SLOPE_75", processes, (_CELL_COMIDS.size,), np.float64)[:] = np.fromiter((slope75_dict[cid] for cid in _CELL_COMIDS), dtype=np.float64, count=len(_CELL_COMIDS))

def run_main_loop(
    num_cells: int,
    params: dict,
    quiet: bool,
    processes: int,
) -> HydraulicData:
    """
    Run the per-cell simulation loop (serial or parallel).

    Parameters
    ----------
    num_cells : int
        Number of stream cells to process (length of the extracted cell lists).
    params : dict
        Simulation parameters produced by :func:`read_main_input_file`.
    quiet : bool
        If True, suppress progress bars.
    processes : int
        Number of worker processes. ``1`` runs serially.

    Returns
    -------
    HydraulicData
        An instance bound to the shared output array and containing any
        requested cross-section export data.
    """
    want_xs = bool(params.get('s_xs_output_file'))
    cross_section_data: list | None = [] if want_xs else None

    LOG.info('Looking at ' + str(num_cells) + ' stream cells')

    if processes == 1:
        for i_entry_cell in tqdm.tqdm(range(num_cells), total=num_cells, disable=quiet):
            item = calculate_hydraulic_data_for_cell(i_entry_cell)
            if cross_section_data is not None and item is not None:
                cross_section_data.append(item)

        hydraulic_data = get_hydraulic_data(params)
        if cross_section_data is not None:
            hydraulic_data.add_cross_section_data(cross_section_data)
        return hydraulic_data
    
    args = get_init_parallel_args(ARRAY_NAMES)

    with Pool(processes=processes, initializer=init_parallel, initargs=(*args, params)) as pool:
        chunksize = min(1_000, num_cells // (processes * 4) + 1)
        for item in tqdm.tqdm(pool.imap(calculate_hydraulic_data_for_cell, range(num_cells), chunksize=chunksize), total=num_cells, disable=quiet):
            if cross_section_data is not None and item is not None:
                cross_section_data.append(item)

    hydraulic_data = get_hydraulic_data(params)
    if cross_section_data is not None:
        hydraulic_data.add_cross_section_data(cross_section_data)
    return hydraulic_data


def handle_processes(processes: int | Literal["auto"], s_input_stream_path: str) -> int:
    """
    Resolve the desired number of worker processes.

    Parameters
    ----------
    processes : int or {"auto"}
        If an integer, values ``< 1`` map to ``os.cpu_count() - 1``. If
        ``"auto"``, ARC chooses serial vs. parallel based on a heuristic using
        the stream raster size.
    s_input_stream_path : str
        Path to the stream raster (used for the heuristic when ``processes="auto"``).

    Returns
    -------
    int
        Number of worker processes to use.
    """
    if isinstance(processes, int):
        if processes < 1:
            return max(os.cpu_count() - 1, 1)
        return processes
    
    if isinstance(processes, str):
        if not processes == "auto":
            raise ValueError(f"Invalid value for processes: {processes}. Must be an integer or 'auto'.")
        
        # Some testing reveals that before 35k stream cells, the overhead of parallel processing outweighs the benefits, so we'll just run serially in those cases
        # To avoid reading it, I note the rough relationship between number of stream cells and the raster size is that number of stream cells is about (RasterXSize * RasterYSize) / 600, so we'll use that to determine whether to run in parallel or not
        ds: gdal.Dataset = gdal.Open(s_input_stream_path)
        if (ds.RasterXSize * ds.RasterYSize) / 600 < 35_000:
            return 1
        
        return max(os.cpu_count() - 1, 1)
        
    raise ValueError(f"Invalid type for processes: {type(processes)}. Must be an integer or 'auto'.")

def create_array(name: str, processes: int, shape: tuple, dtype: np.dtype, fill_value = 0) -> np.ndarray:
    """
    Allocate an array either in-process or in shared memory.

    Parameters
    ----------
    name : str
        Global name to assign, and (when parallel) the shared-memory segment name.
    processes : int
        Number of worker processes. If ``processes == 1``, allocates a normal
        NumPy array. Otherwise allocates a :mod:`multiprocessing.shared_memory`
        backed array.
    shape : tuple
        Array shape.
    dtype : numpy.dtype
        Array dtype.
    fill_value : scalar, optional
        Initial fill value for the array.

    Returns
    -------
    numpy.ndarray
        The allocated array.
    """
    dtype = np.dtype(dtype)
    if processes == 1:
        arr = np.full(
            shape, 
            fill_value, 
            dtype=dtype
        )
        globals()[name] = arr
        return arr
    
    size = int(dtype.itemsize * np.prod(shape))
    shm = shared_memory.SharedMemory(name=name, create=True, size=size)
    arr = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
    arr.fill(fill_value)
    _set_shared(name, shm)
    globals()[name] = arr
    return arr

def _main(MIF_Name: str, args: dict, quiet: bool = False, processes: int | Literal["auto"] = 1):
    """
    Internal driver for ARC.

    This function performs the end-to-end workflow: parse inputs, resolve the
    process count, read and pad rasters, allocate shared arrays (if requested),
    precompute cross-section ordinate indices, run the per-cell computation
    loop, and write output files.

    Parameters
    ----------
    MIF_Name : str
        Path to the ARC model input file (MIF).
    args : dict
        Parameter overrides (keys match the MIF parameter strings).
    quiet : bool, optional
        If True, suppress progress bars and most log output.
    processes : int or {"auto"}, optional
        Number of worker processes.

    Returns
    -------
    None
        Outputs are written to disk based on configured paths.
    """
    starttime = datetime.now()  
    params = read_main_input_file(MIF_Name, args)
    processes = handle_processes(processes, params['s_input_stream_path'])
    if processes > 1:
        LOG.info(f'Using {processes} processes for computation.')

    ### Read Main Input File ###
    
    ### Read the Flow Information ###
    id_flow_dict = read_flow_file(params['s_input_flow_file_path'], params['s_flow_file_id'], params['s_flow_file_baseflow'], params['s_flow_file_qmax'])

    ### Read Raster Data ###
    ### Imbed the Stream and DEM data within a larger Raster to help with the boundary issues. ###
    i_boundary_number = max(1, params['i_general_direction_distance'], params['i_general_slope_distance'])
    dm_elevation, dncols, dnrows, dcellsize, dyll, dyur, dxll, dxur, dlat, dem_geotransform, dem_projection, dem_maxx, dem_miny, dem_dy = read_and_pad_and_maybe_make_shared(params['s_input_dem_path'], processes, i_boundary_number, np.float32, "_DEM")
    dm_stream, sncols, snrows, scellsize, syll, syur, sxll, sxur, slat, strm_geotransform, strm_projection, maxx, miny, dy = read_and_pad_and_maybe_make_shared(params['s_input_stream_path'], processes, i_boundary_number, np.int64, "_STREAMS")
    dm_land_use, lncols, lnrows, lcellsize, lyll, lyur, lxll, lxur, llat, land_geotransform, land_projection, maxx, miny, dy = read_and_pad_and_maybe_make_shared(params['s_input_land_use_path'], processes, i_boundary_number, np.uint8, "_LAND_COVER")

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
    b_modified_dem = add_100_if_elevation_less_than_0(dm_elevation)

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

    ##### Begin Calculations #####
    # Create output rasters
    _BATHYMETRY = create_array("_BATHYMETRY", processes, (nrows + i_boundary_number * 2, ncols + i_boundary_number * 2), np.float32, fill_value=np.nan)
    if params['s_output_flood']:
        create_array("_OUT_FLOOD", processes, (nrows + i_boundary_number * 2, ncols + i_boundary_number * 2), np.uint8)

    # Get the list of stream locations
    flow_ids = np.fromiter(id_flow_dict.keys(), count=len(id_flow_dict), dtype=np.int64)
    ia_valued_row_indices, ia_valued_column_indices = np.where(np.isin(dm_stream, flow_ids, kind='table'))
    for arr, name in zip([ia_valued_row_indices, ia_valued_column_indices], ["_CELL_ROWS", "_CELL_COLS"]):
        create_array(name, processes, arr.shape, arr.dtype)[:] = arr[:]

    # This array will hold all the data for each stream cell. The first 8 columns are 'COMID', 'Row', 'Col', 'DEM_Elev', 'QBaseflow', 'Slope', 'XS_Angle', 'BaseElev', and then we have 5 columns repeated for each increment with 'q', 'v', 't', 'wse', 'p'. 
    create_array("_OUTPUT_DATA_ARRAY", processes, (len(ia_valued_row_indices), 8 + params['i_number_of_increments']*5), np.float64, fill_value=np.nan)

    # Get the cell dx and dy coordinates
    dx, dy, dproject = convert_cell_size(dcellsize, dem_dy, dyll, dyur, dem_projection)
    LOG.info('Cellsize X = ' + str(dx))
    LOG.info('Cellsize Y = ' + str(dy))

    # create a reach average slope before we go stream cell by stream cell
    stream_slope_dicts = initialize_stream_slope_dictionaries(params, dx, dy, dem_geotransform, dem_projection, quiet, processes)

    create_array("_CELL_COMIDS", processes, (ia_valued_row_indices.size,), np.int64)[:] = dm_stream[ia_valued_row_indices, ia_valued_column_indices]
    _build_flow_arrays(id_flow_dict, params['s_flow_file_baseflow'], params['s_flow_file_qmax'], processes)
    _build_reach_slope_arrays(stream_slope_dicts, params, processes)
    
    # Make all Land Cover that is a stream look like water
    i_lc_water_value = params['i_lc_water_value']
    dm_land_use[ia_valued_row_indices,ia_valued_column_indices] = i_lc_water_value
    
    ### Read in the Manning Table ###
    read_manning_table(params['s_input_mannings_path'], dm_land_use, processes)

    # Add params to global variable for use in parallel processing
    params["dx"] = dx
    params["dy"] = dy
    params["i_boundary_number"] = i_boundary_number
    params["nrows"] = nrows
    params["ncols"] = ncols
    params["b_modified_dem"] = b_modified_dem
    global _PARAMS
    _PARAMS = params

    # Create index arrays
    for arr, name in zip(CrossSection.create_cross_section_ordinates(params), ["_INDEX_ARRAYS", "_Z_DISTANCE_ARRAY", "_INDEX_FRACT_ARRAYS"]):
        global_arr = create_array(name, processes, arr.shape, arr.dtype)
        global_arr[:] = arr[:]

    # Extract some parameters
    b_bathy_use_banks = params['b_bathy_use_banks']
    s_output_bathymetry_path = params['s_output_bathymetry_path']

    ### Begin the stream cell solution loop ###
    hydraulic_data = run_main_loop(len(ia_valued_row_indices), params, quiet, processes)

    # Create the output VDT Database file - datatypes are figured out automatically
    if not hydraulic_data.has_vdt_data():
        LOG.warning('No VDT data was generated, so no hydraulic output files will be created.')
        return
    
    # At this point, release all memory except for bathymetry, output array, and elevation
    close_shared_arrays([name for name in ARRAY_NAMES if name not in {"_BATHYMETRY", "_OUTPUT_DATA_ARRAY", "_DEM"}])
    
    hydraulic_data.save_files(id_flow_dict, params['s_flow_file_qmax'])

    # Write the output rasters
    if len(s_output_bathymetry_path) > 1:
        #Make sure all the bathymetry points are above the DEM elevation
        if not b_bathy_use_banks:
            _BATHYMETRY = np.where(_BATHYMETRY>dm_elevation, np.nan, _BATHYMETRY)
        # remove the increase in elevation, if negative elevations were present
        if b_modified_dem:
            # Subtract 100 only for cells that are not NaN
            _BATHYMETRY[~np.isnan(_BATHYMETRY)] -= 100
        # # Joseph was testing a simple smoothing algorithm here to attempt to reduce variation in the bank based bathmetry (functions but doesn't provide better results)
        # if b_bathy_use_banks:
        #     dm_output_bathymetry = smooth_bathymetry_gaussian_numba(dm_output_bathymetry)
        write_output_raster(s_output_bathymetry_path, _BATHYMETRY[i_boundary_number:nrows + i_boundary_number, i_boundary_number:ncols + i_boundary_number], ncols, nrows, dem_geotransform, dem_projection, "GTiff", gdal.GDT_Float32)

    if len(_PARAMS['s_output_flood']) > 1:
        write_output_raster(_PARAMS['s_output_flood'], _OUT_FLOOD[i_boundary_number:nrows + i_boundary_number, i_boundary_number:ncols + i_boundary_number], ncols, nrows, dem_geotransform, dem_projection, "GTiff", gdal.GDT_Byte)
        
    # Log the compute time
    d_sim_time = datetime.now() - starttime
    i_sim_time_s = int(d_sim_time.seconds)

    if i_sim_time_s < 60:
        LOG.info('Simulation Took ' + str(i_sim_time_s) + ' seconds')
    else:
        LOG.info('Simulation Took ' + str(int(i_sim_time_s / 60)) + ' minutes and ' + str(i_sim_time_s - (int(i_sim_time_s / 60) * 60)) + ' seconds')
        
def main(MIF_Name: str, args: dict, quiet: bool = False, processes: int | Literal["auto"] = 1):
    """
    Public entry point for ARC simulations.

    This wrapper calls :func:`_main` and ensures that shared-memory resources are
    cleaned up if an exception occurs.

    Parameters
    ----------
    MIF_Name : str
        Path to the ARC model input file (MIF).
    args : dict
        Parameter overrides (keys match the MIF parameter strings).
    quiet : bool, optional
        If True, suppress progress bars and most log output.
    processes : int or {"auto"}, optional
        Number of worker processes.
    """
    try:
        return _main(MIF_Name, args, quiet, processes)
    except Exception as e:
        LOG.error(f"An error occurred during processing: {e}")
        raise
    finally:
        close_shared_arrays()
        reset_globals()

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
        
    main(MIF_Name, {}, quiet=False, processes=1)
