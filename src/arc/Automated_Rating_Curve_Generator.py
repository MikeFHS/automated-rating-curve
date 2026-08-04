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

import ast
import json
import sys
import os
import math
import warnings
from typing import Literal

import tqdm
import yaml
import numpy as np
import pandas as pd
import networkx as nx
from datetime import datetime
import geopandas as gpd
from scipy.optimize import OptimizeWarning, brentq
from scipy.signal import find_peaks
from shapely.geometry import LineString, MultiLineString
from osgeo import gdal
from pyproj import CRS, Geod
from numba import njit, vectorize
from multiprocessing import Pool, shared_memory

from arc import LOG
from arc.cross_section import CrossSection, calc_bankfull_elevation, calculate_discharge_from_wse, _calculate_all
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
_CELL_SOURCE_STREAM_IDS: np.ndarray = None
_CELL_QBASE: np.ndarray = None
_CELL_QMAX: np.ndarray = None
_CELL_BATHY_DEPTH: np.ndarray = None
_CELL_BATHY_WIDTH: np.ndarray = None
_CELL_REACH_SLOPE: np.ndarray = None
_CELL_SLOPE_25: np.ndarray = None
_CELL_SLOPE_75: np.ndarray = None
_CELL_REACH_INFLECT_BANK_INDEX: np.ndarray = None
_CELL_REACH_INFLECT_TERRACE_INDEX: np.ndarray = None
_MANUAL_CROSS_SECTION_RECORDS: dict[int, dict] | None = None
_PRECOMPUTED_CROSS_SECTION_RECORDS: list[dict | None] | None = None

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
    '_CELL_SOURCE_STREAM_IDS',
    '_CELL_QBASE',
    '_CELL_QMAX',
    '_CELL_BATHY_DEPTH',
    '_CELL_BATHY_WIDTH',
    '_CELL_REACH_SLOPE',
    '_CELL_SLOPE_25',
    '_CELL_SLOPE_75',
    '_CELL_REACH_INFLECT_BANK_INDEX',
    '_CELL_REACH_INFLECT_TERRACE_INDEX',
]

MIN_SLOPE = 1e-8
MIN_SLOPE_DECIMAL_PLACES = -int(math.log10(MIN_SLOPE))
DEPTH_INCREMENT_BIG = 0.5
DEPTH_INCREMENT_MEDIUM = 0.05
DEPTH_INCREMENT_SMALL = 0.01

# Temporary diagnostic output. When enabled, ARC saves one reach-average
# INFLECT curve plot per analyzed reach while building the reach-scale bank
# indices used by the bathymetry workflow.
TEMP_PLOT_REACH_INFLECT_CURVES = False
TEMP_REACH_INFLECT_PLOT_SUBDIRECTORY = 'Reach_Inflect_Curve_Plots'
REACH_BANK_ELEVATION_SMOOTHING_WINDOW = 20

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
        _HYDRAULIC_DATA.associate_with_reach_inflect_terrace_index(_CELL_REACH_INFLECT_TERRACE_INDEX)
    return _HYDRAULIC_DATA

def _get_reach_inflect_plot_directory(params: dict) -> str:
    """Return the directory where temporary reach INFLECT plots should be saved.

    The plots are diagnostic artifacts, so ARC keeps them next to the most
    relevant user-facing output when possible. Representative cross-section
    runs take priority because those workflows also depend on reach-scale
    INFLECT analysis. If that file is not configured, ARC falls back to the
    bathymetry raster directory and finally to the current working directory.
    """
    candidate_output = (
        params.get('s_representative_cross_section_file')
        or params.get('s_output_bathymetry_path')
        or params.get('s_xs_output_file')
        or ''
    )
    base_dir = os.path.dirname(candidate_output) if candidate_output else os.getcwd()
    return os.path.join(base_dir, TEMP_REACH_INFLECT_PLOT_SUBDIRECTORY)

def _plot_reach_average_inflect_curve(
    reach_id: int,
    depth_values: np.ndarray,
    mean_curve: np.ndarray,
    bank_index: int,
    terrace_index: int,
    plot_directory: str,
) -> None:
    """Write a temporary PNG of the reach-average INFLECT curve.

    Parameters
    ----------
    reach_id : int
        Stream-reach identifier used to label the output figure.
    depth_values : numpy.ndarray
        Depth axis aligned to ``mean_curve``. This is built from the actual
        staged bank-height-driven depth samples used by the sampled cross
        sections in the reach rather than from the older fixed 0.10 m spacing.
    mean_curve : numpy.ndarray
        Reach-average ``d2W/dy2`` curve computed from the sampled cross
        sections belonging to the reach.
    bank_index : int
        Index of the maximum-inflection point currently used to define the
        reach-scale bank depth.
    terrace_index : int
        Index of the negative-inflection point currently used to define the
        reach-scale terrace depth.
    plot_directory : str
        Destination directory for the diagnostic PNG.
    """
    if mean_curve.size == 0 or depth_values.size == 0:
        return

    try:
        import matplotlib.pyplot as plt
    except Exception as ex:
        LOG.warning(f'Unable to create temporary reach INFLECT plots because matplotlib is unavailable: {ex}')
        return

    os.makedirs(plot_directory, exist_ok=True)
    curve_length = min(mean_curve.size, depth_values.size)
    mean_curve = np.asarray(mean_curve[:curve_length], dtype=np.float64)
    depth_values = np.asarray(depth_values[:curve_length], dtype=np.float64)
    bank_index = int(min(max(bank_index, 0), mean_curve.size - 1))
    terrace_index = int(min(max(terrace_index, 0), mean_curve.size - 1))
    bank_depth = depth_values[bank_index]
    terrace_depth = depth_values[terrace_index]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(depth_values, mean_curve, color='steelblue', linewidth=2.0)
    ax.scatter([bank_depth], [mean_curve[bank_index]], color='crimson', zorder=3, label='Bank Index')
    ax.axvline(bank_depth, color='crimson', linestyle='--', linewidth=1.0)
    ax.scatter([terrace_depth], [mean_curve[terrace_index]], color='darkgreen', zorder=3, label='Terrace Index')
    ax.axvline(terrace_depth, color='darkgreen', linestyle=':', linewidth=1.0)
    ax.set_title(f'Reach {reach_id} Mean INFLECT Curve')
    ax.set_xlabel('Depth Above Thalweg (m)')
    ax.set_ylabel('Mean d2W/dy2')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')

    output_path = os.path.join(plot_directory, f'reach_{reach_id}_inflect_curve.png')
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

def _set_shared(name: str, shm: shared_memory.SharedMemory):
    """
    We need the shared memory objects to persist somewhere; otherwise, the memory is freed and the numpy arrays point to invalid memory.
    These must last the lifetime of the program! Reason being, bathymetry is the last thing written, and we need the shared memory to persist until then.
    """
    global _SHARED_MEMORYS
    _SHARED_MEMORYS[name] = shm

def reset_globals():
    for name in ARRAY_NAMES + ['_CROSS_SECTION', '_HYDRAULIC_DATA', '_MANUAL_CROSS_SECTION_RECORDS', '_PRECOMPUTED_CROSS_SECTION_RECORDS']:
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

def line_slope_from_dem(line_geom: LineString, dm_elevation: np.ndarray, dem_geotransform, length_m, pad_distance: int = 0):
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
    pad_distance : int, optional
        Number of cells of zero padding ARC added around the DEM array. The
        geotransform still describes the original unpadded raster, so sampled
        row/column indices must be shifted by this amount before indexing the
        padded DEM array.

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
    nrows_padded, ncols_padded = dm_elevation.shape
    nrows = nrows_padded - 2 * pad_distance
    ncols = ncols_padded - 2 * pad_distance

    def xy_to_rowcol(x, y):
        """
        Convert map coordinates (x,y) to DEM row/col indices.
        Assumes a north-up grid (gt2 == gt4 == 0).

        The GDAL geotransform describes the original raster, not ARC's padded
        in-memory array. This helper therefore computes indices in the original
        DEM space first, validates them against the unpadded bounds, and only
        then shifts them into the padded array.
        """
        # column: straightforward with positive pixel width
        col = int((x - gt0) / gt1)

        # row: geotransform[5] is typically negative for north-up rasters
        if gt5 < 0:
            row = int((gt3 - y) / abs(gt5))
        else:
            row = int((y - gt3) / gt5)

        # Clip against the original, unpadded raster extent described by the
        # geotransform. Coordinates outside that extent should not sample the
        # artificial zero border that ARC added for neighborhood operations.
        if row < 0 or row >= nrows or col < 0 or col >= ncols:
            return None
        return row + pad_distance, col + pad_distance

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


def _get_optional_parameter_value(sl_lines: list[str], primary_key: str, secondary_key: str = ''):
    """Read an optional MIF/YAML parameter while supporting a legacy alias.

    ARC historically uses mixed-case parameter names in its MIF examples, while
    this bathymetry enhancement introduces lower-case keys requested by the
    user. This helper lets the parser accept either spelling without forcing the
    rest of the code to care which form the input file used.

    Parameters
    ----------
    sl_lines : list[str]
        Parsed input-file lines in ARC's internal ``key<TAB>value`` form.
    primary_key : str
        Preferred parameter name to read first.
    secondary_key : str, optional
        Backwards-compatible alias to try when the preferred key is not
        present.

    Returns
    -------
    object
        The parsed value returned by :func:`get_parameter_name`, or an empty
        string when neither key was provided.
    """
    value = get_parameter_name(sl_lines, primary_key)
    if value == '' and secondary_key:
        value = get_parameter_name(sl_lines, secondary_key)
    return value


def _parse_optional_float_parameter(value, parameter_name: str) -> float | None:
    """Convert an optional ARC parameter into ``float`` or ``None``.

    The MIF parser returns ``''`` when a parameter is absent. Treating that as
    ``None`` here keeps the validation logic explicit and avoids scattering
    string checks throughout the bathymetry code.

    Parameters
    ----------
    value : object
        Raw parameter value from the MIF/YAML parser.
    parameter_name : str
        Human-readable name used in validation error messages.

    Returns
    -------
    float or None
        Parsed floating-point value, or ``None`` when the parameter was not
        supplied.
    """
    if value in ('', None):
        return None
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{parameter_name} must be numeric when provided.") from exc


def _build_bathymetry_powerlaw_config(sl_lines: list[str], s_strmshp_path: str) -> dict:
    """Parse and validate the optional drainage-area bathymetry parameters.

    The new bathymetry mode replaces the discharge-driven depth estimate with
    drainage-area power laws:

    ``depth = coefficient_depth * drainage_area ** exponent_depth``
    ``width = coefficient_width * drainage_area ** exponent_width``

    The width relationship is intentionally a fallback-only tool. ARC still
    prefers to detect banks directly from land cover / DEM evidence, and only
    uses the estimated width if those searches fail.

    Parameters
    ----------
    sl_lines : list[str]
        Parsed input-file lines in ARC's internal ``key<TAB>value`` form.
    s_strmshp_path : str
        Path to the stream vector dataset. Required because the drainage-area
        attribute must be read from this file.

    Returns
    -------
    dict
        Normalized configuration describing whether the power-law mode is fully
        configured, plus the parsed field names and coefficients.
    """
    drainage_area_field = _get_optional_parameter_value(
        sl_lines, 'drainage_area_field', 'Drainage_Area_Field'
    )
    coefficient_depth = _parse_optional_float_parameter(
        _get_optional_parameter_value(sl_lines, 'coefficient_depth', 'Coefficient_Depth'),
        'coefficient_depth',
    )
    exponent_depth = _parse_optional_float_parameter(
        _get_optional_parameter_value(sl_lines, 'exponent_depth', 'Exponent_Depth'),
        'exponent_depth',
    )
    coefficient_width = _parse_optional_float_parameter(
        _get_optional_parameter_value(sl_lines, 'coefficient_width', 'Coefficient_Width'),
        'coefficient_width',
    )
    exponent_width = _parse_optional_float_parameter(
        _get_optional_parameter_value(sl_lines, 'exponent_width', 'Exponent_Width'),
        'exponent_width',
    )

    provided_flags = {
        'drainage_area_field': drainage_area_field != '',
        'coefficient_depth': coefficient_depth is not None,
        'exponent_depth': exponent_depth is not None,
        'coefficient_width': coefficient_width is not None,
        'exponent_width': exponent_width is not None,
    }
    any_provided = any(provided_flags.values())
    all_provided = all(provided_flags.values())

    if any_provided and not s_strmshp_path:
        raise ValueError(
            "StrmShp_File is required when configuring drainage-area bathymetry "
            "parameters because ARC reads the drainage area attribute from that dataset."
        )

    if any_provided and not all_provided:
        missing = [name for name, supplied in provided_flags.items() if not supplied]
        raise ValueError(
            "The drainage-area bathymetry mode requires all five optional "
            "parameters together. Missing: " + ", ".join(missing)
        )

    return {
        'enabled': all_provided,
        'drainage_area_field': drainage_area_field if all_provided else '',
        'coefficient_depth': coefficient_depth,
        'exponent_depth': exponent_depth,
        'coefficient_width': coefficient_width,
        'exponent_width': exponent_width,
    }

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
    b_build_representative_cross_section = to_bool(
        get_parameter_name(sl_lines, 'Build_Representative_Cross_Section', False)
    )
    s_representative_cross_section_file = get_parameter_name(
        sl_lines,
        'Representative_Cross_Section_File',
    )
    s_reach_id_field = _get_optional_parameter_value(
        sl_lines,
        'reach_id',
        'Reach_ID',
    )
    s_downstream_reach_id_field = _get_optional_parameter_value(
        sl_lines,
        'downstream_reach_id',
        'Downstream_Reach_ID',
    )
    if b_build_representative_cross_section and not s_representative_cross_section_file:
        raise ValueError(
            'Build_Representative_Cross_Section requires Representative_Cross_Section_File.'
        )

    # Bathymetry can now be driven by either a discharge column (legacy path) or
    # a drainage-area power law (new optional path). Parse the power-law
    # configuration first so the baseflow validation below can decide whether
    # omitting Flow_File_BF is intentional.
    bathymetry_powerlaw = _build_bathymetry_powerlaw_config(sl_lines, s_strmshp_path)

    # check for baseflow parameters for bathymetry estimation. If not provided, disable bathymetry estimation.
    s_flow_file_baseflow = get_parameter_name(sl_lines,  'Flow_File_BF')
    s_flow_file_qmax = get_parameter_name(sl_lines,  'Flow_File_QMax')
    s_output_bathymetry_path = get_parameter_name(sl_lines,  'AROutBATHY', get_parameter_name(sl_lines,  'BATHY_Out_File'))
    if s_flow_file_baseflow == '' and len(s_output_bathymetry_path) > 1 and not bathymetry_powerlaw['enabled']:
        LOG.warning(
            'Flow_File_BF was not provided and the drainage-area bathymetry '
            'parameters were not fully configured; disabling bathymetry estimation.'
        )
        s_output_bathymetry_path = ''
    if len(s_output_bathymetry_path) > 1:
        if not s_reach_id_field or not s_downstream_reach_id_field:
            raise ValueError(
                'Bathymetry output requires both reach_id and downstream_reach_id '
                'to be provided in the input file.'
            )

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
        'b_use_bathymetry_powerlaw': bathymetry_powerlaw['enabled'],
        's_bathymetry_drainage_area_field': bathymetry_powerlaw['drainage_area_field'],
        'd_bathymetry_coefficient_depth': bathymetry_powerlaw['coefficient_depth'],
        'd_bathymetry_exponent_depth': bathymetry_powerlaw['exponent_depth'],
        'd_bathymetry_coefficient_width': bathymetry_powerlaw['coefficient_width'],
        'd_bathymetry_exponent_width': bathymetry_powerlaw['exponent_width'],
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
        'b_build_representative_cross_section': b_build_representative_cross_section,
        's_representative_cross_section_file': s_representative_cross_section_file,
        's_reach_id_field': s_reach_id_field,
        's_downstream_reach_id_field': s_downstream_reach_id_field,
        's_manual_cross_section_file': get_parameter_name(sl_lines, 'Manual_Cross_Sections_File'),
        'i_lc_water_value': int(get_parameter_name(sl_lines,  'LC_Water_Value', 80)), # Find the value in the land cover dataset that corresponds to water. This is used to find the banks of the river if b_FindBanksBasedOnLandCover is set to True
        'i_number_of_increments': int(get_parameter_name(sl_lines,  'VDT_Database_NumIterations', 15)), # Find the number of increments to use in the velocity, depth, and top width database
        'b_FindBanksBasedOnLandCover': b_FindBanksBasedOnLandCover, # Find the true/false variable to find the banks of the river based on the land cover dataset instead of the DEM
        'b_reach_average_curve_file': b_reach_average_curve_file, # Find the true/false variable to use a reach-average curve file
        's_output_flood': get_parameter_name(sl_lines,  'AROutFLOOD'), # Find the path to the output flood file

    }

    return params


def _power_law_geometry_from_drainage_area(
    drainage_area: float,
    coefficient_depth: float,
    exponent_depth: float,
    coefficient_width: float,
    exponent_width: float,
) -> tuple[float, float]:
    """Estimate bathymetry target depth and fallback width from drainage area.

    Parameters
    ----------
    drainage_area : float
        Drainage-area attribute value read from the stream vector dataset.
    coefficient_depth, exponent_depth : float
        Power-law parameters used to estimate bankfull depth.
    coefficient_width, exponent_width : float
        Power-law parameters used to estimate bankfull width.

    Returns
    -------
    tuple[float, float]
        ``(estimated_depth, estimated_width)``.
    """
    estimated_depth = coefficient_depth * (drainage_area ** exponent_depth)
    estimated_width = coefficient_width * (drainage_area ** exponent_width)
    return float(estimated_depth), float(estimated_width)


def build_bathymetry_geometry_dict(
    s_strmshp_path: str,
    s_flow_file_id: str,
    drainage_area_field: str,
    coefficient_depth: float,
    exponent_depth: float,
    coefficient_width: float,
    exponent_width: float,
) -> dict[int, dict[str, float]]:
    """Read stream attributes and convert them into per-reach bathymetry targets.

    The resulting dictionary lets the per-cell compute loop operate entirely on
    numeric arrays. All vector I/O is completed once up front, which keeps the
    hot loop simple and multiprocessing-friendly.

    Parameters
    ----------
    s_strmshp_path : str
        Path to the vector dataset identified by ``StrmShp_File``.
    s_flow_file_id : str
        Reach identifier field shared by the flow file and stream vector.
    drainage_area_field : str
        Field in ``s_strmshp_path`` containing drainage area values.
    coefficient_depth, exponent_depth, coefficient_width, exponent_width : float
        Power-law coefficients and exponents used to estimate target geometry.

    Returns
    -------
    dict[int, dict[str, float]]
        Mapping ``reach_id -> {"depth": depth, "width": width}``.
    """
    gdf_stream = gpd.read_file(s_strmshp_path)
    required_columns = {s_flow_file_id, drainage_area_field}
    missing_columns = sorted(required_columns.difference(gdf_stream.columns))
    if missing_columns:
        raise KeyError(
            "The stream vector dataset is missing the columns required for "
            "drainage-area bathymetry: " + ", ".join(missing_columns)
        )

    attribute_df = gdf_stream[[s_flow_file_id, drainage_area_field]].copy()
    attribute_df = attribute_df.dropna(subset=[s_flow_file_id, drainage_area_field])
    attribute_df[s_flow_file_id] = pd.to_numeric(attribute_df[s_flow_file_id], errors='raise').astype(np.int64)
    attribute_df[drainage_area_field] = pd.to_numeric(attribute_df[drainage_area_field], errors='raise')

    bathymetry_geometry: dict[int, dict[str, float]] = {}
    for reach_id, group in attribute_df.groupby(s_flow_file_id, sort=False):
        drainage_area = float(group[drainage_area_field].iloc[0])
        if not np.isfinite(drainage_area) or drainage_area <= 0.0:
            raise ValueError(
                f"Drainage area for reach {reach_id} must be positive in field "
                f"{drainage_area_field}."
            )

        estimated_depth, estimated_width = _power_law_geometry_from_drainage_area(
            drainage_area,
            coefficient_depth,
            exponent_depth,
            coefficient_width,
            exponent_width,
        )
        if not np.isfinite(estimated_depth) or estimated_depth <= 0.0:
            raise ValueError(
                f"Estimated depth for reach {reach_id} was not positive. "
                "Check coefficient_depth, exponent_depth, and the drainage area values."
            )
        if not np.isfinite(estimated_width) or estimated_width <= 0.0:
            raise ValueError(
                f"Estimated width for reach {reach_id} was not positive. "
                "Check coefficient_width, exponent_width, and the drainage area values."
            )

        bathymetry_geometry[int(reach_id)] = {
            'depth': estimated_depth,
            'width': estimated_width,
        }

    return bathymetry_geometry

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


def _parse_manual_cross_section_array(value, dtype=float) -> np.ndarray:
    """Parse one serialized manual cross-section array from the input file."""
    if isinstance(value, np.ndarray):
        return value.astype(dtype, copy=False)
    if value in (None, "", "[]"):
        return np.array([], dtype=dtype)
    if isinstance(value, (list, tuple)):
        return np.asarray(value, dtype=dtype)

    try:
        parsed = json.loads(value)
    except Exception:
        parsed = ast.literal_eval(value)

    return np.asarray(parsed, dtype=dtype)


def load_manual_cross_section_records(
    manual_cross_section_file: str,
    manual_id_field: str,
    i_boundary_number: int,
) -> tuple[dict[int, dict], float]:
    """Load manual ARC cross sections and convert them to padded-grid indices.

    Parameters
    ----------
    manual_cross_section_file : str
        Path to the manual cross-section table written by gap-crossing.
    manual_id_field : str
        Column that should match ``Flow_File_ID`` in the ARC flow table.
    i_boundary_number : int
        Padding offset used by ARC's internal raster arrays.

    Returns
    -------
    tuple
        ``(records, required_x_section_distance)`` where ``records`` maps the
        manual ID to the parsed cross-section data, and the distance term is the
        minimum ``X_Section_Dist`` needed to hold the longest supplied profile.
    """
    if manual_cross_section_file.endswith(".parquet"):
        manual_df = pd.read_parquet(manual_cross_section_file)
    else:
        separator = "\t" if manual_cross_section_file.lower().endswith((".tsv", ".txt")) else ","
        manual_df = pd.read_csv(manual_cross_section_file, sep=separator)

    required_columns = {
        manual_id_field,
        "Row",
        "Col",
        "Ordinate_Dist",
        "XS1_Profile",
        "XS2_Profile",
        "LC1_Profile",
        "LC2_Profile",
        "XS1_Row",
        "XS1_Col",
        "XS2_Row",
        "XS2_Col",
    }
    missing_columns = sorted(required_columns.difference(manual_df.columns))
    if missing_columns:
        raise KeyError(
            "Manual cross-section file is missing required columns: "
            + ", ".join(missing_columns)
        )

    records: dict[int, dict] = {}
    required_x_section_distance = 0.0
    for _, row in manual_df.iterrows():
        manual_id = int(row[manual_id_field])
        xs1_profile = _parse_manual_cross_section_array(row["XS1_Profile"], dtype=np.float64)
        xs2_profile = _parse_manual_cross_section_array(row["XS2_Profile"], dtype=np.float64)
        lc1_profile = _parse_manual_cross_section_array(row["LC1_Profile"], dtype=np.uint8)
        lc2_profile = _parse_manual_cross_section_array(row["LC2_Profile"], dtype=np.uint8)
        xs1_row = _parse_manual_cross_section_array(row["XS1_Row"], dtype=np.int64) + i_boundary_number
        xs1_col = _parse_manual_cross_section_array(row["XS1_Col"], dtype=np.int64) + i_boundary_number
        xs2_row = _parse_manual_cross_section_array(row["XS2_Row"], dtype=np.int64) + i_boundary_number
        xs2_col = _parse_manual_cross_section_array(row["XS2_Col"], dtype=np.int64) + i_boundary_number

        if len(xs1_profile) == 0 or len(xs2_profile) == 0:
            raise ValueError(f"Manual cross section {manual_id} did not contain profile values on both sides.")
        if not (len(xs1_profile) == len(lc1_profile) == len(xs1_row) == len(xs1_col)):
            raise ValueError(f"Manual cross section {manual_id} has mismatched side-1 array lengths.")
        if not (len(xs2_profile) == len(lc2_profile) == len(xs2_row) == len(xs2_col)):
            raise ValueError(f"Manual cross section {manual_id} has mismatched side-2 array lengths.")

        ordinate_dist = float(row["Ordinate_Dist"])
        max_half_distance = max((len(xs1_profile) - 1) * ordinate_dist, (len(xs2_profile) - 1) * ordinate_dist)
        required_x_section_distance = max(required_x_section_distance, 2.0 * max_half_distance)

        source_stream_id = row.get("Source_Stream_ID", manual_id)
        if pd.isna(source_stream_id):
            source_stream_id = manual_id

        records[manual_id] = {
            "manual_id": manual_id,
            "source_stream_id": int(source_stream_id),
            "row": int(row["Row"]) + i_boundary_number,
            "col": int(row["Col"]) + i_boundary_number,
            "xs_angle": float(row.get("XS_Angle", 0.0) or 0.0),
            "ordinate_dist": ordinate_dist,
            "xs1_profile": xs1_profile,
            "xs2_profile": xs2_profile,
            "lc1_profile": lc1_profile,
            "lc2_profile": lc2_profile,
            "xs1_row": xs1_row,
            "xs1_col": xs1_col,
            "xs2_row": xs2_row,
            "xs2_col": xs2_col,
        }

    return records, required_x_section_distance


def apply_manual_cross_section_data(x_section: CrossSection, manual_record: dict) -> None:
    """Populate a :class:`CrossSection` instance from a manual input record."""
    x_section.row = manual_record["row"]
    x_section.col = manual_record["col"]
    x_section.d_xs_direction = manual_record["xs_angle"]
    x_section.d_ordinate_dist = manual_record["ordinate_dist"]
    x_section.xs1_n = len(manual_record["xs1_profile"])
    x_section.xs2_n = len(manual_record["xs2_profile"])
    x_section.i_precompute_angle_closest = 0

    x_section.da_xs_profile1[:] = 0.0
    x_section.da_xs_profile2[:] = 0.0
    x_section.ia_lc_xs1[:] = 0
    x_section.ia_lc_xs2[:] = 0

    x_section.da_xs_profile1[:x_section.xs1_n] = manual_record["xs1_profile"]
    x_section.da_xs_profile2[:x_section.xs2_n] = manual_record["xs2_profile"]
    x_section.ia_lc_xs1[:x_section.xs1_n] = manual_record["lc1_profile"]
    x_section.ia_lc_xs2[:x_section.xs2_n] = manual_record["lc2_profile"]
    x_section.ia_xc_row1_index_main = manual_record["xs1_row"]
    x_section.ia_xc_column1_index_main = manual_record["xs1_col"]
    x_section.ia_xc_row2_index_main = manual_record["xs2_row"]
    x_section.ia_xc_column2_index_main = manual_record["xs2_col"]
    # Mirror the main indices into the "second" arrays because manual sections
    # are already explicitly defined and do not need interpolation offsets.
    x_section.ia_xc_row1_index_second = manual_record["xs1_row"]
    x_section.ia_xc_column1_index_second = manual_record["xs1_col"]
    x_section.ia_xc_row2_index_second = manual_record["xs2_row"]
    x_section.ia_xc_column2_index_second = manual_record["xs2_col"]

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
        d_wse = np.round(thalweg + d_inc_y * i_entry_elevation, 3)

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
            if (Q <= prev_q) or (Q > d_q_sum + 1.0):
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

def dict_stream_slopes_from_endpoints(dm_stream, dem_geotransform, dem_projection, s_strmshp_path, s_flow_file_id, quiet, pad_distance):
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
        slope_pct, slope_deg, z_start, z_end, length_m = line_slope_from_dem(
            StrmSHP_geom.iloc[0],
            _DEM,
            dem_geotransform,
            length_m,
            pad_distance=pad_distance,
        )
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

def initialize_stream_slope_dictionaries(params: dict, dx, dy, dem_geotransform, dem_projection, quiet, processes, i_boundary_number):
    s_stream_slope_method = params['s_stream_slope_method']
    if s_stream_slope_method == 'reach_average' or s_stream_slope_method == 'local_average_corrected':
        dict_stream_slopes, dict_stream_slopes_25th, dict_stream_slopes_75th = create_reach_average_slope_dicts(_STREAMS, dx, dy, quiet, params['i_general_slope_distance'], processes)
        return (dict_stream_slopes, dict_stream_slopes_25th, dict_stream_slopes_75th)
    elif s_stream_slope_method == 'end_points':
        dict_stream_slopes = dict_stream_slopes_from_endpoints(
            _STREAMS,
            dem_geotransform,
            dem_projection,
            params['s_strmshp_path'],
            params['s_flow_file_id'],
            quiet,
            i_boundary_number,
        )
        return (dict_stream_slopes, None, None)
    
    return (None, None, None)


def _get_cell_bathymetry_inputs(i_entry_cell: int, i_row_cell: int, i_column_cell: int, params: dict) -> tuple[float, float, float | None, float | None]:
    """Return the bathymetry-driving inputs for a sampled stream cell.

    The reach-average INFLECT prepass and the main hydraulic loop both need to
    derive bathymetry from the same per-cell baseflow, slope, and optional
    drainage-area power-law targets. Centralizing that logic keeps the
    representative INFLECT prepass consistent with the actual production
    bathymetry workflow.
    """
    d_q_baseflow = _CELL_QBASE[i_entry_cell]
    d_bathy_target_depth = None if _CELL_BATHY_DEPTH is None else _CELL_BATHY_DEPTH[i_entry_cell]
    d_bathy_target_width = None if _CELL_BATHY_WIDTH is None else _CELL_BATHY_WIDTH[i_entry_cell]
    i_general_slope_distance = params['i_general_slope_distance']
    s_stream_slope_method = params['s_stream_slope_method']
    dx = params['dx']
    dy = params['dy']

    if s_stream_slope_method == 'local_average':
        d_slope_use = get_local_average_stream_slope_information(
            i_row_cell,
            i_column_cell,
            _DEM,
            _STREAMS,
            dx,
            dy,
            i_general_slope_distance,
        )
    elif s_stream_slope_method == 'reach_average' or s_stream_slope_method == 'end_points':
        d_slope_use = _CELL_REACH_SLOPE[i_entry_cell]
    elif s_stream_slope_method == 'local_average_corrected':
        d_slope_use = get_local_average_stream_slope_information(
            i_row_cell,
            i_column_cell,
            _DEM,
            _STREAMS,
            dx,
            dy,
            i_general_slope_distance,
        )
        d_slope_25th = _CELL_SLOPE_25[i_entry_cell]
        d_slope_75th = _CELL_SLOPE_75[i_entry_cell]
        if d_slope_use < d_slope_25th:
            d_slope_use = d_slope_25th
        elif d_slope_use > d_slope_75th:
            d_slope_use = d_slope_75th
    else:
        d_slope_use = get_local_average_stream_slope_information(
            i_row_cell,
            i_column_cell,
            _DEM,
            _STREAMS,
            dx,
            dy,
            i_general_slope_distance,
        )

    return d_q_baseflow, d_slope_use, d_bathy_target_depth, d_bathy_target_width


def _replace_slope_with_smoothed_bank_grade(
    d_slope_use: float,
    bank_search_result: dict | None,
) -> float:
    """Replace a cell slope with its network-smoothed bank-surface grade.

    ``_smooth_reach_bank_elevations`` stores one longitudinal grade on every
    staged bank result. Once that result exists, the grade is authoritative for
    bathymetry and hydraulic calculations. A zero grade is raised only to
    ARC's numerical ``MIN_SLOPE`` so square roots and Manning calculations
    remain defined without reintroducing the former 0.001 minimum grade.
    """
    if not isinstance(bank_search_result, dict):
        return float(d_slope_use)

    try:
        smoothed_grade = float(
            bank_search_result.get(
                "network_reach_bank_elevation_grade",
                np.nan,
            )
        )
    except (TypeError, ValueError):
        return float(d_slope_use)
    if not np.isfinite(smoothed_grade) or smoothed_grade < 0.0:
        return float(d_slope_use)
    return float(max(smoothed_grade, MIN_SLOPE))


def _apply_bathymetry_to_cross_section(
    x_section: CrossSection,
    params: dict,
    bank_search_result: dict | None = None,
) -> None:
    """Burn a previously staged bathymetry depth into a cross section.

    Bank finding and hydraulic-depth estimation have already finished before
    this helper is called. The burn methods therefore use the bank geometry and
    ``bathymetry_depth`` stored in ``bank_search_result`` and do not solve
    Manning's equation while mutating the cross-section profiles.
    """
    s_output_bathymetry_path = params['s_output_bathymetry_path']
    if s_output_bathymetry_path == '':
        return

    if not params['b_bathy_use_banks']:
        x_section.Calculate_Bathymetry_Based_on_WSE_or_LC(
            _BATHYMETRY,
            bank_search_result=bank_search_result,
        )
    else:
        x_section.Calculate_Bathymetry_Based_on_RiverBank_Elevations(
            _BATHYMETRY,
            bank_search_result=bank_search_result,
        )


def _stage_cross_section_bathymetry_depths(
    sampled_records: list[dict | None],
    params: dict,
    quiet: bool,
) -> None:
    """Estimate one bathymetry depth per cell after all banks have been found.

    This is deliberately a separate pass over the cached, unmodified cross
    sections. Drainage-area power-law depths are copied directly when present.
    Otherwise, the baseflow and local/reach slope are passed to the hydraulic
    depth solver using the already selected bank geometry. The resulting depth
    is stored with the bank result for the later bathymetry-only burn pass.
    """
    x_section = get_cross_section(params['dx'], params['dy'], _DEM, _LAND_COVER, _STREAMS, params)

    for i_entry_cell in tqdm.tqdm(
        range(_CELL_COMIDS.size),
        total=_CELL_COMIDS.size,
        disable=quiet,
    ):
        sampled_record = sampled_records[i_entry_cell]
        if sampled_record is None:
            continue

        reach_bank_index = None
        if _CELL_REACH_INFLECT_BANK_INDEX is not None:
            reach_bank_index = float(_CELL_REACH_INFLECT_BANK_INDEX[i_entry_cell])
        _replay_precomputed_cross_section(
            x_section,
            sampled_record,
            reach_bank_index=reach_bank_index,
        )

        i_row_cell = int(_CELL_ROWS[i_entry_cell])
        i_column_cell = int(_CELL_COLS[i_entry_cell])
        (
            d_q_baseflow,
            d_slope_use,
            d_bathy_target_depth,
            _d_bathy_target_width,
        ) = _get_cell_bathymetry_inputs(
            i_entry_cell,
            i_row_cell,
            i_column_cell,
            params,
        )

        existing_result = sampled_record.get("bank_search_result")
        staged_result = dict(existing_result) if isinstance(existing_result, dict) else {}
        original_slope_use = float(d_slope_use)
        d_slope_use = _replace_slope_with_smoothed_bank_grade(
            d_slope_use,
            staged_result,
        )
        # Retain both values so output diagnostics show when the smoothed bank
        # surface replaced the raster/flowline-derived slope.
        staged_result["bathymetry_depth_original_slope"] = original_slope_use
        staged_result["bathymetry_depth_smoothed_bank_slope"] = float(
            d_slope_use
        )
        using_target_depth = x_section._is_valid_bathymetry_target(
            d_bathy_target_depth
        )

        if using_target_depth:
            bathymetry_depth = float(d_bathy_target_depth)
            depth_source = "drainage_area_power_law"
        else:
            bathymetry_depth = x_section.calculate_hydraulic_bathymetry_depth(
                d_q_baseflow,
                d_slope_use,
                staged_result,
            )
            depth_source = "baseflow_manning"
            staged_result["hydraulic_bathymetry_depth"] = float(bathymetry_depth)

        # These fields make the depth decision explicit and allow the burn pass
        # to operate without receiving discharge, slope, or target parameters.
        staged_result["bathymetry_depth"] = float(bathymetry_depth)
        # Preserve the former in-method gate exactly: a target depth can be
        # applied without baseflow, while a hydraulically solved depth is only
        # applied when baseflow is positive. This flag also distinguishes a
        # legitimate computed zero from the no-baseflow early-return case.
        staged_result["bathymetry_should_apply"] = bool(
            using_target_depth or d_q_baseflow > 0.0
        )
        staged_result["bathymetry_depth_source"] = depth_source
        staged_result["bathymetry_depth_baseflow"] = float(d_q_baseflow)
        staged_result["bathymetry_depth_slope"] = float(d_slope_use)
        sampled_record["bank_search_result"] = staged_result

    _smooth_reach_bathymetry_depths(sampled_records, params)


def _compute_filtered_reach_median_depths(
    sampled_records: list[dict | None],
    source_reach_ids: np.ndarray,
) -> tuple[dict[int, float], dict[int, dict]]:
    """Calculate an interquartile-filtered median depth for each reach."""
    grouped_depths: dict[int, list[float]] = {}
    for entry_index, sampled_record in enumerate(sampled_records):
        if sampled_record is None:
            continue
        bank_result = sampled_record.get("bank_search_result")
        if not isinstance(bank_result, dict):
            continue

        candidate_depth = float(bank_result.get("bathymetry_depth", np.nan))
        if (
            bool(bank_result.get("bathymetry_should_apply", False))
            and np.isfinite(candidate_depth)
            and 0.0 < candidate_depth < 25.0
        ):
            reach_id = int(source_reach_ids[entry_index])
            grouped_depths.setdefault(reach_id, []).append(candidate_depth)

    reach_medians: dict[int, float] = {}
    reach_statistics: dict[int, dict] = {}
    for reach_id, candidate_depths in grouped_depths.items():
        candidate_array = np.asarray(candidate_depths, dtype=np.float64)
        q25, q75 = np.percentile(candidate_array, [25.0, 75.0])
        retained_depths = candidate_array[
            (candidate_array >= q25) & (candidate_array <= q75)
        ]
        # Interpolated quartiles can exclude both observations on a two-cell
        # reach. Use both valid values rather than reducing an empty array.
        if retained_depths.size == 0:
            retained_depths = candidate_array

        median_depth = float(np.median(retained_depths))
        reach_medians[reach_id] = median_depth
        reach_statistics[reach_id] = {
            "q25": float(q25),
            "q75": float(q75),
            "median": median_depth,
            "candidate_count": int(candidate_array.size),
            "retained_count": int(retained_depths.size),
        }

    return reach_medians, reach_statistics


def _enforce_non_decreasing_downstream_reach_depths(
    reach_network_graph: nx.DiGraph,
    reach_median_depths: dict[int, float],
) -> dict[int, float]:
    """Ensure reach depth stays equal or increases moving downstream."""
    if len(reach_median_depths) == 0:
        return {}

    graph = reach_network_graph.copy()
    graph.add_nodes_from(int(reach_id) for reach_id in reach_median_depths)
    condensed_graph = nx.condensation(graph)
    node_to_component = condensed_graph.graph["mapping"]
    component_depths: dict[int, float] = {}

    for component_id in nx.topological_sort(condensed_graph):
        members = condensed_graph.nodes[component_id]["members"]
        local_depths = [
            float(reach_median_depths[int(reach_id)])
            for reach_id in members
            if int(reach_id) in reach_median_depths
            and np.isfinite(reach_median_depths[int(reach_id)])
        ]
        upstream_depths = [
            component_depths[int(predecessor_id)]
            for predecessor_id in condensed_graph.predecessors(component_id)
            if int(predecessor_id) in component_depths
        ]
        available_depths = local_depths + upstream_depths
        if available_depths:
            # The deepest incoming branch controls a confluence. Taking the
            # maximum also raises a shallower downstream local median.
            component_depths[int(component_id)] = float(max(available_depths))

    constrained_depths: dict[int, float] = {}
    for reach_id in graph.nodes:
        component_id = int(node_to_component[reach_id])
        if component_id in component_depths:
            constrained_depths[int(reach_id)] = component_depths[component_id]
    return constrained_depths


def _smooth_reach_bathymetry_depths(
    sampled_records: list[dict | None],
    params: dict,
) -> None:
    """Apply filtered reach medians and downstream-monotonic depth constraints."""
    source_reach_ids = (
        _CELL_SOURCE_STREAM_IDS
        if _CELL_SOURCE_STREAM_IDS is not None
        else _CELL_COMIDS
    )
    reach_medians, reach_statistics = _compute_filtered_reach_median_depths(
        sampled_records,
        source_reach_ids,
    )
    if len(reach_medians) == 0:
        return

    reach_network_graph, _ = _build_reach_network_graph(
        params.get("s_strmshp_path", ""),
        params.get("s_reach_id_field", ""),
        params.get("s_downstream_reach_id_field", ""),
    )
    constrained_depths = _enforce_non_decreasing_downstream_reach_depths(
        reach_network_graph,
        reach_medians,
    )

    for entry_index, sampled_record in enumerate(sampled_records):
        if sampled_record is None:
            continue
        reach_id = int(source_reach_ids[entry_index])
        if reach_id not in constrained_depths:
            continue

        existing_result = sampled_record.get("bank_search_result")
        bank_result = (
            dict(existing_result) if isinstance(existing_result, dict) else {}
        )
        statistics = reach_statistics.get(reach_id)
        bank_result["cross_section_bathymetry_depth"] = float(
            bank_result.get("bathymetry_depth", np.nan)
        )
        bank_result["cross_section_bathymetry_depth_source"] = (
            bank_result.get("bathymetry_depth_source")
        )
        if statistics is not None:
            bank_result["bathymetry_depth_reach_q25"] = statistics["q25"]
            bank_result["bathymetry_depth_reach_q75"] = statistics["q75"]
            bank_result["bathymetry_depth_reach_median"] = statistics["median"]
            bank_result["bathymetry_depth_reach_candidate_count"] = statistics[
                "candidate_count"
            ]
            bank_result["bathymetry_depth_reach_retained_count"] = statistics[
                "retained_count"
            ]

        constrained_depth = float(constrained_depths[reach_id])
        bank_result["bathymetry_depth"] = constrained_depth
        bank_result["bathymetry_should_apply"] = True
        bank_result["bathymetry_depth_source"] = (
            "downstream_monotonic_reach_median"
        )
        bank_result["bathymetry_depth_reach_id"] = reach_id
        bank_result["bathymetry_depth_network_constrained"] = constrained_depth
        bank_result["bathymetry_depth_monotonic_adjustment"] = float(
            constrained_depth - reach_medians.get(reach_id, constrained_depth)
        )
        sampled_record["bank_search_result"] = bank_result



def _build_precomputed_cross_section_record(
    x_section: CrossSection,
    dem_low_point_elev: float,
    bathymetry_applied: bool = False,
    inflect_curve: np.ndarray | None = None,
    bank_search_result: dict | None = None,
) -> dict:
    """Capture the current cross section so it can be replayed later.

    ARC now moves through three distinct cross-section stages:

    1. sample the raw DEM/land-cover cross section for every stream cell,
    2. determine banks for every cached section after the reach-scale INFLECT
       curves have been assembled, and
    3. optionally apply bathymetry to the cached section before hydraulics.

    This record stores the currently active state of the section, plus enough
    metadata to replay it later without touching the DEM again.
    """
    record = {
        "row": int(x_section.row),
        "col": int(x_section.col),
        "xs_angle": float(x_section.d_xs_direction),
        "ordinate_dist": float(x_section.d_ordinate_dist),
        "xs1_profile": x_section.da_xs_profile1[:x_section.xs1_n].copy(),
        "xs2_profile": x_section.da_xs_profile2[:x_section.xs2_n].copy(),
        "lc1_profile": x_section.ia_lc_xs1[:x_section.xs1_n].copy(),
        "lc2_profile": x_section.ia_lc_xs2[:x_section.xs2_n].copy(),
        "xs1_row": x_section.ia_xc_row1_index_main[:x_section.xs1_n].copy(),
        "xs1_col": x_section.ia_xc_column1_index_main[:x_section.xs1_n].copy(),
        "xs2_row": x_section.ia_xc_row2_index_main[:x_section.xs2_n].copy(),
        "xs2_col": x_section.ia_xc_column2_index_main[:x_section.xs2_n].copy(),
        "dem_low_point_elev": float(dem_low_point_elev),
        "bathymetry_applied": bool(bathymetry_applied),
    }
    if inflect_curve is not None:
        record["inflect_curve"] = np.asarray(inflect_curve, dtype=np.float64).copy()
    if bank_search_result is not None:
        record["bank_search_result"] = dict(bank_search_result)
    return record


def _build_cross_section_export_record(
    x_section: CrossSection,
    params: dict,
    i_cell_comid: int,
    i_row_cell: int,
    i_column_cell: int,
    d_slope_use: float,
    inflect_curve: np.ndarray | None,
) -> dict:
    """Build the persisted per-cell cross-section record used by ARC outputs."""
    b_modified_dem = params['b_modified_dem']
    return {
        'COMID': int(i_cell_comid),
        'Row': int(i_row_cell - x_section.i_boundary_number),
        'Col': int(i_column_cell - x_section.i_boundary_number),
        'XS1_Profile': x_section.da_xs_profile1[0:x_section.xs1_n].copy() - 100 if b_modified_dem else x_section.da_xs_profile1[0:x_section.xs1_n].copy(),
        'Ordinate_Dist': float(x_section.d_ordinate_dist),
        'Manning_N_Raster1': x_section.mannings_n1[:x_section.xs1_n].copy(),
        'XS2_Profile': x_section.da_xs_profile2[0:x_section.xs2_n].copy() - 100 if b_modified_dem else x_section.da_xs_profile2[0:x_section.xs2_n].copy(),
        'Manning_N_Raster2': x_section.mannings_n2[:x_section.xs2_n].copy(),
        'r1': int(x_section.ia_xc_row1_index_main[x_section.xs1_n-1] - x_section.i_boundary_number),
        'c1': int(x_section.ia_xc_column1_index_main[x_section.xs1_n-1] - x_section.i_boundary_number),
        'r2': int(x_section.ia_xc_row2_index_main[x_section.xs2_n-1] - x_section.i_boundary_number),
        'c2': int(x_section.ia_xc_column2_index_main[x_section.xs2_n-1] - x_section.i_boundary_number),
        'Slope': float(d_slope_use),
        'Thalweg': float(x_section.get_thalweg() - 100 if b_modified_dem else x_section.get_thalweg()),
        'Inflect_D2W_Dy2': None if inflect_curve is None else np.asarray(inflect_curve, dtype=np.float64).copy(),
    }


def _sample_cross_section_for_cell(
    x_section: CrossSection,
    i_entry_cell: int,
    params: dict,
) -> tuple[bool, int, int, int, float]:
    """Sample, low-spot adjust, and resample one stream-cell cross section.

    This helper performs only the DEM/land-cover sampling stage. It does not
    attempt bank detection or bathymetry. That separation is intentional: ARC
    first samples *all* stream-cell cross sections, then computes reach-scale
    INFLECT curves and bank locations across the full sampled population, and
    only after that applies bathymetry.
    """
    i_row_cell = int(_CELL_ROWS[i_entry_cell])
    i_column_cell = int(_CELL_COLS[i_entry_cell])
    i_cell_comid = int(_CELL_COMIDS[i_entry_cell])
    using_manual_cross_sections = bool(params.get('s_manual_cross_section_file'))

    if using_manual_cross_sections:
        manual_record = _MANUAL_CROSS_SECTION_RECORDS.get(i_cell_comid)
        if manual_record is None:
            raise KeyError(f"Manual cross section for ID {i_cell_comid} was not found.")
        apply_manual_cross_section_data(x_section, manual_record)
    else:
        d_stream_direction, d_xs_direction = get_stream_direction_information(
            i_row_cell,
            i_column_cell,
            _STREAMS,
            params['i_general_direction_distance'],
        )
        if d_xs_direction > np.pi:
            i_precompute_angle_closest = round((d_xs_direction - np.pi) / x_section.d_precompute_angles)
        else:
            i_precompute_angle_closest = round(d_xs_direction / x_section.d_precompute_angles)

        x_section.set_cross_section(i_row_cell, i_column_cell, i_precompute_angle_closest, d_xs_direction)

        i_low_spot_range = params['i_low_spot_range']
        if i_low_spot_range > 0:
            x_section.adjust_cross_section_to_lowest_point(i_low_spot_range)
            i_row_cell, i_column_cell = x_section.get_row_col()

        if x_section.has_angles_to_test():
            x_section.test_angles_and_reset_cross_section(i_row_cell, i_column_cell)

    if not x_section.is_valid():
        return False, i_cell_comid, i_row_cell, i_column_cell, float("nan")

    i_row_cell, i_column_cell = x_section.get_row_col()
    return True, i_cell_comid, i_row_cell, i_column_cell, float(x_section.get_thalweg())


def _replay_precomputed_cross_section(
    x_section: CrossSection,
    precomputed_record: dict,
    reach_bank_index: float | None = None,
) -> None:
    """Load a cached cross section back into a reusable sampler instance."""
    apply_manual_cross_section_data(x_section, precomputed_record)
    x_section.set_reach_scale_inflect_bank_index(reach_bank_index)

# Function for identifying top inflection point peaks
def top_peaks_id(peaks_array, num_peaks):
    if len(peaks_array[0]) < num_peaks:
        peak_range = len(peaks_array[0])
    else: 
        peak_range = num_peaks
    peak_indices = peaks_array[0]
    max_peaks = []
    for i in range(0, peak_range): # Here is where to define number of peaks looking for 
        current_max = 0 
        current_max_index = 0
        for j in range(len(peak_indices)):
            if abs(peaks_array[1]['peak_heights'][j]) > current_max:
                current_max = abs(peaks_array[1]['peak_heights'][j])
                current_max_index = j
        peaks_array[1]['peak_heights'] = np.delete(peaks_array[1]['peak_heights'], current_max_index)
        max_peaks.append(peak_indices[current_max_index])
        peak_indices = np.delete(peak_indices, current_max_index)
    return max_peaks

def _build_reach_inflect_index_dictionaries(
    grouped_curves: dict[int, list[tuple[np.ndarray, np.ndarray]]],
    params: dict,
) -> tuple[dict[int, float], dict[int, float]]:
    """Aggregate per-cell INFLECT curves into reach-scale bank/terrace depths.

    ``_calculate_inflect_curve_with_depths`` now produces both the
    ``d2W/dy2`` curve and the exact depth axis used to evaluate that curve.
    This helper therefore returns physical depths above the thalweg, not
    synthetic curve indices, so the reach-scale INFLECT bank finder and the
    representative-cross-section workflow both use the same staged geometry.
    """
    inflect_bank_index_dict: dict[int, float] = {}
    inflect_terrace_index_dict: dict[int, float] = {}
    plot_directory = _get_reach_inflect_plot_directory(params) if TEMP_PLOT_REACH_INFLECT_CURVES else ''
    max_peak_ratio = 2 # The ratio of max peak:detected peak. Default val 2 means the detected peak must be one half the magnitude of the maximum peak. 
    distance_val = 5 # The minimum distance required between individual peaks, unitless. Must be greater or equal to 1. 
    width_val = 2 # The minumum width of an individual peak at the base, unitless

    for reach_id, curves in grouped_curves.items():
        min_length = min(curve.shape[0] for curve, _ in curves)
        if min_length <= 0:
            continue
        aligned_curves = np.vstack([curve[:min_length] for curve, _ in curves])
        aligned_depths = np.vstack([depths[:min_length] for _, depths in curves])
        inflections_array = np.nanmean(aligned_curves, axis=0)
        mean_depth_values = np.nanmean(aligned_depths, axis=0)

        # identify top three peaks (across positive and negative)
        peaks_pos = find_peaks(inflections_array, height=max(inflections_array)/max_peak_ratio, distance=distance_val, width=width_val) #, prominence=prominence_val) # require peaks to be at least half the mag of max peak
        inflections_array_neg = [-i for i in inflections_array] # invert all signs to detect negative peaks
        peaks_neg = find_peaks(inflections_array_neg, height=max(inflections_array_neg)/max_peak_ratio, distance=distance_val, width=width_val) #, prominence=prominence_val) # require peaks to be at least half the mag of max peak

        # potential bank locations are the postiive peaks
        bank_indices = top_peaks_id(peaks_pos, 3)

        # order the bank indices from lowest to highest
        bank_indices = sorted(bank_indices)

        # remove the bank indices if it is 0 or if it is the last index of the array (the end of the cross section)
        bank_indices = [i for i in bank_indices if i > 0 and i < len(inflections_array) - 1]

        # if no positive peaks are detected, set bank_index to 0 (the start of the array)
        if len(bank_indices) <= 0:
            bank_index = -1
        else:
            bank_index = int(bank_indices[0])

        # if bank_index is 0, try to find the next lowest index in bank_indices that is not a depth value of 0 to see if it is a better candidate for the bank location
        if bank_index >= 0 and inflections_array[bank_index] <= 0.0:
            if len(bank_indices) > 1:
                for i in range(1, len(bank_indices)):
                    if inflections_array[bank_indices[i]] > 0.0:
                        bank_index = int(bank_indices[i]) - 1
                        break
            # Otherwise we don't know what the bank index is, so add None to the dictionary and set terrace_index to a large number
            else:
                bank_index = -1

        if bank_index > 0:
            # potential negative peaks are the terrace locations
            terrace_indices = top_peaks_id(peaks_neg, 3)
            # sort the terrace indices from lowest to highest
            terrace_indices = sorted(terrace_indices)
            # remove the terrace indices if it is the first or last index of the array (the start or end of the cross section)
            terrace_indices = [i for i in terrace_indices if i > 0 and i < len(inflections_array) - 1]
            if len(terrace_indices) <= 0:
                terrace_index = len(inflections_array) + 20
            elif len(terrace_indices) == 1:
                terrace_index = terrace_indices[0]
                if terrace_index < bank_index:
                    terrace_index = -1
            elif len(terrace_indices) > 1:
                terrace_index = -1
                for terrace_index in terrace_indices:
                    if terrace_index > bank_index:
                        break
                if terrace_index <= bank_index:
                    terrace_index = -1
        else:
            terrace_index = -1

        bank_depth = -1.0
        if bank_index >= 0 and bank_index < mean_depth_values.size:
            bank_depth = float(mean_depth_values[bank_index])

        terrace_depth = -1.0
        if terrace_index >= 0 and terrace_index < mean_depth_values.size:
            terrace_depth = float(mean_depth_values[terrace_index])

        inflect_bank_index_dict[int(reach_id)] = bank_depth
        inflect_terrace_index_dict[int(reach_id)] = terrace_depth
        if TEMP_PLOT_REACH_INFLECT_CURVES:
            _plot_reach_average_inflect_curve(
                int(reach_id),
                mean_depth_values,
                inflections_array,
                bank_index,
                terrace_index,
                plot_directory,
            )

    return inflect_bank_index_dict, inflect_terrace_index_dict


def _populate_reach_inflect_arrays(
    inflect_bank_index_dict: dict[int, float],
    inflect_terrace_index_dict: dict[int, float],
    processes: int,
) -> None:
    """Create shared per-cell arrays from reach-level INFLECT depths."""
    source_ids = _CELL_SOURCE_STREAM_IDS if _CELL_SOURCE_STREAM_IDS is not None else _CELL_COMIDS
    create_array("_CELL_REACH_INFLECT_BANK_INDEX", processes, (_CELL_COMIDS.size,), np.float64, fill_value=-1.0)
    create_array("_CELL_REACH_INFLECT_TERRACE_INDEX", processes, (_CELL_COMIDS.size,), np.float64, fill_value=-1.0)

    if inflect_bank_index_dict:
        _CELL_REACH_INFLECT_BANK_INDEX[:] = np.fromiter(
            (inflect_bank_index_dict.get(int(stream_id), -1.0) for stream_id in source_ids),
            dtype=np.float64,
            count=len(_CELL_COMIDS),
        )

    if inflect_terrace_index_dict:
        _CELL_REACH_INFLECT_TERRACE_INDEX[:] = np.fromiter(
            (inflect_terrace_index_dict.get(int(stream_id), -1.0) for stream_id in source_ids),
            dtype=np.float64,
            count=len(_CELL_COMIDS),
        )


def _get_local_bank_search_result(
    x_section: CrossSection,
    params: dict,
    d_bathy_target_width: float | None,
) -> dict:
    """Run the appropriate local bank-search workflow for the current section."""
    if params['b_bathy_use_banks']:
        return x_section.get_bank_elevation_search_result(
            d_bathy_target_width=d_bathy_target_width,
        )
    return x_section.get_wse_or_lc_bank_search_result(
        d_bathy_target_width=d_bathy_target_width,
    )


def _annotate_cross_sections_with_bank_search_results(
    sampled_records: list[dict | None],
    params: dict,
    quiet: bool,
) -> None:
    """Run the bank-search hierarchy for every cached sampled cross section."""
    x_section = get_cross_section(params['dx'], params['dy'], _DEM, _LAND_COVER, _STREAMS, params)

    for i_entry_cell in tqdm.tqdm(range(_CELL_COMIDS.size), total=_CELL_COMIDS.size, disable=quiet):
        sampled_record = sampled_records[i_entry_cell]
        if sampled_record is None:
            continue

        reach_bank_index = None
        if _CELL_REACH_INFLECT_BANK_INDEX is not None:
            reach_bank_index = float(_CELL_REACH_INFLECT_BANK_INDEX[i_entry_cell])
        _replay_precomputed_cross_section(x_section, sampled_record, reach_bank_index=reach_bank_index)

        (_, _, _, d_bathy_target_width) = _get_cell_bathymetry_inputs(
            i_entry_cell,
            int(_CELL_ROWS[i_entry_cell]),
            int(_CELL_COLS[i_entry_cell]),
            params,
        )
        bank_search_result = _get_local_bank_search_result(
            x_section,
            params,
            d_bathy_target_width,
        )
        sampled_record["bank_search_result"] = dict(bank_search_result)


def _compute_mean_reach_stream_direction(stream_directions: list[float]) -> float:
    """Estimate one representative along-stream axis for a reach.

    ``get_stream_direction_information`` returns an angle for each stream cell,
    but ARC only needs the line orientation here, not the sign. This helper
    therefore averages directions modulo ``pi`` using a doubled-angle mean.
    """
    if len(stream_directions) == 0:
        return 0.0

    # Collapse opposite bearings onto the same [0, pi) axis. A reach directed
    # northeast and one directed southwest describe the same line for the
    # coordinate projection performed by the caller.
    angles = np.mod(np.asarray(stream_directions, dtype=np.float64), np.pi)

    # Doubling axial angles turns them into ordinary circular data. Their
    # vector mean can then be halved to recover the mean unoriented axis while
    # avoiding the discontinuity between angles just below pi and just above 0.
    sin_2theta = np.nanmean(np.sin(2.0 * angles))
    cos_2theta = np.nanmean(np.cos(2.0 * angles))
    if not np.isfinite(sin_2theta) or not np.isfinite(cos_2theta):
        return float(angles[0])

    mean_angle = 0.5 * math.atan2(sin_2theta, cos_2theta)
    if mean_angle < 0.0:
        mean_angle += np.pi
    return float(mean_angle)


def _compute_raw_bank_elevation_from_result(
    x_section: CrossSection,
    bank_search_result: dict | None,
) -> float:
    """Extract one representative bank elevation from a local bank result.

    Bank elevations equal to the sampled thalweg are treated as unresolved
    placeholders and excluded from the reach-smoothing input.
    """
    if bank_search_result is None:
        return np.nan

    thalweg = float(x_section.get_thalweg())

    try:
        bank_elev_1 = float(bank_search_result.get("bank_elev_1"))
    except Exception:
        bank_elev_1 = np.nan
    try:
        bank_elev_2 = float(bank_search_result.get("bank_elev_2"))
    except Exception:
        bank_elev_2 = np.nan

    # Either side may be missing or may still equal the thalweg placeholder.
    # Only genuinely elevated, finite banks are allowed to seed smoothing.
    valid_bank_elevations = np.asarray(
        [
            elev
            for elev in (bank_elev_1, bank_elev_2)
            if np.isfinite(elev) and not np.isclose(elev, thalweg)
        ],
        dtype=np.float64,
    )
    if valid_bank_elevations.size == 0:
        return np.nan

    # The lower of the two banks is the first elevation at which water can
    # leave the channel, so it is the conservative bankfull control elevation.
    bank_elevation = float(np.nanmin(valid_bank_elevations))

    return bank_elevation


def _exclude_thalweg_equal_bank_elevations(
    bank_elevations: np.ndarray,
    thalweg_elevations: np.ndarray,
) -> np.ndarray:
    """Replace bank elevations at or below their sampled thalweg with NaN.

    The function retains its original name for compatibility, but now rejects
    below-thalweg values as well. Such a value cannot represent a physical bank
    and must not become either a reach outlet minimum or a per-cell anchor.
    """
    bank_elevations = np.asarray(bank_elevations, dtype=np.float64).copy()
    thalweg_elevations = np.asarray(thalweg_elevations, dtype=np.float64)
    if bank_elevations.shape != thalweg_elevations.shape:
        raise ValueError(
            "Bank-elevation and thalweg-elevation arrays must have the same shape."
        )

    at_or_below_thalweg = (
        np.isfinite(bank_elevations)
        & np.isfinite(thalweg_elevations)
        & (
            (bank_elevations < thalweg_elevations)
            | np.isclose(bank_elevations, thalweg_elevations)
        )
    )
    bank_elevations[at_or_below_thalweg] = np.nan
    return bank_elevations


def _get_bank_search_result_for_smoothing(
    x_section: CrossSection,
    sampled_record: dict,
    params: dict,
    i_entry_cell: int,
) -> dict:
    """Return a bank-search result for the reach-smoothing prepass.

    The staged bathymetry workflow now tries to give every sampled cross
    section a smoothed reach-scale bank elevation. If an earlier prepass did
    not store a bank-search result for this sampled section, or stored a
    non-dictionary placeholder, ARC reruns the local bank hierarchy here so
    the reach smoother still has a consistent input record to work with.
    """
    # Prefer the cached result from the bank-search prepass; copying prevents
    # later annotations from unexpectedly mutating a shared dictionary.
    bank_search_result = sampled_record.get("bank_search_result")
    if isinstance(bank_search_result, dict):
        return dict(bank_search_result)

    # A missing cache entry is recoverable: reconstruct the cell-specific
    # target width and run the same configured local search hierarchy now.
    (_, _, _, d_bathy_target_width) = _get_cell_bathymetry_inputs(
        i_entry_cell,
        int(sampled_record["row"]),
        int(sampled_record["col"]),
        params,
    )
    if params['b_bathy_use_banks']:
        return x_section.get_bank_elevation_search_result(
            d_bathy_target_width=d_bathy_target_width,
        )
    return x_section.get_wse_or_lc_bank_search_result(
        d_bathy_target_width=d_bathy_target_width,
    )


def _reconstruct_reach_bank_width_with_fallback(
    x_section: CrossSection,
    bank_search_result: dict,
    median_width: float,
    q75: float,
    maximum_width_increase_cells: int = 10,
) -> dict:
    """Rebuild an outlier width and validate the resulting bank geometry.

    The median reach width is attempted first. If the sampled ordinate spacing
    cannot represent that target with valid bank indices, each subsequent
    attempt widens the target by one cross-section cell, through at most ten
    cells. A reconstruction is accepted only when it is valid, was actually
    rebuilt, and its resolved bank-to-bank width remains at or below the
    original reach q75 cutoff. If no candidate passes, ARC replaces the outlier
    with the minimum resolvable one-cell channel instead of retaining it.
    """
    cell_size = float(x_section.d_ordinate_dist)
    maximum_width_increase_cells = max(int(maximum_width_increase_cells), 0)
    attempted_target_widths: list[float] = []

    for width_increase_cells in range(maximum_width_increase_cells + 1):
        target_width = float(
            median_width + width_increase_cells * cell_size
        )
        attempted_target_widths.append(target_width)
        candidate_result = x_section.build_bank_search_result_from_target_width(
            bank_search_result,
            target_width,
            "filter_bank_width_to_reach_median",
        )
        candidate_width = x_section.get_top_width_from_bank_search_result(
            candidate_result
        )
        candidate_was_rebuilt = bool(
            candidate_result.get("reach_top_width_filter_applied", False)
        )
        candidate_is_valid = bool(candidate_result.get("is_valid", False))
        candidate_has_bank_indices = (
            int(candidate_result.get("i_bank_1_index", 0)) > 0
            and int(candidate_result.get("i_bank_2_index", 0)) > 0
        )
        candidate_within_q75 = (
            np.isfinite(candidate_width)
            and candidate_width <= q75 + np.finfo(np.float64).eps * max(abs(q75), 1.0)
        )
        if (
            candidate_was_rebuilt
            and candidate_is_valid
            and candidate_has_bank_indices
            and candidate_within_q75
        ):
            candidate_result["reach_top_width_filter_reconstruction_attempts"] = int(
                width_increase_cells + 1
            )
            candidate_result["reach_top_width_filter_width_increase_cells"] = int(
                width_increase_cells
            )
            candidate_result["reach_top_width_filter_selected_target_top_width"] = float(
                target_width
            )
            candidate_result["reach_top_width_filter_attempted_target_widths"] = tuple(
                attempted_target_widths
            )
            candidate_result["reach_top_width_filter_post_validation_passed"] = True
            candidate_result["reach_top_width_filter_one_cell_fallback_applied"] = False
            return candidate_result

    # No median-based target produced usable banks within q75. Explicitly
    # replace the original outlier with indices (1, 1), which represents the
    # smallest channel available at the current raster/cross-section spacing.
    fallback_result = x_section.build_one_cell_bank_search_result(
        bank_search_result,
        "fallback_to_one_cell_channel_after_width_reconstruction",
    )
    fallback_result["reach_top_width_filter_reconstruction_attempts"] = int(
        len(attempted_target_widths)
    )
    fallback_result["reach_top_width_filter_width_increase_cells"] = int(
        maximum_width_increase_cells
    )
    fallback_result["reach_top_width_filter_selected_target_top_width"] = float(
        cell_size
    )
    fallback_result["reach_top_width_filter_attempted_target_widths"] = tuple(
        attempted_target_widths
    )
    fallback_result["reach_top_width_filter_post_validation_passed"] = bool(
        fallback_result.get("is_valid", False)
    )
    return fallback_result


def _apply_reach_top_width_filter(
    x_section: CrossSection,
    sampled_records: list[dict | None],
    reach_entries: list[dict],
    reach_id: int,
) -> dict | None:
    """Replace reach top-width outliers with the reach-median width geometry.

    ARC groups sampled cross sections by reach before it smooths bank
    elevations. This helper uses those same grouped sections to evaluate local
    bank-to-bank top width at each stream cell, compute the 25th, 50th, and
    75th percentile widths for the reach, and replace any outlier bank result
    with bank indices that match the reach-median width as closely as the
    sampled cross-section spacing allows. The percentile summary is returned
    so later reach-level geometry fallbacks can reuse the same median width.
    """
    # First pass: replay each cached cross section so width is measured from
    # that section's own profile and bank indices, not the previous section's.
    widths_by_entry_index: dict[int, float] = {}
    reach_widths: list[float] = []

    for reach_entry in reach_entries:
        entry_index = int(reach_entry["entry_index"])
        sampled_record = sampled_records[entry_index]
        if sampled_record is None:
            continue

        reach_bank_index = None
        if _CELL_REACH_INFLECT_BANK_INDEX is not None:
            reach_bank_index = float(_CELL_REACH_INFLECT_BANK_INDEX[entry_index])
        _replay_precomputed_cross_section(x_section, sampled_record, reach_bank_index=reach_bank_index)

        bank_search_result = sampled_record.get("bank_search_result")
        top_width = x_section.get_top_width_from_bank_search_result(bank_search_result)
        if np.isfinite(top_width) and top_width > 0.0:
            widths_by_entry_index[entry_index] = float(top_width)
            reach_widths.append(float(top_width))

    if len(reach_widths) == 0:
        return None

    # Retain the central interquartile band and use its median as the first
    # reconstruction target for widths outside that band.
    reach_width_array = np.asarray(reach_widths, dtype=np.float64)
    q25 = float(np.percentile(reach_width_array, 25))
    median_width = float(np.percentile(reach_width_array, 50))
    q75 = float(np.percentile(reach_width_array, 75))
    if not np.isfinite(q25) or not np.isfinite(median_width) or not np.isfinite(q75):
        return None

    # Second pass: replace only finite widths outside the reach band. Missing
    # or invalid bank results are handled separately by the median-fill helper.
    for reach_entry in reach_entries:
        entry_index = int(reach_entry["entry_index"])
        sampled_record = sampled_records[entry_index]
        if sampled_record is None:
            continue

        bank_search_result = sampled_record.get("bank_search_result")
        observed_top_width = widths_by_entry_index.get(entry_index, float("nan"))
        if not isinstance(bank_search_result, dict):
            continue

        if np.isfinite(observed_top_width) and (observed_top_width < q25 or observed_top_width > q75):
            reach_bank_index = None
            if _CELL_REACH_INFLECT_BANK_INDEX is not None:
                reach_bank_index = float(_CELL_REACH_INFLECT_BANK_INDEX[entry_index])
            _replay_precomputed_cross_section(x_section, sampled_record, reach_bank_index=reach_bank_index)

            # Convert the median physical width back to bank indices, then
            # validate the resolved geometry. If the median is not representable,
            # progressively widen it by one cell up to ten times before using
            # the deterministic one-cell fallback.
            updated_bank_result = _reconstruct_reach_bank_width_with_fallback(
                x_section,
                bank_search_result,
                median_width,
                q75,
            )
            updated_bank_result["reach_top_width_filter_reach_id"] = int(reach_id)
            updated_bank_result["reach_top_width_filter_q25"] = float(q25)
            updated_bank_result["reach_top_width_filter_median_top_width"] = float(median_width)
            updated_bank_result["reach_top_width_filter_q75"] = float(q75)
            updated_bank_result["reach_top_width_filter_observed_top_width"] = float(observed_top_width)
            updated_bank_result["reach_top_width_filter_final_top_width"] = float(
                x_section.get_top_width_from_bank_search_result(updated_bank_result)
            )
            sampled_record["bank_search_result"] = updated_bank_result
        else:
            bank_search_result["reach_top_width_filter_applied"] = False
            bank_search_result["reach_top_width_filter_reach_id"] = int(reach_id)
            bank_search_result["reach_top_width_filter_q25"] = float(q25)
            bank_search_result["reach_top_width_filter_median_top_width"] = float(median_width)
            bank_search_result["reach_top_width_filter_q75"] = float(q75)
            bank_search_result["reach_top_width_filter_observed_top_width"] = float(observed_top_width)
            bank_search_result["reach_top_width_filter_final_top_width"] = float(observed_top_width)

    return {
        "reach_id": int(reach_id),
        "q25": float(q25),
        "median_top_width": float(median_width),
        "q75": float(q75),
    }


def _apply_reach_median_top_width_to_missing_bank(
    x_section: CrossSection,
    sampled_records: list[dict | None],
    reach_entries: list[dict],
    reach_top_width_stats: dict | None,
) -> None:
    """Assign reach-median bank indices to sections lacking a valid bank result.

    Reach smoothing already computes a representative bankfull width from the
    valid sections in the reach. This helper reuses that median width to
    rebuild bank indices for any cross section whose local bank search still
    returned an invalid result, allowing the later bathymetry stages to carry
    a reach-consistent width geometry into the smoothed-bank workflow.
    """
    if not isinstance(reach_top_width_stats, dict):
        return

    median_top_width = float(reach_top_width_stats.get("median_top_width", np.nan))
    if not np.isfinite(median_top_width) or median_top_width <= 0.0:
        return

    reach_id = int(reach_top_width_stats.get("reach_id", -1))

    for reach_entry in reach_entries:
        entry_index = int(reach_entry["entry_index"])
        sampled_record = sampled_records[entry_index]
        if sampled_record is None:
            continue

        bank_search_result = sampled_record.get("bank_search_result")
        # Preserve every valid local/filtered bank pair. This helper supplies
        # geometry only where the local search hierarchy failed completely.
        if isinstance(bank_search_result, dict) and bool(bank_search_result.get("is_valid", False)):
            bank_search_result["reach_median_bank_fill_applied"] = False
            bank_search_result["reach_median_bank_fill_reach_id"] = reach_id
            bank_search_result["reach_median_bank_fill_target_top_width"] = float(median_top_width)
            continue

        reach_bank_index = None
        if _CELL_REACH_INFLECT_BANK_INDEX is not None:
            reach_bank_index = float(_CELL_REACH_INFLECT_BANK_INDEX[entry_index])
        _replay_precomputed_cross_section(x_section, sampled_record, reach_bank_index=reach_bank_index)

        # Rebuild indices independently for this profile because equal target
        # widths can correspond to different indices at different resolutions.
        updated_bank_result = x_section.build_bank_search_result_from_target_width(
            bank_search_result,
            median_top_width,
            "fill_missing_bank_with_reach_median_top_width",
        )
        updated_bank_result["reach_median_bank_fill_applied"] = bool(updated_bank_result.get("is_valid", False))
        updated_bank_result["reach_median_bank_fill_reach_id"] = reach_id
        updated_bank_result["reach_median_bank_fill_target_top_width"] = float(median_top_width)
        updated_bank_result["reach_median_bank_fill_final_top_width"] = float(
            x_section.get_top_width_from_bank_search_result(updated_bank_result)
        )
        sampled_record["bank_search_result"] = updated_bank_result


def _estimate_minimum_smoothed_bank_elevation(
    x_section: CrossSection,
    minimum_bank_height: float = 0.6,
) -> float:
    """Build a conservative bank-elevation seed from the sampled profile.

    Some sampled cross sections never yield a valid local multi-cell bank
    width, but the reach smoother still needs a finite elevation for those
    sections so they can inherit the reach-scale trend. ARC therefore falls
    back to the first sampled ordinates beside the thalweg when available and
    otherwise uses the thalweg, which will be left out of the reach-smoothing median calculation.
    """
    thalweg = float(x_section.get_thalweg())
    candidate_elevations = []
    if x_section.xs1_n > 1:
        candidate_elevations.append(float(x_section.da_xs_profile1[1]))
    if x_section.xs2_n > 1:
        candidate_elevations.append(float(x_section.da_xs_profile2[1]))

    valid_candidates = [
        elevation
        for elevation in candidate_elevations
        if np.isfinite(elevation) and elevation > thalweg
    ]
    if len(valid_candidates) == 0:
        return thalweg

    return float(np.nanmedian(np.asarray(valid_candidates, dtype=np.float64)))


def _moving_window_median(
                          values: np.ndarray,
                          window_size: int,
                        ) -> np.ndarray:
    """Apply a local median smoother, referenced to the thalweg.

    When ``reference_values`` is supplied, this function computes a 
    smoothed profile for the stream reach.
    """
    if values.size == 0:
        return values.copy()

    values = np.asarray(values, dtype=np.float64)


    half_window = max(int(window_size) // 2, 0)
    smoothed = np.zeros(values.size, dtype=np.float64)
    for i in range(values.size):
        start = max(0, i - half_window)
        stop = min(values.size, i + half_window + 1)
        window_values = values[start:stop]
        finite_window_values = window_values[np.isfinite(window_values)]
        if finite_window_values.size > 0:
            smoothed[i] = float(np.nanmin(finite_window_values))

    finite_smoothed = smoothed[np.isfinite(smoothed)]
    return finite_smoothed


def _fit_monotone_decreasing_bank_line(
    along_stream_coordinates: np.ndarray,
    smoothed_bank_elevations: np.ndarray,
    thalweg_elevations: np.ndarray,
) -> np.ndarray:
    """Finalize a thalweg-parallel smoothed bank-elevation line.

    The moving-window smoother now operates on bank height above the thalweg
    and reconstructs a bank-elevation line that already follows the exact
    reach thalweg slope. This helper therefore only performs safety checks to
    ensure the smoothed bank elevation stays above the local thalweg.
    """
    del along_stream_coordinates
    if smoothed_bank_elevations.size == 0:
        return smoothed_bank_elevations.copy()
    if smoothed_bank_elevations.size == 1:
        return np.maximum(smoothed_bank_elevations.copy(), thalweg_elevations + 0.01)
    return np.maximum(
        np.asarray(smoothed_bank_elevations, dtype=np.float64).copy(),
        np.asarray(thalweg_elevations, dtype=np.float64) + 0.01,
    )

def _coerce_optional_reach_identifier(value) -> int | None:
    """Convert a reach identifier read from a table into ``int`` or ``None``."""
    if value in ('', None):
        return None
    # Geospatial drivers commonly return IDs as integer-like floats or strings,
    # so try the lossless/common integer form before accepting float text.
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass

    try:
        return int(value)
    except (TypeError, ValueError):
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return None


def _measure_reach_geometry_length(
    geometry,
    source_crs,
) -> float:
    """Measure a reach geometry in meters using its declared CRS.

    Shapely's ``geometry.length`` is expressed in native coordinate units. For
    a geographic stream layer that value is degrees and is therefore not a
    usable distance for the bank-surface slope calculations, whose raster-cell
    stations are measured in meters. Geographic geometries are measured on
    their CRS ellipsoid; projected geometries are converted from their declared
    linear unit to meters. A missing/unparseable CRS retains the native length
    as a compatibility fallback because its physical unit is unknowable.
    """
    if geometry is None:
        return np.nan
    try:
        if geometry.is_empty:
            return np.nan
        native_length = float(geometry.length)
    except Exception:
        return np.nan
    if not np.isfinite(native_length) or native_length <= 0.0:
        return np.nan

    try:
        parsed_crs = CRS.from_user_input(source_crs)
    except Exception:
        return native_length

    if parsed_crs.is_geographic:
        try:
            geodesic_length = abs(
                float(parsed_crs.get_geod().geometry_length(geometry))
            )
            if np.isfinite(geodesic_length) and geodesic_length > 0.0:
                return geodesic_length
        except Exception:
            return native_length

    # Projected CRS axis metadata reports the conversion from its native unit
    # (meters, international feet, US survey feet, etc.) to SI meters.
    try:
        axis_info = parsed_crs.axis_info
        unit_to_meters = float(axis_info[0].unit_conversion_factor)
        length_meters = native_length * unit_to_meters
        if np.isfinite(length_meters) and length_meters > 0.0:
            return length_meters
    except Exception:
        pass
    return native_length


def _build_reach_network_graph(
    s_strmshp_path: str,
    reach_id_field: str,
    downstream_reach_id_field: str,
) -> tuple[nx.DiGraph, dict[int, int | None]]:
    """Read the stream network table and build an upstream-to-downstream graph.

    Parameters
    ----------
    s_strmshp_path : str
        Path to the stream vector dataset used by ARC.
    reach_id_field : str
        Field containing each reach identifier.
    downstream_reach_id_field : str
        Field containing the immediate downstream reach identifier.

    Returns
    -------
    tuple
        ``(graph, downstream_map)`` where ``graph`` is a
        :class:`networkx.DiGraph` with one node per reach and one directed edge
        per upstream-to-downstream connection, and ``downstream_map`` stores
        the immediate downstream reach ID for each reach when available.
    """
    graph = nx.DiGraph()
    downstream_map: dict[int, int | None] = {}

    if not s_strmshp_path:
        raise ValueError(
            'StrmShp_File is required to build the reach bank-elevation network.'
        )
    if not reach_id_field or not downstream_reach_id_field:
        raise ValueError(
            'Both reach_id and downstream_reach_id are required to build the '
            'reach bank-elevation network.'
        )

    try:
        gdf_stream = gpd.read_file(s_strmshp_path)
    except Exception as ex:
        raise ValueError(
            'Unable to read StrmShp_File for downstream reach smoothing: '
            + str(ex)
        )

    required_columns = {reach_id_field, downstream_reach_id_field}
    missing_columns = sorted(required_columns.difference(gdf_stream.columns))
    if missing_columns:
        raise ValueError(
            'Unable to build the reach bank-elevation network because '
            + ', '.join(missing_columns)
            + ' was not found in StrmShp_File.'
        )

    # Collapse duplicate feature rows to one topology record per reach. The
    # first occurrence supplies its downstream link and its distance weight.
    # Resolve the layer CRS once so every graph-node length uses meters rather
    # than blindly accepting degrees or another native coordinate unit.
    stream_crs = gdf_stream.crs
    reach_records: dict[int, tuple[int | None, float]] = {}
    for _, row in gdf_stream.iterrows():
        reach_id = _coerce_optional_reach_identifier(row[reach_id_field])
        if reach_id is None:
            continue

        downstream_reach_id = _coerce_optional_reach_identifier(row[downstream_reach_id_field])
        # A self-link cannot advance a downstream path; treat it as an outlet.
        if downstream_reach_id == reach_id:
            downstream_reach_id = None

        # Geometry length is the interpolation distance assigned to both the
        # reach node and its downstream edge. Convert it to meters so it has the
        # same unit as ordered raster-cell stations. This is essential for
        # geographic flowlines, where ``geometry.length`` alone returns degrees
        # and can shorten a multi-kilometer reach to a small decimal value.
        reach_length = 1.0
        geometry = row.geometry
        geometry_length = _measure_reach_geometry_length(
            geometry,
            stream_crs,
        )
        if np.isfinite(geometry_length) and geometry_length > 0.0:
            reach_length = geometry_length

        if reach_id not in reach_records:
            reach_records[reach_id] = (downstream_reach_id, reach_length)

    if len(reach_records) == 0:
        raise ValueError(
            'Unable to build the reach bank-elevation network because no valid '
            'reach_id values were found in StrmShp_File.'
        )

    # Add all reaches as nodes first so outlets and reaches whose downstream
    # IDs fall outside the input dataset remain represented in the graph.
    for reach_id, (_, reach_length) in reach_records.items():
        graph.add_node(reach_id, length=float(reach_length))

    for reach_id, (downstream_reach_id, reach_length) in reach_records.items():
        downstream_map[reach_id] = downstream_reach_id
        if downstream_reach_id is None or downstream_reach_id not in reach_records:
            continue
        graph.add_edge(reach_id, downstream_reach_id, length=float(reach_length))

    if graph.number_of_nodes() == 0:
        raise ValueError(
            'Unable to build the reach bank-elevation network because no valid '
            'stream reaches were added to the graph.'
        )

    return graph, downstream_map

def _fill_segment(
    start_index: int,
    end_index: int,
    start_elevation: float,
    path_stations: np.array,
    path_smoothed: np.array,
    end_elevation: float | None = None,
) -> None:
    """Fill one downstream path interval from an upstream elevation anchor.

    ``path_stations`` are distances to an outlet, so they decrease as ``idx``
    moves downstream. If a lower downstream anchor exists, the helper draws a
    straight line between the anchors; otherwise it holds the elevation flat.
    Existing values are combined with ``min`` because a
    reach can receive candidates from overlapping segments or network paths.
    """
    start_station = path_stations[start_index]
    end_station = path_stations[end_index]
    # Distance-to-outlet should decrease downstream. Fall back to index spacing
    # when network distances are missing or do not provide a positive interval.
    delta_station = start_station - end_station
    if not np.isfinite(delta_station) or delta_station <= 0.0:
        delta_station = float(max(end_index - start_index, 1))

    # Interpolate only toward a genuinely lower downstream control. Otherwise
    # retain a flat line; monotonicity permits equality but never a rise.
    if end_elevation is not None and np.isfinite(end_elevation) and float(end_elevation) < float(start_elevation):
        slope = (float(end_elevation) - float(start_elevation)) / float(delta_station)
    else:
        slope = 0.0

    for idx in range(start_index, end_index + 1):
        distance_from_start = start_station - path_stations[idx]
        if not np.isfinite(distance_from_start):
            distance_from_start = float(idx - start_index)
        smoothed_elevation = float(start_elevation) + float(slope) * float(distance_from_start)
        # At confluences/overlaps, retain the lower candidate so the assembled
        # network surface cannot be raised by another path traversal.
        if np.isfinite(path_smoothed[idx]):
            path_smoothed[idx] = min(float(path_smoothed[idx]), smoothed_elevation)
        else:
            path_smoothed[idx] = smoothed_elevation


def _anchor_interpolated_bank_surface_to_cell_observations(
    observed_cell_minimum_bank_elevations: np.ndarray,
    interpolated_bank_elevations: np.ndarray,
    reach_fractions: np.ndarray,
    reach_length: float,
    downstream_control_elevation: float,
    minimum_grade: float = MIN_SLOPE,
    upstream_control_ceiling: float | None = None,
    thalweg_elevations: np.ndarray | None = None,
    lower_bound: float = -np.inf,
    upper_bound: float = np.inf,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a downstream-monotonic surface using cell observations as anchors.

    Cells must be ordered from upstream to downstream. At each cell, ARC first
    predicts an elevation from the active upstream anchor and slope. Before an
    observation can become an anchor, it must fall within the reach-level
    ``lower_bound`` and ``upper_bound`` and, when a finite thalweg is available,
    it must be detectably higher than that cell's thalweg. If the filtered
    observation is lower than the prediction, the observation becomes a new
    anchor. ARC recalculates the grade between the nearest upstream anchor and
    that new low point and rewrites every cell in that interval using the
    fitted grade. At the new anchor, ARC then resets the outgoing grade toward
    the fixed reach-outlet control. A later lower observation repeats the
    process from the most recently accepted anchor. The monotonic cap remains
    in place as a defensive constraint, so every accepted anchor and
    interpolated cell supplies ``minimum_grade`` from the prior cell. An
    outlet-feasibility floor also prevents an observation from pulling the
    extrapolated surface below the network outlet. Thus a valid lower
    observation adjusts its approaching segment instead of creating an
    immediate drop or propagating an unbounded steep grade downstream. For a
    non-headwater reach, ``upstream_control_ceiling`` also prevents the first
    cell from rising above the incoming network control.

    Returns
    -------
    tuple
        ``(surface, outgoing_grades, observation_anchor_mask)``. The outgoing
        grade at a cell is the slope used to predict the next downstream cell.
    """
    observed = np.asarray(
        observed_cell_minimum_bank_elevations,
        dtype=np.float64,
    )
    interpolated = np.asarray(interpolated_bank_elevations, dtype=np.float64)
    fractions = np.asarray(reach_fractions, dtype=np.float64)
    if observed.shape != interpolated.shape or observed.shape != fractions.shape:
        raise ValueError(
            "Observed elevations, interpolated elevations, and reach fractions "
            "must have the same shape."
        )
    thalwegs = None
    if thalweg_elevations is not None:
        thalwegs = np.asarray(thalweg_elevations, dtype=np.float64)
        if thalwegs.shape != observed.shape:
            raise ValueError(
                "Thalweg elevations and observed bank elevations must have "
                "the same shape."
            )
    lower_bound = float(lower_bound)
    upper_bound = float(upper_bound)
    if lower_bound > upper_bound:
        raise ValueError("Bank-elevation lower_bound cannot exceed upper_bound.")
    if observed.size == 0:
        return (
            interpolated.copy(),
            np.empty_like(interpolated),
            np.zeros(0, dtype=bool),
        )

    # Build the anchor-eligibility mask once so the first cell and all later
    # cells use exactly the same reach outlier and thalweg tests. Non-finite
    # thalwegs do not disqualify an otherwise valid bank observation because
    # there is no local bed elevation against which it can be checked.
    valid_observation_mask = (
        np.isfinite(observed)
        & (observed >= lower_bound)
        & (observed <= upper_bound)
    )
    if thalwegs is not None:
        finite_thalweg_mask = np.isfinite(thalwegs)
        valid_observation_mask &= (
            ~finite_thalweg_mask
            | (
                (observed > thalwegs)
                & ~np.isclose(observed, thalwegs)
            )
        )

    reach_length = float(reach_length)
    if not np.isfinite(reach_length) or reach_length <= 0.0:
        reach_length = 1.0
    minimum_grade = float(max(minimum_grade, 0.0))
    stations = np.maximum.accumulate(
        np.clip(fractions, 0.0, 1.0) * reach_length
    )

    surface = np.empty_like(interpolated)
    outgoing_grades = np.full_like(interpolated, minimum_grade)
    anchor_mask = np.zeros(observed.size, dtype=bool)

    def _grade_to_outlet(anchor_station: float, anchor_elevation: float) -> float:
        remaining_distance = reach_length - anchor_station
        if remaining_distance <= 0.0:
            return minimum_grade
        return float(
            max(
                (anchor_elevation - downstream_control_elevation)
                / remaining_distance,
                minimum_grade,
            )
        )

    def _minimum_outlet_feasible_elevation(station: float) -> float:
        """Return the lowest anchor that can still reach the outlet safely."""
        remaining_distance = max(reach_length - station, 0.0)
        return float(
            downstream_control_elevation
            + minimum_grade * remaining_distance
        )

    anchor_station = float(stations[0])
    anchor_elevation = float(interpolated[0])
    if valid_observation_mask[0] and float(observed[0]) < anchor_elevation:
        anchor_elevation = float(observed[0])
        anchor_mask[0] = True
    if (
        upstream_control_ceiling is not None
        and np.isfinite(upstream_control_ceiling)
        and anchor_elevation > float(upstream_control_ceiling)
    ):
        anchor_elevation = float(upstream_control_ceiling)
        # A fully capped observation did not alter the interpolation and is not
        # treated as an active anchor in the output diagnostics.
        anchor_mask[0] = not np.isclose(
            anchor_elevation,
            float(interpolated[0]),
        )
    # A first-cell observation cannot be used below the elevation required to
    # reach the fixed outlet while retaining the minimum downstream grade.
    anchor_elevation = max(
        anchor_elevation,
        _minimum_outlet_feasible_elevation(anchor_station),
    )
    surface[0] = anchor_elevation
    anchor_index = 0
    active_grade = _grade_to_outlet(anchor_station, anchor_elevation)
    outgoing_grades[0] = active_grade

    for cell_index in range(1, observed.size):
        station = float(stations[cell_index])
        distance_from_anchor = max(station - anchor_station, 0.0)
        predicted_elevation = (
            anchor_elevation - active_grade * distance_from_anchor
        )

        # An observation below the active line becomes the next anchor only
        # after it passes the reach bounds and thalweg filter. Comparing with
        # the active line (rather than the original baseline) ensures each new
        # anchor is evaluated against the grade established by the prior one.
        cell_distance = max(
            station - float(stations[cell_index - 1]),
            0.0,
        )
        maximum_monotonic_elevation = (
            float(surface[cell_index - 1])
            - minimum_grade * cell_distance
        )

        # Do not accept a low anchor beneath the elevation from which the
        # fixed outlet can still be reached at ``minimum_grade``. Clipping an
        # extreme observation to this floor preserves its lowering influence
        # without allowing the downstream interpolation to run below zero or
        # below the network-estimated outlet merely through extrapolation.
        feasible_observation_elevation = max(
            float(observed[cell_index]),
            _minimum_outlet_feasible_elevation(station),
        ) if valid_observation_mask[cell_index] else np.nan
        observation_used = (
            valid_observation_mask[cell_index]
            and feasible_observation_elevation < predicted_elevation
        )

        if observation_used:
            new_anchor_elevation = min(
                feasible_observation_elevation,
                maximum_monotonic_elevation,
            )
            anchor_distance = station - anchor_station

            if anchor_distance > 0.0:
                # Fit the full interval to the newly discovered low point.
                # Rewriting this interval distributes the elevation change
                # between anchors instead of preserving a sharp one-cell step.
                active_grade = max(
                    (anchor_elevation - new_anchor_elevation)
                    / anchor_distance,
                    minimum_grade,
                )
                segment_slice = slice(anchor_index, cell_index + 1)
                distances_from_upstream_anchor = np.maximum(
                    stations[segment_slice] - anchor_station,
                    0.0,
                )
                # Rewrite the complete segment as one vectorized operation;
                # accepted-anchor intervals do not overlap except at their
                # endpoint, so the complete pass remains linear in cell count.
                surface[segment_slice] = (
                    anchor_elevation
                    - active_grade * distances_from_upstream_anchor
                )
                # The approaching grade applies through the cell immediately
                # upstream of the new anchor. The new anchor receives a fresh
                # outgoing grade toward the reach outlet below.
                outgoing_grades[anchor_index:cell_index] = active_grade
            else:
                # Repeated station values provide no distance over which to
                # calculate a new grade. Retain the active grade while still
                # accepting the lower elevation as the local anchor.
                surface[cell_index] = new_anchor_elevation

            # Make the accepted low point the nearest upstream anchor for the
            # next segment, but do not extrapolate its steep approaching grade.
            # Reset the active grade toward the fixed outlet so the surface is
            # bounded. A later low observation will back-fit only the interval
            # beginning here and will then perform the same outlet reset.
            anchor_index = cell_index
            anchor_station = station
            anchor_elevation = float(surface[cell_index])
            anchor_mask[cell_index] = True
            active_grade = _grade_to_outlet(
                anchor_station,
                anchor_elevation,
            )
        else:
            surface[cell_index] = min(
                predicted_elevation,
                maximum_monotonic_elevation,
            )
        outgoing_grades[cell_index] = active_grade

    return surface, outgoing_grades, anchor_mask

def _reach_length(reach_id: int,
                  reach_network_graph: nx.DiGraph,) -> float:
    reach_length = float(
        reach_network_graph.nodes[reach_id].get('length', 1.0)
    )
    if not np.isfinite(reach_length) or reach_length <= 0.0:
        return 1.0
    return reach_length

def _estimate_network_smoothed_reach_min_bank_elevations(
    reach_network_graph: nx.DiGraph,
    reach_min_bank_elevation_dict: dict[int, float],
    reach_max_bank_elevation_dict: dict[int, float],
    reach_cell_bank_observations: dict[int, dict] | None = None,
) -> dict[int, float]:
    """Place each reach minimum at its outlet and assign connected grades.

    Each value in ``reach_min_bank_elevation_dict`` is treated as an observation
    at the downstream endpoint of that reach. Missing outlet observations are
    interpolated along headwater-to-outlet paths. Outlet controls are lowered
    where necessary so every connected reach falls downstream by at least
    ``MIN_SLOPE``. Equal observed minima therefore receive a very small
    numerical decline instead of a zero grade.

    A reach with both upstream and downstream network context obtains its grade
    from the difference between the lowest incoming outlet and its own outlet.
    Headwaters use a separate upstream-to-downstream initialization because
    they have no incoming network control. The highest filtered raw bank is
    assigned to the upstream endpoint, the headwater's minimum bank remains at
    its outlet, and their difference over graph reach length defines the
    initial grade. A headwater without usable observations falls back to its
    immediate downstream neighbor's grade. An isolated reach, with neither a
    predecessor nor successor, is oriented from its filtered maximum toward
    its filtered minimum and uses those endpoints over graph reach length to
    define both its flow direction and initial grade. An outlet uses the lowest incoming
    predecessor minimum as its upstream endpoint and its own lowest filtered
    bank as its downstream endpoint. Their difference over graph reach length
    supplies the outlet grade. If those controls would rise downstream, the
    local outlet minimum is lowered only enough to preserve ``MIN_SLOPE``. The
    selected grade is stored on the graph node as ``bank_elevation_grade`` for
    :func:`_interpolate_reach_bank_elevation_surface`.

    When ``reach_cell_bank_observations`` is supplied, the function also
    interpolates each reach to its ordered cells and walks those cells from
    upstream to downstream. An observed minimum below the active interpolation
    becomes a new anchor. The slope from the nearest upstream anchor to the new
    low point is recalculated and applied across that interval. From the new
    anchor, the outgoing slope is reset toward the fixed reach outlet until
    another lower observation establishes the next segment. Every consecutive
    cell is constrained to fall by at least ``MIN_SLOPE``, and accepted anchors
    are kept high enough to reach the outlet without crossing beneath it. The
    anchored surface, anchor mask, and per-cell outgoing grades are stored on
    the corresponding graph node for the final mapping pass in
    :func:`_smooth_reach_bank_elevations`.

    Returns
    -------
    dict
        Reach ID to the smoothed elevation at that reach's downstream endpoint.
    """
    if len(reach_min_bank_elevation_dict) == 0:
        return {}
    if reach_network_graph.number_of_nodes() == 0:
        raise ValueError(
            'The reach bank-elevation network was empty, so network smoothing '
            'could not be performed.'
        )

    # Use ARC's numerical slope floor when two observed reach minima are equal.
    minimum_grade = MIN_SLOPE

    # Start with every finite observed minimum at the outlet of its reach.
    outlet_elevation_candidates: dict[int, list[float]] = {}
    for reach_id, elevation in reach_min_bank_elevation_dict.items():
        if reach_id in reach_network_graph and np.isfinite(elevation):
            outlet_elevation_candidates.setdefault(int(reach_id), []).append(
                float(elevation)
            )

    headwaters = [
        int(node)
        for node in reach_network_graph.nodes
        if reach_network_graph.in_degree(node) == 0
    ]
    if len(headwaters) == 0:
        headwaters = [int(node) for node in reach_network_graph.nodes]
    isolated_reaches = {
        int(node)
        for node in reach_network_graph.nodes
        if (
            reach_network_graph.in_degree(node) == 0
            and reach_network_graph.out_degree(node) == 0
        )
    }

    # Fill graph reaches without a local observation by interpolating outlet
    # elevations along every downstream path. Endpoint extrapolation uses the
    # first or last observed path slope, which gives unobserved terminal
    # reaches the same grade as their nearest observed neighbor.
    for headwater in headwaters:
        path = [headwater]
        visited: set[int] = set()
        current_node = headwater
        while (
            reach_network_graph.out_degree(current_node) > 0
            and current_node not in visited
        ):
            visited.add(current_node)
            successors = list(reach_network_graph.successors(current_node))
            if len(successors) == 0:
                break
            current_node = int(successors[0])
            path.append(current_node)

        path_stations = np.cumsum(
            np.asarray([_reach_length(node, reach_network_graph) for node in path], dtype=np.float64)
        )
        path_observed = np.asarray(
            [
                float(reach_min_bank_elevation_dict.get(node, np.nan))
                for node in path
            ],
            dtype=np.float64,
        )
        observed_indices = np.flatnonzero(np.isfinite(path_observed))
        if observed_indices.size == 0:
            continue

        if observed_indices.size == 1:
            only_index = int(observed_indices[0])
            only_station = float(path_stations[only_index])
            only_elevation = float(path_observed[only_index])
            path_filled = (
                only_elevation
                - minimum_grade * (path_stations - only_station)
            )
        else:
            observed_stations = path_stations[observed_indices]
            observed_elevations = path_observed[observed_indices]
            path_filled = np.interp(
                path_stations,
                observed_stations,
                observed_elevations,
            )

            first_delta = float(observed_stations[1] - observed_stations[0])
            first_grade = (
                max(
                    (
                        float(observed_elevations[0])
                        - float(observed_elevations[1])
                    )
                    / first_delta,
                    minimum_grade,
                )
                if first_delta > 0.0
                else minimum_grade
            )
            leading_mask = path_stations < observed_stations[0]
            path_filled[leading_mask] = (
                float(observed_elevations[0])
                + first_grade
                * (float(observed_stations[0]) - path_stations[leading_mask])
            )

            last_delta = float(observed_stations[-1] - observed_stations[-2])
            last_grade = (
                max(
                    (
                        float(observed_elevations[-2])
                        - float(observed_elevations[-1])
                    )
                    / last_delta,
                    minimum_grade,
                )
                if last_delta > 0.0
                else minimum_grade
            )
            trailing_mask = path_stations > observed_stations[-1]
            path_filled[trailing_mask] = (
                float(observed_elevations[-1])
                - last_grade
                * (path_stations[trailing_mask] - float(observed_stations[-1]))
            )

        for path_index, node in enumerate(path):
            if np.isfinite(path_filled[path_index]):
                outlet_elevation_candidates.setdefault(int(node), []).append(
                    float(path_filled[path_index])
                )

    # A confluence can place the same downstream reach on several headwater
    # paths. Retain the lowest candidate so the shared outlet cannot rise along
    # any incoming path.
    smoothed_reach_outlet_elevations: dict[int, float] = {}
    for node, candidates in outlet_elevation_candidates.items():
        finite_candidates = np.asarray(candidates, dtype=np.float64)
        finite_candidates = finite_candidates[np.isfinite(finite_candidates)]
        if finite_candidates.size > 0:
            smoothed_reach_outlet_elevations[int(node)] = float(
                np.nanmin(finite_candidates)
            )

    # Keep every downstream outlet at or below the lowest incoming outlet.
    # Equality is allowed; repeated relaxation also handles a malformed cyclic
    # graph without depending on topological sort.
    for _ in range(max(reach_network_graph.number_of_nodes(), 1)):
        changed = False
        for downstream_reach_id in reach_network_graph.nodes:
            downstream_reach_id = int(downstream_reach_id)
            downstream_elevation = smoothed_reach_outlet_elevations.get(
                downstream_reach_id
            )
            predecessor_elevations = [
                smoothed_reach_outlet_elevations[int(predecessor_id)]
                for predecessor_id in reach_network_graph.predecessors(
                    downstream_reach_id
                )
                if int(predecessor_id) in smoothed_reach_outlet_elevations
            ]
            if downstream_elevation is None or len(predecessor_elevations) == 0:
                continue
            incoming_outlet_elevation = float(np.nanmin(predecessor_elevations))
            maximum_outlet_elevation = (
                incoming_outlet_elevation
                - minimum_grade * _reach_length(downstream_reach_id, reach_network_graph)
            )
            if downstream_elevation > maximum_outlet_elevation:
                smoothed_reach_outlet_elevations[downstream_reach_id] = float(
                    maximum_outlet_elevation
                )
                changed = True
        if not changed:
            break

    # Calculate a grade for every non-headwater reach from its lowest incoming
    # outlet to its own outlet. Interior reaches retain these direct grades;
    # outlet reaches are handled explicitly below using their filtered minima.
    reach_grades: dict[int, float] = {}
    for reach_id, outlet_elevation in smoothed_reach_outlet_elevations.items():
        predecessor_elevations = [
            smoothed_reach_outlet_elevations[int(predecessor_id)]
            for predecessor_id in reach_network_graph.predecessors(reach_id)
            if int(predecessor_id) in smoothed_reach_outlet_elevations
        ]
        if len(predecessor_elevations) == 0:
            continue
        incoming_outlet_elevation = float(np.nanmin(predecessor_elevations))
        reach_grades[reach_id] = max(
            (incoming_outlet_elevation - outlet_elevation)
            / _reach_length(reach_id, reach_network_graph),
            minimum_grade,
        )

    # Outlet reaches have an upstream endpoint supplied by their immediate
    # predecessor but no downstream-neighbor control. Use the outlet reach's
    # own lowest filtered bank as the missing downstream endpoint and calculate
    # a unique linear grade between those two elevations. The shared cell pass
    # later walks this line upstream-to-downstream and promotes any lower
    # intermediate filtered banks to anchors, matching headwater processing.
    outlets = [
        int(node)
        for node in reach_network_graph.nodes
        if reach_network_graph.out_degree(node) == 0
    ]
    for outlet_reach_id in outlets:
        # Isolated nodes also satisfy the graph definition of an outlet, but
        # they have neither incoming nor outgoing controls and receive their
        # own maximum-to-minimum processing below.
        if outlet_reach_id in isolated_reaches:
            continue
        predecessor_ids = [
            int(predecessor_id)
            for predecessor_id in reach_network_graph.predecessors(
                outlet_reach_id
            )
            if int(predecessor_id) in smoothed_reach_outlet_elevations
        ]
        if len(predecessor_ids) == 0:
            reach_grades[outlet_reach_id] = minimum_grade
            continue

        incoming_outlet_elevation = float(
            np.nanmin(
                [
                    smoothed_reach_outlet_elevations[predecessor_id]
                    for predecessor_id in predecessor_ids
                ]
            )
        )
        filtered_outlet_minimum = reach_min_bank_elevation_dict.get(
            outlet_reach_id
        )
        if (
            filtered_outlet_minimum is not None
            and np.isfinite(filtered_outlet_minimum)
        ):
            reach_length = _reach_length(
                outlet_reach_id,
                reach_network_graph,
            )
            # A local minimum above the incoming control would create a rise.
            # Lower only that infeasible endpoint enough to retain the required
            # numerical downstream fall; otherwise use the filtered value
            # directly as requested.
            maximum_monotonic_outlet = (
                incoming_outlet_elevation - minimum_grade * reach_length
            )
            outlet_minimum_to_use = min(
                float(filtered_outlet_minimum),
                maximum_monotonic_outlet,
            )
            outlet_grade = max(
                (
                    incoming_outlet_elevation
                    - outlet_minimum_to_use
                )
                / reach_length,
                minimum_grade,
            )
            reach_grades[outlet_reach_id] = outlet_grade
            smoothed_reach_outlet_elevations[outlet_reach_id] = (
                outlet_minimum_to_use
            )
            node_data = reach_network_graph.nodes[outlet_reach_id]
            node_data["outlet_upstream_bank_elevation"] = (
                incoming_outlet_elevation
            )
            node_data["outlet_filtered_minimum_bank_elevation"] = float(
                filtered_outlet_minimum
            )
            node_data["outlet_bank_elevation_to_use"] = (
                outlet_minimum_to_use
            )
            node_data["bank_elevation_grade_source"] = (
                "outlet_upstream_minimum_to_filtered_minimum"
            )
            continue

        # If an outlet has no usable filtered bank, retain the former slope
        # inheritance so a direct caller or unsampled terminal reach still
        # receives a deterministic downstream endpoint.
        inherited_grades = [
            reach_grades[predecessor_id]
            for predecessor_id in predecessor_ids
            if predecessor_id in reach_grades
        ]
        outlet_grade = (
            max(float(np.nanmedian(inherited_grades)), minimum_grade)
            if inherited_grades
            else minimum_grade
        )
        reach_grades[outlet_reach_id] = outlet_grade
        smoothed_reach_outlet_elevations[outlet_reach_id] = (
            incoming_outlet_elevation - outlet_grade
            * _reach_length(outlet_reach_id, reach_network_graph)
        )
        reach_network_graph.nodes[outlet_reach_id][
            "bank_elevation_grade_source"
        ] = "upstream_neighbor_fallback"

    # A graph-isolated stream is simultaneously a headwater and an outlet in
    # degree terms, but processing it through both branches obscures which
    # endpoint controls its surface. Handle it exactly once: the filtered
    # maximum is the upstream endpoint, the filtered minimum is the downstream
    # endpoint, and the graph length converts their drop into a positive grade.
    # The shared per-cell pass below then performs all normal low-bank anchoring.
    for isolated_reach_id in isolated_reaches:
        maximum_bank_elevation = reach_max_bank_elevation_dict.get(
            isolated_reach_id
        )
        minimum_bank_elevation = reach_min_bank_elevation_dict.get(
            isolated_reach_id
        )
        if (
            maximum_bank_elevation is None
            or minimum_bank_elevation is None
            or not np.isfinite(maximum_bank_elevation)
            or not np.isfinite(minimum_bank_elevation)
        ):
            # Without both endpoints the flow direction cannot be inferred.
            # Retain a finite minimum-grade fallback when a local minimum is
            # available; the existing missing-control checks handle no-minimum
            # reaches consistently with the rest of the network.
            if (
                minimum_bank_elevation is not None
                and np.isfinite(minimum_bank_elevation)
            ):
                reach_grades[isolated_reach_id] = minimum_grade
                smoothed_reach_outlet_elevations[isolated_reach_id] = float(
                    minimum_bank_elevation
                )
                reach_network_graph.nodes[isolated_reach_id][
                    "bank_elevation_grade_source"
                ] = "isolated_minimum_grade_fallback"
            continue

        maximum_bank_elevation = float(maximum_bank_elevation)
        minimum_bank_elevation = float(minimum_bank_elevation)
        isolated_grade = max(
            (
                maximum_bank_elevation
                - minimum_bank_elevation
            )
            / _reach_length(isolated_reach_id, reach_network_graph),
            minimum_grade,
        )
        reach_grades[isolated_reach_id] = isolated_grade
        smoothed_reach_outlet_elevations[isolated_reach_id] = (
            minimum_bank_elevation
        )
        node_data = reach_network_graph.nodes[isolated_reach_id]
        node_data["isolated_upstream_bank_elevation"] = (
            maximum_bank_elevation
        )
        node_data["isolated_downstream_bank_elevation"] = (
            minimum_bank_elevation
        )
        node_data["bank_elevation_flow_direction"] = (
            "ordered_filtered_maximum_to_minimum"
        )
        node_data["bank_elevation_grade_source"] = (
            "isolated_filtered_maximum_to_minimum"
        )

    # Headwaters receive a unique upstream-to-downstream initialization. Their
    # filtered maximum is treated as the upstream endpoint, while the minimum
    # observation already assigned to the reach outlet remains the downstream
    # endpoint. Setting this grade before cell interpolation creates the linear,
    # monotonically decreasing baseline that the shared anchor routine will
    # subsequently lower wherever an eligible raw bank falls beneath it.
    for headwater_reach_id in headwaters:
        # Isolated reaches were completely initialized above and must not be
        # overwritten by the ordinary headwater branch.
        if headwater_reach_id in isolated_reaches:
            continue
        maximum_bank_elevation = reach_max_bank_elevation_dict.get(
            headwater_reach_id
        )
        minimum_bank_elevation = reach_min_bank_elevation_dict.get(
            headwater_reach_id
        )
        if (
            maximum_bank_elevation is not None
            and minimum_bank_elevation is not None
            and np.isfinite(minimum_bank_elevation)
        ):
            minimum_bank_elevation = float(minimum_bank_elevation)
            headwater_grade = max(
                (
                    maximum_bank_elevation
                    - minimum_bank_elevation
                )
                / _reach_length(headwater_reach_id, reach_network_graph),
                minimum_grade,
            )
            reach_grades[headwater_reach_id] = headwater_grade
            node_data = reach_network_graph.nodes[headwater_reach_id]
            node_data["headwater_upstream_bank_elevation"] = float(
                maximum_bank_elevation
            )
            node_data["headwater_outlet_bank_elevation"] = (
                minimum_bank_elevation
            )
            node_data["bank_elevation_grade_source"] = (
                "headwater_filtered_maximum_to_minimum"
            )
            continue

        # Preserve downstream-grade inheritance only as a fallback for a
        # headwater with no finite filtered raw bank observations.
        successors = [
            int(successor_id)
            for successor_id in reach_network_graph.successors(
                headwater_reach_id
            )
        ]
        downstream_grades = [
            reach_grades[successor_id]
            for successor_id in successors
            if successor_id in reach_grades
        ]
        reach_grades[headwater_reach_id] = (
            max(float(np.nanmedian(downstream_grades)), minimum_grade)
            if len(downstream_grades) > 0
            else minimum_grade
        )
        reach_network_graph.nodes[headwater_reach_id][
            "bank_elevation_grade_source"
        ] = "downstream_neighbor_fallback"

    for reach_id, grade in reach_grades.items():
        reach_network_graph.nodes[reach_id]['bank_elevation_grade'] = float(
            grade
        )
    for reach_id in smoothed_reach_outlet_elevations:
        reach_network_graph.nodes[reach_id]['bank_outlet_elevation'] = float(
            smoothed_reach_outlet_elevations[reach_id]
        )

    missing_reaches = sorted(
        int(reach_id)
        for reach_id in reach_min_bank_elevation_dict
        if reach_id not in smoothed_reach_outlet_elevations
    )
    if missing_reaches:
        raise ValueError(
            'Network smoothing did not produce bank elevations for reach_id '
            + ', '.join(map(str, missing_reaches[:10]))
            + ('...' if len(missing_reaches) > 10 else '')
            + '.'
        )

    # Cell observations are optional so direct reach-control callers retain
    # the original API behavior. The production smoothing workflow supplies
    # them after ordering every sampled reach upstream-to-downstream.
    if reach_cell_bank_observations:
        for reach_id, cell_observations in reach_cell_bank_observations.items():
            reach_id = int(reach_id)
            if (
                reach_id not in reach_network_graph
                or reach_id not in smoothed_reach_outlet_elevations
            ):
                continue
            ordered_coordinates = np.asarray(
                cell_observations.get("ordered_coordinates", []),
                dtype=np.float64,
            )
            observed_elevations = np.asarray(
                cell_observations.get("observed_elevations", []),
                dtype=np.float64,
            )
            thalweg_elevations = np.asarray(
                cell_observations.get(
                    "thalweg_elevations",
                    np.full(observed_elevations.shape, np.nan),
                ),
                dtype=np.float64,
            )
            (
                baseline_surface,
                reach_fractions,
                _downstream_reach_id,
                downstream_control,
            ) = _interpolate_reach_bank_elevation_surface(
                reach_network_graph,
                reach_id,
                ordered_coordinates,
                smoothed_reach_outlet_elevations,
            )
            reach_length = _reach_length(reach_id, reach_network_graph)
            predecessor_controls = [
                smoothed_reach_outlet_elevations[int(predecessor_id)]
                for predecessor_id in reach_network_graph.predecessors(reach_id)
                if int(predecessor_id) in smoothed_reach_outlet_elevations
            ]
            upstream_control_ceiling = (
                float(np.nanmin(predecessor_controls))
                if predecessor_controls
                else None
            )
            anchored_surface, outgoing_grades, anchor_mask = (
                _anchor_interpolated_bank_surface_to_cell_observations(
                    observed_elevations,
                    baseline_surface,
                    reach_fractions,
                    reach_length,
                    downstream_control,
                    minimum_grade,
                    upstream_control_ceiling,
                    thalweg_elevations,
                    float(cell_observations.get("lower_bound", -np.inf)),
                    float(cell_observations.get("upper_bound", np.inf)),
                )
            )
            node_data = reach_network_graph.nodes[reach_id]
            node_data["baseline_cell_bank_elevation_surface"] = baseline_surface
            node_data["observed_anchored_cell_bank_elevation_surface"] = (
                anchored_surface
            )
            node_data["cell_bank_elevation_reach_fractions"] = reach_fractions
            node_data["cell_bank_elevation_outgoing_grades"] = outgoing_grades
            node_data["cell_bank_elevation_observation_anchor_mask"] = (
                anchor_mask
            )

    return smoothed_reach_outlet_elevations


def _interpolate_reach_bank_elevation_surface(
    reach_network_graph: nx.DiGraph,
    reach_id: int,
    ordered_coordinates: np.ndarray,
    network_smoothed_reach_min_bank_elevations: dict[int, float],
) -> tuple[np.ndarray, np.ndarray, int | None, float]:
    """Interpolate network bank elevations at the cross sections of one reach.

    The graph smoother treats each reach minimum as an observation at that
    reach's downstream endpoint and stores the reach grade on the graph node.
    This helper reconstructs the upstream endpoint by adding
    ``grade * reach_length`` to the outlet control, then evaluates that line at
    every ordered cross section. Headwater and outlet grades have already been
    initialized from their filtered maximum and reach minimum, or inherited
    from downstream when filtered observations are unavailable. Outlet grades
    have already been calculated from their incoming minimum and their own
    filtered downstream minimum. Isolated grades have been calculated directly
    between their filtered maximum and minimum.

    Returns
    -------
    tuple
        ``(surface_elevations, reach_fractions, downstream_reach_id,
        outlet_control_elevation)``. The surface and fractions follow the
        supplied upstream-to-downstream cross-section order, and the returned
        control is the minimum at the downstream end of the current reach.
    """
    # Normalize the caller's station/distance values once so all following
    # indexing and arithmetic use a predictable floating-point array.
    ordered_coordinates = np.asarray(ordered_coordinates, dtype=np.float64)

    # An empty reach has no surface to build. Return all four result components
    # in their empty/missing forms so callers do not need a special code path.
    if ordered_coordinates.size == 0:
        return (
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.float64),
            None,
            np.nan,
        )

    # Every sampled reach must have received an outlet control from the
    # headwater-to-outlet network smoothing pass before interpolation begins.
    if reach_id not in network_smoothed_reach_min_bank_elevations:
        raise ValueError(
            'Network smoothing did not return a bank elevation for reach_id '
            + str(reach_id)
            + '.'
        )

    # The current reach's smoothed minimum is explicitly anchored at its
    # downstream endpoint rather than being applied to its upstream cell.
    downstream_control_elevation = float(
        network_smoothed_reach_min_bank_elevations[reach_id]
    )

    reach_length = float(
        _reach_length(reach_id, reach_network_graph)
    )
    if not np.isfinite(reach_length) or reach_length <= 0.0:
        reach_length = 1.0
    reach_grade = float(
        reach_network_graph.nodes[reach_id].get(
            'bank_elevation_grade',
            0.0,
        )
    )
    if not np.isfinite(reach_grade) or reach_grade < 0.0:
        reach_grade = 0.0
    upstream_control_elevation = (
        downstream_control_elevation + reach_grade * reach_length
    )

    # The successor is returned for diagnostics only. Its outlet minimum does
    # not anchor the current reach because the current minimum is now located
    # at the current reach's own outlet.
    successors = list(reach_network_graph.successors(reach_id))
    downstream_reach_id = int(successors[0]) if len(successors) > 0 else None

    # Convert the physical stations into dimensionless 0-to-1 interpolation
    # fractions so the same calculation works for reaches of every length.
    if ordered_coordinates.size == 1:
        # A lone sampled stream cell represents the reach outlet so the reach
        # minimum is assigned to it, consistent with the outlet-control model.
        reach_fractions = np.ones(1, dtype=np.float64)
    else:
        # ``ordered_coordinates`` may increase or decrease because the mean
        # stream axis has no inherent sign. Dividing by the signed span yields
        # 0 at the first (upstream) section and 1 at the last (downstream)
        # section in either case, while retaining unequal cell spacing.
        coordinate_span = float(ordered_coordinates[-1] - ordered_coordinates[0])
        if np.isfinite(coordinate_span) and not np.isclose(coordinate_span, 0.0):
            reach_fractions = (
                (ordered_coordinates - float(ordered_coordinates[0]))
                / coordinate_span
            )
            reach_fractions = np.clip(reach_fractions, 0.0, 1.0)
        else:
            # Duplicate/degenerate projected coordinates still have a stable
            # stream order, so distribute them uniformly in that order.
            reach_fractions = np.linspace(
                0.0,
                1.0,
                ordered_coordinates.size,
                dtype=np.float64,
            )

    # Linear interpolation descends from the grade-derived upstream endpoint to
    # the observed/smoothed minimum at the outlet of the current reach.
    surface_elevations = (
        upstream_control_elevation
        + reach_fractions
        * (downstream_control_elevation - upstream_control_elevation)
    )
    # Eliminate any floating-point or coordinate-order artifact that could
    # introduce a local rise between consecutive downstream stream cells.
    surface_elevations = np.minimum.accumulate(surface_elevations)
    # Return the current reach's outlet control as well as the surface. The
    # caller stores both endpoint controls and every cell fraction as metadata.
    return (
        np.asarray(surface_elevations, dtype=np.float64),
        np.asarray(reach_fractions, dtype=np.float64),
        downstream_reach_id,
        float(downstream_control_elevation),
    )


def _nearest_current_cell(reference_entries: list[dict], rows: np.array, cols: np.array) -> int | None:
    """
    Find the current-reach cell closest to any cell in a neighboring reach.
    This identifies a topological endpoint without relying on DEM elevation,
    which may be flat or quantized over an entire FABDEM reach.
    """
    if len(reference_entries) == 0:
        return None
    reference_rows = np.asarray(
        [int(entry['row']) for entry in reference_entries],
        dtype=np.float64,
    )
    reference_cols = np.asarray(
        [int(entry['col']) for entry in reference_entries],
        dtype=np.float64,
    )
    best_index = None
    best_distance_squared = np.inf
    for entry_index, (row, col) in enumerate(zip(rows, cols)):
        distances_squared = (
            np.square(reference_rows - float(row))
            + np.square(reference_cols - float(col))
        )
        candidate_distance_squared = float(np.nanmin(distances_squared))
        if candidate_distance_squared < best_distance_squared:
            best_distance_squared = candidate_distance_squared
            best_index = entry_index
    return best_index

def _order_reach_stream_cells_from_network(
    reach_network_graph: nx.DiGraph,
    reach_id: int,
    reach_entries: list[dict],
    grouped_reach_entries: dict[int, list[dict]],
    fallback_order: np.ndarray,
    raw_bank_elevations: np.ndarray,
    dx: float,
    dy: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Order a reach's sampled raster cells from upstream to downstream.

    The directed reach graph identifies which neighboring reach lies
    downstream. Within the current reach, an 8-connected raster graph measures
    along-channel distance from the cell touching that downstream reach. This
    avoids assigning interpolation values using an unsigned straight-line axis,
    which can reverse flat reaches or scramble curved reaches.

    ``fallback_order`` is used only when no connected neighboring reach is
    sampled. In that case the older endpoint-elevation check supplies the sign.
    The returned stations always start at zero upstream and increase toward the
    downstream end.
    """
    # A zero- or one-cell reach is already ordered and has no measurable
    # internal along-stream distance.
    entry_count = len(reach_entries)
    if entry_count <= 1:
        return np.arange(entry_count, dtype=np.int64), np.zeros(entry_count, dtype=np.float64)

    # Split the sampled records into compact raster-coordinate arrays. The
    # lookup maps a neighboring (row, col) back to its index in ``reach_entries``.
    rows = np.asarray([int(entry['row']) for entry in reach_entries], dtype=np.int64)
    cols = np.asarray([int(entry['col']) for entry in reach_entries], dtype=np.int64)
    cell_indices_by_location: dict[tuple[int, int], list[int]] = {}
    for entry_index, (row, col) in enumerate(zip(rows, cols)):
        cell_indices_by_location.setdefault((int(row), int(col)), []).append(entry_index)

    # Build the actual raster-cell path for this reach. Edge weights retain
    # physical raster spacing, including the longer distance across diagonals.
    cell_graph = nx.Graph()
    cell_graph.add_nodes_from(range(entry_count))
    for entry_index, (row, col) in enumerate(zip(rows, cols)):
        for row_offset in (-1, 0, 1):
            for col_offset in (-1, 0, 1):
                if row_offset == 0 and col_offset == 0:
                    continue
                neighbor_indices = cell_indices_by_location.get(
                    (int(row + row_offset), int(col + col_offset)),
                    [],
                )
                step_distance = math.hypot(
                    float(col_offset) * float(dx),
                    float(row_offset) * float(dy),
                )
                for neighbor_index in neighbor_indices:
                    cell_graph.add_edge(
                        entry_index,
                        int(neighbor_index),
                        length=step_distance,
                    )

    # For a normal reach, the raster cell nearest its graph successor is the
    # downstream endpoint from which within-reach path distances are measured.
    downstream_anchor = None
    for successor_id in reach_network_graph.successors(reach_id):
        downstream_anchor = _nearest_current_cell(
            grouped_reach_entries.get(int(successor_id), []),
            rows, 
            cols
        )
        if downstream_anchor is not None:
            break

    upstream_anchor = None
    if downstream_anchor is None:
        # Outlet reaches have no successor. Locate their upstream end beside a
        # predecessor, then choose the farthest cell on this reach as downstream.
        upstream_reference_entries: list[dict] = []
        for predecessor_id in reach_network_graph.predecessors(reach_id):
            upstream_reference_entries.extend(
                grouped_reach_entries.get(int(predecessor_id), [])
            )
        upstream_anchor = _nearest_current_cell(upstream_reference_entries, rows, cols)
        if upstream_anchor is not None:
            distances_from_upstream = nx.single_source_dijkstra_path_length(
                cell_graph,
                upstream_anchor,
                weight='length',
            )
            if len(distances_from_upstream) > 0:
                downstream_anchor = max(
                    distances_from_upstream,
                    key=distances_from_upstream.get,
                )

    if downstream_anchor is None:
        # A disconnected or isolated reach has no topological endpoint
        # reference. Start with the stable projection order. For a truly
        # isolated reach, explicitly put the filtered maximum before the
        # filtered minimum so its order records the inferred downhill flow
        # direction used by the network estimator.
        order = np.asarray(fallback_order, dtype=np.int64).copy()
        is_isolated_reach = (
            reach_network_graph.in_degree(reach_id) == 0
            and reach_network_graph.out_degree(reach_id) == 0
        )
        ordered_bank_elevations = raw_bank_elevations[order]
        finite_positions = np.flatnonzero(
            np.isfinite(ordered_bank_elevations)
        )
        direction_was_resolved = False
        if is_isolated_reach and finite_positions.size >= 2:
            finite_elevations = ordered_bank_elevations[finite_positions]
            maximum_position = int(
                finite_positions[int(np.nanargmax(finite_elevations))]
            )
            minimum_position = int(
                finite_positions[int(np.nanargmin(finite_elevations))]
            )
            if maximum_position != minimum_position:
                if maximum_position > minimum_position:
                    order = order[::-1]
                direction_was_resolved = True

        # If extrema cannot resolve direction (flat/insufficient observations),
        # retain the prior robust endpoint-mean orientation fallback.
        sample_size = min(10, order.size)
        upstream_slice = raw_bank_elevations[order[:sample_size]]
        downstream_slice = raw_bank_elevations[order[-sample_size:]]
        upstream_finite = upstream_slice[np.isfinite(upstream_slice)]
        downstream_finite = downstream_slice[np.isfinite(downstream_slice)]
        if (
            not direction_was_resolved
            and upstream_finite.size > 0
            and downstream_finite.size > 0
        ):
            if float(np.nanmean(upstream_finite)) < float(np.nanmean(downstream_finite)):
                order = order[::-1]
        return order, np.arange(order.size, dtype=np.float64)

    # Measure each connected cell's shortest along-reach distance to the
    # downstream endpoint. If raster gaps leave a cell disconnected, use its
    # physical straight-line distance to the endpoint as a deterministic
    # fallback rather than dropping the cross section.
    distance_to_downstream = nx.single_source_dijkstra_path_length(
        cell_graph,
        downstream_anchor,
        weight='length',
    )
    anchor_row = float(rows[downstream_anchor])
    anchor_col = float(cols[downstream_anchor])
    distances = np.asarray(
        [
            float(distance_to_downstream.get(entry_index, math.hypot(
                (float(cols[entry_index]) - anchor_col) * float(dx),
                (float(rows[entry_index]) - anchor_row) * float(dy),
            )))
            for entry_index in range(entry_count)
        ],
        dtype=np.float64,
    )

    # Larger distance-to-downstream values are upstream. Convert the descending
    # distances to stations that increase from zero toward the downstream cell.
    order = np.argsort(-distances, kind='stable')
    ordered_distances = distances[order]
    upstream_distance = float(ordered_distances[0])
    ordered_stations = upstream_distance - ordered_distances
    ordered_stations = np.maximum.accumulate(ordered_stations)
    return np.asarray(order, dtype=np.int64), np.asarray(ordered_stations, dtype=np.float64)

def _smooth_reach_bank_elevations(
    sampled_records: list[dict | None],
    params: dict,
    quiet: bool,
) -> None:
    """Create one network-smoothed bank elevation for each sampled section.

    After ARC estimates local bank indices for every sampled cross section, it
    reorders the sections within each reach from upstream to downstream using
    the directed reach topology and an 8-connected path through that reach's
    raster cells. ARC then evaluates local bank-to-bank
    top width at each stream cell, replaces any width outside the 25th-75th
    percentile band with bank locations that match the reach-median width,
    uses that same reach-median width to fill sections whose local bank
    indices remained invalid, and treats the minimum detected bank elevation
    as the downstream endpoint of that reach. The network assigns connected
    reach grades, while each headwater constructs its initial grade between its
    highest filtered raw bank and its outlet minimum. Each outlet constructs
    its grade between the lowest incoming predecessor minimum and its own
    lowest filtered bank. An isolated reach uses its filtered maximum and
    minimum to infer flow direction and construct its initial grade. ARC then
    walks the cells upstream-to-downstream, using lower filtered banks as
    anchors for refitted monotonic segments. The
    interpolated elevation remains only the vertical bathymetry control while
    the filtered local bank indices and top width are preserved for each
    sampled cross section.
    """
    # Source-stream IDs preserve the original reach grouping when processing
    # has reassigned cell COMIDs; otherwise the cell COMID is the reach key.
    source_ids = _CELL_SOURCE_STREAM_IDS if _CELL_SOURCE_STREAM_IDS is not None else _CELL_COMIDS
    x_section = get_cross_section(params['dx'], params['dy'], _DEM, _LAND_COVER, _STREAMS, params)
    grouped_reach_entries: dict[int, list[dict]] = {}
    reach_network_graph, reach_downstream_map = _build_reach_network_graph(
        params.get('s_strmshp_path', ''),
        params.get('s_reach_id_field', ''),
        params.get('s_downstream_reach_id_field', ''),
    )

    # Prepass: restore each sampled profile, ensure it has a local bank result,
    # and gather the location/direction metadata used for reach grouping.
    for i_entry_cell in tqdm.tqdm(range(_CELL_COMIDS.size), total=_CELL_COMIDS.size, disable=quiet):
        sampled_record = sampled_records[i_entry_cell]
        if sampled_record is None:
            continue

        reach_bank_index = None
        if _CELL_REACH_INFLECT_BANK_INDEX is not None:
            reach_bank_index = float(_CELL_REACH_INFLECT_BANK_INDEX[i_entry_cell])
        _replay_precomputed_cross_section(x_section, sampled_record, reach_bank_index=reach_bank_index)

        bank_search_result = _get_bank_search_result_for_smoothing(
            x_section,
            sampled_record,
            params,
            i_entry_cell,
        )
        sampled_record["bank_search_result"] = dict(bank_search_result)

        row = int(sampled_record["row"])
        col = int(sampled_record["col"])
        # Direction is sampled locally but later averaged as an unoriented axis
        # so all cells in the reach can be projected onto one ordering line.
        stream_direction, _ = get_stream_direction_information(
            row,
            col,
            _STREAMS,
            params['i_general_direction_distance'],
        )
        grouped_reach_entries.setdefault(int(source_ids[i_entry_cell]), []).append(
            {
                "entry_index": i_entry_cell,
                "row": row,
                "col": col,
                "stream_direction": float(stream_direction),
            }
        )

    # Reach pass: normalize horizontal bank geometry, order sections along the
    # reach, and reduce the local elevations to one observed reach minimum.
    reach_summaries: dict[int, dict] = {}
    for reach_id, reach_entries in grouped_reach_entries.items():
        if len(reach_entries) == 0:
            continue

        # First standardize the horizontal channel geometry. Valid widths in
        # the central reach band remain local; outliers are rebuilt to the
        # median, and invalid searches receive that same median as a fallback.
        reach_top_width_stats = _apply_reach_top_width_filter(
            x_section,
            sampled_records,
            reach_entries,
            int(reach_id),
        )
        _apply_reach_median_top_width_to_missing_bank(
            x_section,
            sampled_records,
            reach_entries,
            reach_top_width_stats,
        )

        # Calculate the legacy straight-axis projection. It is retained only as
        # an ordering fallback for isolated reaches that have neither a sampled
        # graph successor nor a sampled graph predecessor.
        mean_direction = _compute_mean_reach_stream_direction(
            [entry["stream_direction"] for entry in reach_entries]
        )
        along_stream_unit_x = math.cos(mean_direction)
        along_stream_unit_y = math.sin(mean_direction)

        along_stream_coordinates = np.asarray(
            [
                entry["col"] * along_stream_unit_x + entry["row"] * along_stream_unit_y
                for entry in reach_entries
            ],
            dtype=np.float64,
        )
        # Replay profiles again because the shared CrossSection instance only
        # contains one sampled section at a time. Extract the lower valid bank
        # after width filtering/filling and retain which search method found it.
        raw_bank_elevations = np.full(len(reach_entries), np.nan, dtype=np.float64)
        thalweg_elevations = np.full(len(reach_entries), np.nan, dtype=np.float64)
        function_used_by_entry_index: dict[int, str | None] = {}
        for elevation_index, entry in enumerate(reach_entries):
            sampled_record = sampled_records[int(entry["entry_index"])]
            if sampled_record is None:
                continue

            reach_bank_index = None
            if _CELL_REACH_INFLECT_BANK_INDEX is not None:
                reach_bank_index = float(_CELL_REACH_INFLECT_BANK_INDEX[int(entry["entry_index"])])
            _replay_precomputed_cross_section(x_section, sampled_record, reach_bank_index=reach_bank_index)
            thalweg_elevations[elevation_index] = float(x_section.get_thalweg())

            bank_search_result = sampled_record.get("bank_search_result")
            bank_elevation_to_use = _compute_raw_bank_elevation_from_result(
                x_section,
                bank_search_result,
            )
            if bank_elevation_to_use != 0.0 and not np.isnan(bank_elevation_to_use):
                raw_bank_elevations[elevation_index] = bank_elevation_to_use
                function_used_by_entry_index[int(entry["entry_index"])] = (
                    bank_search_result.get("function_used") if isinstance(bank_search_result, dict) else None
                )
            else:
                continue

        # Width filtering can rebuild a bank result after the earlier side-level
        # validation. Exclude any rebuilt result that still resolves to the
        # thalweg before q2/q97 or the reach minimum is calculated.
        raw_bank_elevations = _exclude_thalweg_equal_bank_elevations(
            raw_bank_elevations,
            thalweg_elevations,
        )

        # Filter the remaining raw bank elevations to keep outliers out.
        finite_mask = np.isfinite(raw_bank_elevations)
        reach_bank_elevations = raw_bank_elevations[finite_mask]
        lower_bound = -np.inf
        upper_bound = np.inf

        if reach_bank_elevations.size >= 4:
            q2, q97 = np.percentile(reach_bank_elevations, [2, 97])
            lower_bound = float(q2)
            upper_bound = float(q97)

            outlier_mask = finite_mask & (
                (raw_bank_elevations < lower_bound)
                | (raw_bank_elevations > upper_bound)
            )
            raw_bank_elevations[outlier_mask] = np.nan

        # The mean-axis projection supplies only an isolated-reach fallback.
        # Normal reaches are ordered along their connected raster-cell paths and
        # oriented using the graph's explicit downstream successor.
        fallback_order = np.argsort(along_stream_coordinates)
        order, ordered_stream_stations = _order_reach_stream_cells_from_network(
            reach_network_graph,
            int(reach_id),
            reach_entries,
            grouped_reach_entries,
            fallback_order,
            raw_bank_elevations,
            float(params['dx']),
            float(params['dy']),
        )

        # From this point forward, ``ordered_coordinates`` means physical
        # distance along the graph-oriented raster path
        ordered_coordinates = ordered_stream_stations
        ordered_raw_bank_elevations = raw_bank_elevations[order]
        ordered_thalweg_elevations = thalweg_elevations[order]
        finite_ordered_bank_elevations = ordered_raw_bank_elevations[np.isfinite(ordered_raw_bank_elevations)]
        if finite_ordered_bank_elevations.size == 0:
            # This reach has no bank control above its thalweg. Leave it out of
            # the observation dictionary, but retain its summary so the network
            # can interpolate a control and apply it to the reach's sections.
            minimum_bank_elevation = np.nan
            maximum_bank_elevation = np.nan
        else:
            # The minimum supplies the outlet control used throughout the
            # network. Headwaters also use the maximum below as their unique
            # upstream endpoint before the per-cell anchoring pass.
            minimum_bank_elevation = float(
                np.nanmin(finite_ordered_bank_elevations)
            )
            # Headwaters use the opposite endpoint of this same filtered range
            # to reconstruct their upstream control without rescanning cells.
            maximum_bank_elevation = float(
                np.nanmax(finite_ordered_bank_elevations)
            )

        reach_summaries[int(reach_id)] = {
            'reach_entries': reach_entries,
            'order': order.copy(),
            'ordered_coordinates': ordered_coordinates.copy(),
            'ordered_raw_bank_elevations': ordered_raw_bank_elevations.copy(),
            'ordered_thalweg_elevations': ordered_thalweg_elevations.copy(),
            'bank_elevation_lower_bound': lower_bound,
            'bank_elevation_upper_bound': upper_bound,
            'function_used_by_entry_index': dict(function_used_by_entry_index),
            'mean_direction': float(mean_direction),
            'minimum_bank_elevation': minimum_bank_elevation,
            'maximum_bank_elevation': maximum_bank_elevation,
        }

    # Omit reaches without a finite local observation; graph interpolation may
    # still populate graph nodes lying between reaches that do have controls.
    reach_min_bank_elevation_dict = {
        int(reach_id): float(summary['minimum_bank_elevation'])
        for reach_id, summary in reach_summaries.items()
        if np.isfinite(summary['minimum_bank_elevation'])
    }
    if len(reach_summaries) > 0 and len(reach_min_bank_elevation_dict) == 0:
        raise ValueError(
            'Network bank-elevation smoothing could not proceed because no '
            'finite minimum bank elevations were found for any reach.'
        )
    # produce the reach_max_bank_elevation_dict for headwater initialization
    reach_max_bank_elevation_dict = {
        int(reach_id): float(summary['maximum_bank_elevation'])
        for reach_id, summary in reach_summaries.items()
        if np.isfinite(summary['maximum_bank_elevation'])
    }
    if len(reach_summaries) > 0 and len(reach_max_bank_elevation_dict) == 0:
        raise ValueError(
            'Network bank-elevation smoothing could not proceed because no '
            'finite maximum bank elevations were found for any reach.'
        )
    # Treat each reach minimum as an observation at that reach's outlet. The
    # graph pass fills missing outlet controls, enforces a downstream fall, and
    # stores the slope assigned to every reach on its graph node. Ordered cell
    # observations are passed into the same estimator so they can become new
    # upstream-to-downstream interpolation anchors rather than being overlaid
    # after interpolation.
    reach_cell_bank_observations = {
        int(reach_id): {
            "ordered_coordinates": np.asarray(
                summary["ordered_coordinates"],
                dtype=np.float64,
            ),
            "observed_elevations": np.asarray(
                summary["ordered_raw_bank_elevations"],
                dtype=np.float64,
            ),
            # This value is calculated directly from ``raw_bank_elevations``
            # after percentile and thalweg filtering. The network estimator
            # uses it only to initialize headwater upstream endpoints.
            "filtered_maximum_elevation": float(
                summary["maximum_bank_elevation"]
            ),
            # Pass the same reach-level outlier thresholds and cell thalwegs
            # into the anchor routine. This keeps anchor eligibility explicit
            # even though the prepass has already replaced known outliers and
            # thalweg-equal observations with NaN.
            "thalweg_elevations": np.asarray(
                summary["ordered_thalweg_elevations"],
                dtype=np.float64,
            ),
            "lower_bound": float(summary["bank_elevation_lower_bound"]),
            "upper_bound": float(summary["bank_elevation_upper_bound"]),
        }
        for reach_id, summary in reach_summaries.items()
    }
    network_smoothed_reach_min_bank_elevations = _estimate_network_smoothed_reach_min_bank_elevations(
        reach_network_graph,
        reach_min_bank_elevation_dict,
        reach_max_bank_elevation_dict,
        reach_cell_bank_observations,
    )

    # Final pass: interpolate the graph-smoothed controls to every cross section
    # while preserving local bank indices/top width as horizontal geometry.
    for reach_id, reach_summary in reach_summaries.items():
        reach_entries = reach_summary['reach_entries']
        order = np.asarray(reach_summary['order'], dtype=np.int64)
        ordered_coordinates = np.asarray(reach_summary['ordered_coordinates'], dtype=np.float64)
        ordered_raw_bank_elevations = np.asarray(reach_summary['ordered_raw_bank_elevations'], dtype=np.float64)
        function_used_by_entry_index = reach_summary['function_used_by_entry_index']
        mean_direction = float(reach_summary['mean_direction'])
        minimum_bank_elevation = float(reach_summary['minimum_bank_elevation'])

        if reach_id not in network_smoothed_reach_min_bank_elevations:
            raise ValueError(
                'Network smoothing did not return a bank elevation for reach_id '
                + str(reach_id)
                + '.'
            )
        # The node value is the reach's outlet minimum. The interpolation
        # helper uses the graph-node slope to reconstruct the upstream endpoint.
        reach_outlet_network_elevation = float(
            network_smoothed_reach_min_bank_elevations[reach_id]
        )
        (
            network_interpolated_bank_elevation_surface,
            reach_interpolation_fractions,
            graph_downstream_reach_id,
            downstream_surface_control_elevation,
        ) = _interpolate_reach_bank_elevation_surface(
            reach_network_graph,
            int(reach_id),
            ordered_coordinates,
            network_smoothed_reach_min_bank_elevations,
        )
        # The estimator already walked these cells upstream-to-downstream and
        # stored its observation-anchored result on the graph node. Reuse that
        # exact surface and its piecewise outgoing grades here so bank elevation
        # and hydraulic slope remain consistent.
        reach_node_data = reach_network_graph.nodes[int(reach_id)]
        interpolated_bank_elevation_surface = np.asarray(
            reach_node_data.get(
                "observed_anchored_cell_bank_elevation_surface",
                network_interpolated_bank_elevation_surface,
            ),
            dtype=np.float64,
        )
        cell_outgoing_grades = np.asarray(
            reach_node_data.get(
                "cell_bank_elevation_outgoing_grades",
                np.full(
                    network_interpolated_bank_elevation_surface.size,
                    reach_node_data.get("bank_elevation_grade", MIN_SLOPE),
                    dtype=np.float64,
                ),
            ),
            dtype=np.float64,
        )
        observation_anchor_mask = np.asarray(
            reach_node_data.get(
                "cell_bank_elevation_observation_anchor_mask",
                np.zeros(
                    network_interpolated_bank_elevation_surface.size,
                    dtype=bool,
                ),
            ),
            dtype=bool,
        )
        # Preserve a downstream ID from the source table even when that reach
        # was outside the graph; graph topology is authoritative when present.
        # The successor value is its own outlet control. The current reach's
        # ``downstream_surface_control_elevation`` is its outlet minimum.
        downstream_reach_id = graph_downstream_reach_id
        if downstream_reach_id is None:
            downstream_reach_id = reach_downstream_map.get(int(reach_id))
        downstream_network_elevation = (
            float(network_smoothed_reach_min_bank_elevations[downstream_reach_id])
            if downstream_reach_id in network_smoothed_reach_min_bank_elevations
            else np.nan
        )

        # Map the upstream-to-downstream surface values back to the original
        # sampled-record indices. This is where the ordered elevation surface
        # becomes the per-cross-section bathymetry input.
        for reach_order, (ordered_position, target_bank_elevation) in enumerate(
            zip(order, interpolated_bank_elevation_surface)
        ):
            reach_entry = reach_entries[int(ordered_position)]
            sampled_record = sampled_records[int(reach_entry["entry_index"])]
            if sampled_record is None:
                continue

            reach_bank_index = None
            if _CELL_REACH_INFLECT_BANK_INDEX is not None:
                reach_bank_index = float(_CELL_REACH_INFLECT_BANK_INDEX[int(reach_entry["entry_index"])])
            _replay_precomputed_cross_section(x_section, sampled_record, reach_bank_index=reach_bank_index)

            current_bank_search_result = sampled_record.get("bank_search_result")
            # Rebuild the result at the network elevation while carrying forward
            # the locally chosen bank pair and the search method that produced it.
            updated_bank_result = x_section.build_bank_search_result_from_smoothed_elevation(
                current_bank_search_result,
                float(target_bank_elevation),
                function_used_by_entry_index.get(int(reach_entry["entry_index"])),
            )
            # Keep the raw/local/network values side by side. Bathymetry reads
            # ``smoothed_bank_elevation``; the remaining fields document how
            # that per-cell value was produced and allow exported diagnostics
            # to compare it with the original detected bank elevation.
            updated_bank_result["raw_bank_elevation"] = float(
                ordered_raw_bank_elevations[reach_order]
            )
            updated_bank_result["observed_cell_minimum_bank_elevation"] = float(
                ordered_raw_bank_elevations[reach_order]
            )
            updated_bank_result["locally_smoothed_bank_elevation"] = float(
                interpolated_bank_elevation_surface[reach_order]
            )
            updated_bank_result["smoothed_bank_elevation"] = float(target_bank_elevation)
            updated_bank_result["network_interpolated_bank_elevation"] = float(
                network_interpolated_bank_elevation_surface[reach_order]
            )
            updated_bank_result["observed_anchored_bank_elevation"] = float(
                target_bank_elevation
            )
            updated_bank_result["observed_bank_elevation_used_as_anchor"] = (
                bool(observation_anchor_mask[reach_order])
            )
            updated_bank_result["network_reach_interpolation_fraction"] = float(
                reach_interpolation_fractions[reach_order]
            )
            updated_bank_result["reach_minimum_bank_elevation"] = minimum_bank_elevation
            updated_bank_result["network_smoothed_reach_minimum_bank_elevation"] = (
                reach_outlet_network_elevation
            )
            updated_bank_result["network_smoothed_reach_outlet_bank_elevation"] = (
                reach_outlet_network_elevation
            )
            updated_bank_result["network_reach_upstream_bank_elevation"] = float(
                network_interpolated_bank_elevation_surface[0]
            )
            updated_bank_result["network_reach_bank_elevation_grade"] = float(
                cell_outgoing_grades[reach_order]
            )
            updated_bank_result["network_reach_base_bank_elevation_grade"] = float(
                reach_node_data.get('bank_elevation_grade', MIN_SLOPE)
            )
            updated_bank_result["downstream_reach_id"] = (
                int(downstream_reach_id) if downstream_reach_id is not None else -1
            )
            updated_bank_result["downstream_network_bank_elevation"] = downstream_network_elevation
            updated_bank_result["downstream_surface_control_elevation"] = float(
                downstream_surface_control_elevation
            )
            updated_bank_result["reach_order_index"] = int(reach_order)
            updated_bank_result["reach_stream_direction"] = float(mean_direction)
            updated_bank_result["along_stream_coordinate"] = float(ordered_coordinates[reach_order])
            sampled_record["bank_search_result"] = updated_bank_result


def _finalize_cross_section_records(
    sampled_records: list[dict | None],
    params: dict,
    quiet: bool,
    collect_cross_section_data: bool = False,
) -> tuple[list[dict | None] | None, list[dict | None]]:
    """Finalize banks and depths, burn staged bathymetry, and build exports.

    Once local bank indices have been estimated for all sampled cross sections,
    ARC smooths the implied bank elevations along each reach, preserves the
    locally detected bank indices and top widths, then estimates every
    non-target hydraulic depth in a separate pass over the unmodified cached
    profiles. A final pass burns those stored depths into the profiles and
    bathymetry raster.
    """
    if params['s_output_bathymetry_path']:
        _smooth_reach_bank_elevations(
            sampled_records,
            params,
            quiet,
        )
        # Hydraulic depths must be solved from the finalized bank geometry,
        # but before any profile is lowered by the bathymetry burn.
        _stage_cross_section_bathymetry_depths(
            sampled_records,
            params,
            quiet,
        )

    x_section = get_cross_section(params['dx'], params['dy'], _DEM, _LAND_COVER, _STREAMS, params)
    cross_section_data = [None] * _CELL_COMIDS.size if collect_cross_section_data else None
    finalized_records = [None] * _CELL_COMIDS.size

    for i_entry_cell in tqdm.tqdm(range(_CELL_COMIDS.size), total=_CELL_COMIDS.size, disable=quiet):
        sampled_record = sampled_records[i_entry_cell]
        if sampled_record is None:
            continue

        reach_bank_index = None
        if _CELL_REACH_INFLECT_BANK_INDEX is not None:
            reach_bank_index = float(_CELL_REACH_INFLECT_BANK_INDEX[i_entry_cell])
        _replay_precomputed_cross_section(x_section, sampled_record, reach_bank_index=reach_bank_index)

        i_row_cell = int(_CELL_ROWS[i_entry_cell])
        i_column_cell = int(_CELL_COLS[i_entry_cell])
        d_q_baseflow, d_slope_use, d_bathy_target_depth, d_bathy_target_width = _get_cell_bathymetry_inputs(
            i_entry_cell,
            i_row_cell,
            i_column_cell,
            params,
        )
        d_slope_use = _replace_slope_with_smoothed_bank_grade(
            d_slope_use,
            sampled_record.get("bank_search_result"),
        )

        bathymetry_applied = False
        if params['s_output_bathymetry_path']:
            _apply_bathymetry_to_cross_section(
                x_section,
                params,
                bank_search_result=sampled_record.get("bank_search_result"),
            )
            bathymetry_applied = True

        # Export records and staged representative hydraulics should reflect the
        # finalized profile geometry, so re-sample Manning's n after any
        # low-spot, angle, and bathymetry adjustments.
        x_section.set_mannings_n_values(_MANNINGS_N)

        inflect_curve = sampled_record.get("inflect_curve")
        finalized_records[i_entry_cell] = _build_precomputed_cross_section_record(
            x_section,
            sampled_record["dem_low_point_elev"],
            bathymetry_applied=bathymetry_applied,
            inflect_curve=inflect_curve,
            bank_search_result=sampled_record.get("bank_search_result"),
        )

        if collect_cross_section_data:
            export_row, export_col = x_section.get_row_col()
            cross_section_data[i_entry_cell] = _build_cross_section_export_record(
                x_section,
                params,
                int(_CELL_COMIDS[i_entry_cell]),
                export_row,
                export_col,
                d_slope_use,
                inflect_curve,
            )

    return cross_section_data, finalized_records

def calculate_hydraulic_data_for_cell(i_entry_cell: int):
    """
    Compute hydraulic increments for a single stream cell.

    This function is the core per-cell kernel. It reads per-cell metadata
    (row/col, COMID, baseflow, qmax) from shared/global arrays, then either
    replays the staged precomputed cross section or samples a fresh section if
    no staged cache is available. If bathymetry has not already been applied to
    the cached section, the function applies it here using the precomputed bank
    search result. It then fills the shared output array with hydraulic
    results.

    Parameters
    ----------
    i_entry_cell : int
        Index into the per-cell arrays (rows/cols/COMIDs/flows). This is *not*
        a raster index; it is the index of the extracted stream-cell list.

    Returns
    -------
    dict or None
        If a cross-section-based export is enabled, returns a per-cell record
        containing the sampled profile and metadata. Otherwise returns
        ``None``.
    """
    i_row_cell = _CELL_ROWS[i_entry_cell]
    i_column_cell = _CELL_COLS[i_entry_cell]
    i_cell_comid = _CELL_COMIDS[i_entry_cell]
    d_q_maximum = _CELL_QMAX[i_entry_cell]
    i_number_of_increments = _PARAMS['i_number_of_increments']
    i_general_direction_distance = _PARAMS['i_general_direction_distance']
    using_manual_cross_sections = bool(_PARAMS.get('s_manual_cross_section_file'))
    manual_record = None
    precomputed_record = None
    if using_manual_cross_sections:
        manual_record = _MANUAL_CROSS_SECTION_RECORDS.get(int(i_cell_comid))
        if manual_record is None:
            raise KeyError(f"Manual cross section for ID {i_cell_comid} was not found.")
    if _PRECOMPUTED_CROSS_SECTION_RECORDS is not None:
        precomputed_record = _PRECOMPUTED_CROSS_SECTION_RECORDS[i_entry_cell]

    dx = _PARAMS['dx']
    dy = _PARAMS['dy']
    d_q_baseflow, d_slope_use, d_bathy_target_depth, d_bathy_target_width = _get_cell_bathymetry_inputs(
        i_entry_cell,
        i_row_cell,
        i_column_cell,
        _PARAMS,
    )
    d_slope_use = _replace_slope_with_smoothed_bank_grade(
        d_slope_use,
        (
            precomputed_record.get("bank_search_result")
            if isinstance(precomputed_record, dict)
            else None
        ),
    )

    x_section = get_cross_section(dx, dy, _DEM, _LAND_COVER, _STREAMS, _PARAMS)
    if precomputed_record is not None:
        reach_bank_index = None
        if _CELL_REACH_INFLECT_BANK_INDEX is not None:
            reach_bank_index = float(_CELL_REACH_INFLECT_BANK_INDEX[i_entry_cell])
        _replay_precomputed_cross_section(x_section, precomputed_record, reach_bank_index=reach_bank_index)
        i_row_cell, i_column_cell = x_section.get_row_col()
        d_dem_low_point_elev = precomputed_record["dem_low_point_elev"]
    elif using_manual_cross_sections:
        apply_manual_cross_section_data(x_section, manual_record)
        d_dem_low_point_elev = x_section.get_thalweg()


    # Adjust cross-section angle to ensure shortest top-width at a specified
    # depth when ARC is sampling the section from rasters itself. Manual cross
    # sections are already fixed and should not be reoriented here.
    if (precomputed_record is None and not using_manual_cross_sections) and x_section.has_angles_to_test():
        x_section.test_angles_and_reset_cross_section(i_row_cell, i_column_cell)

    # Burn bathymetry profile into cross-section profile
    # "Be the banks for your river" - Needtobreathe
            
    # If you don't have a cross-section, skip it and fill in empty values for the reach average processing
    hydraulic_data = get_hydraulic_data(_PARAMS)
    if not x_section.is_valid():
        hydraulic_data.add_empty_x_section_for_curve_file(i_cell_comid, d_slope_use, i_entry_cell)
        return

    if precomputed_record is None and _CELL_REACH_INFLECT_BANK_INDEX is not None:
        x_section.set_reach_scale_inflect_bank_index(float(_CELL_REACH_INFLECT_BANK_INDEX[i_entry_cell]))

    if precomputed_record is None or not bool(precomputed_record.get("bathymetry_applied", False)):
        _apply_bathymetry_to_cross_section(
            x_section,
            _PARAMS,
            bank_search_result=None if precomputed_record is None else precomputed_record.get("bank_search_result"),
        )


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
        d_q_sum = calculate_discharge_from_wse(d_maxflow_wse_final, slope_use_squared, *x_sect_args)
    elif np.round(f_lower, 5) == 0 or np.round(f_upper, 5) == 0:          
        # if the f_lower or f_upper is equal to zero, it's probably close enough to be the WSE we are looking for, so we'll use it
        d_maxflow_wse_final = np.round(wse_lower, 3) if np.round(f_lower, 5) == 0 else np.round(wse_upper, 3)
        d_q_sum = calculate_discharge_from_wse(d_maxflow_wse_final, slope_use_squared, *x_sect_args)

    # Let's see if the volume-fill approach gave us a better answer and use that if it did
    # To find the depth / wse where the maximum flow occurs we use two sets of incremental depths.  The first is 0.5m followed by 0.05m
    d_maxflow_wse_initial, d_q_sum_test = find_wse(101, d_maxflow_wse_initial, DEPTH_INCREMENT_BIG, d_q_maximum, x_sect_args, d_slope_use)


    # Based on using depth increments of 0.5, now lets fine-tune the wse using depth increments of 0.05
    d_maxflow_wse_initial = max(d_maxflow_wse_initial - 0.5, thalweg)
    d_maxflow_wse_med = d_maxflow_wse_initial
    d_maxflow_wse_med, d_q_sum_test = find_wse(101, d_maxflow_wse_med, DEPTH_INCREMENT_MEDIUM, d_q_maximum, x_sect_args, d_slope_use)

    # Based on using depth increments of 0.05, now lets fine-tune the wse even more using depth increments of 0.01
    d_maxflow_wse_med = max(d_maxflow_wse_med - 0.05, thalweg)
    d_maxflow_wse_final_test = d_maxflow_wse_med
    d_maxflow_wse_final_test, d_q_sum_test = find_wse(2501, d_maxflow_wse_med, DEPTH_INCREMENT_SMALL, d_q_maximum, x_sect_args, d_slope_use)

    # let's see if the iterative method gave use a better result and use that if it did
    if abs(d_q_sum_test - d_q_maximum) < abs(d_q_sum-d_q_maximum):
        d_maxflow_wse_final = d_maxflow_wse_final_test
        d_q_sum = d_q_sum_test

    slope_obj_args = (d_maxflow_wse_initial, DEPTH_INCREMENT_SMALL, d_q_maximum, x_sect_args)
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
        slope_lower = max(d_slope_use - potential_slope_error, MIN_SLOPE) # Avoids domain error, taking sqrt of negative number, in find wse
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
            trial_slope_use = np.round(trial_slope_use, MIN_SLOPE_DECIMAL_PLACES)
            # Optionally, recompute d_maxflow_wse_final and d_q_sum with the new slope:
            d_maxflow_wse_final_test, d_q_sum_test = find_wse(
                2501, 
                d_maxflow_wse_initial, 
                DEPTH_INCREMENT_SMALL, 
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
            trial_slope_use = np.round(slope_lower, MIN_SLOPE_DECIMAL_PLACES)
            # Optionally, recompute d_maxflow_wse_final and d_q_sum with the new slope:
            d_maxflow_wse_final_test, d_q_sum_test = find_wse(
                2501, 
                d_maxflow_wse_initial, 
                DEPTH_INCREMENT_SMALL, 
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
            trial_slope_use = np.round(slope_upper, MIN_SLOPE_DECIMAL_PLACES)
            # Optionally, recompute d_maxflow_wse_final and d_q_sum with the new slope:
            d_maxflow_wse_final_test, d_q_sum_test = find_wse(
                2501, 
                d_maxflow_wse_initial, 
                DEPTH_INCREMENT_SMALL, 
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
        slope_lower = max(d_slope_use - potential_slope_error, MIN_SLOPE) # Avoids domain error, taking sqrt of negative number, in find wse
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
            trial_slope_use = np.round(trial_slope_use, MIN_SLOPE_DECIMAL_PLACES)
        
            # Optionally, recompute d_maxflow_wse_final and d_q_sum with the new slope:
            d_maxflow_wse_final_test, d_q_sum_test = find_wse(
                2501, 
                d_maxflow_wse_initial, 
                DEPTH_INCREMENT_SMALL, 
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
            trial_slope_use = np.round(slope_lower, MIN_SLOPE_DECIMAL_PLACES)
            # Optionally, recompute d_maxflow_wse_final and d_q_sum with the new slope:
            d_maxflow_wse_final_test, d_q_sum_test = find_wse(
                2501, 
                d_maxflow_wse_initial, 
                DEPTH_INCREMENT_SMALL, 
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
            trial_slope_use = np.round(slope_upper, MIN_SLOPE_DECIMAL_PLACES)
            # Optionally, recompute d_maxflow_wse_final and d_q_sum with the new slope:
            d_maxflow_wse_final_test, d_q_sum_test = find_wse(
                2501, 
                d_maxflow_wse_initial, 
                DEPTH_INCREMENT_SMALL, 
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
        # round d_q_sum to the 3rd decimal place
        d_q_sum = round(d_q_sum, 3)
        # Now lets get a set number of increments between the low elevation and the elevation where Qmax hits
        d_inc_y = round((d_maxflow_wse_final - thalweg) / i_number_of_increments, 3)
        flood_increments_args = x_section.get_flood_increment_args()
        i_start_elevation_index, i_last_elevation_index = flood_increments(i_number_of_increments + 1, 
                                                                        d_inc_y, 
                                                                        flood_increments_args, thalweg, d_slope_use, 
                                                                        d_q_sum, _OUTPUT_DATA_ARRAY, i_entry_cell, hydraulic_data.b_modified_dem)
        
        if i_last_elevation_index > i_start_elevation_index:
            if d_q_baseflow > 0.001 and hydraulic_data.is_start_q_greater_than_baseflow(i_start_elevation_index, d_q_baseflow, i_entry_cell):
                hydraulic_data.set_q_at_index(i_start_elevation_index + 1, d_q_baseflow - 0.001, i_entry_cell)
                
            # Process each of the elevations to the output file if feasbile values were produced
            hydraulic_data.set_vdt_data(i_cell_comid, d_q_baseflow, d_slope_use, i_entry_cell)

    # Gather up all the values for the stream cell if we are going to build a reach average curve file
    hydraulic_data.set_non_vdt_data(i_start_elevation_index, i_last_elevation_index, i_cell_comid, i_row_cell, i_column_cell,
                                    d_slope_use, d_dem_low_point_elev, i_entry_cell)
    
    if hydraulic_data.wants_cross_section_records():
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
        globals()['_PRECOMPUTED_CROSS_SECTION_RECORDS'] = params.get('_precomputed_cross_section_records')

def _build_flow_arrays(
    id_flow_dict: dict,
    baseflow_key: str,
    qmax_key: str,
    processes: int,
    bathymetry_geometry_dict: dict[int, dict[str, float]] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if baseflow_key == '':
        create_array("_CELL_QBASE", processes, (_CELL_COMIDS.size,), np.float64, fill_value=0.0)
    else:
        create_array("_CELL_QBASE", processes, (_CELL_COMIDS.size,), np.float64)[:] = np.fromiter((id_flow_dict[cid][baseflow_key] for cid in _CELL_COMIDS), dtype=np.float64, count=len(_CELL_COMIDS))
    create_array("_CELL_QMAX", processes, (_CELL_COMIDS.size,), np.float64)[:] = np.fromiter((id_flow_dict[cid][qmax_key] for cid in _CELL_COMIDS), dtype=np.float64, count=len(_CELL_COMIDS))

    if bathymetry_geometry_dict is None:
        return

    # Manual cross-section runs may use manual IDs in the flow file while the
    # stream vector still stores drainage area by the source stream ID. Reuse
    # the same source-ID fallback that ARC already applies for reach-average
    # slopes so the optional power-law bathymetry mode behaves consistently.
    source_ids = _CELL_SOURCE_STREAM_IDS if _CELL_SOURCE_STREAM_IDS is not None else _CELL_COMIDS
    create_array("_CELL_BATHY_DEPTH", processes, (_CELL_COMIDS.size,), np.float64)[:] = np.fromiter(
        (bathymetry_geometry_dict[int(stream_id)]['depth'] for stream_id in source_ids),
        dtype=np.float64,
        count=len(_CELL_COMIDS),
    )
    create_array("_CELL_BATHY_WIDTH", processes, (_CELL_COMIDS.size,), np.float64)[:] = np.fromiter(
        (bathymetry_geometry_dict[int(stream_id)]['width'] for stream_id in source_ids),
        dtype=np.float64,
        count=len(_CELL_COMIDS),
    )

def _build_reach_slope_arrays(stream_slope_dicts: tuple[dict], params: dict, processes: int) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    method = params['s_stream_slope_method']
    slope_ids = _CELL_SOURCE_STREAM_IDS if _CELL_SOURCE_STREAM_IDS is not None else _CELL_COMIDS
    if method in {'reach_average', 'end_points'}:
        slope_dict = stream_slope_dicts[0]
        create_array("_CELL_REACH_SLOPE", processes, (_CELL_COMIDS.size,), np.float64)[:] = np.fromiter((slope_dict[cid] for cid in slope_ids), dtype=np.float64, count=len(_CELL_COMIDS))
    if method == 'local_average_corrected':
        slope25_dict = stream_slope_dicts[1]
        slope75_dict = stream_slope_dicts[2]
        create_array("_CELL_SLOPE_25", processes, (_CELL_COMIDS.size,), np.float64)[:] = np.fromiter((slope25_dict[cid] for cid in slope_ids), dtype=np.float64, count=len(_CELL_COMIDS))
        create_array("_CELL_SLOPE_75", processes, (_CELL_COMIDS.size,), np.float64)[:] = np.fromiter((slope75_dict[cid] for cid in slope_ids), dtype=np.float64, count=len(_CELL_COMIDS))

def compute_cross_section_data(
    params: dict,
    processes: int,
    quiet: bool,
    collect_cross_section_data: bool = False,
) -> tuple[dict[int, int], dict[int, int], list[dict | None] | None, list[dict | None] | None]:
    """Precompute sampled sections, banks, and staged bathymetry.

    The cross-section workflow is intentionally split into ordered passes:

    1. sample every stream-cell cross section from the DEM (or manual input),
       apply the low-spot adjustment, and resample the final profile,
    2. compute reach-scale INFLECT curves and use them, together with the
       legacy bank heuristics, to identify banks for every cached section, and
    3. once the banks are known, optionally apply bathymetry and build any
       requested cross-section export records.
    """
    x_section = get_cross_section(params['dx'], params['dy'], _DEM, _LAND_COVER, _STREAMS, params)
    grouped_curves: dict[int, list[tuple[np.ndarray, np.ndarray]]] = {}
    source_ids = _CELL_SOURCE_STREAM_IDS if _CELL_SOURCE_STREAM_IDS is not None else _CELL_COMIDS
    sampled_records = [None] * _CELL_COMIDS.size

    for i_entry_cell in tqdm.tqdm(range(_CELL_COMIDS.size), total=_CELL_COMIDS.size, disable=quiet):
        i_reach_id = int(source_ids[i_entry_cell])
        valid, i_cell_comid, i_row_cell, i_column_cell, d_dem_low_point_elev = _sample_cross_section_for_cell(
            x_section,
            i_entry_cell,
            params,
        )
        if not valid:
            continue

        depth_values, curve = x_section.get_representative_inflect_curve_with_depths()
        sampled_records[i_entry_cell] = _build_precomputed_cross_section_record(
            x_section,
            d_dem_low_point_elev,
            bathymetry_applied=False,
            inflect_curve=curve,
        )
        if curve.size > 0 and depth_values.size > 0:
            grouped_curves.setdefault(i_reach_id, []).append((curve.copy(), depth_values.copy()))

    inflect_bank_index_dict, inflect_terrace_index_dict = _build_reach_inflect_index_dictionaries(
        grouped_curves,
        params,
    )
    _populate_reach_inflect_arrays(
        inflect_bank_index_dict,
        inflect_terrace_index_dict,
        processes,
    )
    _annotate_cross_sections_with_bank_search_results(
        sampled_records,
        params,
        quiet,
    )
    cross_section_data, precomputed_records = _finalize_cross_section_records(
        sampled_records,
        params,
        quiet,
        collect_cross_section_data=collect_cross_section_data,
    )

    return inflect_bank_index_dict, inflect_terrace_index_dict, cross_section_data, precomputed_records


def compute_cross_section_and_inflect_curve(
    params: dict,
    processes: int,
    quiet: bool,
    collect_cross_section_data: bool = False,
) -> list[dict | None] | None:
    """Build the staged cross-section cache and optional export records."""
    if TEMP_PLOT_REACH_INFLECT_CURVES:
        LOG.info('Temporary reach INFLECT plots will be written to ' + _get_reach_inflect_plot_directory(params))
    inflect_bank_index_dict, inflect_terrace_index_dict, cross_section_data, precomputed_records = compute_cross_section_data(
        params,
        processes,
        quiet,
        collect_cross_section_data=collect_cross_section_data,
    )
    global _PRECOMPUTED_CROSS_SECTION_RECORDS
    _PRECOMPUTED_CROSS_SECTION_RECORDS = precomputed_records
    params['_precomputed_cross_section_records'] = precomputed_records
    return cross_section_data

def run_main_loop(
    num_cells: int,
    params: dict,
    quiet: bool,
    processes: int,
    precomputed_cross_section_data: list[dict | None] | None = None,
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
    # Representative cross sections are rebuilt later from stored per-cell
    # cross-section records using INFLECT-limited 0.10 m hydraulic staging, so
    # either output requires retaining the sampled profile records. When the
    # reach-INFLECT prepass already built those records, reuse them directly
    # instead of collecting duplicate copies in the hydraulic loop.
    want_xs = bool(
        params.get('s_xs_output_file')
        or (
            params.get('b_build_representative_cross_section')
            and params.get('s_representative_cross_section_file')
        )
    )
    cross_section_data: list | None = None
    collect_cross_section_data = False
    if want_xs:
        if precomputed_cross_section_data is not None:
            cross_section_data = precomputed_cross_section_data
        else:
            cross_section_data = []
            collect_cross_section_data = True

    LOG.info('Looking at ' + str(num_cells) + ' stream cells')

    if processes == 1:
        for i_entry_cell in tqdm.tqdm(range(num_cells), total=num_cells, disable=quiet):
            item = calculate_hydraulic_data_for_cell(i_entry_cell)
            if collect_cross_section_data and item is not None:
                cross_section_data.append(item)

        hydraulic_data = get_hydraulic_data(params)
        if cross_section_data is not None:
            hydraulic_data.add_cross_section_data(cross_section_data)
        return hydraulic_data
    
    args = get_init_parallel_args(ARRAY_NAMES)

    with Pool(processes=processes, initializer=init_parallel, initargs=(*args, params)) as pool:
        chunksize = min(1_000, num_cells // (processes * 4) + 1)
        for item in tqdm.tqdm(pool.imap(calculate_hydraulic_data_for_cell, range(num_cells), chunksize=chunksize), total=num_cells, disable=quiet):
            if collect_cross_section_data and item is not None:
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
    bathymetry_geometry_dict = None
    if params['b_use_bathymetry_powerlaw']:
        bathymetry_geometry_dict = build_bathymetry_geometry_dict(
            params['s_strmshp_path'],
            params['s_flow_file_id'],
            params['s_bathymetry_drainage_area_field'],
            params['d_bathymetry_coefficient_depth'],
            params['d_bathymetry_exponent_depth'],
            params['d_bathymetry_coefficient_width'],
            params['d_bathymetry_exponent_width'],
        )

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

    ### Accessing the manual cross-section file that will by-pass the creation of the cross-sections in ARC. 
    manual_cross_section_file = params.get('s_manual_cross_section_file', '')
    manual_cross_section_records = None
    if manual_cross_section_file:
        manual_cross_section_records, required_x_section_distance = load_manual_cross_section_records(
            manual_cross_section_file,
            params['s_flow_file_id'],
            i_boundary_number,
        )
        if required_x_section_distance > params['d_x_section_distance']:
            LOG.info(
                "Increasing X_Section_Dist from "
                + str(params['d_x_section_distance'])
                + " to "
                + str(required_x_section_distance)
                + " to accommodate the supplied manual cross sections."
            )
            params['d_x_section_distance'] = required_x_section_distance

    # Get the list of stream locations. In manual mode, the location list comes
    # from the manual cross-section file rather than from the stream raster.
    flow_ids = np.fromiter(id_flow_dict.keys(), count=len(id_flow_dict), dtype=np.int64)
    if manual_cross_section_records:
        matching_flow_ids = [int(flow_id) for flow_id in flow_ids if int(flow_id) in manual_cross_section_records]
        if len(matching_flow_ids) == 0:
            raise ValueError(
                "No IDs were shared between the ARC flow file and the manual cross-section file."
            )
        ia_valued_row_indices = np.asarray(
            [manual_cross_section_records[flow_id]['row'] for flow_id in matching_flow_ids],
            dtype=np.int64,
        )
        ia_valued_column_indices = np.asarray(
            [manual_cross_section_records[flow_id]['col'] for flow_id in matching_flow_ids],
            dtype=np.int64,
        )
        create_array("_CELL_COMIDS", processes, (len(matching_flow_ids),), np.int64)[:] = np.asarray(matching_flow_ids, dtype=np.int64)
        create_array("_CELL_SOURCE_STREAM_IDS", processes, (len(matching_flow_ids),), np.int64)[:] = np.asarray(
            [manual_cross_section_records[flow_id]['source_stream_id'] for flow_id in matching_flow_ids],
            dtype=np.int64,
        )
        global _MANUAL_CROSS_SECTION_RECORDS
        _MANUAL_CROSS_SECTION_RECORDS = {
            flow_id: manual_cross_section_records[flow_id]
            for flow_id in matching_flow_ids
        }
    else:
        ia_valued_row_indices, ia_valued_column_indices = np.where(np.isin(dm_stream, flow_ids, kind='table'))
        create_array("_CELL_COMIDS", processes, (ia_valued_row_indices.size,), np.int64)[:] = dm_stream[ia_valued_row_indices, ia_valued_column_indices]

    for arr, name in zip([ia_valued_row_indices, ia_valued_column_indices], ["_CELL_ROWS", "_CELL_COLS"]):
        create_array(name, processes, arr.shape, arr.dtype)[:] = arr[:]

    # This array will hold all the data for each stream cell. The first 8 columns are 'COMID', 'Row', 'Col', 'DEM_Elev', 'QBaseflow', 'Slope', 'XS_Angle', 'BaseElev', and then we have 5 columns repeated for each increment with 'q', 'v', 't', 'wse', 'p'. 
    create_array("_OUTPUT_DATA_ARRAY", processes, (len(ia_valued_row_indices), 8 + params['i_number_of_increments']*5), np.float64, fill_value=np.nan)

    # Get the cell dx and dy coordinates
    dx, dy, dproject = convert_cell_size(dcellsize, dem_dy, dyll, dyur, dem_projection)
    LOG.info('Cellsize X = ' + str(dx))
    LOG.info('Cellsize Y = ' + str(dy))

    # create a reach average slope before we go stream cell by stream cell
    stream_slope_dicts = initialize_stream_slope_dictionaries(
        params,
        dx,
        dy,
        dem_geotransform,
        dem_projection,
        quiet,
        processes,
        i_boundary_number,
    )

    _build_flow_arrays(
        id_flow_dict,
        params['s_flow_file_baseflow'],
        params['s_flow_file_qmax'],
        processes,
        bathymetry_geometry_dict=bathymetry_geometry_dict,
    )
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

    want_xs = bool(
        params.get('s_xs_output_file')
        or (
            params.get('b_build_representative_cross_section')
            and params.get('s_representative_cross_section_file')
        )
    )

    # Precompute cross-section data before performing hydraulic calculations.
    precomputed_cross_section_data = compute_cross_section_and_inflect_curve(
        params,
        processes,
        quiet,
        collect_cross_section_data=want_xs,
    )

    # Extract some parameters
    b_bathy_use_banks = params['b_bathy_use_banks']
    s_output_bathymetry_path = params['s_output_bathymetry_path']

    # If b_build_representative_cross_section is False, then we want to generate hydraulic_data
    if params['b_build_representative_cross_section'] is False:
        ### Begin the stream cell solution loop ###
        hydraulic_data = run_main_loop(
            len(ia_valued_row_indices),
            params,
            quiet,
            processes,
            precomputed_cross_section_data=precomputed_cross_section_data,
        )

        # Create the output VDT Database file - datatypes are figured out automatically
        if not hydraulic_data.has_vdt_data():
            LOG.warning('No VDT data was generated, so no hydraulic output files will be created.')
            if precomputed_cross_section_data is not None:
                hydraulic_data.add_cross_section_data(precomputed_cross_section_data)
                hydraulic_data.save_cross_section_outputs_only()
        else:
            # At this point, release all memory except for bathymetry, output array, and elevation
            close_shared_arrays([name for name in ARRAY_NAMES if name not in {"_BATHYMETRY", "_OUTPUT_DATA_ARRAY", "_DEM"}])
            hydraulic_data.save_files(id_flow_dict, params['s_flow_file_qmax'])
    elif precomputed_cross_section_data is not None:
        hydraulic_data = get_hydraulic_data(params)
        hydraulic_data.add_cross_section_data(precomputed_cross_section_data)
        hydraulic_data.save_cross_section_outputs_only()

    # Write the output rasters
    if len(s_output_bathymetry_path) > 1:
        #Make sure all the bathymetry points are above the DEM elevation
        if not b_bathy_use_banks:
            _BATHYMETRY = np.where(_BATHYMETRY>dm_elevation, np.nan, _BATHYMETRY)
        # remove the increase in elevation, if negative elevations were present
        if b_modified_dem:
            # Subtract 100 only for cells that are not NaN
            _BATHYMETRY[~np.isnan(_BATHYMETRY)] -= 100
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
