"""Sampling a cross section from rasters: the values at evenly spaced points (ordinates) across the stream, centred
on a stream cell.

ARC's convention for directions and sides, used throughout (the stream's direction, the cross section's XS_Angle in
the outputs, and a manual cross section's XS_Angle):

- A direction is an angle on the raster's grid, in radians from the +column axis towards the +row axis. 0 runs along a
  row towards the higher columns (east), and pi / 2 along a column towards the higher rows (south on a north-up
  raster, north on a south-up one), so on a north-up raster the angle turns clockwise from east.
- A cross section runs from its left end to its right, as seen looking back along the stream direction on the grid
  drawn with row 0 at the top (on a north-up raster, as on the map), so its right end lies in the direction
  stream direction - pi / 2. That is looking upstream when the direction is the way the water flows, but
  arc.pipeline's stream directions run from 0 up to pi and don't say which way that is, so its left and right
  aren't the river's. The outputs use sides instead: side 1 of the cross-section file is the half towards the
  higher rows.

The ordinates step one cell along whichever axis the cross section crosses fastest, so every ordinate lands on a row or
column of cell centres and consecutive ordinates are in different cells. Values are interpolated linearly between the
four cells around each ordinate, as scipy.ndimage.map_coordinates does with order=1. Ordinates beyond the raster's edge
get OFF_RASTER_ELEVATION (9999), which the hydraulics treat as a wall.
"""
import numpy as np
from numba import njit, types
from numba.extending import overload

from .xsection import XSection

OFF_RASTER_ELEVATION = 9999.0  # the value of an ordinate off the raster, a wall that water can't spread past


def sample_cross_section(
        dem: np.ndarray,
        manning_n: np.ndarray,
        row: int,
        col: int,
        stream_direction: float,
        cross_section_length: float,
        dx: float,
        dy: float) -> XSection:
    """
    Sample a cross section from a DEM and a Manning's n raster.

    The cross section is centred on cell (row, col) and runs square to the stream, cross_section_length metres long.
    stream_direction is the stream's direction (see the notes above for the convention). dx and dy are the cells'
    width and height in metres. The values run from the left end of the cross section to the right, with the
    stream cell in the middle. They are float32 for a float32 raster and float64 for any other.
    """
    elevations, mannings_n_values, ordinate_distance = _sample(
        dem, manning_n, int(row), int(col), float(stream_direction), float(cross_section_length), float(dx), float(dy))
    return XSection(elevations, mannings_n_values, ordinate_distance)

def _compute_dem_coordinates(
        row: int,
        col: int,
        stream_direction: float,
        cross_section_length: float,
        dx: float,
        dy: float):
    """
    The column and row coordinates of a cross section's ordinates, from its left end to its right (see the notes
    above), and the distance between ordinates in metres.
    """
    # The cross section runs perpendicular to the stream, from the left bank to the right bank
    xs_direction = stream_direction - np.pi / 2

    # Step one cell along whichever axis the cross section crosses fastest, so every ordinate lands on a
    # row or column of cell centres. The same number of ordinates go out to each side, so the stream cell
    # is always sampled
    ordinate_distance = 1 / max(abs(np.cos(xs_direction)) / dx, abs(np.sin(xs_direction)) / dy)
    num_points_per_side = int(cross_section_length / 2 / ordinate_distance)
    x_coords = ordinate_distance * np.arange(-num_points_per_side, num_points_per_side + 1)
    y_coords = np.zeros_like(x_coords)

    # Rotate the coordinates onto the cross section
    rotation_matrix = np.array([[np.cos(xs_direction), -np.sin(xs_direction)],
                                [np.sin(xs_direction), np.cos(xs_direction)]])
    rotated_coords = rotation_matrix @ np.vstack((x_coords, y_coords))

    # Translate the coordinates to the DEM grid
    dem_x_coords = col + rotated_coords[0, :] / dx
    dem_y_coords = row + rotated_coords[1, :] / dy

    return dem_x_coords, dem_y_coords, ordinate_distance


# --- The same sampling, compiled ------------------------------------------------------------------------------------
# These give the ordinates _compute_dem_coordinates gives, and interpolate at them as map_coordinates does, bit for
# bit: the same two cells along each axis, the same weights, multiplied and summed in the same order.


def _values_like(raster):
    """An empty array of the type a raster's sampled values take: float32 for a float32 raster and float64 for any
    other. Compiled code gets the type when it compiles, so it samples an integer raster without copying it."""
    return np.empty(0, np.float32 if raster.dtype == np.float32 else np.float64)


@overload(_values_like)
def _compiled_values_like(raster):
    if raster.dtype == types.float32:
        return lambda raster: np.empty(0, np.float32)
    return lambda raster: np.empty(0, np.float64)


@njit(cache=True, error_model="numpy")
def _ordinates(stream_direction, cross_section_length, dx, dy):
    """The cross section's direction as a cosine and sine, the spacing of its ordinates and how many go out to each
    side, as _compute_dem_coordinates works them out."""
    xs_direction = stream_direction - np.pi / 2
    cos, sin = np.cos(xs_direction), np.sin(xs_direction)
    spacing = 1 / max(abs(cos) / dx, abs(sin) / dy)
    return cos, sin, spacing, int(cross_section_length / 2 / spacing)


@njit(cache=True, error_model="numpy")
def _interpolate(raster, x, y):
    """The raster's value at column x and row y, or OFF_RASTER_ELEVATION off the raster."""
    rows, cols = raster.shape
    if not (0.0 <= x <= cols - 1 and 0.0 <= y <= rows - 1):
        return OFF_RASTER_ELEVATION
    # The cells either side along each axis (x and y aren't negative, so int() is floor), and like map_coordinates,
    # the last two on the last cell
    c, r = min(int(x), max(cols - 2, 0)), min(int(y), max(rows - 2, 0))
    fx, fy = x - c, y - r
    c1, r1 = min(c + 1, cols - 1), min(r + 1, rows - 1)
    wy0, wx0 = 1.0 - fy, 1.0 - fx
    wy1, wx1 = 1.0 - wy0, 1.0 - wx0  # scipy's second weights, which aren't always fy and fx
    value = raster[r, c] * wy0 * wx0
    value += raster[r, c1] * wy0 * wx1
    value += raster[r1, c] * wy1 * wx0
    value += raster[r1, c1] * wy1 * wx1
    return value


@njit(cache=True, error_model="numpy")
def sample_elevations(dem, row, col, stream_direction, cross_section_length, dx, dy):
    """The elevations and ordinate spacing of the cross section sample_cross_section takes with these arguments,
    for compiled code."""
    cos, sin, spacing, half = _ordinates(stream_direction, cross_section_length, dx, dy)
    elevations = np.empty(2 * half + 1, _values_like(dem).dtype)
    for k in range(-half, half + 1):
        distance = spacing * k
        elevations[k + half] = _interpolate(dem, col + cos * distance / dx, row + sin * distance / dy)
    return elevations, spacing


@njit(cache=True, error_model="numpy")
def _sample(dem, manning_n, row, col, stream_direction, cross_section_length, dx, dy):
    cos, sin, spacing, half = _ordinates(stream_direction, cross_section_length, dx, dy)
    elevations = np.empty(2 * half + 1, _values_like(dem).dtype)
    mannings_n = np.empty(2 * half + 1, _values_like(manning_n).dtype)
    for k in range(-half, half + 1):
        distance = spacing * k
        x, y = col + cos * distance / dx, row + sin * distance / dy
        elevations[k + half] = _interpolate(dem, x, y)
        mannings_n[k + half] = _interpolate(manning_n, x, y)
    return elevations, mannings_n, spacing
