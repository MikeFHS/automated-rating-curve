"""Sampling a cross section from rasters: the values at evenly spaced points (ordinates) across the stream, centred
on a stream cell.

ARC's convention for directions and sides, used throughout the code, is:

- A direction is an angle on the raster's grid, in radians from the +column axis towards the +row axis. 0 runs along a
  row towards the higher columns (east), and pi / 2 along a column towards the higher rows (south on a north-up
  raster, north on a south-up one), so on a north-up raster the angle turns clockwise from east.
- A cross section runs from its left end to its right, as seen looking back along the stream direction on the grid
  drawn with row 0 at the top (on a north-up raster, as on the map), so its right end lies in the direction
  stream direction - pi / 2. That is looking upstream when the direction is the way the water flows, but
  arc.pipeline's stream directions run from 0 up to pi (the angle search can turn them a little past either end) and
  don't say which way that is, so its left and right aren't the river's. The outputs use sides instead: side 1 of the cross-section file is the half towards the
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

def _values_like(raster, like=None):
    """An empty array of the type a raster's sampled values take: float32 for a float32 raster and float64 for any
    other, or like's type, if like (an array) is given. Compiled code gets the type when it compiles, so it samples an
    integer raster without copying it. The values are worked out in float64 whatever the raster's type, so a float32
    raster sampled like a float64 array gives what the float64 copy of it would, without the copy."""
    if like is not None:
        return like[:0]
    return np.empty(0, np.float32 if raster.dtype == np.float32 else np.float64)


@overload(_values_like)
def _compiled_values_like(raster, like=None):
    if like is not None and not isinstance(like, (types.NoneType, types.Omitted)):
        return lambda raster, like=None: like[:0]
    
    if raster.dtype == types.float32:
        return lambda raster, like=None: np.empty(0, np.float32)
    
    return lambda raster, like=None: np.empty(0, np.float64)


@njit(cache=True, error_model="numpy")
def _ordinates(stream_direction, cross_section_length, dx, dy):
    """The cross section's direction as a cosine and sine, the spacing of its ordinates and how many go out to each
    side."""
    xs_direction = stream_direction - np.pi / 2
    cos, sin = np.cos(xs_direction), np.sin(xs_direction)
    spacing = 1 / max(abs(cos) / dx, abs(sin) / dy)
    return cos, sin, spacing, int(cross_section_length / 2 / spacing)


@njit(cache=True, error_model="numpy")
def _corners(rows, cols, x, y):
    """The four cells around column x and row y of a raster rows by cols, which must be on it, and their weights."""
    # The cells either side along each axis (x and y aren't negative, so int() is floor), and like map_coordinates,
    # the last two on the last cell
    c, r = min(int(x), max(cols - 2, 0)), min(int(y), max(rows - 2, 0))
    fx, fy = x - c, y - r
    c1, r1 = min(c + 1, cols - 1), min(r + 1, rows - 1)
    wy0, wx0 = 1.0 - fy, 1.0 - fx
    wy1, wx1 = 1.0 - wy0, 1.0 - wx0  # scipy's second weights, which aren't always fy and fx
    return r, c, r1, c1, wy0, wx0, wy1, wx1


@njit(cache=True, error_model="numpy")
def _interpolate(raster, x, y):
    """The raster's value at column x and row y, or OFF_RASTER_ELEVATION off the raster."""
    rows, cols = raster.shape
    if not (0.0 <= x <= cols - 1 and 0.0 <= y <= rows - 1):
        return OFF_RASTER_ELEVATION
    
    r, c, r1, c1, wy0, wx0, wy1, wx1 = _corners(rows, cols, x, y)
    value = raster[r, c] * wy0 * wx0
    value += raster[r, c1] * wy0 * wx1
    value += raster[r1, c] * wy1 * wx0
    value += raster[r1, c1] * wy1 * wx1
    return value


@njit(cache=True, error_model="numpy")
def _class_value(classes, water, lookup, water_value, r, c):
    return water_value if water[r, c] else lookup[classes[r, c]]


@njit(cache=True, error_model="numpy")
def _interpolate_classes(classes, water, lookup, water_value, x, y):
    """As _interpolate, on the raster whose cells are lookup[classes], or water_value where water is set."""
    rows, cols = classes.shape
    if not (0.0 <= x <= cols - 1 and 0.0 <= y <= rows - 1):
        return OFF_RASTER_ELEVATION
    
    r, c, r1, c1, wy0, wx0, wy1, wx1 = _corners(rows, cols, x, y)
    value = _class_value(classes, water, lookup, water_value, r, c) * wy0 * wx0
    value += _class_value(classes, water, lookup, water_value, r, c1) * wy0 * wx1
    value += _class_value(classes, water, lookup, water_value, r1, c) * wy1 * wx0
    value += _class_value(classes, water, lookup, water_value, r1, c1) * wy1 * wx1
    return value


@njit(cache=True, error_model="numpy")
def sample_classes(classes, water, lookup, water_value, row, col, stream_direction, cross_section_length, dx, dy):
    """As sample_elevations, from the float64 raster whose cells are lookup[classes], or water_value where water is
    set, without making it: a Manning's n raster kept as its land cover classes."""
    cos, sin, spacing, half = _ordinates(stream_direction, cross_section_length, dx, dy)
    values = np.empty(2 * half + 1)
    for k in range(-half, half + 1):
        distance = spacing * k
        values[k + half] = _interpolate_classes(classes, water, lookup, water_value, col + cos * distance / dx,
                                                row + sin * distance / dy)
    return values


@njit(cache=True, error_model="numpy")
def ordinates_above(raster, row, col, stream_direction, cross_section_length, dx, dy, height, like=None):
    """How many ordinates either side of its centre the cross section sample_elevations takes with these arguments
    needs, to take in on each side an ordinate more than `height` above its centre (as stored; ordinates off the raster
    are), or all of them if it doesn't. like is as sample_elevations takes it."""
    cos, sin, spacing, half = _ordinates(stream_direction, cross_section_length, dx, dy)
    stored = np.empty(1, _values_like(raster, like).dtype)
    center = ordinate_value(raster, row, col, cos, sin, spacing, 0, dx, dy, stored)
    needed = 0
    for step in (-1, 1):
        k = 0
        while k < half:
            k += 1
            if ordinate_value(raster, row, col, cos, sin, spacing, step * k, dx, dy, stored) - center > height:
                break

        needed = max(needed, k)

    return needed


@njit(cache=True, error_model="numpy")
def sample_above(raster, row, col, stream_direction, cross_section_length, dx, dy, height, like=None):
    """sample_elevations of the cross section cut to the ordinates either side ordinates_above says it needs (at least
    one): the same elevations and spacing, with each ordinate sampled once."""
    cos, sin, spacing, half = _ordinates(stream_direction, cross_section_length, dx, dy)
    room = max(half, 1)
    values = np.empty(2 * room + 1, _values_like(raster, like).dtype)
    values[room] = _interpolate(raster, col + cos * (spacing * 0) / dx, row + sin * (spacing * 0) / dy)
    sampled = np.zeros(2, np.int64)  # out to where on each side
    for side in range(2):
        step = 2 * side - 1
        k = 0
        while k < half:
            k += 1
            distance = spacing * (step * k)
            values[room + step * k] = _interpolate(raster, col + cos * distance / dx, row + sin * distance / dy)
            if values[room + step * k] - values[room] > height:
                break

        sampled[side] = k

    needed = max(sampled[0], sampled[1], 1)
    for side in range(2):
        step = 2 * side - 1
        for k in range(sampled[side] + 1, needed + 1):
            distance = spacing * (step * k)
            values[room + step * k] = _interpolate(raster, col + cos * distance / dx, row + sin * distance / dy)

    return values[room - needed:room + needed + 1].copy(), spacing


@njit(cache=True, error_model="numpy")
def ordinate_value(raster, row, col, cos, sin, spacing, k, dx, dy, stored):
    """Ordinate k (negative to the left) of the cross section with this direction (cos, sin) and spacing, as
    sample_elevations samples and stores it: stored is a one-element array of the stored type, which it goes
    through."""
    distance = spacing * k
    stored[0] = _interpolate(raster, col + cos * distance / dx, row + sin * distance / dy)
    return stored[0]


@njit(cache=True, error_model="numpy")
def sample_elevations(dem, row, col, stream_direction, cross_section_length, dx, dy, like=None):
    """The elevations and ordinate spacing of the cross section sample_cross_section takes with these arguments,
    for compiled code. like, an array, stores the elevations in its type instead of the DEM's (see _values_like)."""
    cos, sin, spacing, half = _ordinates(stream_direction, cross_section_length, dx, dy)
    elevations = np.empty(2 * half + 1, _values_like(dem, like).dtype)
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
