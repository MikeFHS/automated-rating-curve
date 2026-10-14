"""Which way a cross section runs: the stream's direction at a cell, and the search that turns a cross section to
where the channel is narrowest.

The stream's direction at a cell is the line that best fits its reach's cells in a square around it (Gen_Dir_Dist
cells each way), measured in metres, and the cross section runs square to it. The search then tries turning the
cross section a little each way (Degree_Manip and Degree_Interval) and keeps the direction where the water surface
TEST_DEPTH (5 m) above the stream cell is narrowest, since a cross section square to a valley crosses it by the
shortest way. At 5 m the water usually spans the floodplain, so the search follows the valley rather than the
rasterised edges of a channel a cell or two wide.

Where the water reaches the end of a cross section, or of the raster, on one side, the widths there are compared
only as far out as every direction's cross section reaches, so such ties go to the direction the search started from.
"""
from __future__ import annotations

import math

import numpy as np
from numba import njit

from arc.hydraulics import top_widths
from arc.xsection.sampling import OFF_RASTER_ELEVATION, sample_elevations

TEST_DEPTH = 5.0  # how far above the stream cell the search measures the water's width
_SAME_DIRECTION = 1e-12  # radians


@njit(cache=True, error_model="numpy")
def stream_direction(streams, row, col, distance, dx, dy):
    """The direction of the stream at a cell, as sample_cross_section takes it (see arc.xsection.sampling for the
    convention).

    It's the principal axis, in metres, of the cells of the cell's reach (the cells holding the same ID in the stream
    raster) within `distance` cells of it along each axis, the cell included. It is between 0 and pi, and says
    nothing about which way along that line the water flows. pi / 2 for a cell with no other cell of its reach within
    the distance, so that its cross section runs along its row. Compiled, so compiled code can call it too.
    """
    if not streams[row, col] > 0:
        raise ValueError("The cell isn't a stream cell.")
    reach = streams[row, col]
    first_row, last_row = max(row - distance, 0), min(row + distance, streams.shape[0] - 1)
    first_col, last_col = max(col - distance, 0), min(col + distance, streams.shape[1] - 1)
    n = 0
    sx = sy = sxx = syy = sxy = 0.0
    for r in range(first_row, last_row + 1):
        y = (r - row) * dy
        for c in range(first_col, last_col + 1):
            if streams[r, c] == reach:
                x = (c - col) * dx
                n += 1
                sx += x
                sy += y
                sxx += x * x
                syy += y * y
                sxy += x * y
    if n < 2:
        return np.pi / 2
    # The principal axis of the cells' positions, from their covariances
    direction = 0.5 * math.atan2(2.0 * (sxy - sx * sy / n), (sxx - sx * sx / n) - (syy - sy * sy / n))
    if direction < 0.0:
        direction += np.pi
    return direction


def angle_offsets(degree_manipulation: float, degree_interval: float) -> np.ndarray:
    """The turns from the stream's direction for the search to try, in radians: none, then one interval each way,
    then two, and so on to half of degree_manipulation (both in degrees)."""
    offsets = [0.0]
    if degree_manipulation > 0.0 and degree_interval > 0.0:
        for step in range(1, int(degree_manipulation / (2.0 * degree_interval)) + 1):
            offsets += [-step * degree_interval, step * degree_interval]
    return np.multiply(offsets, math.pi / 180.0)


@njit(cache=True, error_model="numpy")
def _reach(elevations, center, step, spacing):
    """How far the cross section reaches on one side before its first ordinate off the raster (or NaN)."""
    end = elevations.size - 1 if step > 0 else 0
    j = center
    while j != end and elevations[j + step] < OFF_RASTER_ELEVATION:
        j += step
    return spacing * abs(j - center)


@njit(cache=True, error_model="numpy")
def _narrowest_direction(dem, row, col, stream_direction, cross_section_length, dx, dy, offsets, test_depth):
    tried = np.empty(offsets.size)
    widths = np.empty((offsets.size, 2))
    reaches = np.empty((offsets.size, 2))
    count = 0
    for offset in offsets:
        # The same cross section, turned half a turn, is the same direction with its sides swapped. Keep every
        # direction's left side on the same side, and try each direction once.
        offset = offset % np.pi
        if offset > np.pi / 2:
            offset -= np.pi
        repeat = False
        for j in range(count):
            if abs(tried[j] - offset) < _SAME_DIRECTION:
                repeat = True
        if repeat:
            continue
        elevations, spacing = sample_elevations(dem, row, col, stream_direction + offset, cross_section_length, dx,
                                                dy)
        center = elevations.size // 2
        left, right = top_widths(elevations, spacing, elevations[center] + test_depth)
        left_reach, right_reach = _reach(elevations, center, -1, spacing), _reach(elevations, center, 1, spacing)
        # Water that reaches the raster's edge, or data it can't see past, spans at least the reach
        tried[count] = offset
        widths[count, 0] = left if left <= left_reach else left_reach
        widths[count, 1] = right if right <= right_reach else right_reach
        reaches[count, 0], reaches[count, 1] = left_reach, right_reach
        count += 1

    left_limit, right_limit = reaches[:count, 0].min(), reaches[:count, 1].min()
    best, narrowest = 0, np.inf
    for j in range(count):
        width = min(widths[j, 0], left_limit) + min(widths[j, 1], right_limit)
        if width < narrowest:
            best, narrowest = j, width
    return stream_direction + tried[best]


def narrowest_direction(dem: np.ndarray, row: int, col: int, stream_direction: float, cross_section_length: float,
                        dx: float, dy: float, offsets: np.ndarray, test_depth: float = TEST_DEPTH) -> float:
    """The stream direction, turned by one of the offsets (see angle_offsets), whose cross section is narrowest
    test_depth above the stream cell.

    The arguments are sample_cross_section's. Each side's width is compared only as far out as every direction's
    cross section reaches on that side (the raster's edge, or the first NaN, can be nearer than its end). Of
    directions equally narrow, the first offset wins, so with angle_offsets the direction the search started from.
    An offset of more than a quarter turn either way is taken half a turn round, which is the same cross section with
    its sides swapped.
    """
    offsets = np.asarray(offsets, dtype=np.float64).ravel()
    if offsets.size == 0 or (offsets.size == 1 and offsets[0] == 0.0):
        return float(stream_direction)
    return float(_narrowest_direction(dem, int(row), int(col), float(stream_direction), float(cross_section_length),
                                      float(dx), float(dy), offsets, float(test_depth)))
