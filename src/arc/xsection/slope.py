"""Stream slopes from the DEM at a reach's stream cells.

The slope between two stream cells is the difference in their elevations over the distance between them along the
stream: the difference in their stations (see arc.xsection.stream_path), or the straight line between them where
that's 0, as for two cells side by side across the stream. Not the larger of the two: on a stream at an angle to the
grid, the straight line between two of its stair-step cells is longer than the stream between them.

The slope methods (Stream_Slope_Method), and when each can't give a slope:

- local_average: each cell's slope is the mean of its slopes to the reach's other cells within Gen_Slope_Dist cells
  of it along each axis, kept between 0.00001 and 0.5 (local_average_slopes). None for a cell with no other cell of
  its reach that near, and the reach's slope is the median of its cells'. With a stream network, arc.pipeline counts
  the cells of the reach's main-stem neighbours too (main_stem_above, and the reach below), so that a cell near a
  junction isn't averaged over one side of it alone: one-sided, the slopes jumped across junctions.
- reach_average: every cell takes its reach's median slope, from the slopes between all pairs of the reach's cells
  within Gen_Slope_Dist cells of each other along each axis, leaving out level pairs. The median is of the slopes
  between two percentiles of them, 5 and 95 by default (reach_median_slope). None for a reach without a pair of cells
  that isn't level. A uniform slope, with both percentiles the same, is a slope like any other.
- local_average_corrected (the default): each cell's local average, kept between its reach's two percentiles
  (corrected_local_slopes). None for a reach without percentiles, as for reach_average.
- end_points: the drop between the ends of the reach's stream line over its length (end_point_slope). None where the
  ends can't be read on the DEM.
- local_fit: each cell's slope is the fall of the straight line fitted by least squares to the elevations of the
  2 * Gen_Slope_Dist + 1 cells nearest it along the stream, and the reach's is the fall of the line through all its
  cells (local_fit_slopes). None for a reach with fewer than two cells.

Whatever the method, a reach without a slope takes its neighbours' in the stream network
(fill_unresolved_reach_slopes), and a cell without one takes its reach's. arc.pipeline skips the cells of a reach
left without a slope.
"""
from __future__ import annotations

import math

import networkx as nx
import numpy as np
from numba import njit

from arc.xsection.stream_path import _columns_between, _row_index

MIN_SLOPE = 1e-5
MAX_SLOPE = 0.5
NO_DATA = -9999.0  # an elevation at or below this is missing


@njit(cache=True, error_model="numpy")
def _between(rows, cols, stations, dx, dy, i, k):
    """How far apart two cells are along the stream: the difference in their stations, or the straight line between
    them where that's 0."""
    along = abs(stations[k] - stations[i])
    if along > 0.0:
        return along
    
    across, down = (cols[k] - cols[i]) * dx, (rows[k] - rows[i]) * dy
    return math.sqrt(across * across + down * down)


@njit(cache=True, error_model="numpy")
def _local_average_slopes(z, rows, cols, stations, dx, dy, distance):
    n = rows.size
    order, sorted_cols, row_start, first_row = _row_index(rows, cols)
    last_row = first_row + row_start.size - 2
    slopes = np.empty(n)
    for i in range(n):
        total = 0.0
        count = 0
        for row in range(max(rows[i] - distance, first_row), min(rows[i] + distance, last_row) + 1):
            lo, hi = _columns_between(sorted_cols, row_start[row - first_row], row_start[row - first_row + 1],
                                      cols[i] - distance, cols[i] + distance)
            for p in range(lo, hi):
                k = order[p]
                if k == i:
                    continue
                length = _between(rows, cols, stations, dx, dy, i, k)
                if length > 0.0:
                    total += abs(z[i] - z[k]) / length
                    count += 1

        if count == 0:  # no other cell of the reach near enough to measure a slope to
            slopes[i] = np.nan
            continue

        slope = total / count
        slope = max(MIN_SLOPE, slope)
        slope = min(MAX_SLOPE, slope)
        slopes[i] = slope

    return slopes


@njit(cache=True, error_model="numpy")
def _pair_slopes(z, rows, cols, stations, dx, dy, distance):
    """The slope between each pair of cells within `distance` cells of each other along each axis, rounded to 1e-8,
    for the pairs whose slope rounds to more than 0. Each pair counts once."""
    n = rows.size
    order, sorted_cols, row_start, first_row = _row_index(rows, cols)
    last_row = first_row + row_start.size - 2
    slopes = np.empty(n * (2 * distance + 1) + 16)  # room for every pair along a path of cells
    size = 0
    for p in range(n):
        i = order[p]
        for row in range(rows[i], min(rows[i] + distance, last_row) + 1):
            lo, hi = _columns_between(sorted_cols, row_start[row - first_row], row_start[row - first_row + 1],
                                      cols[i] - distance, cols[i] + distance)
            if row == rows[i]:
                lo = p + 1  # in the cell's own row only the cells after it, so that each pair counts once

            for q in range(lo, hi):
                k = order[q]
                length = _between(rows, cols, stations, dx, dy, i, k)
                if length == 0.0:
                    continue

                slope = np.rint(abs(z[i] - z[k]) / length * 1e8) / 1e8  # np.round(..., 8), without its pow
                if slope > 0.0:
                    if size == slopes.size:
                        slopes = np.concatenate((slopes, np.empty(slopes.size)))
                    slopes[size] = slope
                    size += 1

    return slopes[:size]


@njit(cache=True, error_model="numpy")
def _round_significant(x, digits):
    """x rounded to `digits` significant figures."""
    if x == 0.0:
        return 0.0
    
    if not np.isfinite(x):
        return x
    
    factor = 10.0 ** (digits - 1 - int(math.floor(math.log10(abs(x)))))
    return math.floor(x * factor + 0.5) / factor


@njit(cache=True, error_model="numpy")
def _reach_median_slope(z, rows, cols, stations, dx, dy, distance, low, high):
    if rows.size < 2:
        return np.nan, np.nan, np.nan
    
    slopes = _pair_slopes(z, rows, cols, stations, dx, dy, distance)
    if slopes.size == 0:
        return np.nan, np.nan, np.nan
    
    # The percentiles are of the slopes rounded to 8 significant figures. That changes nothing below 1, where the
    # slopes already have 8 decimals, so only those above need it
    rounded = slopes
    if slopes.max() >= 1.0:
        rounded = slopes.copy()
        for k in range(slopes.size):
            if slopes[k] >= 1.0:
                rounded[k] = _round_significant(slopes[k], 8)

    percentiles = np.percentile(rounded, np.array([low, high]))
    lower, upper = np.round(percentiles[0], 8), np.round(percentiles[1], 8)
    kept = np.empty(slopes.size)
    count = 0
    for slope in slopes:
        if lower <= slope <= upper:
            kept[count] = slope
            count += 1

    if count == 0:  # two slopes, neither between their percentiles
        return np.median(slopes), lower, upper
    
    return np.median(kept[:count]), lower, upper


@njit(cache=True, error_model="numpy")
def _fitted_line_slope(s, z, lo, hi):
    """The slope m of the least-squares line z = m s + b through the points lo to hi - 1, NaN if their stations are
    all the same."""
    n = hi - lo
    mean_s = 0.0
    mean_z = 0.0
    for k in range(lo, hi):
        mean_s += s[k]
        mean_z += z[k]

    mean_s /= n
    mean_z /= n
    sxx = 0.0
    sxz = 0.0
    for k in range(lo, hi):
        ds = s[k] - mean_s
        sxx += ds * ds
        sxz += ds * (z[k] - mean_z)

    return sxz / sxx if sxx > 0.0 else np.nan


@njit(cache=True, error_model="numpy")
def _local_fit_slopes(z, stations, distance):
    n = z.size
    slopes = np.full(n, np.nan)
    index = np.empty(n, np.int64)
    count = 0
    for i in range(n):
        if math.isfinite(z[i]) and math.isfinite(stations[i]):
            index[count] = i
            count += 1

    if count < 2:
        return slopes, np.nan
    
    index = index[:count]
    order = index[np.argsort(stations[index], kind="mergesort")]
    s = stations[order]
    elevations = z[order]
    whole = _fitted_line_slope(s, elevations, 0, count)
    if math.isnan(whole):
        return slopes, np.nan
    
    width = min(2 * distance + 1, count)
    for p in range(count):
        lo = min(max(p - distance, 0), count - width)
        m = _fitted_line_slope(s, elevations, lo, lo + width)
        if math.isnan(m):
            continue
        # The fall per metre the way the reach's line falls
        fall = -m if whole < 0.0 else (m if whole > 0.0 else abs(m))
        slopes[order[p]] = min(max(fall, MIN_SLOPE), MAX_SLOPE)

    return slopes, min(max(abs(whole), MIN_SLOPE), MAX_SLOPE)


def _reach_arrays(z, rows, cols, stations):
    rows = np.asarray(rows, dtype=np.int64)
    cols = np.asarray(cols, dtype=np.int64)
    z = np.asarray(z)
    stations = np.asarray(stations, dtype=np.float64)
    if not rows.ndim == 1 or not rows.shape == cols.shape == z.shape == stations.shape:
        raise ValueError(f"z, rows, cols and stations must be 1-D and the same length, not {z.shape}, {rows.shape}, "
                         f"{cols.shape} and {stations.shape}.")
    
    return z, rows, cols, stations


def local_average_slopes(z, rows, cols, stations, dx: float, dy: float, distance: int) -> np.ndarray:
    """Each of a reach's cells' slope, the mean of its slopes to the reach's other cells within `distance` cells of
    it along each axis, from `MIN_SLOPE` to `MAX_SLOPE`. NaN for a cell with no other cell that near.

    `z` is the DEM at the cells, rows and cols are the cells, which should be all of the reach's cells, and stations are
    their distances along the stream from one end (along_stream_stations).
    """
    z, rows, cols, stations = _reach_arrays(z, rows, cols, stations)
    if rows.size == 0:
        return np.empty(0)
    
    return _local_average_slopes(z, rows, cols, stations, float(dx), float(dy), int(distance))


def local_fit_slopes(z, stations, distance: int) -> tuple[np.ndarray, float]:
    """Each of a reach's cells' slope from a straight line fitted to the elevations along the stream, and the reach's
    slope (the local_fit method).

    A cell's line is the least-squares fit of elevation against station through the 2 * distance + 1 cells nearest
    it along the stream: itself and `distance` either side, or more on one side near the reach's ends, so that every
    line has as many cells while the reach has them. The reach's line goes through all of its cells. A slope is the
    line's fall per metre the way the reach's line falls, kept between `MIN_SLOPE` and `MAX_SLOPE`, so a stretch rising against
    the reach takes the least slope. A rise between cells offsets a fall, where the methods that average slopes
    between pairs of cells count both as falls.

    `z` is the DEM at the cells, NaN where it has no data (those cells are left out, and get NaN), and stations are their
    distances along the stream from one end (along_stream_stations). All NaN for a reach with fewer than two cells
    with elevations, or whose cells all share one station.
    """
    z = np.asarray(z, dtype=np.float64)
    stations = np.asarray(stations, dtype=np.float64)
    if not z.ndim == 1 or z.shape != stations.shape:
        raise ValueError(f"z and stations must be 1-D and the same length, not {z.shape} and {stations.shape}.")
    
    if int(distance) < 1:
        raise ValueError(f"distance must be at least 1 cell, not {distance!r}.")
    
    slopes, reach_slope = _local_fit_slopes(z, stations, int(distance))
    return slopes, float(reach_slope)


def reach_median_slope(z, rows, cols, stations, dx: float, dy: float, distance: int, low_percentile: float = 5,
                       high_percentile: float = 95) -> tuple[float, float, float]:
    """A reach's slope, and the percentiles its slope is taken between.

    The slopes are between each pair of the reach's cells within `distance` cells of each other along each axis,
    leaving out those that round to 0 at 1e-8. The reach's slope is the median of those between the two percentiles,
    or of all of them if none is. All three are NaN for a reach without slopes: fewer than two cells, no two within
    the distance, or every pair level. The arguments are as for local_average_slopes.
    """
    z, rows, cols, stations = _reach_arrays(z, rows, cols, stations)
    if rows.size == 0:
        return math.nan, math.nan, math.nan
    
    slope, lower, upper = _reach_median_slope(z, rows, cols, stations, float(dx), float(dy), int(distance),
                                              float(low_percentile), float(high_percentile))
    return float(slope), float(lower), float(upper)


def corrected_local_slopes(local_slopes, lower: float, upper: float) -> np.ndarray:
    """Local average slopes kept between their reach's percentiles. A NaN stays NaN."""
    local_slopes = np.asarray(local_slopes, dtype=np.float64)
    return np.where(local_slopes < lower, lower, np.where(local_slopes > upper, upper, local_slopes))


def main_stem_above(network: nx.DiGraph, reach: int, among=None) -> int | None:
    """The reach's upstream neighbour on its main stem: the one with the most reaches above it, and of those the
    lowest ID, among the reaches in `among` if it's given. None if the reach has no such neighbour."""
    upstream = [p for p in network.predecessors(reach) if among is None or p in among]
    if not upstream:
        return None
    
    return max(upstream, key=lambda p: (len(nx.ancestors(network, p)), -p))


def fill_unresolved_reach_slopes(network: nx.DiGraph, slopes: dict) -> set:
    """Give the reaches without a slope (NaN) their neighbours' slope, in place, and return the reaches still left
    without one.

    A reach takes the mean of the slopes of its upstream neighbour with the most reaches above it and its downstream
    neighbour, or the one of those it has, among neighbours with a slope. This repeats while any reach gets filled,
    each time from the reaches filled before. network runs from each reach to the one downstream of it, and reaches
    not in it aren't filled.
    """
    unresolved = {reach for reach, slope in slopes.items() if math.isnan(slope)}
    while True:
        filled = {}
        for reach in sorted(unresolved):
            if reach not in network:
                continue

            main_stem = main_stem_above(network, reach, slopes.keys() - unresolved)
            downstream = [s for s in network.successors(reach) if s in slopes and s not in unresolved]
            neighbour_slopes = []
            if main_stem is not None:
                neighbour_slopes.append(slopes[main_stem])
            if downstream:
                neighbour_slopes.append(slopes[downstream[0]])
            if neighbour_slopes:
                filled[reach] = sum(neighbour_slopes) / len(neighbour_slopes)

        if not filled:
            return unresolved
        slopes.update(filled)
        unresolved -= filled.keys()


def end_point_slope(line, length: float, dem: np.ndarray, geotransform) -> float:
    """The drop between the ends of a reach's line over its length in metres, at least 0.00001 (the end_points
    method).

    `line` is a shapely LineString or MultiLineString in the DEM's coordinates, of which only the longest part counts.
    geotransform is the DEM's GDAL geotransform, north-up or south-up but not rotated. Each end is the DEM at the
    cell holding it, and an end off the raster or without data (at or below -9999, or NaN) moves along the line, 2% of
    its length at a time, to the first point with data, which shortens the length between them. NaN if no point has
    any.
    """
    from shapely.geometry import LineString, MultiLineString

    if line is None or line.is_empty or not length > 0.0:
        return math.nan
    
    if isinstance(line, MultiLineString):
        whole = line.length
        line = max(line.geoms, key=lambda part: part.length)
        length *= line.length / whole
    elif not isinstance(line, LineString):
        line = LineString(line)

    if len(line.coords) < 2:
        return math.nan

    z_start, start = _first_elevation_along(line, dem, geotransform)
    z_end, end = _first_elevation_along(LineString(line.coords[::-1]), dem, geotransform)
    if math.isnan(z_start) or math.isnan(z_end):
        return math.nan
    
    along = (1.0 - end - start) * length
    if not along > 0.0:
        return math.nan
    
    return max(round(abs(z_end - z_start) / along, 8), MIN_SLOPE)


def _first_elevation_along(line, dem, geotransform, step=0.02) -> tuple[float, float]:
    """The DEM at the first point with data along the line from its start, and how far along it is as a share of
    the line's length."""
    for i in range(int(round(1 / step)) + 1):
        fraction = i * step
        point = line.interpolate(fraction, normalized=True)
        z = _elevation_at(point.x, point.y, dem, geotransform)
        if not math.isnan(z):
            return z, fraction
        
    return math.nan, math.nan


def _elevation_at(x, y, dem, geotransform) -> float:
    col = math.floor((x - geotransform[0]) / geotransform[1])
    row = math.floor((y - geotransform[3]) / geotransform[5])
    if not (0 <= row < dem.shape[0] and 0 <= col < dem.shape[1]):
        return math.nan
    z = float(dem[row, col])
    return z if math.isfinite(z) and z > NO_DATA else math.nan
