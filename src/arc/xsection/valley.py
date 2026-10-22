"""Overbanks square to the valley (Valley_Sections).

Manning's equation takes the water to flow square to the cross section. In the channel it follows the channel, and a
cross section square to the channel is right. Over the floodplain of a meandering river it runs straight down the
valley, crossing the channel's loops, while each stream cell's cross section is square to the channel wherever the
channel happens to point: where the channel crosses the valley, it cuts the floodplain at a slant, so the water looks
wider than it is, and the slope it takes is the channel's, gentler than the valley's by the channel's sinuosity. On a
synthetic meandering river (docs/valley_sections.md), against a 2D model, ARC's water over the floodplain swung from
0.3 m too low to 0.1 m too high from one meander to the next, and was out by 0.15 m on average.

With Valley_Sections, each stream cell's cross section keeps its channel, between its banks, and takes each overbank
from a line out from that bank square to the valley: one line across the valley, kinked at the banks. The valley's
direction is the stream's, smoothed along it about a meander either way (Valley_Dir_Widths of the reach's channel
widths, arc.pipeline.valley_windows), and its sinuosity is the stream's length over the smoothed line's
(valley_axes). The water lines it crosses beyond the banks (the river's other bends, other streams, lakes) are filled
to their lips (fill_other_water). The two join in one profile (splice).

The overbank water runs a shorter way down the valley than the channel's, by the sinuosity, so it falls more steeply:
the valley's slope is the channel's times the sinuosity (slope_ratio takes the channel's slope averaged over the same
stretch). ARC's hydraulics take one slope, the channel's, so the overbanks' Manning's n is divided by the square root
of the valley's slope over the channel's instead, which gives their conveyance the valley's slope: Manning's
discharge, A R^(2/3) S^(1/2) / n, is the same with n / sqrt(S_v / S) and S as with n and S_v, whatever the depth. It's
what HEC-RAS does with an overbank's flow length shorter than its channel's: the standard step's friction losses,
worked out over the channel's length, come out the same.

Where the valley runs within a degree of the cell's cross section's own direction and the stream barely winds
(sinuosity under 1.005), there's nothing to change, and the cross section is left as it is.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numba import njit

from arc.xsection.sampling import OFF_RASTER_ELEVATION, _interpolate, _interpolate_classes
from arc.xsection.stream_path import along_stream_stations
from arc.xsection.xsection import Profile

MAX_SINUOSITY = 3.0  # the valley's slope at most three times the channel's
MIN_TURN = math.radians(1.0)  # turns smaller than this, with a sinuosity under MIN_SINUOSITY, leave the section be
MIN_SINUOSITY = 1.005


@dataclass(eq=False, repr=False)
class Valley:
    """A cross section's overbanks square to the valley: the valley's direction (the stream's direction, as
    arc.xsection.sampling takes it, so the lines run square to it), the stream's sinuosity there, the overbanks'
    slope over the channel's (slope_ratio), and the ground and Manning's n at the ordinates of the two lines out from
    the cross section's banks, as one array: the left line's from its far end in, the stream cell's, and the right
    line's from the bank out, ordinate_distance apart (and as far from the banks). full_half is how many ordinates
    each line would have out to the raster's edge; rows and cols are the ordinates' cells."""
    direction: float
    sinuosity: float
    slope_ratio: float
    elevations: np.ndarray
    mannings_n: np.ndarray
    ordinate_distance: float
    full_half: int
    rows: np.ndarray
    cols: np.ndarray


def valley_axes(rows, cols, reaches, dx: float, dy: float, cells_each_way,
                slopes=None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Each stream cell's valley direction (radians from +col towards +row, in [0, pi)), its stream's sinuosity (at
    least 1, at most MAX_SINUOSITY), and the channel slopes (slopes, one a cell) averaged the same way: the steps from
    each of the reach's cells to the next, by their stations along it (arc.xsection.stream_path), weighted by a
    Gaussian of half cells_each_way cells along the stream either side of the cell (out to three times that, as far as
    the reach goes), one number for all, or one a cell, of which each reach takes its cells' median. The direction is the weighted sum's, the sinuosity the steps' weighted length over that sum's, and
    the slope the weighted mean of the cells' (those that are positive). NaN for a reach of fewer than three cells,
    and for the slopes without slopes.

    A meander's steps that cross the valley one way are cancelled by those crossing it back, but only all of them: a
    straight line between the cells either side, a window with sharp ends, swings with the meanders until it spans a
    whole number of them, while the Gaussian's error only falls as it lengthens. On the synthetic meanders of
    docs/valley_sections.md, once cells_each_way spans about a meander along the stream, the direction is within a few
    degrees of the valley's."""
    rows, cols, reaches = np.asarray(rows), np.asarray(cols), np.asarray(reaches)
    direction = np.full(rows.size, np.nan)
    sinuosity = np.full(rows.size, np.nan)
    mean_slope = np.full(rows.size, np.nan)
    cell = math.sqrt((dx * dx + dy * dy) / 2.0)
    windows = np.broadcast_to(np.asarray(cells_each_way, np.float64), rows.shape)
    order = np.argsort(reaches, kind="stable")
    starts = np.flatnonzero(np.r_[True, reaches[order][1:] != reaches[order][:-1]])
    for begin, end in zip(starts, np.r_[starts[1:], order.size]):
        ks = order[begin:end]
        if ks.size < 3:
            continue
        sigma = 0.5 * float(np.median(windows[ks])) * cell
        stations = along_stream_stations(rows[ks], cols[ks], dx, dy)
        by_station = np.argsort(stations, kind="stable")
        ks, stations = ks[by_station], stations[by_station]
        across, down = np.diff(cols[ks] * float(dx)), np.diff(rows[ks] * float(dy))
        sum_across, sum_down, length = _smoothed(stations, across, down, np.diff(stations),
                                                 0.5 * (stations[1:] + stations[:-1]), sigma)
        chord = np.hypot(sum_across, sum_down)
        ok = chord > 0.0
        direction[ks[ok]] = np.arctan2(sum_down[ok], sum_across[ok]) % math.pi
        sinuosity[ks[ok]] = np.clip(length[ok] / chord[ok], 1.0, MAX_SINUOSITY)
        if slopes is not None:
            mean_slope[ks] = _smoothed_values(stations, np.asarray(slopes, np.float64)[ks], sigma)
    return direction, sinuosity, mean_slope


def slope_ratio(channel_slope: float, mean_slope: float, sinuosity: float) -> float:
    """The overbanks' slope over the channel's, between 1 / MAX_SINUOSITY and MAX_SINUOSITY. The water falls as far
    down the valley as along the channel, over a length shorter by the sinuosity, so the valley's slope is the
    channel's times the sinuosity; but a cell's channel slope is its own, over Gen_Slope_Dist cells, steeper where the
    channel runs down the valley and gentler where it crosses it. So the valley's slope is the channel's slope averaged
    as the sinuosity is (valley_axes's mean_slope) times the sinuosity, which the cell's own then divides. On the
    synthetic meanders, the cells' own slopes times the sinuosity swung 12% to 29% either side of the valley's slope;
    averaged, they were within 1% of it, or 9% on the tightest meanders. The sinuosity where either slope isn't a
    positive number."""
    if not (channel_slope > 0.0 and mean_slope > 0.0):
        return float(sinuosity)
    return float(min(max(mean_slope * sinuosity / channel_slope, 1.0 / MAX_SINUOSITY), MAX_SINUOSITY))


@njit(cache=True, error_model="numpy")
def _smoothed(stations, across, down, lengths, middles, sigma):
    """At each station, the sums of the steps' vectors and lengths weighted by a Gaussian of sigma metres along the
    stream from their middles, out to three sigma (valley_axes). Both stations and middles ascend."""
    n, m = stations.size, middles.size
    sum_across, sum_down, total = np.zeros(n), np.zeros(n), np.zeros(n)
    first = 0
    for i in range(n):
        while first < m and middles[first] < stations[i] - 3.0 * sigma:
            first += 1
        j = first
        while j < m and middles[j] <= stations[i] + 3.0 * sigma:
            w = math.exp(-0.5 * ((middles[j] - stations[i]) / sigma) ** 2)
            sum_across[i] += w * across[j]
            sum_down[i] += w * down[j]
            total[i] += w * lengths[j]
            j += 1
    return sum_across, sum_down, total


@njit(cache=True, error_model="numpy")
def _smoothed_values(stations, values, sigma):
    """At each station, the mean of the positive values weighted by a Gaussian of sigma metres along the stream from
    their stations, out to three sigma (NaN for none). The stations ascend."""
    n = stations.size
    out = np.full(n, np.nan)
    first = 0
    for i in range(n):
        while first < n and stations[first] < stations[i] - 3.0 * sigma:
            first += 1
        total = weight = 0.0
        j = first
        while j < n and stations[j] <= stations[i] + 3.0 * sigma:
            if values[j] > 0.0:
                w = math.exp(-0.5 * ((stations[j] - stations[i]) / sigma) ** 2)
                total += w * values[j]
                weight += w
            j += 1
        if weight > 0.0:
            out[i] = total / weight
    return out


def turn(a: float, b: float) -> float:
    """The angle between two undirected directions, in [0, pi/2]."""
    d = abs((a - b) % math.pi)
    return min(d, math.pi - d)


def worth_splitting(section_direction: float, valley_direction: float, sinuosity: float) -> bool:
    """Whether the valley is far enough from the cross section's own direction, or the stream winds enough, to change
    anything (see the notes above)."""
    if not (np.isfinite(valley_direction) and np.isfinite(sinuosity)):
        return False
    return turn(section_direction, valley_direction) >= MIN_TURN or sinuosity >= MIN_SINUOSITY


def fill_other_water(elevations: np.ndarray, wet: np.ndarray) -> np.ndarray:
    """The overbanks' ground (as Valley holds it) with each stretch of water along either line filled to the lower of
    the dry ground either side of it (the line's far side alone, where it starts at the bank): where the river winds
    back on itself the lines cross its other bends, whose flow, below their banks, is the channel's flow again,
    counted once already; and the water in lakes and other streams isn't the river's flowing down the valley either.
    The stream cell, in the middle, is never water here."""
    z = np.array(elevations, dtype=np.float64)
    wet = np.asarray(wet, dtype=np.bool_)
    _fill_side(z, wet, -1)
    _fill_side(z, wet, 1)
    return z


@njit(cache=True, error_model="numpy")
def _fill_side(z, wet, step):
    """fill_other_water along one line (step -1 for the left), in place."""
    h = z.size // 2
    k = 1
    while k <= h:
        if not wet[h + step * k]:
            k += 1
            continue
        start = k
        while k <= h and wet[h + step * k]:
            k += 1
        lip = np.inf  # the ground either side, but not the stream cell's
        if start > 1:
            lip = z[h + step * (start - 1)]
        if k <= h:
            lip = min(lip, z[h + step * k])
        if lip < np.inf:
            for j in range(start, k):
                z[h + step * j] = max(z[h + step * j], lip)


@njit(cache=True, error_model="numpy")
def ray_values(raster, x0, y0, cos, sin, spacing, count, dx, dy):
    """A raster's values at count ordinates spacing metres apart along a line from column x0, row y0 (not counting
    it) in the direction (cos, sin), interpolated as arc.xsection.sampling samples a cross section."""
    values = np.empty(count)
    for k in range(1, count + 1):
        distance = spacing * k
        values[k - 1] = _interpolate(raster, x0 + cos * distance / dx, y0 + sin * distance / dy)
    return values


@njit(cache=True, error_model="numpy")
def ray_classes(classes, water, lookup, water_value, x0, y0, cos, sin, spacing, count, dx, dy):
    """ray_values on the Manning's n raster kept as its land cover classes (arc.pipeline.ManningsN)."""
    values = np.empty(count)
    for k in range(1, count + 1):
        distance = spacing * k
        values[k - 1] = _interpolate_classes(classes, water, lookup, water_value, x0 + cos * distance / dx,
                                             y0 + sin * distance / dy)
    return values


@njit(cache=True, error_model="numpy")
def ray_reach(raster, x0, y0, cos, sin, spacing, dx, dy, level):
    """How many ordinates along the line (as ray_values) it takes to reach one above level, or off the raster (the
    first off it counted), and how many to leave the raster."""
    rows, cols = raster.shape
    edge = _steps_off(x0, cos * spacing / dx, cols) if cos * spacing / dx != 0.0 else 1 << 62
    edge = min(edge, _steps_off(y0, sin * spacing / dy, rows) if sin * spacing / dy != 0.0 else 1 << 62)
    # The first ordinate off the raster, as _interpolate tells it, which rounding can put a step either side
    while edge > 1 and not _on_raster(x0, y0, cos, sin, spacing, edge - 1, dx, dy, rows, cols):
        edge -= 1
    while _on_raster(x0, y0, cos, sin, spacing, edge, dx, dy, rows, cols):
        edge += 1
    for k in range(1, edge):
        distance = spacing * k
        if _interpolate(raster, x0 + cos * distance / dx, y0 + sin * distance / dy) > level:
            return k, edge
    return edge, edge


@njit(cache=True, error_model="numpy")
def _steps_off(start, step, size):
    """How many steps of `step` from `start` it takes to leave [0, size - 1]."""
    room = (size - 1 - start) / step if step > 0.0 else start / -step
    return max(int(math.floor(room)) + 1, 1)


@njit(cache=True, error_model="numpy")
def _on_raster(x0, y0, cos, sin, spacing, k, dx, dy, rows, cols):
    distance = spacing * k
    x, y = x0 + cos * distance / dx, y0 + sin * distance / dy
    return 0.0 <= x <= cols - 1 and 0.0 <= y <= rows - 1


def splice(profile: Profile, left_bank: float, right_bank: float, valley: Valley) -> Profile:
    """One profile of the channel and the overbanks (see the notes above): profile's vertices between its banks, a
    vertex at each bank on its ground, and beyond each bank the ordinates of its line, as far beyond it as they are
    along the line. The overbanks' Manning's n, the bank vertices' included (a segment takes its inner vertex's n), is
    the lines' over the square root of the slope ratio."""
    stations, elevations, mannings_n, center = profile
    new_stations, new_elevations, new_n, new_center = _splice(
        stations, elevations, mannings_n, int(center), float(left_bank), float(right_bank), valley.elevations,
        valley.mannings_n, float(valley.ordinate_distance), 1.0 / math.sqrt(valley.slope_ratio))
    return Profile(new_stations, new_elevations, new_n, int(new_center))


@njit(cache=True, error_model="numpy")
def _splice(stations, elevations, mannings_n, center, left_bank, right_bank, valley_z, valley_n, spacing, scale):
    """splice's arrays and centre."""
    h = valley_z.size // 2
    first = np.searchsorted(stations, -left_bank, side="right")  # the first vertex inside the banks
    end = np.searchsorted(stations, right_bank, side="left")  # and the first beyond them
    size = h + 1 + (end - first) + 1 + h
    new_stations, new_elevations, new_n = np.empty(size), np.empty(size), np.empty(size)
    i = 0
    for k in range(h, 0, -1):
        new_stations[i] = -left_bank - k * spacing
        new_elevations[i] = valley_z[h - k]
        new_n[i] = valley_n[h - k] * scale
        i += 1
    new_stations[i] = -left_bank
    new_elevations[i] = np.interp(-left_bank, stations, elevations)
    new_n[i] = valley_n[h - 1] * scale if h > 0 else mannings_n[center]
    i += 1
    for j in range(first, end):
        new_stations[i], new_elevations[i], new_n[i] = stations[j], elevations[j], mannings_n[j]
        i += 1
    new_stations[i] = right_bank
    new_elevations[i] = np.interp(right_bank, stations, elevations)
    new_n[i] = valley_n[h + 1] * scale if h > 0 else mannings_n[center]
    i += 1
    for k in range(1, h + 1):
        new_stations[i] = right_bank + k * spacing
        new_elevations[i] = valley_z[h + k]
        new_n[i] = valley_n[h + k] * scale
        i += 1
    return new_stations, new_elevations, new_n, h + 1 + (center - first)
