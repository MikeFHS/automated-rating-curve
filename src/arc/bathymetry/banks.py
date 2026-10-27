"""Finding a cross section's banks, as distances from the stream cell.

Every method gives the distance from the stream cell to each bank, which can fall between ordinates, and the
channel's top width is the sum of the two. find_banks tries the methods in turn and keeps the first that resolves a
channel at least two ordinate spacings wide. Where none does, a drainage-area width prior that says the channel is no
wider than a cell, which no DEM can show, makes it the prior's width. A channel whose width is given, by a prior or a
reach's median width, can be any width, narrower than a cell included: its banks are wherever the width puts them.
"""
from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np
from numba import njit

from arc.hydraulics import top_widths
from arc.xsection.sampling import OFF_RASTER_ELEVATION
from arc.xsection.xsection import XSection

# Ground less than this far above the stream cell is flat water
FLAT_WATER_DEPTH = 0.1
# The ground under a channel's water, out to a bank found from the DEM, falls no further than this below the stream
# cell. Where it falls further, as across a valley from a stream on its side, the water drains away from the stream,
# and the ground that stops it is the valley's far side, not a bank
MAX_BANK_DIP = 2.0
# The width-to-depth search looks no higher than this above the stream cell, and on each side no further out than the
# first ordinate higher than it, which is as far out as arc.pipeline samples a cross section to find its banks
MAX_BANK_HEIGHT = 25.0
# The width-to-depth search's stages are at least this far apart (m), so that it looks past the centimetres of noise
# on a DEM's water or bed (see _width_to_depth_banks)
MIN_STAGE_STEP = 0.1

METHODS = ("none", "single_cell", "land_cover", "width_to_depth_ratio", "flat_water", "target_width")
_NONE, _SINGLE_CELL, _LAND_COVER, _WIDTH_TO_DEPTH, _FLAT_WATER, _TARGET_WIDTH = range(6)
_NO_LAND_COVER = np.empty(0)


class Banks(NamedTuple):
    """Where a cross section's banks are, and how they were found.

    `left` and `right` are the distances in metres from the stream cell to the left and right banks (NaN if the method
    found none), and the elevations are the ground's there. A bank a width put within a spacing of the stream cell,
    where the DEM can't show one, takes the ground at the neighbouring ordinate instead. A single-cell channel is one
    the DEM can't resolve: as wide as the width power laws say, or else one spacing, the stream cell's own width. The
    `single_cell` attribute indicates whether the channel is a single cell, and `valid` says the banks can be used: a
    channel whose width was given (single cell or target width) with the cross section
    on the raster either side, or one found at least two ordinate spacings wide.
    """
    method: str
    left: float
    right: float
    left_elevation: float
    right_elevation: float
    single_cell: bool
    valid: bool

    @property
    def top_width(self) -> float:
        return self.left + self.right


@njit(cache=True, error_model="numpy")
def _on_raster_ordinates(elevations, center, step):
    """How many ordinates beyond the stream cell on one side come before the first wall (or NaN)."""
    end = elevations.size - 1 if step > 0 else 0
    count = 0
    j = center
    while j != end and elevations[j + step] < OFF_RASTER_ELEVATION:
        count += 1
        j += step

    return count


@njit(cache=True, error_model="numpy")
def _ground_at(elevations, center, step, spacing, distance):
    """The ground's elevation `distance` metres from the stream cell on one side, linear between ordinates."""
    last = elevations.size - 1 - center if step > 0 else center
    position = distance / spacing
    if not position >= 0.0:
        return np.nan
    
    if position >= last:
        return float(elevations[center + step * last])
    
    k = int(position)
    z = float(elevations[center + step * k])
    return z + (position - k) * (float(elevations[center + step * (k + 1)]) - z)


@njit(cache=True, error_model="numpy")
def _resolved(left, right, spacing):
    """Whether banks this far out resolve a channel, which needs to be at least two ordinate spacings wide."""
    return left >= 0.0 and right >= 0.0 and left + right >= 2.0 * spacing


@njit(cache=True, error_model="numpy")
def _is_single_cell(target_width, spacing, left_ordinates, right_ordinates):
    """Whether a width prior makes a single-cell channel: no wider than one ordinate spacing, at any resolution."""
    if not 0.0 < target_width < np.inf or left_ordinates < 1 or right_ordinates < 1:
        return False
    
    return target_width <= spacing


@njit(cache=True, error_model="numpy")
def _side_land_cover_bank(land_cover, center, step, water):
    """How many ordinate spacings from the stream cell the boundary with the first ordinate that isn't water is, or
    NaN if the side runs off the raster (NaN land cover) or out of ordinates first."""
    end = land_cover.size - 1 if step > 0 else 0
    j = center
    while j != end:
        j += step
        value = land_cover[j]
        if value == water:
            continue
        if math.isnan(value):
            return np.nan
        return abs(j - center) - 0.5
    
    return np.nan


@njit(cache=True, error_model="numpy")
def _land_cover_banks(land_cover, water, spacing):
    center = land_cover.size // 2
    if not land_cover[center] == water:
        return np.nan, np.nan
    
    return (spacing * _side_land_cover_bank(land_cover, center, -1, water),
            spacing * _side_land_cover_bank(land_cover, center, 1, water))


@njit(cache=True, error_model="numpy")
def _sift_down(heap, i, size):
    """Restore a binary min-heap below position i."""
    value = heap[i]
    while True:
        child = 2 * i + 1
        if child >= size:
            break
        if child + 1 < size and heap[child + 1] < heap[child]:
            child += 1
        if not heap[child] < value:
            break
        heap[i] = heap[child]
        i = child

    heap[i] = value


@njit(cache=True, error_model="numpy")
def _width_to_depth_banks(elevations, spacing):
    """The banks at the stage where the ratio of top width to depth stops falling, and that stage's height above the
    stream cell, or NaNs. The stages tried are the heights of the ordinates above the stream cell, lowest first, out
    on each side to the first more than `MAX_BANK_HEIGHT` above it: as far as water that high could reach, which is as
    far as any stage's width looks. A stage less than `MIN_STAGE_STEP` above the last one tried is skipped, unless it's
    the highest (a short or gently rising floodplain may have no ordinate that much above the bank). This handles DEM noise well. 

    The search usually stops a few stages up, so the heights come off a heap as needed rather than all being sorted.
    """
    center = elevations.size // 2
    thalweg = float(elevations[center])
    heap = np.empty(elevations.size)
    size = 0
    for step in (-1, 1):
        end = elevations.size - 1 if step > 0 else 0
        j = center
        while j != end:
            j += step
            height = float(elevations[j]) - thalweg
            if height > MAX_BANK_HEIGHT:
                break
            if height > 0.0:
                heap[size] = height
                size += 1
                
    for i in range(size // 2 - 1, -1, -1):
        _sift_down(heap, i, size)

    last_left = last_right = last_height = np.nan
    last_ratio = np.inf
    tried = -np.inf
    while size > 0:
        height = heap[0]
        size -= 1
        heap[0] = heap[size]
        _sift_down(heap, 0, size)
        if height - tried < MIN_STAGE_STEP - 1e-9 and size > 0:  # stages 0.1 m apart in rounding count as 0.1 m
            continue

        tried = height
        left, right = top_widths(elevations, spacing, thalweg + height)
        # The widths and the ratio are rounded to the millimetre, which decides where the search stops
        ratio = np.round((np.round(left, 3) + np.round(right, 3)) / height, 3)
        if ratio > last_ratio:
            return last_left, last_right, last_height
        
        last_left, last_right, last_height, last_ratio = left, right, height, ratio

    return np.nan, np.nan, np.nan


@njit(cache=True, error_model="numpy")
def _flat_water_banks(elevations, spacing):
    return top_widths(elevations, spacing, float(elevations[elevations.size // 2]) + FLAT_WATER_DEPTH)


@njit(cache=True, error_model="numpy")
def _target_width_banks(width, spacing, left_ordinates, right_ordinates):
    """Half the width on each side, as far as each side reaches, with what one side can't hold on the other."""
    left_available = left_ordinates * spacing
    right_available = right_ordinates * spacing
    left = min(0.5 * width, left_available)
    right = min(0.5 * width, right_available)
    remaining = max(width - left - right, 0.0)
    left_room = left_available - left
    right_room = right_available - right
    if remaining > 0.0 and left_room + right_room > 0.0:
        left = min(left + remaining * left_room / (left_room + right_room), left_available)
        right = min(right + remaining * right_room / (left_room + right_room), right_available)

    return left, right


@njit(cache=True, error_model="numpy")
def _on_the_raster(elevations, spacing, left, right):
    """Banks where the water's edges are, but none (NaN) on a side whose water runs off the raster: its edge there is
    in the wall beyond it (OFF_RASTER_ELEVATION), not on a bank."""
    center = elevations.size // 2
    if left > spacing * _on_raster_ordinates(elevations, center, -1):
        left = np.nan
    if right > spacing * _on_raster_ordinates(elevations, center, 1):
        right = np.nan

    return left, right


@njit(cache=True, error_model="numpy")
def _drains_away(elevations, spacing, step, distance):
    """Whether the ground under the water out to a bank `distance` metres from the stream cell on one side falls more
    than MAX_BANK_DIP below the stream cell."""
    center = elevations.size // 2
    lowest = float(elevations[center]) - MAX_BANK_DIP
    j = center
    for _ in range(int(distance / spacing)):
        j += step
        if elevations[j] < lowest:
            return True
        
    return False


@njit(cache=True, error_model="numpy")
def _held(elevations, spacing, left, right):
    """Banks where the DEM's water's edges are, but none (NaN) on a side whose water runs off the raster (see
    _on_the_raster) or drains away from the stream (_drains_away)."""
    left, right = _on_the_raster(elevations, spacing, left, right)
    if left >= 0.0 and _drains_away(elevations, spacing, -1, left):
        left = np.nan
    if right >= 0.0 and _drains_away(elevations, spacing, 1, right):
        right = np.nan

    return left, right


@njit(cache=True, error_model="numpy")
def _search(elevations, spacing, land_cover, water, target_width):
    """The method find_banks uses, and the distances to the left and right banks."""
    center = elevations.size // 2
    if land_cover.size == elevations.size:
        left, right = _land_cover_banks(land_cover, water, spacing)
        if _resolved(left, right, spacing):
            return _LAND_COVER, left, right
        
    left, right, _ = _width_to_depth_banks(elevations, spacing)
    left, right = _held(elevations, spacing, left, right)
    if _resolved(left, right, spacing):
        return _WIDTH_TO_DEPTH, left, right
    
    left, right = _held(elevations, spacing, *_flat_water_banks(elevations, spacing))
    if _resolved(left, right, spacing):
        return _FLAT_WATER, left, right
    
    if _is_single_cell(target_width, spacing, _on_raster_ordinates(elevations, center, -1),
                       _on_raster_ordinates(elevations, center, 1)):
        return _SINGLE_CELL, 0.5 * target_width, 0.5 * target_width
    
    return _NONE, np.nan, np.nan


@njit(cache=True, error_model="numpy")
def _at_least(distance, spacing):
    return spacing if distance < spacing else distance  # NaN stays NaN


@njit(cache=True, error_model="numpy")
def _with_elevations(elevations, spacing, method, left, right):
    """The method and the banks, with the ground's elevations at them (see Banks for a given width's banks within a
    spacing of the stream cell)."""
    center = elevations.size // 2
    left_at, right_at = left, right
    if method == _SINGLE_CELL or method == _TARGET_WIDTH:
        left_at, right_at = _at_least(left, spacing), _at_least(right, spacing)

    return (method, left, right, _ground_at(elevations, center, -1, spacing, left_at),
            _ground_at(elevations, center, 1, spacing, right_at))


@njit(cache=True, error_model="numpy")
def _find_banks(elevations, spacing, land_cover, water, target_width):
    """find_banks for arrays: the method used, the distances to the left and right banks, and their elevations."""
    method, left, right = _search(elevations, spacing, land_cover, water, target_width)
    return _with_elevations(elevations, spacing, method, left, right)


def find_banks(xs: XSection, *, target_width: float | None = None, land_cover: np.ndarray | None = None,
               water_value: float | None = None) -> Banks:
    """Find a cross section's banks, trying in turn:

    1. The land cover, or a water mask (water_value 1), if given (see banks_by_land_cover).
    2. The stage where the ratio of top width to depth stops falling (see banks_by_width_to_depth_ratio).
    3. The water's edges 0.1 m above the stream cell (see banks_by_flat_water).

    It keeps the first to resolve a channel at least two ordinate spacings wide. If none does, a drainage-area width
    prior (target_width) no wider than one ordinate spacing makes a single-cell channel that wide: a channel too narrow
    for any DEM to show. Otherwise the banks are not valid, and the bathymetry gives the channel its reach's median
    width, or its prior's, or one cell (single_cell_banks). None of 1 to 3 gives a bank on a side whose water runs off
    the raster: its edge is the raster's, not a bank. Nor do 2 and 3 on a side whose water drains away from the stream,
    over ground more than MAX_BANK_DIP below it.

    The prior comes last so that it doesn't override the banks the water mask, the land cover or the DEM find.
    """
    if land_cover is None:
        values, water = _NO_LAND_COVER, np.nan
    else:
        values, water = _land_cover_values(xs, land_cover, water_value)
    method, left, right, left_elevation, right_elevation = _find_banks(
        xs.elevations, float(xs.ordinate_distance), values, water, _optional(target_width))
    return Banks(METHODS[method], left, right, left_elevation, right_elevation, method == _SINGLE_CELL,
                 method != _NONE)


def banks_by_land_cover(xs: XSection, land_cover: np.ndarray, water_value: float) -> Banks:
    """Banks where the water in the land cover ends: half way between the last ordinate of water out from the stream
    cell, which has to be water too, and the first that isn't. No bank on a side whose water runs to the end of the
    cross section or off the raster (land cover NaN, as sample_land_cover gives)."""
    values, water = _land_cover_values(xs, land_cover, water_value)
    left, right = _land_cover_banks(values, water, float(xs.ordinate_distance))
    return _banks(xs, _LAND_COVER, left, right, _resolved(left, right, xs.ordinate_distance))


def banks_by_width_to_depth_ratio(xs: XSection) -> Banks:
    """Banks at the water's edges at the last stage before the ratio of top width to depth rises.

    The stages tried are the heights of the ordinates above the stream cell, lowest first, out on each side to the
    first more than 25 m above it, so the terrain beyond that doesn't change where the search stops. As the water
    fills the channel the ratio falls, and it rises once the water spills over the banks. No banks if it never rises.
    No bank on a side whose water runs off the raster.
    """
    spacing = float(xs.ordinate_distance)
    left, right, _ = _width_to_depth_banks(xs.elevations, spacing)
    left, right = _held(xs.elevations, spacing, left, right)
    return _banks(xs, _WIDTH_TO_DEPTH, left, right, _resolved(left, right, spacing))


def banks_by_flat_water(xs: XSection) -> Banks:
    """Banks at the water's edges 0.1 m above the stream cell, where the ground stops being flat water. No bank on a
    side whose water runs off the raster, or drains away from the stream."""
    spacing = float(xs.ordinate_distance)
    left, right = _held(xs.elevations, spacing, *_flat_water_banks(xs.elevations, spacing))
    return _banks(xs, _FLAT_WATER, left, right, _resolved(left, right, spacing))


def banks_for_width(xs: XSection, width: float) -> Banks:
    """Banks that make a channel a given width, such as a reach's median width: half the width either side of the
    stream cell, as far as that side stays on the raster, with what one side can't hold on the other. Any width will
    do, narrower than a cell included, so long as the cross section is on the raster either side of the stream cell.
    """
    spacing = float(xs.ordinate_distance)
    center = xs.elevations.size // 2
    left_ordinates = _on_raster_ordinates(xs.elevations, center, -1)
    right_ordinates = _on_raster_ordinates(xs.elevations, center, 1)
    width = _optional(width)
    if not 0.0 < width < np.inf or left_ordinates < 1 or right_ordinates < 1:
        return _banks(xs, _TARGET_WIDTH, np.nan, np.nan, False)
    left, right = _target_width_banks(width, spacing, left_ordinates, right_ordinates)
    return _banks(xs, _TARGET_WIDTH, left, right, True)


def single_cell_banks(xs: XSection) -> Banks:
    """The banks of a channel one cell wide, the stream cell's own width: half a spacing either side of it, if the
    ordinates either side are on the raster. Their elevations are those ordinates' (see Banks)."""
    spacing = float(xs.ordinate_distance)
    center = xs.elevations.size // 2
    valid = _on_raster_ordinates(xs.elevations, center, -1) >= 1 and _on_raster_ordinates(xs.elevations, center, 1) >= 1
    return _banks(xs, _SINGLE_CELL, 0.5 * spacing, 0.5 * spacing, valid)


def set_bank_distances(xs: XSection, banks: Banks) -> None:
    """Store valid banks on the cross section, for the hydraulics to divide it there. Otherwise -1, for none."""
    xs.left_bank_distance = float(banks.left) if banks.valid else -1.0
    xs.right_bank_distance = float(banks.right) if banks.valid else -1.0


def bank_control_elevation(banks: Banks, thalweg: float) -> float:
    """The lower of the two bank elevations above the stream cell, the first level at which water can leave the
    channel, or NaN if neither is above it."""
    above = [elevation for elevation in (banks.left_elevation, banks.right_elevation) if elevation > thalweg]
    return min(above) if above else math.nan


def in_bank(xs: XSection, banks: Banks) -> np.ndarray:
    """Which ordinates are between valid banks, the banks included. None are for banks that aren't valid."""
    n = xs.elevations.size
    offsets = (np.arange(n) - n // 2) * float(xs.ordinate_distance)
    if not banks.valid:
        return np.zeros(n, dtype=bool)
    tolerance = 1e-9 * float(xs.ordinate_distance)
    return (offsets >= -banks.left - tolerance) & (offsets <= banks.right + tolerance)


def set_in_bank_roughness(xs: XSection, banks: Banks, mannings_n: float) -> None:
    """Give the ordinates between valid banks (see in_bank) the water's Manning's n."""
    xs.mannings_n[in_bank(xs, banks)] = mannings_n


def _banks(xs: XSection, method: int, left: float, right: float, valid: bool) -> Banks:
    _, left, right, left_elevation, right_elevation = _with_elevations(
        xs.elevations, float(xs.ordinate_distance), method, float(left), float(right))
    return Banks(METHODS[method], left, right, left_elevation, right_elevation, method == _SINGLE_CELL, bool(valid))


def _land_cover_values(xs: XSection, land_cover: np.ndarray, water_value: float | None) -> tuple[np.ndarray, float]:
    if water_value is None:
        raise TypeError("Give the land cover's water value with the land cover.")
    values = np.asarray(land_cover, dtype=np.float64)
    if values.shape != xs.elevations.shape:
        raise ValueError(f"The land cover has shape {values.shape}, but the cross section has {xs.elevations.shape}.")
    return values, float(water_value)


def _optional(value: float | None) -> float:
    return math.nan if value is None else float(value)
