"""Carving the bathymetry into a cross section.

A channel is carved as a trapezoid between its banks, however wide it is, narrower than a cell included. Its top is
at the reference level at the banks, and its sides slope down to the bed over trapezoid_height of the top width,
measured from the nearer bank. Its bed is flat, depth below the reference level. A channel without valid banks is
carved one cell wide (single_cell_banks).

The carve does two things. The ordinates between the banks, the banks included, take the channel's elevation there.
Those are what the bathymetry raster and the cross-section file get, so a channel narrower than a cell sets just its
stream cell, to the bed. And the cross section gets the channel's exact profile (XSection.profile), for the
hydraulics: the ground outside the banks, the channel's corners between them, and the ground running straight from
each bank's top out to the first ordinate beyond it.

Without bank elevations (Bathy_Use_Banks) the reference level is the stream cell's elevation, and the channel only ever lowers the
ground. With bank elevations the reference level is the bank elevation given, and the channel sets the ground whether that raises
or lowers it.
"""
from __future__ import annotations

import math

import numpy as np
from numba import njit

from arc.bathymetry.banks import Banks, _ground_at, single_cell_banks
from arc.bathymetry.depth import _check_trapezoid_height
from arc.hydraulics import DepthRoughness, _check_slope_factor, _parameters
from arc.rating_curve import section_hydraulics
from arc.xsection.xsection import Profile, XSection

MIN_DEPTH = 1e-6  # metres: the depth carved_depth gives a channel whose ground below the level carries its flow
_MIN_LOG_DEPTH = math.log(MIN_DEPTH)
_MAX_LOG_DEPTH = math.log(1e4)  # no channel is solved deeper than 10 km


@njit(cache=True, error_model="numpy")
def _channel_elevation(offset, left, right, reference, bed, depth, slope_width, tolerance):
    """The channel's elevation offset metres from the stream cell (negative to the left), at or between its banks."""
    to_bank = max(min(offset + left, right - offset), 0.0)  # the distance to the nearer bank
    if slope_width > 0.0:
        return bed + depth * max(1.0 - to_bank / slope_width, 0.0)
    
    return reference if to_bank <= tolerance else bed  # a rectangle: the banks are its top


@njit(cache=True, error_model="numpy")
def _n_at(mannings_n, center, spacing, offset, tolerance):
    """The Manning's n of the segment between ordinates that offset falls in, which is its inner ordinate's."""
    k = math.floor((abs(offset) + tolerance) / spacing)
    index = center + k if offset > 0.0 else center - k
    return mannings_n[min(max(index, 0), mannings_n.size - 1)]


@njit(cache=True, error_model="numpy")
def _ground(ground, center, spacing, offset):
    return _ground_at(ground, center, 1 if offset > 0.0 else -1, spacing, abs(offset))


@njit(cache=True, error_model="numpy")
def _ordinate_range(size, spacing, left, right):
    """The first and last ordinates between the banks, the banks included, and the tolerance for being at one."""
    center = size // 2
    tolerance = 1e-9 * spacing
    first = max(center - math.floor((left + tolerance) / spacing), 0)
    last = min(center + math.floor((right + tolerance) / spacing), size - 1)
    return first, last, tolerance


@njit(cache=True, error_model="numpy")
def _carve(elevations, spacing, left, right, reference, depth, trapezoid_height, lower_only, changed):
    """Carve the channel into the ordinates between its banks, in place, marking the ones it sets in changed."""
    center = elevations.size // 2
    bed = reference - depth
    slope_width = trapezoid_height * (left + right)
    first, last, tolerance = _ordinate_range(elevations.size, spacing, left, right)
    for k in range(first, last + 1):
        elevation = _channel_elevation((k - center) * spacing, left, right, reference, bed, depth, slope_width,
                                       tolerance)
        if not lower_only or elevation < elevations[k]:
            elevations[k] = elevation
            changed[k] = True


@njit(cache=True, error_model="numpy")
def _push(stations, elevations, n, count, station, elevation, mannings_n):
    """Add a vertex to the profile, unless it repeats the last one, and return the new count."""
    if count > 0 and stations[count - 1] == station and elevations[count - 1] == elevation:
        return count
    stations[count] = station
    elevations[count] = elevation
    n[count] = mannings_n
    return count + 1


@njit(cache=True, error_model="numpy")
def _channel_profile(ground, mannings_n, spacing, left, right, reference, depth, trapezoid_height, lower_only):
    """The profile of the channel carved into ground, the ordinates before the carve: its stations, elevations and
    Manning's n, and the stream cell's index."""
    size = ground.size
    center = size // 2
    bed = reference - depth
    slope_width = trapezoid_height * (left + right)
    first, last, tolerance = _ordinate_range(size, spacing, left, right)

    # Where the profile between the banks bends: the banks, the channel's corners and the stream cell, and without
    # bank elevations the ordinates too, since there the ground can be below the channel
    candidates = np.empty(last - first + 4)
    count = 0
    if slope_width > 0.0:
        candidates[0], candidates[1] = -left + slope_width, right - slope_width
        count = 2
    candidates[count] = 0.0
    count += 1
    if lower_only:
        for k in range(first, last + 1):
            offset = (k - center) * spacing
            if -left + tolerance < offset < right - tolerance:
                candidates[count] = offset
                count += 1

    inside = np.sort(candidates[:count])
    points = np.empty(count + 2)
    points[0] = -left
    total = 1
    for i in range(count):
        if inside[i] != points[total - 1]:
            points[total] = inside[i]
            total += 1

    points[total] = right
    total += 1

    # The channel at each of them, the banks' taken from inside the channel, and how far the ground is below it
    inner_top = bed + depth if slope_width > 0.0 else bed
    channel = np.empty(total)
    below = np.empty(total)
    for i in range(total):
        if i == 0 or i == total - 1:
            channel[i] = inner_top
        else:
            channel[i] = _channel_elevation(points[i], left, right, reference, bed, depth, slope_width, tolerance)
        below[i] = channel[i] - _ground(ground, center, spacing, points[i])

    # The part between the banks, from the left bank's top to the right's
    capacity = 2 * total + 1
    middle_stations = np.empty(capacity)
    middle = np.empty(capacity)
    middle_n = np.empty(capacity)
    bank_top = _channel_elevation(-left, left, right, reference, bed, depth, slope_width, tolerance)
    if lower_only:
        bank_top = min(bank_top, channel[0] - below[0])
    vertices = _push(middle_stations, middle, middle_n, 0, -left, bank_top,
                     _n_at(mannings_n, center, spacing, -left, tolerance))
    for i in range(total):
        if lower_only and i > 0 and below[i - 1] * below[i] < 0.0:
            # The ground crosses the channel between these points
            fraction = below[i - 1] / (below[i - 1] - below[i])
            station = points[i - 1] + fraction * (points[i] - points[i - 1])
            elevation = channel[i - 1] + fraction * (channel[i] - channel[i - 1])
            vertices = _push(middle_stations, middle, middle_n, vertices, station, elevation,
                             _n_at(mannings_n, center, spacing, station, tolerance))
        elevation = min(channel[i], channel[i] - below[i]) if lower_only else channel[i]
        vertices = _push(middle_stations, middle, middle_n, vertices, points[i], elevation,
                         _n_at(mannings_n, center, spacing, points[i], tolerance))
        
    bank_top = _channel_elevation(right, left, right, reference, bed, depth, slope_width, tolerance)
    if lower_only:
        bank_top = min(bank_top, channel[total - 1] - below[total - 1])

    vertices = _push(middle_stations, middle, middle_n, vertices, right, bank_top,
                     _n_at(mannings_n, center, spacing, right, tolerance))

    # With the ground outside the banks either side, which never shares a station with a bank
    outside = size - 1 - last
    length = first + vertices + outside
    stations = np.empty(length)
    profile = np.empty(length)
    n = np.empty(length)
    for k in range(first):
        stations[k] = (k - center) * spacing

    profile[:first] = ground[:first]
    n[:first] = mannings_n[:first]
    stations[first:first + vertices] = middle_stations[:vertices]
    profile[first:first + vertices] = middle[:vertices]
    n[first:first + vertices] = middle_n[:vertices]
    for k in range(outside):
        stations[first + vertices + k] = (last + 1 + k - center) * spacing

    profile[first + vertices:] = ground[last + 1:]
    n[first + vertices:] = mannings_n[last + 1:]

    stream_cell = first  # always among the points, strictly between the banks
    while stream_cell < length - 1 and stations[stream_cell] != 0.0:
        stream_cell += 1

    return stations, profile, n, stream_cell


@njit(cache=True, error_model="numpy")
def _log_conveyance_carved(log_depth, ground, mannings_n, spacing, left, right, reference, trapezoid_height,
                           lower_only, left_bank, right_bank, k_decay, shallow_factor, deep_factor):
    """ln of the channel's conveyance with its water at the reference level, carved exp(log_depth) deep, as the
    hydraulics work it out on the profile _channel_profile gives (the channel's between left_bank and right_bank, or
    the whole section's where they're -1)."""
    stations, elevations, n, center = _channel_profile(ground, mannings_n, spacing, left, right, reference,
                                                       math.exp(log_depth), trapezoid_height, lower_only)
    k = section_hydraulics(stations, elevations, n, center, left_bank, right_bank, reference, k_decay,
                           shallow_factor, deep_factor)[5]
    return math.log(k) if k > 0.0 else -np.inf


@njit(cache=True, error_model="numpy")
def _carved_depth(ground, mannings_n, spacing, left, right, reference, trapezoid_height, lower_only, left_bank,
                  right_bank, k_decay, shallow_factor, deep_factor, target, guess):
    """The depth at which the carved channel's ln conveyance at the reference level is target (see carved_depth).
    The conveyance rises with the depth, so the root is bracketed out from guess, doubling the step in ln depth, and
    then found by false position, halving the end that stays put (the Illinois method)."""
    u = math.log(guess)
    f = _log_conveyance_carved(u, ground, mannings_n, spacing, left, right, reference, trapezoid_height, lower_only,
                               left_bank, right_bank, k_decay, shallow_factor, deep_factor) - target
    if f == 0.0:
        return guess
    
    step = 0.25
    if f < 0.0:
        lo, f_lo = u, f
        hi = lo + step
        f_hi = _log_conveyance_carved(hi, ground, mannings_n, spacing, left, right, reference, trapezoid_height,
                                      lower_only, left_bank, right_bank, k_decay, shallow_factor, deep_factor) - target
        while f_hi < 0.0:
            if hi > _MAX_LOG_DEPTH:
                return math.nan
            
            lo, f_lo = hi, f_hi
            step *= 2.0
            hi = lo + step
            f_hi = _log_conveyance_carved(hi, ground, mannings_n, spacing, left, right, reference, trapezoid_height,
                                          lower_only, left_bank, right_bank, k_decay, shallow_factor,
                                          deep_factor) - target
    else:
        hi, f_hi = u, f
        lo = hi - step
        f_lo = _log_conveyance_carved(lo, ground, mannings_n, spacing, left, right, reference, trapezoid_height,
                                      lower_only, left_bank, right_bank, k_decay, shallow_factor, deep_factor) - target
        while f_lo > 0.0:
            if lo < _MIN_LOG_DEPTH:
                return MIN_DEPTH  # the ground below the level carries the flow already
            
            hi, f_hi = lo, f_lo
            step *= 2.0
            lo = hi - step
            f_lo = _log_conveyance_carved(lo, ground, mannings_n, spacing, left, right, reference, trapezoid_height,
                                          lower_only, left_bank, right_bank, k_decay, shallow_factor,
                                          deep_factor) - target
    if f_hi == 0.0:
        return math.exp(hi)
    if f_lo == 0.0:
        return math.exp(lo)
    
    side = 0
    for _ in range(200):
        if math.isfinite(f_lo) and math.isfinite(f_hi):
            u = (lo * f_hi - hi * f_lo) / (f_hi - f_lo)
        else:
            u = 0.5 * (lo + hi)
        f = _log_conveyance_carved(u, ground, mannings_n, spacing, left, right, reference, trapezoid_height,
                                   lower_only, left_bank, right_bank, k_decay, shallow_factor, deep_factor) - target
        if abs(f) <= 1e-13 or hi - lo <= 1e-13 * (1.0 + abs(u)):
            break

        if f < 0.0:
            lo, f_lo = u, f
            if side == -1:
                f_hi *= 0.5
            side = -1
        else:
            hi, f_hi = u, f
            if side == 1:
                f_lo *= 0.5
            side = 1

    return math.exp(u)


def carved_depth(xs: XSection, banks: Banks, q: float, slope: float, guess: float, *, trapezoid_height: float,
                 roughness: DepthRoughness | None = None, slope_factor: float = 1.0,
                 bank_elevation: float | None = None, divide: Banks | None = None) -> float:
    """How deep to carve the channel (as carve_channel would, with these banks and bank_elevation) for the cross
    section to carry q at its reference level in its channel, by the hydraulics the rating curves use: the carved
    profile, Manning's n varying with depth as roughness says, and slope_factor. The channel is the part between
    the banks the hydraulics divide it at, as arc.bathymetry.set_bank_distances gives them: `divide` if given (with
    bathymetry and without Bathy_Use_Banks, the bank tops beyond the water's edge it's carved to), and otherwise the
    banks if they're valid, and the whole cross section if not, though the carve is then one cell wide
    (single_cell_banks).

    `guess`, a depth near it, starts the search. `MIN_DEPTH` where the ground already below the level carries q (without
    bank elevations, the carve keeps ground below the channel), and `guess` itself where the carve would carve nothing.
    """
    trapezoid_height = _check_trapezoid_height(trapezoid_height)
    carved = banks if banks.valid else single_cell_banks(xs)
    spacing = float(xs.ordinate_distance)
    elevations = np.asarray(xs.elevations, dtype=np.float64)
    reference = float(elevations[elevations.size // 2]) if bank_elevation is None else float(bank_elevation)
    if not (carved.valid and math.isfinite(reference) and carved.left > 1e-9 * spacing
            and carved.right > 1e-9 * spacing and xs.mannings_n.size == elevations.size):
        return guess
    
    hydraulic = banks if divide is None else divide
    left_bank, right_bank = (float(hydraulic.left), float(hydraulic.right)) if hydraulic.valid else (-1.0, -1.0)
    target = math.log(q / (math.sqrt(slope) * _check_slope_factor(slope_factor)))
    return float(_carved_depth(elevations, np.asarray(xs.mannings_n, dtype=np.float64), spacing, float(carved.left),
                               float(carved.right), reference, trapezoid_height, bank_elevation is None, left_bank,
                               right_bank, *_parameters(roughness), target, float(guess)))


def carve_channel(xs: XSection, banks: Banks, depth: float, *, trapezoid_height: float,
                  bank_elevation: float | None = None, keep_ground: bool = False) -> np.ndarray:
    """Carve a channel depth below its reference level into the cross section (see the notes above): set the
    ordinates between its banks, and give the cross section the channel's exact profile. With keep_ground, the
    cross section keeps the ordinates as they were before its first carve, and the banks of its last
    (XSection.ground and carved_banks, for arc.xsection.subcell).

    Returns which ordinates it set, for burn_into_raster. Sets none, and no profile, for a depth or reference level
    that isn't finite, or a channel without valid banks where the cross section isn't on the raster either side of
    the stream cell.
    """
    trapezoid_height = _check_trapezoid_height(trapezoid_height)
    elevations = xs.elevations
    center = elevations.size // 2
    spacing = float(xs.ordinate_distance)
    changed = np.zeros(elevations.size, dtype=bool)
    if xs.mannings_n.size != elevations.size:
        raise ValueError("Carving a channel needs the cross section's Manning's n, for its profile.")
    
    reference = float(elevations[center]) if bank_elevation is None else float(bank_elevation)
    if not (math.isfinite(depth) and math.isfinite(reference)):
        return changed
    
    if not banks.valid:
        banks = single_cell_banks(xs)
        if not banks.valid:
            return changed
        
    if not (banks.left > 1e-9 * spacing and banks.right > 1e-9 * spacing):
        return changed  # no width either side of the stream cell to carve
    
    left, right, lower_only = float(banks.left), float(banks.right), bank_elevation is None
    if keep_ground:
        if xs.ground is None:
            xs.ground = elevations.copy()
        xs.carved_banks = (left, right)
        
    stations, profile, n, stream_cell = _channel_profile(elevations, xs.mannings_n, spacing, left, right, reference,
                                                         float(depth), trapezoid_height, lower_only)
    _carve(elevations, spacing, left, right, reference, float(depth), trapezoid_height, lower_only, changed)
    xs.profile = Profile(stations, profile, n, int(stream_cell))
    return changed
