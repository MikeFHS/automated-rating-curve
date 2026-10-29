"""Hydraulic geometry and Manning's discharge for a cross section.

A cross section's ground is a profile: a polyline through vertices at known distances from the stream cell, each with
an elevation and a Manning's n (arc.xsection.xsection.Profile). Sampled ordinates are a profile a spacing apart, and a
carved channel adds its corners between them (see hydraulic_profile).

The water surface starts at the stream cell and spreads outward on both sides until it meets ground higher than
itself. Ground is straight between vertices, so the water's edge falls part way along a segment. Low ground beyond a
higher point stays dry until the water rises above that point, and both ends of the cross section act as vertical
walls.

Discharge is Manning's equation in SI units. Each wetted segment takes the Manning's n of its end nearer the stream
cell. The section's roughness is the composite n = (sum(P_i * n_i**1.5) / P) ** (2/3), so its conveyance is
K = A**(5/3) / sum(P_i * n_i**1.5) ** (2/3), and its discharge is Q = K * sqrt(slope) * slope_factor.

With banks, a cross section is divided at them into a left overbank, the channel and a right overbank, and its
conveyance is the sum of theirs (the divided channel method). The banks change how the conveyance adds up, not where
the water is. A bank that is negative or NaN (not found), or at or beyond the end of its side, leaves that side all
channel.

Manning's n can vary with the depth of the water over it (DepthRoughness):

    n(h) = n0 * (deep_factor + (shallow_factor - deep_factor) / (1 + k_decay * h)),   h = max(wse - z, 0)

so ground at the water's edge is `shallow_factor` times rougher than n0, falling towards `deep_factor` times n0 in deep
water. Each segment takes the depth over its inner end. Factors of 1 leave n as it is.
"""
from __future__ import annotations

import math
import weakref
from collections import deque
from typing import NamedTuple

import numpy as np
from numba import njit

from arc.xsection.valley import splice
from arc.xsection.xsection import Profile, XSection


class HydraulicGeometry(NamedTuple):
    area: float
    wetted_perimeter: float
    hydraulic_radius: float
    top_width: float
    mannings_n: float  # composite roughness of the wetted perimeter


class CompoundGeometry(NamedTuple):
    """The wetted geometry of each subsection of a cross section divided at its banks."""
    left_overbank: HydraulicGeometry
    channel: HydraulicGeometry
    right_overbank: HydraulicGeometry


class DepthRoughness(NamedTuple):
    """How Manning's n varies with the depth of the water (see the notes above)."""
    k_decay: float = 6.0  # per metre of depth
    shallow_factor: float = 2.0  # at least 1
    deep_factor: float = 1.0  # above 0, at most 1

    def scale(self, n0, depth):
        """n0 scaled for water depth deep over it; depths below 0 count as 0."""
        _check_roughness(self)
        h = np.maximum(depth, 0.0)
        return n0 * (self.deep_factor + (self.shallow_factor - self.deep_factor) / (1.0 + self.k_decay * h))


_INVERSE_CBRT_MAGIC = np.int64(0x553EF0FF289DD796)


@njit(cache=True, error_model="numpy")
def _inverse_cbrt(x):
    """x**(-1/3) for 1e-300 < x < 1e300, to within about 1e-15, without dividing.

    Subtracting a third of the float's bits from a constant gives a first guess within 3.5%. Each step multiplies
    by the start of the binomial series for (1 - e)**(-1/3), where e = 1 - x * r**3, and so has fourth-order
    convergence: two steps reach float64 precision. This takes well under half the time of two fractional powers.
    """
    r = np.int64(_INVERSE_CBRT_MAGIC - np.float64(x).view(np.int64) // 3).view(np.float64)
    for _ in range(2):
        e = 1.0 - x * r * r * r
        r = r * (1.0 + e * (1.0 / 3.0 + e * (2.0 / 9.0 + e * (14.0 / 81.0))))

    return r


@njit(cache=True, error_model="numpy")
def _conveyance(area, weighted_perimeter):
    """A**(5/3) / W**(2/3), where W = sum(P_i * n_i**1.5), as A**2 * (A * W**2)**(-1/3)."""
    if area <= 0.0:
        return 0.0
    
    x = area * weighted_perimeter * weighted_perimeter
    if 1e-300 < x < 1e300:
        return area * area * _inverse_cbrt(x)
    
    return area ** (5.0 / 3.0) / weighted_perimeter ** (2.0 / 3.0)  # out of range, NaN, or no roughness


@njit(cache=True, error_model="numpy")
def _scaled_n_15(n0, depth, k_decay, shallow_factor, deep_factor):
    """(n0 scaled for water depth deep over it) ** 1.5."""
    if shallow_factor == deep_factor and depth == depth:  # noqa: PLR0124
        # n doesn't vary with depth: (shallow - deep) / (1 + k * depth) is 0 at any depth but NaN (k > 0 makes the
        # denominator at least 1, or inf), and deep + 0 is deep, so this is the same number without the division
        n = n0 * deep_factor
    else:
        n = n0 * (deep_factor + (shallow_factor - deep_factor) / (1.0 + k_decay * max(depth, 0.0)))

    return n * math.sqrt(n)


# --- Top widths on the ordinates ------------------------------------------------------------------------------------


@njit(cache=True, error_model="numpy")
def _side_top_width(elevations, center, step, spacing, wse):
    """The distance from the centre out to one water edge, given water above the centre."""
    end = elevations.size - 1 if step > 0 else 0
    j = center
    while j != end:
        z_out = elevations[j + step]
        if not wse > z_out:
            z_in = elevations[j]
            return spacing * (abs(j - center) + (wse - z_in) / (z_out - z_in))
        
        j += step

    return spacing * abs(end - center)  # the water reaches the wall at the end


@njit(cache=True, error_model="numpy")
def top_widths(elevations, spacing, wse):
    """The water's top width to the left and to the right of the stream cell (the middle ordinate) at a water
    surface elevation, which are the distances from the stream cell to the water's edges. Both zero when the water is
    at or below the stream cell."""
    center = elevations.size // 2
    if not wse > elevations[center]:
        return 0.0, 0.0
    
    return (_side_top_width(elevations, center, -1, spacing, wse),
            _side_top_width(elevations, center, 1, spacing, wse))


# --- Profiles -----------------------------------------------------------------------------------------------------
# Two vertices at the same distance are a vertical face, which adds its wetted height to the wetted perimeter and
# nothing to the area or the top width. A face at a bank is in the channel if it rises outward from the channel, and
# in the overbank if it falls, since that is the side whose water it holds.


def uniform_profile(elevations: np.ndarray, mannings_n: np.ndarray, spacing: float) -> Profile:
    """The profile of ordinates a spacing apart, with the stream cell in the middle."""
    center = elevations.size // 2
    stations = np.arange(-center, elevations.size - center, dtype=np.float64)  # (k - center) * spacing, in one array
    stations *= float(spacing)
    return Profile(stations, elevations, mannings_n, center)


# How many cross sections keep their last splice (hydraulic_profile): the most recent, so that the steps that work
# a cross section out again and again in a row, or a reach's at a time, find it, but a run doesn't hold one for every
# cross section at once, as much memory as their valleys
KEPT_SPLICES = 512
# The kept splices, oldest first: each a weak reference to its cross section and the splice's id, which hold none of
# their memory once the cross sections go
_kept_splices: deque = deque()


def hydraulic_profile(xs: XSection) -> Profile:
    """The ground the hydraulics use: the carved channel's exact profile if the cross section has one, and otherwise
    its ordinates; with a valley's overbanks (arc.xsection.valley), its channel between its banks joined to them."""
    valley = xs.valley
    if valley is None or not (xs.left_bank_distance > 0.0 and xs.right_bank_distance > 0.0):
        if xs.profile is not None:
            return xs.profile
        return uniform_profile(xs.elevations, xs.mannings_n, xs.ordinate_distance)

    # Spliced once for the same valley, ground and banks: the cache holds them, so none is a new object in their place
    sources = (valley, xs.profile, xs.elevations, xs.mannings_n, xs.left_bank_distance, xs.right_bank_distance)
    cached = xs._spliced
    if cached is not None and all(a is b or (isinstance(a, float) and a == b) for a, b in zip(cached[0], sources)):
        return cached[1]

    profile = xs.profile if xs.profile is not None else uniform_profile(xs.elevations, xs.mannings_n,
                                                                          xs.ordinate_distance)
    spliced = splice(profile, float(xs.left_bank_distance), float(xs.right_bank_distance), valley)
    xs._spliced = cached = (sources, spliced)
    _kept_splices.append((weakref.ref(xs), id(cached)))
    if len(_kept_splices) > KEPT_SPLICES:
        # The oldest kept goes, unless its cross section has spliced again since
        oldest, entry = _kept_splices.popleft()
        oldest = oldest()
        if oldest is not None and id(oldest._spliced) == entry:
            oldest._spliced = None

    return spliced


@njit(cache=True, error_model="numpy")
def _profile_side(stations, elevations, n0, center, step, wse, bank, extent, k_decay, shallow_factor, deep_factor):
    """The area, wetted perimeter, top width and sum(P_i * n_i**1.5) of the part of one side in the channel, then of
    the part in the overbank.

    The water covers every vertex from the centre out to vertex `extent`, then meets the ground part way along the
    next segment, or the wall if `extent` is the end. A negative extent walks out from the centre to the first vertex
    at or above wse. A bank that is negative or NaN, or at or beyond the side's last vertex, leaves the side all
    channel.
    """
    end = elevations.size - 1 if step > 0 else 0
    walk = extent < 0  # the water's edge is found on the way out: the first vertex at or above wse
    origin = stations[center]
    divided = bank >= 0.0 and bank < abs(stations[end] - origin)
    channel_area = channel_perimeter = channel_top_width = channel_weighted = 0.0
    overbank_area = overbank_perimeter = overbank_top_width = overbank_weighted = 0.0
    j = center
    z_in = elevations[j]
    d_in = abs(stations[j] - origin)
    while True:
        if j == end:
            # The water reaches the wall at the end, in the overbank if the bank is inside the section
            depth_end = wse - elevations[end]
            n_15 = _scaled_n_15(n0[end], depth_end, k_decay, shallow_factor, deep_factor)
            if divided:
                overbank_perimeter += depth_end
                overbank_weighted += depth_end * n_15
            else:
                channel_perimeter += depth_end
                channel_weighted += depth_end * n_15
            break

        z_out = elevations[j + step]
        d_out = abs(stations[j + step] - origin)
        width = d_out - d_in
        rise = z_out - z_in
        depth_in = wse - z_in
        full = wse > z_out if walk else j != extent
        if full:
            wet = 1.0  # the fraction of the segment under water
            depth_out = wse - z_out
        else:
            wet = depth_in / rise  # the water meets the ground part way along the segment
            depth_out = 0.0

        length = math.sqrt(width * width + rise * rise)
        n_15 = _scaled_n_15(n0[j], depth_in, k_decay, shallow_factor, deep_factor)

        if not divided or d_out < bank or (d_out == bank and (width > 0.0 or rise > 0.0)):
            in_channel = wet
        elif d_in >= bank:
            in_channel = 0.0
        else:
            in_channel = min(wet, (bank - d_in) / width)  # the bank divides the segment

        if in_channel == wet:
            channel_area += width * wet * (depth_in + depth_out) / 2
            channel_perimeter += wet * length
            channel_weighted += wet * length * n_15
            channel_top_width += width * wet
        elif in_channel == 0.0:
            overbank_area += width * wet * (depth_in + depth_out) / 2
            overbank_perimeter += wet * length
            overbank_weighted += wet * length * n_15
            overbank_top_width += width * wet
        else:
            depth_bank = depth_in - in_channel * rise
            outer = wet - in_channel
            channel_area += width * in_channel * (depth_in + depth_bank) / 2
            channel_perimeter += in_channel * length
            channel_weighted += in_channel * length * n_15
            channel_top_width += width * in_channel
            overbank_area += width * outer * (depth_bank + depth_out) / 2
            overbank_perimeter += outer * length
            overbank_weighted += outer * length * n_15
            overbank_top_width += width * outer

        if not full:
            break

        j += step
        z_in, d_in = z_out, d_out

    return (channel_area, channel_perimeter, channel_top_width, channel_weighted,
            overbank_area, overbank_perimeter, overbank_top_width, overbank_weighted)


@njit(cache=True, error_model="numpy")
def _profile_sides(stations, elevations, n0, center, step, levels, first, bank, k_decay, shallow_factor, deep_factor,
                   out):
    """_profile_side, walking out to the water's edge, at each water surface levels[first:], which ascend: out[i] gets
    its eight sums at levels[i], each summed in the order _profile_side sums it, so the same numbers. Each segment's
    geometry is worked out once for all the water surfaces over it, which takes half the time of one at a time."""
    m = levels.size
    end = elevations.size - 1 if step > 0 else 0
    origin = stations[center]
    divided = bank >= 0.0 and bank < abs(stations[end] - origin)
    constant_n = shallow_factor == deep_factor
    j = center
    z_in = elevations[j]
    d_in = abs(stations[j] - origin)
    while first < m:
        if j == end:
            # The water reaches the wall at the end
            for i in range(first, m):
                depth_end = levels[i] - elevations[end]
                n_15 = _scaled_n_15(n0[end], depth_end, k_decay, shallow_factor, deep_factor)
                if divided:
                    out[i, 5] += depth_end
                    out[i, 7] += depth_end * n_15
                else:
                    out[i, 1] += depth_end
                    out[i, 3] += depth_end * n_15
            break
        
        z_out = elevations[j + step]
        d_out = abs(stations[j + step] - origin)
        width = d_out - d_in
        rise = z_out - z_in
        length = math.sqrt(width * width + rise * rise)
        # The water surfaces that meet the ground on this segment, which are the lowest of those still walking out
        split = first
        while split < m and not levels[split] > z_out:
            split += 1

        for i in range(first, split):
            depth_in = levels[i] - z_in
            wet = depth_in / rise
            n_15 = _scaled_n_15(n0[j], depth_in, k_decay, shallow_factor, deep_factor)
            _add_segment(out, i, _in_channel(divided, bank, d_in, d_out, width, rise, wet), width, rise, length, wet,
                         depth_in, 0.0, n_15)
            
        if split < m:
            # Those over it all the way, wet 1: how much of it is in the channel is the same for all of them, so each
            # way of dividing it has a loop of its own
            wet = 1.0
            in_channel = _in_channel(divided, bank, d_in, d_out, width, rise, wet)
            n_15_fixed = _scaled_n_15(n0[j], 0.0, k_decay, shallow_factor, deep_factor)
            if in_channel == wet:
                for i in range(split, m):
                    depth_in, depth_out = levels[i] - z_in, levels[i] - z_out
                    n_15 = n_15_fixed if constant_n and depth_in == depth_in else \
                        _scaled_n_15(n0[j], depth_in, k_decay, shallow_factor, deep_factor)  # noqa: PLR0124
                    out[i, 0] += width * wet * (depth_in + depth_out) / 2
                    out[i, 1] += wet * length
                    out[i, 3] += wet * length * n_15
                    out[i, 2] += width * wet
            elif in_channel == 0.0:
                for i in range(split, m):
                    depth_in, depth_out = levels[i] - z_in, levels[i] - z_out
                    n_15 = n_15_fixed if constant_n and depth_in == depth_in else \
                        _scaled_n_15(n0[j], depth_in, k_decay, shallow_factor, deep_factor)  # noqa: PLR0124
                    out[i, 4] += width * wet * (depth_in + depth_out) / 2
                    out[i, 5] += wet * length
                    out[i, 7] += wet * length * n_15
                    out[i, 6] += width * wet
            else:
                for i in range(split, m):
                    depth_in, depth_out = levels[i] - z_in, levels[i] - z_out
                    n_15 = n_15_fixed if constant_n and depth_in == depth_in else \
                        _scaled_n_15(n0[j], depth_in, k_decay, shallow_factor, deep_factor)  # noqa: PLR0124
                    _add_segment(out, i, in_channel, width, rise, length, wet, depth_in, depth_out, n_15)

        first = split
        j += step
        z_in, d_in = z_out, d_out


@njit(cache=True, error_model="numpy")
def _in_channel(divided, bank, d_in, d_out, width, rise, wet):
    """How much of a segment's wet share is in the channel, as _profile_side divides it."""
    if not divided or d_out < bank or (d_out == bank and (width > 0.0 or rise > 0.0)):
        return wet
    
    if d_in >= bank:
        return 0.0
    
    return min(wet, (bank - d_in) / width)  # the bank divides the segment


@njit(cache=True, error_model="numpy")
def _add_segment(sums, i, in_channel, width, rise, length, wet, depth_in, depth_out, n_15):
    """Add a segment, in_channel of it in the channel (_in_channel), to _profile_side's eight sums, sums[i], as
    _profile_side does."""
    if in_channel == wet:
        sums[i, 0] += width * wet * (depth_in + depth_out) / 2
        sums[i, 1] += wet * length
        sums[i, 3] += wet * length * n_15
        sums[i, 2] += width * wet
    elif in_channel == 0.0:
        sums[i, 4] += width * wet * (depth_in + depth_out) / 2
        sums[i, 5] += wet * length
        sums[i, 7] += wet * length * n_15
        sums[i, 6] += width * wet
    else:
        depth_bank = depth_in - in_channel * rise
        outer = wet - in_channel
        sums[i, 0] += width * in_channel * (depth_in + depth_bank) / 2
        sums[i, 1] += in_channel * length
        sums[i, 3] += in_channel * length * n_15
        sums[i, 2] += width * in_channel
        sums[i, 4] += width * outer * (depth_bank + depth_out) / 2
        sums[i, 5] += outer * length
        sums[i, 7] += outer * length * n_15
        sums[i, 6] += width * outer


@njit(cache=True, error_model="numpy")
def _profile_parts(stations, elevations, n0, center, left_bank, right_bank, wse, left, right, k_decay, shallow_factor,
                   deep_factor):
    lo = _profile_side(stations, elevations, n0, center, -1, wse, left_bank, left, k_decay, shallow_factor,
                       deep_factor)
    hi = _profile_side(stations, elevations, n0, center, 1, wse, right_bank, right, k_decay, shallow_factor,
                       deep_factor)
    channel = (lo[0] + hi[0], lo[1] + hi[1], lo[2] + hi[2], lo[3] + hi[3])
    return (lo[4], lo[5], lo[6], lo[7]), channel, (hi[4], hi[5], hi[6], hi[7])


@njit(cache=True, error_model="numpy")
def profile_compound_geometry(stations, elevations, mannings_n, center, left_bank, right_bank, wse, k_decay,
                              shallow_factor, deep_factor):
    """The left overbank's, the channel's and the right overbank's area, wetted perimeter, top width and
    sum(P_i * n_i**1.5) at a water surface elevation (see the notes above). All zero when the water is at or below
    the stream cell."""
    if not wse > elevations[center]:
        dry = (0.0, 0.0, 0.0, 0.0)
        return dry, dry, dry
    
    return _profile_parts(stations, elevations, mannings_n, center, left_bank, right_bank, wse, -1, -1, k_decay,
                          shallow_factor, deep_factor)


@njit(cache=True, error_model="numpy")
def _parts_conveyance(parts):
    left, channel, right = parts
    return _conveyance(left[0], left[3]) + _conveyance(channel[0], channel[3]) + _conveyance(right[0], right[3])


@njit(cache=True, error_model="numpy")
def profile_conveyance(stations, elevations, mannings_n, center, left_bank, right_bank, wse, k_decay, shallow_factor,
                       deep_factor):
    """A profile's conveyance at a water surface elevation: the sum of its subsections'."""
    return _parts_conveyance(profile_compound_geometry(stations, elevations, mannings_n, center, left_bank, right_bank,
                                                       wse, k_decay, shallow_factor, deep_factor))


@njit(cache=True, error_model="numpy")
def _profile_interval_conveyance(stations, elevations, n0, center, left_bank, right_bank, wse, left, right, k_decay,
                                 shallow_factor, deep_factor):
    return _parts_conveyance(_profile_parts(stations, elevations, n0, center, left_bank, right_bank, wse, left, right,
                                            k_decay, shallow_factor, deep_factor))


@njit(cache=True, error_model="numpy")
def _profile_solve_interval(stations, elevations, n0, center, left_bank, right_bank, left, right, k_decay,
                            shallow_factor, deep_factor, lo, hi, k_lo, k_hi, target):
    """The water surface elevation between lo and hi where the conveyance reaches target, given that it is below
    target at lo and not below it at hi, with the same vertices wet throughout. "Regula falsi" on the conveyance, with
    the Illinois method's halving to keep it moving, bisecting whenever that would leave the bracket."""
    f_lo, f_hi = k_lo - target, k_hi - target
    side = 0
    for _ in range(200):
        if hi - lo <= 1e-12 * (1.0 + abs(hi)):
            break

        u = hi - f_hi * (hi - lo) / (f_hi - f_lo)
        if not lo < u < hi:
            u = 0.5 * (lo + hi)

        f = _profile_interval_conveyance(stations, elevations, n0, center, left_bank, right_bank, u, left, right,
                                         k_decay, shallow_factor, deep_factor) - target
        if f == 0.0:
            return u
        
        if f > 0.0:
            hi, f_hi = u, f
            if side == 1:
                f_lo *= 0.5
            side = 1
        else:
            lo, f_lo = u, f
            if side == -1:
                f_hi *= 0.5
            side = -1

    return hi


@njit(cache=True, error_model="numpy")
def wse_for_profile_conveyance(stations, elevations, mannings_n, center, left_bank, right_bank, target, k_decay,
                               shallow_factor, deep_factor, max_wse=np.inf):
    """The lowest water surface elevation whose conveyance reaches target.

    It raises the water surface one interval at a time, from the stream cell to the next vertex on either side that
    stops the water, and solves within the first interval whose conveyance reaches the target. Where conveyance jumps
    past the target (water spilling over a high point into low ground beyond it), this returns the elevation of that
    high point. The stream cell's elevation for a target of zero or less, and NaN if the target isn't reached at or
    below max_wse, or ever.
    """
    n = elevations.size
    level = float(elevations[center])
    if target <= 0.0:
        return level

    left = right = center
    k_top = np.nan  # the conveyance at the top of the last interval, which is this level
    while True:
        # Once the water is above this level, each side's edge moves past any ground no higher than it
        old_left, old_right = left, right
        while right < n - 1 and elevations[right + 1] <= level:
            right += 1
        while left > 0 and elevations[left - 1] <= level:
            left -= 1

        if level > max_wse:
            return np.nan

        if k_top == k_top and right - old_right <= 1 and old_left - left <= 1:  # noqa: PLR0124
            # Each edge moved past the vertex at this level at most, onto ground above it: the water is the last
            # interval's at its top, and so is its conveyance, bit for bit
            k_start = k_top
        else:
            k_start = _profile_interval_conveyance(stations, elevations, mannings_n, center, left_bank, right_bank,
                                                   level, left, right, k_decay, shallow_factor, deep_factor)
            
        if k_start >= target:
            return level
        
        next_right = elevations[right + 1] if right < n - 1 else np.inf
        next_left = elevations[left - 1] if left > 0 else np.inf
        end = min(next_right, next_left)
        if end == np.inf:
            # Both sides are at their walls, and conveyance grows without bound with the depth
            height = 1.0
            top = level + height
            k_top = _profile_interval_conveyance(stations, elevations, mannings_n, center, left_bank, right_bank, top,
                                                 left, right, k_decay, shallow_factor, deep_factor)
            if not k_top > k_start:
                return np.nan  # no width between the walls, so this section never carries any water
            
            while k_top < target and top < max_wse:
                height *= 2.0
                top = level + height
                k_top = _profile_interval_conveyance(stations, elevations, mannings_n, center, left_bank, right_bank,
                                                     top, left, right, k_decay, shallow_factor, deep_factor)
        else:
            top = end
            k_top = _profile_interval_conveyance(stations, elevations, mannings_n, center, left_bank, right_bank, top,
                                                 left, right, k_decay, shallow_factor, deep_factor)
            
        if k_top >= target:
            answer = _profile_solve_interval(stations, elevations, mannings_n, center, left_bank, right_bank, left,
                                             right, k_decay, shallow_factor, deep_factor, level, top, k_start, k_top,
                                             target)
            return answer if answer <= max_wse else np.nan
        
        if end == np.inf or end > max_wse:
            return np.nan
        
        level = end


# --- A cross section's hydraulics -----------------------------------------------------------------------------------


def hydraulic_geometry(xs: XSection, *, wse: float | None = None, depth: float | None = None,
                       roughness: DepthRoughness | None = None) -> HydraulicGeometry:
    """The whole cross section's wetted geometry at a water surface elevation, or at a depth above the stream cell.
    Zero when dry.

    The banks don't change the geometry (see compound_geometry for each subsection's). With roughness, Manning's n
    varies with the depth, which changes the composite n and nothing else.
    """
    profile = hydraulic_profile(xs)
    wse = _resolve_wse(profile, wse, depth)
    return _to_hydraulic_geometry(*profile_compound_geometry(*profile, -1.0, -1.0, wse, *_parameters(roughness))[1])


def compound_geometry(xs: XSection, *, wse: float | None = None, depth: float | None = None,
                      roughness: DepthRoughness | None = None) -> CompoundGeometry:
    """The wetted geometry of the left overbank, the channel and the right overbank. Without banks, it's all
    channel."""
    profile = hydraulic_profile(xs)
    wse = _resolve_wse(profile, wse, depth)
    return CompoundGeometry(*(_to_hydraulic_geometry(*part) for part in profile_compound_geometry(
        *profile, *_bank_distances(xs), wse, *_parameters(roughness))))


def discharge(xs: XSection, slope: float, *, wse: float | None = None, depth: float | None = None,
              roughness: DepthRoughness | None = None, slope_factor: float = 1.0) -> float:
    """Manning's discharge in m^3/s at a water surface elevation, or at a depth above the stream cell.

    With banks, this is the sum of the subsections' discharges, which is not Manning's equation applied to the whole
    section's hydraulic_geometry. With roughness, Manning's n varies with the depth. `slope_factor` multiplies the
    square root of the slope, and so the discharge.
    """
    profile = hydraulic_profile(xs)
    wse = _resolve_wse(profile, wse, depth)
    k = profile_conveyance(*profile, *_bank_distances(xs), wse, *_parameters(roughness))
    return k * _sqrt_slope(slope) * _check_slope_factor(slope_factor)


def wse_for_discharge(xs: XSection, q: float, slope: float, *, roughness: DepthRoughness | None = None,
                      slope_factor: float = 1.0) -> float:
    """The lowest water surface elevation at which the cross section carries q m^3/s, or NaN if it never does.

    Discharge doesn't always rise with the water level: when water spreads onto flat ground, the wetted perimeter can
    grow faster than the area. Taking the lowest match gives the level that a rising water surface reaches first.
    Where the discharge jumps past q, as water spills over a high point into low ground beyond it, this is the high
    point's elevation. `roughness` and `slope_factor` are as for discharge.
    """
    target = q / (_sqrt_slope(slope) * _check_slope_factor(slope_factor))
    return float(wse_for_profile_conveyance(*hydraulic_profile(xs), *_bank_distances(xs), target,
                                            *_parameters(roughness)))


def _bank_distances(xs: XSection) -> tuple[float, float]:
    return float(xs.left_bank_distance), float(xs.right_bank_distance)


_CONSTANT_N = (1.0, 1.0, 1.0)  # depth-varying roughness parameters that leave n as it is


def _parameters(roughness: DepthRoughness | None) -> tuple[float, float, float]:
    """The profile functions' roughness parameters: roughness's, checked, or ones that leave n as it is."""
    return _CONSTANT_N if roughness is None else _check_roughness(roughness)


def _to_hydraulic_geometry(area: float, perimeter: float, top_width: float, weighted_perimeter: float) -> HydraulicGeometry:
    if perimeter > 0.0:
        return HydraulicGeometry(area, perimeter, area / perimeter, top_width, (weighted_perimeter / perimeter) ** (2 / 3))
    if perimeter == 0.0:
        return HydraulicGeometry(0.0, 0.0, 0.0, 0.0, 0.0)  # dry
    
    return HydraulicGeometry(math.nan, math.nan, math.nan, math.nan, math.nan)  # NaN in the ground


def _resolve_wse(profile: Profile, wse: float | None, depth: float | None) -> float:
    if (wse is None) == (depth is None):
        raise TypeError("Give exactly one of wse or depth.")
    return float(wse) if depth is None else float(profile.elevations[profile.center]) + float(depth)


def _sqrt_slope(slope: float) -> float:
    if not slope > 0.0:
        raise ValueError(f"slope must be positive, not {slope!r}.")
    return math.sqrt(slope)


def _check_slope_factor(slope_factor: float) -> float:
    slope_factor = float(slope_factor)
    if not (math.isfinite(slope_factor) and slope_factor > 0.0):
        raise ValueError(f"slope_factor must be finite and positive, not {slope_factor!r}.")
    return slope_factor


def _check_roughness(roughness: DepthRoughness) -> tuple[float, float, float]:
    """The parameters as floats, checked."""
    k_decay, shallow_factor, deep_factor = (float(v) for v in roughness)
    if not (math.isfinite(shallow_factor) and shallow_factor >= 1.0):
        raise ValueError(f"shallow_factor must be finite and >= 1, not {shallow_factor!r}.")
    if not (math.isfinite(deep_factor) and 0.0 < deep_factor <= 1.0):
        raise ValueError(f"deep_factor must be finite and in (0, 1], not {deep_factor!r}.")
    if not (math.isfinite(k_decay) and k_decay > 0.0):
        raise ValueError(f"k_decay must be finite and > 0, not {k_decay!r}.")
    return k_decay, shallow_factor, deep_factor
