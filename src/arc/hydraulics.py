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
water. Factors of 1 leave n as it is. Each segment takes its mean depth, the mean of the depths at the two ends of its
wet part: where the water meets the ground part way along it, half the depth at its inner end, and an end wall's,
half the depth at its foot. Where a bank divides a segment, each part takes its own. The depth at one end would
make n depend on the spacing: the segment at the water's edge would take the depth a whole spacing in from the edge.

Flow-through threads (XSection.threads, arc.xsection.threads): where a bar splits the river, the threads the stream
cell isn't in carry water at the same water surface below the bar's top, joined to the stream above and below the
cross section. The water spreads from every vertex of a thread as it does from the stream cell. Each vertex then has a
connection level (connection_levels): the least, over the stream cell's vertex and the threads' vertices, of the
highest ground between that vertex and this one. A segment holds water where its water surface is above the
connection level of either end, the part of it below the water surface: the stream cell's spreading alone gives the
rule above. Beyond the outermost spreading vertex on a side the connection levels never fall going out, so a walk out
along the side stops at the first vertex there whose level the water isn't above, as the walk to the water's edge
does. A cross section without threads takes the code paths that don't look at connection levels at all.
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

    # Spliced once for the same valley, ground and banks: the cache holds them, so none is a new object in their place.
    # Checked a field at a time, without a generator: it's asked for millions of times, nearly always found
    left, right = xs.left_bank_distance, xs.right_bank_distance
    cached = xs._spliced
    if cached is not None:
        kept = cached[0]
        if kept[0] is valley and kept[1] is xs.profile and kept[2] is xs.elevations and kept[3] is xs.mannings_n \
                and (kept[4] is left or (isinstance(kept[4], float) and kept[4] == left)) \
                and (kept[5] is right or (isinstance(kept[5], float) and kept[5] == right)):
            return cached[1]

    sources = (valley, xs.profile, xs.elevations, xs.mannings_n, left, right)
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
            n_15 = _scaled_n_15(n0[end], 0.5 * depth_end, k_decay, shallow_factor, deep_factor)
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

        if not divided or d_out < bank or (d_out == bank and (width > 0.0 or rise > 0.0)):
            in_channel = wet
        elif d_in >= bank:
            in_channel = 0.0
        else:
            in_channel = min(wet, (bank - d_in) / width)  # the bank divides the segment

        if in_channel == wet:
            n_15 = _scaled_n_15(n0[j], 0.5 * (depth_in + depth_out), k_decay, shallow_factor, deep_factor)
            channel_area += width * wet * (depth_in + depth_out) / 2
            channel_perimeter += wet * length
            channel_weighted += wet * length * n_15
            channel_top_width += width * wet
        elif in_channel == 0.0:
            n_15 = _scaled_n_15(n0[j], 0.5 * (depth_in + depth_out), k_decay, shallow_factor, deep_factor)
            overbank_area += width * wet * (depth_in + depth_out) / 2
            overbank_perimeter += wet * length
            overbank_weighted += wet * length * n_15
            overbank_top_width += width * wet
        else:
            depth_bank = depth_in - in_channel * rise
            outer = wet - in_channel
            n_15 = _scaled_n_15(n0[j], 0.5 * (depth_in + depth_bank), k_decay, shallow_factor, deep_factor)
            channel_area += width * in_channel * (depth_in + depth_bank) / 2
            channel_perimeter += in_channel * length
            channel_weighted += in_channel * length * n_15
            channel_top_width += width * in_channel
            n_15 = _scaled_n_15(n0[j], 0.5 * (depth_bank + depth_out), k_decay, shallow_factor, deep_factor)
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
                   out, connect=None):
    """_profile_side, walking out to the water's edge, at each water surface levels[first:], which ascend: out[i] gets
    its eight sums at levels[i], each summed in the order _profile_side sums it, so the same numbers. Each segment's
    geometry is worked out once for all the water surfaces over it, which takes half the time of one at a time.
    With connection levels (connect, not empty), _side_connected's sums instead."""
    if connect is not None:  # noqa: SIM102 (apart, so that compiling without connect drops the branch)
        if connect.size > 0:
            _sides_connected(stations, elevations, n0, center, step, levels, first, bank, connect, k_decay,
                             shallow_factor, deep_factor, out)
            return

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
                n_15 = _scaled_n_15(n0[end], 0.5 * depth_end, k_decay, shallow_factor, deep_factor)
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
            _add_segment(out, i, _in_channel(divided, bank, d_in, d_out, width, rise, wet), width, rise, length, wet,
                         depth_in, 0.0, n0[j], k_decay, shallow_factor, deep_factor)

        if split < m:
            # Those over it all the way, wet 1: how much of it is in the channel is the same for all of them, so each
            # way of dividing it has a loop of its own
            wet = 1.0
            in_channel = _in_channel(divided, bank, d_in, d_out, width, rise, wet)
            n_15_fixed = _scaled_n_15(n0[j], 0.0, k_decay, shallow_factor, deep_factor)
            if in_channel == wet:
                for i in range(split, m):
                    depth_in, depth_out = levels[i] - z_in, levels[i] - z_out
                    depth = 0.5 * (depth_in + depth_out)
                    n_15 = n_15_fixed if constant_n and depth == depth else \
                        _scaled_n_15(n0[j], depth, k_decay, shallow_factor, deep_factor)  # noqa: PLR0124
                    out[i, 0] += width * wet * (depth_in + depth_out) / 2
                    out[i, 1] += wet * length
                    out[i, 3] += wet * length * n_15
                    out[i, 2] += width * wet
            elif in_channel == 0.0:
                for i in range(split, m):
                    depth_in, depth_out = levels[i] - z_in, levels[i] - z_out
                    depth = 0.5 * (depth_in + depth_out)
                    n_15 = n_15_fixed if constant_n and depth == depth else \
                        _scaled_n_15(n0[j], depth, k_decay, shallow_factor, deep_factor)  # noqa: PLR0124
                    out[i, 4] += width * wet * (depth_in + depth_out) / 2
                    out[i, 5] += wet * length
                    out[i, 7] += wet * length * n_15
                    out[i, 6] += width * wet
            else:
                for i in range(split, m):
                    depth_in, depth_out = levels[i] - z_in, levels[i] - z_out
                    _add_segment(out, i, in_channel, width, rise, length, wet, depth_in, depth_out, n0[j], k_decay,
                                 shallow_factor, deep_factor)

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
def _add_segment(sums, i, in_channel, width, rise, length, wet, depth_in, depth_out, n0, k_decay, shallow_factor,
                 deep_factor):
    """Add a segment, in_channel of it in the channel (_in_channel), with Manning's n0 before scaling for depth, to
    _profile_side's eight sums, sums[i], as _profile_side does."""
    if in_channel == wet:
        n_15 = _scaled_n_15(n0, 0.5 * (depth_in + depth_out), k_decay, shallow_factor, deep_factor)
        sums[i, 0] += width * wet * (depth_in + depth_out) / 2
        sums[i, 1] += wet * length
        sums[i, 3] += wet * length * n_15
        sums[i, 2] += width * wet
    elif in_channel == 0.0:
        n_15 = _scaled_n_15(n0, 0.5 * (depth_in + depth_out), k_decay, shallow_factor, deep_factor)
        sums[i, 4] += width * wet * (depth_in + depth_out) / 2
        sums[i, 5] += wet * length
        sums[i, 7] += wet * length * n_15
        sums[i, 6] += width * wet
    else:
        depth_bank = depth_in - in_channel * rise
        outer = wet - in_channel
        n_15 = _scaled_n_15(n0, 0.5 * (depth_in + depth_bank), k_decay, shallow_factor, deep_factor)
        sums[i, 0] += width * in_channel * (depth_in + depth_bank) / 2
        sums[i, 1] += in_channel * length
        sums[i, 3] += in_channel * length * n_15
        sums[i, 2] += width * in_channel
        n_15 = _scaled_n_15(n0, 0.5 * (depth_bank + depth_out), k_decay, shallow_factor, deep_factor)
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
                              shallow_factor, deep_factor, connect=None):
    """The left overbank's, the channel's and the right overbank's area, wetted perimeter, top width and
    sum(P_i * n_i**1.5) at a water surface elevation (see the notes above). All zero when the water is at or below
    the stream cell. connect: the profile's connection levels (connection_levels) where it has threads, or None."""
    if not wse > elevations[center]:
        dry = (0.0, 0.0, 0.0, 0.0)
        return dry, dry, dry

    if connect is not None:  # noqa: SIM102 (apart, so that compiling without connect drops the branch)
        if connect.size > 0:
            return _connected_parts(stations, elevations, mannings_n, center, left_bank, right_bank, wse, connect,
                                    False, k_decay, shallow_factor, deep_factor)

    return _profile_parts(stations, elevations, mannings_n, center, left_bank, right_bank, wse, -1, -1, k_decay,
                          shallow_factor, deep_factor)


@njit(cache=True, error_model="numpy")
def _parts_conveyance(parts):
    left, channel, right = parts
    return _conveyance(left[0], left[3]) + _conveyance(channel[0], channel[3]) + _conveyance(right[0], right[3])


@njit(cache=True, error_model="numpy")
def profile_conveyance(stations, elevations, mannings_n, center, left_bank, right_bank, wse, k_decay, shallow_factor,
                       deep_factor, connect=None):
    """A profile's conveyance at a water surface elevation: the sum of its subsections'."""
    return _parts_conveyance(profile_compound_geometry(stations, elevations, mannings_n, center, left_bank, right_bank,
                                                       wse, k_decay, shallow_factor, deep_factor, connect))


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
                               shallow_factor, deep_factor, max_wse=np.inf, connect=None):
    """The lowest water surface elevation whose conveyance reaches target.

    It raises the water surface one interval at a time, from the stream cell to the next vertex on either side that
    stops the water, and solves within the first interval whose conveyance reaches the target. Where conveyance jumps
    past the target (water spilling over a high point into low ground beyond it), this returns the elevation of that
    high point. The stream cell's elevation for a target of zero or less, and NaN if the target isn't reached at or
    below max_wse, or ever. With connection levels (connect, not empty), the intervals run between them instead
    (_connected_wse).
    """
    if connect is not None:  # noqa: SIM102 (apart, so that compiling without connect drops the branch)
        if connect.size > 0:
            return _connected_wse(stations, elevations, mannings_n, center, left_bank, right_bank, target, connect,
                                  k_decay, shallow_factor, deep_factor, max_wse)

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


# --- Flow-through threads (see the notes above) ---------------------------------------------------------------------


@njit(cache=True, error_model="numpy")
def connection_levels(stations, elevations, center, threads):
    """Each vertex's connection level: the lowest water surface above which the water reaches it, spreading along the
    profile from the stream cell's vertex or from any vertex of a thread, the least over those of the highest ground
    between that one and this, both included. threads holds a row for each thread, the stations of its two edges;
    every vertex from one to the other spreads water. Without threads it's the highest ground out from the stream
    cell, the rule of the notes above. Two more values follow the vertices' levels: the indices of the outermost
    spreading vertices, the leftmost and the rightmost, beyond which the levels never fall going out."""
    n = elevations.size
    seeds = np.zeros(n, np.bool_)
    seeds[center] = True
    for t in range(threads.shape[0]):
        for i in range(np.searchsorted(stations, threads[t, 0], side="left"),
                       np.searchsorted(stations, threads[t, 1], side="right")):
            seeds[i] = True

    # The nearest spreading vertex on either side has the lowest ground between: a farther one's way passes it
    out = np.empty(n + 2)
    leftmost, rightmost = center, center
    for i in range(n):
        if seeds[i]:
            leftmost = min(leftmost, i)
            rightmost = max(rightmost, i)
    out[n], out[n + 1] = leftmost, rightmost
    level = np.inf
    for i in range(n):
        level = elevations[i] if seeds[i] else max(level, elevations[i])
        out[i] = level

    level = np.inf
    for i in range(n - 1, -1, -1):
        level = elevations[i] if seeds[i] else max(level, elevations[i])
        out[i] = min(out[i], level)

    return out


@njit(cache=True, error_model="numpy")
def _segment_sums(wse, z_in, z_out, d_in, d_out, width, rise, length, divided, bank, n0, n_15_fixed, constant_n,
                  k_decay, shallow_factor, deep_factor):
    """One segment's shares of _profile_side's eight sums at wse, the part of it below wse: from its inner end, as
    _profile_side and _add_segment take it, or where the ground falls outward to the water, from its outer end.
    n_15_fixed is the segment's n**1.5 where n doesn't vary with the depth (constant_n)."""
    depth_in, depth_out = wse - z_in, wse - z_out
    if depth_in > 0.0:
        if depth_out > 0.0:
            wet = 1.0
        else:
            wet = depth_in / rise  # the water meets the ground part way along the segment
            depth_out = 0.0

        in_channel = _in_channel(divided, bank, d_in, d_out, width, rise, wet)
        if in_channel == wet:
            depth = 0.5 * (depth_in + depth_out)
            n_15 = n_15_fixed if constant_n and depth == depth else \
                _scaled_n_15(n0, depth, k_decay, shallow_factor, deep_factor)  # noqa: PLR0124
            return (width * wet * (depth_in + depth_out) / 2, wet * length, width * wet, wet * length * n_15,
                    0.0, 0.0, 0.0, 0.0)
        if in_channel == 0.0:
            depth = 0.5 * (depth_in + depth_out)
            n_15 = n_15_fixed if constant_n and depth == depth else \
                _scaled_n_15(n0, depth, k_decay, shallow_factor, deep_factor)  # noqa: PLR0124
            return (0.0, 0.0, 0.0, 0.0,
                    width * wet * (depth_in + depth_out) / 2, wet * length, width * wet, wet * length * n_15)

        depth_bank = depth_in - in_channel * rise
        outer = wet - in_channel
        n_15 = _scaled_n_15(n0, 0.5 * (depth_in + depth_bank), k_decay, shallow_factor, deep_factor)
        n_15_outer = _scaled_n_15(n0, 0.5 * (depth_bank + depth_out), k_decay, shallow_factor, deep_factor)
        return (width * in_channel * (depth_in + depth_bank) / 2, in_channel * length, width * in_channel,
                in_channel * length * n_15,
                width * outer * (depth_bank + depth_out) / 2, outer * length, width * outer,
                outer * length * n_15_outer)

    if not depth_out > 0.0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    # The ground falls outward to the water: wet from a of the way out to the outer end, where it's depth_out deep
    a = 1.0 + depth_out / rise
    if not divided or d_out < bank or (d_out == bank and (width > 0.0 or rise > 0.0)):
        split = 1.0
    elif d_in >= bank:
        split = 0.0
    else:
        split = (bank - d_in) / width  # the channel's share, out from the inner end

    channel = overbank = (0.0, 0.0, 0.0, 0.0)
    if split > a:
        channel = _wet_part(a, min(1.0, split), width, rise, length, depth_in, n0, n_15_fixed, constant_n, k_decay,
                            shallow_factor, deep_factor)
    if split < 1.0:
        overbank = _wet_part(max(a, split), 1.0, width, rise, length, depth_in, n0, n_15_fixed, constant_n, k_decay,
                             shallow_factor, deep_factor)
    return (channel[0], channel[1], channel[2], channel[3], overbank[0], overbank[1], overbank[2], overbank[3])


@njit(cache=True, error_model="numpy")
def _wet_part(a, b, width, rise, length, depth_in, n0, n_15_fixed, constant_n, k_decay, shallow_factor, deep_factor):
    """The area, wetted perimeter, top width and P * n**1.5 of the part of a segment from a to b of the way out from
    its inner end, under water all along."""
    depth_a = max(depth_in - a * rise, 0.0)
    depth_b = max(depth_in - b * rise, 0.0)
    share = b - a
    depth = 0.5 * (depth_a + depth_b)
    n_15 = n_15_fixed if constant_n and depth == depth else \
        _scaled_n_15(n0, depth, k_decay, shallow_factor, deep_factor)  # noqa: PLR0124
    return width * share * (depth_a + depth_b) / 2, share * length, width * share, share * length * n_15


@njit(cache=True, error_model="numpy")
def _side_connected(stations, elevations, n0, center, step, wse, bank, connect, inclusive, k_decay, shallow_factor,
                    deep_factor):
    """_profile_side's eight sums with connection levels: every segment on the side either end of which is connected
    below wse (at or below it, with inclusive, which gives the water just above wse) holds the part of it below wse,
    which can lie at either end of it (_segment_sums). The wall at the side's end is wet where its foot is
    connected. Beyond the side's outermost spreading vertex, the walk stops at the first vertex not connected below
    wse (see the notes above)."""
    n = elevations.size
    end = n - 1 if step > 0 else 0
    limit = int(connect[n + 1]) if step > 0 else int(connect[n])  # the side's outermost spreading vertex
    origin = stations[center]
    divided = bank >= 0.0 and bank < abs(stations[end] - origin)
    constant_n = shallow_factor == deep_factor
    channel_area = channel_perimeter = channel_top_width = channel_weighted = 0.0
    overbank_area = overbank_perimeter = overbank_top_width = overbank_weighted = 0.0
    j = center
    z_in, d_in = elevations[j], abs(stations[j] - origin)
    while j != end:
        if (j - limit) * step > 0 and (connect[j] > wse or (connect[j] == wse and not inclusive)):
            break  # beyond it, and none farther out holds water

        o = j + step
        z_out, d_out = elevations[o], abs(stations[o] - origin)
        level = min(connect[j], connect[o])
        if (level < wse or (inclusive and level == wse)) and (wse > z_in or wse > z_out):
            width, rise = d_out - d_in, z_out - z_in
            length = math.sqrt(width * width + rise * rise)
            n_15_fixed = _scaled_n_15(n0[j], 0.0, k_decay, shallow_factor, deep_factor) if constant_n else 0.0
            sums = _segment_sums(wse, z_in, z_out, d_in, d_out, width, rise, length, divided, bank, n0[j], n_15_fixed,
                                 constant_n, k_decay, shallow_factor, deep_factor)
            channel_area += sums[0]
            channel_perimeter += sums[1]
            channel_top_width += sums[2]
            channel_weighted += sums[3]
            overbank_area += sums[4]
            overbank_perimeter += sums[5]
            overbank_top_width += sums[6]
            overbank_weighted += sums[7]

        j = o
        z_in, d_in = z_out, d_out

    if connect[end] < wse or (inclusive and connect[end] == wse):
        depth_end = wse - elevations[end]
        if depth_end > 0.0:
            n_15 = _scaled_n_15(n0[end], 0.5 * depth_end, k_decay, shallow_factor, deep_factor)
            if divided:
                overbank_perimeter += depth_end
                overbank_weighted += depth_end * n_15
            else:
                channel_perimeter += depth_end
                channel_weighted += depth_end * n_15

    return (channel_area, channel_perimeter, channel_top_width, channel_weighted,
            overbank_area, overbank_perimeter, overbank_top_width, overbank_weighted)


@njit(cache=True, error_model="numpy")
def _first_above(levels, lo, hi, value):
    """The index of the first of levels[lo:hi], which ascend, above value, or hi if none is."""
    while lo < hi:
        middle = (lo + hi) // 2
        if levels[middle] > value:
            hi = middle
        else:
            lo = middle + 1
    return lo


@njit(cache=True, error_model="numpy")
def _sides_connected(stations, elevations, n0, center, step, levels, first, bank, connect, k_decay, shallow_factor,
                     deep_factor, out):
    """_side_connected at each water surface levels[first:], which ascend, added to out[i] as _profile_sides adds
    its sums: the same numbers, each segment's geometry worked out once for all the levels it holds water at, those
    above its connection level."""
    m = levels.size
    if first >= m:
        return

    n = elevations.size
    end = n - 1 if step > 0 else 0
    limit = int(connect[n + 1]) if step > 0 else int(connect[n])  # the side's outermost spreading vertex
    origin = stations[center]
    divided = bank >= 0.0 and bank < abs(stations[end] - origin)
    constant_n = shallow_factor == deep_factor
    top = levels[m - 1]
    j = center
    z_in, d_in = elevations[j], abs(stations[j] - origin)
    while j != end:
        if (j - limit) * step > 0 and connect[j] >= top:
            break  # beyond it, and none farther out holds water at any of the levels

        o = j + step
        z_out, d_out = elevations[o], abs(stations[o] - origin)
        level = min(connect[j], connect[o])
        if level < top:
            lo = _first_above(levels, first, m, level)  # the first water surface above its connection level

            width, rise = d_out - d_in, z_out - z_in
            length = math.sqrt(width * width + rise * rise)
            n_15_fixed = _scaled_n_15(n0[j], 0.0, k_decay, shallow_factor, deep_factor) if constant_n else 0.0
            # The levels the segment holds water at part of the way (above its lower end), then all of it (above its
            # higher end): ranges of the ascending levels
            partial = _first_above(levels, lo, m, min(z_in, z_out))
            full = _first_above(levels, partial, m, max(z_in, z_out))
            for i in range(partial, full):
                sums = _segment_sums(levels[i], z_in, z_out, d_in, d_out, width, rise, length, divided, bank, n0[j],
                                     n_15_fixed, constant_n, k_decay, shallow_factor, deep_factor)
                for q in range(8):
                    out[i, q] += sums[q]

            if full < m:
                # All of it under water: how much of it is in the channel is the same at every level, so each way of
                # dividing it has a loop of its own, as _profile_sides has, summing as _segment_sums does
                in_channel = _in_channel(divided, bank, d_in, d_out, width, rise, 1.0)
                if in_channel == 1.0 or in_channel == 0.0:
                    q0 = 0 if in_channel == 1.0 else 4
                    for i in range(full, m):
                        depth_in, depth_out = levels[i] - z_in, levels[i] - z_out
                        depth = 0.5 * (depth_in + depth_out)
                        n_15 = n_15_fixed if constant_n and depth == depth else                             _scaled_n_15(n0[j], depth, k_decay, shallow_factor, deep_factor)  # noqa: PLR0124
                        out[i, q0] += width * 1.0 * (depth_in + depth_out) / 2
                        out[i, q0 + 1] += 1.0 * length
                        out[i, q0 + 2] += width * 1.0
                        out[i, q0 + 3] += 1.0 * length * n_15
                else:
                    for i in range(full, m):
                        sums = _segment_sums(levels[i], z_in, z_out, d_in, d_out, width, rise, length, divided, bank,
                                             n0[j], n_15_fixed, constant_n, k_decay, shallow_factor, deep_factor)
                        for q in range(8):
                            out[i, q] += sums[q]

        j = o
        z_in, d_in = z_out, d_out

    for i in range(first, m):
        depth_end = levels[i] - elevations[end]
        if connect[end] < levels[i] and depth_end > 0.0:
            n_15 = _scaled_n_15(n0[end], 0.5 * depth_end, k_decay, shallow_factor, deep_factor)
            if divided:
                out[i, 5] += depth_end
                out[i, 7] += depth_end * n_15
            else:
                out[i, 1] += depth_end
                out[i, 3] += depth_end * n_15


@njit(cache=True, error_model="numpy")
def _connected_parts(stations, elevations, n0, center, left_bank, right_bank, wse, connect, inclusive, k_decay,
                     shallow_factor, deep_factor):
    """_profile_parts with connection levels (_side_connected)."""
    lo = _side_connected(stations, elevations, n0, center, -1, wse, left_bank, connect, inclusive, k_decay,
                         shallow_factor, deep_factor)
    hi = _side_connected(stations, elevations, n0, center, 1, wse, right_bank, connect, inclusive, k_decay,
                         shallow_factor, deep_factor)
    channel = (lo[0] + hi[0], lo[1] + hi[1], lo[2] + hi[2], lo[3] + hi[3])
    return (lo[4], lo[5], lo[6], lo[7]), channel, (hi[4], hi[5], hi[6], hi[7])


@njit(cache=True, error_model="numpy")
def _connected_conveyance(stations, elevations, n0, center, left_bank, right_bank, wse, connect, inclusive, k_decay,
                          shallow_factor, deep_factor):
    return _parts_conveyance(_connected_parts(stations, elevations, n0, center, left_bank, right_bank, wse, connect,
                                              inclusive, k_decay, shallow_factor, deep_factor))


@njit(cache=True, error_model="numpy")
def _connected_solve(stations, elevations, n0, center, left_bank, right_bank, connect, k_decay, shallow_factor,
                     deep_factor, lo, hi, k_lo, k_hi, target):
    """_profile_solve_interval with connection levels, between two of them, where the same segments hold water."""
    f_lo, f_hi = k_lo - target, k_hi - target
    side = 0
    for _ in range(200):
        if hi - lo <= 1e-12 * (1.0 + abs(hi)):
            break

        u = hi - f_hi * (hi - lo) / (f_hi - f_lo)
        if not lo < u < hi:
            u = 0.5 * (lo + hi)

        f = _connected_conveyance(stations, elevations, n0, center, left_bank, right_bank, u, connect, False, k_decay,
                                  shallow_factor, deep_factor) - target
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
def _connected_wse(stations, elevations, n0, center, left_bank, right_bank, target, connect, k_decay, shallow_factor,
                   deep_factor, max_wse):
    """wse_for_profile_conveyance with connection levels. Between two of them the same segments hold water, so the
    conveyance changes smoothly; at one it can jump, as a bar's top goes under and the water joins a thread's. The
    water surface rises from the stream cell one interval at a time: if the conveyance just above an interval's
    bottom (inclusive) reaches the target, that's the answer, as it is where the water spills over a high point;
    if it reaches it at the interval's top, it's solved within."""
    level = float(elevations[center])
    if target <= 0.0:
        return level

    if level > max_wse:
        return np.nan

    k_bottom = _connected_conveyance(stations, elevations, n0, center, left_bank, right_bank, level, connect, True,
                                     k_decay, shallow_factor, deep_factor)
    if k_bottom >= target:
        return level

    levels = np.sort(connect[:elevations.size])
    i = np.searchsorted(levels, level, side="right")
    while True:
        while i < levels.size and not levels[i] > level:
            i += 1

        if i < levels.size:
            top = levels[i]
            k_top = _connected_conveyance(stations, elevations, n0, center, left_bank, right_bank, top, connect, False,
                                          k_decay, shallow_factor, deep_factor)
        else:
            # Past every connection level, and conveyance grows without bound with the depth
            height = 1.0
            top = level + height
            k_top = _connected_conveyance(stations, elevations, n0, center, left_bank, right_bank, top, connect, False,
                                          k_decay, shallow_factor, deep_factor)
            if not k_top > k_bottom:
                return np.nan

            while k_top < target and top < max_wse:
                height *= 2.0
                top = level + height
                k_top = _connected_conveyance(stations, elevations, n0, center, left_bank, right_bank, top, connect,
                                              False, k_decay, shallow_factor, deep_factor)

        if k_top >= target:
            answer = _connected_solve(stations, elevations, n0, center, left_bank, right_bank, connect, k_decay,
                                      shallow_factor, deep_factor, level, top, k_bottom, k_top, target)
            return answer if answer <= max_wse else np.nan

        if i >= levels.size or top > max_wse:
            return np.nan

        level = top
        k_bottom = _connected_conveyance(stations, elevations, n0, center, left_bank, right_bank, level, connect, True,
                                         k_decay, shallow_factor, deep_factor)
        if k_bottom >= target:
            return level


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
    return _to_hydraulic_geometry(*profile_compound_geometry(*profile, -1.0, -1.0, wse, *_parameters(roughness),
                                                             hydraulic_connect(xs, profile))[1])


def compound_geometry(xs: XSection, *, wse: float | None = None, depth: float | None = None,
                      roughness: DepthRoughness | None = None) -> CompoundGeometry:
    """The wetted geometry of the left overbank, the channel and the right overbank. Without banks, it's all
    channel."""
    profile = hydraulic_profile(xs)
    wse = _resolve_wse(profile, wse, depth)
    return CompoundGeometry(*(_to_hydraulic_geometry(*part) for part in profile_compound_geometry(
        *profile, *_bank_distances(xs), wse, *_parameters(roughness), hydraulic_connect(xs, profile))))


def discharge(xs: XSection, slope: float, *, wse: float | None = None, depth: float | None = None,
              roughness: DepthRoughness | None = None, slope_factor: float = 1.0) -> float:
    """Manning's discharge in m^3/s at a water surface elevation, or at a depth above the stream cell.

    With banks, this is the sum of the subsections' discharges, which is not Manning's equation applied to the whole
    section's hydraulic_geometry. With roughness, Manning's n varies with the depth. `slope_factor` multiplies the
    square root of the slope, and so the discharge.
    """
    profile = hydraulic_profile(xs)
    wse = _resolve_wse(profile, wse, depth)
    k = profile_conveyance(*profile, *_bank_distances(xs), wse, *_parameters(roughness),
                           hydraulic_connect(xs, profile))
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
    profile = hydraulic_profile(xs)
    return float(wse_for_profile_conveyance(*profile, *_bank_distances(xs), target, *_parameters(roughness), np.inf,
                                            hydraulic_connect(xs, profile)))


def hydraulic_connect(xs: XSection, profile: Profile | None = None) -> np.ndarray | None:
    """The connection levels of a cross section's hydraulic profile (connection_levels) where it has threads, for
    the compiled functions' `connect`, and None where it has none. profile is hydraulic_profile(xs), if known. Kept
    on the cross section until its profile or threads change."""
    threads = xs.threads
    if threads is None or len(threads) == 0:
        return None

    profile = hydraulic_profile(xs) if profile is None else profile
    cached = xs._connect
    if cached is not None and cached[0] is profile.elevations and cached[1] is threads and \
            cached[2].size == profile.stations.size + 2 and cached[3] == profile.center:
        return cached[2]

    connect = connection_levels(np.asarray(profile.stations, np.float64), np.asarray(profile.elevations, np.float64),
                                int(profile.center), np.asarray(threads, np.float64))
    xs._connect = (profile.elevations, threads, connect, int(profile.center))
    return connect


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
