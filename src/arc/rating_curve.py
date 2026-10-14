"""A cross section's rating curve: the flow increments written to the VDT database.

The water surface that carries a stream cell's maximum flow is the lowest at which the discharge reaches it
(arc.hydraulics.wse_for_discharge). The rating curve is then taken at equal steps of depth from the stream cell up to
that water surface (VDT_Database_NumIterations steps): at each, the discharge, the velocity (discharge over area), the
top width, the water surface elevation and the wetted perimeter. The hydraulics are the cross section's own, a carved
channel's exact shape included, divided at its banks, with Manning's n varying with depth if asked (see
arc.hydraulics).

Some cross sections need care:

- Where discharge falls as the water rises, as it can where water spreads onto flat ground, the increment looks up a
  centimetre at a time, as far as the next increment, for a water surface carrying more than the last increment. If
  there isn't one, the last increment is repeated, so the discharges never fall.
- Where water spills over a bank into lower ground, the discharge jumps, and the lowest water surface carrying the
  maximum flow is the bank's top, where the channel carries less. A cross section only gets a rating curve if the
  discharge at its top is within half of the maximum flow either way, and no increment may carry more than 1% more
  than that discharge.
- With a baseflow above 0.001 m^3/s, the first increment's discharge is set just below the baseflow if it was at or
  above it.
"""
from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np
from numba import njit

from arc.hydraulics import (DepthRoughness, _check_slope_factor, _conveyance, _parameters, hydraulic_profile,
                            profile_compound_geometry, wse_for_discharge)
from arc.xsection.xsection import XSection

FIX_UP_STEP = 0.01  # metres: how far at a time to look above an increment whose discharge fell
DISCHARGE_CAP = 1.01  # an increment's discharge may exceed the discharge at the top by 1%
ACCEPTABLE = (0.5, 1.5)  # the discharge at the top, as a share of the maximum flow, that gives a rating curve
BASEFLOW_MARGIN = 0.001  # the first increment's discharge is set this far below the baseflow
Q, V, T, WSE, P = range(5)


class RatingCurve(NamedTuple):
    """A cross section's flow increments (see the notes above). increments is (count, 5): discharge, velocity, top
    width, water surface elevation and wetted perimeter (q, v, t, wse, p)."""
    increments: np.ndarray
    max_wse: float  # the lowest water surface carrying the maximum flow
    start: int  # the last increment, counting from 1, whose geometry wasn't valid, or 0
    last: int  # the last increment, counting from 1, with its own values, or 0

    @property
    def valid(self) -> bool:
        """Whether the increments go in the VDT database: at least one has its own values."""
        return self.last > max(self.start, 0)


@njit(cache=True, error_model="numpy")
def section_hydraulics(stations, elevations, mannings_n, center, left_bank, right_bank, wse, k_decay, shallow_factor,
                       deep_factor):
    """Area, wetted perimeter, top width and conveyance of a cross section's profile at a water surface elevation,
    and the channel's area and conveyance. The conveyance is the sum of the subsections' where there are banks, and
    all channel where not."""
    left, channel, right = profile_compound_geometry(stations, elevations, mannings_n, center, left_bank, right_bank,
                                                     wse, k_decay, shallow_factor, deep_factor)
    area = left[0] + channel[0] + right[0]
    perimeter = left[1] + channel[1] + right[1]
    top_width = left[2] + channel[2] + right[2]
    channel_k = _conveyance(channel[0], channel[3])
    k = _conveyance(left[0], left[3]) + channel_k + _conveyance(right[0], right[3])
    return area, perimeter, top_width, k, channel[0], channel_k


@njit(cache=True, error_model="numpy")
def _increments(stations, elevations, mannings_n, center, left_bank, right_bank, k_decay, shallow_factor, deep_factor,
                thalweg, max_wse, count, discharge_scale, q_top, out):
    """The flow increments from the thalweg up to max_wse (see the notes above): fills out (count, 5) and returns the
    last increment whose geometry wasn't valid and the last with its own values, counting from 1. q_top is the
    discharge at max_wse."""
    step = (max_wse - thalweg) / count
    start = 0
    last = 0
    prev_t = prev_a = prev_p = prev_q = prev_v = prev_wse = 0.0
    for i in range(1, count + 1):
        wse = thalweg + step * i
        area, perimeter, top_width, k, _, _ = section_hydraulics(stations, elevations, mannings_n, center, left_bank,
                                                                 right_bank, wse, k_decay, shallow_factor, deep_factor)
        row = out[i - 1]
        if top_width > 0.0 and area > 0.0 and perimeter > 0.0:
            q = k * discharge_scale
            v = q / area
            if q < prev_q:
                # Look a centimetre at a time, up to the next increment, for a discharge above the last one
                candidate = wse + FIX_UP_STEP
                upper = thalweg + step * (i + 1)
                while candidate < upper:
                    area, perimeter, top_width, k, _, _ = section_hydraulics(
                        stations, elevations, mannings_n, center, left_bank, right_bank, candidate, k_decay,
                        shallow_factor, deep_factor)
                    q_candidate = k * discharge_scale
                    if area > prev_a and perimeter > prev_p and q_candidate > prev_q and q_candidate <= q_top:
                        wse = candidate
                        q = q_candidate
                        v = q_candidate / area
                        break
                    candidate += FIX_UP_STEP
            if q <= prev_q or q > q_top * DISCHARGE_CAP:
                # Repeat the last increment
                row[Q], row[V], row[T], row[WSE], row[P] = prev_q, prev_v, prev_t, prev_wse, prev_p
                continue
            row[Q], row[V], row[T], row[WSE], row[P] = q, v, top_width, wse, perimeter
            prev_t, prev_a, prev_p, prev_q, prev_v, prev_wse = top_width, area, perimeter, q, v, wse
            last = i
        else:
            start = i
            row[Q] = row[V] = row[T] = row[WSE] = row[P] = 0.0
    return start, last


def _banks(xs: XSection) -> tuple[float, float]:
    return float(xs.left_bank_distance), float(xs.right_bank_distance)


def max_flow_wse(xs: XSection, q_max: float, slope: float, *, roughness: DepthRoughness | None = None,
                 slope_factor: float = 1.0) -> float:
    """The lowest water surface elevation at which the cross section carries q_max (arc.hydraulics.wse_for_discharge),
    or NaN if it never does or the slope isn't above 0."""
    if not slope > 0.0:
        return math.nan
    return wse_for_discharge(xs, q_max, slope, roughness=roughness, slope_factor=slope_factor)


def rating_curve(xs: XSection, q_max: float, slope: float, increments: int, *, baseflow: float = 0.0,
                 roughness: DepthRoughness | None = None, slope_factor: float = 1.0) -> RatingCurve | None:
    """A cross section's rating curve up to its maximum flow (see the notes above), or None if it never carries the
    maximum flow, or the discharge at the top is more than half of it away. A maximum flow that isn't above 0, or no
    increments, give a curve without any (start -1 and last 0).

    increments is VDT_Database_NumIterations.
    """
    count = max(int(increments), 0)
    out = np.full((count, 5), np.nan)
    if not q_max > 0.0:
        return RatingCurve(out, math.nan, -1, 0)
    wse = max_flow_wse(xs, q_max, slope, roughness=roughness, slope_factor=slope_factor)
    if not math.isfinite(wse):
        return None
    profile = hydraulic_profile(xs)
    left, right = _banks(xs)
    parameters = _parameters(roughness)
    discharge_scale = math.sqrt(slope) * _check_slope_factor(slope_factor)
    q_top = section_hydraulics(*profile, left, right, wse, *parameters)[3] * discharge_scale
    if not ACCEPTABLE[0] * q_max <= q_top <= ACCEPTABLE[1] * q_max:
        return None
    if count == 0:
        return RatingCurve(out, wse, -1, 0)
    thalweg = float(profile.elevations[profile.center])
    start, last = _increments(*profile, left, right, *parameters, thalweg, wse, count, discharge_scale,
                              float(q_top), out)
    if last > start and baseflow > BASEFLOW_MARGIN and out[start, Q] >= baseflow:
        out[start, Q] = baseflow - BASEFLOW_MARGIN
    return RatingCurve(out, wse, int(start), int(last))
