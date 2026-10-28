"""The standard-step method: water surfaces linked along the stream network (Standard_Step).

Without it, each cross section's rating curve is its own: the water is at normal depth for the cross section's
geometry and slope, whatever the cross sections around it are like. With it, the water surfaces come from the energy
equation between neighbouring cross sections instead, marched upstream from the network's outlets, as in a
one-dimensional hydraulic model (Chow, 1959). Between a cross section u and the one below it, d, L metres apart along
the stream:

    WSE_u + h_u = WSE_d + h_d + L Sf + C |h_u - h_w| / n,    h = a V^2 / 2g

with Sf = ((Q_u + Q_d) / ((K_u + K_d) f))^2, the average-conveyance friction slope (K the conveyance, f the
slope_adjustment_factor), a the velocity-head coefficient of the channel and the overbanks, and C 0.1 where the flow
speeds up going downstream and 0.3 where it slows. The eddy loss (the last term) is reckoned on the change in velocity
head from u to the cross section w, the first EDDY_LENGTH (300 m, Standard_Step_Eddy_Length) or more downstream in the
reach (or its last), n cross sections below u, and each step takes 1/n of it. A steady widening or narrowing is charged
in full that way, but not the cross sections' differences from their neighbours: with cross sections a cell apart, those
come and go over tens of metres, and charging each step's change (n = 1, as HEC-RAS does with cross sections hundreds of
metres apart) added up to half the friction on a big braided river, and raised its water metres above normal depth. The
window has its own cost: a real change in velocity head, as where the water draws down to critical depth above a steep
reach, is charged over the window above it, raising the water there. On Chow's (1959) channel drawn down to critical
depth (testing/test_benchmarks.py), a window of 300 m puts the water up to 0.03 m above the exact profile, as 1 km
does, but 0.002 m above normal depth upstream of the drawdown, where 1 km put it 0.006 m above; on a channel whose width
varies by a fifth from cell to cell, the eddy loss raises the water 0.07 m (1 km: 0.02 m; each step's own: 0.46 m).

The upstream water surface is the subcritical root, the lowest above critical depth and above the cross section's water
at the flow before (so that it never falls as the flow rises, which the lowest root can in a compound channel), or
critical depth where there's none (where the water below is supercritical, or the cross section above needs more energy
even at critical depth).

Where the water speeds up going downstream, the eddy loss (the contraction loss, C 0.1) is reckoned at the velocity
head the water would have without it, the root with the expansion loss alone, and the water then raised from there to
make it up. Reckoned at the water's own velocity head, the loss would grow with the water faster than its energy near
critical depth, where (1 + C / n) Fr^2 > 1: the energy equation's root near the one without the loss then only touches
zero, or isn't there, and the step landed on a root far above it, as if critical depth were at a Froude number of
0.95 (uniform flow at 0.986, n 1, stood 0.02 m above normal depth). Reckoned so, the loss in a steady drawdown is
C (1 - Fr^2) / (1 - (1 - C) Fr^2) of the gain in velocity head: C where the water is slow, falling to nothing at
critical depth, so that the water draws down to the energy equation's own critical depth. The expansion loss falls as
the water rises, so it's the water's own: its root is the lowest above the root without it, and below the water surface
whose velocity head is the far end's.

Each reach's rated cells are put in order along the stream (arc.bathymetry.smoothing.order_reach): by the reaches next
to it in the network, on the stream raster if they have no rating curves, or else by where its line ends, and only
failing those by its stream cells' elevations. A march from the wrong end piles water up from there, and the
elevations, the carved beds above all, are noisy where the water is flat. Their rating curves are then worked out
again at flows common to the reach: Q_j = j/N of each cell's maximum flow, j = 1 to N (VDT_Database_NumIterations),
with the first at or above the baseflow set just below it, so that the baseflow is one of them (rating_curve does the
same when that is the first). A reach starts from the water surface of the first rated cell of the reach below it at
the same j (the same share of that reach's maximum flow), and an outlet from its last cell's normal depth
(arc.hydraulics.wse_for_discharge), both at least critical depth. Where the reaches below have no rated cells (one in
a lake, or one the flow file leaves out), the reach starts as an outlet does, but no higher than the water at the top of the first reach further down that has
them, risen by the reach's median slope over the straight distance between them: normal depth at one cross section, a
choke as like as not on flat water, could otherwise stand the whole reach metres above the river below it. The water
below isn't passed on in full, as it is from a reach next to it, as the gap's fall is only an estimate. Each
increment's velocity, top width and wetted perimeter are then the cell's own cross section's at its water surface.

The baseflow stays in the channel. The bathymetry carves each channel to carry its baseflow, by Manning's equation,
with its water at a level (its bank elevation with Bathy_Use_Banks, its stream cell's otherwise), by the hydraulics
the rating curves use (arc.bathymetry.channel.carved_depth), so normal depth puts the baseflow there. The step
needn't: where a channel was carved for a steeper slope than the levels fall, its water loses more to friction than
the banks drop, and rises over them going upstream. So at every flow up to its baseflow,
a cell whose channel's depth came from its baseflow (not a drainage-area depth) has its water held at most at its
level (baseflow_levels), the energy above that taken as lost there, as over a riffle, and the march goes on upstream
from the held water. A steep section is held there too where its critical depth is higher: supercritical water is below
critical depth. With the baseflow one of the flows, curve2flood reads the held water at it, not a line from the flows
either side of it.

Each cross section has a table of LEVELS water surfaces, from its bed to well above the top of its own rating curve:
it brackets each step's water surface (or, where the water below stands higher than the table reaches, the search goes
on above it), and gives the critical depth, and an outlet's normal depth where wse_for_discharge has none. Each water
surface is then found with the exact hydraulics, so that no interpolation error builds up from one step to the next.
Where the energy jumps across what's needed instead, as where the water tops a high point of a cross section and the
low ground beyond joins it at once (arc.hydraulics), its area, conveyance and velocity head jumping, the water surface
is the high point's level for a range of water below, and it's taken on the side with energy enough, so that the step
above always gets the same hydraulics from it.
"""
from __future__ import annotations

import math
from collections import deque
from collections.abc import Callable
from typing import NamedTuple

import networkx as nx
import numpy as np
from numba import njit

from arc.bathymetry.smoothing import order_reach
from arc.hydraulics import (
    DepthRoughness,
    _check_slope_factor,
    _parameters,
    hydraulic_profile,
    wse_for_discharge,
)
from arc.outputs.rating_curves import RatingCurves
from arc.rating_curve import (
    BASEFLOW_MARGIN,
    WSE,
    P,
    Q,
    T,
    V,
    _banks,
    section_hydraulics,
    section_hydraulics_at,
)
from arc.xsection.xsection import XSection

G = 9.81 # m/s^2: the acceleration of gravity, for the velocity head
LEVELS = 240  # water surfaces in each cross section's table
TOLERANCE = 1e-6  # metres: how closely each step's water surface is found
CONTRACTION = 0.1  # loss coefficients on the change in velocity head, where the flow speeds up going downstream...
EXPANSION = 0.3  # ...and where it slows
EDDY_LENGTH = 300.0  # metres: the eddy loss is reckoned on the change in velocity head over this far downstream
OUTLET_REACH = 3.0  # cells: how near an outlet's last cross section a water surface given for it must be


@njit(cache=True, error_model="numpy")
def _alpha(area, conveyance, channel_area, channel_conveyance):
    """The velocity-head coefficient of the channel and the overbanks taken together:
    sum(K_i^3 / A_i^2) A^2 / K^3, over the channel and the two overbanks lumped as one."""
    terms = 0.0
    if channel_area > 0.0:
        terms += channel_conveyance ** 3 / channel_area ** 2

    overbank_area, overbank_conveyance = area - channel_area, conveyance - channel_conveyance
    if overbank_area > 1e-9 and overbank_conveyance > 0.0:
        terms += overbank_conveyance ** 3 / overbank_area ** 2

    return area * area * terms / conveyance ** 3 if conveyance > 0.0 else 1.0


@njit(cache=True, error_model="numpy")
def _state(stations, elevations, mannings_n, center, left_bank, right_bank, wse, k_decay, shallow_factor,
           deep_factor):
    """The area, conveyance and velocity-head coefficient at a water surface, exactly."""
    a, _, _, k, a_channel, k_channel = section_hydraulics(stations, elevations, mannings_n, center, left_bank,
                                                          right_bank, wse, k_decay, shallow_factor, deep_factor)
    return max(a, 1e-9), k, _alpha(a, k, a_channel, k_channel)


@njit(cache=True, error_model="numpy")
def _excess(stations, elevations, mannings_n, center, left_bank, right_bank, k_decay, shallow_factor, deep_factor,
            wse, q_up, energy_down, eddy_head, eddy_steps, conveyance_down, flow_sum, length, slope_factor, extra):
    """The energy at a water surface of the upstream cross section, less what the energy equation needs there: with
    the expansion loss where the water slows going downstream, and extra, the contraction loss (see _step_up)."""
    area, conveyance, alpha = _state(stations, elevations, mannings_n, center, left_bank, right_bank, wse, k_decay,
                                     shallow_factor, deep_factor)
    head = alpha * (q_up / area) ** 2 / (2.0 * G)
    friction = (flow_sum / max((conveyance_down + conveyance) * slope_factor, 1e-9)) ** 2
    return wse + head - (energy_down + length * friction + EXPANSION * max(head - eddy_head, 0.0) / eddy_steps
                         + extra)


@njit(cache=True, error_model="numpy")
def _root(stations, elevations, mannings_n, center, left_bank, right_bank, k_decay, shallow_factor, deep_factor,
          low, high, q_up, energy_down, eddy_head, eddy_steps, conveyance_down, flow_sum, length, slope_factor,
          extra):
    """The water surface between low (energy short) and high (enough) where the energy equation holds, by the
    Illinois variant of the false position method, to TOLERANCE."""
    f_low = _excess(stations, elevations, mannings_n, center, left_bank, right_bank, k_decay, shallow_factor,
                    deep_factor, low, q_up, energy_down, eddy_head, eddy_steps, conveyance_down, flow_sum, length,
                    slope_factor, extra)
    f_high = _excess(stations, elevations, mannings_n, center, left_bank, right_bank, k_decay, shallow_factor,
                     deep_factor, high, q_up, energy_down, eddy_head, eddy_steps, conveyance_down, flow_sum, length,
                     slope_factor, extra)
    if f_low >= 0.0:
        return low
    
    if f_high <= 0.0:
        return high
    
    wse = high
    side = 0
    for _ in range(100):
        wse = (low * f_high - high * f_low) / (f_high - f_low)
        f = _excess(stations, elevations, mannings_n, center, left_bank, right_bank, k_decay, shallow_factor,
                    deep_factor, wse, q_up, energy_down, eddy_head, eddy_steps, conveyance_down, flow_sum, length,
                    slope_factor, extra)
        if f > 0.0:
            high, f_high = wse, f
            if side == -1:
                f_low *= 0.5
            side = -1
        else:
            low, f_low = wse, f
            if side == 1:
                f_high *= 0.5
            side = 1

        if abs(f) < 1e-12:
            return wse
        
        if high - low < TOLERANCE:
            break

    # The side with energy enough: where the energy jumps across what's needed, as where the water tops a high point
    # of the cross section and the low ground beyond joins it, the bracket closes on the jump, and the side its last
    # guess fell on would decide the hydraulics the next step up takes from it
    return high


@njit(cache=True, error_model="numpy")
def _velocity_head(alpha, area, q):
    """a V^2 / 2g of q through an area."""
    return alpha * (q / area) ** 2 / (2.0 * G)


@njit(cache=True, error_model="numpy")
def _critical(levels, alpha, area, q, ordered=False):
    """The level of a table whose water surface plus velocity head is least, carrying q: critical depth. The first
    such level, or the first whose energy is NaN, as np.argmin picks. ordered says the table is (_ordered), so that
    the search can stop at the first level at or above the least energy so far: no energy is NaN or below its level."""
    best = 0
    least = levels[0] + _velocity_head(alpha[0], area[0], q)
    stop = ordered and abs(q) < 1e100  # and so is every velocity head finite
    for i in range(1, levels.size):
        if stop and levels[i] >= least:
            break

        energy = levels[i] + _velocity_head(alpha[i], area[i], q)
        if energy < least or (energy != energy and least == least):  # noqa: PLR0124
            best, least = i, energy

    return levels[best]


@njit(cache=True, error_model="numpy")
def _ordered(levels, alpha, area):
    """Whether a table's levels ascend, all finite, with its alpha finite and at least 0 and its area finite."""
    for i in range(levels.size):
        if not (np.isfinite(levels[i]) and np.isfinite(alpha[i]) and alpha[i] >= 0.0 and np.isfinite(area[i])):
            return False
        
        if i > 0 and not levels[i] > levels[i - 1]:
            return False
        
    return True


@njit(cache=True, error_model="numpy")
def _lowest_root(levels, alpha, area, conveyance, stations, elevations, mannings_n, center, left_bank, right_bank,
                 k_decay, shallow_factor, deep_factor, q_up, length, slope_factor, lowest, energy_down, flow_sum,
                 conveyance_down, eddy_head, eddy_steps, extra, ordered):
    """The lowest water surface at or above lowest where the energy is enough (_excess, with extra), and whether
    that is lowest itself, and the highest water surface the search worked the hydraulics out at."""
    if _excess(stations, elevations, mannings_n, center, left_bank, right_bank, k_decay, shallow_factor, deep_factor,
               lowest, q_up, energy_down, eddy_head, eddy_steps, conveyance_down, flow_sum, length, slope_factor,
               extra) >= 0.0:
        return lowest, True, lowest
    
    # The first level above lowest with energy enough, by the table, brackets the root with the level below it
    first = -1
    start = 0
    if ordered:
        # The levels up to lowest, by bisection
        high_index = levels.size
        while start < high_index:
            middle = (start + high_index) // 2
            if levels[middle] > lowest:
                high_index = middle
            else:
                start = middle + 1

    for i in range(start, levels.size):
        if levels[i] > lowest:
            head = _velocity_head(alpha[i], area[i], q_up)
            loss = EXPANSION * max(head - eddy_head, 0.0) / eddy_steps
            friction = (flow_sum / max((conveyance_down + conveyance[i]) * slope_factor, 1e-9)) ** 2
            if levels[i] + head - (energy_down + length * friction + loss + extra) >= 0.0:
                first = i
                break

    if first >= 0:
        low, high = max(levels[first - 1], lowest), levels[first]
    else:
        # The water below stands higher than the table reaches, backing water up beyond it: look higher
        low = max(levels[levels.size - 1], lowest)
        rise = max(energy_down - low, 0.0) + 0.5
        high = low + rise
        enough = False
        for _ in range(60):
            if _excess(stations, elevations, mannings_n, center, left_bank, right_bank, k_decay, shallow_factor,
                       deep_factor, high, q_up, energy_down, eddy_head, eddy_steps, conveyance_down, flow_sum, length,
                       slope_factor, extra) >= 0.0:
                enough = True
                break

            low, rise = high, 2.0 * rise
            high = low + rise

        if not enough:
            return lowest, True, high
        
    return _root(stations, elevations, mannings_n, center, left_bank, right_bank, k_decay, shallow_factor, deep_factor,
                 low, high, q_up, energy_down, eddy_head, eddy_steps, conveyance_down, flow_sum, length,
                 slope_factor, extra), False, high


@njit(cache=True, error_model="numpy")
def _step_up(levels, alpha, area, conveyance, stations, elevations, mannings_n, center, left_bank, right_bank, k_decay,
             shallow_factor, deep_factor, q_up, length, slope_factor, floor, wse_down, q_down, critical_down,
             conveyance_down, head_down, eddy_head, eddy_steps, ordered=False):
    """step_up, given the upstream cross section's table and profile, and the downstream one's water surface, its
    critical depth, and its conveyance and velocity head there. Also returns the upstream critical depth, which the
    next step up takes as its critical_down, and the highest water surface it worked out the upstream cross section's
    hydraulics at, beyond its table's levels. ordered is whether the table is (_ordered).

    The water surface is the lowest root with the expansion loss alone (which is the root with the eddy loss where the
    water slows going downstream); where the water there speeds up going downstream instead, the contraction loss is
    reckoned at its velocity head, and the water raised from there to make it up (see the notes above)."""
    critical = _critical(levels, alpha, area, q_up, ordered)
    if wse_down < critical_down:
        # The water below is supercritical, and no subcritical water surface steps up from it
        return max(critical, floor), True, critical, max(critical, floor)
    
    energy_down = wse_down + head_down
    flow_sum = q_up + q_down
    lowest = max(critical, floor)
    wse, at_lowest, top = _lowest_root(levels, alpha, area, conveyance, stations, elevations, mannings_n, center,
                                       left_bank, right_bank, k_decay, shallow_factor, deep_factor, q_up, length,
                                       slope_factor, lowest, energy_down, flow_sum, conveyance_down, eddy_head,
                                       eddy_steps, 0.0, ordered)
    water_area, _, water_alpha = _state(stations, elevations, mannings_n, center, left_bank, right_bank, wse, k_decay,
                                        shallow_factor, deep_factor)
    head = _velocity_head(water_alpha, water_area, q_up)
    if eddy_head > head:
        # The water speeds up going downstream: the contraction loss at this velocity head, made up from here
        contraction = CONTRACTION * (eddy_head - head) / eddy_steps
        wse, raised_none, raised_top = _lowest_root(levels, alpha, area, conveyance, stations, elevations, mannings_n,
                                                    center, left_bank, right_bank, k_decay, shallow_factor,
                                                    deep_factor, q_up, length, slope_factor, wse, energy_down,
                                                    flow_sum, conveyance_down, eddy_head, eddy_steps, contraction,
                                                    ordered)
        at_lowest, top = at_lowest and raised_none, max(top, raised_top)

    return wse, at_lowest and wse == critical, critical, top


@njit(cache=True, error_model="numpy")
def _march(levels, alpha, area, conveyance, stations, elevations, mannings_n, starts, centers, left_banks, right_banks,
           k_decay, shallow_factor, deep_factor, flows, start_wse, holds, hold_levels, lengths, far, slope_factor):
    """A reach's water surfaces at each of its flows (see apply_standard_step), its cross sections in order downstream:
    each one's table (levels, alpha, area and conveyance, a row each) and profile (cross section m's from starts[m] to
    starts[m + 1]), the last one's water surface at each flow before the checks (start_wse), the flows each one holds
    its water at most at its hold level, the lengths from each to the next and its eddy loss's far end (far). Also the
    number of steps, and for each cross section how many of its water surfaces are at critical depth and how many are
    held, and the highest water surface its hydraulics were worked out at."""
    size, increments = flows.shape
    wse = np.empty((size, increments))
    heads = np.empty((size, increments))
    steps = 0
    critical_at = np.zeros(size, np.int64)
    held_at = np.zeros(size, np.int64)
    highest = np.full(size, -np.inf)
    last = size - 1
    ordered = np.empty(size, np.bool_)  # whether each table is (_ordered), for the critical depth searches
    for m in range(size):
        ordered[m] = _ordered(levels[m], alpha[m], area[m])

    for j in range(increments):
        # No cross section's water falls as the flow rises
        floor = wse[last, j - 1] if j > 0 else -np.inf
        critical = _critical(levels[last], alpha[last], area[last], flows[last, j], ordered[last])
        w = max(start_wse[j], critical, floor)
        if holds[last, j] and w > hold_levels[last]:
            w = max(hold_levels[last], floor)
            held_at[last] += 1

        wse[last, j] = w
        highest[last] = max(highest[last], w)
        a, b = starts[last], starts[last + 1]
        area_down, conveyance_down, alpha_down = _state(stations[a:b], elevations[a:b], mannings_n[a:b], centers[last],
                                                        left_banks[last], right_banks[last], w, k_decay,
                                                        shallow_factor, deep_factor)
        heads[last, j] = _velocity_head(alpha_down, area_down, flows[last, j])
        for m in range(size - 2, -1, -1):
            floor = wse[m, j - 1] if j > 0 else -np.inf
            end = far[m]
            a, b = starts[m], starts[m + 1]
            w, at_critical, critical, top = _step_up(levels[m], alpha[m], area[m], conveyance[m], stations[a:b],
                                                elevations[a:b], mannings_n[a:b], centers[m], left_banks[m],
                                                right_banks[m], k_decay, shallow_factor, deep_factor, flows[m, j],
                                                lengths[m], slope_factor, floor, wse[m + 1, j], flows[m + 1, j],
                                                critical, conveyance_down, heads[m + 1, j], heads[end, j],
                                                float(end - m), ordered[m])
            if holds[m, j] and w > hold_levels[m]:
                # The baseflow stays in its channel: the energy over its level is lost here
                w, at_critical = max(hold_levels[m], floor), False
                held_at[m] += 1

            wse[m, j] = w
            highest[m] = max(highest[m], top, w)
            area_down, conveyance_down, alpha_down = _state(stations[a:b], elevations[a:b], mannings_n[a:b],
                                                            centers[m], left_banks[m], right_banks[m], w, k_decay,
                                                            shallow_factor, deep_factor)
            heads[m, j] = _velocity_head(alpha_down, area_down, flows[m, j])
            steps += 1
            critical_at[m] += int(at_critical)

    return wse, steps, critical_at, held_at, highest


@njit(cache=True, error_model="numpy")
def _rewrite(stations, elevations, mannings_n, starts, centers, left_banks, right_banks, k_decay, shallow_factor,
             deep_factor, flows, wse, out):
    """A reach's increments at its flows and water surfaces, each with its own cross section's hydraulics there (the
    profiles as _march takes them)."""
    for m in range(flows.shape[0]):
        a, b = starts[m], starts[m + 1]
        for j in range(flows.shape[1]):
            q, w = flows[m, j], wse[m, j]
            area, perimeter, top_width, _, _, _ = section_hydraulics(stations[a:b], elevations[a:b], mannings_n[a:b],
                                                                     centers[m], left_banks[m], right_banks[m], w,
                                                                     k_decay, shallow_factor, deep_factor)
            out[m, j, Q], out[m, j, V], out[m, j, T], out[m, j, WSE], out[m, j, P] = \
                q, q / max(area, 1e-9), top_width, w, perimeter


@njit(cache=True, error_model="numpy")
def _table(stations, elevations, mannings_n, center, left_bank, right_bank, levels, k_decay, shallow_factor,
           deep_factor):
    """Area, wetted perimeter, top width, conveyance and velocity-head coefficient at each level."""
    m = levels.size
    area = np.empty(m)
    perimeter = np.empty(m)
    width = np.empty(m)
    conveyance = np.empty(m)
    alpha = np.empty(m)
    ascending = True
    for i in range(1, m):
        ascending = ascending and levels[i] > levels[i - 1]

    if ascending:
        # All at once, which gives the same numbers in half the time
        rows = section_hydraulics_at(stations, elevations, mannings_n, center, left_bank, right_bank, levels, k_decay,
                                     shallow_factor, deep_factor)
        for i in range(m):
            area[i], perimeter[i], width[i], conveyance[i] = rows[i, 0], rows[i, 1], rows[i, 2], rows[i, 3]
            alpha[i] = _alpha(rows[i, 0], rows[i, 3], rows[i, 4], rows[i, 5])

        return area, perimeter, width, conveyance, alpha
    
    for i in range(m):
        a, p, t, k, a_channel, k_channel = section_hydraulics(stations, elevations, mannings_n, center, left_bank,
                                                              right_bank, levels[i], k_decay, shallow_factor,
                                                              deep_factor)
        area[i], perimeter[i], width[i], conveyance[i] = a, p, t, k
        alpha[i] = _alpha(a, k, a_channel, k_channel)

    return area, perimeter, width, conveyance, alpha


class HydraulicTable:
    """A cross section's area, top width, wetted perimeter, conveyance and velocity-head coefficient at LEVELS water
    surfaces, from just above its bed to `top` and some way beyond: a metre for each two metres of depth, and at
    least three metres."""

    def __init__(self, xs: XSection, top: float, roughness: DepthRoughness | None = None):
        profile = hydraulic_profile(xs)
        # The cross section as the compiled functions take it: its profile and banks, then the roughness parameters
        self.section = (*profile, *_banks(xs))
        self.parameters = _parameters(roughness)
        self.bed = float(profile.elevations[profile.center])
        span = max(top - self.bed, 0.5)
        self.levels = self.bed + np.linspace(0.005, span + max(3.0, 0.5 * span), LEVELS)
        area, self.perimeter, self.width, self.conveyance, self.alpha = _table(*self.section, self.levels, *self.parameters)
        self.area = np.maximum(area, 1e-9)

    def state(self, wse: float) -> tuple[float, float, float]:
        """The area, conveyance and velocity-head coefficient at a water surface, worked out exactly."""
        return _state(*self.section, float(wse), *self.parameters)

    def normal(self, q: float, slope: float, slope_factor: float = 1.0) -> float:
        """The lowest water surface carrying q by Manning's equation, or the top level if none does."""
        discharges = self.conveyance * math.sqrt(max(slope, 0.0)) * slope_factor
        above = np.flatnonzero(discharges >= q)
        if above.size == 0:
            return float(self.levels[-1])
        
        i = int(above[0])
        if i == 0:
            return float(self.levels[0])
        
        share = (q - discharges[i - 1]) / (discharges[i] - discharges[i - 1])
        return float(self.levels[i - 1] + share * (self.levels[i] - self.levels[i - 1]))

    def specific_energy(self, q: float) -> np.ndarray:
        """The water surface plus the velocity head at each level, carrying q."""
        return self.levels + self.alpha * (q / self.area) ** 2 / (2.0 * G)

    def critical(self, q: float) -> float:
        """The water surface of least energy for q: critical depth."""
        return float(_critical(self.levels, self.alpha, self.area, float(q)))


def velocity_head(table: HydraulicTable, wse: float, q: float) -> float:
    """The velocity head, a V^2 / 2g, of q at a water surface, worked out exactly."""
    area, _, alpha = table.state(wse)
    return float(_velocity_head(alpha, area, float(q)))


def step_up(down: HydraulicTable, wse_down: float, q_down: float, up: HydraulicTable, q_up: float, length: float,
            slope_factor: float = 1.0, floor: float = -math.inf, eddy_head: float | None = None,
            eddy_steps: int = 1) -> tuple[float, bool]:
    """The water surface at the cross section `up`, `length` metres upstream of `down`, by the energy equation (see
    the notes above), and whether it is critical depth for want of a subcritical root. up's table brackets the root,
    and the exact hydraulics then find it.

    The water surface is the lowest root at or above floor (the cross section's water at a smaller flow), or floor
    itself if the energy there is already enough: a compound channel can have more than one root, and the lowest can
    otherwise drop as the flow rises.

    The eddy loss is C |h_up - eddy_head| / eddy_steps: with eddy_head the velocity head eddy_steps cross sections
    downstream, the step's share of the change over them (see apply_standard_step). By default, the change from down
    alone, as the energy equation above has it. Where the water speeds up going downstream, h_up is the velocity head
    without the loss (see the notes above)."""
    area_d, conveyance_d, alpha_d = down.state(wse_down)
    head_d = _velocity_head(alpha_d, area_d, float(q_down))
    if eddy_head is None:
        eddy_head, eddy_steps = head_d, 1

    wse, critical, _, _ = _step_up(up.levels, up.alpha, up.area, up.conveyance, *up.section, *up.parameters, float(q_up),
                                float(length), float(slope_factor), float(floor), float(wse_down), float(q_down),
                                down.critical(q_down), conveyance_d, head_d, float(eddy_head),
                                float(max(int(eddy_steps), 1)))
    return float(wse), bool(critical)


class StepSummary(NamedTuple):
    """What apply_standard_step did: the cells it rewrote, how many steps it took and went critical, and how many
    water surfaces (a cell at a flow) it held at the baseflow level. critical_at and held_at give the last two for
    each cell, in the order of the sections it was given."""
    cells: int
    steps: int
    critical: int
    held: int = 0
    given: int = 0  # outlets started from the water surfaces given for them (outlet_wse)
    critical_at: np.ndarray | None = None
    held_at: np.ndarray | None = None


def flows_for(max_flow: float, baseflow: float, increments: int) -> np.ndarray:
    """A cell's flows: j/N of its maximum flow, j = 1 to N, the first at or above its baseflow set just below it, so
    that the baseflow is one of them (none is if the baseflow is above the maximum flow)."""
    flows = max_flow * np.arange(1, increments + 1) / increments
    if increments and baseflow > BASEFLOW_MARGIN:
        reaching = np.flatnonzero(flows >= baseflow)
        if reaching.size:
            j = int(reaching[0])
            node = baseflow - BASEFLOW_MARGIN
            if j == 0 or node > flows[j - 1]:
                flows[j] = node

    return flows


def apply_standard_step(network: nx.DiGraph, rows: np.ndarray, cols: np.ndarray, reaches: np.ndarray, sections: list,
                        slopes: np.ndarray, max_flow: np.ndarray, baseflow: np.ndarray, curves: RatingCurves,
                        dx: float, dy: float, *, roughness: DepthRoughness | None = None,
                        slope_factor: float = 1.0, eddy_length: float = EDDY_LENGTH, stream_cells=None, outlets=None,
                        elevations: np.ndarray | None = None,
                        baseflow_levels: np.ndarray | None = None,
                        ensure_level: Callable[[int, float], bool] | None = None,
                        outlet_wse: dict | None = None) -> StepSummary:
    """Rewrite the rating curves of the cells that have them (see the notes above), in place.

    `rows`, `cols` and `reaches` are each cell's stream cell and reach, and `sections` its cross section (anything with an
    `xs` and a `usable` flag, or None). `network` runs from each reach to the one downstream (reach_network), and must
    hold every reach with a rated cell. `eddy_length` is how far downstream the eddy loss looks (see the notes above);
    0 charges each step the change from the cross section below alone. `baseflow_levels` is each cell's baseflow
    level, the water surface its channel was carved to carry its baseflow at, or NaN (and None) for none: at every
    flow up to its baseflow, its water is held at most there.

    A reach whose neighbours have no rated cells is ordered by their cells on the stream raster (`stream_cells`, each
    reach's rows and columns), or by the cell where its line ends (`outlets`, each reach's (row, col)), and failing
    both with its higher end upstream, by `elevations` (each cell's stream cell's, NaN for none) or else its cross
    sections' beds (see order_reach).

    `outlet_wse` gives the water surfaces to start some outlets from instead of normal depth, as a neighbouring tile's
    rating curves do where the reach leaves this one: by reach, a list of points (row, col, flows, wse), each a place
    on the raster (rows and columns, fractional) and its water surface at increasing flows. An outlet takes the point
    nearest its last cross section, if one is within OUTLET_REACH cells (_given_wse): its water at the flows within
    the point's, and beyond them normal depth as far above or below as at the nearest of those.

    `ensure_level(k, level)`, if given, makes cell k's cross section reach above the water surface level on both sides
    (a trimmed one, see arc.pipeline), and says whether it changed it. It is asked for each table's top level once the
    table is made (which is made again if it did), and for the highest water surface the march worked each cross
    section out at after it, which is marched again if any changed.
    """
    slope_factor = _check_slope_factor(slope_factor)
    increments = curves.number_of_increments
    rated = [k for k, section in enumerate(sections)
             if section is not None and section.usable and increments and np.isfinite(curves.increments[k, -1, Q])]
    by_reach: dict[int, list[int]] = {}
    for k in rated:
        by_reach.setdefault(int(reaches[k]), []).append(k)

    missing = sorted(reach for reach in by_reach if reach not in network)
    if missing:
        raise ValueError("These reaches have rating curves but aren't in the stream network: "
                         + ", ".join(map(str, missing[:10])) + ("..." if len(missing) > 10 else "") + ".")
    
    try:
        downstream_first = [reach for reach in nx.topological_sort(network) if reach in by_reach][::-1]
    except nx.NetworkXUnfeasible:
        raise ValueError("The stream network has a loop, so it has no outlet to start the standard step from.") from None

    cells = {reach: (np.asarray(rows)[ks].astype(np.int64), np.asarray(cols)[ks].astype(np.int64)) for reach, ks in by_reach.items()}
    upstream_wse: dict[int, np.ndarray] = {}  # each reach's first cell's water surface at each j
    first_cell: dict[int, tuple[int, int]] = {}  # and that cell
    levels_all = np.full(len(sections), math.nan) if baseflow_levels is None else np.asarray(baseflow_levels, dtype=np.float64)
    steps = given = 0
    critical_at, held_at = np.zeros(len(sections), np.int64), np.zeros(len(sections), np.int64)
    for reach in downstream_first:
        ks = np.array(by_reach[reach])
        heights = np.asarray(elevations, dtype=np.float64)[ks] if elevations is not None else \
            np.array([float(sections[k].xs.elevations[sections[k].xs.elevations.size // 2]) for k in ks])
        order, stations = order_reach(network, reach, *cells[reach], dx, dy, cells, heights, stream_cells=stream_cells, outlets=outlets)
        ks, stations = ks[order], np.asarray(stations, dtype=np.float64)
        tops = [float(np.nanmax(curves.increments[k, :, WSE])) for k in ks]
        tables = [_table_for(sections[k].xs, top, roughness, k, ensure_level) for k, top in zip(ks, tops)]
        flows = np.array([flows_for(float(max_flow[k]), float(baseflow[k]), increments) for k in ks])
        # Each cell's level and the flows it holds the water at most at it: those up to its baseflow
        levels = levels_all[ks]
        holds = (flows <= np.asarray(baseflow, dtype=np.float64)[ks, None]) & np.isfinite(levels)[:, None]
        below, cap = _reach_below(network, reach, upstream_wse), np.full(increments, math.inf)
        if below is not None and not network.has_edge(reach, below):
            # Past reaches without rated cells: an outlet, its water no higher than the water below risen by the slope
            row, col = first_cell[below]
            gap = math.hypot((int(rows[ks[-1]]) - row) * dy, (int(cols[ks[-1]]) - col) * dx)
            cap = upstream_wse[below] + float(np.median(np.asarray(slopes)[ks])) * gap
            below = None

        if below is not None:
            start = upstream_wse[below]
        else:
            start = np.array([_outlet_wse(sections[ks[-1]].xs, tables[-1], flows[-1, j], float(slopes[ks[-1]]),
                                          roughness, slope_factor, cap[j]) for j in range(increments)])
            if outlet_wse and reach in outlet_wse:
                wse_given = _given_wse(outlet_wse[reach], float(rows[ks[-1]]), float(cols[ks[-1]]), flows[-1], dx, dy)
                inside = np.flatnonzero(np.isfinite(wse_given))
                if inside.size:
                    # The given water, and beyond its flows, normal depth as far above or below as at the nearest
                    start = start + np.interp(np.arange(increments), inside, (wse_given - start)[inside])
                    given += 1
            
        far = _eddy_ends(stations, eddy_length)
        while True:
            packed = _packed(tables)
            wse, reach_steps, reach_critical, reach_held, highest = _march(
                *packed, *tables[-1].parameters, flows, start, holds, levels, np.abs(np.diff(stations)), far,
                slope_factor)
            if ensure_level is None:
                break

            changed = [ensure_level(int(k), float(level)) for k, level in zip(ks, highest)]
            if not any(changed):
                break

            # The march worked a cross section out above where it reached: again, with it reaching further
            tables = [HydraulicTable(sections[k].xs, top, roughness) if grew else table
                      for k, top, table, grew in zip(ks, tops, tables, changed)]
            
        steps += reach_steps
        critical_at[ks], held_at[ks] = reach_critical, reach_held
        upstream_wse[reach] = wse[0]
        first_cell[reach] = (int(rows[ks[0]]), int(cols[ks[0]]))
        out = curves.increments[ks]
        _rewrite(*packed[4:], *tables[-1].parameters, flows, wse, out)
        curves.increments[ks] = out

    return StepSummary(len(rated), steps, int(critical_at.sum()), int(held_at.sum()), given, critical_at, held_at)


def _given_wse(points: list, row: float, col: float, flows: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """The water surface at each flow given for an outlet whose last cross section is at (row, col): that of the point
    nearest it, if one is within OUTLET_REACH cells, interpolated by flow at the flows within the point's, NaN at the
    others (and at all, without a point near enough)."""
    nearest, best = None, OUTLET_REACH * max(dx, dy)
    for point in points:
        distance = math.hypot((point[0] - row) * dy, (point[1] - col) * dx)
        if distance <= best:
            nearest, best = point, distance

    out = np.full(flows.size, math.nan)
    if nearest is not None:
        q, w = nearest[2], nearest[3]
        inside = (flows >= q[0]) & (flows <= q[-1])
        out[inside] = np.interp(flows[inside], q, w)
        
    return out


def _table_for(xs: XSection, top: float, roughness: DepthRoughness | None, k: int,
               ensure_level: Callable[[int, float], bool] | None) -> HydraulicTable:
    """A cross section's table, made again if ensure_level has to extend it to reach the table's top level."""
    table = HydraulicTable(xs, top, roughness)
    if ensure_level is not None and ensure_level(int(k), float(table.levels[-1])):
        table = HydraulicTable(xs, top, roughness)
    return table


def _outlet_wse(xs: XSection, table: HydraulicTable, q: float, slope: float, roughness: DepthRoughness | None,
                slope_factor: float, cap: float) -> float:
    """An outlet's water surface at a flow, before the checks: its normal depth, by the table where
    wse_for_discharge has none, and no higher than cap."""
    wse = wse_for_discharge(xs, q, slope, roughness=roughness, slope_factor=slope_factor)
    if not math.isfinite(wse):
        wse = table.normal(q, slope, slope_factor)

    return min(wse, cap)


def _packed(tables: list[HydraulicTable]) -> tuple:
    """A reach's tables and cross sections as _march takes them: the tables' levels, alpha, area and conveyance, a row
    for each, then the profiles' stations, elevations and Manning's n end to end, where each starts, and each one's
    centre and banks."""
    sections = [table.section for table in tables]
    starts = np.zeros(len(sections) + 1, dtype=np.int64)
    np.cumsum([section[0].size for section in sections], out=starts[1:])
    return (np.array([table.levels for table in tables]), np.array([table.alpha for table in tables]),
            np.array([table.area for table in tables]), np.array([table.conveyance for table in tables]),
            *(np.concatenate([section[i] for section in sections]).astype(np.float64, copy=False) for i in range(3)),
            starts, np.array([section[3] for section in sections], dtype=np.int64),
            np.array([section[4] for section in sections], dtype=np.float64),
            np.array([section[5] for section in sections], dtype=np.float64))


def _reach_below(network: nx.DiGraph, reach: int, worked_out) -> int | None:
    """The nearest reach downstream whose water surfaces are worked out, past any that have no rated cells, or None
    for an outlet."""
    seen, queue = {reach}, deque(network.successors(reach))
    while queue:
        below = queue.popleft()
        if below in worked_out:
            return below
        
        if below not in seen:
            seen.add(below)
            queue.extend(network.successors(below))

    return None


@njit(cache=True, error_model="numpy")
def _eddy_ends(stations, eddy_length):
    """For each of a reach's cross sections, in order downstream, the first below it at least eddy_length metres
    away along the stream, or the reach's last; at least the next one."""
    size = stations.size
    ends = np.empty(size, dtype=np.int64)
    for m in range(size):
        i = min(m + 1, size - 1)
        while i < size - 1 and abs(stations[i] - stations[m]) < eddy_length:
            i += 1
        ends[m] = i
        
    return ends
