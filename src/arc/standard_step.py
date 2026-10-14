"""The standard-step method: water surfaces linked along the stream network (Standard_Step).

Without it, each cross section's rating curve is its own: the water is at normal depth for the cross section's
geometry and slope, whatever the cross sections around it are like. With it, the water surfaces come from the energy
equation between neighbouring cross sections instead, marched upstream from the network's outlets, as in a
one-dimensional hydraulic model (Chow, 1959). Between a cross section u and the one below it, d, L metres apart along
the stream:

    WSE_u + a_u V_u^2 / 2g = WSE_d + a_d V_d^2 / 2g + L Sf + C |a_u V_u^2 / 2g - a_d V_d^2 / 2g|

with Sf = ((Q_u + Q_d) / ((K_u + K_d) f))^2, the average-conveyance friction slope (K the conveyance, f the
slope_adjustment_factor), a the velocity-head coefficient of the channel and the overbanks, and C 0.1 where the flow
speeds up going downstream and 0.3 where it slows. The upstream water surface is the subcritical root, the lowest above
critical depth, or critical depth where there's none (where the water below is supercritical, or the cross section above
needs more energy even at critical depth).

Each reach's rated cells are put in order along the stream (arc.bathymetry.order_reach), and their rating curves are
worked out again at flows common to the reach: Q_j = j/N of each cell's maximum flow, j = 1 to N
(VDT_Database_NumIterations), with the first set just below the baseflow if it was at or above it, as rating_curve
does. A reach starts from the water surface of the first rated cell of the reach below it at the same j (the same share
of that reach's maximum flow), and an outlet from its last cell's normal depth (arc.hydraulics.wse_for_discharge), both
at least critical depth. Each
increment's velocity, top width and wetted perimeter are then the cell's own cross section's at its water surface.

Each cross section has a table of LEVELS water surfaces, from its bed to well above the top of its own rating curve:
it brackets each step's water surface (or, where the water below stands higher than the table reaches, the search goes
on above it), and gives the critical depth, and an outlet's normal depth where wse_for_discharge has none. Each water
surface is then found with the exact hydraulics, so that no interpolation error builds up from one step to the next.
"""
from __future__ import annotations

import math
from typing import NamedTuple

import networkx as nx
import numpy as np
from numba import njit

from arc.bathymetry.smoothing import order_reach
from arc.hydraulics import DepthRoughness, _check_slope_factor, _parameters, hydraulic_profile, wse_for_discharge
from arc.outputs.rating_curves import RatingCurves
from arc.rating_curve import BASEFLOW_MARGIN, P, Q, T, V, WSE, _banks, section_hydraulics
from arc.xsection.xsection import XSection

G = 9.81
LEVELS = 240  # water surfaces in each cross section's table
TOLERANCE = 1e-6  # metres: how closely each step's water surface is found
CONTRACTION = 0.1  # loss coefficients on the change in velocity head, where the flow speeds up going downstream...
EXPANSION = 0.3  # ...and where it slows


@njit(cache=True, error_model="numpy")
def _alpha(area, conveyance, channel_area, channel_conveyance):
    """The velocity-head coefficient of the channel and the overbanks taken together:
    sum(K_i^3 / A_i^2) A^2 / K^3."""
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
            wse, q_up, energy_down, head_down, conveyance_down, flow_sum, length, slope_factor):
    """The energy at a water surface of the upstream cross section, less what the energy equation needs there."""
    area, conveyance, alpha = _state(stations, elevations, mannings_n, center, left_bank, right_bank, wse, k_decay,
                                     shallow_factor, deep_factor)
    head = alpha * (q_up / area) ** 2 / (2.0 * G)
    coefficient = CONTRACTION if head_down > head else EXPANSION
    friction = (flow_sum / max((conveyance_down + conveyance) * slope_factor, 1e-9)) ** 2
    return wse + head - (energy_down + length * friction + coefficient * abs(head - head_down))


@njit(cache=True, error_model="numpy")
def _root(stations, elevations, mannings_n, center, left_bank, right_bank, k_decay, shallow_factor, deep_factor,
          low, high, q_up, energy_down, head_down, conveyance_down, flow_sum, length, slope_factor):
    """The water surface between low (energy short) and high (enough) where the energy equation holds, by the
    Illinois variant of the false position method, to TOLERANCE."""
    f_low = _excess(stations, elevations, mannings_n, center, left_bank, right_bank, k_decay, shallow_factor,
                    deep_factor, low, q_up, energy_down, head_down, conveyance_down, flow_sum, length, slope_factor)
    f_high = _excess(stations, elevations, mannings_n, center, left_bank, right_bank, k_decay, shallow_factor,
                     deep_factor, high, q_up, energy_down, head_down, conveyance_down, flow_sum, length, slope_factor)
    if f_low >= 0.0:
        return low
    if f_high <= 0.0:
        return high
    wse = high
    side = 0
    for _ in range(100):
        wse = (low * f_high - high * f_low) / (f_high - f_low)
        f = _excess(stations, elevations, mannings_n, center, left_bank, right_bank, k_decay, shallow_factor,
                    deep_factor, wse, q_up, energy_down, head_down, conveyance_down, flow_sum, length, slope_factor)
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
        if high - low < TOLERANCE or abs(f) < 1e-12:
            break
    return wse


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
    for i in range(m):
        a, p, t, k, a_channel, k_channel = section_hydraulics(stations, elevations, mannings_n, center, left_bank,
                                                              right_bank, levels[i], k_decay, shallow_factor,
                                                              deep_factor)
        area[i], perimeter[i], width[i], conveyance[i] = a, p, t, k
        alpha[i] = _alpha(a, k, a_channel, k_channel)
    return area, perimeter, width, conveyance, alpha


class HydraulicTable:
    """A cross section's area, top width, wetted perimeter, conveyance and velocity-head coefficient at LEVELS water
    surfaces, from just above its bed to `top` and some way beyond: a metre and a half for each two metres of depth,
    and at least three."""

    def __init__(self, xs: XSection, top: float, roughness: DepthRoughness | None = None):
        profile = hydraulic_profile(xs)
        # The cross section as the compiled functions take it: its profile and banks, then the roughness parameters
        self.section = (*profile, *_banks(xs))
        self.parameters = _parameters(roughness)
        self.bed = float(profile.elevations[profile.center])
        span = max(top - self.bed, 0.5)
        self.levels = self.bed + np.linspace(0.005, span + max(3.0, 0.5 * span), LEVELS)
        area, self.perimeter, self.width, self.conveyance, self.alpha = _table(*self.section, self.levels,
                                                                               *self.parameters)
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
        return float(self.levels[int(np.argmin(self.specific_energy(q)))])


def step_up(down: HydraulicTable, wse_down: float, q_down: float, up: HydraulicTable, q_up: float, length: float,
            slope_factor: float = 1.0) -> tuple[float, bool]:
    """The water surface at the cross section `up`, `length` metres upstream of `down`, by the energy equation (see
    the notes above), and whether it is critical depth for want of a subcritical root. up's table brackets the root,
    and the exact hydraulics then find it."""
    if wse_down < down.critical(q_down):
        # The water below is supercritical, and no subcritical water surface steps up from it
        return up.critical(q_up), True
    area_d, conveyance_d, alpha_d = down.state(wse_down)
    head_d = alpha_d * (q_down / area_d) ** 2 / (2.0 * G)
    head = up.alpha * (q_up / up.area) ** 2 / (2.0 * G)
    energy = up.levels + head
    loss = np.where(head_d > head, CONTRACTION, EXPANSION) * np.abs(head - head_d)
    friction = ((q_up + q_down) / np.maximum((conveyance_d + up.conveyance) * slope_factor, 1e-9)) ** 2
    excess = energy - (wse_down + head_d + length * friction + loss)
    critical = int(np.argmin(energy))
    if excess[critical] >= 0.0:
        return float(up.levels[critical]), True
    above = np.flatnonzero(excess[critical:] >= 0.0)
    arguments = (float(q_up), wse_down + head_d, head_d, conveyance_d, float(q_up + q_down), float(length),
                 float(slope_factor))
    if above.size:
        i1 = critical + int(above[0])
        low, high = float(up.levels[i1 - 1]), float(up.levels[i1])
    else:
        # The water below stands higher than the table reaches, backing water up beyond it: look higher
        low = float(up.levels[-1])
        rise = max(wse_down + head_d - low, 0.0) + 0.5
        high = low + rise
        for _ in range(60):
            if _excess(*up.section, *up.parameters, high, *arguments) >= 0.0:
                break
            low, rise = high, 2.0 * rise
            high = low + rise
        else:
            return float(up.levels[critical]), True
    return float(_root(*up.section, *up.parameters, low, high, *arguments)), False


class StepSummary(NamedTuple):
    """What apply_standard_step did: the cells it rewrote, and how many steps it took and went critical."""
    cells: int
    steps: int
    critical: int


def flows_for(max_flow: float, baseflow: float, increments: int) -> np.ndarray:
    """A cell's flows: j/N of its maximum flow, j = 1 to N, the first set just below a baseflow it reaches."""
    flows = max_flow * np.arange(1, increments + 1) / increments
    if increments and baseflow > BASEFLOW_MARGIN and flows[0] >= baseflow:
        flows[0] = baseflow - BASEFLOW_MARGIN
    return flows


def apply_standard_step(network: nx.DiGraph, rows: np.ndarray, cols: np.ndarray, reaches: np.ndarray, sections: list,
                        slopes: np.ndarray, max_flow: np.ndarray, baseflow: np.ndarray, curves: RatingCurves,
                        dx: float, dy: float, *, roughness: DepthRoughness | None = None,
                        slope_factor: float = 1.0) -> StepSummary:
    """Rewrite the rating curves of the cells that have them (see the notes above), in place.

    rows, cols and reaches are each cell's stream cell and reach, and sections its cross section (anything with an
    `xs` and a `usable` flag, or None). network runs from each reach to the one downstream (reach_network), and must
    hold every reach with a rated cell.
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
        raise ValueError("The stream network has a loop, so it has no outlet to start the standard step from.") \
            from None

    cells = {reach: (np.asarray(rows)[ks].astype(np.int64), np.asarray(cols)[ks].astype(np.int64))
             for reach, ks in by_reach.items()}
    upstream_wse: dict[int, np.ndarray] = {}  # each reach's first cell's water surface at each j
    steps = critical_steps = 0
    for reach in downstream_first:
        ks = np.array(by_reach[reach])
        beds = np.array([float(sections[k].xs.elevations[sections[k].xs.elevations.size // 2]) for k in ks])
        order, stations = order_reach(network, reach, *cells[reach], dx, dy, cells, beds)
        ks, stations = ks[order], np.asarray(stations, dtype=np.float64)
        tables = [HydraulicTable(sections[k].xs, float(np.nanmax(curves.increments[k, :, WSE])), roughness)
                  for k in ks]
        flows = np.array([flows_for(float(max_flow[k]), float(baseflow[k]), increments) for k in ks])
        below = next((s for s in network.successors(reach) if s in upstream_wse), None)
        wse = np.empty((ks.size, increments))
        for j in range(increments):
            last = tables[-1]
            if below is not None:
                start = upstream_wse[below][j]
            else:
                start = wse_for_discharge(sections[ks[-1]].xs, flows[-1, j], float(slopes[ks[-1]]),
                                          roughness=roughness, slope_factor=slope_factor)
                if not math.isfinite(start):
                    start = last.normal(flows[-1, j], float(slopes[ks[-1]]), slope_factor)
            wse[-1, j] = max(start, last.critical(flows[-1, j]))
            for m in range(ks.size - 2, -1, -1):
                wse[m, j], critical = step_up(tables[m + 1], wse[m + 1, j], flows[m + 1, j], tables[m], flows[m, j],
                                              abs(stations[m + 1] - stations[m]), slope_factor)
                steps += 1
                critical_steps += int(critical)
        upstream_wse[reach] = wse[0]
        for i, k in enumerate(ks):
            _rewrite(curves.increments[k], sections[k].xs, flows[i], wse[i], roughness)
    return StepSummary(len(rated), steps, critical_steps)


def _rewrite(out: np.ndarray, xs: XSection, flows: np.ndarray, wse: np.ndarray,
             roughness: DepthRoughness | None) -> None:
    """A cell's increments at its flows and water surfaces, with its own cross section's hydraulics there."""
    profile = hydraulic_profile(xs)
    left_bank, right_bank = _banks(xs)
    parameters = _parameters(roughness)
    for j, (q, w) in enumerate(zip(flows, wse)):
        area, perimeter, top_width, _, _, _ = section_hydraulics(*profile, left_bank, right_bank, float(w),
                                                                 *parameters)
        out[j, Q], out[j, V], out[j, T], out[j, WSE], out[j, P] = q, q / max(area, 1e-9), top_width, w, perimeter
