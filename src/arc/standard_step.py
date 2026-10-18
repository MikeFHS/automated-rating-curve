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
head from u to the cross section w, the first EDDY_LENGTH (1 km) or more downstream in the reach (or its last), n cross
sections below u, and each step takes 1/n of it. A steady widening or narrowing is charged in full that way, but not the
cross sections' differences from their neighbours: with cross sections a cell apart, those come and go over tens of
metres, and charging each step's change (n = 1, as HEC-RAS does with cross sections hundreds of metres apart) added up
to half the friction on a big braided river, and raised its water metres above normal depth.

The upstream water surface is the subcritical root, the lowest above critical depth and above the cross section's water
at the flow before (so that it never falls as the flow rises, which the lowest root can in a compound channel), or
critical depth where there's none (where the water below is supercritical, or the cross section above needs more energy
even at critical depth).

Each reach's rated cells are put in order along the stream (arc.bathymetry.order_reach): by the reaches next to it in
the network, on the stream raster if they have no rating curves, or else by where its line ends, and only failing
those by its stream cells' elevations. A march from the wrong end piles water up from there, and the elevations, the
carved beds above all, are noisy where the water is flat. Their rating curves are then worked out again at flows
common to the reach: Q_j = j/N of each cell's maximum flow, j = 1 to N
(VDT_Database_NumIterations), with the first at or above the baseflow set just below it, so that the baseflow is one
of them (rating_curve does the same when that is the first). A reach starts from the water surface of the first rated cell of the reach below it at the same j (the same share
of that reach's maximum flow), and an outlet from its last cell's normal depth (arc.hydraulics.wse_for_discharge), both
at least critical depth. Where the reaches below have no rated cells (one in a lake, or one the flow file leaves out),
the reach starts as an outlet does, but no higher than the water at the top of the first reach further down that has
them, risen by the reach's median slope over the straight distance between them: normal depth at one cross section, a
choke as like as not on flat water, could otherwise stand the whole reach metres above the river below it. The water
below isn't passed on in full, as it is from a reach next to it, as the gap's fall is only an estimate. Each
increment's velocity, top width and wetted perimeter are then the cell's own cross section's at its water surface.

The baseflow stays in the channel. The bathymetry carves each channel to carry its baseflow, by Manning's equation,
with its water at a level (its bank elevation with Bathy_Use_Banks, its stream cell's otherwise), so normal depth puts
the baseflow there. The step needn't: where a channel was carved for a steeper slope than the levels fall, its water
loses more to friction than the banks drop, and rises over them going upstream (on 68 test sites, the friction with
full banks added up to 1.26 times the banks' fall, and the baseflow stood over half a metre above the banks at 23% of
cells, and over 0.1 m at 55%). So at every flow up to its baseflow, a cell's water is held at most at its level
(baseflow_levels), the energy above that taken as lost there, as over a riffle, and the march goes on upstream from
the held water. A steep section is held there too where its critical depth is higher: supercritical water is below
critical depth. With the baseflow one of the flows, curve2flood reads the held water at it, not a line from the flows
either side of it.

Each cross section has a table of LEVELS water surfaces, from its bed to well above the top of its own rating curve:
it brackets each step's water surface (or, where the water below stands higher than the table reaches, the search goes
on above it), and gives the critical depth, and an outlet's normal depth where wse_for_discharge has none. Each water
surface is then found with the exact hydraulics, so that no interpolation error builds up from one step to the next.
"""
from __future__ import annotations

import math
from collections import deque
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
EDDY_LENGTH = 1000.0  # metres: the eddy loss is reckoned on the change in velocity head over this far downstream


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
            wse, q_up, energy_down, eddy_head, eddy_steps, conveyance_down, flow_sum, length, slope_factor):
    """The energy at a water surface of the upstream cross section, less what the energy equation needs there."""
    area, conveyance, alpha = _state(stations, elevations, mannings_n, center, left_bank, right_bank, wse, k_decay,
                                     shallow_factor, deep_factor)
    head = alpha * (q_up / area) ** 2 / (2.0 * G)
    coefficient = CONTRACTION if eddy_head > head else EXPANSION
    friction = (flow_sum / max((conveyance_down + conveyance) * slope_factor, 1e-9)) ** 2
    return wse + head - (energy_down + length * friction + coefficient * abs(head - eddy_head) / eddy_steps)


@njit(cache=True, error_model="numpy")
def _root(stations, elevations, mannings_n, center, left_bank, right_bank, k_decay, shallow_factor, deep_factor,
          low, high, q_up, energy_down, eddy_head, eddy_steps, conveyance_down, flow_sum, length, slope_factor):
    """The water surface between low (energy short) and high (enough) where the energy equation holds, by the
    Illinois variant of the false position method, to TOLERANCE."""
    f_low = _excess(stations, elevations, mannings_n, center, left_bank, right_bank, k_decay, shallow_factor,
                    deep_factor, low, q_up, energy_down, eddy_head, eddy_steps, conveyance_down, flow_sum, length,
                    slope_factor)
    f_high = _excess(stations, elevations, mannings_n, center, left_bank, right_bank, k_decay, shallow_factor,
                     deep_factor, high, q_up, energy_down, eddy_head, eddy_steps, conveyance_down, flow_sum, length,
                     slope_factor)
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
                    slope_factor)
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


def velocity_head(table: HydraulicTable, wse: float, q: float) -> float:
    """The velocity head, a V^2 / 2g, of q at a water surface, worked out exactly."""
    area, _, alpha = table.state(wse)
    return alpha * (q / area) ** 2 / (2.0 * G)


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
    alone, as the energy equation above has it."""
    if wse_down < down.critical(q_down):
        # The water below is supercritical, and no subcritical water surface steps up from it
        return max(up.critical(q_up), floor), True
    area_d, conveyance_d, alpha_d = down.state(wse_down)
    head_d = alpha_d * (q_down / area_d) ** 2 / (2.0 * G)
    if eddy_head is None:
        eddy_head, eddy_steps = head_d, 1
    eddy_steps = max(int(eddy_steps), 1)
    head = up.alpha * (q_up / up.area) ** 2 / (2.0 * G)
    energy = up.levels + head
    loss = np.where(eddy_head > head, CONTRACTION, EXPANSION) * np.abs(head - eddy_head) / eddy_steps
    friction = ((q_up + q_down) / np.maximum((conveyance_d + up.conveyance) * slope_factor, 1e-9)) ** 2
    excess = energy - (wse_down + head_d + length * friction + loss)
    critical = float(up.levels[int(np.argmin(energy))])
    arguments = (float(q_up), wse_down + head_d, float(eddy_head), float(eddy_steps), conveyance_d,
                 float(q_up + q_down), float(length), float(slope_factor))
    lowest = max(critical, float(floor))
    if _excess(*up.section, *up.parameters, lowest, *arguments) >= 0.0:
        return lowest, lowest == critical
    above = np.flatnonzero((excess >= 0.0) & (up.levels > lowest))
    if above.size:
        i1 = int(above[0])
        low, high = max(float(up.levels[i1 - 1]), lowest), float(up.levels[i1])
    else:
        # The water below stands higher than the table reaches, backing water up beyond it: look higher
        low = max(float(up.levels[-1]), lowest)
        rise = max(wse_down + head_d - low, 0.0) + 0.5
        high = low + rise
        for _ in range(60):
            if _excess(*up.section, *up.parameters, high, *arguments) >= 0.0:
                break
            low, rise = high, 2.0 * rise
            high = low + rise
        else:
            return lowest, lowest == critical
    return float(_root(*up.section, *up.parameters, low, high, *arguments)), False


class StepSummary(NamedTuple):
    """What apply_standard_step did: the cells it rewrote, how many steps it took and went critical, and how many
    water surfaces (a cell at a flow) it held at the baseflow level."""
    cells: int
    steps: int
    critical: int
    held: int = 0


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
                        baseflow_levels: np.ndarray | None = None) -> StepSummary:
    """Rewrite the rating curves of the cells that have them (see the notes above), in place.

    rows, cols and reaches are each cell's stream cell and reach, and sections its cross section (anything with an
    `xs` and a `usable` flag, or None). network runs from each reach to the one downstream (reach_network), and must
    hold every reach with a rated cell. eddy_length is how far downstream the eddy loss looks (see the notes above);
    0 charges each step the change from the cross section below alone. baseflow_levels is each cell's baseflow
    level, the water surface its channel was carved to carry its baseflow at, or NaN (and None) for none: at every
    flow up to its baseflow, its water is held at most there.

    A reach whose neighbours have no rated cells is ordered by their cells on the stream raster (stream_cells, each
    reach's rows and columns), or by the cell where its line ends (outlets, each reach's (row, col)), and failing
    both with its higher end upstream, by elevations (each cell's stream cell's, NaN for none) or else its cross
    sections' beds (see order_reach).
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
    first_cell: dict[int, tuple[int, int]] = {}  # and that cell
    levels_all = np.full(len(sections), math.nan) if baseflow_levels is None else \
        np.asarray(baseflow_levels, dtype=np.float64)
    steps = critical_steps = held = 0
    for reach in downstream_first:
        ks = np.array(by_reach[reach])
        heights = np.asarray(elevations, dtype=np.float64)[ks] if elevations is not None else \
            np.array([float(sections[k].xs.elevations[sections[k].xs.elevations.size // 2]) for k in ks])
        order, stations = order_reach(network, reach, *cells[reach], dx, dy, cells, heights,
                                      stream_cells=stream_cells, outlets=outlets)
        ks, stations = ks[order], np.asarray(stations, dtype=np.float64)
        tables = [HydraulicTable(sections[k].xs, float(np.nanmax(curves.increments[k, :, WSE])), roughness)
                  for k in ks]
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
        wse = np.empty((ks.size, increments))
        heads = np.empty((ks.size, increments))
        far = _eddy_ends(stations, eddy_length)
        for j in range(increments):
            last = tables[-1]
            if below is not None:
                start = upstream_wse[below][j]
            else:
                start = wse_for_discharge(sections[ks[-1]].xs, flows[-1, j], float(slopes[ks[-1]]),
                                          roughness=roughness, slope_factor=slope_factor)
                if not math.isfinite(start):
                    start = last.normal(flows[-1, j], float(slopes[ks[-1]]), slope_factor)
                start = min(start, cap[j])
            # No cross section's water falls as the flow rises
            floors = wse[:, j - 1] if j > 0 else np.full(ks.size, -math.inf)
            wse[-1, j] = max(start, last.critical(flows[-1, j]), floors[-1])
            if holds[-1, j] and wse[-1, j] > levels[-1]:
                wse[-1, j] = max(levels[-1], floors[-1])
                held += 1
            heads[-1, j] = velocity_head(last, wse[-1, j], flows[-1, j])
            for m in range(ks.size - 2, -1, -1):
                end = far[m]
                wse[m, j], critical = step_up(tables[m + 1], wse[m + 1, j], flows[m + 1, j], tables[m], flows[m, j],
                                              abs(stations[m + 1] - stations[m]), slope_factor, floors[m],
                                              eddy_head=heads[end, j], eddy_steps=end - m)
                if holds[m, j] and wse[m, j] > levels[m]:
                    # The baseflow stays in its channel: the energy over its level is lost here
                    wse[m, j], critical = max(levels[m], floors[m]), False
                    held += 1
                heads[m, j] = velocity_head(tables[m], wse[m, j], flows[m, j])
                steps += 1
                critical_steps += int(critical)
        upstream_wse[reach] = wse[0]
        first_cell[reach] = (int(rows[ks[0]]), int(cols[ks[0]]))
        for i, k in enumerate(ks):
            _rewrite(curves.increments[k], sections[k].xs, flows[i], wse[i], roughness)
    return StepSummary(len(rated), steps, critical_steps, held)


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


def _eddy_ends(stations: np.ndarray, eddy_length: float) -> np.ndarray:
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
