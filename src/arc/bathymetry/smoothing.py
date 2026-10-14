"""Smoothing the bank elevations along each reach.

The smoothed bank elevation is the level a channel is carved below with Bathy_Use_Banks (see bathymetry_depth and
carve_channel). smooth_bank_elevations does it all, a reach at a time:

1. Widths: a channel narrower than its reach's 25th percentile width or wider than its 75th, or one without valid
   banks, gets banks for the reach's median width instead, unless its banks are where the water ends in the water
   mask or the land cover (filter_reach_widths).
2. Observations: each cross section's lower bank above its stream cell is an observation of the bank elevation. With
   four or more in a reach, those below its 2nd percentile or above its 97th are left out (reach_bank_observations).
3. Order: the cross sections are put in order from upstream to downstream (order_reach).
4. Levels: the bank elevation is the stream cells' elevations, fitted never to rise downstream, plus the bank's
   height: the 10th percentile of the observations' heights above their stream cells within 500 m along the reach,
   or the reach's, or the whole site's where there are none. The sum is fitted never to rise again, and is never
   below the stream cell (reach_bank_levels).

So the bank elevation follows the shape of the stream, and only its height comes from the banks: a few low banks can't
pull a whole reach down. Only a site without any observation gets no bank elevations (NaN).

With smooth=False (Bathy_Smooth_Banks off), each cross section keeps its own banks and its own observation as its bank
elevation, none left out. One without an observation takes its stream cell's elevation plus the reach's bank height
(the 10th percentile of its observations' heights), or the whole site's. The cross sections are still put in order.
"""
from __future__ import annotations

import math
from typing import Mapping, NamedTuple, Sequence

import networkx as nx
import numpy as np
from numba import njit

from arc.bathymetry.banks import Banks, bank_control_elevation, banks_for_width
from arc.xsection.stream_path import along_stream_stations, downstream_order
from arc.xsection.xsection import XSection

OUTLIER_PERCENTILES = (2, 97)  # observations outside these percentiles of their reach's are left out
ORIENTING_CELLS = 10  # how many cross sections at each end decide which end of an unconnected reach is upstream
BANK_HEIGHT_QUANTILE = 0.1  # the bank's height above the water: this quantile of the observations' heights...
BANK_HEIGHT_WINDOW = 500.0  # ...within this many metres either side along the reach


class ReachWidths(NamedTuple):
    """The percentiles of a reach's valid channel widths, which filter_reach_widths works from."""
    q25: float
    median: float
    q75: float


class ReachSections(NamedTuple):
    """A reach's cross sections: each one's stream cell, the section, and its banks from find_banks."""
    rows: np.ndarray
    cols: np.ndarray
    sections: Sequence[XSection]
    banks: Sequence[Banks]


class SmoothedReach(NamedTuple):
    """A reach after smoothing. The arrays are in the order the cross sections were given, except order and
    stations, which are in order from upstream to downstream."""
    banks: list[Banks]  # after the width filter (as given, unsmoothed)
    bank_elevations: np.ndarray  # the smoothed bank elevations (each cross section's own), NaN if there are none
    observations: np.ndarray  # each cross section's observation, NaN for none or an outlier
    order: np.ndarray  # the cross sections from upstream to downstream
    stations: np.ndarray  # how far each is along the stream from the upstream one, in metres
    widths: ReachWidths | None  # None if no cross section had valid banks


# --- One reach ----------------------------------------------------------------------------------------------------


def filter_reach_widths(sections: Sequence[XSection], banks: Sequence[Banks]) -> tuple[list[Banks], ReachWidths | None]:
    """A reach's banks with its outlying widths replaced, and the percentiles of its valid widths.

    A valid channel narrower than the 25th percentile or wider than the 75th, and a cross section without valid
    banks, get banks for the median width (banks_for_width), or for as much of it as the cross section holds. Where
    that isn't possible (the cross section is off the raster on one side of the stream cell), the banks stay as they
    were. Banks found where the water ends in the water mask or the land cover stay as they are too: the water shows
    the channel's width, where the DEM's banks are only an estimate of it. They count towards the percentiles all the
    same. The percentiles are None, and the banks unchanged, if no cross section has valid banks.
    """
    if len(sections) != len(banks):
        raise ValueError(f"There are {len(sections)} cross sections but {len(banks)} banks.")
    banks = list(banks)
    widths = np.array([b.top_width for b in banks if b.valid and math.isfinite(b.top_width) and b.top_width > 0.0])
    if widths.size == 0:
        return banks, None
    q25, median, q75 = (float(p) for p in np.percentile(widths, [25, 50, 75]))
    for k, (xs, b) in enumerate(zip(sections, banks)):
        if b.valid and b.method == "land_cover":
            continue
        if not b.valid or b.top_width < q25 or b.top_width > q75:
            rebuilt = banks_for_width(xs, median)
            if rebuilt.valid:
                banks[k] = rebuilt
    return banks, ReachWidths(q25, median, q75)


def reach_bank_observations(sections: Sequence[XSection], banks: Sequence[Banks]) -> tuple[np.ndarray, float, float]:
    """Each of a reach's cross sections' observation of its bank elevation, and the bounds outside which they were
    left out.

    An observation is the lower of a cross section's valid banks above its stream cell (bank_control_elevation), or
    NaN. With four or more, those below the 2nd percentile or above the 97th become NaN too, and those are the
    bounds; otherwise the bounds are infinite.
    """
    observations = np.array([bank_control_elevation(b, float(xs.elevations[xs.elevations.size // 2])) if b.valid
                             else math.nan for xs, b in zip(sections, banks)], dtype=np.float64)
    finite = np.isfinite(observations)
    if np.count_nonzero(finite) < 4:
        return observations, -math.inf, math.inf
    lower, upper = (float(p) for p in np.percentile(observations[finite], OUTLIER_PERCENTILES))
    observations[finite & ((observations < lower) | (observations > upper))] = math.nan
    return observations, lower, upper


def order_reach(network: nx.DiGraph, reach: int, rows, cols, dx: float, dy: float,
                cells: Mapping[int, tuple[np.ndarray, np.ndarray]], observations) -> tuple[np.ndarray, np.ndarray]:
    """A reach's cross sections in order from upstream to downstream, and their stations in metres from the upstream
    one.

    The downstream end is nearest the first reach downstream that has cross sections, or, failing that, farthest
    along the stream from the one nearest any reach upstream (see downstream_order). cells holds each reach's cross
    sections' stream cells. A reach with neither is ordered from one end of its cells to the other, then turned round
    to put the higher end upstream: an unconnected reach's highest observation before its lowest, or else whichever
    end's first ten cross sections have the higher mean observation.
    """
    rows = np.asarray(rows, dtype=np.int64)
    cols = np.asarray(cols, dtype=np.int64)
    downstream = next((cells[s] for s in network.successors(reach) if s in cells and len(cells[s][0]) > 0), None)
    upstream = [cells[p] for p in network.predecessors(reach) if p in cells]
    upstream_cells = (np.concatenate([c[0] for c in upstream]), np.concatenate([c[1] for c in upstream])) \
        if upstream else None
    ordered = downstream_order(rows, cols, dx, dy, downstream_cells=downstream, upstream_cells=upstream_cells)
    if ordered is not None:
        return ordered

    stations = along_stream_stations(rows, cols, dx, dy)
    order = np.argsort(stations, kind="stable")
    stations = stations[order]
    if _higher_end_is_downstream(np.asarray(observations, dtype=np.float64)[order],
                                 network.in_degree(reach) == 0 and network.out_degree(reach) == 0):
        order, stations = order[::-1].copy(), stations[-1] - stations[::-1]
    return order, stations


def _higher_end_is_downstream(ordered_observations: np.ndarray, unconnected: bool) -> bool:
    finite = np.flatnonzero(np.isfinite(ordered_observations))
    if unconnected and finite.size >= 2:
        highest = finite[np.argmax(ordered_observations[finite])]
        lowest = finite[np.argmin(ordered_observations[finite])]
        if highest != lowest:
            return bool(highest > lowest)
    # The ten observations nearest each end, whichever cross sections they're at
    observed = ordered_observations[finite]
    count = min(ORIENTING_CELLS, observed.size)
    return bool(count > 0 and observed[:count].mean() < observed[-count:].mean())


# --- The network --------------------------------------------------------------------------------------------------


def reach_network(reach_ids, downstream_ids) -> nx.DiGraph:
    """The stream network from the stream layer's reach and downstream reach IDs: an edge from each reach to the one
    downstream of it.

    A reach's first row counts. A downstream ID that is missing (None, NaN or ''), the reach itself, or not a reach in
    the table makes it an outlet.
    """
    records: dict[int, int | None] = {}
    for reach, downstream in zip(reach_ids, downstream_ids):
        reach = _reach_id(reach)
        if reach is None or reach in records:
            continue
        downstream = _reach_id(downstream)
        records[reach] = None if downstream == reach else downstream
    if not records:
        raise ValueError("The stream network has no reaches.")
    network = nx.DiGraph()
    network.add_nodes_from(records)
    for reach, downstream in records.items():
        if downstream is not None and downstream in records:
            network.add_edge(reach, downstream)
    return network


def _reach_id(value) -> int | None:
    if value is None or (isinstance(value, str) and value.strip() == ""):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return int(number) if math.isfinite(number) else None


# --- The bank level along a reach: the water surface plus the banks' height --------------------------------------


@njit(cache=True, error_model="numpy")
def _falling_fit(values):
    n = values.size
    sums = np.empty(n)
    counts = np.empty(n, np.int64)
    top = 0
    for k in range(n):
        if not math.isfinite(values[k]):
            continue
        sums[top], counts[top] = values[k], 1
        top += 1
        # a rise downstream: pool the two runs at their mean, and look again
        while top > 1 and sums[top - 2] / counts[top - 2] < sums[top - 1] / counts[top - 1]:
            sums[top - 2] += sums[top - 1]
            counts[top - 2] += counts[top - 1]
            top -= 1
    fitted = np.full(n, np.nan)
    block, left = 0, counts[0] if top > 0 else 0
    for k in range(n):
        if not math.isfinite(values[k]):
            continue
        fitted[k] = sums[block] / counts[block]
        left -= 1
        if left == 0 and block + 1 < top:
            block += 1
            left = counts[block]
    return fitted


def falling_fit(values) -> np.ndarray:
    """The sequence that never rises closest to values in least squares (pool-adjacent-violators), in order; a NaN
    is left out and stays NaN."""
    return _falling_fit(np.asarray(values, dtype=np.float64))


@njit(cache=True, error_model="numpy")
def _running_quantile(values, stations, quantile, half_window):
    n = values.size
    out = np.full(n, np.nan)
    part = np.empty(n)
    for i in range(n):
        count = 0
        for j in range(n):
            if abs(stations[j] - stations[i]) <= half_window and math.isfinite(values[j]):
                part[count] = values[j]
                count += 1
        if count > 0:
            out[i] = np.quantile(part[:count], quantile)
    return out


def reach_bank_levels(thalwegs, observations, stations, fallback_height: float, *,
                      quantile: float = BANK_HEIGHT_QUANTILE, half_window: float = BANK_HEIGHT_WINDOW) -> np.ndarray:
    """A reach's bank elevations, its cross sections in order from upstream to downstream (see the notes above).

    thalwegs are the stream cells' elevations, observations the cross sections' observations (NaN for none) and
    stations their distances along the reach in metres. The water surface is the thalwegs fitted never to rise
    (falling_fit). Each cross section's bank height is the quantile of the observations' heights above their stream
    cells within half_window metres, or the reach's quantile, or failing that fallback_height (the site's); the bank
    elevation is the water surface plus that, fitted never to rise, and never below the stream cell. All NaN if the
    reach has no height to take.
    """
    thalwegs = np.asarray(thalwegs, dtype=np.float64)
    observations = np.asarray(observations, dtype=np.float64)
    stations = np.asarray(stations, dtype=np.float64)
    heights = observations - thalwegs
    heights[~(heights > 0.0)] = math.nan
    height = _running_quantile(heights, stations, float(quantile), float(half_window))
    reach_height = float(np.quantile(heights[np.isfinite(heights)], quantile)) if np.isfinite(heights).any() \
        else float(fallback_height)
    height = np.where(np.isfinite(height), height, reach_height)
    if not np.isfinite(height).all():
        return np.full(thalwegs.size, math.nan)
    known = np.isfinite(thalwegs)
    water = np.interp(stations, stations[known], _falling_fit(thalwegs[known])) if known.any() else thalwegs
    return np.maximum(_falling_fit(water + height), np.where(known, thalwegs, -math.inf))


# --- Everything ---------------------------------------------------------------------------------------------------


def smooth_bank_elevations(network: nx.DiGraph, reaches: Mapping[int, ReachSections], dx: float,
                           dy: float, *, smooth: bool = True) -> dict[int, SmoothedReach]:
    """Smooth the bank elevations of every reach's cross sections (see the notes above), or with smooth=False, give
    each its own.

    network runs from each reach to the one downstream of it (reach_network), and orders each reach's cross sections.
    Every reach with cross sections must be in it. dx and dy are the cells' width and height in metres.
    """
    missing = sorted(reach for reach in reaches if reach not in network)
    if missing:
        raise ValueError("These reaches have cross sections but aren't in the stream network: "
                         + ", ".join(map(str, missing[:10])) + ("..." if len(missing) > 10 else "") + ".")
    cells = {reach: (np.asarray(r.rows, dtype=np.int64), np.asarray(r.cols, dtype=np.int64))
             for reach, r in reaches.items() if len(r.sections) > 0}

    prepared = {}
    for reach, sections in reaches.items():
        if len(sections.sections) == 0:
            continue
        thalwegs = np.array([float(xs.elevations[xs.elevations.size // 2]) for xs in sections.sections])
        if smooth:
            banks, widths = filter_reach_widths(sections.sections, sections.banks)
            observations, _, _ = reach_bank_observations(sections.sections, banks)
        else:
            banks, widths = list(sections.banks), None
            observations = np.array([bank_control_elevation(b, z) if b.valid else math.nan
                                     for b, z in zip(banks, thalwegs)], dtype=np.float64)
        order, stations = order_reach(network, reach, sections.rows, sections.cols, dx, dy, cells, observations)
        prepared[reach] = (banks, widths, observations, order, stations, thalwegs)

    # The whole site's bank height, for reaches without observations of their own
    heights = [observations - thalwegs for _, _, observations, _, _, thalwegs in prepared.values()]
    heights = np.concatenate(heights) if heights else np.empty(0)
    heights = heights[heights > 0.0]
    site_height = float(np.quantile(heights, BANK_HEIGHT_QUANTILE)) if heights.size else math.nan

    smoothed = {}
    for reach, (banks, widths, observations, order, stations, thalwegs) in prepared.items():
        if smooth:
            bank_elevations = np.full(observations.size, math.nan)
            bank_elevations[order] = reach_bank_levels(thalwegs[order], observations[order], stations, site_height)
        else:
            reach_heights = observations - thalwegs
            reach_heights = reach_heights[reach_heights > 0.0]
            height = float(np.quantile(reach_heights, BANK_HEIGHT_QUANTILE)) if reach_heights.size else site_height
            bank_elevations = np.where(np.isfinite(observations), observations, thalwegs + height)
        smoothed[reach] = SmoothedReach(banks, bank_elevations, observations, order, stations, widths)
    return smoothed
