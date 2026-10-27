"""Filling in and smoothing the channel depths along each reach.

This step comes between working out each cross section's depth (bathymetry_depth) and carving it (carve_channel).
smooth_channel_depths does it all, taking each reach's cross sections from upstream to downstream in the order the bank
smoothing put them in (smooth_bank_elevations):

1. A depth that isn't above 0 and finite is replaced by the one before it along the stream. The first in a reach
   takes the shallowest last depth of the reaches flowing into it, or failing that 0.5 m (fill_reach_depths).
2. The bed, each cross section's reference level minus its depth, is smoothed with a running median of five cross
   sections (fewer at a reach's ends). With the bed cap on (Bathy_Bed_Cap), it then rises or falls no more than
   MAX_BED_GRADE metres per metre along the stream, starting from the lowest bed flowing into the reach
   (smooth_reach_bed). The depth carved is the reference level's height above the smoothed bed, or 0 where the bed is
   above it.

With Bathy_Smooth_Bed off, the pipeline passes a window of 1 and no cap: only the depths are filled in, and each cross
section keeps its own bed.

The reference level is the smoothed bank elevation with Bathy_Use_Banks, and otherwise the stream cell's elevation.
The cap is `MAX_SLOPE`, so a bed may be as steep as a stream. A missing bed (no reference level) is
left out of the median and the cap, and its cross section keeps its depth.
"""
from __future__ import annotations

import math
from typing import Mapping, NamedTuple, Sequence

import networkx as nx
import numpy as np
from numba import njit

from arc.bathymetry.smoothing import ReachSections, SmoothedReach
from arc.xsection.slope import MAX_SLOPE

DEFAULT_DEPTH = 0.5  # the depth filled in with none before it to take
MEDIAN_WINDOW = 5  # how many cross sections the running median of the bed takes in
MAX_BED_GRADE = MAX_SLOPE  # the bed cap: how much the bed may rise or fall per metre along the stream
SHORTEST_STEP = 0.1  # metres: the cap allows at least this much distance between cross sections


class ChannelDepths(NamedTuple):
    """A reach's depths after filling in and smoothing, in the order its cross sections were given."""
    depths: np.ndarray  # the depths to carve
    beds: np.ndarray  # the smoothed bed elevations, NaN where the reference level is missing
    filled: np.ndarray  # the depths after filling in, before smoothing


def _upstream_to_downstream(network: nx.DiGraph) -> list:
    try:
        return list(nx.topological_sort(network))
    except nx.NetworkXUnfeasible:
        raise ValueError("The stream network has a loop, so its reaches can't be put in order downstream.") from None


def fill_reach_depths(network: nx.DiGraph, depths: Mapping[int, Sequence[float]],
                      orders: Mapping[int, np.ndarray]) -> dict[int, np.ndarray]:
    """Each reach's depths, in the order given, with those that aren't above 0 and finite (NaN included) filled in.

    `orders` holds each reach's cross sections from upstream to downstream, as indices into its depths. Going
    downstream, a depth is filled in with the one before it, the first in a reach with the shallowest last depth of the
    reaches flowing into it (those with cross sections), or with neither, 0.5 m.
    """
    _check_reaches(network, depths)
    filled: dict[int, np.ndarray] = {}
    last_depths: dict[int, float] = {}
    for reach in _upstream_to_downstream(network):
        if reach not in depths:
            continue

        values = np.array(depths[reach], dtype=np.float64)
        order = np.asarray(orders[reach], dtype=np.int64)
        if order.size != values.size:
            raise ValueError(f"Reach {reach} has {values.size} depths but {order.size} cross sections in order.")
        
        inflows = [last_depths[p] for p in network.predecessors(reach) if p in last_depths]
        previous = min(inflows) if inflows else DEFAULT_DEPTH
        for i in order:
            if not 0.0 < values[i] < math.inf:
                values[i] = previous
            previous = values[i]

        if order.size > 0:
            last_depths[reach] = previous
        filled[reach] = values

    return filled


@njit(cache=True, error_model="numpy")
def _smooth_bed(beds, stations, window, max_grade, inflow_bed, inflow_distance):
    n = beds.size
    pad = window // 2
    smoothed = np.full(n, np.nan)
    values = np.empty(2 * pad + 1)
    for i in range(n):
        if math.isnan(beds[i]):
            continue

        count = 0
        for k in range(max(i - pad, 0), min(i + pad + 1, n)):
            if not math.isnan(beds[k]):
                values[count] = beds[k]
                count += 1
        smoothed[i] = np.median(values[:count])

    if not max_grade >= 0.0:
        return smoothed

    last = -1  # the last cross section with a bed
    for i in range(n):
        if math.isnan(smoothed[i]):
            continue

        if last >= 0:
            limit = max_grade * max(stations[i] - stations[last], SHORTEST_STEP)
            smoothed[i] = min(max(smoothed[i], smoothed[last] - limit), smoothed[last] + limit)
        elif not math.isnan(inflow_bed):
            limit = max_grade * max(inflow_distance + (stations[i] - stations[0]), SHORTEST_STEP)
            smoothed[i] = min(max(smoothed[i], inflow_bed - limit), inflow_bed + limit)
        last = i

    return smoothed


def smooth_reach_bed(beds, stations, *, window: int = MEDIAN_WINDOW, max_bed_grade: float | None = MAX_BED_GRADE,
                     inflow_bed: float | None = None, inflow_distance: float = 0.0) -> np.ndarray:
    """A reach's bed elevations, in order from upstream to downstream, smoothed.

    Each bed becomes the median of those within window // 2 cross sections of it. A cross section without a bed
    (NaN) is left out, and stays NaN. With max_bed_grade, the beds then change from each to the next by no more than
    that times the distance between them along the stream (stations, in metres), and at least `SHORTEST_STEP` apart.
    The first is held that way to `inflow_bed`, the lowest bed flowing into the reach, `inflow_distance` upstream of the
    first cross section.
    """
    beds = np.asarray(beds, dtype=np.float64)
    stations = np.asarray(stations, dtype=np.float64)
    if beds.shape != stations.shape or beds.ndim != 1:
        raise ValueError("The beds and stations must be 1-d and the same length.")
    
    if int(window) < 1:
        raise ValueError(f"The median window must be at least 1 cross section, not {window!r}.")
    
    grade = math.nan if max_bed_grade is None else float(max_bed_grade)
    if grade < 0.0:
        raise ValueError(f"max_bed_grade can't be negative ({max_bed_grade!r}).")
    
    return _smooth_bed(beds, stations, int(window), grade, math.nan if inflow_bed is None else float(inflow_bed),
                       float(inflow_distance))


def smooth_channel_depths(network: nx.DiGraph, reaches: Mapping[int, ReachSections],
                          smoothed: Mapping[int, SmoothedReach], depths: Mapping[int, Sequence[float]], dx: float,
                          dy: float, *, use_banks: bool, max_bed_grade: float | None = MAX_BED_GRADE,
                          window: int = MEDIAN_WINDOW) -> dict[int, ChannelDepths]:
    """Fill in and smooth every reach's depths along the network (see the notes above).

    `reaches` and `smoothed` are what `smooth_bank_elevations` took and gave, and `depths` holds each reach's depths from
    `bathymetry_depth`, in the same order. `use_banks` is `Bathy_Use_Banks`: whether the reference level is the smoothed
    bank elevation or the stream cell's. `max_bed_grade=None` turns the bed cap off.
    """
    missing = sorted(reach for reach in depths if reach not in reaches or reach not in smoothed)
    if missing:
        raise ValueError("These reaches have depths but no cross sections or bank smoothing: "
                         + ", ".join(map(str, missing[:10])) + ("..." if len(missing) > 10 else "") + ".")
    
    for reach, values in depths.items():
        if len(values) != len(reaches[reach].sections) or len(values) != smoothed[reach].order.size:
            raise ValueError(f"Reach {reach} has {len(values)} depths but {len(reaches[reach].sections)} cross "
                             f"sections.")
        
    filled = fill_reach_depths(network, depths, {reach: smoothed[reach].order for reach in depths})

    results: dict[int, ChannelDepths] = {}
    ends: dict[int, tuple[float, int, int]] = {}  # each reach's last bed and its stream cell
    for reach in _upstream_to_downstream(network):
        if reach not in depths:
            continue

        sections, order, stations = reaches[reach], smoothed[reach].order, smoothed[reach].stations
        if use_banks:
            references = np.asarray(smoothed[reach].bank_elevations, dtype=np.float64)
        else:
            references = np.array([float(xs.elevations[xs.elevations.size // 2]) for xs in sections.sections])

        rows = np.asarray(sections.rows, dtype=np.int64)[order]
        cols = np.asarray(sections.cols, dtype=np.int64)[order]

        inflow_bed, inflow_distance = None, 0.0
        inflows = [ends[p] for p in network.predecessors(reach) if p in ends]
        if inflows and order.size > 0:
            bed, row, col = min(inflows, key=lambda end: end[0])
            inflow_bed = bed
            inflow_distance = math.hypot((cols[0] - col) * dx, (rows[0] - row) * dy)

        beds = smooth_reach_bed(references[order] - filled[reach][order], stations, window=window,
                                max_bed_grade=max_bed_grade, inflow_bed=inflow_bed, inflow_distance=inflow_distance)
        with_bed = np.flatnonzero(~np.isnan(beds))
        if with_bed.size > 0:
            last = with_bed[-1]
            ends[reach] = (float(beds[last]), int(rows[last]), int(cols[last]))

        reach_beds = np.full(order.size, math.nan)
        reach_beds[order] = beds
        carved = filled[reach].copy()
        known = ~np.isnan(reach_beds) & ~np.isnan(references)
        carved[known] = np.maximum(references[known] - reach_beds[known], 0.0)
        results[reach] = ChannelDepths(carved, reach_beds, filled[reach])
        
    return results


def _check_reaches(network: nx.DiGraph, reaches) -> None:
    missing = sorted(reach for reach in reaches if reach not in network)
    if missing:
        raise ValueError("These reaches have cross sections but aren't in the stream network: "
                         + ", ".join(map(str, missing[:10])) + ("..." if len(missing) > 10 else "") + ".")
