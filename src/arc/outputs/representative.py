"""Representative cross sections: each reach's hydraulics averaged over its cross sections, stage by stage.

Every 0.10 m above each cross section's stream cell, up to 25 m, the cross section's area, wetted perimeter, top
width, discharge and velocity are worked out on its own profile, a carved channel's exact shape included. With banks,
the velocity is the channel's (its discharge over its area); without them it's the whole section's. At each stage,
the cross sections whose area, velocity and top width are all within two standard deviations of the reach's means are
averaged, and a reach stops at the first stage where any cross section's hydraulics aren't finite. The
representative cross section is then built from the averages: its area and top width kept from falling as the stage
rises, and its depth at each stage the one that grows the width linearly from the stage below to give that area.

The means are summed as numpy sums them, so they are numpy's nanmean and nanstd to the bit.
"""
from __future__ import annotations

import os
from typing import NamedTuple, Sequence

import numpy as np
import pandas as pd
from numba import njit

from arc import LOG
from arc.hydraulics import DepthRoughness, _check_roughness, _check_slope_factor, hydraulic_profile
from arc.outputs.tables import write_table
from arc.rating_curve import section_hydraulics
from arc.xsection.xsection import XSection

DEPTH_INCREMENT = 0.10  # metres between stages
MAX_DEPTH = 25.0  # the highest stage above the stream cell

REPRESENTATIVE_CROSS_SECTION_COLUMNS = [
    'COMID', 'Cross_Section_Count', 'Hydraulic_Sample_Count', 'Depth_Stage_Index', 'Depth_Stage_Meters',
    'Stream_Slope', 'Reach_Inflect_Terrace_Depth', 'Representative_Thalweg_Elevation', 'Mean_Discharge', 'Mean_Depth',
    'Mean_Velocity', 'Representative_Velocity', 'Mean_Top_Width', 'Mean_Cross_Sectional_Area', 'Mean_WSE',
    'Representative_Cross_Sectional_Area', 'Representative_Depth_Increment', 'Representative_Depth',
    'Representative_Top_Width', 'Representative_Stage_Elevation', 'Representative_Left_Station',
    'Representative_Right_Station',
]
_INTEGER_COLUMNS = ['COMID', 'Cross_Section_Count', 'Hydraulic_Sample_Count', 'Depth_Stage_Index']


class RepresentativeSample(NamedTuple):
    """One cross section of a reach, with its banks set on it for the channel velocity."""
    comid: int
    xs: XSection
    thalweg: float
    slope: float


def monotonic_cumulative_max(values: np.ndarray) -> np.ndarray:
    """values made non-decreasing, skipping NaNs."""
    result = np.asarray(values, dtype=np.float64).copy()
    running_max = -np.inf
    for i in range(result.size):
        if np.isnan(result[i]):
            continue

        if result[i] < running_max:
            result[i] = running_max
        else:
            running_max = result[i]

    return result


def depths_from_width_and_area(widths: np.ndarray, areas: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Each stage's depth increment and depth, growing the width linearly between stages to give each area."""
    increments = np.zeros(widths.size)
    depths = np.zeros(widths.size)
    previous_width = previous_area = previous_depth = 0.0
    for i in range(widths.size):
        width, area = float(widths[i]), float(areas[i])
        delta_area = max(area - previous_area, 0.0)
        denominator = previous_width + width
        delta = (2.0 * delta_area) / denominator if denominator > 0.0 and delta_area > 0.0 else 0.0
        increments[i] = delta
        depths[i] = previous_depth + delta
        previous_width, previous_area, previous_depth = width, area, depths[i]

    return increments, depths


@njit(cache=True, error_model="numpy")
def _block_sum(values, start, count):
    """values[start:start + count], at most 128 of them, summed as numpy sums a block: in 8 interleaved sums."""
    if count < 8:
        total = 0.0
        for i in range(start, start + count):
            total += values[i]
        return total
    
    s0, s1, s2, s3 = values[start], values[start + 1], values[start + 2], values[start + 3]
    s4, s5, s6, s7 = values[start + 4], values[start + 5], values[start + 6], values[start + 7]
    i = start + 8
    blocks_end = start + count - count % 8
    while i < blocks_end:
        s0 += values[i]
        s1 += values[i + 1]
        s2 += values[i + 2]
        s3 += values[i + 3]
        s4 += values[i + 4]
        s5 += values[i + 5]
        s6 += values[i + 6]
        s7 += values[i + 7]
        i += 8

    total = ((s0 + s1) + (s2 + s3)) + ((s4 + s5) + (s6 + s7))
    while i < start + count:
        total += values[i]
        i += 1

    return total


@njit(cache=True, error_model="numpy")
def _sum(values, start, count):
    """values[start:start + count] summed as numpy sums float64: halved, the halves' lengths kept to multiples of 8,
    down to blocks of at most 128 (_block_sum), and the halves' sums added up. The means and deviations below are
    numpy's to the bit. The halves are worked through with a stack, as numba can't load recursion from its cache."""
    if count <= 128:
        return _block_sum(values, start, count)
    
    # Each half is stacked to be summed, and the whole it came from to add up their sums once they are
    node_starts = np.empty(256, np.int64)
    node_counts = np.empty(256, np.int64)
    halves_summed = np.empty(256, np.bool_)
    sums = np.empty(128)
    node_starts[0], node_counts[0], halves_summed[0] = start, count, False
    nodes, summed = 1, 0
    while nodes > 0:
        nodes -= 1
        first, length = node_starts[nodes], node_counts[nodes]
        if length <= 128:
            sums[summed] = _block_sum(values, first, length)
            summed += 1
        elif halves_summed[nodes]:
            summed -= 1
            sums[summed - 1] = sums[summed - 1] + sums[summed]
        else:
            half = length // 2
            half -= half % 8
            halves_summed[nodes] = True
            node_starts[nodes + 1], node_counts[nodes + 1], halves_summed[nodes + 1] = first + half, length - half, False
            node_starts[nodes + 2], node_counts[nodes + 2], halves_summed[nodes + 2] = first, half, False
            nodes += 3

    return sums[0]


@njit(cache=True, error_model="numpy")
def _keep_within_two_deviations(values, count, keep, squares):
    """Leave kept only those of values[:count] within two standard deviations of their mean, or equal to it where
    they don't vary."""
    mean = _sum(values, 0, count) / count
    for i in range(count):
        difference = values[i] - mean
        squares[i] = difference * difference

    deviation = np.sqrt(_sum(squares, 0, count) / count)
    for i in range(count):
        if deviation == 0.0:
            keep[i] = keep[i] and values[i] == mean
        else:
            keep[i] = keep[i] and abs(values[i] - mean) <= 2.0 * deviation


@njit(cache=True, error_model="numpy")
def _staged_means(stations, elevations, mannings_n, starts, centers, lefts, rights, thalwegs, slopes, k_decay,
                  shallow_factor, deep_factor, slope_factor, stage_count):
    """A reach's stages: for each stage from the first, how many of its cross sections were averaged (0 for none, and
    no row), and their mean discharge, velocity, top width, area and water surface elevation. The cross sections'
    profiles are end to end in stations, elevations and mannings_n, cross section j's from starts[j] to
    starts[j + 1], and only those with a slope above 0 count. At the first stage where one's hydraulics aren't all
    finite, the reach stops, and only the stages before it are returned."""
    count = centers.size
    averaged = np.zeros(stage_count, np.int64)
    means = np.zeros((stage_count, 5))
    values = np.empty((5, count))  # each counted cross section's discharge, velocity, top width, area and WSE
    keep = np.empty(count, np.bool_)
    scratch = np.empty(count)
    for stage in range(1, stage_count + 1):
        depth = stage * DEPTH_INCREMENT
        n = 0
        for j in range(count):
            slope = slopes[j]
            if not slope > 0.0:
                continue

            start, end = starts[j], starts[j + 1]
            wse = thalwegs[j] + depth
            area, perimeter, top_width, k, channel_area, channel_k = section_hydraulics(
                stations[start:end], elevations[start:end], mannings_n[start:end], centers[j], lefts[j], rights[j],
                wse, k_decay, shallow_factor, deep_factor)
            # With banks, the velocity is the channel's
            scale = np.sqrt(slope) * slope_factor
            if area <= 0.0 or perimeter <= 0.0 or k <= 0.0:
                area = perimeter = velocity = discharge = 0.0
            else:
                discharge = k * scale
                if lefts[j] >= 0.0 or rights[j] >= 0.0:
                    velocity = channel_k * scale / channel_area if channel_area > 0.0 else 0.0
                else:
                    velocity = discharge / area

            if not (np.isfinite(area) and np.isfinite(perimeter) and np.isfinite(velocity)
                    and np.isfinite(discharge) and np.isfinite(top_width)):
                return averaged[:stage - 1], means[:stage - 1]
            
            if area <= 0.0 or top_width <= 0.0 or discharge < 0.0:
                continue

            values[0, n] = discharge
            values[1, n] = velocity
            values[2, n] = top_width
            values[3, n] = area
            values[4, n] = wse
            n += 1

        if n == 0:
            continue

        keep[:n] = True
        for column in (3, 1, 2):  # the area, velocity and top width
            _keep_within_two_deviations(values[column], n, keep, scratch)

        kept = 0
        for i in range(n):
            if keep[i]:
                kept += 1

        if kept == 0:
            continue

        averaged[stage - 1] = kept
        for column in range(5):
            m = 0
            for i in range(n):
                if keep[i]:
                    scratch[m] = values[column, i]
                    m += 1
            means[stage - 1, column] = _sum(scratch, 0, kept) / kept

    return averaged, means


def _reach_columns(comid: int, group: Sequence[RepresentativeSample], parameters, slope_factor: float,
                   stage_count: int) -> dict | None:
    """One reach's rows, as columns, or None if it has none."""
    profiles = [hydraulic_profile(sample.xs) for sample in group]
    starts = np.zeros(len(group) + 1, dtype=np.int64)
    starts[1:] = np.cumsum([profile.stations.size for profile in profiles])
    vertices = [np.concatenate([np.asarray(getattr(profile, name), dtype=np.float64) for profile in profiles])
                for name in ("stations", "elevations", "mannings_n")]
    averaged, means = _staged_means(
        *vertices, starts, np.array([profile.center for profile in profiles], dtype=np.int64),
        np.array([sample.xs.left_bank_distance for sample in group], dtype=np.float64),
        np.array([sample.xs.right_bank_distance for sample in group], dtype=np.float64),
        np.array([sample.thalweg for sample in group], dtype=np.float64),
        np.array([sample.slope for sample in group], dtype=np.float64), *parameters, slope_factor, stage_count)
    stages = np.flatnonzero(averaged) + 1
    if stages.size == 0:
        return None
    
    rows = stages.size
    depths = stages * DEPTH_INCREMENT
    thalweg = float(np.nanmean([s.thalweg for s in group]))
    slopes = [s.slope for s in group if s.slope > 0.0]
    discharge, velocity, top_width, area, wse = means[stages - 1].T
    # The representative cross section: its area and top width kept from falling as the stage rises
    representative_area = monotonic_cumulative_max(area)
    fallback = representative_area / np.maximum(depths, 1e-9)
    width = monotonic_cumulative_max(np.where(top_width > 0.0, top_width, fallback))
    increments, representative_depths = depths_from_width_and_area(width, representative_area)
    return {
        'COMID': np.full(rows, comid, dtype=np.int64),
        'Cross_Section_Count': np.full(rows, len(group), dtype=np.int64),
        'Hydraulic_Sample_Count': averaged[stages - 1],
        'Depth_Stage_Index': stages.astype(np.int64),
        'Depth_Stage_Meters': depths,
        'Stream_Slope': np.full(rows, float(np.nanmean(slopes)) if slopes else np.nan),
        'Reach_Inflect_Terrace_Depth': np.full(rows, depths[-1]),
        'Representative_Thalweg_Elevation': np.full(rows, thalweg),
        'Mean_Discharge': discharge,
        'Mean_Depth': depths,
        'Mean_Velocity': velocity,
        'Representative_Velocity': velocity,
        'Mean_Top_Width': top_width,
        'Mean_Cross_Sectional_Area': area,
        'Mean_WSE': wse,
        'Representative_Cross_Sectional_Area': representative_area,
        'Representative_Depth_Increment': increments,
        'Representative_Depth': representative_depths,
        'Representative_Top_Width': width,
        'Representative_Stage_Elevation': np.full(rows, thalweg) + representative_depths,
        'Representative_Left_Station': -0.5 * width,
        'Representative_Right_Station': 0.5 * width,
    }


def representative_cross_section_dataframe(samples: Sequence[RepresentativeSample], *,
                                           roughness: DepthRoughness | None = None,
                                           slope_factor: float = 1.0) -> pd.DataFrame:
    """Each reach's representative cross section, one row per 0.10 m stage (see the notes above). The reaches are
    the samples' COMIDs, in increasing order."""
    parameters = (1.0, 1.0, 1.0) if roughness is None else _check_roughness(roughness)
    slope_factor = _check_slope_factor(slope_factor)
    grouped: dict[int, list[RepresentativeSample]] = {}
    for sample in samples:
        grouped.setdefault(int(sample.comid), []).append(sample)

    stage_count = _stage_count()
    reaches = [_reach_columns(comid, grouped[comid], parameters, slope_factor, stage_count) for comid in sorted(grouped)]
    reaches = [columns for columns in reaches if columns is not None]
    if not reaches:
        return pd.DataFrame(columns=REPRESENTATIVE_CROSS_SECTION_COLUMNS)
    
    return pd.DataFrame({column: np.concatenate([reach[column] for reach in reaches])
                         for column in REPRESENTATIVE_CROSS_SECTION_COLUMNS})


def _stage_count() -> int:
    return max(int(round(min(MAX_DEPTH, 25.0) / DEPTH_INCREMENT)), 0)


def highest_stage(thalweg: float) -> float:
    """The water surface of a cross section's highest stage, the highest level its hydraulics are worked out at."""
    return thalweg + _stage_count() * DEPTH_INCREMENT


def write_representative_cross_sections(df: pd.DataFrame | None, path: os.PathLike) -> pd.DataFrame:
    """Write the representative cross sections rounded to 6 decimals, as CSV or, for a .parquet path, Parquet
    (arc.outputs.tables), and return them."""
    if df is None or df.empty:
        # typed, so that an empty Parquet file still has integer and float columns
        df = pd.DataFrame({col: pd.Series(dtype=np.int64 if col in _INTEGER_COLUMNS else np.float64)
                           for col in REPRESENTATIVE_CROSS_SECTION_COLUMNS})
    else:
        df = df.copy()
        numeric = [c for c in df.columns if c not in _INTEGER_COLUMNS]
        df[numeric] = df[numeric].round(6)
        
    write_table(df, path)
    LOG.info('Finished writing ' + str(path))
    return df
