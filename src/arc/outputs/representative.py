"""Representative hydraulic-stage exports and sample-based helper functions.

build_representative_cross_section_dataframe computes filtered mean hydraulics
at 0.10 m depth stages and fits reach-length-based celerity-K coefficients.
write_representative_tables writes the stage and coefficient CSVs. These
calculations use sampled profiles, bank indices, depth-varying roughness and a
slope adjustment factor. REPRESENTATIVE_STAGE_COLUMNS defines the stage schema.

The notes below describe the separate RepresentativeSample-based helper API:

Representative cross sections: each reach's hydraulics averaged over its cross sections, stage by stage.

Every 0.10 m above each cross section's stream cell, up to 25 m, the cross section's area, wetted perimeter, top
width, discharge and velocity are worked out on its own profile, a carved channel's exact shape included. With banks,
the velocity is the channel's (its discharge over its area); without them it's the whole section's. At each stage,
the cross sections whose area, velocity and top width are all within two standard deviations of the reach's means are
averaged, and a reach stops at the first stage where any cross section's hydraulics aren't finite. The
representative cross section is then built from the averages: its area and top width kept from falling as the stage
rises, and its depth at each stage the one that grows the width linearly from the stage below to give that area.

The discharge is Manning's, K sqrt(S) f (f the slope_adjustment_factor), at a slope S that is the cross section's own
slope, the water at normal depth, unless the cross section has a rating curve from the standard step (a sample's
curve). Then, at each of the rating curve's water surfaces, it is the curve's flow, so the representative cross
sections average the water the VDT database has. Between two of them it goes from one flow to the other
(BETWEEN_INCREMENTS): with "conveyance", in step with the cross section's conveyance K, Q_1 + (Q_2 - Q_1) (K - K_1) /
(K_2 - K_1), and never beyond Q_1 or Q_2, so that a curve at normal depth (flows in proportion to K) gives normal
depth's discharge between its water surfaces too; with "linear", linear in the water surface, as the VDT database is
read. Below and above them it is Manning's at the curve's friction slope there, the slope at which the cross section
carries the curve's flow, (Q / (f K))^2: the water stays as far above (or below) normal depth as at the curve's ends.
Of water surfaces held at the baseflow level at several flows, the highest flow's counts. A cross
section's discharge never falls as its stage rises: where it would, as where water spills onto flat ground, the
discharge below is held, as the rating curves hold their last increment.

With banks, the channel carries its share of the discharge, its conveyance over the whole section's at one friction
slope, and the velocity is the channel's (Mean_Velocity, Representative_Velocity); without banks it's the whole
section's. Mean_Section_Velocity is always the whole section's, discharge over area, the VDT database's velocity. The
friction slope each cross section's discharge implies at each stage, (Q / (f K))^2, is averaged as Mean_Friction_Slope:
its own slope where it has no curve and its discharge isn't held.

The means are summed as numpy sums them, so they are numpy's nanmean and nanstd to the bit.
"""
from __future__ import annotations

import os
from pathlib import Path
from collections.abc import Sequence
from typing import NamedTuple

import numpy as np
import pandas as pd
from numba import njit

from arc import LOG
from arc.hydraulics import (
    DepthRoughness,
    _check_roughness,
    _check_slope_factor,
    hydraulic_profile,
)
from arc.outputs.tables import write_table
from arc.rating_curve import section_hydraulics
from arc.xsection.xsection import XSection

DEPTH_INCREMENT = 0.10  # metres between stages
MAX_DEPTH = 25.0  # the highest stage above the stream cell
# How a rating curve's discharge goes between its water surfaces (see the notes above): "conveyance" or "linear"
BETWEEN_INCREMENTS = "linear"

REPRESENTATIVE_CROSS_SECTION_COLUMNS = [
    'COMID', 'Cross_Section_Count', 'Hydraulic_Sample_Count', 'Depth_Stage_Index', 'Depth_Stage_Meters',
    'Stream_Slope', 'Reach_Inflect_Terrace_Depth', 'Representative_Thalweg_Elevation', 'Mean_Discharge', 'Mean_Depth',
    'Mean_Velocity', 'Representative_Velocity', 'Mean_Top_Width', 'Mean_Cross_Sectional_Area', 'Mean_WSE',
    'Representative_Cross_Sectional_Area', 'Representative_Depth_Increment', 'Representative_Depth',
    'Representative_Top_Width', 'Representative_Stage_Elevation', 'Representative_Left_Station',
    'Representative_Right_Station', 'Mean_Friction_Slope', 'Mean_Section_Velocity',
]
_INTEGER_COLUMNS = ['COMID', 'Cross_Section_Count', 'Hydraulic_Sample_Count', 'Depth_Stage_Index']


class RepresentativeSample(NamedTuple):
    """One cross section of a reach, with its banks set on it for the channel velocity."""
    comid: int
    xs: XSection
    thalweg: float
    slope: float
    curve: np.ndarray | None = None  # its rating curve from the standard step, (count, 2): water surfaces and flows


@njit(cache=True, error_model="numpy")
def _curve_nodes(stations, elevations, mannings_n, starts, centers, lefts, rights, curve_starts, curve_wse,
                 curve_q, k_decay, shallow_factor, deep_factor, slope_factor):
    """Each cross section's rating curve as _staged_means reads it (see the notes above), the profiles as it takes
    them and cross section j's curve from curve_starts[j] to curve_starts[j + 1]: where each one's start, the water
    surfaces rising, their flows, the conveyances there, and the friction slopes. A water surface lower than the one
    before, or where the flow or conveyance isn't above 0, is left out."""
    count = centers.size
    node_starts = np.zeros(count + 1, np.int64)
    levels = np.empty(curve_wse.size)
    flows = np.empty(curve_wse.size)
    conveyances = np.empty(curve_wse.size)
    slopes = np.empty(curve_wse.size)
    m = 0
    for j in range(count):
        start, end = starts[j], starts[j + 1]
        first = m
        for i in range(curve_starts[j], curve_starts[j + 1]):
            wse, q = curve_wse[i], curve_q[i]
            if not (np.isfinite(wse) and np.isfinite(q) and q > 0.0) or (m > first and wse < levels[m - 1]):
                continue

            k = section_hydraulics(stations[start:end], elevations[start:end], mannings_n[start:end], centers[j],
                                   lefts[j], rights[j], wse, k_decay, shallow_factor, deep_factor)[3]
            if not (np.isfinite(k) and k > 0.0):
                continue

            if m > first and wse == levels[m - 1]:
                m -= 1  # held at the baseflow level: the higher flow's

            levels[m] = wse
            flows[m] = q
            conveyances[m] = k
            slopes[m] = (q / (slope_factor * k)) ** 2
            m += 1

        node_starts[j + 1] = m

    return node_starts, levels[:m], flows[:m], conveyances[:m], slopes[:m]


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
def _staged_means(stations, elevations, mannings_n, starts, centers, lefts, rights, thalwegs, slopes, node_starts,
                  node_levels, node_flows, node_conveyances, node_slopes, by_conveyance, k_decay, shallow_factor,
                  deep_factor, slope_factor, stage_count):
    """A reach's stages: for each stage from the first, how many of its cross sections were averaged (0 for none, and
    no row), and their mean discharge, velocity, top width, area, water surface elevation, friction slope and section
    velocity. The cross sections' profiles are end to end in stations, elevations and mannings_n, cross section j's
    from starts[j] to starts[j + 1], and only those with a slope above 0 count. Its rating curve is at the water
    surfaces node_levels[node_starts[j]:node_starts[j + 1]], with those flows, conveyances and friction slopes
    (_curve_nodes), its discharge between them in step with its conveyance if by_conveyance and otherwise linear in the
    water surface, and where it has none, its discharge is at its slope. At the first stage where one's hydraulics
    aren't all finite, the reach stops, and only the stages before it are returned."""
    count = centers.size
    averaged = np.zeros(stage_count, np.int64)
    means = np.zeros((stage_count, 7))
    # each counted cross section's discharge, velocity, top width, area, WSE, friction slope and section velocity
    values = np.empty((7, count))
    keep = np.empty(count, np.bool_)
    scratch = np.empty(count)
    held = np.zeros(count)  # each cross section's discharge at the stage below, which it doesn't fall from
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
            if area <= 0.0 or perimeter <= 0.0 or k <= 0.0:
                area = perimeter = velocity = section_velocity = discharge = 0.0
            else:
                first, last = node_starts[j], node_starts[j + 1]
                curved = last > first and node_levels[first] <= wse <= node_levels[last - 1]
                if curved:
                    # Within the rating curve: from the flow at the water surface below to the one above
                    upper = min(first + np.searchsorted(node_levels[first:last], wse), last - 1)
                    lower = max(upper - 1, first)
                    q1, q2 = node_flows[lower], node_flows[upper]
                    k1, k2 = node_conveyances[lower], node_conveyances[upper]
                    if upper == lower or wse == node_levels[upper]:
                        discharge = q2
                    elif by_conveyance and k2 != k1:
                        discharge = min(max(q1 + (q2 - q1) * (k - k1) / (k2 - k1), min(q1, q2)), max(q1, q2))
                    else:
                        discharge = np.interp(wse, node_levels[first:last], node_flows[first:last])
                else:
                    if last > first:  # beyond its ends: at its friction slope there
                        slope = node_slopes[first] if wse < node_levels[first] else node_slopes[last - 1]
                    discharge = k * np.sqrt(slope) * slope_factor

                if curved or discharge < held[j]:
                    # The friction slope its discharge implies: its own where it carries its own slope's
                    discharge = max(discharge, held[j])
                    slope = (discharge / (slope_factor * k)) ** 2
                held[j] = discharge
                section_velocity = discharge / area
                # With banks, the velocity is the channel's: its share of the discharge, at one friction slope
                if lefts[j] >= 0.0 or rights[j] >= 0.0:
                    velocity = discharge * channel_k / k / channel_area if channel_area > 0.0 else 0.0
                else:
                    velocity = section_velocity

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
            values[5, n] = slope
            values[6, n] = section_velocity
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
        for column in range(7):
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
    sections = (*vertices, starts, np.array([profile.center for profile in profiles], dtype=np.int64),
                np.array([sample.xs.left_bank_distance for sample in group], dtype=np.float64),
                np.array([sample.xs.right_bank_distance for sample in group], dtype=np.float64))
    # Each cross section's rating curve, end to end, none for those without one
    curves = [np.empty((0, 2)) if sample.curve is None else np.asarray(sample.curve, dtype=np.float64).reshape(-1, 2)
              for sample in group]
    curve_starts = np.zeros(len(group) + 1, dtype=np.int64)
    curve_starts[1:] = np.cumsum([curve.shape[0] for curve in curves])
    curve = np.concatenate(curves)
    nodes = _curve_nodes(*sections, curve_starts, curve[:, 0].copy(), curve[:, 1].copy(), *parameters, slope_factor)
    averaged, means = _staged_means(
        *sections, np.array([sample.thalweg for sample in group], dtype=np.float64),
        np.array([sample.slope for sample in group], dtype=np.float64), *nodes, BETWEEN_INCREMENTS == "conveyance",
        *parameters, slope_factor, stage_count)
    stages = np.flatnonzero(averaged) + 1
    if stages.size == 0:
        return None
    
    rows = stages.size
    depths = stages * DEPTH_INCREMENT
    thalweg = float(np.nanmean([s.thalweg for s in group]))
    slopes = [s.slope for s in group if s.slope > 0.0]
    discharge, velocity, top_width, area, wse, friction, section_velocity = means[stages - 1].T
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
        'Mean_Friction_Slope': friction,
        'Mean_Section_Velocity': section_velocity,
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
    return max(round(min(MAX_DEPTH, 25.0) / DEPTH_INCREMENT), 0)


def highest_stage(thalweg: float) -> float:
    """The water surface of a cross section's highest stage, the highest level its hydraulics are worked out at."""
    return thalweg + _stage_count() * DEPTH_INCREMENT


def write_representative_cross_sections(df: pd.DataFrame | None, path: os.PathLike,
                                        metadata: dict[str, str] | None = None) -> pd.DataFrame:
    """Write the representative cross sections rounded to 6 decimals, as CSV or, for a .parquet path, Parquet with
    metadata (arc.outputs.tables), and return them."""
    if df is None or df.empty:
        # typed, so that an empty Parquet file still has integer and float columns
        df = pd.DataFrame({col: pd.Series(dtype=np.int64 if col in _INTEGER_COLUMNS else np.float64)
                           for col in REPRESENTATIVE_CROSS_SECTION_COLUMNS})
    else:
        df = df.copy()
        numeric = [c for c in df.columns if c not in _INTEGER_COLUMNS]
        df[numeric] = df[numeric].round(6)

    write_table(df, path, metadata)
    LOG.info('Finished writing ' + str(path))
    return df


# Representative stage export. Its stage schema is separate from the
# retained sample-based API above.

def reach_lengths_from_network(network, records, length_field, reach_id):
    """Match full reach lengths to exported COMIDs without changing the network.

    Prefer COMID and validate the selected lengths. Networks may name
    their raster ID differently; in that case reach_id supplies the equivalent
    COMID column for this lookup alone. No geometry or attributes are modified.
    """
    comids = {int(record['COMID']) for record in records if record is not None}
    if not comids:
        return {}
    id_field = 'COMID' if 'COMID' in network.columns else reach_id
    if id_field not in network.columns or length_field not in network.columns:
        raise ValueError(f'StrmShp_File must contain {id_field} and the Stream_Length_m column {length_field}.')
    ids = pd.to_numeric(network[id_field], errors='coerce')
    lengths = pd.to_numeric(network[length_field], errors='coerce')
    selected = pd.DataFrame({'COMID': ids, 'Length': lengths}).loc[ids.isin(comids)]
    if selected.COMID.duplicated().any():
        raise ValueError('Duplicate stream IDs prevent an unambiguous celerity K reach-length lookup.')
    missing = comids - set(selected.COMID)
    if missing:
        raise ValueError(f'No stream-length record in StrmShp_File for COMIDs {sorted(missing)}.')
    if not (np.isfinite(selected.Length) & (selected.Length > 0)).all():
        raise ValueError(f'Stream_Length_m column {length_field} must contain finite positive lengths in meters.')
    return dict(zip(selected.COMID.astype(int), selected.Length))

def write_representative_tables(frame, path):
    """Write the representative stage and companion coefficient CSVs.

    Stages exclude coefficient columns and round non-integer
    values to six decimals. Coefficients retain precision and their signed p.
    Both destinations are CSV even when the supplied suffix is .parquet.
    """
    coefficients = frame[REPRESENTATIVE_CROSS_SECTION_COEFFICIENT_COLUMNS].drop_duplicates('COMID')
    stages = frame[REPRESENTATIVE_STAGE_COLUMNS].copy()
    numeric = [column for column in stages if column not in
               ('COMID', 'Cross_Section_Count', 'Hydraulic_Sample_Count', 'Depth_Stage_Index')]
    stages[numeric] = stages[numeric].round(6)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    stages.to_csv(path, index=False)
    companion = representative_cross_section_coefficients_path(path)
    coefficients.to_csv(companion, index=False)
    LOG.info('Finished writing %s and %s', path, companion)

@njit(cache=True)
def _calculate_all(
    da_xs_profile1: np.ndarray,
    xs1_n: int,
    mannings_n1: np.ndarray,
    da_xs_profile2: np.ndarray,
    xs2_n: int,
    mannings_n2: np.ndarray,
    d_ordinate_dist: float,
    wse: float,
    sqrt_slope: float,
    k_decay: float = 6.0,
    shallow_factor: float = 2.0, deep_factor: float = 1.0, slope_adjustment_factor: float = 1.0, bank_index1: int = -1, bank_index2: int = -1,
    channel_velocity: bool = False
):
    wse = np.round(wse, 3)

    if 0 < bank_index1 < xs1_n and 0 < bank_index2 < xs2_n:
        A, P, T, conveyance = _compound_section_conveyance(
            da_xs_profile1[:xs1_n], mannings_n1[:xs1_n], bank_index1,
            da_xs_profile2[:xs2_n], mannings_n2[:xs2_n], bank_index2,
            wse, d_ordinate_dist, k_decay, shallow_factor, deep_factor,
        )
        if A <= 0.0 or P <= 0.0 or conveyance <= 0.0:
            return 0.0, 0.0, 0.0, 0.0, np.round(T, 3), 0.0
        # An equivalent n is diagnostic only; discharge uses summed conveyance.
        equivalent_n = A * (A / P)**(2.0 / 3.0) / conveyance
        Q = np.round(conveyance * sqrt_slope * slope_adjustment_factor, 3)
        A = np.round(A, 3)
        V = np.round(Q / A, 3) if A > 0.0 else 0.0
        if channel_velocity:
            channel_area, _, _, channel_conveyance = _compound_section_conveyance(
                da_xs_profile1[:xs1_n], mannings_n1[:xs1_n], bank_index1,
                da_xs_profile2[:xs2_n], mannings_n2[:xs2_n], bank_index2,
                wse, d_ordinate_dist, k_decay, shallow_factor, deep_factor,
                channel_only=True,
            )
            V = np.round(channel_conveyance * sqrt_slope * slope_adjustment_factor / channel_area, 3) if channel_area > 0.0 else np.nan
        return A, np.round(P, 3), V, Q, np.round(T, 3), np.round(equivalent_n, 4)

    # Pass depth-scaling parameters to geometry functions
    A1, P1, np1, T1 = _calculate_stream_geometry_and_topwidth(
        da_xs_profile1[:xs1_n], wse, d_ordinate_dist, mannings_n1[:xs1_n], k_decay, shallow_factor, deep_factor
    )
    A2, P2, np2, T2 = _calculate_stream_geometry_and_topwidth(
        da_xs_profile2[:xs2_n], wse, d_ordinate_dist, mannings_n2[:xs2_n], k_decay, shallow_factor, deep_factor
    )

    T = np.round(T1 + T2, 3)
    A = np.round(A1 + A2, 3)
    P = np.round(P1 + P2, 3)

    if A <= 0.0 or P <= 0.0:
        return 0.0, 0.0, 0.0, 0.0, T, 0.0

    # Composite roughness formula remains unchanged; np1 and np2 already incorporate depth scaling
    d_composite_n = np.round(((np1 + np2) / P)**(2 / 3), 4)

    # Manning's discharge calculation
    Q = np.round((1 / d_composite_n) * A * (A / P)**(2 / 3) * (sqrt_slope * slope_adjustment_factor), 3)
    V = np.round(Q / A, 3)

    return A, P, V, Q, T, d_composite_n

@njit(cache=True)
def _check_for_negative_depths(da_y_depth: np.ndarray):
    # Take action if there are values < 0
    lt_0_in_depths = False
    i_target_index = 0
    for i_target_index, value in enumerate(da_y_depth[1:]):
        if value <= 0:
            lt_0_in_depths = True
            break

    return lt_0_in_depths, i_target_index

@njit(cache=True)
def _get_distance_to_use(da_y_depth: np.ndarray, i_target_index: int, d_ordinate_dist: float):
    return d_ordinate_dist * da_y_depth[i_target_index - 1] / (np.abs(da_y_depth[i_target_index - 1]) + np.abs(da_y_depth[i_target_index]))

@njit(cache=True)
def _calculate_top_width_up_to_point(i_target_index: int, d_dist_use: float, d_ordinate_dist: float):
    return d_ordinate_dist * (i_target_index - 1) + d_dist_use

@njit(cache=True)
def _calculate_top_width_from_all(da_y_depth: np.ndarray, d_ordinate_dist: float):
    return d_ordinate_dist * (da_y_depth.shape[0] - 1)

@njit(cache=True)
def _get_stream_depths(d_wse: float, profile: np.ndarray):
    da_y_depth = d_wse - profile

    if da_y_depth.shape[0] <= 0 or da_y_depth[0] <= 1e-16:
        return None

    return da_y_depth

@njit(cache=True)
def _adjust_n_by_depth(
    n0: np.ndarray,
    depth: np.ndarray,
    shallow_factor: float,
    deep_factor: float,
    k_decay: float,
) -> np.ndarray:
    """
    Apply a bounded exponential adjustment to baseline Manning's n.

    n(h) = n0 * (
        deep_factor
        + (shallow_factor - deep_factor) / (1.0 + k_decay * h)
    )

    Parameters
    ----------
    n0 : np.ndarray
        Baseline Manning's n values. Preprocess NoData and ensure
        finite, positive values before calling.
    depth : np.ndarray
        Signed local depths, aligned with n0.
    shallow_factor : float
        Multiplier as wet depth approaches zero; must be >= 1.
    deep_factor : float
        Large-depth limiting multiplier; must be in (0, 1].
    k_decay : float
        Positive decay coefficient in inverse depth units.

    Notes
    -----
    Setting both factors to 1 reproduces the baseline.
    Neither input array is modified.
    Dry locations must still be excluded by the geometry calculation.
    """
    if not np.isfinite(shallow_factor) or shallow_factor < 1.0:
        raise ValueError(
            "shallow_factor must be finite and >= 1."
        )

    if (
        not np.isfinite(deep_factor)
        or deep_factor <= 0.0
        or deep_factor > 1.0
    ):
        raise ValueError(
            "deep_factor must be finite and in (0, 1]."
        )

    if not np.isfinite(k_decay) or k_decay <= 0.0:
        raise ValueError(
            "k_decay must be finite and > 0."
        )

    if (
        n0.ndim != 1
        or depth.ndim != 1
        or n0.size != depth.size
    ):
        raise ValueError(
            "n0 and depth must be matching 1-D arrays."
        )

    # Clamp only the depths used for roughness scaling.
    # Preserve signed input depths for bank-intersection calculations.
    h = np.maximum(depth, 0.0)


    updated_mannning_n = n0 * (
        deep_factor
        + (shallow_factor - deep_factor) / (1.0 + k_decay * h)
    )

    return updated_mannning_n

@njit(cache=True)
def _bank_subsection_geometry(profile, roughness, bank_index, wse, spacing,
                              k_decay, shallow_factor, deep_factor):
    """Integrate connected wet bed segments into channel and overbank parts.

    Columns are area, bed perimeter, perimeter-weighted n**1.5, and width.
    Vertical subdivision interfaces contribute no solid wetted perimeter.
    Stop at the first dry barrier, as in the existing side-profile calculation.
    """
    parts = np.zeros((2, 4))
    depths = wse - profile
    if len(depths) < 2 or depths[0] <= 0.0:
        return parts
    scaled_n = _adjust_n_by_depth(roughness, depths, shallow_factor, deep_factor, k_decay)
    for j in range(1, len(profile)):
        h0, h1 = depths[j - 1], depths[j]
        if h0 <= 0.0:
            break
        width = spacing
        n = scaled_n[j]
        if h1 <= 0.0:
            width *= h0 / (h0 - h1)
            h1 = 0.0
            n = scaled_n[j - 1]
        perimeter = np.hypot(width, h1 - h0)
        part = 0 if j <= bank_index else 1
        parts[part, 0] += width * (h0 + h1) * 0.5
        parts[part, 1] += perimeter
        parts[part, 2] += perimeter * n**1.5
        parts[part, 3] += width
        if depths[j] <= 0.0:
            break
    return parts

@njit(cache=True)
def _compound_section_conveyance(profile1, n1, bank1, profile2, n2, bank2,
                                 wse, spacing, k_decay, shallow_factor, deep_factor,
                                 channel_only=False):
    """Sum conveyances (SI units), optionally returning only main-channel geometry and conveyance."""
    left = _bank_subsection_geometry(profile1, n1, bank1, wse, spacing,
                                    k_decay, shallow_factor, deep_factor)
    right = _bank_subsection_geometry(profile2, n2, bank2, wse, spacing,
                                     k_decay, shallow_factor, deep_factor)
    sections = np.empty((3, 4))
    sections[0] = left[0] + right[0]
    sections[1] = left[1]
    sections[2] = right[1]
    area, perimeter, width, conveyance = 0.0, 0.0, 0.0, 0.0
    for i in range(1 if channel_only else 3):
        a, p, weighted_n, t = sections[i]
        area += a
        perimeter += p
        width += t
        if a > 0.0 and p > 0.0:
            n = (weighted_n / p)**(2.0 / 3.0)
            if n < 0.0001:
                n = 0.035
            conveyance += a * (a / p)**(2.0 / 3.0) / n
    return area, perimeter, width, conveyance

@njit(cache=True)
def _calculate_stream_geometry_and_topwidth(
    da_xs_profile: np.ndarray,
    d_wse: float,
    d_ordinate_dist: float,
    da_n_profile: np.ndarray,
    k_decay: float = 1.0,
    shallow_factor: float = 2.0, deep_factor: float = 1.0
) -> tuple[float, ...]:
    d_area, d_perimeter, d_composite_n, d_top_width = 0.0, 0.0, 0.0, 0.0

    da_y_depth = _get_stream_depths(d_wse, da_xs_profile)
    if da_y_depth is None:
        return 0.0, 0.0, 0.0, 0.0

    # set da_n_profile to 0.03 and adjust it based on depth
    # da_n_profile = np.full_like(da_n_profile, 1.0)

    # Transition between the configured shallow and deep roughness factors.
    da_scaled_n = _adjust_n_by_depth(
        n0=da_n_profile,
        depth=da_y_depth,
        shallow_factor=shallow_factor,
        deep_factor=deep_factor,
        k_decay=k_decay
    )

    # da_scaled_n = da_n_profile

    lt_0_in_depths, i_target_index = _check_for_negative_depths(da_y_depth)

    if lt_0_in_depths:
        i_target_index += 1
        d_dist_use = _get_distance_to_use(da_y_depth, i_target_index, d_ordinate_dist)

        d_area = np.sum(d_ordinate_dist * 0.5 * (da_y_depth[1:i_target_index] + da_y_depth[:i_target_index-1])) + 0.5 * d_dist_use * da_y_depth[i_target_index-1]
        d_perimeter_i = calculate_hypotnuse(d_dist_use, da_y_depth[i_target_index - 1])
        perim_array = calculate_hypotnuse(d_ordinate_dist, (da_y_depth[1:i_target_index] - da_y_depth[:i_target_index-1]))

        d_perimeter = np.sum(perim_array) + d_perimeter_i
        d_composite_n = np.sum(perim_array[:i_target_index-1] * da_scaled_n[1:i_target_index]**1.5) + d_perimeter_i * da_scaled_n[i_target_index - 1]**1.5
        d_top_width = _calculate_top_width_up_to_point(i_target_index, d_dist_use, d_ordinate_dist)

    else:
        d_area = np.sum(d_ordinate_dist * 0.5 * (da_y_depth[2:] + da_y_depth[1:-1]))
        perim_array = calculate_hypotnuse(d_ordinate_dist, da_y_depth[1:] - da_y_depth[:-1])

        d_perimeter = np.sum(perim_array[1:])
        # Match roughness to the endpoints of the retained perimeter segments.
        d_composite_n = np.sum(perim_array[1:] * da_scaled_n[2:]**1.5)
        d_top_width = _calculate_top_width_from_all(da_y_depth, d_ordinate_dist)

    return d_area, d_perimeter, d_composite_n, d_top_width

@njit(cache=True)
def calculate_hypotnuse(d_side_one: float, d_side_two: float):
    """
    Calculates the hypotenuse distance of a right triangle

    Parameters
    ----------
    d_side_one: float
        Length of the first right triangle side
    d_side_two: float
        Length of the second right triangle side

    Returns
    -------
    d_distance: float
        Length of the hypotenuse

    """

    # Calculate the distance
    d_distance = (d_side_one ** 2 + d_side_two ** 2)**(1/2)

    # Return to the calling function
    return d_distance

REPRESENTATIVE_DEPTH_INCREMENT = 0.10

REPRESENTATIVE_STAGE_COLUMNS = [
    'COMID',
    'Cross_Section_Count',
    'Hydraulic_Sample_Count',
    'Depth_Stage_Index',
    'Depth_Stage_Meters',
    'Stream_Slope',
    'Reach_Inflect_Terrace_Depth',
    'Representative_Thalweg_Elevation',
    'Mean_Discharge',
    'Mean_Depth',
    'Mean_Velocity',
    'Representative_Velocity',
    'Mean_Top_Width',
    'Mean_Cross_Sectional_Area',
    'Mean_WSE',
    'Representative_Cross_Sectional_Area',
    'Representative_Depth_Increment',
    'Representative_Depth',
    'Representative_Top_Width',
    'Representative_Stage_Elevation',
    'Representative_Left_Station',
    'Representative_Right_Station',
]

REPRESENTATIVE_CROSS_SECTION_COEFFICIENT_COLUMNS = [
    'COMID',
    'Cross_Section_Count',
    'Celerity_K_Power_Law_A',
    'Celerity_K_Power_Law_p',
]

def representative_cross_section_coefficients_path(representative_cross_section_file):
    """Derive the companion coefficients filename: <stem>_exponents<suffix>."""
    path = Path(representative_cross_section_file)
    return path.with_name(f'{path.stem}_exponents{path.suffix}')

def _filter_stage_samples_to_two_standard_deviations(
    areas: np.ndarray,
    velocities: np.ndarray,
    top_widths: np.ndarray,
    perimeters: np.ndarray,
    discharges: np.ndarray,
) -> np.ndarray:
    """Keep stage samples whose key hydraulics are near the reach mean.

    The representative section is meant to summarize the reach, so a sampled
    cross-section must have area, velocity, and top width all within two
    standard deviations of that stage's mean before it contributes to the
    final reach mean.
    """
    keep_mask = np.isfinite(areas) & np.isfinite(velocities) & np.isfinite(top_widths) & np.isfinite(perimeters) & np.isfinite(discharges)
    for values in (areas, velocities, top_widths, perimeters, discharges):
        finite_values = values[np.isfinite(values)]
        if finite_values.size == 0:
            return np.zeros(values.shape, dtype=bool)

        mean_value = float(np.nanmean(finite_values))
        standard_deviation = float(np.nanstd(finite_values))
        if standard_deviation == 0.0:
            keep_mask &= values == mean_value
        else:
            keep_mask &= np.abs(values - mean_value) <= (2.0 * standard_deviation)

    return keep_mask

def _build_representative_hydraulic_rows_for_reach(
    comid: int,
    group: list[dict],
    max_depth: float,
    k_decay: float = 6.0,
    shallow_factor: float = 2.0, deep_factor: float = 1.0,
    slope_adjustment_factor: float = 1.0,
) -> list[dict]:
    """Build staged hydraulic means for one reach up to a fixed depth cap.

    ARC evaluates each contributing cross section every 0.10 meters above its
    local thalweg up to ``max_depth`` meters, computing area, perimeter,
    main-channel velocity, total discharge, and total top width at each stage.
    Without valid bank stations, velocity falls back to the whole-section mean.
    If any non-finite hydraulic value appears, the sweep stops early and the
    last successful depth becomes the effective cap for that reach.
    """
    max_depth = float(min(max_depth, 25.0))
    stage_count = max(int(round(max_depth / REPRESENTATIVE_DEPTH_INCREMENT)), 0)
    if stage_count <= 0:
        return []

    representative_thalweg = float(
        np.nanmean([float(record['Thalweg']) for record in group if record.get('Thalweg') is not None])
    )
    valid_slopes = [
        float(record['Slope'])
        for record in group
        if record.get('Slope') is not None and float(record['Slope']) > 0.0
    ]
    representative_stream_slope = float(np.nanmean(valid_slopes)) if valid_slopes else np.nan
    rows: list[dict] = []
    last_valid_stage_depth = 0.0

    for stage_index in range(1, stage_count + 1):
        stage_depth = float(stage_index * REPRESENTATIVE_DEPTH_INCREMENT)
        samples = []

        for record in group:
            thalweg = record.get('Thalweg')
            slope = record.get('Slope')
            if thalweg is None or slope is None or slope <= 0.0:
                continue

            left_profile = np.asarray(record['XS1_Profile'], dtype=np.float64)
            right_profile = np.asarray(record['XS2_Profile'], dtype=np.float64)
            left_n = np.asarray(record['Manning_N_Raster1'], dtype=np.float64)
            right_n = np.asarray(record['Manning_N_Raster2'], dtype=np.float64)
            ordinate_dist = float(record['Ordinate_Dist'])
            wse = float(thalweg + stage_depth)
            sqrt_slope = float(slope) ** 0.5

            area, perimeter, velocity, discharge, top_width, d_composite_n = _calculate_all(
                left_profile,
                left_profile.size,
                left_n,
                right_profile,
                right_profile.size,
                right_n,
                ordinate_dist,
                wse,
                sqrt_slope,
                k_decay,
                shallow_factor,
                deep_factor,
                slope_adjustment_factor,
                int(record.get("Bank_Index1", -1)),
                int(record.get("Bank_Index2", -1)),
                channel_velocity=True,
            )

            if not all(np.isfinite(value) for value in (area, perimeter, velocity, discharge, top_width, d_composite_n)):
                for row in rows:
                    row['Reach_Inflect_Terrace_Depth'] = last_valid_stage_depth
                return rows

            if area <= 0.0 or top_width <= 0.0 or discharge < 0.0:
                continue

            samples.append((float(area), float(velocity), float(top_width),
                            float(perimeter), float(discharge), float(wse)))

        if not samples:
            continue
        sample_values = np.asarray(samples, dtype=np.float64)
        area_values, velocity_values, top_width_values, perimeter_values, discharge_values, wse_values = sample_values.T

        # Remove hydraulic outliers before averaging each stage's samples.
        sample_keep_mask = _filter_stage_samples_to_two_standard_deviations(
            area_values,
            velocity_values,
            top_width_values,
            perimeter_values,
            discharge_values,
        )
        if not sample_keep_mask.any():
            continue

        rows.append(
            {
                'COMID': int(comid),
                'Cross_Section_Count': int(len(group)),
                'Hydraulic_Sample_Count': int(np.count_nonzero(sample_keep_mask)),
                'Depth_Stage_Index': int(stage_index),
                'Depth_Stage_Meters': stage_depth,
                'Stream_Slope': representative_stream_slope,
                'Reach_Inflect_Terrace_Depth': max_depth,
                'Representative_Thalweg_Elevation': representative_thalweg,
                'Mean_Discharge': float(np.nanmean(discharge_values[sample_keep_mask])),
                'Mean_Depth': stage_depth,
                'Mean_Velocity': float(np.nanmean(velocity_values[sample_keep_mask])),
                'Mean_Top_Width': float(np.nanmean(top_width_values[sample_keep_mask])),
                'Mean_Cross_Sectional_Area': float(np.nanmean(area_values[sample_keep_mask])),
                'Mean_WSE': float(np.nanmean(wse_values[sample_keep_mask])),
            }
        )
        last_valid_stage_depth = stage_depth
    if rows:
        for row in rows:
            row['Reach_Inflect_Terrace_Depth'] = last_valid_stage_depth
    return rows

def fit_celerity_k_power_law(discharge, velocity, reach_length_m):
    """Fit K = A Q^p, where K = L / ((5/3) V), in seconds.

    Equal weights are used for each unique positive discharge, matching the
    representative hydraulic curve used by the analysis scripts. A is strictly
    positive and p retains its fitted sign, so K stays positive for Q > 0.
    Negative p describes decreasing K; positive p describes increasing K. Only finite
    discharge values greater than zero with positive velocities are used. Failed or
    insufficient fits return NaN coefficients without discarding stage rows.
    """
    if not np.isfinite(reach_length_m) or reach_length_m <= 0:
        return np.nan, np.nan
    pairs = pd.DataFrame({'Q': discharge, 'V': velocity}).replace([np.inf, -np.inf], np.nan).dropna()
    # Exclude zero and negative discharges before grouping or fitting.
    pairs = pairs.loc[pairs.Q > 0]
    pairs = pairs.loc[pairs.V > 0]
    pairs = pairs.groupby('Q', as_index=False).V.median().sort_values('Q')
    if len(pairs) < 3:
        return np.nan, np.nan
    q = pairs.Q.to_numpy(dtype=float)
    k = reach_length_m / ((5.0 / 3.0) * pairs.V.to_numpy(dtype=float))
    if not (np.isfinite(k) & (k > 0)).all():
        return np.nan, np.nan
    # Minimize squared log(K) errors so large low-flow K values do not dominate.
    log_q, log_k = np.log(q), np.log(k)
    centered_q = log_q - log_q.mean()
    try:
        slope = np.linalg.lstsq(centered_q[:, None], log_k - log_k.mean(), rcond=None)[0][0]
        exponent = float(slope)
        with np.errstate(over='raise', under='ignore', invalid='raise'):
            amplitude = float(np.exp(log_k.mean() - exponent * log_q.mean()))
    except (np.linalg.LinAlgError, ValueError, FloatingPointError):
        return np.nan, np.nan
    if not np.isfinite([amplitude, exponent]).all() or amplitude <= 0:
        return np.nan, np.nan
    return amplitude, exponent

def build_representative_cross_section_dataframe(
    cross_section_data: list[dict],
    k_decay: float = 6.0,
    shallow_factor: float = 2.0, deep_factor: float = 1.0,
    slope_adjustment_factor: float = 1.0,
    reach_lengths_m: dict[int, float] | None = None,
) -> pd.DataFrame:
    """Build representative cross sections from filtered staged hydraulic means.

    When representative cross sections are enabled, ARC now ignores the qmax-
    based VDT staging for this export. Instead it:

    1. Collects the sampled cross-section geometry, Manning's n arrays, slope,
       and thalweg elevation for every stream cell.
    2. Recomputes hydraulic properties every 0.10 meters above each
       cross-section thalweg up to a 25 meter maximum depth.
       The sweep truncates earlier if any non-finite hydraulic outputs appear.
    3. Removes cross sections whose area, velocity, or top width are outside
       two standard deviations of the reach mean at the current stage.
    4. Takes the mean discharge, velocity, top width, area, and WSE across the
       retained reach samples at each stage.
    5. Derives representative cross-section dimensions from the staged mean
       top width and staged mean area.
    6. Fits K = A Q^p for each COMID using celerity = (5/3) V and
       repeats the fitted coefficients on every row belonging to that COMID.

    Parameters
    ----------
    cross_section_data : list of dict
        Per-cell sampled cross-section records collected during the ARC main
        loop. These records include the profiles, Manning arrays, local slope,
        and thalweg elevation needed to rebuild the representative hydraulic
        stage database.
    reach_lengths_m : dict, optional
        Reach lengths in meters keyed by COMID. Missing or invalid lengths
        leave the power-law coefficients NaN, preserving the hydraulic rows.

    Returns
    -------
    pandas.DataFrame
        Long-format representative cross-section table with one row per reach
        and 0.10 m depth stage, including repeated per-COMID coefficients for
        internal use. The writer separates those coefficients from the stage CSV.
    """
    columns = REPRESENTATIVE_STAGE_COLUMNS + [
        col for col in REPRESENTATIVE_CROSS_SECTION_COEFFICIENT_COLUMNS
        if col not in REPRESENTATIVE_STAGE_COLUMNS
    ]
    if not cross_section_data:
        return pd.DataFrame(columns=columns)

    grouped_records: dict[int, list[dict]] = {}
    for i, record in enumerate(cross_section_data):
        if record is None:
            continue
        comid = int(record['COMID'])
        grouped_records.setdefault(comid, []).append(record)

    rows: list[dict] = []
    for comid, group in grouped_records.items():
        rows.extend(_build_representative_hydraulic_rows_for_reach(int(comid), group, 25.0, k_decay, shallow_factor, deep_factor, slope_adjustment_factor))

    representative_df = pd.DataFrame(rows)
    if representative_df.empty:
        return pd.DataFrame(columns=columns)


    representative_groups = []
    for comid, group in representative_df.groupby('COMID', sort=True):
        group = group.sort_values('Depth_Stage_Index').copy()
        group['Mean_Discharge'] = group['Mean_Discharge'].to_numpy(dtype=np.float64)
        group['Representative_Cross_Sectional_Area'] = group['Mean_Cross_Sectional_Area'].to_numpy(dtype=np.float64)
        group['Representative_Depth_Increment'] = group['Mean_Depth'].to_numpy(dtype=np.float64)
        group['Representative_Top_Width'] = group['Mean_Top_Width'].to_numpy(dtype=np.float64)
        group['Representative_Depth'] = group['Mean_Depth'].to_numpy(dtype=np.float64)
        # Preserve the filtered mean of per-section Q_channel / A_channel.
        group['Representative_Velocity'] = group['Mean_Velocity']
        group['Representative_Stage_Elevation'] = (
            group['Representative_Thalweg_Elevation'].to_numpy(dtype=np.float64) + group['Mean_Depth'].to_numpy(dtype=np.float64)
        )
        group['Representative_Left_Station'] = -0.5 * group['Representative_Top_Width']
        group['Representative_Right_Station'] = 0.5 * group['Representative_Top_Width']
        length_m = (reach_lengths_m or {}).get(int(comid), np.nan)
        amplitude, exponent = fit_celerity_k_power_law(
            group['Mean_Discharge'], group['Representative_Velocity'], length_m,
        )
        group['Celerity_K_Power_Law_A'] = amplitude
        group['Celerity_K_Power_Law_p'] = exponent
        if reach_lengths_m is not None and not np.isfinite(amplitude):
            LOG.warning('No celerity K power-law fit for COMID %s: invalid length or insufficient hydraulic data.', comid)
        representative_groups.append(group)

    representative_df = pd.concat(representative_groups, ignore_index=True)
    representative_df = representative_df[columns]

    int_columns = ['COMID', 'Cross_Section_Count', 'Hydraulic_Sample_Count', 'Depth_Stage_Index']
    representative_df[int_columns] = representative_df[int_columns].astype(int)
    return representative_df
