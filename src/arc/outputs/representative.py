"""Representative cross sections: each reach's hydraulics averaged over its cross sections, stage by stage.

Every 0.10 m above each cross section's stream cell, up to 25 m, the cross section's area, wetted perimeter, top
width, discharge and velocity are worked out on its own profile, a carved channel's exact shape included. With banks,
the velocity is the channel's (its discharge over its area); without them it's the whole section's. At each stage,
the cross sections whose area, velocity, top width, wetted perimeter and discharge are all within two standard
deviations of the reach's means are averaged, and a reach stops at the first stage where any cross section's
hydraulics aren't finite. The representative cross section is those averages: at each stage, its area and top width
are the means', its depth is the stage's, and it is symmetric about the reach's mean bed elevation.

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

Each reach's travel time K, in seconds, for Muskingum routing, is fitted as a power law of the discharge, K = A Q^p
(fit_celerity_k_power_law), to K = L / ((5/3) V) at its stages: L the reach's length (the Stream_Length_m field of the
stream lines, reach_lengths_from_network) and V its Representative_Velocity. A and p are on each of the reach's rows,
and are written beside the representative cross sections, one row per reach (representative_cross_section_coefficients_path).

The means are summed as numpy sums them, so they are numpy's nanmean and nanstd to the bit.
"""
from __future__ import annotations

import os
from collections.abc import Iterable, Sequence
from pathlib import Path
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
# Each reach's celerity K power law, in the table beside the representative cross sections
REPRESENTATIVE_CROSS_SECTION_COEFFICIENT_COLUMNS = [
    'COMID', 'Cross_Section_Count', 'Celerity_K_Power_Law_A', 'Celerity_K_Power_Law_p',
]
_COEFFICIENTS = ['Celerity_K_Power_Law_A', 'Celerity_K_Power_Law_p']
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
    # each counted cross section's discharge, velocity, top width, area, WSE, friction slope and section velocity,
    # the means' columns, then its wetted perimeter, which only the outlier filter uses
    values = np.empty((8, count))
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
            values[7, n] = perimeter
            n += 1

        if n == 0:
            continue

        keep[:n] = True
        for column in (3, 1, 2, 7, 0):  # the area, velocity, top width, wetted perimeter and discharge
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
    # The representative cross section is the means at each stage, at the stage's depth
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
        'Representative_Cross_Sectional_Area': area,
        'Representative_Depth_Increment': depths,
        'Representative_Depth': depths,
        'Representative_Top_Width': top_width,
        'Representative_Stage_Elevation': np.full(rows, thalweg) + depths,
        'Representative_Left_Station': -0.5 * top_width,
        'Representative_Right_Station': 0.5 * top_width,
        'Mean_Friction_Slope': friction,
        'Mean_Section_Velocity': section_velocity,
    }


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


def with_celerity_k(df: pd.DataFrame, reach_lengths: dict[int, float] | None = None) -> pd.DataFrame:
    """The representative cross sections with each reach's celerity K power law, A and p
    (fit_celerity_k_power_law), on its rows: NaN for a reach without a length in reach_lengths, or whose stages are
    too few to fit. Warns of a reach it can't fit when reach_lengths is given."""
    df = df.copy()
    amplitudes, exponents = np.full(len(df), np.nan), np.full(len(df), np.nan)
    discharge, velocity = df['Mean_Discharge'].to_numpy(), df['Representative_Velocity'].to_numpy()
    for comid, rows in df.groupby('COMID', sort=False).indices.items():
        length = (reach_lengths or {}).get(int(comid), np.nan)
        amplitudes[rows], exponents[rows] = fit_celerity_k_power_law(discharge[rows], velocity[rows], length)
        if reach_lengths is not None and not np.isfinite(amplitudes[rows[0]]):
            LOG.warning('No celerity K power-law fit for COMID %s: invalid length or insufficient hydraulic data.',
                        comid)
    df['Celerity_K_Power_Law_A'] = amplitudes
    df['Celerity_K_Power_Law_p'] = exponents
    return df


def representative_cross_section_dataframe(samples: Sequence[RepresentativeSample], *,
                                           roughness: DepthRoughness | None = None,
                                           slope_factor: float = 1.0,
                                           reach_lengths: dict[int, float] | None = None) -> pd.DataFrame:
    """Each reach's representative cross section, one row per 0.10 m stage (see the notes above), with its celerity
    K power law from reach_lengths, each reach's length in metres by its COMID (with_celerity_k). The reaches are the
    samples' COMIDs, in increasing order."""
    parameters = (1.0, 1.0, 1.0) if roughness is None else _check_roughness(roughness)
    slope_factor = _check_slope_factor(slope_factor)
    grouped: dict[int, list[RepresentativeSample]] = {}
    for sample in samples:
        grouped.setdefault(int(sample.comid), []).append(sample)

    stage_count = _stage_count()
    reaches = [_reach_columns(comid, grouped[comid], parameters, slope_factor, stage_count) for comid in sorted(grouped)]
    reaches = [columns for columns in reaches if columns is not None]
    if not reaches:
        return pd.DataFrame(columns=REPRESENTATIVE_CROSS_SECTION_COLUMNS + _COEFFICIENTS)

    return with_celerity_k(pd.DataFrame({column: np.concatenate([reach[column] for reach in reaches])
                                         for column in REPRESENTATIVE_CROSS_SECTION_COLUMNS}), reach_lengths)


def _stage_count() -> int:
    return max(round(min(MAX_DEPTH, 25.0) / DEPTH_INCREMENT), 0)


def highest_stage(thalweg: float) -> float:
    """The water surface of a cross section's highest stage, the highest level its hydraulics are worked out at."""
    return thalweg + _stage_count() * DEPTH_INCREMENT


def reach_lengths_from_network(network: pd.DataFrame, comids: Iterable[int], length_field: str,
                               reach_id: str) -> dict[int, float]:
    """Each reach's full length in metres, from the stream lines' length_field (Stream_Length_m), by COMID.

    Prefer COMID and validate the selected lengths. Networks may name
    their raster ID differently; in that case reach_id supplies the equivalent
    COMID column for this lookup alone. No geometry or attributes are modified.
    """
    comids = {int(comid) for comid in comids}
    if not comids:
        return {}
    if network is None:
        raise ValueError('StrmShp_File is required to read representative reach lengths.')
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


def representative_cross_section_coefficients_path(representative_cross_section_file):
    """Derive the companion coefficients filename: <stem>_exponents<suffix>."""
    path = Path(representative_cross_section_file)
    return path.with_name(f'{path.stem}_exponents{path.suffix}')


def write_representative_cross_sections(df: pd.DataFrame | None, path: os.PathLike,
                                        metadata: dict[str, str] | None = None) -> pd.DataFrame:
    """Write the representative cross sections rounded to 6 decimals, and beside them each reach's celerity K power
    law (REPRESENTATIVE_CROSS_SECTION_COEFFICIENT_COLUMNS) at representative_cross_section_coefficients_path(path):
    both as CSV or, for a .parquet path, Parquet with metadata (arc.outputs.tables). Returns the cross sections as
    written."""
    if df is None or df.empty:
        # typed, so that an empty Parquet file still has integer and float columns
        stages = pd.DataFrame({col: pd.Series(dtype=np.int64 if col in _INTEGER_COLUMNS else np.float64)
                               for col in REPRESENTATIVE_CROSS_SECTION_COLUMNS})
        coefficients = pd.DataFrame({col: pd.Series(dtype=np.int64 if col in _INTEGER_COLUMNS else np.float64)
                                     for col in REPRESENTATIVE_CROSS_SECTION_COEFFICIENT_COLUMNS})
    else:
        stages = df[REPRESENTATIVE_CROSS_SECTION_COLUMNS].copy()
        numeric = [c for c in stages.columns if c not in _INTEGER_COLUMNS]
        stages[numeric] = stages[numeric].round(6)
        if not all(column in df.columns for column in _COEFFICIENTS):
            df = with_celerity_k(df)
        coefficients = df[REPRESENTATIVE_CROSS_SECTION_COEFFICIENT_COLUMNS].drop_duplicates('COMID')
        coefficients = coefficients.reset_index(drop=True)

    companion = representative_cross_section_coefficients_path(path)
    write_table(stages, path, metadata)
    write_table(coefficients, companion, metadata)
    LOG.info(f'Finished writing {path} and {companion}')
    return stages
