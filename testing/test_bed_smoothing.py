from __future__ import annotations

import math

import networkx as nx
import numpy as np
import pytest

from arc.bathymetry import Banks
from arc.bathymetry.bed_smoothing import (
    MAX_BED_GRADE,
    fill_reach_depths,
    smooth_channel_depths,
    smooth_reach_bed,
)
from arc.bathymetry.smoothing import ReachSections, SmoothedReach, order_reach
from arc.xsection.slope import MAX_SLOPE
from arc.xsection.xsection import XSection

CELL = 10.0
GRADE = 0.01  # a bed cap of 1 cm per metre, gentle enough to see what it does
NO_BANKS = Banks("none", math.nan, math.nan, math.nan, math.nan, False, False)


def line_graph(*reaches) -> nx.DiGraph:
    network = nx.DiGraph()
    network.add_nodes_from(reaches)
    network.add_edges_from(zip(reaches[:-1], reaches[1:]))
    return network


def network_inputs(network, cells, banks, thalwegs=None):
    """What smooth_channel_depths takes, for cross sections at cells (reach, row, col) in that order, each reach
    ordered as smooth_bank_elevations orders it, with these smoothed bank elevations and stream cell elevations."""
    thalwegs = np.zeros(len(cells)) if thalwegs is None else np.asarray(thalwegs, dtype=float)
    indices: dict[int, list[int]] = {}
    for k, (reach, _, _) in enumerate(cells):
        indices.setdefault(reach, []).append(k)
    positions = {reach: (np.array([cells[k][1] for k in ks]), np.array([cells[k][2] for k in ks]))
                 for reach, ks in indices.items()}
    reaches, smoothed = {}, {}
    for reach, ks in indices.items():
        rows, cols = positions[reach]
        sections = [XSection(np.array([thalwegs[k] + 5.0, thalwegs[k], thalwegs[k] + 5.0]), np.full(3, 0.035), CELL)
                    for k in ks]
        reaches[reach] = ReachSections(rows, cols, sections, [NO_BANKS] * len(ks))
        bank_elevations = np.array([banks[k] for k in ks], dtype=float)
        order, stations = order_reach(network, reach, rows, cols, CELL, CELL, positions, bank_elevations)
        smoothed[reach] = SmoothedReach([NO_BANKS] * len(ks), bank_elevations, bank_elevations, order, stations, None)
    return reaches, smoothed, indices


def smoothed_depths(network, cells, banks, depths, thalwegs=None, **options):
    """smooth_channel_depths' depths and beds, back in the order of cells, with a bed cap of GRADE unless told
    otherwise."""
    reaches, smoothed, indices = network_inputs(network, cells, banks, thalwegs)
    options.setdefault("use_banks", True)
    options.setdefault("max_bed_grade", GRADE)
    result = smooth_channel_depths(network, reaches, smoothed, {r: [depths[k] for k in ks] for r, ks in indices.items()},
                                   CELL, CELL, **options)
    out_depths, out_beds = np.empty(len(cells)), np.empty(len(cells))
    for reach, ks in indices.items():
        out_depths[ks], out_beds[ks] = result[reach].depths, result[reach].beds
    return out_depths, out_beds


def along_row(reach_lengths, row=0):
    """Reaches one after another along a row, one cross section per cell."""
    cells, col = [], 0
    for reach, length in reach_lengths:
        cells += [(reach, row, col + k) for k in range(length)]
        col += length
    return cells


# --- Filling in depths --------------------------------------------------------------------------------------------


def test_depths_that_aren_t_above_0_and_finite_take_the_one_before() -> None:
    depths = [1.0, 0.0, -1.0, 2.0, math.inf, 3.0]

    filled = fill_reach_depths(line_graph(1), {1: depths}, {1: np.arange(6)})

    np.testing.assert_array_equal(filled[1], [1.0, 1.0, 1.0, 2.0, 2.0, 3.0])


def test_a_deep_depth_is_kept() -> None:
    depths = [2.0, 25.0, 25.5, 70.0, 3.0]

    filled = fill_reach_depths(line_graph(1), {1: depths}, {1: np.arange(5)})

    np.testing.assert_array_equal(filled[1], depths)


def test_depths_are_filled_in_down_the_order_given() -> None:
    filled = fill_reach_depths(line_graph(1), {1: [0.0, 1.0, 0.0, 2.0]}, {1: np.array([3, 2, 1, 0])})

    np.testing.assert_array_equal(filled[1], [1.0, 1.0, 2.0, 2.0])


def test_a_reach_s_first_depths_take_the_shallowest_last_depth_flowing_in() -> None:
    network = nx.DiGraph([(1, 3), (2, 3)])

    filled = fill_reach_depths(network, {1: [1.0, 1.5], 2: [0.8], 3: [0.0, -1.0, 2.0]},
                               {1: np.arange(2), 2: np.arange(1), 3: np.arange(3)})

    np.testing.assert_array_equal(filled[3], [0.8, 0.8, 2.0])


def test_with_no_depth_before_it_a_depth_becomes_half_a_metre() -> None:
    filled = fill_reach_depths(line_graph(1), {1: [0.0, math.nan, 1.2]}, {1: np.arange(3)})

    np.testing.assert_array_equal(filled[1], [0.5, 0.5, 1.2])


def test_a_nan_depth_is_filled_in_and_so_is_the_next_with_the_depth_before_them() -> None:
    depths = [1.2, math.nan, -1.0, 1.3]

    filled = fill_reach_depths(line_graph(1, 2), {1: depths, 2: [1.0]}, {1: np.arange(4), 2: np.arange(1)})

    np.testing.assert_array_equal(filled[1], [1.2, 1.2, 1.2, 1.3])


# --- Smoothing a reach's bed --------------------------------------------------------------------------------------


def test_the_bed_is_the_median_of_the_five_cross_sections_around_it() -> None:
    beds = np.array([10.0, 11.0, 9.0, 14.0, 8.0, 12.0])

    smoothed = smooth_reach_bed(beds, CELL * np.arange(6), max_bed_grade=None)

    np.testing.assert_array_equal(smoothed, [10.0, 10.5, 10.0, 11.0, 10.5, 12.0])


def test_the_bed_cap_is_the_steepest_stream_slope() -> None:
    assert MAX_BED_GRADE == MAX_SLOPE == 0.5


def test_a_bed_cap_of_1_cm_a_metre_holds_the_bed_back_on_a_reach_falling_2_percent() -> None:
    """A 1 m channel down a reach falling 2%: capped at 1 cm a metre, its bed falls 1 m per 100 m, so the channel is
    gone after 120 m (1.2 m deep at first, from the median at the reach's end)."""
    banks = 100.0 - 0.02 * CELL * np.arange(30)

    capped = smooth_reach_bed(banks - 1.0, CELL * np.arange(30), max_bed_grade=GRADE)
    uncapped = smooth_reach_bed(banks - 1.0, CELL * np.arange(30), max_bed_grade=None)

    np.testing.assert_allclose((banks - capped)[[0, 5, 10, 12]], [1.2, 0.7, 0.2, 0.0], atol=1e-9)
    assert np.all(banks[12:] - capped[12:] <= 1e-9)
    np.testing.assert_allclose((banks - uncapped)[2:-2], 1.0, atol=1e-9)
    np.testing.assert_allclose((banks - uncapped)[[0, -1]], [1.2, 0.8], atol=1e-9)


def test_at_the_default_cap_the_bed_follows_a_reach_falling_2_percent() -> None:
    banks = 100.0 - 0.02 * CELL * np.arange(30)

    smoothed = smooth_reach_bed(banks - 1.0, CELL * np.arange(30))

    np.testing.assert_allclose(smoothed, smooth_reach_bed(banks - 1.0, CELL * np.arange(30), max_bed_grade=None))


def test_the_cap_allows_at_least_a_tenth_of_a_metre_between_cross_sections() -> None:
    smoothed = smooth_reach_bed([100.0, 90.0, 90.0], [0.0, 0.0, 0.0], window=1, max_bed_grade=GRADE)

    np.testing.assert_allclose(smoothed, [100.0, 99.999, 99.998])


def test_the_first_bed_is_held_to_the_lowest_bed_flowing_in() -> None:
    smoothed = smooth_reach_bed([95.0, 95.0], [0.0, 10.0], window=1, inflow_bed=98.0, inflow_distance=10.0,
                                max_bed_grade=GRADE)

    np.testing.assert_allclose(smoothed, [97.9, 97.8])


def test_a_missing_bed_is_left_out_of_the_median_and_the_cap() -> None:
    """The medians are 10.5, 10, none, 11 and 11, and the cap lets them change 0.1 m per 10 m, across the gap too."""
    smoothed = smooth_reach_bed([10.0, 11.0, math.nan, 9.0, 13.0], [0.0, 10.0, 20.0, 30.0, 40.0],
                                max_bed_grade=GRADE)

    np.testing.assert_allclose(smoothed, [10.5, 10.4, math.nan, 10.6, 10.7])


def test_a_bed_cap_or_median_window_that_can_t_be_raises() -> None:
    with pytest.raises(ValueError):
        smooth_reach_bed([1.0, 2.0], [0.0, 1.0], max_bed_grade=-0.01)
    with pytest.raises(ValueError):
        smooth_reach_bed([1.0, 2.0], [0.0, 1.0], window=0)
    with pytest.raises(ValueError):
        smooth_reach_bed([1.0, 2.0], [0.0])


# --- Along the network --------------------------------------------------------------------------------------------


def test_the_depth_is_the_bank_elevation_s_height_above_the_smoothed_bed() -> None:
    """Beds of 99, 98.9, 96.8, 98.7 and 98.6 m, whose medians are 98.9, 98.8, 98.7, 98.65 and 98.6."""
    cells = along_row([(1, 5), (2, 1)])
    banks = [100.0, 99.9, 99.8, 99.7, 99.6, 99.5]

    depths, beds = smoothed_depths(line_graph(1, 2), cells, banks, [1.0, 1.0, 3.0, 1.0, 1.0, 1.0])

    np.testing.assert_allclose(beds[:5], [98.9, 98.8, 98.7, 98.65, 98.6])
    np.testing.assert_allclose(depths[:5], [1.1, 1.1, 1.1, 1.05, 1.0])


def test_a_window_of_one_and_no_cap_leave_each_bed_as_it_is() -> None:
    """What the pipeline does with Bathy_Smooth_Bed off: the missing depth is filled in, and nothing else changes."""
    cells = along_row([(1, 5), (2, 1)])
    banks = [100.0, 99.9, 99.8, 99.7, 99.6, 99.5]

    depths, beds = smoothed_depths(line_graph(1, 2), cells, banks, [1.0, 1.0, 3.0, math.nan, 1.0, 1.0], window=1,
                                   max_bed_grade=None)

    np.testing.assert_allclose(depths, [1.0, 1.0, 3.0, 3.0, 1.0, 1.0])
    np.testing.assert_allclose(beds[:5], [99.0, 98.9, 96.8, 96.7, 98.6])


def test_where_the_bed_is_above_the_bank_elevation_the_depth_is_0() -> None:
    cells = along_row([(1, 3), (2, 1)])

    depths, _ = smoothed_depths(line_graph(1, 2), cells, [100.0, 100.0, 90.0, 90.0], [1.0] * 4)

    assert depths[2] == 0.0


def test_without_bank_elevations_the_bed_is_the_stream_cell_s_elevation_less_the_depth() -> None:
    """Stream cells 0.3 m up and down about smooth banks: the bed smoothed is the bed carved, so it comes out
    smooth."""
    rng = np.random.default_rng(0)
    cells = along_row([(1, 50), (2, 1)])
    banks = 100.0 - 0.005 * CELL * np.arange(51)
    thalwegs = banks - 1.0 + rng.normal(0.0, 0.3, 51)

    depths, beds = smoothed_depths(line_graph(1, 2), cells, banks, [0.8] * 51, thalwegs=thalwegs, use_banks=False)

    known = depths > 0.0
    np.testing.assert_allclose((thalwegs - depths)[known], beds[known])
    assert np.std(np.diff(thalwegs[:50] - depths[:50])) < 0.1


def test_a_reach_without_neighbours_is_smoothed_along_its_cells_in_metres() -> None:
    """A reach by itself falling 0.5%, a 1 m channel, its cross sections 10 m apart along the stream."""
    cells = along_row([(1, 50)])
    banks = 100.0 - 0.005 * CELL * np.arange(50)

    depths, _ = smoothed_depths(line_graph(1), cells, banks, [1.0] * 50)

    np.testing.assert_allclose(depths[[0, 25, 49]], [1.05, 1.0, 0.95], atol=1e-9)


def test_where_the_banks_drop_3_m_the_capped_reach_below_starts_without_a_channel() -> None:
    """Reach 1 falls into reach 2, whose banks are 3 m lower. Capped, reach 2's first bed is held within 0.1 m of
    reach 1's last, 10 m upstream, and falls 1 cm a metre from there, so the channel starts 220 m down. Uncapped, it
    is 1 m deep all along."""
    network = line_graph(1, 2, 3)
    cells = along_row([(1, 20), (2, 40)])
    banks = 100.0 - 0.001 * CELL * np.arange(60) - np.where(np.arange(60) >= 20, 3.0, 0.0)

    capped, beds = smoothed_depths(network, cells, banks, [1.0] * 60)
    uncapped, _ = smoothed_depths(network, cells, banks, [1.0] * 60, max_bed_grade=None)

    assert beds[20] == pytest.approx(beds[19] - 0.1)
    assert np.flatnonzero(capped[20:] > 0.0)[0] == 22
    np.testing.assert_allclose(uncapped[22:58], 1.0, atol=1e-9)


def test_at_the_default_cap_the_reach_below_a_3_m_drop_has_its_channel_from_the_start() -> None:
    network = line_graph(1, 2, 3)
    cells = along_row([(1, 20), (2, 40)])
    banks = 100.0 - 0.001 * CELL * np.arange(60) - np.where(np.arange(60) >= 20, 3.0, 0.0)

    depths, _ = smoothed_depths(network, cells, banks, [1.0] * 60, max_bed_grade=MAX_BED_GRADE)

    assert np.all(depths[20:58] > 0.9)


def test_the_first_bed_is_held_to_the_lower_of_two_reaches_flowing_in() -> None:
    """Reaches 1 and 2, with beds at 98 and 97 m, join to make reach 3, whose own bed is at 95 m. Its first cross
    section is a cell diagonal from each of their last."""
    network = nx.DiGraph([(1, 3), (2, 3)])
    cells = [(1, 0, k) for k in range(5)] + [(2, 2, k) for k in range(5)] + [(3, 1, 5 + k) for k in range(5)]
    banks = [100.0] * 10 + [97.0] * 5
    depths = [2.0] * 5 + [3.0] * 5 + [2.0] * 5

    _, beds = smoothed_depths(network, cells, banks, depths)

    assert beds[10] == pytest.approx(97.0 - 0.01 * math.hypot(CELL, CELL))


def test_a_missing_bank_elevation_takes_out_only_its_own_cross_section() -> None:
    """It's left out of the running median and the cap, and its cross section keeps its depth."""
    cells = along_row([(1, 50), (2, 1)])
    banks = 100.0 - 0.005 * CELL * np.arange(51)
    banks[20] = math.nan

    depths, beds = smoothed_depths(line_graph(1, 2), cells, banks, [1.0] * 51)

    assert np.isnan(beds[20]) and depths[20] == 1.0
    assert not np.isnan(beds[:20]).any() and not np.isnan(beds[21:]).any()


def test_depths_come_back_in_the_order_the_cross_sections_were_given() -> None:
    rng = np.random.default_rng(2)
    cells = along_row([(1, 30), (2, 1)])
    banks = list(100.0 - 0.01 * CELL * np.arange(31) + rng.normal(0.0, 0.1, 31))
    depths = list(rng.uniform(0.5, 2.0, 31))
    shuffle = rng.permutation(30)

    expected, _ = smoothed_depths(line_graph(1, 2), cells, banks, depths)
    got, _ = smoothed_depths(line_graph(1, 2), [cells[k] for k in shuffle] + cells[30:],
                             [banks[k] for k in shuffle] + banks[30:], [depths[k] for k in shuffle] + depths[30:])

    np.testing.assert_array_equal(got[:30], expected[shuffle])


def test_reaches_that_can_t_be_smoothed_raise() -> None:
    cells = along_row([(1, 3)])
    reaches, smoothed, _ = network_inputs(line_graph(1), cells, [100.0] * 3)

    with pytest.raises(ValueError, match="no cross sections"):
        smooth_channel_depths(line_graph(1), reaches, smoothed, {2: [1.0]}, CELL, CELL, use_banks=True)
    with pytest.raises(ValueError, match="3 cross sections"):
        smooth_channel_depths(line_graph(1), reaches, smoothed, {1: [1.0, 1.0]}, CELL, CELL, use_banks=True)
    with pytest.raises(ValueError, match="loop"):
        smooth_channel_depths(nx.DiGraph([(1, 1)]), reaches, smoothed, {1: [1.0] * 3}, CELL, CELL, use_banks=True)
    with pytest.raises(ValueError, match="aren't in the stream network"):
        smooth_channel_depths(line_graph(5), reaches, smoothed, {1: [1.0] * 3}, CELL, CELL, use_banks=True)
