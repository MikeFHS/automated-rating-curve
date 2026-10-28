from __future__ import annotations

import math

import networkx as nx
import numpy as np
import pytest

from arc.bathymetry import Banks, banks_for_width, single_cell_banks
from arc.bathymetry.smoothing import (
    BANK_HEIGHT_QUANTILE,
    BANK_HEIGHT_WINDOW,
    JUNCTION_TAPER,
    MIN_BANK_OBSERVATIONS,
    ReachSections,
    ReachWidths,
    SmoothedReach,
    below_ceilings,
    falling_fit,
    filter_reach_widths,
    order_reach,
    reach_bank_levels,
    reach_bank_observations,
    reach_network,
    smooth_bank_elevations,
    taper_junctions,
)
from arc.xsection.stream_path import along_stream_stations
from arc.xsection.xsection import XSection

BED = 100.0
NO_BANKS = Banks("none", math.nan, math.nan, math.nan, math.nan, False, False)


def flat_section(spacing=1.0, ordinates=101, bed=BED) -> XSection:
    elevations = np.full(ordinates, bed)
    return XSection(elevations, np.full(ordinates, 0.035), spacing)


def section_with_banks(width, spacing=1.0) -> tuple[XSection, Banks]:
    xs = flat_section(spacing)
    return xs, banks_for_width(xs, width)


def banks_at(left_elevation, right_elevation, valid=True) -> Banks:
    return Banks("target_width", 5.0, 5.0, left_elevation, right_elevation, False, valid)


def line_graph(*reaches) -> nx.DiGraph:
    network = nx.DiGraph()
    network.add_nodes_from(reaches)
    network.add_edges_from(zip(reaches[:-1], reaches[1:]))
    return network


# --- The width filter ---------------------------------------------------------------------------------------------


def test_outlying_widths_are_rebuilt_at_the_reach_median() -> None:
    """Widths of 10, 20, 20, 20 and 40 m have their 25th, 50th and 75th percentiles all at 20 m, so the 10 and 40 m
    channels are rebuilt 20 m wide, half each side. The others keep their own banks, even at a percentile."""
    sections, banks = map(list, zip(*(section_with_banks(w) for w in [10.0, 20.0, 20.0, 20.0, 40.0])))
    banks[2] = Banks("width_to_depth_ratio", 5.0, 15.0, BED, BED, False, True)

    filtered, widths = filter_reach_widths(sections, banks)

    assert widths == ReachWidths(20.0, 20.0, 20.0)
    assert [b.top_width for b in filtered] == pytest.approx([20.0] * 5)
    assert filtered[1:4] == banks[1:4]
    assert filtered[0].method == filtered[4].method == "target_width"
    assert (filtered[0].left, filtered[0].right) == pytest.approx((10.0, 10.0))


def test_banks_where_the_water_ends_are_kept_whatever_their_width() -> None:
    """Banks from the water mask or the land cover show the channel's width, so they aren't rebuilt, though they count
    towards the percentiles. With widths of 10, 10 and 10 m from the water, a 40 m channel from the DEM is rebuilt
    10 m wide. Among channels 20 m wide from the DEM, a 10 m one is rebuilt 20 m wide, unless it's from the water."""
    sections, banks = map(list, zip(*(section_with_banks(w) for w in [10.0, 10.0, 10.0, 40.0])))
    water = [b._replace(method="land_cover") for b in banks[:3]]

    filtered, widths = filter_reach_widths(sections, [*water, banks[3]])

    assert widths == ReachWidths(10.0, 10.0, 17.5)
    assert filtered[:3] == water
    assert filtered[3].method == "target_width" and filtered[3].top_width == pytest.approx(10.0)

    sections, banks = map(list, zip(*(section_with_banks(w) for w in [10.0, 20.0, 20.0, 20.0, 20.0, 10.0])))
    banks[0] = banks[0]._replace(method="land_cover")

    filtered, widths = filter_reach_widths(sections, banks)

    assert widths == ReachWidths(12.5, 20.0, 20.0)
    assert filtered[0] == banks[0]
    assert filtered[5].method == "target_width" and filtered[5].top_width == pytest.approx(20.0)


def test_a_median_between_one_and_two_spacings_is_rebuilt_as_it_is() -> None:
    """The median is 10 m, on a cross section with 8 m spacing: banks needn't be whole spacings apart."""
    sections, banks = map(list, zip(*(section_with_banks(w) for w in [10.0, 10.0, 10.0, 20.0])))
    wide = flat_section(spacing=8.0)
    sections.append(wide)
    banks.append(banks_for_width(wide, 40.0))

    filtered, widths = filter_reach_widths(sections, banks)

    assert widths == ReachWidths(10.0, 10.0, 20.0)
    assert (filtered[4].left, filtered[4].right) == pytest.approx((5.0, 5.0))
    assert filtered[4].valid and filtered[4].method == "target_width"


def test_a_median_wider_than_a_cross_section_holds_is_as_wide_as_it_can_be() -> None:
    """With walls a metre and two metres either side of the stream cell, a 10 m median makes a channel 3 m wide."""
    sections, banks = map(list, zip(*(section_with_banks(w) for w in [10.0, 10.0, 10.0, 12.0])))
    walled = flat_section()
    walled.elevations[:49] = walled.elevations[53:] = 9999.0
    sections.append(walled)
    banks.append(Banks("width_to_depth_ratio", 1.0, 1.0, BED, BED, False, True))

    filtered, _ = filter_reach_widths(sections, banks)

    assert (filtered[4].left, filtered[4].right) == pytest.approx((1.0, 2.0))
    assert filtered[4].valid


def test_a_channel_that_can_t_be_rebuilt_is_left_alone() -> None:
    """With no ordinate on the raster on one side of the stream cell, no channel there is valid."""
    sections, banks = map(list, zip(*(section_with_banks(w) for w in [10.0, 10.0, 10.0, 12.0])))
    walled = flat_section(spacing=8.0)
    walled.elevations[:50] = 9999.0
    outlier = Banks("width_to_depth_ratio", 1.0, 40.0, BED, BED, False, True)
    sections.append(walled)
    banks.append(outlier)

    filtered, _ = filter_reach_widths(sections, banks)

    assert filtered[4] == outlier


def test_cross_sections_without_valid_banks_get_the_median_width() -> None:
    sections, banks = map(list, zip(*(section_with_banks(w) for w in [10.0, 12.0, 14.0])))
    sections += [flat_section(), flat_section(spacing=8.0)]
    banks += [NO_BANKS, NO_BANKS]

    filtered, widths = filter_reach_widths(sections, banks)

    assert widths.median == pytest.approx(12.0)
    assert filtered[3].valid and filtered[3].top_width == pytest.approx(12.0)
    assert filtered[4].valid and filtered[4].top_width == pytest.approx(12.0)  # one and a half of its spacings


def test_a_reach_without_valid_banks_is_left_alone() -> None:
    sections = [flat_section(), flat_section()]

    assert filter_reach_widths(sections, [NO_BANKS, NO_BANKS]) == ([NO_BANKS, NO_BANKS], None)


def test_a_single_cell_channel_is_one_spacing_wide() -> None:
    """The stream cell's own width, so a reach of them has a median of one spacing."""
    sections = [flat_section(spacing=10.0) for _ in range(4)]
    banks = [single_cell_banks(xs) for xs in sections]

    assert banks[0].top_width == pytest.approx(10.0)
    assert filter_reach_widths(sections, banks)[1] == ReachWidths(10.0, 10.0, 10.0)


# --- Observations -------------------------------------------------------------------------------------------------


def test_an_observation_is_the_lower_bank_above_the_stream_cell() -> None:
    """A bank at or below the stream cell is ignored, and banks that aren't valid give no observation."""
    sections = [flat_section() for _ in range(5)]
    banks = [banks_at(103.0, 102.0), banks_at(99.0, 102.5), banks_at(BED, 101.5), banks_at(BED, BED),
             banks_at(104.0, 104.0, valid=False)]

    observations, lower, upper = reach_bank_observations(sections, banks)

    np.testing.assert_array_equal(observations, [102.0, 102.5, 101.5, np.nan, np.nan])
    assert (lower, upper) == (-math.inf, math.inf)


def test_observations_outside_the_2nd_and_97th_percentiles_are_left_out() -> None:
    elevations = [100.5, 101.0, 101.0, 101.0, 101.0, 110.0]
    sections = [flat_section(bed=99.0) for _ in elevations]

    observations, lower, upper = reach_bank_observations(sections, [banks_at(z, z) for z in elevations])

    assert (lower, upper) == pytest.approx(tuple(np.percentile(elevations, [2, 97])))
    np.testing.assert_array_equal(observations, [np.nan, 101.0, 101.0, 101.0, 101.0, np.nan])


def test_fewer_than_four_observations_are_all_kept() -> None:
    sections = [flat_section(bed=99.0) for _ in range(3)]

    observations, lower, upper = reach_bank_observations(sections, [banks_at(z, z) for z in [100.0, 101.0, 150.0]])

    np.testing.assert_array_equal(observations, [100.0, 101.0, 150.0])
    assert (lower, upper) == (-math.inf, math.inf)


def test_a_bank_from_the_dem_above_its_ceiling_is_left_out() -> None:
    """Above the water that carries the largest flow, a bank from the DEM is a valley wall. The water mask's banks
    stay, and so does an observation without a ceiling."""
    observations = [102.0, 103.0, 104.0, 105.0, 101.0]
    banks = [banks_at(0.0, 0.0), banks_at(0.0, 0.0)._replace(method="land_cover"), banks_at(0.0, 0.0),
             banks_at(0.0, 0.0)._replace(method="width_to_depth_ratio"), banks_at(0.0, 0.0)]
    ceilings = [101.0, 101.0, math.nan, 106.0, 101.0]

    np.testing.assert_array_equal(below_ceilings(observations, banks, ceilings),
                                  [np.nan, 103.0, 104.0, 105.0, 101.0])
    np.testing.assert_array_equal(below_ceilings(observations, banks, None), observations)


# --- Ordering a reach's cross sections ----------------------------------------------------------------------------


def hairpin():
    """Cells east along row 10, down through column 16, and back west along row 13."""
    cells = [(10, c) for c in range(5, 16)] + [(11, 16), (12, 16)] + [(13, c) for c in range(15, 4, -1)]
    return np.array([r for r, _ in cells]), np.array([c for _, c in cells])


def test_a_reach_is_ordered_towards_the_reach_downstream_of_it() -> None:
    rows, cols = hairpin()
    network = line_graph(1, 2)

    order, stations = order_reach(network, 1, rows[::-1], cols[::-1], 10.0, 10.0,
                                  {1: (rows, cols), 2: (np.array([13]), np.array([4]))}, np.zeros(rows.size))

    assert order.tolist() == list(range(rows.size - 1, -1, -1))
    # The reach's stations (arc.xsection.stream_path), which round the bend a little inside its cells' path
    assert stations[-1] == pytest.approx(along_stream_stations(rows, cols, 10.0, 10.0).max(), rel=1e-12)
    assert 200.0 < stations[-1] < 200.0 + 2 * math.hypot(10, 10) + 10.0


def test_an_outlet_is_ordered_away_from_the_reach_upstream_of_it() -> None:
    rows, cols = hairpin()
    network = line_graph(9, 1)

    order, _ = order_reach(network, 1, rows, cols, 10.0, 10.0, {1: (rows, cols), 9: (np.array([13]), np.array([4]))},
                           np.zeros(rows.size))

    assert order.tolist() == list(range(rows.size - 1, -1, -1))


def test_the_first_reach_downstream_with_cross_sections_sets_the_downstream_end() -> None:
    """Reach 1 splits into reaches 2 and 3, and reach 2 has no cross sections, so reach 3 sets the end."""
    rows, cols = hairpin()
    network = nx.DiGraph()
    network.add_edges_from([(1, 2), (1, 3)])
    cells = {1: (rows, cols), 2: (np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)),
             3: (np.array([13]), np.array([4]))}

    order, _ = order_reach(network, 1, rows, cols, 10.0, 10.0, cells, np.zeros(rows.size))

    assert order.tolist() == list(range(rows.size))


def test_a_reach_by_itself_is_ordered_along_its_cells_from_its_highest_observation() -> None:
    """Along its cells, not a straight line through them, which would mix up the hairpin's two arms, and with
    stations in metres: 10 m a cell along its straight arm, up to the cells the bend's smoothing reaches
    (arc.xsection.stream_path.SMOOTHING_CELLS)."""
    rows, cols = hairpin()
    observations = np.linspace(90.0, 110.0, rows.size)  # rising along the cells, so the far end is upstream
    network = nx.DiGraph()
    network.add_node(1)

    order, stations = order_reach(network, 1, rows, cols, 10.0, 10.0, {1: (rows, cols)}, observations)

    assert order.tolist() == list(range(rows.size - 1, -1, -1))
    np.testing.assert_allclose(np.diff(stations)[:7], 10.0)
    assert np.all(np.diff(stations) > 0.0)


def test_a_reach_whose_neighbours_have_no_cross_sections_is_turned_by_its_end_observations() -> None:
    """Its first ten cross sections' mean observation is higher than its last ten's, so that end is upstream."""
    rows, cols = np.zeros(30, dtype=np.int64), np.arange(30)
    observations = np.r_[np.full(10, 105.0), np.full(10, np.nan), np.full(10, 95.0)]
    network = line_graph(1, 2)

    order, stations = order_reach(network, 1, rows, cols, 10.0, 10.0, {1: (rows, cols)}, observations)

    assert order.tolist() == list(range(30))
    np.testing.assert_allclose(stations, 10.0 * np.arange(30))


@pytest.mark.parametrize("neighbour", ["downstream", "upstream"])
def test_a_reach_whose_neighbours_have_no_cross_sections_is_ordered_by_where_they_lie(neighbour: str) -> None:
    """Neither neighbour has cross sections, but on the stream raster the reach below lies beside its first cell (or
    the reach above beside its last), so that end is downstream, though the observations are higher there and the
    reach's line ends at the other."""
    rows, cols = np.zeros(30, dtype=np.int64), np.arange(1, 31)
    observations = np.linspace(105.0, 95.0, 30)
    network = line_graph(1, 2) if neighbour == "downstream" else line_graph(2, 1)
    stream_cells = {1: (rows, cols), 2: (np.array([0]), np.array([0 if neighbour == "downstream" else 31]))}

    order, stations = order_reach(network, 1, rows, cols, 10.0, 10.0, {1: (rows, cols)}, observations,
                                  stream_cells=stream_cells, outlets={1: (0, 31)})

    assert order.tolist() == list(range(29, -1, -1))
    np.testing.assert_allclose(stations, 10.0 * np.arange(30))


def test_a_reach_with_no_neighbour_on_the_stream_raster_is_ordered_by_where_its_line_ends() -> None:
    """Reach 2 is in the network but not on the stream raster, and the reach's line ends beside its first cell."""
    rows, cols = np.zeros(30, dtype=np.int64), np.arange(30)
    observations = np.linspace(105.0, 95.0, 30)
    network = line_graph(1, 2)

    order, _ = order_reach(network, 1, rows, cols, 10.0, 10.0, {1: (rows, cols)}, observations,
                           stream_cells={1: (rows, cols)}, outlets={1: (0, -1)})

    assert order.tolist() == list(range(29, -1, -1))


def test_a_reach_is_turned_by_the_observations_nearest_its_ends() -> None:
    """Its high end's last dozen cross sections have no observation (no valid banks there). The ten observations
    nearest each end still say which end is higher."""
    # Given from the high end, so the stations start at the low end and the reach has to be turned round
    rows, cols = np.zeros(40, dtype=np.int64), np.arange(39, -1, -1)
    observations = 95.0 + 10.0 * cols / 39.0
    observations[cols >= 28] = np.nan
    network = line_graph(1, 2)
    cells = {1: (rows, cols)}

    order, stations = order_reach(network, 1, rows, cols, 10.0, 10.0, cells, observations)

    assert order.tolist() == list(range(40))  # from column 39 down to column 0
    np.testing.assert_allclose(stations, 10.0 * np.arange(40))


# --- The network --------------------------------------------------------------------------------------------------


def test_a_network_is_built_from_the_stream_table() -> None:
    """Missing, self and unknown downstream IDs make outlets, and a reach's first row counts."""
    network = reach_network([1, 2, 3, 4, 2, 5], [2, 3, None, 99, 4, "4"])

    assert sorted(network.nodes) == [1, 2, 3, 4, 5]
    assert sorted(network.edges) == [(1, 2), (2, 3), (5, 4)]
    assert reach_network([7], [7]).out_degree(7) == 0


# --- The bank level: the water surface plus the banks' height -----------------------------------------------------


def reference_falling_fit(values) -> np.ndarray:
    """Pool adjacent violators the slow way: merge the first rising pair of runs at their mean, and start again."""
    runs = [[float(v)] for v in values]
    merged = True
    while merged:
        merged = False
        for k in range(len(runs) - 1):
            if np.mean(runs[k]) < np.mean(runs[k + 1]):
                runs[k:k + 2] = [runs[k] + runs[k + 1]]
                merged = True
                break
    return np.concatenate([[np.mean(run)] * len(run) for run in runs])


def test_the_falling_fit_pools_each_rise_at_its_mean() -> None:
    np.testing.assert_allclose(falling_fit([5.0, 3.0, 4.0, 2.0, 6.0, 1.0]), [5.0, 3.75, 3.75, 3.75, 3.75, 1.0])
    np.testing.assert_array_equal(falling_fit([3.0, 2.0, 2.0, 1.0]), [3.0, 2.0, 2.0, 1.0])


def test_the_falling_fit_is_the_least_squares_sequence_that_never_rises() -> None:
    rng = np.random.default_rng(0)
    for _ in range(300):
        n = int(rng.integers(1, 40))
        values = rng.normal(0.0, 1.0, n) - rng.uniform(0.0, 0.2) * np.arange(n)

        fitted = falling_fit(values)

        np.testing.assert_allclose(fitted, reference_falling_fit(values), rtol=0.0, atol=1e-12)
        assert np.all(np.diff(fitted) <= 1e-12)


def test_the_falling_fit_leaves_nan_out() -> None:
    np.testing.assert_allclose(falling_fit([3.0, math.nan, 5.0, 1.0]), [4.0, math.nan, 4.0, 1.0])


def test_the_bank_height_is_the_10th_percentile_within_500_m_or_of_the_5_nearest() -> None:
    assert (BANK_HEIGHT_QUANTILE, BANK_HEIGHT_WINDOW, MIN_BANK_OBSERVATIONS) == (0.1, 500.0, 5)


def test_a_steady_reach_s_bank_elevation_is_its_water_plus_its_banks_height() -> None:
    stations = 30.0 * np.arange(100)
    thalwegs = 100.0 - 0.001 * stations

    bank = reach_bank_levels(thalwegs, thalwegs + 1.5, stations, math.nan)

    np.testing.assert_allclose(bank, thalwegs + 1.5, atol=1e-9)


def test_the_height_comes_from_the_banks_nearby() -> None:
    """Banks 1 m above the water for the first kilometre and 2 m for the second: at each end, only its own are
    within 500 m."""
    stations = 10.0 * np.arange(201)
    thalwegs = 103.0 - 0.0015 * stations
    heights = np.where(stations < 1000.0, 1.0, 2.0)

    bank = reach_bank_levels(thalwegs, thalwegs + heights, stations, math.nan)

    assert bank[0] == pytest.approx(thalwegs[0] + 1.0)
    assert bank[-1] == pytest.approx(thalwegs[-1] + 2.0)


def test_the_height_is_a_low_percentile_of_the_banks() -> None:
    """Heights of 1 to 10 m in turn, all within 500 m of each other: their 10th percentile."""
    stations = 10.0 * np.arange(50)
    thalwegs = 100.0 - 0.01 * stations
    heights = 1.0 + np.arange(50) % 10

    bank = reach_bank_levels(thalwegs, thalwegs + heights, stations, math.nan)

    np.testing.assert_allclose(bank - thalwegs, np.quantile(heights, 0.1), atol=1e-9)


def test_a_reach_that_runs_mild_then_steep_keeps_its_banks_above_the_water() -> None:
    """A straight line from the reach's top to its outlet would pass under the whole mild stretch."""
    stations = 30.0 * np.arange(300)
    thalwegs = np.where(stations < 6000.0, 165.0 - 0.0001 * stations, 164.4 - 8.0 * (stations - 6000.0) / 3000.0)

    bank = reach_bank_levels(thalwegs, thalwegs + 1.5, stations, math.nan)

    np.testing.assert_allclose(bank, thalwegs + 1.5, atol=1e-9)


def test_the_bank_elevation_never_rises_downstream_or_goes_below_the_stream_cell() -> None:
    """A stream cell 2 m above its neighbours, as a bridge in the DEM: the water surface passes under it, and the
    bank elevation stops at the stream cell."""
    stations = 10.0 * np.arange(50)
    thalwegs = 100.0 - 0.001 * stations
    thalwegs[25] += 2.0

    bank = reach_bank_levels(thalwegs, thalwegs + 0.5, stations, math.nan)

    assert np.all(bank >= thalwegs)
    assert bank[25] == thalwegs[25]
    assert np.all(np.diff(np.delete(bank, 25)) <= 1e-12)


def test_observations_at_or_below_the_stream_cell_are_left_out() -> None:
    stations = 10.0 * np.arange(40)
    thalwegs = 100.0 - 0.01 * stations

    bank = reach_bank_levels(thalwegs, thalwegs + np.where(np.arange(40) % 2, -0.5, 1.0), stations, math.nan)

    np.testing.assert_allclose(bank, thalwegs + 1.0, atol=1e-9)


def test_a_reach_without_observations_takes_the_fallback_height() -> None:
    stations = 10.0 * np.arange(40)
    thalwegs = 100.0 - 0.01 * stations

    np.testing.assert_allclose(reach_bank_levels(thalwegs, np.full(40, math.nan), stations, 0.8), thalwegs + 0.8)
    assert np.isnan(reach_bank_levels(thalwegs, np.full(40, math.nan), stations, math.nan)).all()


def test_a_lone_high_bank_doesn_t_set_the_height_of_the_cross_sections_round_it() -> None:
    """A 0.9 m bank at km 1.5, the only one within 500 m of km 1.0-2.0, and 0.2 m banks every 100 m over the reach's
    first and last 400 m. The 5 nearest to any cross section hold at least four of the low ones, so the height is
    0.2 m all along; the 500 m window alone would have made it 0.9 m round km 1.5."""
    stations = 10.0 * np.arange(301)
    thalwegs = 100.0 - 0.001 * stations
    heights = np.full(301, math.nan)
    heights[[0, 10, 20, 30, 40, 260, 270, 280, 290, 300]] = 0.2
    heights[150] = 0.9

    bank = reach_bank_levels(thalwegs, thalwegs + heights, stations, math.nan)

    np.testing.assert_allclose(bank, thalwegs + 0.2, atol=1e-9)


def test_a_window_with_5_or_more_observations_takes_their_10th_percentile_as_before() -> None:
    """A bank every 100 m, so every 500 m window holds 6 to 11: each cross section's height is the 10th percentile of
    those, not of the 5 nearest."""
    stations = 10.0 * np.arange(101)
    thalwegs = 100.0 - 0.001 * stations
    heights = np.full(101, math.nan)
    heights[::10] = np.random.default_rng(1).uniform(0.2, 2.0, 11)
    observed = np.isfinite(heights)
    window = np.array([np.quantile(heights[observed & (np.abs(stations - s) <= 500.0)], 0.1) for s in stations])

    bank = reach_bank_levels(thalwegs, thalwegs + heights, stations, math.nan)

    np.testing.assert_allclose(bank, np.maximum(falling_fit(thalwegs + window), thalwegs), atol=1e-9)


def test_a_reach_with_fewer_than_5_observations_takes_their_10th_percentile_everywhere() -> None:
    """Banks 0.4, 1.0 and 1.6 m up at the ends and the middle of a 3 km reach: 0.52 m everywhere, even at the
    middle, whose 500 m window holds only the 1.0 m bank."""
    stations = 10.0 * np.arange(301)
    thalwegs = 100.0 - 0.001 * stations
    heights = np.full(301, math.nan)
    heights[[0, 150, 300]] = [0.4, 1.0, 1.6]

    bank = reach_bank_levels(thalwegs, thalwegs + heights, stations, 5.0)

    np.testing.assert_allclose(bank, thalwegs + np.quantile([0.4, 1.0, 1.6], 0.1), atol=1e-9)


def test_the_bank_height_needs_at_least_one_observation() -> None:
    stations = 10.0 * np.arange(40)
    thalwegs = 100.0 - 0.01 * stations
    with pytest.raises(ValueError):
        reach_bank_levels(thalwegs, thalwegs + 1.0, stations, math.nan, min_observations=0)


def made_up_reach(thalwegs, observations, cell=30.0, phantom=False):
    """A reach of cross sections one cell apart along a row, flowing into a reach of one cross section without an
    observation of its own (as below the Du Page reach), and with phantom an inflow with no cross sections."""
    count = len(thalwegs)
    network = line_graph(1, 2)
    if phantom:
        network.add_edge(0, 1)

    def section(z):
        elevations = np.full(9, z + 5.0)
        elevations[4] = z
        return XSection(elevations, np.full(9, 0.035), cell)

    def banks(z):
        return Banks("test", 30.0, 30.0, z, z, False, True)
    last = float(thalwegs[-1]) - 0.01
    reaches = {1: ReachSections(np.zeros(count, np.int64), np.arange(count), [section(z) for z in thalwegs],
                                [banks(z) for z in observations]),
               2: ReachSections(np.zeros(1, np.int64), np.array([count]), [section(last)], [banks(last)])}
    return network, reaches


def test_an_inflow_without_cross_sections_doesn_t_cap_the_reach() -> None:
    """Each reach's bank elevation comes from its own cross sections, so an inflow without any can't hold it down.
    (Its last 500 m are ramped towards reach 2's, 1 cm lower.)"""
    thalwegs = 100.0 - 0.001 * 30.0 * np.arange(300)
    network, reaches = made_up_reach(thalwegs, thalwegs + 1.5, phantom=True)

    smoothed = smooth_bank_elevations(network, reaches, 30.0, 30.0)[1]

    away = 30.0 * (299 - np.arange(300)) >= JUNCTION_TAPER
    np.testing.assert_allclose(smoothed.bank_elevations[away], (thalwegs + 1.5)[away], atol=1e-9)
    np.testing.assert_allclose(smoothed.bank_elevations, thalwegs + 1.5, atol=0.01)


def test_the_site_s_height_fills_in_a_reach_without_observations() -> None:
    """Reach 1's banks are 1.2 m above its water where they're above it at all; reach 3, flat, has no banks."""
    thalwegs = 100.0 - 0.001 * 30.0 * np.arange(60)
    network, reaches = made_up_reach(thalwegs, np.r_[thalwegs[:30] + 1.2, thalwegs[30:] - 1.0])
    flat = XSection(np.r_[np.full(4, 55.0), 50.0, np.full(4, 55.0)], np.full(9, 0.035), 30.0)
    reaches[3] = ReachSections(np.full(10, 5, np.int64), np.arange(10), [flat] * 10, [NO_BANKS] * 10)
    network.add_node(3)

    smoothed = smooth_bank_elevations(network, reaches, 30.0, 30.0)

    np.testing.assert_allclose(smoothed[3].bank_elevations, 51.2, atol=1e-9)


# --- Junctions ----------------------------------------------------------------------------------------------------


def flat_reach(level, count=20, spacing=50.0) -> SmoothedReach:
    """A smoothed reach of count cross sections spacing metres apart, upstream first, all with one bank elevation."""
    return SmoothedReach([NO_BANKS] * count, np.full(count, float(level)), np.full(count, math.nan), np.arange(count),
                         spacing * np.arange(count), None)


def test_two_reaches_meet_at_their_mean_and_are_left_alone_500_m_away() -> None:
    """Reach 1's banks at 104 m flow into reach 2's at 102 m: each 950 m long reach is ramped to 103 m at the junction,
    linearly over its 500 m nearest it."""
    smoothed = {1: flat_reach(104.0), 2: flat_reach(102.0)}

    tapered = taper_junctions(line_graph(1, 2), smoothed, {1: np.full(20, 100.0), 2: np.full(20, 100.0)})

    stations = 50.0 * np.arange(20)
    np.testing.assert_allclose(tapered[1].bank_elevations,
                               104.0 - np.clip(1.0 - (950.0 - stations) / JUNCTION_TAPER, 0.0, 1.0))
    np.testing.assert_allclose(tapered[2].bank_elevations, 102.0 + np.clip(1.0 - stations / JUNCTION_TAPER, 0.0, 1.0))
    assert tapered[1].bank_elevations[-1] == tapered[2].bank_elevations[0] == pytest.approx(103.0)
    np.testing.assert_array_equal(smoothed[1].bank_elevations, 104.0)  # the levels given are left as they were


def test_the_ramp_rises_to_a_higher_reach_below_but_never_goes_below_the_stream_cells() -> None:
    """1 (101 m) meets 2 (105 m) at 103 m, rising on the way; 2 meets 3 (99 m) at 102 m, but 2's stream cells are at
    104 m, so it stops there."""
    smoothed = {1: flat_reach(101.0), 2: flat_reach(105.0), 3: flat_reach(99.0)}
    thalwegs = {1: np.full(20, 100.0), 2: np.full(20, 104.0), 3: np.full(20, 96.0)}

    tapered = taper_junctions(line_graph(1, 2, 3), smoothed, thalwegs)

    assert tapered[1].bank_elevations[-1] == pytest.approx(103.0)
    assert np.all(np.diff(tapered[1].bank_elevations) >= 0.0)
    assert tapered[2].bank_elevations[0] == pytest.approx(104.0)  # 103 m, held at the stream cells
    assert tapered[2].bank_elevations[-1] == pytest.approx(104.0)  # 102 m, held at the stream cells
    assert tapered[3].bank_elevations[0] == pytest.approx(102.0)
    for reach, result in tapered.items():
        assert np.all(result.bank_elevations >= thalwegs[reach])


def test_only_the_main_stem_is_tapered_into_the_reach_below() -> None:
    """Reach 1, with reach 4 above it, is reach 3's main stem; reach 2, with none above it, keeps its levels."""
    network = reach_network([1, 2, 3, 4], [3, 3, None, 1])
    smoothed = {reach: flat_reach(level) for reach, level in {1: 106.0, 2: 109.0, 3: 104.0, 4: 108.0}.items()}

    tapered = taper_junctions(network, smoothed, {reach: np.full(20, 100.0) for reach in smoothed})

    np.testing.assert_array_equal(tapered[2].bank_elevations, 109.0)
    assert tapered[1].bank_elevations[-1] == tapered[3].bank_elevations[0] == pytest.approx(105.0)
    assert tapered[4].bank_elevations[-1] == tapered[1].bank_elevations[0] == pytest.approx(107.0)


def test_a_reach_without_bank_elevations_is_left_out_of_the_taper() -> None:
    smoothed = {1: flat_reach(104.0), 2: flat_reach(math.nan)}

    tapered = taper_junctions(line_graph(1, 2), smoothed, {1: np.full(20, 100.0), 2: np.full(20, 100.0)})

    np.testing.assert_array_equal(tapered[1].bank_elevations, 104.0)
    assert np.isnan(tapered[2].bank_elevations).all()


def test_the_junction_taper_must_be_positive() -> None:
    with pytest.raises(ValueError, match="above 0 m"):
        taper_junctions(line_graph(1, 2), {1: flat_reach(104.0)}, {1: np.full(20, 100.0)}, taper=0.0)


def two_reaches(height_above: float, height_below: float, count: int = 40, cell: float = 30.0):
    """Reach 1 flowing into reach 2 along a row of cells, count cross sections each, on a stream falling 1 m per km,
    with banks height_above and height_below above it. Also the stream cells' elevations, reach 1's first."""
    thalwegs = 100.0 - 0.001 * cell * np.arange(2 * count)

    def section(z):
        elevations = np.full(9, z + 5.0)
        elevations[4] = z
        return XSection(elevations, np.full(9, 0.035), cell)

    reaches = {}
    for reach, part, height in ((1, slice(0, count), height_above), (2, slice(count, 2 * count), height_below)):
        z = thalwegs[part]
        reaches[reach] = ReachSections(np.zeros(count, np.int64), np.arange(2 * count)[part], [section(v) for v in z],
                                       [Banks("test", cell, cell, v + height, v + height, False, True) for v in z])
    return line_graph(1, 2), reaches, thalwegs


def test_smoothed_reaches_meet_at_a_junction_and_keep_their_own_heights_away_from_it() -> None:
    """Reach 1's banks are 1 m above the stream and reach 2's 3 m: they meet 2 m above it at the junction."""
    network, reaches, thalwegs = two_reaches(1.0, 3.0)

    smoothed = smooth_bank_elevations(network, reaches, 30.0, 30.0)

    upper, lower = smoothed[1].bank_elevations, smoothed[2].bank_elevations
    assert upper[-1] == pytest.approx(lower[0])
    assert upper[-1] == pytest.approx(0.5 * (thalwegs[39] + 1.0 + thalwegs[40] + 3.0))
    stations = 30.0 * np.arange(40)
    far_above, far_below = stations[-1] - stations >= JUNCTION_TAPER, stations >= JUNCTION_TAPER
    np.testing.assert_allclose(upper[far_above], thalwegs[:40][far_above] + 1.0)
    np.testing.assert_allclose(lower[far_below], thalwegs[40:][far_below] + 3.0)


def test_unsmoothed_bank_elevations_aren_t_tapered() -> None:
    network, reaches, thalwegs = two_reaches(1.0, 3.0)

    smoothed = smooth_bank_elevations(network, reaches, 30.0, 30.0, smooth=False)

    np.testing.assert_allclose(smoothed[1].bank_elevations, thalwegs[:40] + 1.0)
    np.testing.assert_allclose(smoothed[2].bank_elevations, thalwegs[40:] + 3.0)


# --- Everything ---------------------------------------------------------------------------------------------------


def channel_section(bank_top, bed, width=20.0, spacing=2.0, ordinates=81) -> XSection:
    """A 1:1 channel whose sides rise from the bed to bank_top at width / 2 from the stream cell, then a floodplain
    rising 1 cm per metre."""
    distance = spacing * np.abs(np.arange(ordinates) - ordinates // 2)
    side = np.clip(distance - (width / 2 - (bank_top - bed)), 0.0, bank_top - bed)
    elevations = bed + side + 0.01 * np.maximum(distance - width / 2, 0.0)
    return XSection(elevations, np.full(ordinates, 0.035), spacing)


def small_network():
    """Reaches 1 and 2 join as reach 3, each of 20 cross sections along a row of 10 m cells."""
    network = reach_network([1, 2, 3], [3, 3, None])
    reaches = {}
    starts = {1: (0, 0), 2: (2, 0), 3: (1, 21)}
    tops = {1: 110.0, 2: 108.0, 3: 104.0}
    for reach, (row, col) in starts.items():
        sections = [channel_section(tops[reach] - 0.05 * k, tops[reach] - 2.0 - 0.05 * k) for k in range(20)]
        if reach == 3:
            sections[10] = channel_section(103.3, 101.3)  # a bank lower than the ones either side
        banks = [banks_for_width(xs, 20.0) for xs in sections]
        rows, cols = np.full(20, row), np.arange(col, col + 20)
        reaches[reach] = ReachSections(rows, cols, sections, banks)
    return network, reaches


def test_each_reach_s_bank_elevations_fall_downstream_and_stay_above_its_stream_cells() -> None:
    network, reaches = small_network()

    smoothed = smooth_bank_elevations(network, reaches, 10.0, 10.0)

    for reach, result in smoothed.items():
        ordered = result.bank_elevations[result.order]
        thalwegs = np.array([xs.elevations[xs.elevations.size // 2] for xs in reaches[reach].sections])[result.order]
        assert np.all(np.isfinite(ordered))
        assert np.all(np.diff(ordered) <= 1e-12)
        assert np.all(ordered >= thalwegs)


def test_results_come_back_in_the_order_the_cross_sections_were_given() -> None:
    network, reaches = small_network()
    expected = smooth_bank_elevations(network, reaches, 10.0, 10.0)[3]
    shuffle = np.random.default_rng(0).permutation(20)
    given = reaches[3]
    reaches[3] = ReachSections(given.rows[shuffle], given.cols[shuffle], [given.sections[k] for k in shuffle],
                               [given.banks[k] for k in shuffle])

    shuffled = smooth_bank_elevations(network, reaches, 10.0, 10.0)[3]

    np.testing.assert_allclose(shuffled.bank_elevations, expected.bank_elevations[shuffle], rtol=1e-15)
    np.testing.assert_array_equal(shuffled.observations, expected.observations[shuffle])
    assert shuffled.order.tolist() == np.argsort(shuffle)[expected.order].tolist()


def test_each_cross_section_keeps_its_own_banks() -> None:
    network, reaches = small_network()

    smoothed = smooth_bank_elevations(network, reaches, 10.0, 10.0)

    assert smoothed[1].banks == list(reaches[1].banks)
    assert smoothed[1].widths == ReachWidths(20.0, 20.0, 20.0)


def test_unsmoothed_each_cross_section_keeps_its_own_banks_and_bank_elevation() -> None:
    """With smooth=False, a channel twice as wide as the rest of its reach keeps its banks, reach 3's low bank stays
    where it is, and a cross section without banks takes its stream cell plus the reach's bank height (2 m here)."""
    network, reaches = small_network()
    reaches[1].banks[5] = banks_for_width(reaches[1].sections[5], 40.0)
    reaches[2].banks[3] = NO_BANKS

    smoothed = smooth_bank_elevations(network, reaches, 10.0, 10.0, smooth=False)

    assert smoothed[1].banks == list(reaches[1].banks) and smoothed[1].widths is None
    assert smooth_bank_elevations(network, reaches, 10.0, 10.0)[1].banks[5].top_width == pytest.approx(20.0)
    assert smoothed[3].bank_elevations[10] == pytest.approx(103.3)
    np.testing.assert_allclose(smoothed[2].bank_elevations, [108.0 - 0.05 * k for k in range(20)])
    assert sorted(smoothed[3].order.tolist()) == list(range(20))


def test_the_width_filter_s_banks_aren_t_observations() -> None:
    """A cross section without banks gets banks at the reach's median width, for the hydraulics, but the ground
    there isn't a bank: it gives no observation."""
    network, reaches = small_network()
    assert np.isfinite(smooth_bank_elevations(network, reaches, 10.0, 10.0)[1].observations[5])
    reaches[1].banks[5] = NO_BANKS

    smoothed = smooth_bank_elevations(network, reaches, 10.0, 10.0)[1]

    assert smoothed.banks[5].valid and smoothed.banks[5].top_width == pytest.approx(20.0)
    assert np.isnan(smoothed.observations[5])


def test_a_reach_whose_banks_are_walls_takes_the_site_s_bank_height() -> None:
    """Reach 2's banks are 3 m up, above its ceilings 2 m up, so it has no observation and takes the site's 1 m, from
    reach 1 (500 m from the junction, past the taper)."""
    network, reaches, thalwegs = two_reaches(1.0, 3.0)
    reaches[2] = reaches[2]._replace(ceilings=thalwegs[40:] + 2.0)

    smoothed = smooth_bank_elevations(network, reaches, 30.0, 30.0)

    assert np.isnan(smoothed[2].observations).all()
    far_below = 30.0 * np.arange(40) >= JUNCTION_TAPER
    np.testing.assert_allclose(smoothed[2].bank_elevations[far_below], thalwegs[40:][far_below] + 1.0, atol=1e-9)


def test_where_every_bank_is_above_its_ceiling_they_are_all_kept() -> None:
    """With the largest flow below every bank, as in deep channels without a water mask, the ceilings would leave
    the site without a bank elevation. The DEM's banks are all there is, so they stay."""
    network, reaches, thalwegs = two_reaches(1.0, 3.0)
    expected = smooth_bank_elevations(network, reaches, 30.0, 30.0)
    for reach, part in ((1, slice(0, 40)), (2, slice(40, 80))):
        reaches[reach] = reaches[reach]._replace(ceilings=thalwegs[part] + 0.5)

    smoothed = smooth_bank_elevations(network, reaches, 30.0, 30.0)

    for reach in (1, 2):
        np.testing.assert_array_equal(smoothed[reach].bank_elevations, expected[reach].bank_elevations)


def test_a_reach_with_cross_sections_must_be_in_the_network() -> None:
    network, reaches = small_network()
    network.remove_node(2)

    with pytest.raises(ValueError, match="aren't in the stream network: 2"):
        smooth_bank_elevations(network, reaches, 10.0, 10.0)


def test_without_an_observation_anywhere_no_reach_gets_a_bank_elevation() -> None:
    """What to do with that is up to the caller: the pipeline drops them only with Bathy_Use_Banks, which is where
    they're needed."""
    network, reaches = small_network()
    for reach, sections in reaches.items():
        reaches[reach] = sections._replace(banks=[NO_BANKS] * len(sections.banks))

    smoothed = smooth_bank_elevations(network, reaches, 10.0, 10.0)

    assert all(np.isnan(result.bank_elevations).all() for result in smoothed.values())
