"""Flow-through threads (Flow_Through_Threads): the hydraulics spreading water from a river's other threads, finding
them on the water mask, carving them, and the workflow with them (arc.xsection.threads)."""
from __future__ import annotations

import numpy as np
import pytest
from osgeo import gdal, osr
from test_pipeline import _write_raster, write_inputs

from arc import hydraulics, pipeline
from arc.bathymetry import Banks
from arc.bathymetry.channel import carve_channel, carved_depth
from arc.config import Configs
from arc.hydraulics import (
    connection_levels,
    hydraulic_connect,
    hydraulic_geometry,
    wse_for_discharge,
)
from arc.rating_curve import rating_curve, section_hydraulics, section_hydraulics_at
from arc.standard_step import HydraulicTable
from arc.xsection.stream_path import stream_raster_cells, stream_row_starts
from arc.xsection.threads import (
    MAX_THREAD_RISE,
    ThreadFinder,
    in_threads,
    outside,
    reach_out,
)
from arc.xsection.xsection import XSection

N = 0.035
SLOPE = 0.001
NO_THREADS = np.empty((0, 2))


def make(z, spacing=1.0, threads=None) -> XSection:
    xs = XSection(np.asarray(z, np.float64), np.full(len(z), N), spacing)
    if threads is not None:
        xs.threads = np.asarray(threads, np.float64)
    return xs


def bar_section(bar: float = 2.0) -> XSection:
    """Two channels 10 m wide and 3 m deep either side of a bar 6 m wide topping out bar metres up: the stream cell's
    from -5 to 5 m, the other's from 11 to 21 m, each ordinate 1 m apart; walls 3 m up beyond."""
    stations = np.arange(-30, 31)
    z = np.full(stations.size, 3.0)
    z[(stations >= -5) & (stations <= 5)] = 0.0
    z[(stations >= 11) & (stations <= 21)] = 0.0
    z[(stations >= 6) & (stations <= 10)] = bar
    return make(z)


THREAD = [(10.5, 21.5)]  # the other channel's water's edges, half way to the dry ordinates either side


# --- Connection levels and the hydraulics -----------------------------------------------------------------------


def test_without_threads_the_connection_levels_are_the_highest_ground_out_from_the_stream_cell() -> None:
    z = np.array([5.0, 1.0, 3.0, 0.0, 2.0, 1.0, 4.0])
    stations = np.arange(7.0) - 3.0
    # Then the outermost spreading vertices' indices: the stream cell's, either side
    np.testing.assert_array_equal(connection_levels(stations, z, 3, NO_THREADS),
                                  [5.0, 3.0, 3.0, 0.0, 2.0, 2.0, 4.0, 3.0, 3.0])


def test_a_thread_s_vertices_spread_water_from_their_own_ground() -> None:
    z = np.array([5.0, 1.0, 3.0, 0.0, 2.0, 1.0, 4.0])
    stations = np.arange(7.0) - 3.0
    levels = connection_levels(stations, z, 3, np.array([[-2.5, -1.5]]))
    np.testing.assert_array_equal(levels, [5.0, 1.0, 3.0, 0.0, 2.0, 2.0, 4.0, 1.0, 3.0])


@pytest.mark.parametrize("seed", range(40))
def test_today_s_connection_levels_give_today_s_hydraulics(seed: int) -> None:
    # Bit for bit, but for the water surface solve, whose intervals are bracketed apart
    rng = np.random.default_rng(seed)
    n = int(rng.integers(5, 50))
    stations = np.maximum.accumulate(np.cumsum(rng.uniform(0.0, 3.0, n) * (rng.random(n) > 0.1)))
    center = int(rng.integers(1, n - 1))
    stations = stations - stations[center]
    z = np.cumsum(rng.normal(0.0, 0.3, n))
    z[center] = z.min() - 0.5
    n0 = rng.uniform(0.02, 0.1, n)
    banks = (-1.0, -1.0) if seed % 3 == 0 else tuple(rng.uniform(0.0, 10.0, 2))
    roughness = (1.0, 1.0, 1.0) if seed % 2 else (6.0, 2.0, 1.0)
    connect = connection_levels(stations, z, center, NO_THREADS)
    for wse in z[center] + rng.uniform(0.01, 4.0, 4):
        old = hydraulics.profile_compound_geometry(stations, z, n0, center, *banks, wse, *roughness)
        new = hydraulics.profile_compound_geometry(stations, z, n0, center, *banks, wse, *roughness, connect)
        np.testing.assert_array_equal(np.array(new), np.array(old))
    for target in rng.uniform(1.0, 300.0, 3):
        old = hydraulics.wse_for_profile_conveyance(stations, z, n0, center, *banks, target, *roughness)
        new = hydraulics.wse_for_profile_conveyance(stations, z, n0, center, *banks, target, *roughness, np.inf,
                                                    connect)
        assert new == pytest.approx(old, abs=1e-9, nan_ok=True)
    levels = np.sort(z[center] + rng.uniform(0.0, 4.0, 6))
    # Each side's sums are bit for bit; compiled into the levels' function, the conveyance can differ in its last bits
    np.testing.assert_allclose(section_hydraulics_at(stations, z, n0, center, *banks, levels, *roughness, connect),
                               section_hydraulics_at(stations, z, n0, center, *banks, levels, *roughness),
                               rtol=1e-14, atol=0.0)


@pytest.mark.parametrize("seed", range(30))
def test_with_threads_the_levels_together_are_the_levels_one_at_a_time(seed: int) -> None:
    rng = np.random.default_rng(100 + seed)
    n = int(rng.integers(8, 60))
    stations = np.maximum.accumulate(np.cumsum(rng.uniform(0.0, 3.0, n) * (rng.random(n) > 0.1)))
    center = int(rng.integers(1, n - 1))
    stations = stations - stations[center]
    z = np.cumsum(rng.normal(0.0, 0.5, n))
    n0 = rng.uniform(0.02, 0.1, n)
    banks = (-1.0, -1.0) if seed % 3 == 0 else tuple(rng.uniform(0.0, 10.0, 2))
    roughness = (1.0, 1.0, 1.0) if seed % 2 else (6.0, 2.0, 1.0)
    lows = rng.choice(n, 3, replace=False)
    threads = np.sort(np.array([[stations[i] - 0.1, stations[i] + 0.1] for i in lows]), axis=0)
    connect = connection_levels(stations, z, center, threads)
    levels = np.sort(z[center] + rng.uniform(0.0, 4.0, 9))
    together = section_hydraulics_at(stations, z, n0, center, *banks, levels, *roughness, connect)
    one_at_a_time = np.array([section_hydraulics(stations, z, n0, center, *banks, level, *roughness, connect)
                              for level in levels])
    np.testing.assert_array_equal(together, one_at_a_time)


@pytest.mark.parametrize("depth", [0.5, 1.0, 1.9])
def test_a_thread_holds_water_below_the_bar_s_top(depth: float) -> None:
    alone = hydraulic_geometry(bar_section(), depth=depth)
    both = hydraulic_geometry(make(bar_section().elevations, threads=THREAD), depth=depth)
    assert both.area == pytest.approx(2.0 * alone.area, rel=1e-12)
    assert both.top_width == pytest.approx(2.0 * alone.top_width, rel=1e-12)


def test_above_the_bar_s_top_a_thread_changes_nothing() -> None:
    alone = hydraulic_geometry(bar_section(), depth=2.5)
    both = hydraulic_geometry(make(bar_section().elevations, threads=THREAD), depth=2.5)
    for a, b in zip(alone, both):
        assert b == pytest.approx(a, rel=1e-12)


def test_two_alike_channels_share_the_flow_at_one_water_surface() -> None:
    # Lumped as one subsection, two alike channels have twice the area and wetted perimeter, so the same hydraulic
    # radius: twice the conveyance of one, which carries half the flow
    q = 20.0
    both = wse_for_discharge(make(bar_section().elevations, threads=THREAD), q, SLOPE)
    one = wse_for_discharge(bar_section(), q / 2.0, SLOPE)
    assert both == pytest.approx(one, abs=1e-9)
    assert both < wse_for_discharge(bar_section(), q, SLOPE)


def test_where_the_flow_needs_the_bar_overtopped_the_lowest_water_surface_is_its_top() -> None:
    # Below the top the single channel can't carry it; just above, the other channel joins
    xs = bar_section(bar=1.0)
    q_top = hydraulics.discharge(xs, SLOPE, wse=1.0)
    q_over = hydraulics.discharge(make(xs.elevations, threads=THREAD), SLOPE, wse=1.0 + 1e-9)
    assert q_over > q_top * 1.5
    assert wse_for_discharge(xs, 1.2 * q_top, SLOPE) == pytest.approx(1.0, abs=1e-9)


def test_a_thread_s_water_spreads_out_from_it_onto_lower_ground() -> None:
    z = bar_section().elevations.copy()
    z[52:56] = 0.5  # low ground beyond the thread, joined to it below the wall
    xs = make(z, threads=THREAD)
    geometry = hydraulic_geometry(xs, depth=1.0)
    alone = hydraulic_geometry(make(bar_section().elevations, threads=THREAD), depth=1.0)
    assert geometry.top_width > alone.top_width + 3.0


def test_the_connection_levels_follow_the_cross_section_s_profile() -> None:
    xs = make(bar_section().elevations, threads=THREAD)
    first = hydraulic_connect(xs)
    assert hydraulic_connect(xs) is first  # kept
    xs.elevations = xs.elevations.copy()
    xs.elevations[25] = -1.0
    assert hydraulic_connect(xs) is not first  # worked out again for new ground
    xs.threads = None
    assert hydraulic_connect(xs) is None


def test_the_rating_curve_and_the_standard_step_s_table_take_the_threads() -> None:
    alone, both = bar_section(), make(bar_section().elevations, threads=THREAD)
    curve_alone = rating_curve(alone, 30.0, SLOPE, 10)
    curve_both = rating_curve(both, 30.0, SLOPE, 10)
    assert curve_both.max_wse < curve_alone.max_wse
    table_alone, table_both = HydraulicTable(alone, 2.0), HydraulicTable(both, 2.0)
    below = table_both.levels < 1.9
    np.testing.assert_allclose(table_both.area[below], 2.0 * table_alone.area[below], rtol=1e-12)


def test_the_hydraulics_dry_at_or_below_the_stream_cell_with_threads_too() -> None:
    xs = make(bar_section().elevations - np.where(np.arange(61) > 40, 1.0, 0.0), threads=THREAD)
    assert hydraulic_geometry(xs, depth=0.0).area == 0.0


# --- Finding threads -----------------------------------------------------------------------------------------------


def river_grid(kind: str):
    """A river along row 20 of a 41 x 121 grid of 10 m cells, its water rows 19 to 21, and beside it, rows 26 to 28
    from column 40 to 80: 'bar' joined to the river at both ends, 'slough' at its downstream end only, 'stream' a bar
    thread with a stream line of its own, 'pond' joined nowhere."""
    mask = np.zeros((41, 121), np.uint8)
    mask[19:22, :] = 1
    mask[26:29, 40:81] = 1
    if kind in ("bar", "stream"):
        mask[21:27, 38:41] = 1
        mask[21:27, 80:83] = 1
    elif kind == "slough":
        mask[21:27, 80:83] = 1
    streams = np.zeros(mask.shape, np.int32)
    streams[20, :] = 7
    if kind == "stream":
        streams[27, 40:81] = 8
    return mask, streams


def finder_for(mask, streams, network=None) -> ThreadFinder:
    rows, cols, ids = stream_raster_cells(streams)
    return ThreadFinder(mask, (stream_row_starts(rows, mask.shape[0]), cols), ids, 10.0, 10.0, network)


def column_section(col: int):
    """A cross section down column col of river_grid, 10 m ordinates on the cells, its middle on the river's row 20:
    its cells, and ground 5 m up but for the river's water and the water beside it, at 0."""
    z = np.full(41, 5.0)
    z[18:23] = 0.0
    z[25:30] = 0.0
    return np.arange(41), np.full(41, col), z


@pytest.mark.parametrize("kind, expected", [("bar", 1), ("slough", 0), ("stream", 0), ("pond", 0)])
def test_a_thread_joins_the_river_both_ways(kind: str, expected: int) -> None:
    mask, streams = river_grid(kind)
    finder = finder_for(mask, streams)
    rows, cols, z = column_section(60)
    threads = finder.threads(rows, cols, z, 10.0, 20, 60, 0.0, 7)  # the river runs along the rows
    assert len(threads) == expected
    if expected:
        # Rows 26 to 28: 6 to 8 ordinates right of the stream cell, edges half way to the dry ordinates
        np.testing.assert_allclose(threads[0], [55.0, 85.0])


def test_a_thread_far_above_the_stream_cell_is_a_terrace_s_water() -> None:
    mask, streams = river_grid("bar")
    rows, cols, z = column_section(60)
    z[25:30] = MAX_THREAD_RISE + 0.5
    assert len(finder_for(mask, streams).threads(rows, cols, z, 10.0, 20, 60, 0.0, 7)) == 0


def long_thread_grid(kind: str, cell: float):
    """A river 3.4 km long along the x axis, its water from y = -15 to 15 m and its stream line at y = 0, and beside
    it a thread from y = 45 to 75 m and x = 200 to 3200 m: 'bar' joined to the river at both ends, 'slough' at its
    downstream end only. Cells of this size; the grid's row i0 is at y = 0."""
    i0 = round(40.0 / cell)
    y = (np.arange(round(140.0 / cell) + 1) - i0) * cell
    x = (np.arange(round(3400.0 / cell)) + 0.5) * cell
    X, Y = np.meshgrid(x, y)
    water = (np.abs(Y) < 15.0) | ((Y > 45.0) & (Y < 75.0) & (X > 200.0) & (X < 3200.0))
    ends = [(3170.0, 3200.0)] if kind == "slough" else [(200.0, 230.0), (3170.0, 3200.0)]
    for low, high in ends:
        water |= (Y > -15.0) & (Y < 75.0) & (X > low) & (X < high)
    streams = np.zeros(water.shape, np.int32)
    streams[i0, :] = 7
    return water.astype(np.uint8), streams, i0


@pytest.mark.parametrize("cell", [10.0, 5.0, 2.5])
@pytest.mark.parametrize("kind, expected", [("bar", 1), ("slough", 0)])
def test_threads_are_the_same_at_any_cell_size(kind: str, expected: int, cell: float) -> None:
    # The thread joins the river 1.5 km either side of the cross section: within THREAD_REACH whatever the cells
    mask, streams, i0 = long_thread_grid(kind, cell)
    rows, cols, ids = stream_raster_cells(streams)
    finder = ThreadFinder(mask, (stream_row_starts(rows, mask.shape[0]), cols), ids, cell, cell)
    col = int(1700.0 / cell)
    half = mask.shape[0] - 1 - i0
    section_rows = np.arange(i0 - half, i0 + half + 1)
    z = np.where([0 <= r < mask.shape[0] and mask[r, col] > 0 for r in section_rows], 0.0, 5.0)
    threads = finder.threads(section_rows, np.full(section_rows.size, col), z, cell, i0, col, 0.0, 7)
    assert len(threads) == expected
    if expected:
        np.testing.assert_allclose(threads[0], [45.0, 75.0], atol=cell)


def test_the_river_is_its_reach_and_its_neighbours() -> None:
    import networkx as nx
    network = nx.DiGraph([(1, 2), (3, 2), (2, 4)])
    finder = finder_for(*river_grid("bar"), network)
    np.testing.assert_array_equal(finder.river(2), [1, 2, 3, 4])
    np.testing.assert_array_equal(finder.river(4), [2, 4])


def test_threads_beyond_the_banks() -> None:
    threads = np.array([[-30.0, -12.0], [-8.0, 6.0], [4.0, 9.0]])
    np.testing.assert_allclose(outside(threads, 10.0, 5.0), [[-30.0, -12.0], [5.0, 6.0], [5.0, 9.0]])
    assert outside(np.array([[-3.0, 2.0]]), 10.0, 5.0) is None
    assert reach_out(threads) == (30.0, 9.0)
    xs = make(np.zeros(11), threads=[[2.0, 3.0]])
    assert in_threads(xs).tolist() == [False] * 7 + [True, True] + [False] * 2


# --- Carving them --------------------------------------------------------------------------------------------------


def test_a_thread_is_carved_as_deep_as_the_channel_lowering_only() -> None:
    xs = make(bar_section().elevations, threads=THREAD)
    banks = Banks("water_edge", 5.5, 5.5, 0.0, 0.0, False, True)
    changed = carve_channel(xs, banks, 2.0, trapezoid_height=0.0)
    stations = np.arange(61) - 30
    thread = (stations >= 11) & (stations <= 21)
    np.testing.assert_allclose(xs.elevations[thread], -2.0)
    assert changed[thread].all()
    assert (xs.elevations[(stations >= 6) & (stations <= 10)] == 2.0).all()  # the bar stays


def test_with_a_thread_the_channel_is_carved_shallower_and_carries_the_flow_at_its_level() -> None:
    banks = Banks("water_edge", 5.5, 5.5, 0.0, 0.0, False, True)
    alone = bar_section()
    both = make(bar_section().elevations, threads=THREAD)
    q = 40.0
    d_alone = carved_depth(alone, banks, q, SLOPE, 1.0, trapezoid_height=0.1)
    d_both = carved_depth(both, banks, q, SLOPE, 1.0, trapezoid_height=0.1,
                          divide=banks._replace(right=21.5))
    assert 0.0 < d_both < d_alone
    carve_channel(both, banks, d_both, trapezoid_height=0.1)
    both.left_bank_distance, both.right_bank_distance = 5.5, 21.5
    assert hydraulics.discharge(both, SLOPE, wse=0.0 + 1e-12) == pytest.approx(q, rel=1e-6)


# --- The workflow -------------------------------------------------------------------------------------------------


def bar_valley():
    """A straight valley of 10 m cells falling 0.001 down its columns: a channel 160 m wide and 3 m deep, with a bar
    600 m long and 40 m wide whose top is 2 m above the bed splitting it into two threads, the stream line along the
    left one. The water mask is the channel's water below the bar's top."""
    ny, nx = 60, 200
    x = (np.arange(nx) + 0.5) * 10.0
    y = (np.arange(ny) + 0.5) * 10.0 - 300.0
    X, Y = np.meshgrid(x, y)
    ground = 100.0 - 0.001 * X
    channel = np.abs(Y) <= 80.0
    bar = (np.abs(Y - 10.0) <= 20.0) & (X > 700.0) & (X < 1300.0)
    dem = np.where(channel, ground - 3.0, ground)
    dem = np.where(bar, ground - 1.0, dem)
    mask = (channel & ~bar).astype(np.uint8)
    center = int(np.argmin(np.abs(y - 0.0)))
    left = int(np.argmin(np.abs(y + 45.0)))
    rows = np.where((x > 700.0) & (x < 1300.0), left, center)
    streams = np.zeros(dem.shape, np.int32)
    path = []
    for j, r in enumerate(rows):
        if path and abs(path[-1][0] - r) > 1:
            step = 1 if r > path[-1][0] else -1
            path += [(rr, j - 1) for rr in range(path[-1][0] + step, r, step)]
        path.append((int(r), j))
    path = np.array(path)
    streams[path[:, 0], path[:, 1]] = 1
    return dem, mask, streams, path, x


def run_bar_valley(folder, *, threads: bool, mask: bool = True, bathymetry: bool = False):
    dem, water, streams, path, _ = bar_valley()
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(32612)
    gt = (500000.0, 10.0, 0.0, 4000000.0, 0.0, -10.0)
    reaches = {1: {"start": tuple(path[0]), "end": tuple(path[-1]), "downstream": -1, "baseflow": 100.0,
                   "q_max": 1500.0}}
    inputs = write_inputs(folder, dem, streams, srs, gt, reaches, water_n=N, land_n=N)
    inputs.update({"VDT_Database_NumIterations": 20, "Gen_Slope_Dist": 25, "Flow_Through_Threads": threads})
    if mask:
        _write_raster(folder / "mask.tif", water, gt, srs, gdal.GDT_Byte)
        inputs["BathyWaterMask"] = str(folder / "mask.tif")
    if bathymetry:
        inputs.update({"BATHY_Out_File": str(folder / "bathy.tif"), "Bathy_Trap_H": 0.1})
    return pipeline.run(Configs.from_mapping(inputs), quiet=True, write=False)


def wse_at(results, k, q):
    increments = results.curves.increments[k]
    ok = np.isfinite(increments[:, 0])
    return float(np.interp(q, increments[ok, 0], increments[ok, 3]))


def in_bar(results, x):
    return np.array([700.0 < x[c] < 1300.0 for c in results.cells.cols])


def test_the_workflow_finds_the_bar_s_other_thread_and_lowers_the_water_beside_it(tmp_path) -> None:
    *_, x = bar_valley()
    on = run_bar_valley(tmp_path / "on", threads=True)
    off = run_bar_valley(tmp_path / "off", threads=False)
    bar = in_bar(on, x)
    threaded = np.array([s is not None and s.xs.threads is not None for s in on.sections])
    # None away from the bar: the stream line's step across to the left thread, at the bar's head, cuts across the
    # right thread's start
    clear = np.array([not 650.0 < x[c] < 1350.0 for c in on.cells.cols])
    assert threaded[bar].mean() > 0.8 and not threaded[clear].any()
    middle = [k for k in np.flatnonzero(bar) if 800.0 < x[on.cells.cols[k]] < 1200.0]
    for k in middle:
        assert wse_at(on, k, 150.0) < wse_at(off, k, 150.0) - 0.1  # below the bar's top: the flow shares the threads
        assert wse_at(on, k, 1500.0) == pytest.approx(wse_at(off, k, 1500.0), abs=0.1)
    far = [k for k in np.flatnonzero(~bar) if x[on.cells.cols[k]] > 1500.0]
    for k in far:
        np.testing.assert_array_equal(on.curves.increments[k], off.curves.increments[k])


def test_without_a_water_mask_there_are_no_threads(tmp_path) -> None:
    on = run_bar_valley(tmp_path / "on", threads=True, mask=False)
    off = run_bar_valley(tmp_path / "off", threads=False, mask=False)
    assert all(s is None or s.xs.threads is None for s in on.sections)
    np.testing.assert_array_equal(on.curves.increments, off.curves.increments)


def test_the_bathymetry_carves_the_threads_and_the_channel_shallower(tmp_path) -> None:
    *_, x = bar_valley()
    on = run_bar_valley(tmp_path / "on", threads=True, bathymetry=True)
    off = run_bar_valley(tmp_path / "off", threads=False, bathymetry=True)
    bar = in_bar(on, x)
    middle = [k for k in np.flatnonzero(bar) if 800.0 < x[on.cells.cols[k]] < 1200.0
              and on.sections[k].xs.threads is not None]
    assert middle
    for k in middle:
        assert on.sections[k].carve_depth < off.sections[k].carve_depth


def test_the_standard_step_and_the_representative_cross_sections_take_the_threads(tmp_path) -> None:
    *_, x = bar_valley()
    extra = {"Standard_Step": True, "Build_Representative_Cross_Section": True,
             "Representative_Cross_Section_File": str(tmp_path / "representative.csv"), "Stream_Length_m": "length_m"}

    def run(folder, threads):
        dem, water, streams, path, _ = bar_valley()
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(32612)
        gt = (500000.0, 10.0, 0.0, 4000000.0, 0.0, -10.0)
        reaches = {1: {"start": tuple(path[0]), "end": tuple(path[-1]), "downstream": -1, "baseflow": 100.0,
                       "q_max": 1500.0}}
        inputs = write_inputs(folder, dem, streams, srs, gt, reaches, water_n=N, land_n=N)
        _write_raster(folder / "mask.tif", water, gt, srs, gdal.GDT_Byte)
        inputs.update({"VDT_Database_NumIterations": 20, "Gen_Slope_Dist": 25, "Flow_Through_Threads": threads,
                       "BathyWaterMask": str(folder / "mask.tif"), **extra})
        return pipeline.run(Configs.from_mapping(inputs), quiet=True, write=False)

    on, off = run(tmp_path / "on", True), run(tmp_path / "off", False)
    assert on.representative is not None and len(on.representative)
    middle = [k for k in np.flatnonzero(in_bar(on, x)) if 800.0 < x[on.cells.cols[k]] < 1200.0]
    assert np.median([wse_at(off, k, 150.0) - wse_at(on, k, 150.0) for k in middle]) > 0.1
