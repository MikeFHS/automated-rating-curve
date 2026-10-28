from __future__ import annotations

import math

import networkx as nx
import numpy as np
import pytest
from shapely.geometry import LineString, MultiLineString

from arc.xsection import slope as slope_module
from arc.xsection.slope import (
    MIN_SLOPE,
    corrected_local_slopes,
    end_point_slope,
    fill_unresolved_reach_slopes,
    local_average_slopes,
    local_fit_slopes,
    reach_median_slope,
)
from arc.xsection.stream_path import (
    SMOOTHING_CELLS,
    along_stream_distances,
    along_stream_stations,
    downstream_order,
    stream_cells_by_reach,
)

CELL = 10.0
REACH = 7


def reach_raster(cells, shape=(60, 60)):
    """A stream raster holding REACH at the given (row, col) cells, and those cells' rows and columns."""
    streams = np.zeros(shape, dtype=np.int64)
    rows = np.array([r for r, _ in cells], dtype=np.int64)
    cols = np.array([c for _, c in cells], dtype=np.int64)
    streams[rows, cols] = REACH
    return streams, rows, cols


def hairpin():
    """A stream that runs east along row 10, turns down through column 16, and comes back west along row 13, so the
    two arms are 30 m apart but up to 238 m apart along the stream."""
    return ([(10, c) for c in range(5, 16)] + [(11, 16), (12, 16)] + [(13, c) for c in range(15, 4, -1)])


# --- Distances along the stream -----------------------------------------------------------------------------------


def test_distances_follow_the_stream_one_cell_at_a_time() -> None:
    """With 10 m wide, 20 m tall cells: a step along the row, one across a diagonal, and one down the column."""
    rows, cols = [0, 0, 1, 2], [0, 1, 2, 2]

    distances = along_stream_distances(rows, cols, 10.0, 20.0, 0)

    np.testing.assert_allclose(distances, [0.0, 10.0, 10.0 + math.hypot(10.0, 20.0), 30.0 + math.hypot(10.0, 20.0)],
                               rtol=1e-15)


def test_a_path_takes_the_diagonal_across_a_corner() -> None:
    """Round a corner of whole cells, the path cuts the corner cell's diagonal."""
    distances = along_stream_distances([0, 0, 1, 2], [0, 1, 1, 1], 1.0, 1.0, 3)

    np.testing.assert_allclose(distances, [1.0 + math.sqrt(2.0), 2.0, 1.0, 0.0], rtol=1e-15)


def test_a_gap_in_the_cells_is_jumped_in_a_straight_line() -> None:
    rows, cols = [0] * 8, [0, 1, 2, 3, 4, 8, 9, 10]

    np.testing.assert_allclose(along_stream_distances(rows, cols, CELL, CELL, 0), [0, 10, 20, 30, 40, 80, 90, 100])


def smoothed_stations(rows, cols, dx, dy, distances, cells=SMOOTHING_CELLS) -> np.ndarray:
    """The stations as arc.xsection.stream_path defines them, worked out plainly from each cell's distance along the
    shortest path from the start: the path back from the farthest cell, a step at a time to the neighbour on a
    shortest path nearest the start (the lowest row and then column of ties); each of its cells moved towards the
    mean of the cells up to `cells` either side of it along the path (as many each side, and not across a gap, a step
    more than a diagonal), but by no more than half a cell along a row or a column; the length along those points
    from the start, but across a gap the path's own jump; and every cell the station at its distance along the
    path."""
    index = {(int(r), int(c)): k for k, (r, c) in enumerate(zip(rows, cols))}
    current = int(np.argmax(distances))
    path = [current]
    while distances[current] > 0.0:
        r, c = rows[current], cols[current]
        steps = [index[(r + i, c + j)] for i in (-1, 0, 1) for j in (-1, 0, 1)
                 if (i or j) and (r + i, c + j) in index]
        steps = [v for v in steps if distances[v] < distances[current]
                 and abs(distances[v] + math.hypot((cols[v] - c) * dx, (rows[v] - r) * dy) - distances[current])
                 <= 1e-9 * (1.0 + distances[current])]
        current = min(steps, key=lambda v: (distances[v], rows[v], cols[v]))
        path.append(current)
    path = path[::-1]
    d = [distances[k] for k in path]
    points = [(cols[k] * dx, rows[k] * dy) for k in path]
    runs, start = [], 0
    for i in range(1, len(path) + 1):
        if i == len(path) or d[i] - d[i - 1] > 1.5 * math.hypot(dx, dy):
            runs.append((start, i - 1))
            start = i
    run_of = {i: run for run in runs for i in range(run[0], run[1] + 1)}
    means = []
    for i in range(len(path)):
        first, last = run_of[i]
        h = min(cells, i - first, last - i)
        mean = [np.mean([points[k][axis] for k in range(i - h, i + h + 1)]) for axis in (0, 1)]
        means.append((points[i][0] + np.clip(mean[0] - points[i][0], -dx / 2, dx / 2),
                      points[i][1] + np.clip(mean[1] - points[i][1], -dy / 2, dy / 2)))
    along = [0.0]
    for i in range(1, len(path)):
        along.append(along[-1] + (d[i] - d[i - 1] if run_of[i][0] == i else math.dist(means[i], means[i - 1])))
    return np.interp(distances, d, along)


def test_stations_run_from_one_end_of_the_reach_to_the_other() -> None:
    """The hairpin's cells shuffled: its stations run from 0 at one end to the other end, as the cells' path
    smoothed round the bend has them, which is shorter than the path through the cells' centres."""
    cells = hairpin()
    order = np.random.default_rng(0).permutation(len(cells))
    rows = np.array([cells[k][0] for k in order])
    cols = np.array([cells[k][1] for k in order])
    end = list(order).index(0)

    stations = along_stream_stations(rows, cols, CELL, CELL)

    expected = smoothed_stations(rows, cols, CELL, CELL, shortest_paths_from(rows, cols, end))
    np.testing.assert_allclose(stations, expected, rtol=0, atol=1e-9)
    path = 100.0 + math.hypot(10, 10) + 10.0 + math.hypot(10, 10) + 100.0  # along each arm, and round the bend
    assert stations[end] == 0.0 and stations[list(order).index(len(cells) - 1)] == stations.max() < path
    assert stations.max() > 200.0  # the arms are 100 m long each, and stay so


@pytest.mark.parametrize("degrees", [0.0, 5.0, 10.0, 15.0, 22.5, 30.0, 37.5, 45.0, 60.0, 90.0])
def test_the_stations_of_a_straight_stream_at_any_angle_are_its_length(degrees: float) -> None:
    """A straight stream 300 cells long crossing the grid at an angle: its cells' path is up to 8% longer (at 22.5
    degrees), the stations within 0.3% of the stream's length, and exact along a row, a column or a diagonal."""
    angle = math.radians(degrees)
    cells = []
    for t in np.arange(0.0, 300.0, 0.05):
        cell = (int(round(t * math.sin(angle))), int(round(t * math.cos(angle))))
        if not cells or cells[-1] != cell:
            cells.append(cell)
    rows, cols = np.array([r for r, _ in cells]), np.array([c for _, c in cells])
    length = CELL * math.hypot(rows[-1] - rows[0], cols[-1] - cols[0])

    stations = along_stream_stations(rows, cols, CELL, CELL)

    assert stations.max() == pytest.approx(length, rel=1e-12 if degrees in (0.0, 45.0, 90.0) else 0.003)


def test_a_gap_keeps_its_jump_and_the_stations_don_t_smooth_across_it() -> None:
    """Columns 0 to 4 along row 0, then 8 to 10 along row 1: each run is straight, and the jump between them is the
    straight line from column 4 to column 8."""
    rows, cols = np.array([0, 0, 0, 0, 0, 1, 1, 1]), np.array([0, 1, 2, 3, 4, 8, 9, 10])

    stations = along_stream_stations(rows, cols, CELL, CELL)

    jump = CELL * math.hypot(4, 1)
    expected = np.array([0, 10, 20, 30, 40, 40 + jump, 50 + jump, 60 + jump])
    np.testing.assert_allclose(np.abs(stations - stations[0]), expected, atol=1e-9)


def test_a_reach_is_ordered_from_the_cell_by_the_reach_downstream() -> None:
    """A reach that bends on its way to the reach below it, starting at (0, 0). The path from the downstream end runs
    (2, 1), (1, 1) and across the diagonal to (0, 0); (1, 1) moves to the mean of the three, (2/3, 1) in (col, row),
    so the steps are sqrt(10)/3 and sqrt(13)/3. The corner cell (0, 1), off the path, is 2 m along the shortest path
    from the downstream end, 1/sqrt(2) of the way along the diagonal step."""
    order, stations = downstream_order([0, 0, 1, 2], [0, 1, 1, 1], 1.0, 1.0, downstream_cells=([3], [1]))

    assert order.tolist() == [0, 1, 2, 3]
    first, second = math.sqrt(10.0) / 3.0, math.sqrt(13.0) / 3.0
    total = first + second
    np.testing.assert_allclose(stations, [0.0, total - (first + second / math.sqrt(2.0)), second, total], rtol=1e-12)


def test_a_reach_without_one_downstream_is_ordered_away_from_the_one_upstream() -> None:
    """The cell nearest the reach upstream is the upstream end, and the cell farthest from it downstream."""
    cells = hairpin()
    rows = np.array([r for r, _ in cells])[::-1]
    cols = np.array([c for _, c in cells])[::-1]

    order, stations = downstream_order(rows, cols, CELL, CELL, upstream_cells=([10], [4]))

    assert (rows[order[0]], cols[order[0]]) == (10, 5)
    assert (rows[order[-1]], cols[order[-1]]) == (13, 5)
    assert np.all(np.diff(stations) > 0.0)


def test_a_reach_needs_a_reach_next_to_it_to_know_its_downstream_end() -> None:
    assert downstream_order([0, 0], [0, 1], CELL, CELL) is None
    assert downstream_order([0, 0], [0, 1], CELL, CELL, downstream_cells=([], [])) is None
    order, stations = downstream_order([5], [5], CELL, CELL)
    assert order.tolist() == [0] and stations.tolist() == [0.0]


def shortest_paths_from(rows, cols, source: int) -> np.ndarray:
    """Each cell's distance from cell `source` along the shortest path through neighbouring cells, by networkx."""
    index = {(int(r), int(c)): k for k, (r, c) in enumerate(zip(rows, cols))}
    graph = nx.Graph()
    graph.add_nodes_from(range(len(index)))
    for (r, c), k in index.items():
        for dr, dc in ((0, 1), (1, -1), (1, 0), (1, 1)):
            j = index.get((r + dr, c + dc))
            if j is not None:
                graph.add_edge(k, j, weight=CELL * math.hypot(dr, dc))
    lengths = nx.single_source_dijkstra_path_length(graph, source)
    return np.array([lengths[k] for k in range(len(index))])


def test_winding_reaches_are_ordered_by_their_distance_along_the_stream_to_the_reach_below() -> None:
    """Meandering reaches, their cells shuffled, ending next to the reach below: the order runs from the cell
    farthest along the stream from the reach below to the one nearest it, and the stations are how far along the
    smoothed path."""
    rng = np.random.default_rng(11)
    compared = 0
    while compared < 60:
        heading = rng.uniform(0.0, 2.0 * math.pi)
        x = y = 0.0
        cells = [(0, 0)]
        for _ in range(int(rng.integers(20, 1500))):
            heading += rng.normal(0.0, 0.15)
            x += math.cos(heading) / 3
            y += math.sin(heading) / 3
            cell = (int(round(y)), int(round(x)))
            if cell == cells[-1]:
                continue
            if cell in cells:
                break
            cells.append(cell)
        beyond = (cells[-1][0] + int(round(math.sin(heading) * 1.4)), cells[-1][1] + int(round(math.cos(heading) * 1.4)))
        if len(cells) < 2 or beyond in cells or beyond == cells[-1]:
            continue
        shuffle = rng.permutation(len(cells))
        rows = np.array([cells[k][0] for k in shuffle]) + 200
        cols = np.array([cells[k][1] for k in shuffle]) + 200
        nearest = int(np.argmin(np.hypot(rows - beyond[0] - 200, cols - beyond[1] - 200)))  # the first of any ties

        order, stations = downstream_order(rows, cols, CELL, CELL,
                                           downstream_cells=([beyond[0] + 200], [beyond[1] + 200]))

        to_end = shortest_paths_from(rows, cols, nearest)
        assert sorted(order.tolist()) == list(range(len(cells)))
        assert np.all(np.diff(to_end[order]) <= 1e-9)
        smoothed = smoothed_stations(rows, cols, CELL, CELL, to_end)
        np.testing.assert_allclose(stations, smoothed.max() - smoothed[order], rtol=0, atol=1e-9)
        compared += 1


def test_cells_are_grouped_by_reach() -> None:
    streams = np.zeros((5, 6), dtype=np.int64)
    streams[1, 1:4] = 3
    streams[3, 2:5] = 1

    cells = stream_cells_by_reach(streams)

    assert sorted(cells) == [1, 3]
    assert cells[3][0].tolist() == [1, 1, 1] and cells[3][1].tolist() == [1, 2, 3]
    assert cells[1][0].tolist() == [3, 3, 3] and cells[1][1].tolist() == [2, 3, 4]


# --- Slopes -------------------------------------------------------------------------------------------------------


def hairpin_slopes(distance: int):
    """The hairpin with a bed falling exactly 1 mm per metre along the stream, and both slopes of each kind."""
    streams, rows, cols = reach_raster(hairpin())
    stations = along_stream_stations(rows, cols, CELL, CELL)
    dem = np.full(streams.shape, 500.0)
    dem[rows, cols] = 100.0 - 0.001 * (stations - stations.min())
    z = dem[rows, cols]
    local = local_average_slopes(z, rows, cols, stations, CELL, CELL, distance)
    median = reach_median_slope(z, rows, cols, stations, CELL, CELL, distance)
    return local, median


def test_slopes_are_the_drop_over_the_distance_along_the_stream() -> None:
    """Across the hairpin's two arms the straight line is 30 m but the stream up to 238 m. Measured over the
    straight line, the slopes between the arms would be up to eight times the bed's; along the stream every slope is
    the bed's."""
    local, (median, lower, upper) = hairpin_slopes(distance=5)

    np.testing.assert_allclose(local, 0.001, rtol=1e-9)
    assert (median, lower, upper) == pytest.approx((0.001, 0.001, 0.001), rel=1e-9)


def test_the_local_average_reaches_as_far_downstream_as_upstream() -> None:
    """On a straight row, the average at column 15 of the slopes to columns 10 to 20."""
    cells = [(10, c) for c in range(31)]
    streams, rows, cols = reach_raster(cells)
    dem = np.full(streams.shape, 500.0)
    dem[rows, cols] = 100.0 - 0.001 * (cols - 15.0) ** 2
    stations = along_stream_stations(rows, cols, CELL, CELL)

    local = local_average_slopes(dem[rows, cols], rows, cols, stations, CELL, CELL, 5)

    def mean_slope(columns):
        return np.mean([abs(dem[10, 15] - dem[10, c]) / (CELL * abs(c - 15)) for c in columns if c != 15])

    assert local[15] == pytest.approx(mean_slope(range(10, 21)), rel=1e-12)


@pytest.mark.parametrize(("drop", "expected"), [(0.0, MIN_SLOPE), (0.5 * MIN_SLOPE * CELL, MIN_SLOPE),
                                                 (2e-5 * CELL, 2e-5), (100.0, 0.5)])
def test_local_averages_are_kept_between_0p00001_and_0p5(drop: float, expected: float) -> None:
    rows, cols = np.array([0, 0]), np.array([0, 1])
    z = np.array([100.0, 100.0 - drop])

    assert local_average_slopes(z, rows, cols, along_stream_stations(rows, cols, CELL, CELL), CELL, CELL, 5) \
        == pytest.approx([expected, expected])


def test_a_cell_without_another_within_the_distance_has_no_local_average() -> None:
    """Columns 0 to 2 and 9: column 9 is 7 cells from the others, beyond the distance of 5."""
    rows, cols = np.zeros(4, dtype=np.int64), np.array([0, 1, 2, 9])
    z = np.array([100.0, 99.99, 99.98, 99.91])

    slopes = local_average_slopes(z, rows, cols, along_stream_stations(rows, cols, CELL, CELL), CELL, CELL, 5)

    np.testing.assert_allclose(slopes[:3], 0.001, rtol=1e-9)
    assert math.isnan(slopes[3])


def test_the_reach_median_is_of_the_slopes_between_its_percentiles() -> None:
    """Six cells along a row make 15 pairs. The flat pair is left out, and the median is of the slopes between the
    25th and 75th percentiles."""
    rows, cols = np.zeros(6, dtype=np.int64), np.arange(6)
    z = np.array([10.0, 9.9, 9.9, 9.6, 9.5, 9.0])
    stations = along_stream_stations(rows, cols, CELL, CELL)

    slopes = np.round([abs(z[i] - z[k]) / (CELL * (k - i)) for i in range(6) for k in range(i + 1, 6)], 8)
    slopes = slopes[slopes > 0.0]
    lower, upper = np.round(np.percentile(slopes, [25, 75]), 8)
    kept = slopes[(slopes >= lower) & (slopes <= upper)]

    assert slopes.size == 14 and 0 < kept.size < slopes.size
    assert reach_median_slope(z, rows, cols, stations, CELL, CELL, 10, 25, 75) == pytest.approx(
        (np.median(kept), lower, upper), rel=1e-12)


def test_a_reach_with_two_slopes_takes_their_median() -> None:
    """Three cells with one flat pair leave slopes of 0.05 and 0.1, neither between their percentiles (0.0625 and
    0.0875)."""
    dem = np.zeros((3, 5))
    dem[1, 1:4] = [10.0, 10.0, 9.0]
    rows, cols = np.array([1, 1, 1]), np.array([1, 2, 3])

    result = reach_median_slope(dem[rows, cols], rows, cols, along_stream_stations(rows, cols, CELL, CELL), CELL,
                                CELL, 10, 25, 75)

    assert result == pytest.approx((0.075, 0.0625, 0.0875), rel=1e-12)


@pytest.mark.parametrize(("z", "cols"), [([100.0], [0]), ([100.0, 100.0, 100.0], [0, 1, 2]), ([100.0, 99.0], [0, 20])],
                         ids=["one cell", "level", "too far apart"])
def test_a_reach_without_slopes_has_none(z, cols) -> None:
    rows, cols = np.zeros(len(z), dtype=np.int64), np.array(cols)
    stations = along_stream_stations(rows, cols, CELL, CELL)

    assert all(math.isnan(value) for value in reach_median_slope(np.array(z), rows, cols, stations, CELL, CELL, 10))


@pytest.mark.parametrize("cells", [2, 30])
def test_a_uniform_slope_is_a_slope(cells: int) -> None:
    """Every slope the same, as on a uniform reach or one of two cells, makes both percentiles that slope."""
    rows, cols = np.zeros(cells, dtype=np.int64), np.arange(cells)
    z = 100.0 - 0.001 * CELL * cols
    stations = along_stream_stations(rows, cols, CELL, CELL)

    assert reach_median_slope(z, rows, cols, stations, CELL, CELL, 10) == pytest.approx((0.001, 0.001, 0.001),
                                                                                          rel=1e-9)
    local = local_average_slopes(z, rows, cols, stations, CELL, CELL, 10)
    np.testing.assert_allclose(corrected_local_slopes(local, 0.001, 0.001), 0.001, rtol=1e-9)


def test_corrected_slopes_are_kept_between_the_reach_percentiles() -> None:
    np.testing.assert_allclose(corrected_local_slopes([0.001, 0.003, 0.009, np.nan], 0.002, 0.008),
                               [0.002, 0.003, 0.008, np.nan])


@pytest.mark.parametrize(("stations", "length"), [([0.0, 3.0], 3.0), ([5.0, 5.0], CELL)],
                         ids=["along the stream", "side by side"])
def test_a_slope_between_two_cells_is_over_the_distance_between_their_stations(stations, length) -> None:
    """Two cells a column apart: their slope is over the difference in their stations, even where that's less than
    the 10 m between their centres, as between the stair-step cells of a stream at an angle to the grid; only cells at
    the same station, side by side across the stream, are measured in a straight line."""
    rows, cols = np.array([0, 0]), np.array([0, 1])
    z = np.array([10.0, 9.99])

    slopes = local_average_slopes(z, rows, cols, np.array(stations), CELL, CELL, 1)

    np.testing.assert_allclose(slopes, 0.01 / length, rtol=1e-12)


def reach_with_a_long_row(rng):
    """A reach running 40 cells along a row and then winding away, with random elevations."""
    cells = [(20, c) for c in range(10, 50)]
    x, y, heading = 49.0, 20.0, 0.0
    while len(cells) < 220:
        heading = 0.9 * math.sin(len(cells) / 9.0) + 0.6
        x += math.cos(heading) / 3
        y += math.sin(heading) / 3
        cell = (int(round(y)), int(round(x)))
        if cell != cells[-1] and cell not in cells:
            cells.append(cell)
    order = rng.permutation(len(cells))
    rows = np.array([cells[k][0] for k in order], dtype=np.int64)
    cols = np.array([cells[k][1] for k in order], dtype=np.int64)
    z = 100.0 + rng.normal(0.0, 0.5, rows.size)
    return rows, cols, z, along_stream_stations(rows, cols, CELL, CELL)


def brute_force_length(rows, cols, stations, a, b):
    along = abs(stations[a] - stations[b])
    return along if along > 0.0 else CELL * math.hypot(rows[a] - rows[b], cols[a] - cols[b])


def test_each_pair_of_cells_within_the_distance_gives_one_slope() -> None:
    rows, cols, z, stations = reach_with_a_long_row(np.random.default_rng(8))
    distance = 6
    expected = []
    for a in range(rows.size):
        for b in range(a + 1, rows.size):
            if abs(rows[a] - rows[b]) <= distance and abs(cols[a] - cols[b]) <= distance:
                slope = np.round(abs(z[a] - z[b]) / brute_force_length(rows, cols, stations, a, b), 8)
                if slope > 0.0:
                    expected.append(slope)

    got = slope_module._pair_slopes(z, rows, cols, stations, CELL, CELL, distance)

    np.testing.assert_allclose(np.sort(got), np.sort(expected), rtol=1e-15)


def test_each_local_average_is_over_the_cells_within_the_distance() -> None:
    rows, cols, z, stations = reach_with_a_long_row(np.random.default_rng(9))
    distance = 6
    expected = []
    for a in range(rows.size):
        slopes = [abs(z[a] - z[b]) / brute_force_length(rows, cols, stations, a, b) for b in range(rows.size)
                  if b != a and abs(rows[a] - rows[b]) <= distance and abs(cols[a] - cols[b]) <= distance]
        expected.append(min(max(np.mean(slopes), MIN_SLOPE), 0.5))

    np.testing.assert_allclose(local_average_slopes(z, rows, cols, stations, CELL, CELL, distance), expected,
                               rtol=1e-12)


def test_slopes_at_a_percentile_count_as_between_them() -> None:
    """On a DEM rounded to centimetres slopes tie. Here two tie at the 75th percentile, and counting them makes the
    median 0.0015, where leaving them out would make it 0.00133."""
    dem = np.zeros((3, 6))
    dem[1, 1:5] = [99.98, 99.97, 99.95, 99.94]
    rows, cols = np.ones(4, dtype=np.int64), np.arange(1, 5)
    stations = along_stream_stations(rows, cols, CELL, CELL)

    result = reach_median_slope(dem[rows, cols], rows, cols, stations, CELL, CELL, 10, 25, 75)

    assert result == pytest.approx((0.0015, 0.00108333, 0.0015), rel=1e-12)


# --- Fitted lines (local_fit) -------------------------------------------------------------------------------------


def straight_reach(n: int) -> np.ndarray:
    rows, cols = np.zeros(n, dtype=np.int64), np.arange(n)
    return along_stream_stations(rows, cols, CELL, CELL)


@pytest.mark.parametrize("reverse", [False, True], ids=["falling with the stations", "rising with the stations"])
def test_a_straight_bed_has_its_slope_at_every_cell_whichever_way_the_stations_run(reverse: bool) -> None:
    stations = straight_reach(40)
    z = 100.0 - 0.002 * stations
    if reverse:
        z = z[::-1]

    slopes, reach_slope = local_fit_slopes(z, stations, 5)

    np.testing.assert_allclose(slopes, 0.002, rtol=1e-9)
    assert reach_slope == pytest.approx(0.002, rel=1e-9)


def test_a_rise_offsets_a_fall_where_the_local_average_counts_both() -> None:
    stations = straight_reach(60)
    z = 100.0 - 0.001 * stations + 0.3 * (np.arange(60) % 2)  # a 0.3 m sawtooth on a fall of 0.001
    rows, cols = np.zeros(60, dtype=np.int64), np.arange(60)

    slopes, reach_slope = local_fit_slopes(z, stations, 5)

    assert np.median(slopes) == pytest.approx(0.001, rel=0.05)
    assert reach_slope == pytest.approx(0.001, rel=0.05)
    assert np.median(local_average_slopes(z, rows, cols, stations, CELL, CELL, 5)) > 3 * 0.001


def test_each_line_is_through_the_cells_nearest_along_the_stream_as_many_at_the_ends() -> None:
    stations = straight_reach(30)
    z = 100.0 - 0.001 * stations - 2e-5 * stations ** 1.5  # steepening downstream
    distance = 4

    slopes, _ = local_fit_slopes(z, stations, distance)

    for i in range(30):
        lo = min(max(i - distance, 0), 30 - (2 * distance + 1))
        expected = -np.polyfit(stations[lo:lo + 2 * distance + 1], z[lo:lo + 2 * distance + 1], 1)[0]
        assert slopes[i] == pytest.approx(expected, rel=1e-9)


def test_fitted_slopes_are_kept_between_0p00001_and_0p5() -> None:
    stations = straight_reach(41)
    z = 100.0 - 0.001 * stations
    z[36:] = z[35] + 0.002 * (stations[36:] - stations[35])  # rising against the reach's fall at its end

    slopes, reach_slope = local_fit_slopes(z, stations, 2)
    cliff_slopes, cliff_reach_slope = local_fit_slopes(100.0 - 0.9 * stations, stations, 2)

    assert slopes[-1] == MIN_SLOPE and 0.0005 < reach_slope < 0.001
    assert (cliff_slopes == 0.5).all() and cliff_reach_slope == 0.5


def test_cells_without_an_elevation_are_left_out_of_the_lines() -> None:
    stations = straight_reach(20)
    z = 100.0 - 0.002 * stations + 0.01 * np.sin(np.arange(20))
    gappy = z.copy()
    gappy[7] = np.nan

    slopes, reach_slope = local_fit_slopes(gappy, stations, 3)
    kept, kept_reach = local_fit_slopes(np.delete(z, 7), np.delete(stations, 7), 3)

    assert np.isnan(slopes[7])
    np.testing.assert_allclose(np.delete(slopes, 7), kept, rtol=1e-12)
    assert reach_slope == kept_reach


@pytest.mark.parametrize(("z", "stations"), [([100.0], [0.0]), ([100.0, 99.0], [5.0, 5.0]),
                                             ([100.0, np.nan], [0.0, 10.0])],
                         ids=["one cell", "one station", "one elevation"])
def test_no_line_can_be_fitted_through_one_point(z, stations) -> None:
    slopes, reach_slope = local_fit_slopes(np.array(z), np.array(stations), 5)

    assert np.isnan(slopes).all() and math.isnan(reach_slope)


# --- Filling unresolved reaches from their neighbours -------------------------------------------------------------


def neighbours() -> nx.DiGraph:
    """Reach 3 has reaches 1 (with reach 5 above it) and 2 flowing into it, and flows into reach 4."""
    network = nx.DiGraph()
    network.add_edges_from([(5, 1), (1, 3), (2, 3), (3, 4)])
    return network


def test_a_reach_without_a_slope_takes_the_mean_of_its_main_stem_neighbours() -> None:
    slopes = {5: 0.004, 1: 0.003, 2: 0.009, 3: math.nan, 4: 0.001}

    assert fill_unresolved_reach_slopes(neighbours(), slopes) == set()
    assert slopes[3] == pytest.approx(0.002)


def test_reaches_fill_from_those_filled_before() -> None:
    """Reach 4 has only reach 3 next to it, so it fills once reach 3 has."""
    slopes = {5: 0.004, 1: 0.003, 2: 0.009, 3: math.nan, 4: math.nan}

    assert fill_unresolved_reach_slopes(neighbours(), slopes) == set()
    assert slopes[3] == pytest.approx(0.003)
    assert slopes[4] == pytest.approx(0.003)


def test_the_main_stem_is_chosen_among_neighbours_with_a_slope() -> None:
    """Reach 1 has the most reaches above it but isn't on the raster, so reach 2 is the upstream neighbour."""
    slopes = {2: 0.009, 3: math.nan, 4: 0.001}

    fill_unresolved_reach_slopes(neighbours(), slopes)

    assert slopes[3] == pytest.approx(0.005)


def test_reaches_with_a_slope_keep_it_whatever_their_neighbours() -> None:
    """A uniform reach, with both percentiles the same, keeps its own slope."""
    slopes = {5: 0.004, 1: 0.003, 2: 0.009, 3: 0.0007, 4: 0.001}

    assert fill_unresolved_reach_slopes(neighbours(), slopes) == set()
    assert slopes == {5: 0.004, 1: 0.003, 2: 0.009, 3: 0.0007, 4: 0.001}


def test_reaches_without_neighbours_stay_without_a_slope() -> None:
    network = nx.DiGraph()
    network.add_node(8)
    slopes = {8: math.nan, 9: math.nan, 10: 0.001}

    assert fill_unresolved_reach_slopes(network, slopes) == {8, 9}
    assert math.isnan(slopes[8]) and math.isnan(slopes[9])


# --- End points ---------------------------------------------------------------------------------------------------


GEOTRANSFORM = (0.0, 10.0, 0.0, 100.0, 0.0, -10.0)  # 10 m cells, the top left corner at (0, 100)


def end_point_dem() -> np.ndarray:
    """A 10 x 10 DEM falling 1 m per cell to the south east."""
    rows, cols = np.mgrid[0:10, 0:10]
    return 120.0 - rows - cols


def test_the_end_point_slope_is_the_drop_between_the_ends_over_the_length() -> None:
    line = LineString([(5.0, 95.0), (95.0, 5.0)])  # from cell (0, 0) to cell (9, 9)

    assert end_point_slope(line, 150.0, end_point_dem(), GEOTRANSFORM) == round(18.0 / 150.0, 8)


def test_an_end_off_the_raster_moves_along_the_line_to_it() -> None:
    """A 140 m line along row 4 that ends 45 m east of the raster. In steps of 2% (2.8 m) from that end, the first
    point on the raster is 34% along, in column 9."""
    line = LineString([(5.0, 55.0), (145.0, 55.0)])
    dem = end_point_dem()

    slope = end_point_slope(line, 140.0, dem, GEOTRANSFORM)

    assert slope == pytest.approx((dem[4, 0] - dem[4, 9]) / (0.66 * 140.0))


def test_a_line_with_both_ends_off_the_raster_moves_both() -> None:
    """A 1 km line along row 4 from 40 m west of the raster. In steps of 2% (20 m) the first point on the raster is
    at its west edge, 4% along, and the last is 80 m east of it, 12% from the far end."""
    line = LineString([(-40.0, 55.0), (960.0, 55.0)])
    dem = end_point_dem()

    assert end_point_slope(line, 1000.0, dem, GEOTRANSFORM) == pytest.approx((dem[4, 0] - dem[4, 8]) / 80.0)


def test_an_end_without_data_moves_along_the_line_too() -> None:
    """An end where the DEM has no data moves along the line like an end off the raster."""
    line = LineString([(5.0, 55.0), (95.0, 55.0)])  # along row 4, from column 0 to column 9
    dem = end_point_dem()
    dem[4, 0] = -9999.0

    slope = end_point_slope(line, 90.0, dem, GEOTRANSFORM)

    assert slope == pytest.approx(8.0 / (0.94 * 90.0))  # the first point with data is 6% along, in column 1


def test_a_point_just_beyond_the_left_edge_is_off_the_raster() -> None:
    """3 m west of the raster is off it, so the line's first point on it is 4% along."""
    dem = end_point_dem()
    line = LineString([(-3.0, 55.0), (97.0, 55.0)])

    assert end_point_slope(line, 100.0, dem, GEOTRANSFORM) == pytest.approx(9.0 / 96.0)


def test_a_line_with_no_data_at_either_end_has_no_slope() -> None:
    line = LineString([(200.0, 200.0), (300.0, 300.0)])

    assert math.isnan(end_point_slope(line, 141.0, end_point_dem(), GEOTRANSFORM))


def test_a_line_in_parts_uses_the_longest_part_and_its_share_of_the_length() -> None:
    """The long part runs down column 5 from row 0 to row 8 (80 m of the 100 m of line)."""
    line = MultiLineString([[(55.0, 95.0), (55.0, 15.0)], [(0.5, 0.5), (0.5, 20.5)]])

    assert end_point_slope(line, 200.0, end_point_dem(), GEOTRANSFORM) == pytest.approx(8.0 / 160.0)


def test_the_end_point_slope_is_at_least_0p00001() -> None:
    line = LineString([(5.0, 95.0), (95.0, 95.0)])
    dem = np.full((10, 10), 50.0)

    assert end_point_slope(line, 90.0, dem, GEOTRANSFORM) == MIN_SLOPE
