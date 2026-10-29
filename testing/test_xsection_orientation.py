from __future__ import annotations

import math

import numpy as np
import pytest

from arc.hydraulics import top_widths
from arc.xsection.orientation import (
    TEST_DEPTH,
    angle_offsets,
    narrowest_direction,
    stream_direction,
)
from arc.xsection.sampling import sample_cross_section, sample_elevations

CELL = 10.0
REACH = 7
HALF_A_METRE = 0.5  # a test depth that finds a channel's own width rather than its valley's


def stream_raster(cells, shape=(61, 61)):
    streams = np.zeros(shape, dtype=np.int64)
    for r, c in cells:
        streams[r, c] = REACH
    return streams


def rasterised_line(angle, offset=0.0, size=101, half_length=40):
    """A straight stream through the middle cell, rasterised by rounding points every 0.02 cells along it."""
    centre = size // 2
    cells = {(centre, centre)}
    for t in np.linspace(-half_length, half_length, 4001):
        cells.add((int(math.floor(centre + t * math.sin(angle) + offset * math.cos(angle) + 0.5)),
                   int(math.floor(centre + t * math.cos(angle) - offset * math.sin(angle) + 0.5))))
    return stream_raster(cells, (size, size)), centre


def axis_difference(a: float, b: float) -> float:
    """How far apart two lines' directions are, in radians, whichever way along them each points."""
    d = (a - b) % math.pi
    return min(d, math.pi - d)


# --- The stream's direction ---------------------------------------------------------------------------------------


@pytest.mark.parametrize(("step", "expected"), [((0, 1), 0.0), ((1, 0), math.pi / 2), ((1, 1), math.pi / 4),
                                               ((-1, 1), 3 * math.pi / 4)])
def test_a_straight_stream_runs_along_its_cells(step, expected) -> None:
    streams = stream_raster([(30 + step[0] * k, 30 + step[1] * k) for k in range(-10, 11)])

    assert stream_direction(streams, 30, 30, 5, CELL, CELL) == pytest.approx(expected, abs=1e-12)


def test_the_direction_is_in_metres_where_cells_are_not_square() -> None:
    """With 10 m wide, 20 m tall cells, a stream one row down for two columns across runs at 45 degrees on the
    ground, not the 26.6 degrees it makes in cells."""
    streams = stream_raster([(30 + k, 30 + 2 * k) for k in range(-5, 6)])

    assert stream_direction(streams, 30, 30, 10, 10.0, 20.0) == pytest.approx(math.pi / 4, abs=1e-12)


def test_rows_and_columns_are_treated_alike() -> None:
    """Transposing the raster mirrors a stream's direction about 45 degrees, as it should. (A fit of rows against
    columns doesn't: on straight streams at random angles it moves by up to 5 degrees.)"""
    rng = np.random.default_rng(0)
    error = 0.0
    for _ in range(400):
        streams, centre = rasterised_line(rng.uniform(0.0, math.pi), rng.uniform(-0.5, 0.5))
        transposed = streams.T.copy()

        error = max(error, axis_difference(stream_direction(transposed, centre, centre, 10, CELL, CELL),
                                           math.pi / 2 - stream_direction(streams, centre, centre, 10, CELL, CELL)))

    assert error < 1e-12


def test_straight_streams_are_measured_as_well_near_the_columns_as_near_the_rows() -> None:
    rng = np.random.default_rng(1)
    errors = {"rows": [], "columns": []}
    for _ in range(400):
        angle = rng.uniform(0.0, math.pi)
        streams, centre = rasterised_line(angle, rng.uniform(-0.5, 0.5))
        near = "rows" if min(angle, math.pi - angle) < math.pi / 4 else "columns"
        errors[near].append(axis_difference(stream_direction(streams, centre, centre, 10, CELL, CELL), angle))

    rows, columns = np.degrees(np.mean(errors["rows"])), np.degrees(np.mean(errors["columns"]))
    assert rows < 0.5 and columns < 0.5
    assert abs(rows - columns) < 0.1


def test_a_stream_with_one_step_sideways() -> None:
    """Ten cells down a column, stepping one column over half way, and the same cells along a row: the two
    directions mirror each other."""
    along_a_column = stream_raster([(r, 30 if r <= 30 else 31) for r in range(25, 35)])
    along_a_row = along_a_column.T.copy()

    down = stream_direction(along_a_column, 30, 30, 5, CELL, CELL)
    across = stream_direction(along_a_row, 30, 30, 5, CELL, CELL)

    assert down == pytest.approx(math.pi / 2 - across, abs=1e-12)
    assert math.degrees(across) == pytest.approx(8.34, abs=0.01)


def test_the_square_reaches_as_far_each_way() -> None:
    """A cell of the reach 5 rows and 5 columns on counts, as does one 5 back."""
    ahead = stream_raster([(30, 30), (35, 35)])
    behind = stream_raster([(30, 30), (25, 25)])

    assert stream_direction(ahead, 30, 30, 5, CELL, CELL) == pytest.approx(math.pi / 4)
    assert stream_direction(behind, 30, 30, 5, CELL, CELL) == pytest.approx(math.pi / 4)


def test_a_lone_cell_s_cross_section_runs_along_its_row() -> None:
    """With no other cell of its reach nearby, its stream direction is pi / 2, so its cross section runs along its
    row."""
    streams = stream_raster([(30, 30), (30, 40)])

    assert stream_direction(streams, 30, 30, 5, CELL, CELL) == pytest.approx(math.pi / 2)


def test_only_a_stream_cell_has_a_direction() -> None:
    streams = stream_raster([(30, c) for c in range(20, 41)])

    with pytest.raises(ValueError, match="isn't a stream cell"):
        stream_direction(streams, 31, 30, 5, CELL, CELL)


def test_the_square_stops_at_the_raster_s_edge() -> None:
    """Nothing wraps round to the far side of the raster."""
    streams = stream_raster([(0, c) for c in range(5, 16)] + [(19, 10), (0, 19)], shape=(20, 20))

    assert stream_direction(streams, 0, 10, 5, CELL, CELL) == pytest.approx(0.0, abs=1e-12)


# --- The angle search ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize(("manipulation", "interval", "degrees"), [
    (6.0, 1.0, [0, -1, 1, -2, 2, -3, 3]), (1.1, 1.0, [0]), (0.0, 1.0, [0]), (10.0, 0.0, [0]), (7.0, 2.0, [0, -2, 2]),
    (180.0, 90.0, [0, -90, 90])])
def test_the_offsets_go_out_an_interval_at_a_time_to_half_of_degree_manip(manipulation, interval, degrees) -> None:
    np.testing.assert_allclose(angle_offsets(manipulation, interval), np.radians(degrees), rtol=1e-15)


def test_the_search_samples_as_sample_cross_section_does() -> None:
    """The elevations the search compares are sample_cross_section's, with NaNs, in float32 and off the raster."""
    rng = np.random.default_rng(3)
    for trial in range(400):
        shape = tuple(rng.integers(3, 40, 2))
        dem = rng.normal(100.0, 5.0, shape)
        if trial % 3 == 0:
            dem[rng.random(shape) < 0.05] = np.nan
        if trial % 2 == 0:
            dem = dem.astype(np.float32)
        dx, dy = float(rng.choice([10.0, 7.0, 23.0])), float(rng.choice([10.0, 13.0, 31.0]))
        row, col = int(rng.integers(0, shape[0])), int(rng.integers(0, shape[1]))
        direction = float(rng.uniform(-7.0, 7.0)) if trial % 5 else int(rng.integers(0, 8)) * math.pi / 4
        length = float(rng.uniform(20.0, 600.0))

        expected = sample_cross_section(dem, np.ones_like(dem), row, col, direction, length, dx, dy)
        elevations, spacing = sample_elevations(dem, row, col, direction, length, dx, dy)

        assert spacing == expected.ordinate_distance
        assert elevations.dtype == expected.elevations.dtype
        np.testing.assert_array_equal(elevations, expected.elevations)


def v_valley(angle: float, size: int = 201) -> np.ndarray:
    """A straight channel at an angle through cell (100, 100), its sides rising 0.2 m per metre from it."""
    rows, cols = np.mgrid[0:size, 0:size]
    across = np.abs(-(cols - 100) * CELL * math.sin(angle) + (rows - 100) * CELL * math.cos(angle))
    return 100.0 + 0.2 * across


@pytest.mark.parametrize(("start", "expected"), [(10, 30), (20, 30), (30, 30), (40, 30), (50, 30), (0, 20)])
def test_the_search_turns_the_cross_section_across_the_channel(start, expected) -> None:
    """A channel at 30 degrees, searched within 20 degrees either way of the start, 5 degrees at a time."""
    dem = v_valley(math.radians(30.0))

    direction = narrowest_direction(dem, 100, 100, math.radians(start), 1000.0, CELL, CELL, angle_offsets(40, 5))

    assert math.degrees(direction) == pytest.approx(expected, abs=1e-9)


def test_a_quarter_turn_finds_a_channel_the_stream_direction_runs_across() -> None:
    """A channel down column 100, from a stream direction wrongly flowing east. Turned a quarter turn, the cross
    section runs along a row and crosses the channel, 5 m wide half a metre up."""
    dem = v_valley(math.pi / 2)

    direction = narrowest_direction(dem, 100, 100, 0.0, 1000.0, CELL, CELL, angle_offsets(180, 90),
                                    test_depth=HALF_A_METRE)
    elevations, spacing = sample_elevations(dem, 100, 100, direction, 1000.0, CELL, CELL)
    left, right = top_widths(elevations, spacing, elevations[elevations.size // 2] + HALF_A_METRE)

    assert axis_difference(direction, math.pi / 2) == pytest.approx(0.0, abs=1e-12)
    assert left + right == pytest.approx(2.0 * 0.5 / 0.2, rel=1e-12)


def test_equally_narrow_directions_keep_the_first() -> None:
    """In a round hollow a cross section along a row is as narrow as one along a column."""
    rows, cols = np.mgrid[0:201, 0:201]
    dem = 100.0 + 0.2 * CELL * np.hypot(rows - 100, cols - 100)

    assert narrowest_direction(dem, 100, 100, 0.0, 1000.0, CELL, CELL, [0.0, math.pi / 2]) == 0.0
    assert narrowest_direction(dem, 100, 100, 0.0, 1000.0, CELL, CELL, [math.pi / 2, 0.0]) == math.pi / 2


def uncapped_choice(dem, row, col, start, length, offsets) -> int:
    """Which offset comparing whole widths would pick: the smallest top width 0.5 m up, however far each cross
    section reaches."""
    widths = []
    for offset in offsets:
        elevations, spacing = sample_elevations(dem, row, col, start + offset, length, CELL, CELL)
        left, right = top_widths(elevations, spacing, elevations[elevations.size // 2] + HALF_A_METRE)
        widths.append(left + right)
    return int(np.argmin(widths))


def test_water_reaching_every_cross_section_s_ends_doesnt_choose_the_direction() -> None:
    """On a flat floodplain the water 0.5 m up reaches both ends of every cross section, whose lengths differ: 50 m
    a side along rows and columns, 42 m on the diagonals. Comparing whole widths would pick a diagonal, the
    shortest."""
    dem = np.full((201, 201), 100.0)
    dem[100, 100] = 99.8
    offsets = angle_offsets(90, 45)

    assert narrowest_direction(dem, 100, 100, 0.0, 100.0, CELL, CELL, offsets, test_depth=HALF_A_METRE) == 0.0
    assert uncapped_choice(dem, 100, 100, 0.0, 100.0, offsets) == 1


def channel_by_a_floodplain(shape, row, col, angle=math.radians(30.0)) -> np.ndarray:
    """A channel at an angle through (row, col). On its north-east side the ground rises 1 cm per metre, so the water
    0.5 m up reaches 50 m out, and on its south-west side a floodplain rises 1 mm per metre, so the water reaches 500
    m out. Each side is a plane, which linear interpolation between cells reproduces exactly, so a cross section
    turned from square to the channel is wider by exactly 1 / cos of the turn."""
    rows, cols = np.mgrid[0:shape[0], 0:shape[1]]
    across = -(cols - col) * CELL * math.sin(angle) + (rows - row) * CELL * math.cos(angle)  # > 0 to the south-west
    return np.where(across > 0.0, 100.0 + 0.001 * across, 100.0 - 0.01 * across)


def test_a_side_reaching_the_raster_s_edge_leaves_the_other_side_to_choose() -> None:
    """The channel three cells from the raster's west edge, so the floodplain floods to the edge, which each
    direction's cross section reaches at a different distance. The north-east side still shows which way the
    channel runs. Comparing whole widths would pick the direction that reaches the edge soonest."""
    dem = channel_by_a_floodplain((101, 60), 50, 3)
    offsets = angle_offsets(40, 5)
    start = math.radians(20.0)

    assert math.degrees(narrowest_direction(dem, 50, 3, start, 600.0, CELL, CELL, offsets,
                                            test_depth=HALF_A_METRE)) == pytest.approx(30.0)
    assert math.degrees(start + offsets[uncapped_choice(dem, 50, 3, start, 600.0, offsets)]) == pytest.approx(40.0)


def test_water_reaching_missing_ground_counts_as_reaching_the_end() -> None:
    """As at the raster's edge, but with no data (NaN) 30 m out on the floodplain."""
    dem = channel_by_a_floodplain((101, 101), 50, 50)
    rows, cols = np.mgrid[0:101, 0:101]
    dem[-(cols - 50) * CELL * 0.5 + (rows - 50) * CELL * math.sqrt(0.75) > 30.0] = np.nan

    direction = narrowest_direction(dem, 50, 50, math.radians(20.0), 600.0, CELL, CELL, angle_offsets(40, 5),
                                    test_depth=HALF_A_METRE)

    assert math.degrees(direction) == pytest.approx(30.0)


def test_offsets_half_a_turn_apart_are_the_same_cross_section() -> None:
    dem = v_valley(math.radians(30.0))
    start = math.radians(30.0)

    assert narrowest_direction(dem, 100, 100, start, 1000.0, CELL, CELL, [0.0, math.pi]) == start
    assert narrowest_direction(dem, 100, 100, start, 1000.0, CELL, CELL, [0.2, 0.2 + math.pi, 0.2 - math.pi]) \
        == pytest.approx(start + 0.2)


def test_the_width_is_compared_at_the_test_depth() -> None:
    """A channel at 30 degrees whose sides rise 1 cm per metre for 100 m and then level off. Half a metre up, the
    water reaches 50 m out square to the channel, so the channel's direction is narrowest. 1.5 m up it floods past
    the sides to the ends of every cross section, and the start is kept."""
    angle = math.radians(30.0)
    rows, cols = np.mgrid[0:201, 0:201]
    across = np.abs(-(cols - 100) * CELL * math.sin(angle) + (rows - 100) * CELL * math.cos(angle))
    dem = 100.0 + 0.01 * np.minimum(across, 100.0)
    start, offsets = math.radians(20.0), angle_offsets(20, 10)

    assert math.degrees(narrowest_direction(dem, 100, 100, start, 600.0, CELL, CELL, offsets,
                                            test_depth=0.5)) == pytest.approx(30.0)
    assert narrowest_direction(dem, 100, 100, start, 600.0, CELL, CELL, offsets, test_depth=1.5) == start


def test_the_test_depth_is_5_m() -> None:
    assert TEST_DEPTH == 5.0


def test_the_default_depth_finds_the_valley_where_half_a_metre_finds_the_channel() -> None:
    """A V channel 1 m deep and 50 m wide at 30 degrees, down a valley whose floor is 400 m wide at 60 degrees, with
    walls rising 10 cm a metre, all planes that linear interpolation between cells reproduces. Half a metre up the
    water fills only the channel; 5 m up it floods the valley floor out to 240 m either side of its axis."""
    rows, cols = np.mgrid[0:201, 0:201]

    def across(angle):
        return np.abs(-(cols - 100) * CELL * math.sin(angle) + (rows - 100) * CELL * math.cos(angle))
    valley = 0.1 * np.maximum(across(math.radians(60.0)) - 200.0, 0.0)
    channel = np.maximum(1.0 - across(math.radians(30.0)) / 25.0, 0.0)
    dem = 100.0 + valley - channel
    start, offsets = math.radians(45.0), angle_offsets(40, 5)

    assert math.degrees(narrowest_direction(dem, 100, 100, start, 600.0, CELL, CELL, offsets)) == pytest.approx(60.0)
    assert math.degrees(narrowest_direction(dem, 100, 100, start, 600.0, CELL, CELL, offsets,
                                            test_depth=HALF_A_METRE)) == pytest.approx(30.0)


def test_an_offset_beyond_a_quarter_turn_is_taken_half_a_turn_round() -> None:
    dem = v_valley(math.radians(30.0))
    start = math.radians(30.0)

    assert narrowest_direction(dem, 100, 100, start, 1000.0, CELL, CELL, [math.pi + 0.1]) == pytest.approx(start + 0.1)
    assert narrowest_direction(dem, 100, 100, start, 1000.0, CELL, CELL, [-0.1 - math.pi]) == pytest.approx(start - 0.1)


def test_no_offsets_to_try_keeps_the_direction() -> None:
    dem = v_valley(math.radians(30.0))

    assert narrowest_direction(dem, 100, 100, 0.3, 1000.0, CELL, CELL, angle_offsets(1.1, 1.0)) == 0.3
    assert narrowest_direction(dem, 100, 100, 0.3, 1000.0, CELL, CELL, []) == 0.3


def test_a_reach_s_cells_give_the_stream_raster_s_direction() -> None:
    """reach_direction, from the reach's cells in row order, gives the direction stream_direction gives from the
    raster, bit for bit: with reaches crossing, cells at the raster's edges, windows of any size and cells alone."""
    from arc.xsection.orientation import reach_direction
    from arc.xsection.stream_path import stream_cells_by_reach
    rng = np.random.default_rng(66)
    for _ in range(200):
        shape = (int(rng.integers(1, 40)), int(rng.integers(1, 40)))
        streams = np.where(rng.random(shape) < rng.uniform(0.05, 0.6), rng.integers(1, 4, shape), 0).astype(np.int32)
        reaches = stream_cells_by_reach(streams)
        dx, dy = float(rng.uniform(5.0, 40.0)), float(rng.uniform(5.0, 40.0))
        for reach, (rows, cols) in reaches.items():
            for i in rng.integers(0, rows.size, min(rows.size, 5)):
                distance = int(rng.integers(0, 12))
                row, col = int(rows[i]), int(cols[i])
                expected = stream_direction(streams, row, col, distance, dx, dy)
                got = reach_direction(rows, cols, row, col, distance, dx, dy)
                assert np.float64(got).view(np.uint64) == np.float64(expected).view(np.uint64)
