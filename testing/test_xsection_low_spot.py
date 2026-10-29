from __future__ import annotations

import math

import numpy as np
import pytest

from arc.xsection.low_spot import low_spot_cell
from arc.xsection.sampling import sample_cross_section

# 10 m cells, and a 200 m cross section: 10 ordinates either side. A stream flowing south (pi / 2) has its cross
# section along the row, and one flowing east (0) along the column.
CELL = 10.0
LENGTH = 200.0
SOUTH, NORTH, EAST, WEST = math.pi / 2, 3 * math.pi / 2, 0.0, math.pi


def valley(channel_col, size=41) -> np.ndarray:
    """A V-shaped valley down a column, its sides rising 0.2 m per metre."""
    return np.tile(100.0 + 0.2 * CELL * np.abs(np.arange(size) - channel_col), (size, 1))


def test_the_cross_section_moves_to_the_lowest_ground_nearby() -> None:
    dem = valley(23)

    assert low_spot_cell(dem, 20, 20, SOUTH, LENGTH, CELL, CELL, 5) == (20, 23)
    assert low_spot_cell(dem, 20, 20, NORTH, LENGTH, CELL, CELL, 5) == (20, 23)


def test_a_range_of_n_looks_n_ordinates_out() -> None:
    """With the channel 5 cells away, a range of 5 reaches it and 4 gets as near as it can."""
    dem = valley(25)

    assert low_spot_cell(dem, 20, 20, SOUTH, LENGTH, CELL, CELL, 5) == (20, 25)
    assert low_spot_cell(dem, 20, 20, SOUTH, LENGTH, CELL, CELL, 4) == (20, 24)
    assert low_spot_cell(dem, 20, 20, SOUTH, LENGTH, CELL, CELL, 1) == (20, 21)


def test_the_stream_cell_stays_where_nothing_nearby_is_lower() -> None:
    dem = valley(20)
    dem[20, 23] = 100.0  # as low, but not lower

    assert low_spot_cell(dem, 20, 20, SOUTH, LENGTH, CELL, CELL, 5) == (20, 20)
    assert low_spot_cell(valley(23), 20, 20, SOUTH, LENGTH, CELL, CELL, 0) == (20, 20)


@pytest.mark.parametrize(("direction", "lower", "expected"), [
    pytest.param(SOUTH, [(20, 17), (20, 23)], (20, 23), id="along-a-row-flowing-south"),
    pytest.param(NORTH, [(20, 17), (20, 23)], (20, 23), id="along-a-row-flowing-north"),
    pytest.param(EAST, [(17, 20), (23, 20)], (23, 20), id="along-a-column-flowing-east"),
    pytest.param(WEST, [(17, 20), (23, 20)], (23, 20), id="along-a-column-flowing-west"),
])
def test_of_two_equally_low_ordinates_as_near_the_higher_rows_or_columns_win(direction, lower, expected) -> None:
    """The one towards the higher rows, or for a cross section along a row the higher columns, whichever way the
    stream flows."""
    dem = np.full((41, 41), 100.0)
    for cell in lower:
        dem[cell] = 99.0

    assert low_spot_cell(dem, 20, 20, direction, LENGTH, CELL, CELL, 5) == expected


def test_of_equally_low_ordinates_the_nearest_wins() -> None:
    dem = np.full((41, 41), 100.0)
    dem[20, 24] = dem[20, 18] = dem[20, 15] = 99.0

    assert low_spot_cell(dem, 20, 20, SOUTH, LENGTH, CELL, CELL, 5) == (20, 18)


def lowest_nearby(dem, row, col, along_row: bool, low_spot_range: int) -> tuple[int, int]:
    """The low spot the slow way: of the cells within low_spot_range of the stream cell along its row (or column)
    that are lower than it, the lowest, then the nearest, then the one in the higher column (or row)."""
    candidates = []
    for offset in range(-low_spot_range, low_spot_range + 1):
        cell = (row, col + offset) if along_row else (row + offset, col)
        if offset != 0 and dem[cell] < dem[row, col]:
            candidates.append((dem[cell], abs(offset), offset < 0, cell))
    return min(candidates)[3] if candidates else (row, col)


def test_the_low_spot_along_rows_and_columns_at_random() -> None:
    """Random ground, whole metres for plenty of ties."""
    rng = np.random.default_rng(5)
    for trial in range(300):
        dem = rng.integers(95, 105, (41, 41)).astype(np.float64) if trial % 2 else rng.normal(100.0, 2.0, (41, 41))
        row, col = (int(v) for v in rng.integers(12, 29, 2))
        along_row = trial % 4 < 2
        low_spot_range = int(rng.integers(1, 11))
        expected = lowest_nearby(dem, row, col, along_row, low_spot_range)
        for direction in ((SOUTH, NORTH) if along_row else (EAST, WEST)):
            assert low_spot_cell(dem, row, col, direction, LENGTH, CELL, CELL, low_spot_range) == expected


def test_nan_and_ordinates_off_the_raster_are_never_the_low_spot() -> None:
    """A DEM in feet, 12000 ft up, two cells from its east edge: the ordinates off the raster sample as 9999."""
    dem = np.full((41, 23), 12000.0)
    dem[20, 18] = np.nan

    assert low_spot_cell(dem, 20, 20, SOUTH, LENGTH, CELL, CELL, 10) == (20, 20)
    assert sample_cross_section(dem, dem, 20, 20, SOUTH, LENGTH, CELL, CELL).elevations[-1] == 9999.0


def test_ground_at_or_below_sea_level_can_be_the_low_spot() -> None:
    dem = valley(23) - 101.0

    assert low_spot_cell(dem, 20, 20, SOUTH, LENGTH, CELL, CELL, 5) == (20, 23)


def test_an_oblique_cross_section_moves_to_the_cell_nearest_the_low_ordinate() -> None:
    """A stream flowing south-south-east: its cross section steps a column at a time and 0.4 of a row, so the lowest
    ordinate, three out, is 1.2 rows up, nearest the row above."""
    stream_direction = math.pi / 2 - math.atan(0.4)
    xs = sample_cross_section(np.zeros((41, 41)), np.zeros((41, 41)), 20, 20, stream_direction, LENGTH, CELL, CELL)
    dem = np.full((41, 41), 100.0)
    dem[19, 23] = dem[18, 23] = 90.0

    assert xs.ordinate_distance == pytest.approx(CELL * math.hypot(1.0, 0.4))
    assert low_spot_cell(dem, 20, 20, stream_direction, LENGTH, CELL, CELL, 5) == (19, 23)


def test_the_cells_of_a_cross_section_on_cells_that_aren_t_square() -> None:
    """Cells 10 m wide and 20 m tall: a stream flowing east has its cross section down the column, a row per
    ordinate, 20 m apart."""
    dem = np.tile(100.0 + np.abs(np.arange(41) - 23.0)[:, None], (1, 41))

    assert low_spot_cell(dem, 20, 20, EAST, 400.0, 10.0, 20.0, 5) == (23, 20)
    assert low_spot_cell(dem, 20, 20, WEST, 400.0, 10.0, 20.0, 5) == (23, 20)
