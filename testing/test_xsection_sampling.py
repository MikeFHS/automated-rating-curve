from __future__ import annotations

import numpy as np
import pytest

from arc.xsection.sampling import sample_cross_section

# Most tests use 10 m cells and a 100 m cross section centred on cell (50, 50) of a 101 x 101 raster.
# Ordinates step one cell along whichever axis the section crosses fastest: 10 m along a row or column,
# so 5 fit on each side of the stream cell, and one cell diagonal (14.14 m) at 45 degrees. Rows increase
# downward, so a stream_direction of 0 flows east and pi / 2 flows south.
CELL = 10.0
LENGTH = 100.0
ROW = COL = 50
STEPS = np.arange(-5, 6)  # ordinate positions from the left end to the right end

def test_samples_a_v_shaped_valley() -> None:
    """A south-flowing channel on column 50, with valley sides rising 1 m per 10 m cell."""
    cells_from_channel = np.abs(np.arange(101) - COL)
    dem = np.tile(100.0 + cells_from_channel, (101, 1))
    manning_n = np.tile(np.where(cells_from_channel == 0, 0.035, 0.1), (101, 1))

    xs = sample_cross_section(dem, manning_n, ROW, COL, np.pi / 2, LENGTH, CELL, CELL)

    np.testing.assert_allclose(xs.elevations, [105, 104, 103, 102, 101, 100, 101, 102, 103, 104, 105])
    np.testing.assert_allclose(xs.mannings_n, [0.1] * 5 + [0.035] + [0.1] * 5)
    assert xs.ordinate_distance == pytest.approx(CELL)


@pytest.mark.parametrize(
    ("stream_direction", "expected_elevations"),
    [
        pytest.param(np.pi / 2, [110, 108, 106, 104, 102, 100, 101, 102, 103, 104, 105], id="flowing-south-left-bank-is-west"),
        pytest.param(3 * np.pi / 2, [105, 104, 103, 102, 101, 100, 102, 104, 106, 108, 110], id="flowing-north-left-bank-is-east"),
    ],
)
def test_left_bank_looking_upstream_comes_first(stream_direction: float, expected_elevations: list[float]) -> None:
    """The west side of the valley rises 2 m per cell and the east side 1 m per cell."""
    cells_from_channel = np.arange(101) - COL
    dem = np.tile(100.0 + np.where(cells_from_channel < 0, -2.0 * cells_from_channel, cells_from_channel), (101, 1))

    xs = sample_cross_section(dem, np.full_like(dem, 0.035), ROW, COL, stream_direction, LENGTH, CELL, CELL)

    np.testing.assert_allclose(xs.elevations, expected_elevations)


def test_ordinates_beyond_the_raster_are_walls() -> None:
    """An 11 x 11 raster reaches 5 cells either side of its centre, so a 200 m section runs off both edges.

    Off-raster ordinates get an elevation of 9999, a wall that water cannot spread past. The cells on the
    raster's edge keep their real values.
    """
    dem = np.full((11, 11), 100.0)

    xs = sample_cross_section(dem, np.full_like(dem, 0.035), 5, 5, np.pi / 2, 200.0, CELL, CELL)

    off_raster = np.abs(np.arange(-10, 11)) > 5
    np.testing.assert_allclose(xs.elevations, np.where(off_raster, 9999.0, 100.0))
    np.testing.assert_allclose(xs.mannings_n[~off_raster], 0.035)

def test_a_nan_reaches_the_ordinates_that_interpolate_with_it() -> None:
    """Like map_coordinates, an ordinate is interpolated from its cell and the next one along each axis, even with no
    weight on the next, and on the last row or column from it and the one before. So a NaN in column 9 of 11 reaches
    the ordinates in columns 8, 9 and 10."""
    dem = np.full((11, 11), 100.0)
    dem[5, 9] = np.nan

    xs = sample_cross_section(dem, np.ones_like(dem), 5, 5, np.pi / 2, 100.0, CELL, CELL)

    np.testing.assert_array_equal(xs.elevations, [100.0] * 8 + [np.nan] * 3)

def test_a_float32_dem_sampled_like_float64_is_its_float64_copy() -> None:
    """Sampled with like a float64 array, a float32 DEM gives what its float64 copy gives, bit for bit, as arc.pipeline
    samples it: the elevations, how far out the ground rises above a height, the angle search and the low spot, near
    the raster's edges, with walls, and at any angle."""
    from arc.xsection.low_spot import low_spot_cell
    from arc.xsection.orientation import angle_offsets, narrowest_direction
    from arc.xsection.sampling import (
        OFF_RASTER_ELEVATION,
        ordinates_above,
        sample_elevations,
    )
    rng = np.random.default_rng(67)
    like = np.empty(0)
    offsets = angle_offsets(40.0, 5.0)
    for _ in range(500):
        shape = (int(rng.integers(2, 60)), int(rng.integers(2, 60)))
        dem32 = (100.0 + np.cumsum(rng.normal(0.0, 0.4, shape), axis=1)).astype(np.float32)
        dem32[rng.random(shape) < 0.05] = OFF_RASTER_ELEVATION
        dem64 = dem32.astype(np.float64)
        row, col = int(rng.integers(shape[0])), int(rng.integers(shape[1]))
        direction, length = float(rng.uniform(0.0, np.pi)), float(rng.choice([5.0, 60.0, 400.0]))
        dx, dy = float(rng.choice([1.0, 0.8])), float(rng.choice([1.0, 1.3]))
        got, spacing = sample_elevations(dem32, row, col, direction, length, dx, dy, like)
        expected, expected_spacing = sample_elevations(dem64, row, col, direction, length, dx, dy)
        assert got.dtype == np.float64 and spacing == expected_spacing
        assert np.array_equal(got.view(np.uint64), expected.view(np.uint64))
        height = float(rng.uniform(0.0, 5.0))
        assert ordinates_above(dem32, row, col, direction, length, dx, dy, height, like) == \
            ordinates_above(dem64, row, col, direction, length, dx, dy, height)
        assert narrowest_direction(dem32, row, col, direction, length, dx, dy, offsets, like=like) == \
            narrowest_direction(dem64, row, col, direction, length, dx, dy, offsets)
        assert low_spot_cell(dem32, row, col, direction, length, dx, dy, 3, like=like) == \
            low_spot_cell(dem64, row, col, direction, length, dx, dy, 3)
