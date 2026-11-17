"""Subcell_Channels (arc.xsection.subcell): a narrow channel's ground reconstructed below the cell scale."""
from __future__ import annotations

import numpy as np
import pytest

from arc.bathymetry import Banks, carve_channel
from arc.xsection.subcell import FINE, NARROW_CELLS, reconstruct, reconstruct_all
from arc.xsection.xsection import XSection

SPACING = 30.0
N = 0.035
BED, BANK, SIDE = 100.0, 105.0, 2.0  # bed and bank elevations, and the sides' run per unit of rise


def trapezoid(stations: np.ndarray, bed_width: float) -> np.ndarray:
    """The ground of a trapezoidal channel, its bed centred on station 0, flat beyond its banks."""
    distance = np.abs(stations) - bed_width / 2
    return np.clip(BED + distance / SIDE, BED, BANK)


def cell_means(bed_width: float, cells: int = 21, offset: float = 0.0) -> np.ndarray:
    """The channel's ground averaged over cells SPACING wide, the stream cell's centre offset from the bed's."""
    centres = (np.arange(cells) - cells // 2) * SPACING + offset
    fine = np.linspace(-0.5, 0.5, 601)[None, :] * SPACING + centres[:, None]
    return trapezoid(fine, bed_width).mean(axis=1)


def section(bed_width: float, offset: float = 0.0) -> XSection:
    elevations = cell_means(bed_width, offset=offset)
    xs = XSection(elevations, np.full(elevations.size, N), SPACING)
    half = bed_width / 2 + (BANK - BED) * SIDE
    xs.left_bank_distance = xs.right_bank_distance = half
    return xs


def area(stations: np.ndarray, elevations: np.ndarray, wse: float) -> float:
    """The area under a level over a polyline, integrated finely."""
    s = np.linspace(stations[0], stations[-1], 200001)
    return float(np.trapezoid(np.clip(wse - np.interp(s, stations, elevations), 0.0, None), s))


@pytest.mark.parametrize("bed_width", [30.0, 60.0, 90.0])
def test_a_narrow_channel_keeps_its_cells_means_and_stays_between_its_neighbours(bed_width: float) -> None:
    xs = section(bed_width)
    z = xs.elevations.copy()

    assert reconstruct(xs) and xs.subcell
    profile = xs.profile
    center = z.size // 2
    for k in range(z.size):
        # The FINE points of each reconstructed cell, at its centre and either side
        inside = np.abs(profile.stations - (k - center) * SPACING) < SPACING / 2 - 1e-9
        if inside.sum() == FINE:
            points = profile.elevations[inside]
            assert points.mean() == pytest.approx(z[k], abs=1e-6)
            neighbours = z[max(k - 1, 0):k + 2]
            assert neighbours.min() - 1e-6 <= points.min() and points.max() <= neighbours.max() + 1e-6
    assert profile.stations[profile.center] == 0.0
    assert np.all(np.diff(profile.stations) >= 0.0)


@pytest.mark.parametrize("offset", [0.0, 7.5, 15.0])
def test_a_narrow_channel_s_area_is_nearer_the_truth_than_straight_lines_between_the_cells(offset: float) -> None:
    """A 60 m bed in 30 m cells: joined by straight lines, the cells' means make a V with too little area."""
    xs = section(60.0, offset)
    stations = (np.arange(xs.elevations.size) - xs.elevations.size // 2) * SPACING
    wse = BED + 3.0
    fine = np.linspace(stations[0], stations[-1], 40001)
    exact = area(fine, trapezoid(fine + offset, 60.0), wse)
    linear = area(stations, xs.elevations, wse)

    assert reconstruct(xs)
    reconstructed = area(xs.profile.stations, xs.profile.elevations, wse)
    assert abs(reconstructed - exact) < 0.5 * abs(linear - exact)


def test_a_channel_as_wide_as_narrow_cells_is_left_alone() -> None:
    xs = section(NARROW_CELLS * SPACING)
    assert not reconstruct(xs)
    assert xs.profile is None and not xs.subcell


def test_a_carved_channel_keeps_its_carve_between_its_banks() -> None:
    """The carve's exact profile stays between the banks; out from them the ground before the carve is reconstructed."""
    xs = section(60.0)
    half = xs.left_bank_distance
    banks = Banks("width_to_depth_ratio", half, half, BANK, BANK, False, True)
    carve_channel(xs, banks, 2.0, trapezoid_height=0.2, bank_elevation=BANK, keep_ground=True)
    carved = xs.profile
    assert xs.ground is not None and xs.carved_banks == (half, half)

    assert reconstruct(xs)
    channel = np.abs(carved.stations) <= half + 1e-9
    kept = np.abs(xs.profile.stations) <= half + 1e-9
    np.testing.assert_array_equal(xs.profile.stations[kept], carved.stations[channel])
    np.testing.assert_array_equal(xs.profile.elevations[kept], carved.elevations[channel])
    assert xs.profile.stations.size > carved.stations.size


def test_without_the_ground_before_its_carve_a_carved_channel_is_left_alone() -> None:
    xs = section(60.0)
    half = xs.left_bank_distance
    carve_channel(xs, Banks("width_to_depth_ratio", half, half, BANK, BANK, False, True), 2.0, trapezoid_height=0.2,
                  bank_elevation=BANK)
    assert not reconstruct(xs)


def test_an_uncarved_cross_section_without_banks_is_reconstructed_around_its_stream_cell() -> None:
    xs = section(30.0)
    xs.left_bank_distance = xs.right_bank_distance = -1.0
    assert reconstruct(xs)


def test_many_cross_sections_at_once_keep_their_cells_means() -> None:
    sections = [section(width, offset) for width in (30.0, 60.0, 90.0) for offset in (0.0, 10.0, 20.0)] * 40
    means = [xs.elevations.copy() for xs in sections]

    assert reconstruct_all(sections) == len(sections)
    for xs, z in zip(sections, means):
        center = z.size // 2
        inside = np.abs(xs.profile.stations) < SPACING / 2 - 1e-9
        assert xs.profile.elevations[inside].mean() == pytest.approx(z[center], abs=1e-6)
