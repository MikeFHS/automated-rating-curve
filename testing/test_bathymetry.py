from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

from arc import hydraulics
from arc.bathymetry import (
    Banks,
    bank_control_elevation,
    banks_by_flat_water,
    banks_by_land_cover,
    banks_by_width_to_depth_ratio,
    banks_for_width,
    bathymetry_depth,
    burn_into_raster,
    carve_channel,
    channel_depth,
    find_banks,
    in_bank,
    ordinate_cells,
    power_law_geometry,
    sample_land_cover,
    set_bank_distances,
    set_in_bank_roughness,
    single_cell_banks,
    trapezoid_depth,
)
from arc.hydraulics import DepthRoughness
from arc.xsection.xsection import XSection

BED = 100.0
SLOPE = 0.001
N = 0.03
WALL = 9999.0
WATER = 80
NO_BANKS = Banks("none", math.nan, math.nan, math.nan, math.nan, False, False)


def make_section(elevations, spacing=1.0) -> XSection:
    elevations = np.asarray(elevations, dtype=np.float64)
    return XSection(elevations.copy(), np.full(elevations.size, 0.035), spacing)


def symmetric(side_profile, spacing=1.0) -> XSection:
    """A cross section with the same profile out from the stream cell on both sides."""
    side = np.asarray(side_profile, dtype=np.float64)
    return make_section(np.concatenate([side[:0:-1], side]), spacing)


def channel_and_floodplains() -> XSection:
    """A 20 m bed, banks rising 1:1 to 2 m 12 m out, then floodplains rising 1 cm per metre, at 1 m spacing."""
    distance = np.arange(0.0, 31.0)
    return symmetric(BED + np.clip(distance - 10.0, 0.0, 2.0) + 0.01 * np.maximum(distance - 12.0, 0.0))


def trapezoid_discharge(depth, bottom, top, slope=SLOPE, n=N):
    side = (top - bottom) / 2
    area = depth * (bottom + top) / 2
    perimeter = bottom + 2 * math.hypot(side, depth)
    return area * (area / perimeter) ** (2 / 3) * math.sqrt(slope) / n


def random_section(rng, size, spacing) -> XSection:
    """A random valley whose banks rise, flatten, and sometimes dip, rounded so that some ordinates tie."""
    center = size // 2
    left = np.cumsum(rng.normal(0.2, 0.4, center))
    right = np.cumsum(rng.normal(0.2, 0.4, size - 1 - center))
    elevations = BED + np.concatenate([left[::-1], [0.0], right])
    return make_section(np.round(elevations, int(rng.integers(2, 5))), spacing)


# --- Banks -------------------------------------------------------------------------------------------------------


def test_flat_water_banks_are_where_the_ground_is_0p1_m_above_the_stream_cell() -> None:
    """Banks rising 4 cm per metre reach 0.1 m 2.5 m out."""
    banks = banks_by_flat_water(symmetric(BED + 0.04 * np.arange(0.0, 21.0)))

    assert banks.left == pytest.approx(2.5, rel=1e-12)
    assert banks.right == pytest.approx(2.5, rel=1e-12)
    assert banks.left_elevation == pytest.approx(BED + 0.1, rel=1e-12)
    assert banks.valid


def test_width_to_depth_banks_are_where_the_water_spills_out_of_the_channel() -> None:
    """The ratio of top width to depth falls while the channel fills and rises once the water reaches the
    floodplains, so the banks are at the bank tops, 12 m out."""
    banks = banks_by_width_to_depth_ratio(channel_and_floodplains())

    assert (banks.left, banks.right) == pytest.approx((12.0, 12.0), rel=1e-12)
    assert banks.top_width == pytest.approx(24.0, rel=1e-12)
    assert (banks.left_elevation, banks.right_elevation) == pytest.approx((BED + 2.0, BED + 2.0), rel=1e-12)
    assert banks.valid and not banks.single_cell


def test_width_to_depth_banks_don_t_look_past_ground_25_m_up() -> None:
    """The stages come from the ordinates out to the first more than 25 m above the stream cell on each side, so the
    ground beyond it, however low or high, doesn't move the banks, and a cross section can end there."""
    rng = np.random.default_rng(5)
    for _ in range(300):
        spacing = float(rng.choice([1.0, 10.0]))
        xs = random_section(rng, 2 * int(rng.integers(5, 40)) + 1, spacing)
        center = xs.elevations.size // 2
        left, right = center - int(rng.integers(1, center)), center + int(rng.integers(1, center))
        elevations = xs.elevations.copy()
        elevations[[left, right]] = elevations[center] + 26.0
        beyond = elevations.copy()
        beyond[:left] = BED + rng.uniform(-5.0, 30.0, left)
        beyond[right + 1:] = BED + rng.uniform(-5.0, 30.0, beyond.size - right - 1)

        np.testing.assert_equal(banks_by_width_to_depth_ratio(make_section(beyond, spacing)),
                                banks_by_width_to_depth_ratio(make_section(elevations, spacing)))


def valley_side(dip: float) -> XSection:
    """A stream cell partway up a valley side, at 10 m spacing: the ground rises 1 m in 10 to its left, and to its
    right falls `dip` metres over 2 km to the valley floor, then rises 1 m in 10 up the far side."""
    distance = 10.0 * np.arange(0.0, 301.0)
    right = BED - dip * np.minimum(distance, 2000.0) / 2000.0 + 0.1 * np.maximum(distance - 2000.0, 0.0)
    return make_section(np.concatenate([(BED + 0.1 * distance)[:0:-1], right]), spacing=10.0)


def test_banks_from_the_dem_dont_take_the_far_side_of_a_valley_its_water_drains_into() -> None:
    """Water above a stream cell on a valley side drains away down it, and the ground that stops it is the valley's
    far side, kilometres out (as along a stream on the side of the Salt Lake valley, n40w112): no bank there, from
    either method. Ground dipping no more than MAX_BANK_DIP below the stream cell holds the water in its channel."""
    from arc.bathymetry.banks import MAX_BANK_DIP
    held, drains = valley_side(0.9 * MAX_BANK_DIP), valley_side(30.0)

    assert banks_by_flat_water(held).right > 2000.0 and banks_by_flat_water(held).valid
    assert math.isnan(banks_by_flat_water(drains).right) and not banks_by_flat_water(drains).valid
    assert math.isnan(banks_by_width_to_depth_ratio(drains).right)
    assert find_banks(drains).method == "none"


def flat_water_edge(side: np.ndarray, spacing: float) -> float:
    """How far out from the stream cell (side[0]) the ground first reaches 0.1 m above it, or the end of the side,
    but NaN if the ground before then falls more than MAX_BANK_DIP below the stream cell."""
    from arc.bathymetry.banks import MAX_BANK_DIP
    level = side[0] + 0.1
    for k in range(1, side.size):
        if side[k] >= level:
            return spacing * (k - 1 + (level - side[k - 1]) / (side[k] - side[k - 1]))
        if side[k] < side[0] - MAX_BANK_DIP:
            return math.nan
    return spacing * (side.size - 1)


def test_flat_water_banks_are_where_the_ground_first_reaches_the_flat_water() -> None:
    """On random sections, each side's bank is where the ground first reaches 0.1 m above the stream cell, or the
    end of the side if it never does, unless the water drains away over ground more than MAX_BANK_DIP below it."""
    rng = np.random.default_rng(3)
    for _ in range(300):
        spacing = float(rng.choice([1.0, 10.0]))
        xs = random_section(rng, 2 * int(rng.integers(3, 40)) + 1, spacing)
        center = xs.elevations.size // 2

        banks = banks_by_flat_water(xs)

        assert banks.left == pytest.approx(flat_water_edge(xs.elevations[center::-1], spacing), rel=1e-12, nan_ok=True)
        assert banks.right == pytest.approx(flat_water_edge(xs.elevations[center:], spacing), rel=1e-12, nan_ok=True)


def test_land_cover_banks_are_half_way_to_the_first_land() -> None:
    """Water from the stream cell three ordinates out on the left and five on the right, at 10 m spacing."""
    land_cover = np.full(21, 30.0)
    land_cover[10 - 3:10 + 6] = WATER
    banks = banks_by_land_cover(make_section(np.full(21, BED), spacing=10.0), land_cover, WATER)

    assert (banks.left, banks.right) == pytest.approx((35.0, 55.0), rel=1e-12)
    assert banks.top_width == pytest.approx(90.0, rel=1e-12)  # the nine cells of water


def test_land_cover_banks_are_half_way_to_the_first_land_at_random() -> None:
    """On random land cover: each bank is half way from the last water out from the stream cell to the first land,
    and the banks are valid if the water is at least two cells wide."""
    rng = np.random.default_rng(4)
    for _ in range(300):
        size = 2 * int(rng.integers(3, 30)) + 1
        center = size // 2
        land_cover = np.where(rng.random(size) < 0.3, 30, WATER)
        land_cover[center] = WATER
        land_cover[0] = land_cover[-1] = 30  # land on both sides
        xs = random_section(rng, size, 10.0)
        first_land = [next(k for k in range(1, center + 1) if side[k] != WATER)
                      for side in (land_cover[center::-1], land_cover[center:])]

        banks = banks_by_land_cover(xs, land_cover, WATER)

        assert (banks.left, banks.right) == pytest.approx(((first_land[0] - 0.5) * 10.0,
                                                           (first_land[1] - 0.5) * 10.0), rel=1e-12)
        assert banks.valid == (first_land[0] + first_land[1] - 1 >= 2)


def test_land_cover_needs_water_at_the_stream_cell_and_land_on_each_side() -> None:
    """A side that is water all the way out has no bank, and nor does a stream cell that isn't water, so the search
    goes on to the other methods."""
    xs = channel_and_floodplains()
    all_water_left = np.full(61, float(WATER))
    all_water_left[34:] = 30.0
    stream_cell_on_land = np.full(61, 30.0)
    stream_cell_on_land[25:30] = WATER

    assert not banks_by_land_cover(xs, all_water_left, WATER).valid
    assert math.isnan(banks_by_land_cover(xs, all_water_left, WATER).left)
    assert not banks_by_land_cover(xs, stream_cell_on_land, WATER).valid
    # So the search goes on to the width-to-depth ratio
    assert find_banks(xs, land_cover=all_water_left, water_value=WATER).method == "width_to_depth_ratio"


def test_land_cover_off_the_raster_is_no_bank() -> None:
    land_cover = np.full(21, float(WATER))
    land_cover[:3] = np.nan
    land_cover[15:] = 30.0

    banks = banks_by_land_cover(make_section(np.full(21, BED)), land_cover, WATER)

    assert math.isnan(banks.left)
    assert banks.right == pytest.approx(4.5, rel=1e-12)


@pytest.mark.parametrize("spacing, widest", [(1.0, 1.0), (14.9, 14.9), (15.0, 15.0), (30.0, 30.0)])
def test_a_width_prior_of_one_spacing_or_less_makes_a_single_cell_channel(spacing: float, widest: float) -> None:
    """At any resolution (legacy ARC allowed two spacings below 15 m). The channel is the prior's width, and its banks'
    elevations are the ordinates' either side."""
    xs = symmetric(BED + np.arange(0.0, 11.0), spacing)

    assert find_banks(xs, target_width=widest).single_cell
    assert not find_banks(xs, target_width=widest * 1.001).single_cell
    for width in (widest, 0.3 * spacing):
        banks = find_banks(xs, target_width=width)
        assert (banks.left, banks.right) == (width / 2, width / 2)
        assert (banks.left_elevation, banks.right_elevation) == (BED + 1.0, BED + 1.0)
        assert banks.valid and banks.method == "single_cell"


def test_a_single_cell_channel_is_the_stream_cell_s_width() -> None:
    """Half a spacing either side of the stream cell, with the neighbours' elevations."""
    xs = make_section([BED + 3.0, BED + 1.5, BED, BED + 0.5, BED + 3.0], spacing=10.0)
    banks = single_cell_banks(xs)

    assert (banks.left, banks.right, banks.top_width) == (5.0, 5.0, 10.0)
    assert (banks.left_elevation, banks.right_elevation) == (BED + 1.5, BED + 0.5)
    assert banks.valid and banks.single_cell


def test_banks_a_width_puts_within_a_spacing_take_the_neighbours_ground() -> None:
    """The DEM can't show a bank inside the stream cell, so the ordinate beside it stands in. Further out, the ground
    at the bank, as for any other bank."""
    xs = make_section([BED + 5.0, BED + 3.0, BED + 1.0, BED, BED + 2.0, BED + 4.0, BED + 6.0], spacing=10.0)

    for width in (4.0, 10.0, 16.0, 20.0):
        banks = banks_for_width(xs, width)
        assert (banks.left_elevation, banks.right_elevation) == (BED + 1.0, BED + 2.0)
    banks = banks_for_width(xs, 30.0)  # 15 m out, half way to the next ordinates
    assert (banks.left_elevation, banks.right_elevation) == pytest.approx((BED + 2.0, BED + 3.0))
    # Banks found from the DEM keep the ground where they are, however near: flat water 1 m from the stream cell
    assert banks_by_flat_water(xs).left_elevation == pytest.approx(BED + 0.1)


def test_a_single_cell_channel_needs_an_ordinate_on_the_raster_each_side() -> None:
    xs = make_section([BED + 5.0, WALL, BED, BED + 1.0, BED + 2.0])

    assert not find_banks(xs, target_width=1.0).single_cell
    assert not single_cell_banks(xs).valid


def test_the_search_order() -> None:
    """The land cover, the width-to-depth ratio, flat water, and only then the width prior."""
    xs = channel_and_floodplains()
    land_cover = np.full(61, 30.0)
    land_cover[30 - 5:30 + 6] = WATER

    assert find_banks(xs, target_width=1.5, land_cover=land_cover, water_value=WATER).method == "land_cover"
    assert find_banks(xs, land_cover=land_cover, water_value=WATER).method == "land_cover"
    assert find_banks(xs, target_width=1.5).method == "width_to_depth_ratio"
    assert find_banks(xs).method == "width_to_depth_ratio"
    # A V's width-to-depth ratio never rises, so its banks are flat water's
    v = symmetric(BED + 0.02 * np.arange(0.0, 41.0))
    assert find_banks(v, target_width=1.5).method == find_banks(v).method == "flat_water"
    # A V too steep for its flat water to span two spacings has no valid banks, unless a prior gives it a single cell
    steep = symmetric(BED + 0.5 * np.arange(0.0, 5.0))
    none = find_banks(steep)
    assert (none.method, none.valid) == ("none", False)
    assert math.isnan(none.left)
    assert find_banks(steep, target_width=0.8).method == "single_cell"
    # A prior wider than a single cell gives no banks: the bathymetry gives the channel its reach's median width
    assert not find_banks(steep, target_width=1.5).valid


def test_banks_need_a_channel_two_spacings_wide() -> None:
    """Flat water only 0.8 spacings out each side is too narrow to resolve."""
    xs = symmetric(BED + 0.125 * np.arange(0.0, 11.0))
    banks = banks_by_flat_water(xs)

    assert banks.left == pytest.approx(0.8, rel=1e-12)
    assert not banks.valid


def test_banks_for_a_width_split_it_between_the_sides() -> None:
    xs = make_section(np.full(41, BED))
    walled = make_section(np.concatenate([np.full(17, WALL), np.full(24, BED)]))  # 3 ordinates left of the stream cell

    assert (banks_for_width(xs, 10.0).left, banks_for_width(xs, 10.0).right) == pytest.approx((5.0, 5.0))
    assert (banks_for_width(walled, 10.0).left, banks_for_width(walled, 10.0).right) == pytest.approx((3.0, 7.0))
    # 3 ordinates left of the stream cell and 7 right: of 12 m, the right side can hold only 7
    short_both = make_section([WALL] * 7 + [BED] * 11 + [WALL] * 3)
    assert (banks_for_width(short_both, 12.0).left, banks_for_width(short_both, 12.0).right) == pytest.approx((3.0, 7.0))
    assert banks_for_width(xs, 100.0).top_width == pytest.approx(40.0)  # the whole section
    assert banks_for_width(xs, 100.0).valid
    # Any width, narrower than a cell included
    for width in (0.8, 1.5):
        banks = banks_for_width(xs, width)
        assert (banks.left, banks.right) == pytest.approx((width / 2, width / 2))
        assert banks.valid and banks.method == "target_width" and not banks.single_cell
    assert not banks_for_width(make_section([WALL, BED, BED + 1.0]), 0.5).valid  # off the raster to the left
    assert not banks_for_width(xs, 0.0).valid and not banks_for_width(xs, math.nan).valid


def test_valid_banks_divide_the_hydraulics() -> None:
    xs = channel_and_floodplains()
    banks = find_banks(xs)

    set_bank_distances(xs, banks)

    assert (xs.left_bank_distance, xs.right_bank_distance) == (banks.left, banks.right)
    parts = hydraulics.compound_geometry(xs, wse=103.0)
    assert parts.channel.top_width == pytest.approx(24.0)  # bank to bank
    assert parts.left_overbank.area > 0.0 and parts.right_overbank.area > 0.0
    divided = hydraulics.discharge(xs, SLOPE, wse=103.0)
    set_bank_distances(xs, NO_BANKS)
    assert (xs.left_bank_distance, xs.right_bank_distance) == (-1.0, -1.0)
    assert hydraulics.discharge(xs, SLOPE, wse=103.0) < divided  # as one section, the floodplains hold it back


def test_the_bank_control_elevation_is_the_lower_bank_above_the_stream_cell() -> None:
    banks = Banks("test", 5.0, 7.0, 103.0, 102.5, False, True)

    assert bank_control_elevation(banks, BED) == 102.5
    assert bank_control_elevation(banks._replace(right_elevation=BED), BED) == 103.0  # at the stream cell: ignored
    assert math.isnan(bank_control_elevation(banks._replace(left_elevation=99.0, right_elevation=math.nan), BED))


def test_in_bank_ordinates_get_the_water_s_roughness() -> None:
    xs = make_section(np.full(21, BED), spacing=10.0)
    banks = Banks("test", 35.0, 20.0, BED, BED, False, True)

    assert np.flatnonzero(in_bank(xs, banks)).tolist() == [7, 8, 9, 10, 11, 12]
    set_in_bank_roughness(xs, banks, 0.02)
    assert xs.mannings_n[7:13].tolist() == [0.02] * 6
    assert xs.mannings_n[6] == xs.mannings_n[13] == 0.035
    assert not in_bank(xs, banks._replace(valid=False)).any()
    # A single-cell channel's banks are half a spacing out, so only its stream cell is between them
    assert np.flatnonzero(in_bank(xs, single_cell_banks(xs))).tolist() == [10]


# --- Depth --------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("q", [0.01, 1.0, 20.0, 500.0])
def test_a_trapezoid_s_depth_carries_the_baseflow(q: float) -> None:
    depth = trapezoid_depth(q, 16.0, 24.0, SLOPE, N)

    assert trapezoid_discharge(depth, 16.0, 24.0) == pytest.approx(q, rel=1e-12)


def test_every_channel_is_a_trapezoid_as_wide_as_its_banks() -> None:
    """With a constant n, the cross section's own, and ground nowhere below the channel, the depth is the trapezoid's."""
    xs = channel_and_floodplains()
    xs.mannings_n[:] = N
    banks = find_banks(xs)  # 24 m apart

    assert channel_depth(xs, banks, 20.0, SLOPE, trapezoid_height=0.2) == \
        pytest.approx(trapezoid_depth(20.0, 14.4, 24.0, SLOPE, N), rel=1e-12)
    # Without valid banks, one cell wide: at 1 m spacing, a trapezoid 1 m across the top
    assert channel_depth(xs, banks._replace(valid=False), 2.0, SLOPE, trapezoid_height=0.2) == \
        pytest.approx(trapezoid_depth(2.0, 0.6, 1.0, SLOPE, N), rel=1e-12)


def test_a_single_cell_channel_is_a_trapezoid_whatever_the_ground_beside_it() -> None:
    """A trapezoid of the single cell's width, or as wide as the width prior that made it, if one did."""
    xs = make_section([BED + 3.0, BED + 0.5, BED, BED + 0.8, BED + 3.0], spacing=10.0)
    xs.mannings_n[:] = N

    assert channel_depth(xs, single_cell_banks(xs), 2.0, SLOPE, trapezoid_height=0.2) == \
        pytest.approx(trapezoid_depth(2.0, 6.0, 10.0, SLOPE, N), rel=1e-12)
    prior = find_banks(xs, target_width=4.0)
    assert prior.single_cell
    assert channel_depth(xs, prior, 2.0, SLOPE, trapezoid_height=0.2) == \
        pytest.approx(trapezoid_depth(2.0, 2.4, 4.0, SLOPE, N), rel=1e-12)
    # Narrower is deeper, for the same baseflow
    assert channel_depth(xs, prior, 2.0, SLOPE, trapezoid_height=0.2) > \
        channel_depth(xs, single_cell_banks(xs), 2.0, SLOPE, trapezoid_height=0.2)


@pytest.mark.parametrize("spacing", [1.0, 30.0])
@pytest.mark.parametrize("bank_elevation", [None, BED + 0.4], ids=["water surface", "bank elevation"])
@pytest.mark.parametrize(("roughness", "slope_factor"), [
    (None, 1.0), (DepthRoughness(), 1.0), (DepthRoughness(), 1.7), (DepthRoughness(3.0, 4.0, 0.8), 0.6)],
    ids=["constant n", "depth-varying n", "with a slope factor", "other factors"])
def test_the_carved_channel_carries_the_baseflow_at_its_level_by_the_rating_curve_s_own_hydraulics(
        spacing: float, bank_elevation, roughness, slope_factor: float) -> None:
    """Whatever the roughness and slope factor, the rating curve's hydraulics on the carved cross section carry the
    baseflow with the water at the reference level: the stream cell's elevation, or the bank elevation. Depth-varying
    n is rougher in shallow water, so the channel is carved deeper than a constant n's."""
    offsets = spacing * np.arange(-60, 61)
    xs = make_section(np.where(np.abs(offsets) <= 40.0, BED, BED + 0.1 * (np.abs(offsets) - 40.0)), spacing)
    banks = find_banks(xs)
    set_in_bank_roughness(xs, banks, N)

    depth = channel_depth(xs, banks, 25.0, SLOPE, trapezoid_height=0.2, mannings_n=N, roughness=roughness,
                          slope_factor=slope_factor, bank_elevation=bank_elevation)
    constant_n = channel_depth(xs, banks, 25.0, SLOPE, trapezoid_height=0.2, mannings_n=N,
                               bank_elevation=bank_elevation)
    carve_channel(xs, banks, depth, trapezoid_height=0.2, bank_elevation=bank_elevation)
    set_bank_distances(xs, banks)

    level = BED if bank_elevation is None else bank_elevation
    assert hydraulics.discharge(xs, SLOPE, wse=level, roughness=roughness, slope_factor=slope_factor) == \
        pytest.approx(25.0, rel=1e-10)
    if roughness is not None and slope_factor == 1.0:
        assert depth > constant_n


def test_no_baseflow_is_no_depth_and_no_slope_no_answer() -> None:
    xs = channel_and_floodplains()
    banks = find_banks(xs)

    assert channel_depth(xs, banks, 0.0, SLOPE, trapezoid_height=0.2) == 0.0
    assert math.isnan(channel_depth(xs, banks, 5.0, 0.0, trapezoid_height=0.2))
    assert math.isnan(trapezoid_depth(5.0, 0.0, 0.0, SLOPE, N))


def test_a_valid_target_depth_wins_and_always_applies() -> None:
    xs = channel_and_floodplains()
    banks = find_banks(xs)

    assert bathymetry_depth(xs, banks, 0.0, SLOPE, trapezoid_height=0.2, target_depth=1.7) == (1.7, True, "target_depth")
    solved = bathymetry_depth(xs, banks, 20.0, SLOPE, trapezoid_height=0.2, target_depth=math.nan)
    assert solved == (channel_depth(xs, banks, 20.0, SLOPE, trapezoid_height=0.2), True, "baseflow_manning")
    # Without a target or baseflow there is nothing to carve
    assert bathymetry_depth(xs, banks, 0.0, SLOPE, trapezoid_height=0.2) == (0.0, False, "baseflow_manning")


def test_power_law_geometry() -> None:
    assert power_law_geometry(100.0, 0.3, 0.5, 2.0, 0.4) == pytest.approx((3.0, 2.0 * 100.0 ** 0.4))
    assert power_law_geometry(100.0, 0.3, 0.5, None, 0.4) == (pytest.approx(3.0), None)


@pytest.mark.parametrize("height", [-0.1, 0.51, math.nan])
def test_the_trapezoid_height_is_at_most_a_half(height: float) -> None:
    xs = channel_and_floodplains()
    with pytest.raises(ValueError, match="trapezoid_height"):
        channel_depth(xs, find_banks(xs), 1.0, SLOPE, trapezoid_height=height)
    with pytest.raises(ValueError, match="trapezoid_height"):
        carve_channel(xs, find_banks(xs), 1.0, trapezoid_height=height)


# --- Carving the channel ------------------------------------------------------------------------------------------

BANKS_10_M_OUT = Banks("test", 10.0, 10.0, BED, BED, False, True)


def test_the_channel_is_a_trapezoid_between_its_banks() -> None:
    """Banks 10 m out, 2 m deep, with each side sloping over 0.2 of the 20 m top width: 4 m."""
    xs = make_section(np.full(25, BED))

    changed = carve_channel(xs, BANKS_10_M_OUT, 2.0, trapezoid_height=0.2)

    expected = np.full(25, BED)
    expected[12 - 9:12 + 10] = [99.5, 99.0, 98.5] + [98.0] * 13 + [98.5, 99.0, 99.5]
    np.testing.assert_allclose(xs.elevations, expected, rtol=1e-15)
    assert np.flatnonzero(changed).tolist() == list(range(3, 22))  # the banks themselves stay at 100 m


def test_the_carved_channel_carries_the_baseflow() -> None:
    """A 10 m flat bed between walls rising 3 m per metre, at 5 cm spacing. Its flat-water banks are 0.1 m up the
    walls. Carved, it carries the baseflow with the water at the stream cell's old elevation."""
    offsets = 0.05 * np.arange(-400, 401)
    xs = make_section(np.where(np.abs(offsets) <= 5.0, BED, BED + 3.0 * (np.abs(offsets) - 5.0)), spacing=0.05)
    xs.mannings_n[:] = N
    banks = find_banks(xs)
    depth = channel_depth(xs, banks, 5.0, SLOPE, trapezoid_height=0.2)

    carve_channel(xs, banks, depth, trapezoid_height=0.2)

    assert banks.method == "flat_water"
    assert banks.top_width == pytest.approx(10.0 + 2 * 0.1 / 3, rel=1e-9)
    # Exactly, since the hydraulics see the carved channel's own shape
    assert hydraulics.discharge(xs, SLOPE, wse=BED) == pytest.approx(5.0, rel=1e-9)


def test_without_bank_elevations_the_channel_only_lowers_the_ground() -> None:
    """Ground 6 to 8 m from the stream cell is already below the carved slope."""
    xs = make_section(np.full(25, BED))
    xs.elevations[4:7] = 97.0

    changed = carve_channel(xs, BANKS_10_M_OUT, 2.0, trapezoid_height=0.2)

    assert (xs.elevations[4:7] == 97.0).all()
    assert not changed[4:7].any()
    assert xs.elevations[12] == 98.0


def test_with_bank_elevations_the_channel_can_raise_the_ground_too() -> None:
    """The bank elevation, 101 m, is above this ground, so the bed at 99 m and the slopes up to 101 m at the banks
    raise it. (The user's ruling: bathymetry with bank elevations may fill as well as cut.)"""
    xs = make_section(np.full(25, BED))

    changed = carve_channel(xs, BANKS_10_M_OUT, 2.0, trapezoid_height=0.2, bank_elevation=101.0)

    expected = np.full(25, BED)
    expected[12 - 10:12 + 11] = [101.0, 100.5, 100.0, 99.5] + [99.0] * 13 + [99.5, 100.0, 100.5, 101.0]
    np.testing.assert_allclose(xs.elevations, expected, rtol=1e-15)
    assert np.flatnonzero(changed).tolist() == list(range(2, 23))


def test_with_bank_elevations_the_channel_stays_between_its_banks() -> None:
    """Banks 2 m out at 11 m, 1.5 m deep: the bank ordinates are the channel's top, and the ground beyond them
    stays."""
    xs = make_section([12.0, 11.0, 10.4, 10.0, 10.5, 11.0, 12.0])
    banks = Banks("test", 2.0, 2.0, 11.0, 11.0, False, True)

    carve_channel(xs, banks, 1.5, trapezoid_height=0.1, bank_elevation=11.0)

    assert xs.elevations.tolist() == pytest.approx([12.0, 11.0, 9.5, 9.5, 9.5, 11.0, 12.0])


def test_a_lopsided_channel_carves_one_stream_cell_elevation() -> None:
    """Banks 20 m left and 1 m right of the stream cell, so it lies within the right bank's 4.2 m slope. Each
    ordinate takes the channel's elevation at its distance from the nearer bank."""
    elevations = np.full(43, 13.0)
    elevations[1:23] = 10.0  # the stream cell is ordinate 21
    xs = make_section(elevations)

    carve_channel(xs, Banks("test", 20.0, 1.0, 10.0, 10.0, False, True), 2.0, trapezoid_height=0.2)

    assert xs.elevations[21] == pytest.approx(8.0 + 2.0 * (1.0 - 1.0 / 4.2), rel=1e-15)
    assert xs.elevations[20] == pytest.approx(8.0 + 2.0 * (1.0 - 2.0 / 4.2), rel=1e-15)  # left, but 2 m from the right bank
    assert xs.elevations[11] == 8.0  # 10 m from the left bank and 11 m from the right, beyond both slopes
    assert xs.elevations[5] == pytest.approx(8.0 + 2.0 * (1.0 - 4.0 / 4.2), rel=1e-15)  # 4 m from the left bank

    # The same channel the other way round
    mirrored = make_section(elevations[::-1].copy())
    carve_channel(mirrored, Banks("test", 1.0, 20.0, 10.0, 10.0, False, True), 2.0, trapezoid_height=0.2)
    np.testing.assert_allclose(mirrored.elevations, xs.elevations[::-1], rtol=1e-15)


@pytest.mark.parametrize("use_banks", [False, True])
def test_a_symmetric_channel_is_its_trapezoid_at_every_ordinate(use_banks: bool) -> None:
    """On random sections, banks and trapezoid heights: between the banks each ordinate is on the trapezoid, whose
    sides rise from its bed to the reference level over trapezoid_height of its top width. Without bank elevations
    the ground is only ever lowered."""
    rng = np.random.default_rng(9)
    for _ in range(300):
        spacing = float(rng.choice([1.0, 3.0, 10.0]))
        center = int(rng.integers(3, 30))
        half = float(rng.uniform(1.0, center - 1) * spacing)
        trapezoid_height = float(rng.uniform(0.01, 0.5))
        depth = float(rng.uniform(0.1, 5.0))
        profile = BED + np.abs(np.arange(-center, center + 1)) * spacing * rng.uniform(0.01, 0.3)
        profile += rng.normal(0.0, 0.1, profile.size)
        profile[center] = BED
        xs = make_section(profile, spacing)
        reference = BED + 1.0 if use_banks else BED

        carve_channel(xs, Banks("test", half, half, np.nan, np.nan, False, True), depth,
                      trapezoid_height=trapezoid_height, bank_elevation=reference if use_banks else None)

        to_bank = half - spacing * np.abs(np.arange(-center, center + 1))
        channel = reference - depth + depth * np.maximum(1.0 - to_bank / (2 * half * trapezoid_height), 0.0)
        carved = channel if use_banks else np.minimum(profile, channel)
        np.testing.assert_allclose(xs.elevations, np.where(to_bank >= -1e-9 * spacing, carved, profile), atol=1e-12)


def test_a_single_cell_channel_lowers_only_the_stream_cell() -> None:
    xs = make_section([BED + 3.0, BED + 1.0, BED, BED + 1.0, BED + 3.0])

    changed = carve_channel(xs, single_cell_banks(xs), 0.6, trapezoid_height=0.2)

    assert xs.elevations.tolist() == pytest.approx([BED + 3.0, BED + 1.0, BED - 0.6, BED + 1.0, BED + 3.0])
    assert np.flatnonzero(changed).tolist() == [2]
    # Banks that aren't valid carve the same way
    xs = make_section([BED + 3.0, BED + 1.0, BED, BED + 1.0, BED + 3.0])
    carve_channel(xs, NO_BANKS, 0.6, trapezoid_height=0.2)
    assert xs.elevations[2] == pytest.approx(BED - 0.6)


def test_what_isn_t_carved() -> None:
    flat = lambda: make_section(np.full(25, BED))

    # Any depth is carved, with bank elevations or without (legacy left 25 m or more uncarved without them)
    deep = flat()
    assert carve_channel(deep, BANKS_10_M_OUT, 40.0, trapezoid_height=0.2).any()
    assert deep.elevations[12] == pytest.approx(BED - 40.0)
    assert carve_channel(flat(), BANKS_10_M_OUT, 40.0, trapezoid_height=0.2, bank_elevation=BED).any()
    # A depth or bank elevation that isn't finite
    xs = flat()
    assert not carve_channel(xs, BANKS_10_M_OUT, math.nan, trapezoid_height=0.2, bank_elevation=BED).any()
    assert not carve_channel(xs, BANKS_10_M_OUT, 1.0, trapezoid_height=0.2, bank_elevation=math.nan).any()
    assert (xs.elevations == BED).all()
    # A single-cell channel without a neighbour on the raster
    walled = make_section([WALL, WALL, BED, BED + 1.0, BED + 1.0])
    assert not carve_channel(walled, single_cell_banks(walled), 1.0, trapezoid_height=0.2).any()


def test_a_trapezoid_height_of_zero_is_a_rectangle() -> None:
    xs = make_section(np.full(25, BED))

    carve_channel(xs, BANKS_10_M_OUT, 2.0, trapezoid_height=0.0, bank_elevation=BED + 1.0)

    assert xs.elevations[2] == xs.elevations[22] == BED + 1.0  # the banks are the rectangle's top
    assert (xs.elevations[3:22] == BED - 1.0).all()
    # Its walls are vertical faces in the profile
    geometry = hydraulics.hydraulic_geometry(xs, wse=BED)
    assert (geometry.area, geometry.top_width, geometry.wetted_perimeter) == pytest.approx((20.0, 20.0, 22.0))


# --- The carved channel's profile -----------------------------------------------------------------------------------


def profile_at(xs: XSection, offsets) -> np.ndarray:
    """The profile's elevation at these distances from the stream cell, taking a vertical face's inner end."""
    stations, elevations = xs.profile.stations, xs.profile.elevations
    values = []
    for offset in np.atleast_1d(offsets):
        at = np.flatnonzero(stations == offset)
        if at.size:
            values.append(elevations[at[-1] if offset < 0 else at[0]])
        else:
            values.append(np.interp(offset, stations, elevations))
    return np.array(values)


def test_a_channel_narrower_than_a_cell_is_its_own_shape_in_the_hydraulics() -> None:
    """A 12 m wide trapezoid 1.5 m deep in a stream cell 30 m wide. The raster gets the bed at the stream cell, and
    the ordinates either side keep their ground, but the hydraulics see the channel: at its banks' level its water
    is the trapezoid, 12 m across, and it carries the baseflow its depth was solved for."""
    xs = make_section([BED + 6.0, BED + 4.0, BED + 2.0, BED, BED + 2.0, BED + 4.0, BED + 6.0], spacing=30.0)
    xs.mannings_n[:] = N
    banks = find_banks(xs, target_width=12.0)
    depth = channel_depth(xs, banks, 3.0, SLOPE, trapezoid_height=0.2)

    changed = carve_channel(xs, banks, depth, trapezoid_height=0.2)

    assert banks.single_cell and banks.top_width == 12.0
    assert np.flatnonzero(changed).tolist() == [3]
    assert xs.elevations[3] == pytest.approx(BED - depth)
    assert xs.elevations[[2, 4]].tolist() == [BED + 2.0, BED + 2.0]
    assert xs.profile.stations[xs.profile.center] == 0.0
    geometry = hydraulics.hydraulic_geometry(xs, wse=BED)
    assert geometry.top_width == pytest.approx(12.0, rel=1e-12)
    assert geometry.area == pytest.approx(depth * (12.0 + 7.2) / 2, rel=1e-12)
    assert hydraulics.discharge(xs, SLOPE, wse=BED) == pytest.approx(3.0, rel=1e-9)
    # On the ordinates alone the channel would reach the neighbours, 60 m across
    plain = make_section(xs.elevations, spacing=30.0)
    assert hydraulics.hydraulic_geometry(plain, wse=BED).top_width == pytest.approx(60.0 * depth / (depth + 2.0))


def test_a_two_spacing_trapezoid_is_a_trapezoid_not_a_triangle() -> None:
    """Banks 1 m either side at 1 m spacing: the ordinates are the banks and the stream cell, which on their own make
    a triangle with 5/8 of the trapezoid's area at a trapezoid height of 0.2."""
    xs = make_section(np.full(9, BED))
    banks = Banks("test", 1.0, 1.0, BED, BED, False, True)

    carve_channel(xs, banks, 1.0, trapezoid_height=0.2, bank_elevation=BED)

    assert xs.elevations.tolist() == [BED] * 3 + [BED, BED - 1.0, BED] + [BED] * 3
    assert hydraulics.hydraulic_geometry(xs, wse=BED).area == pytest.approx(1.6)
    assert hydraulics.hydraulic_geometry(make_section(xs.elevations), wse=BED).area == pytest.approx(1.0)


def test_the_ground_runs_straight_from_each_bank_top_to_the_first_ordinate_beyond() -> None:
    """With bank elevations the channel's banks are at the bank elevation, here above the stream cell's ground. The
    ground beside the channel doesn't dip back down to the stream cell's old elevation, as it would if it were
    interpolated between the stream cell and its neighbour: it rises from the bank top to the neighbour."""
    xs = make_section([BED + 6.0, BED + 4.0, BED + 2.0, BED, BED + 2.0, BED + 4.0, BED + 6.0], spacing=30.0)
    banks = find_banks(xs, target_width=10.0)

    carve_channel(xs, banks, 1.0, trapezoid_height=0.2, bank_elevation=BED + 1.8)

    np.testing.assert_allclose(profile_at(xs, [-30.0, -15.0, -5.0, -3.0, 0.0, 3.0, 5.0, 15.0, 30.0]),
                               [BED + 2.0, BED + 1.88, BED + 1.8, BED + 0.8, BED + 0.8, BED + 0.8, BED + 1.8,
                                BED + 1.88, BED + 2.0])
    assert np.all(np.diff(xs.profile.stations) >= 0.0)


def test_without_bank_elevations_the_profile_is_the_lower_of_the_ground_and_the_channel() -> None:
    """A pool below the channel's bed keeps its ground, and where the ground crosses the channel's side the profile
    turns there."""
    xs = make_section(np.full(25, BED))
    xs.elevations[4:7] = 97.0  # 6 to 8 m left of the stream cell, below the 98 m bed
    original = xs.elevations.copy()

    carve_channel(xs, BANKS_10_M_OUT, 2.0, trapezoid_height=0.2)

    offsets = np.linspace(-12.0, 12.0, 481)
    ground = np.interp(offsets, np.arange(-12.0, 13.0), original)
    channel = np.where(np.abs(offsets) <= 10.0, 98.0 + 2.0 * np.maximum(1.0 - (10.0 - np.abs(offsets)) / 4.0, 0.0),
                       np.inf)
    np.testing.assert_allclose(profile_at(xs, offsets), np.minimum(ground, channel), atol=1e-12)


@pytest.mark.parametrize("use_banks", [False, True])
def test_the_profile_goes_through_the_carved_ordinates(use_banks: bool) -> None:
    """On random sections, banks, depths and trapezoid heights: the profile is the carved ordinates' elevation at
    each ordinate, the ground outside the banks, and without bank elevations never above the ground."""
    rng = np.random.default_rng(10)
    for _ in range(300):
        spacing = float(rng.choice([1.0, 7.5, 30.0]))
        xs = random_section(rng, 2 * int(rng.integers(2, 20)) + 1, spacing)
        ground = xs.elevations.copy()
        center = ground.size // 2
        left, right = (float(rng.uniform(0.05, center) * spacing) for _ in range(2))
        if rng.random() < 0.3:
            left = float(rng.integers(1, center + 1) * spacing)  # a bank on an ordinate
        banks = Banks("test", left, right, np.nan, np.nan, False, True)
        reference = BED + float(rng.uniform(0.0, 2.0)) if use_banks else None

        changed = carve_channel(xs, banks, float(rng.uniform(0.05, 3.0)),
                                trapezoid_height=float(rng.uniform(0.0, 0.5)), bank_elevation=reference)

        offsets = (np.arange(ground.size) - center) * spacing
        outside = (offsets < -left - 1e-9 * spacing) | (offsets > right + 1e-9 * spacing)
        assert not changed[outside].any()
        stations = xs.profile.stations
        assert np.all(np.diff(stations) >= 0.0) and stations[xs.profile.center] == 0.0
        interior = ~outside & (np.abs(offsets + left) > 1e-9 * spacing) & (np.abs(offsets - right) > 1e-9 * spacing)
        np.testing.assert_allclose(profile_at(xs, offsets[interior]), xs.elevations[interior], atol=1e-9)
        np.testing.assert_allclose(profile_at(xs, offsets[outside]), ground[outside], atol=1e-9)
        if not use_banks:
            dense = np.linspace(offsets[0], offsets[-1], 2001)
            assert np.all(profile_at(xs, dense) <= np.interp(dense, offsets, ground) + 1e-9)


def test_the_profile_keeps_the_ordinates_roughness() -> None:
    """Each vertex takes the Manning's n of the segment between ordinates it falls in, its inner ordinate's."""
    xs = make_section(np.full(9, BED), spacing=10.0)
    xs.mannings_n[:] = [0.1, 0.09, 0.08, 0.07, 0.03, 0.06, 0.05, 0.04, 0.02]

    carve_channel(xs, Banks("test", 15.0, 4.0, BED, BED, False, True), 1.0, trapezoid_height=0.2,
                  bank_elevation=BED)

    stations, n = xs.profile.stations, xs.profile.mannings_n
    expected = {-40.0: 0.1, -30.0: 0.09, -20.0: 0.08, -15.0: 0.07, -11.2: 0.07, 0.0: 0.03, 0.2: 0.03, 4.0: 0.03,
                10.0: 0.06, 20.0: 0.05, 30.0: 0.04, 40.0: 0.02}
    assert {float(s): float(v) for s, v in zip(np.round(stations, 9), n)} == pytest.approx(expected)


# --- Rasters ------------------------------------------------------------------------------------------------------


def test_ordinate_cells_follow_the_sampled_cross_section() -> None:
    """A stream flowing south (pi / 2) has its cross section along its row, from west (on the left, looking
    upstream) to east. Flowing south-east, each ordinate steps a cell in both directions, from the south-west."""
    rows, cols = ordinate_cells(5, 5, np.pi / 2, 40.0, 10.0, 10.0)
    assert rows.tolist() == [5] * 5
    assert cols.tolist() == [3, 4, 5, 6, 7]

    rows, cols = ordinate_cells(5, 5, np.pi / 4, 40.0, 10.0, 10.0)
    assert list(zip(rows.tolist(), cols.tolist())) == [(6, 4), (5, 5), (4, 6)]


def test_land_cover_is_sampled_from_the_nearest_cells_and_nan_off_the_raster() -> None:
    land_cover = np.arange(25, dtype=np.uint8).reshape(5, 5)

    values = sample_land_cover(land_cover, 2, 2, np.pi / 2, 80.0, 10.0, 10.0)

    assert np.isnan(values[:2]).all() and np.isnan(values[-2:]).all()
    assert values[2:7].tolist() == [10.0, 11.0, 12.0, 13.0, 14.0]


def _burn_all(shape, burns_in_order) -> np.ndarray:
    raster = np.full(shape, np.nan, dtype=np.float32)
    counts = np.zeros(shape, dtype=np.int32)
    for rows, cols, elevations, changed in burns_in_order:
        burn_into_raster(raster, counts, np.array(rows), np.array(cols), np.array(elevations, dtype=np.float64),
                         np.array(changed))
    return raster


def test_overlapping_cross_sections_burn_their_mean_whatever_their_order() -> None:
    """Three cross sections of five ordinates, centred on (0, 2), (2, 2) and (4, 2), all crossing cell (1, 3) off their
    centres, at 1, 2 and 6 m. The ordinate that isn't changed and the one off the raster burn nothing."""
    burns = [([0, 0, 0, 1, 9], [0, 1, 2, 3, 0], [5.0, 5.0, 5.0, 1.0, 5.0], [False, True, True, True, True]),
             ([2, 2, 2, 1, 2], [0, 1, 2, 3, 4], [5.0, 5.0, 5.0, 2.0, 5.0], [True, True, True, True, True]),
             ([4, 4, 4, 1, 4], [0, 1, 2, 3, 4], [5.0, 5.0, 5.0, 6.0, 5.0], [True, True, True, True, True])]
    for order in itertools.permutations(burns):
        raster = _burn_all((5, 5), order)
        assert raster[1, 3] == 3.0  # (1 + 2 + 6) / 3
        assert np.isnan(raster[0, 0])  # never changed
        assert np.isfinite(raster).sum() == 1 + 2 + 4 + 4  # nothing off the raster


def test_a_cross_section_s_centre_cell_takes_its_own_bed() -> None:
    """Cell (1, 1) is the centre of one cross section, at 2 m, and crossed off their centres by two others, at 5 and
    8 m, before and after it: it keeps the centre's 2 m, which is the VDT database's Elev there. A cell at the centre
    of two cross sections takes the mean of their centres, and a centre burns even where its carve left it as it
    was, unless the carve changed nothing at all."""
    own = ([1, 1, 1], [0, 1, 2], [3.0, 2.0, 3.0], [True, True, True])
    before = ([1, 0, 0], [1, 1, 2], [5.0, 9.0, 9.0], [True, True, True])  # centred on (0, 1)
    after = ([2, 2, 1], [0, 1, 1], [9.0, 9.0, 8.0], [True, True, True])  # centred on (2, 1)
    for order in itertools.permutations([own, before, after]):
        assert _burn_all((3, 3), order)[1, 1] == 2.0

    second = ([0, 1, 2], [1, 1, 1], [6.0, 4.0, 6.0], [True, False, True])  # its centre unchanged, still at 4 m
    assert _burn_all((3, 3), [before, own, second])[1, 1] == 3.0
    nothing = ([0, 1, 2], [1, 1, 1], [6.0, 4.0, 6.0], [False, False, False])
    assert np.isnan(_burn_all((3, 3), [nothing])).all()

def filled_gaps(raster: np.ndarray) -> np.ndarray:
    """The gap fill the slow way: each gap with at least four of its eight neighbours in the raster as it was gets
    their mean, added up one at a time in float64, row by row (Python's sum() would compensate its rounding)."""
    filled = raster.copy()
    rows, cols = raster.shape
    for r in range(rows):
        for c in range(cols):
            if not np.isnan(raster[r, c]):
                continue
            around = [float(raster[rr, cc]) for rr in range(max(r - 1, 0), min(r + 2, rows))
                      for cc in range(max(c - 1, 0), min(c + 2, cols)) if not np.isnan(raster[rr, cc])]
            if len(around) >= 4:
                total = 0.0
                for value in around:
                    total += value
                filled[r, c] = total / len(around)
    return filled

def test_burning_cross_sections_all_at_once_is_burning_them_one_by_one() -> None:
    """Burns, which gathers the carved cross sections and burns each cell's ordinates in their order, gives the raster
    burn_into_raster gives, bit for bit: with cross sections crossing, centres shared, ordinates off the raster, and
    cross sections that changed nothing."""
    from arc.bathymetry import Burns
    rng = np.random.default_rng(64)
    for _ in range(300):
        shape = (int(rng.integers(1, 15)), int(rng.integers(1, 15)))
        raster = np.full(shape, np.nan, dtype=np.float32)
        counts = np.zeros(shape, dtype=np.int32)
        burns = Burns(shape)
        for _ in range(int(rng.integers(0, 12))):
            size = 2 * int(rng.integers(0, 6)) + 1
            rows = rng.integers(-2, shape[0] + 2, size)
            cols = rng.integers(-2, shape[1] + 2, size)
            elevations = rng.normal(95.0, 2.0, size)
            changed = rng.random(size) < rng.choice([0.0, 0.5, 1.0])
            burn_into_raster(raster, counts, rows, cols, elevations, changed)
            burns.add(rows, cols, elevations, changed)
        burned = burns.burned().to_array()
        assert np.array_equal(burned.view(np.uint32), raster.view(np.uint32))

