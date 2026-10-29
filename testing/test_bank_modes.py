"""What a bank is in each mode (arc.bathymetry.find_banks' `at`, arc.pipeline.banks_at): with bathymetry, the DEM's
water's edge without Bathy_Use_Banks and the bank tops with it, below the maximum flow's water; without bathymetry, the
bank tops below the maximum flow's water too, or its edges where the channel holds it."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from arc.bathymetry import find_banks
from arc.bathymetry.banks import FLAT_WATER_DEPTH, MASK_REACH, _water_level
from arc.pipeline import banks_at
from arc.xsection.xsection import XSection

SPACING = 10.0
WATER = 100.0  # the DEM's water surface
WALL = 9999.0


def channel(water_cells: int = 2, bank: float = 3.0, edge: float = 0.6, floodplain_slope: float = 0.005,
            half: int = 40, burn: float = 0.0, beyond: np.ndarray | None = None) -> np.ndarray:
    """A cross section of cells: the stream cell and `water_cells` either side at the water surface, then a cell half
    water and half bank `edge` above it, a bank face up to `bank` above the water, and a floodplain beyond."""
    z = np.empty(2 * half + 1)
    for j in range(z.size):
        d = abs(j - half)
        if d <= water_cells:
            z[j] = WATER
        elif d == water_cells + 1:
            z[j] = WATER + edge
        elif d == water_cells + 2:
            z[j] = WATER + bank * 0.7
        else:
            z[j] = WATER + bank + floodplain_slope * SPACING * (d - water_cells - 3)
    z[half] -= burn
    if beyond is not None:
        z = np.where(np.arange(z.size) - half >= water_cells + 3, beyond[: z.size], z)
    return z


def xs(z: np.ndarray) -> XSection:
    return XSection(np.asarray(z, dtype=np.float64), np.full(z.size, 0.035), SPACING)


def mask_of(z: np.ndarray, left: int, right: int) -> np.ndarray:
    """A water mask of the stream cell and `left` and `right` cells either side."""
    half = z.size // 2
    m = np.zeros(z.size)
    m[half - left:half + right + 1] = 1.0
    return m


# --- The water level --------------------------------------------------------------------------------------------


def test_the_water_level_is_the_stream_cell_s_without_a_mask() -> None:
    z = channel(burn=0.5)
    assert _water_level(z, np.empty(0), np.nan) == pytest.approx(WATER - 0.5)


def test_beside_a_burned_stream_cell_the_water_level_is_the_mask_s_flat_water() -> None:
    z = channel(water_cells=3, burn=0.6)
    assert _water_level(z, mask_of(z, 3, 3), 1.0) == pytest.approx(WATER)


@pytest.mark.parametrize("extra", [1, 2, 8])
def test_a_mask_that_runs_onto_the_bank_or_the_floodplain_doesn_t_raise_the_water_level(extra: int) -> None:
    """The edge cell, half bank, and the floodplain beyond make no flat run with the water."""
    z = channel(water_cells=1)
    assert _water_level(z, mask_of(z, 1 + extra, 1 + extra), 1.0) == pytest.approx(WATER)


# --- Without Bathy_Use_Banks: the water's edge -------------------------------------------------------------------


@pytest.mark.parametrize("water_cells", [1, 2, 4])
def test_the_banks_are_half_way_between_the_last_water_cell_and_the_first_land_cell(water_cells: int) -> None:
    z = channel(water_cells=water_cells)
    banks = find_banks(xs(z), at="water")
    assert banks.method == "water_edge"
    assert (banks.left, banks.right) == pytest.approx(((water_cells + 0.5) * SPACING,) * 2)


@pytest.mark.parametrize("mask", ["exact", "wide 2", "spills 8", "narrow 1"])
def test_the_banks_never_take_in_land_whatever_the_mask(mask: str) -> None:
    """The edge cell, half water and half bank, and the bank face stay outside the channel: no bank tops."""
    z = channel(water_cells=3)
    m = {"exact": (3, 3), "wide 2": (5, 5), "spills 8": (3, 11), "narrow 1": (2, 2)}[mask]
    banks = find_banks(xs(z), land_cover=mask_of(z, *m), water_value=1.0, at="water")
    assert (banks.left, banks.right) == pytest.approx((3.5 * SPACING, 3.5 * SPACING))


def test_the_mask_stops_flat_water_running_across_a_floodplain_at_its_level() -> None:
    """Past the mask's edge, the DEM's water may run MASK_REACH cells; farther, the mask's edge holds it."""
    z = channel(water_cells=3, bank=0.05, edge=0.02)  # banks no higher than the water's noise: a flat floodplain
    m = mask_of(z, 3, 3)
    banks = find_banks(xs(z), land_cover=m, water_value=1.0, at="water")
    assert (banks.left, banks.right) == pytest.approx((3.5 * SPACING, 3.5 * SPACING))
    assert MASK_REACH == 2


def test_beside_a_burned_stream_cell_the_banks_are_the_water_s_edge() -> None:
    z = channel(water_cells=3, burn=0.6)
    without = find_banks(xs(z), at="water")
    banks = find_banks(xs(z), land_cover=mask_of(z, 3, 3), water_value=1.0, at="water")
    assert (banks.left, banks.right) == pytest.approx((3.5 * SPACING, 3.5 * SPACING))
    assert not without.valid  # 0.1 m above the burned stream cell is inside the burn: no water to find


def test_water_a_cell_wide_is_no_channel_and_a_prior_makes_it_a_single_cell() -> None:
    z = channel(water_cells=0)
    assert not find_banks(xs(z), at="water").valid
    banks = find_banks(xs(z), at="water", target_width=6.0)
    assert banks.single_cell and banks.top_width == pytest.approx(6.0)


# --- With Bathy_Use_Banks: the bank tops ----------------------------------------------------------------------------


@pytest.mark.parametrize("mask", [None, "exact", "wide 2", "spills 8", "narrow 1"])
def test_the_banks_are_the_bank_tops_mask_or_not(mask: str | None) -> None:
    z = channel(water_cells=3)
    m = None if mask is None else mask_of(z, *{"exact": (3, 3), "wide 2": (5, 5), "spills 8": (3, 11),
                                                "narrow 1": (2, 2)}[mask])
    banks = find_banks(xs(z), land_cover=m, water_value=None if m is None else 1.0, at="tops")
    assert banks.method == "width_to_depth_ratio"
    assert min(banks.left_elevation, banks.right_elevation) == pytest.approx(WATER + 3.0, abs=0.15)


def test_the_plain_search_takes_the_mask_first() -> None:
    """find_banks' default (at="any"), the search ARC ran without bathymetry before 0.5.0: the mask's edge first."""
    z = channel(water_cells=3)
    banks = find_banks(xs(z), land_cover=mask_of(z, 3, 3), water_value=1.0)
    assert banks.method == "land_cover" and banks.top_width == pytest.approx(7 * SPACING)


@pytest.mark.parametrize(("bathymetry", "use_banks", "expected"), [
    (False, False, "divide"), (False, True, "divide"), (True, False, "water"), (True, True, "tops")])
def test_without_bathymetry_the_banks_divide_the_channel_whatever_bathy_use_banks_says(bathymetry: bool,
                                                                                     use_banks: bool,
                                                                                     expected: str) -> None:
    """Without bathymetry the banks divide the channel from its overbanks, which the divided-channel method does at the
    bank tops; Bathy_Use_Banks is the bathymetry's."""
    configs = SimpleNamespace(bathy_out_file="bathy.tif" if bathymetry else None, bathy_use_banks=use_banks)
    assert banks_at(configs) == expected


# --- A DEM that shows the bed (no bathymetry) ----------------------------------------------------------------------

BED_SPACING = 2.0


def bed_channel(half: int = 60) -> np.ndarray:
    """A DEM showing a channel's bed, as lidar with bathymetry does: 20 m wide at the bed, sides 4:1 up to banks 3 m
    high, 22 m out, and a floodplain beyond rising 0.005. Its water at a low flow, 0.75 m deep, is 26 m wide."""
    d = np.abs(np.arange(2 * half + 1) - half) * BED_SPACING
    return WATER + np.where(d <= 22.0, np.clip((d - 10.0) / 4.0, 0.0, None), 3.0 + 0.005 * (d - 22.0))


@pytest.mark.parametrize("mask", [None, "exact", "narrow", "wide", "spills"])
def test_on_a_dem_showing_the_bed_the_banks_are_the_tops_whatever_the_mask(mask: str | None) -> None:
    """The mask is the low flow's water (its edge 13 m out) or wrong; the banks are the bank tops, 22 m out, all the
    same. Taking the mask's edge first would leave the bank faces above the low water in the overbanks."""
    z = bed_channel()
    sides = {"exact": (6, 6), "narrow": (3, 3), "wide": (9, 9), "spills": (6, 20)}
    m = None if mask is None else mask_of(z, *sides[mask])
    section = XSection(z, np.full(z.size, 0.035), BED_SPACING)
    kwargs = {"land_cover": m, "water_value": None if m is None else 1.0}

    banks = find_banks(section, at="divide", **kwargs)

    assert banks.method == "width_to_depth_ratio"
    assert (banks.left, banks.right) == pytest.approx((22.0, 22.0))
    if mask == "exact":
        assert find_banks(section, **kwargs).top_width == pytest.approx(26.0)  # the plain search: the mask's edge


def test_where_the_channel_holds_the_maximum_flow_its_water_s_edges_are_the_banks() -> None:
    """The maximum flow stands 1.7 m deep in the 3 m channel, so no bank is above it in either mode: one there would be
    a valley wall for all the DEM can tell. With Bathy_Use_Banks the banks are the highest stage's below it, 1.5 m up
    on the ground (16 m out); without bathymetry its own edges, 16.8 m out, so that the hydraulics don't divide it.
    Neither is a bank top (max_flow_water)."""
    z = bed_channel()
    section = XSection(z, np.full(z.size, 0.035), BED_SPACING)

    divided = find_banks(section, at="divide", ceiling=1.7)
    capped = find_banks(section, at="tops", ceiling=1.7)

    assert divided.method == capped.method == "max_flow_water"
    assert (divided.left, divided.right) == pytest.approx((16.8, 16.8))
    assert divided.left_elevation == pytest.approx(WATER + 1.7)
    assert (capped.left, capped.right) == pytest.approx((16.0, 16.0), abs=0.01)


def test_where_the_ratio_never_stops_falling_the_banks_are_the_maximum_flow_s_water_s_edges() -> None:
    """A channel without a floodplain, its sides rising ever more steeply, has no bank tops: without bathymetry the
    banks are the edges of the maximum flow's water, which it holds, and not the edges of the water 0.1 m up, which
    would divide the channel at its bed. Without a maximum flow, those."""
    d = np.abs(np.arange(121) - 60) * BED_SPACING
    z = WATER + np.where(d <= 10.0, 0.0, 0.002 * (d - 10.0) ** 2)
    section = XSection(z, np.full(z.size, 0.035), BED_SPACING)

    banks = find_banks(section, at="divide", ceiling=2.0)
    without = find_banks(section, at="divide")

    assert banks.method == "max_flow_water"
    assert banks.left_elevation == pytest.approx(WATER + 2.0) and banks.right_elevation == pytest.approx(WATER + 2.0)
    assert without.method == "water_edge" and without.top_width < 0.5 * banks.top_width


def test_no_bank_top_above_the_water_that_carries_the_maximum_flow() -> None:
    """A valley wall beyond a floodplain the maximum flow doesn't fill isn't a bank: under a ceiling below the wall,
    the banks stay at the channel's tops."""
    half = 40
    beyond = np.array([WATER + 3.0 + (8.0 if abs(j - half) > 12 else 0.0) for j in range(2 * half + 1)])
    z = channel(water_cells=3, bank=3.0, floodplain_slope=0.0, beyond=beyond)
    capped = find_banks(xs(z), at="tops", ceiling=4.0)
    assert min(capped.left_elevation, capped.right_elevation) == pytest.approx(WATER + 3.0, abs=0.15)


def test_under_a_ceiling_a_bank_is_on_the_ground_however_steeply_it_rises_above_it() -> None:
    """Water against a face rising 2 m a cell on the right, and a slope rising 0.3 m a cell on the left, below a
    ceiling 1 m up: the ratio is still falling at the highest stage below it, 0.9 m up. That stage's right edge is
    0.45 of the way up the face, 14.5 m out and 0.9 m up, not 10 m out at the water's level, where it was when the
    ground above the ceiling was made a wall."""
    z = np.full(81, WATER)
    offsets = np.arange(81) - 40
    z[offsets < 0] = WATER + 0.3 * -offsets[offsets < 0]
    z[offsets >= 2] = WATER + 2.0 * (offsets[offsets >= 2] - 1)

    banks = find_banks(xs(z), at="tops", ceiling=1.0)

    assert banks.method == "max_flow_water"
    assert (banks.left, banks.right) == pytest.approx((30.0, 14.5))
    assert banks.left_elevation == pytest.approx(WATER + 0.9) and banks.right_elevation == pytest.approx(WATER + 0.9)


def test_the_bank_tops_are_never_inside_the_water() -> None:
    """Noise on the DEM's water can stop width-to-depth inside it; the search starts above the water level."""
    z = channel(water_cells=6)
    z[z.size // 2 - 2] += 0.08
    z[z.size // 2 + 3] += 0.07
    banks = find_banks(xs(z), land_cover=mask_of(z, 6, 6), water_value=1.0, at="tops")
    assert banks.top_width >= 13 * SPACING


def test_under_a_ceiling_the_bank_tops_are_no_higher_than_the_highest_stage_below_it() -> None:
    """Steep valley sides rising from the water have no stage below the ceiling where the water spills: the banks are
    the water's edges at the highest stage below it, not the valley's sides above it."""
    z = channel(water_cells=3, bank=0.0, edge=0.3, floodplain_slope=0.3)
    banks = find_banks(xs(z), land_cover=mask_of(z, 3, 3), water_value=1.0, at="tops", ceiling=0.5)
    assert max(banks.left_elevation, banks.right_elevation) <= WATER + 0.5
    assert banks.top_width <= 9 * SPACING


def test_without_any_stage_above_the_water_the_banks_are_the_water_s_edge() -> None:
    """Water running to walls (a cross section cut short by the raster's edge beyond its water) has no stage to try."""
    z = channel(water_cells=3)
    half = z.size // 2
    z[: half - 4] = WALL
    z[half + 5:] = WALL
    banks = find_banks(xs(z), land_cover=mask_of(z, 3, 3), water_value=1.0, at="tops")
    assert banks.method in ("water_edge", "land_cover", "width_to_depth_ratio")
    assert banks.top_width <= 8 * SPACING


def test_an_unknown_at_is_refused() -> None:
    with pytest.raises(ValueError, match="at must be one of"):
        find_banks(xs(channel()), at="bankfull")


def test_flat_water_depth_is_the_water_s_tolerance() -> None:
    assert FLAT_WATER_DEPTH == pytest.approx(0.1)
