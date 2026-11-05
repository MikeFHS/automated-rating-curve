from __future__ import annotations

import math

import numpy as np
import pytest

from arc import hydraulics
from arc.hydraulics import (
    CompoundGeometry,
    DepthRoughness,
    HydraulicGeometry,
    compound_geometry,
    discharge,
    hydraulic_geometry,
    wse_for_discharge,
)
from arc.xsection.sampling import sample_cross_section
from arc.xsection.xsection import XSection

BED = 100.0
N = 0.035
SLOPE = 0.001
DRY = HydraulicGeometry(0.0, 0.0, 0.0, 0.0, 0.0)


def make_section(elevations, mannings_n=N, spacing=1.0) -> XSection:
    elevations = np.asarray(elevations, dtype=np.float64)
    mannings_n = np.broadcast_to(np.asarray(mannings_n, dtype=np.float64), elevations.shape).copy()
    return XSection(elevations, mannings_n, spacing)


def trapezoid(bottom_width: float, side_slope: float, ordinates_per_side: int = 60, spacing: float = 1.0) -> XSection:
    """A flat bed at 100 m, with banks rising 1 m for every side_slope metres across. A bottom width of 0 is a V."""
    x = spacing * np.arange(-ordinates_per_side, ordinates_per_side + 1)
    return make_section(BED + np.maximum(np.abs(x) - bottom_width / 2, 0.0) / side_slope, spacing=spacing)


def ridge_and_hollow() -> XSection:
    """A V channel whose right bank tops out in a ridge 3 m up, with a hollow behind the ridge lower than the bed."""
    left = BED + np.arange(10.0, 0.0, -1.0)
    right = [100.0, 101.0, 102.0, 103.0, 99.0, 99.0, 99.0, 104.0, 105.0, 106.0, 107.0]
    return make_section(np.concatenate([left, right]))


def flat_floodplain() -> XSection:
    """A 2 m deep V channel between perfectly flat floodplains 50 m wide, then valley walls."""
    distance = np.abs(np.arange(-60.0, 61.0))
    elevations = np.where(distance <= 2, BED + distance, BED + 2.0)
    return make_section(np.where(distance > 52, BED + 2.0 + (distance - 52), elevations))


def walled_bed() -> XSection:
    """A 20 m wide flat bed with 9999 m walls on both sides, like a section that runs off the raster."""
    elevations = np.full(25, BED)
    mannings_n = np.full(25, N)
    elevations[[0, 1, -2, -1]] = 9999.0
    mannings_n[[0, 1, -2, -1]] = 9999.0  # sampling gives off-raster ordinates a roughness of 9999 too
    return make_section(elevations, mannings_n)


def rough_valley() -> XSection:
    """A valley whose banks steepen and flatten but never dip, with a different roughness at every ordinate."""
    rng = np.random.default_rng(4)
    right = BED + np.concatenate([[0.0], np.cumsum(rng.uniform(0.01, 0.8, 80))])
    left = BED + np.concatenate([[0.0], np.cumsum(rng.uniform(0.01, 0.8, 80))])
    return make_section(np.concatenate([left[:0:-1], right]), rng.uniform(0.02, 0.12, 161), spacing=2.0)


SHAPES = {
    "v-channel": lambda: trapezoid(0.0, 1.5),
    "trapezoid": lambda: trapezoid(10.0, 2.0),
    "ridge-and-hollow": ridge_and_hollow,
    "flat-floodplain": flat_floodplain,
    "walled-bed": walled_bed,
    "rough-valley": rough_valley,
}


def with_banks(xs: XSection, left: float, right: float) -> XSection:
    xs.left_bank_distance = left
    xs.right_bank_distance = right
    return xs


def channel_and_floodplains() -> XSection:
    """A 20 m wide bed, with banks rising 2 m over 2 m to flat floodplains 18 m wide on each side. The channel's
    banks are 12 m either side of the stream cell."""
    distance = np.abs(np.arange(-30.0, 31.0))
    return with_banks(make_section(BED + np.clip(distance - 10.0, 0.0, 2.0)), 12.0, 12.0)


def refine(xs: XSection, factor: int) -> XSection:
    """The same ground with `factor` times as many ordinates. Each new ordinate takes the Manning's n of the
    original ordinate nearer the stream cell, so every segment keeps its roughness."""
    size = xs.elevations.size
    position = np.arange((size - 1) * factor + 1) / factor
    nearer = np.where(position >= size // 2, np.floor(position), np.ceil(position)).astype(int)
    return XSection(np.interp(position, np.arange(size), xs.elevations), xs.mannings_n[nearer],
                    xs.ordinate_distance / factor)


def manning_conveyance(area: float, perimeter: float, mannings_n: float = N) -> float:
    return area * (area / perimeter) ** (2 / 3) / mannings_n


# Pairs of bank distances: on ordinates, between them, at the stream cell, and beyond the ends of the section
BANKS = [(3.0, 5.0), (2.5, 7.25), (0.0, 4.0), (0.0, 0.0), (6.0, 1e6)]


# --- Geometry ----------------------------------------------------------------------------------------------


@pytest.mark.parametrize("side_slope", [1.0, 2.5])
@pytest.mark.parametrize("depth", [0.25, 1.0, 2.37])
def test_v_channel_matches_the_triangle_formulas(side_slope: float, depth: float) -> None:
    geometry = hydraulic_geometry(trapezoid(0.0, side_slope), depth=depth)

    area = side_slope * depth**2
    perimeter = 2 * depth * math.hypot(1, side_slope)
    assert geometry.area == pytest.approx(area, rel=1e-12)
    assert geometry.top_width == pytest.approx(2 * side_slope * depth, rel=1e-12)
    assert geometry.wetted_perimeter == pytest.approx(perimeter, rel=1e-12)
    assert geometry.hydraulic_radius == pytest.approx(area / perimeter, rel=1e-12)


@pytest.mark.parametrize("depth", [0.5, 1.3, 3.0])
def test_trapezoid_matches_the_trapezoid_formulas(depth: float) -> None:
    """A 10 m wide bed, with banks rising 1 m for every 2 m across."""
    geometry = hydraulic_geometry(trapezoid(10.0, 2.0), depth=depth)

    area = (10 + 2 * depth) * depth
    perimeter = 10 + 2 * depth * math.hypot(1, 2)
    assert geometry.area == pytest.approx(area, rel=1e-12)
    assert geometry.top_width == pytest.approx(10 + 4 * depth, rel=1e-12)
    assert geometry.wetted_perimeter == pytest.approx(perimeter, rel=1e-12)
    assert geometry.hydraulic_radius == pytest.approx(area / perimeter, rel=1e-12)


def test_banks_with_different_slopes() -> None:
    """The left bank rises 1 m for every 1 m across, and the right bank 1 m for every 3 m."""
    x = np.arange(-30.0, 31.0)
    geometry = hydraulic_geometry(make_section(BED + np.where(x < 0, -x, x / 3)), depth=2.0)

    assert geometry.area == pytest.approx((1 + 3) * 2.0**2 / 2, rel=1e-12)
    assert geometry.top_width == pytest.approx((1 + 3) * 2.0, rel=1e-12)
    assert geometry.wetted_perimeter == pytest.approx(2.0 * (math.hypot(1, 1) + math.hypot(1, 3)), rel=1e-12)


def test_depth_is_measured_from_the_stream_cell() -> None:
    xs = trapezoid(10.0, 2.0)

    assert hydraulic_geometry(xs, depth=1.5) == hydraulic_geometry(xs, wse=101.5)
    assert discharge(xs, SLOPE, depth=1.5) == discharge(xs, SLOPE, wse=101.5)


@pytest.mark.parametrize("depth", [0.0, -0.5])
def test_dry_when_the_water_surface_is_not_above_the_stream_cell(depth: float) -> None:
    xs = trapezoid(10.0, 2.0)

    assert hydraulic_geometry(xs, depth=depth) == DRY
    assert discharge(xs, SLOPE, depth=depth) == 0.0


def test_water_spreads_out_from_the_stream_cell() -> None:
    """The stream cell sits on a bump 0.5 m above the ground either side of it, which only floods once the
    water reaches the stream cell."""
    distance = np.abs(np.arange(-10.0, 11.0))
    elevations = BED + 0.5 * distance
    elevations[10] = 101.0
    xs = make_section(elevations)

    assert hydraulic_geometry(xs, wse=100.8) == DRY
    # At 101.5 m, each side has two full segments (depths 0.5 to 1.0, then 1.0 to 0.5) and one that the
    # water just reaches the end of (0.5 to 0)
    geometry = hydraulic_geometry(xs, wse=101.5)
    assert geometry.area == pytest.approx(2 * (0.75 + 0.75 + 0.25), rel=1e-12)
    assert geometry.top_width == pytest.approx(6.0, rel=1e-12)
    # A depth is measured from the stream cell, not from the lowest ground
    assert hydraulic_geometry(xs, depth=0.5) == geometry


def test_low_ground_behind_a_ridge_stays_dry_until_the_ridge_is_overtopped() -> None:
    xs = ridge_and_hollow()

    # Below the ridge the section is just the V channel, 2.5 m deep
    assert hydraulic_geometry(xs, wse=102.5).area == pytest.approx(2.5**2, rel=1e-12)
    # Above it, the hollow (bed 99 m) fills too. Left: 3.5**2 / 2. Right: 6 in the channel up to the ridge,
    # then 2.5, 4.5 and 4.5 over the hollow, and 2.025 where the water meets the far side (0.9 m across)
    assert hydraulic_geometry(xs, wse=103.5).area == pytest.approx(3.5**2 / 2 + 6 + 2.5 + 4.5 + 4.5 + 2.025, rel=1e-12)


def test_the_ends_of_the_cross_section_act_as_walls() -> None:
    """Water above all of a 20 m wide flat section is held in by vertical walls at its ends."""
    geometry = hydraulic_geometry(make_section(np.full(21, BED)), depth=2.0)

    assert geometry.area == pytest.approx(40.0, rel=1e-12)
    assert geometry.top_width == pytest.approx(20.0, rel=1e-12)
    assert geometry.wetted_perimeter == pytest.approx(20.0 + 2 * 2.0, rel=1e-12)


def test_off_raster_walls_hold_the_water_in() -> None:
    """A 200 m section sampled from an 11 x 11 raster of 10 m cells runs 50 m off each edge, into 9999 m walls."""
    dem = np.full((11, 11), BED)
    xs = sample_cross_section(dem, np.full_like(dem, N), 5, 5, np.pi / 2, 200.0, 10.0, 10.0)

    geometry = hydraulic_geometry(xs, depth=1.0)

    assert geometry.top_width == pytest.approx(100.0, abs=0.01)
    assert geometry.area == pytest.approx(100.0, abs=0.01)
    assert geometry.wetted_perimeter == pytest.approx(100.0 + 2 * 1.0, abs=0.01)
    assert geometry.mannings_n == pytest.approx(N, rel=1e-12)  # the walls' 9999 roughness doesn't count


@pytest.mark.parametrize("shape", SHAPES)
def test_top_widths_are_each_side_s_share_of_the_top_width(shape: str) -> None:
    """The distances from the stream cell to the water's edges, which add up to the top width. A bank at the stream
    cell puts its whole side in that side's overbank."""
    xs = SHAPES[shape]()
    ground = xs.elevations[xs.elevations < 9000]
    center = xs.elevations.size // 2
    left_side = with_banks(SHAPES[shape](), 0.0, 1e6)
    right_side = with_banks(SHAPES[shape](), 1e6, 0.0)

    for wse in np.concatenate([np.linspace(ground.min() - 1, ground.max() + 3, 200), ground]):
        left, right = hydraulics.top_widths(xs.elevations, xs.ordinate_distance, wse)
        if not wse > xs.elevations[center]:
            assert (left, right) == (0.0, 0.0)
            continue
        assert left == pytest.approx(compound_geometry(left_side, wse=wse).left_overbank.top_width, rel=1e-12)
        assert right == pytest.approx(compound_geometry(right_side, wse=wse).right_overbank.top_width, rel=1e-12)
        assert left + right == pytest.approx(hydraulic_geometry(xs, wse=wse).top_width, rel=1e-12)


def test_mirroring_the_section_changes_nothing() -> None:
    xs = rough_valley()
    mirrored = XSection(xs.elevations[::-1].copy(), xs.mannings_n[::-1].copy(), xs.ordinate_distance)

    for wse in np.linspace(100.0, 130.0, 61):
        assert hydraulic_geometry(mirrored, wse=wse) == pytest.approx(hydraulic_geometry(xs, wse=wse), rel=1e-12)


def test_float32_sections_give_the_same_answers() -> None:
    xs = trapezoid(10.0, 2.0)
    xs32 = XSection(xs.elevations.astype(np.float32), xs.mannings_n.astype(np.float32), xs.ordinate_distance)

    assert hydraulic_geometry(xs32, depth=1.3) == pytest.approx(hydraulic_geometry(xs, depth=1.3), rel=1e-6)
    assert discharge(xs32, SLOPE, depth=1.3) == pytest.approx(discharge(xs, SLOPE, depth=1.3), rel=1e-6)


def test_a_single_ordinate_section_never_carries_water() -> None:
    xs = make_section([BED])

    assert hydraulic_geometry(xs, depth=1.0).area == 0.0
    assert math.isnan(wse_for_discharge(xs, 1.0, SLOPE))


def test_needs_exactly_one_of_wse_and_depth() -> None:
    xs = trapezoid(10.0, 2.0)

    with pytest.raises(TypeError, match="exactly one of wse or depth"):
        hydraulic_geometry(xs)
    with pytest.raises(TypeError, match="exactly one of wse or depth"):
        discharge(xs, SLOPE, wse=101.0, depth=1.0)


@pytest.mark.parametrize("slope", [0.0, -0.001, math.nan])
def test_slope_must_be_positive(slope: float) -> None:
    with pytest.raises(ValueError, match="slope must be positive"):
        discharge(trapezoid(10.0, 2.0), slope, depth=1.0)


# --- Roughness and discharge -------------------------------------------------------------------------------


def test_a_uniform_bed_s_roughness_is_the_composite_roughness() -> None:
    xs = trapezoid(10.0, 2.0)
    xs.mannings_n[:] = 0.042

    assert hydraulic_geometry(xs, depth=1.7).mannings_n == pytest.approx(0.042, rel=1e-12)


def test_composite_roughness_weights_each_segment_by_its_wetted_length() -> None:
    """A V with 45 degree banks, with n = 0.02 up to and including the stream cell and 0.05 to its right.

    Each segment takes the n of its end nearer the stream cell. So at 2 m deep, the two left segments and
    the first right segment are 0.02, the second right segment is 0.05, and all four are sqrt(2) m long.
    """
    xs = trapezoid(0.0, 1.0, ordinates_per_side=10)
    xs.mannings_n[:] = np.where(np.arange(21) <= 10, 0.02, 0.05)

    expected = ((3 * 0.02**1.5 + 0.05**1.5) / 4) ** (2 / 3)
    assert hydraulic_geometry(xs, depth=2.0).mannings_n == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize("depth", [0.5, 1.3, 3.0])
def test_discharge_matches_mannings_equation(depth: float) -> None:
    area = (10 + 2 * depth) * depth
    perimeter = 10 + 2 * depth * math.hypot(1, 2)
    expected = area * (area / perimeter) ** (2 / 3) * math.sqrt(SLOPE) / N

    assert discharge(trapezoid(10.0, 2.0), SLOPE, depth=depth) == pytest.approx(expected, rel=1e-12)


def test_discharge_scales_with_the_square_root_of_slope() -> None:
    xs = trapezoid(10.0, 2.0)

    assert discharge(xs, 4 * SLOPE, depth=1.0) == pytest.approx(2 * discharge(xs, SLOPE, depth=1.0), rel=1e-12)


def test_discharge_rises_with_the_water_level_between_straight_banks() -> None:
    """Including once the water is above the banks and held in by the walls at the ends (127.5 m).

    With uneven banks and roughness this needn't hold: in rough_valley, discharge falls slightly as the water
    spreads over flatter or rougher ground, as it does onto flat_floodplain.
    """
    flows = [discharge(trapezoid(10.0, 2.0), SLOPE, wse=wse) for wse in np.linspace(BED, 160.0, 2001)]

    assert np.all(np.diff(flows) > 0)


# --- The water surface elevation for a discharge ----------------------------------------------------------------


def bisected_wse(xs: XSection, q: float, steps: int = 60) -> float:
    """The water surface elevation carrying q, by bisection within 25 m of the stream cell, for a section whose
    discharge always rises with the water level."""
    lo = float(xs.elevations[xs.elevations.size // 2])
    hi = lo + 25.0
    for _ in range(steps):
        mid = 0.5 * (lo + hi)
        if discharge(xs, SLOPE, wse=mid) < q:
            lo = mid
        else:
            hi = mid
    return hi


@pytest.mark.parametrize("shape", ["v-channel", "trapezoid", "flat-floodplain", "walled-bed", "rough-valley"])
def test_wse_for_discharge_carries_that_discharge(shape: str) -> None:
    xs = SHAPES[shape]()

    for q in np.geomspace(0.01, 5000.0, 60):
        wse = wse_for_discharge(xs, q, SLOPE)
        assert discharge(xs, SLOPE, wse=wse) == pytest.approx(q, rel=1e-6)  # the solve stops within 1e-10 m


def test_wse_for_discharge_agrees_with_bisection() -> None:
    """On a section whose discharge always rises with the water level, so that bisection has one answer."""
    xs = trapezoid(10.0, 2.0)

    for q in [0.5, 5.0, 50.0, 500.0]:
        assert wse_for_discharge(xs, q, SLOPE) == pytest.approx(bisected_wse(xs, q), abs=1e-9)


def test_lowest_wse_is_returned_when_discharge_falls_as_water_spreads() -> None:
    """Water spilling from the channel onto a flat floodplain carries less than the full channel did, so
    discharge falls and then rises again, and two water levels carry the same discharge."""
    xs = flat_floodplain()
    bankfull = discharge(xs, SLOPE, wse=102.0)
    q = 0.9 * bankfull
    assert discharge(xs, SLOPE, wse=102.01) < q

    wse = wse_for_discharge(xs, q, SLOPE)

    assert wse < 102.0
    assert discharge(xs, SLOPE, wse=wse) == pytest.approx(q, rel=1e-9)
    assert max(discharge(xs, SLOPE, wse=level) for level in np.linspace(BED, wse, 2000, endpoint=False)) < q


def test_wse_sits_on_the_ridge_when_overtopping_it_jumps_past_the_discharge() -> None:
    xs = ridge_and_hollow()
    below = discharge(xs, SLOPE, wse=103.0)  # the hollow is still dry
    above = discharge(xs, SLOPE, wse=103.0 + 1e-9)  # the hollow has filled
    assert above > 2 * below

    assert wse_for_discharge(xs, (below + above) / 2, SLOPE) == 103.0


def test_discharge_above_the_highest_ground_is_held_between_the_walls() -> None:
    xs = trapezoid(10.0, 2.0, ordinates_per_side=20)  # the banks top out at 107.5 m
    q = 10 * discharge(xs, SLOPE, wse=107.5)

    wse = wse_for_discharge(xs, q, SLOPE)

    assert wse > 107.5
    assert discharge(xs, SLOPE, wse=wse) == pytest.approx(q, rel=1e-9)


def test_no_discharge_is_no_depth() -> None:
    xs = trapezoid(10.0, 2.0)

    assert wse_for_discharge(xs, 0.0, SLOPE) == BED


# --- Cross sections divided at the banks -------------------------------------------------------------------


def test_the_banks_divide_the_water_into_the_channel_and_the_overbanks() -> None:
    """With the water 1 m over the floodplains, the channel holds its full 2 m deep trapezoid and 1 m of water
    across its 24 m top, and each overbank holds 1 m of water over 18 m of floodplain, against a 1 m wall."""
    parts = compound_geometry(channel_and_floodplains(), wse=103.0)

    channel_perimeter = 20 + 2 * math.hypot(2, 2)  # the lines at the banks add no wetted perimeter
    assert parts.channel == pytest.approx(HydraulicGeometry(68.0, channel_perimeter, 68.0 / channel_perimeter, 24.0, N),
                                          rel=1e-12)
    overbank = HydraulicGeometry(18.0, 18.0 + 1.0, 18.0 / 19.0, 18.0, N)
    assert parts.left_overbank == pytest.approx(overbank, rel=1e-12)
    assert parts.right_overbank == pytest.approx(overbank, rel=1e-12)


def test_discharge_with_banks_adds_up_the_subsections() -> None:
    xs = channel_and_floodplains()
    expected = (manning_conveyance(68.0, 20 + 2 * math.hypot(2, 2)) + 2 * manning_conveyance(18.0, 19.0)) * math.sqrt(SLOPE)

    assert discharge(xs, SLOPE, wse=103.0) == pytest.approx(expected, rel=1e-12)
    # Treating it all as one section, the floodplains' long wetted perimeter holds back the channel's water too
    whole = manning_conveyance(68.0 + 36.0, 20 + 2 * math.hypot(2, 2) + 38.0) * math.sqrt(SLOPE)
    assert expected > 1.1 * whole


def test_the_banks_change_the_discharge_but_not_the_geometry() -> None:
    xs = channel_and_floodplains()
    parts = compound_geometry(xs, wse=103.0)
    whole = hydraulic_geometry(xs, wse=103.0)

    assert sum(part.area for part in parts) == pytest.approx(whole.area, rel=1e-12)
    assert sum(part.wetted_perimeter for part in parts) == pytest.approx(whole.wetted_perimeter, rel=1e-12)
    assert sum(part.top_width for part in parts) == pytest.approx(whole.top_width, rel=1e-12)


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("banks", BANKS)
def test_the_subsections_add_up_to_the_whole_section(shape: str, banks: tuple[float, float]) -> None:
    xs = with_banks(SHAPES[shape](), *banks)
    ground = xs.elevations[xs.elevations < 9000]

    for wse in np.concatenate([np.linspace(ground.min(), ground.max() + 2, 80), ground]):
        parts = compound_geometry(xs, wse=wse)
        whole = hydraulic_geometry(xs, wse=wse)
        for field in ("area", "wetted_perimeter", "top_width"):
            assert sum(getattr(part, field) for part in parts) == pytest.approx(getattr(whole, field), rel=1e-12,
                                                                               abs=1e-12)
        assert min(min(part) for part in parts) >= 0.0


def test_without_banks_the_section_is_all_channel() -> None:
    xs = rough_valley()  # XSection's banks start at -1, not yet found
    assert xs.left_bank_distance < 0 and xs.right_bank_distance < 0

    for wse in [100.5, 104.0, 112.0]:
        parts = compound_geometry(xs, wse=wse)
        assert parts.channel == hydraulic_geometry(xs, wse=wse)
        assert parts.left_overbank == parts.right_overbank == DRY


@pytest.mark.parametrize("banks", [(-1.0, -1.0), (math.nan, math.nan), (-1.0, math.nan), (160.0, 1e9), (math.inf, 500.0)])
def test_banks_that_divide_nothing_leave_one_section(banks: tuple[float, float]) -> None:
    """Undefined banks (negative or NaN), and banks at or beyond the ends of the section (160 m here)."""
    xs = rough_valley()
    divided = with_banks(rough_valley(), *banks)

    for wse in np.linspace(100.0, 160.0, 61):
        assert discharge(divided, SLOPE, wse=wse) == pytest.approx(discharge(xs, SLOPE, wse=wse), rel=1e-12)


def test_water_inside_the_banks_is_all_channel() -> None:
    """At 101.5 m the water's edges are 11.5 m out, inside the banks at 12 m."""
    xs = channel_and_floodplains()

    parts = compound_geometry(xs, wse=101.5)

    assert parts.left_overbank == parts.right_overbank == DRY
    assert parts.channel == pytest.approx(hydraulic_geometry(xs, wse=101.5), rel=1e-12)
    assert discharge(xs, SLOPE, wse=101.5) == pytest.approx(discharge(with_banks(xs, -1.0, -1.0), SLOPE, wse=101.5),
                                                            rel=1e-12)


@pytest.mark.parametrize("banks", [(7.0, 13.0), (7.3, 12.9), (0.4, 0.0)])
def test_a_bank_between_ordinates_divides_its_segment(banks: tuple[float, float]) -> None:
    """The same ground and roughness sampled more finely, so that the banks can fall on or off ordinates, gives
    the same subsections. At 2 m spacing, banks 7 m and 13 m out are half way along a segment. At 1 m they are on
    ordinates."""
    xs = with_banks(rough_valley(), *banks)

    for factor in [2, 3]:
        fine = with_banks(refine(xs, factor), *banks)
        for wse in np.linspace(100.0, 135.0, 141):
            assert np.array(compound_geometry(fine, wse=wse)) == pytest.approx(
                np.array(compound_geometry(xs, wse=wse)), rel=1e-11, abs=1e-11)


def test_a_bank_at_the_stream_cell_puts_that_whole_side_in_its_overbank() -> None:
    xs = with_banks(trapezoid(10.0, 2.0), 0.0, 1e6)
    parts = compound_geometry(xs, wse=102.0)

    # The trapezoid is symmetric, so each side holds half the water
    half = hydraulic_geometry(trapezoid(10.0, 2.0), wse=102.0).area / 2
    assert parts.left_overbank.area == pytest.approx(half, rel=1e-12)
    assert parts.channel.area == pytest.approx(half, rel=1e-12)
    assert parts.right_overbank == DRY

    # With both banks there, the channel has no width, and the sides are separate subsections
    both = compound_geometry(with_banks(trapezoid(10.0, 2.0), 0.0, 0.0), wse=102.0)
    assert both.channel == DRY
    assert both.left_overbank == pytest.approx(both.right_overbank, rel=1e-12)


def test_the_walls_are_in_the_overbanks_or_in_the_channel_without_them() -> None:
    xs = channel_and_floodplains()
    wse = 104.0  # 2 m over the floodplains, against the walls at 30 m

    assert compound_geometry(xs, wse=wse).right_overbank.wetted_perimeter == pytest.approx(18.0 + 2.0, rel=1e-12)
    no_right_overbank = compound_geometry(with_banks(xs, 12.0, 30.0), wse=wse)  # the bank is on the wall
    assert no_right_overbank.right_overbank == DRY
    assert no_right_overbank.channel.wetted_perimeter == pytest.approx(
        10.0 + math.hypot(2, 2) + 18.0 + 2.0 + 10.0 + math.hypot(2, 2), rel=1e-12)


def test_off_raster_walls_bound_the_overbanks() -> None:
    """The 9999 m walls from sampling are in the overbanks, and their 9999 roughness still doesn't count."""
    dem = np.full((11, 11), BED)
    xs = with_banks(sample_cross_section(dem, np.full_like(dem, N), 5, 5, np.pi / 2, 200.0, 10.0, 10.0), 20.0, 20.0)

    parts = compound_geometry(xs, depth=1.0)

    assert parts.channel.top_width == pytest.approx(40.0, abs=0.01)
    assert parts.left_overbank.top_width == pytest.approx(30.0, abs=0.01)
    for part in parts:
        assert part.mannings_n == pytest.approx(N, rel=1e-12)


def test_mirroring_swaps_the_overbanks() -> None:
    xs = with_banks(rough_valley(), 7.3, 21.0)
    mirrored = with_banks(XSection(xs.elevations[::-1].copy(), xs.mannings_n[::-1].copy(), xs.ordinate_distance),
                          21.0, 7.3)

    for wse in np.linspace(100.0, 130.0, 61):
        parts = compound_geometry(xs, wse=wse)
        mirror = compound_geometry(mirrored, wse=wse)
        assert mirror.left_overbank == pytest.approx(parts.right_overbank, rel=1e-12, abs=1e-12)
        assert mirror.channel == pytest.approx(parts.channel, rel=1e-12, abs=1e-12)
        assert discharge(mirrored, SLOPE, wse=wse) == pytest.approx(discharge(xs, SLOPE, wse=wse), rel=1e-12)


def test_a_divided_section_matches_the_hand_calculation() -> None:
    """A V channel 1 m deep between banks 1 m either side, level ground for 1 m beyond each bank, then ground rising
    2 m over 1 m. Water 2 m deep fills the channel (3 m² over 2√2 m of bed) and spreads 1.5 m into each overbank
    (1.25 m² over 1 + √1.25 m)."""
    xs = with_banks(make_section(BED + np.array([3.0, 1.0, 1.0, 0.0, 1.0, 1.0, 3.0])), 1.0, 1.0)
    channel_perimeter, overbank_perimeter = 2.0 * math.sqrt(2.0), 1.0 + math.sqrt(1.25)

    parts = compound_geometry(xs, wse=BED + 2.0)
    whole = hydraulic_geometry(xs, wse=BED + 2.0)

    assert (parts.channel.area, parts.channel.wetted_perimeter, parts.channel.top_width) == \
        pytest.approx((3.0, channel_perimeter, 2.0), rel=1e-12)
    for overbank in (parts.left_overbank, parts.right_overbank):
        assert (overbank.area, overbank.wetted_perimeter, overbank.top_width) == \
            pytest.approx((1.25, overbank_perimeter, 1.5), rel=1e-12)
    assert (whole.area, whole.wetted_perimeter, whole.top_width) == \
        pytest.approx((5.5, channel_perimeter + 2.0 * overbank_perimeter, 5.0), rel=1e-12)
    expected = manning_conveyance(3.0, channel_perimeter) + 2.0 * manning_conveyance(1.25, overbank_perimeter)
    assert discharge(xs, SLOPE, wse=BED + 2.0) == pytest.approx(expected * math.sqrt(SLOPE), rel=1e-12)


def test_banks_at_the_channel_edge_remove_the_dip_onto_a_flat_floodplain() -> None:
    """As one section, discharge falls as the channel spills onto the floodplain. Divided at the channel's edges,
    the floodplain's conveyance starts from nothing and the channel's keeps rising."""
    whole = flat_floodplain()
    divided = with_banks(flat_floodplain(), 2.0, 2.0)
    levels = np.linspace(BED, 110.0, 2001)

    assert discharge(whole, SLOPE, wse=102.01) < discharge(whole, SLOPE, wse=102.0)
    flows = [discharge(divided, SLOPE, wse=wse) for wse in levels]
    assert np.all(np.diff(flows) > 0)


def test_discharge_is_continuous_as_the_water_edge_passes_a_bank() -> None:
    """The bank 7.3 m out is part way up a rising segment, whose ground there is at bank_ground."""
    xs = with_banks(rough_valley(), 30.0, 7.3)
    bank_ground = np.interp(7.3 / 2.0, np.arange(81), xs.elevations[80:])
    assert hydraulic_geometry(xs, wse=bank_ground).top_width > 7.3  # the water gets there across the channel

    below, above = discharge(xs, SLOPE, wse=bank_ground - 1e-7), discharge(xs, SLOPE, wse=bank_ground + 1e-7)
    assert above == pytest.approx(below, rel=1e-6)


def test_conveyance_without_powers_matches_the_formula() -> None:
    """Conveyance A**(5/3) / W**(2/3) is found as A**2 * (A * W**2)**(-1/3), with no power or division."""
    rng = np.random.default_rng(3)
    areas = np.exp(rng.uniform(math.log(1e-8), math.log(1e8), 20000))
    weighted = np.exp(rng.uniform(math.log(1e-6), math.log(1e4), 20000))

    found = np.array([hydraulics._conveyance(a, w) for a, w in zip(areas, weighted)])

    assert found == pytest.approx(areas ** (5 / 3) / weighted ** (2 / 3), rel=1e-14)
    assert hydraulics._conveyance(0.0, 1.0) == 0.0
    assert hydraulics._conveyance(1.0, 0.0) == math.inf  # a wetted perimeter with no roughness
    assert math.isnan(hydraulics._conveyance(math.nan, 1.0))
    assert hydraulics._conveyance(1e120, 1.0) == pytest.approx(1e200, rel=1e-14)  # beyond the fast method's range


@pytest.mark.parametrize("shape", ["v-channel", "trapezoid", "flat-floodplain", "walled-bed", "rough-valley"])
@pytest.mark.parametrize("banks", BANKS)
def test_compound_wse_for_discharge_carries_that_discharge(shape: str, banks: tuple[float, float]) -> None:
    xs = with_banks(SHAPES[shape](), *banks)

    for q in np.geomspace(0.01, 5000.0, 40):
        wse = wse_for_discharge(xs, q, SLOPE)
        assert discharge(xs, SLOPE, wse=wse) == pytest.approx(q, rel=1e-6)  # the solve stops within 1e-10 m


def test_compound_wse_agrees_with_bisection() -> None:
    """On a section whose divided discharge always rises with the water level, so that bisection has one answer."""
    xs = with_banks(flat_floodplain(), 2.0, 2.0)

    for q in [0.5, 5.0, 50.0, 500.0]:
        assert wse_for_discharge(xs, q, SLOPE) == pytest.approx(bisected_wse(xs, q), abs=1e-9)


def test_compound_wse_is_the_lowest_when_an_overbank_s_discharge_falls() -> None:
    """With banks inside the channel, the overbanks hold the channel's upper banks, and spill onto the flat
    floodplain at 102 m, where their discharge falls and so does the total."""
    xs = with_banks(flat_floodplain(), 1.0, 1.0)
    q = 0.97 * discharge(xs, SLOPE, wse=102.0)
    assert discharge(xs, SLOPE, wse=102.01) < q

    wse = wse_for_discharge(xs, q, SLOPE)

    assert wse < 102.0
    assert discharge(xs, SLOPE, wse=wse) == pytest.approx(q, rel=1e-9)
    assert max(discharge(xs, SLOPE, wse=level) for level in np.linspace(BED, wse, 2000, endpoint=False)) < q


def test_compound_wse_sits_on_the_ridge_when_the_overbank_behind_it_floods() -> None:
    xs = with_banks(ridge_and_hollow(), 3.0, 3.0)  # the right bank is on the ridge, with the hollow beyond
    below = discharge(xs, SLOPE, wse=103.0)
    above = discharge(xs, SLOPE, wse=103.0 + 1e-9)
    assert compound_geometry(xs, wse=103.0 + 1e-9).right_overbank.area > 10.0
    assert above > 1.5 * below

    assert wse_for_discharge(xs, (below + above) / 2, SLOPE) == 103.0


def test_compound_discharge_above_the_highest_ground_is_held_between_the_walls() -> None:
    xs = with_banks(trapezoid(10.0, 2.0, ordinates_per_side=20), 5.0, 5.0)  # the banks top out at 107.5 m
    q = 10 * discharge(xs, SLOPE, wse=107.5)

    wse = wse_for_discharge(xs, q, SLOPE)

    assert wse > 107.5
    assert discharge(xs, SLOPE, wse=wse) == pytest.approx(q, rel=1e-9)


def test_compound_edge_cases() -> None:
    single = with_banks(make_section([BED]), 0.0, 0.0)
    assert compound_geometry(single, depth=1.0).channel.area == 0.0
    assert math.isnan(wse_for_discharge(single, 1.0, SLOPE))

    xs = with_banks(trapezoid(10.0, 2.0), 3.0, 4.5)
    assert wse_for_discharge(xs, 0.0, SLOPE) == BED
    assert compound_geometry(xs, depth=0.0) == CompoundGeometry(DRY, DRY, DRY)

    xs32 = with_banks(XSection(xs.elevations.astype(np.float32), xs.mannings_n.astype(np.float32), 1.0), 3.0, 4.5)
    assert discharge(xs32, SLOPE, depth=1.3) == pytest.approx(discharge(xs, SLOPE, depth=1.3), rel=1e-6)
    assert wse_for_discharge(xs32, 20.0, SLOPE) == pytest.approx(wse_for_discharge(xs, 20.0, SLOPE), abs=1e-5)


# --- Depth-varying roughness and the slope factor ------------------------------------------------------------------


def random_roughness(rng) -> DepthRoughness:
    return DepthRoughness(float(rng.uniform(0.5, 10.0)), float(rng.uniform(1.0, 3.0)), float(rng.uniform(0.2, 1.0)))


def test_the_depth_scaling_follows_its_formula() -> None:
    """n = n0 * (deep_factor + (shallow_factor - deep_factor) / (1 + k_decay * depth)), a dry depth counting as 0,
    for arrays as for single values."""
    n0 = np.full(4, 0.03)
    depth = np.array([-1.0, 0.0, 0.5, 100.0])

    scaled = DepthRoughness(2.0, 2.5, 0.6).scale(n0, depth)

    assert scaled == pytest.approx(0.03 * (0.6 + 1.9 / (1.0 + 2.0 * np.array([0.0, 0.0, 0.5, 100.0]))), rel=1e-15)
    assert DepthRoughness(2.0, 1.0, 1.0).scale(n0, depth) == pytest.approx(n0, rel=1e-15)
    assert np.array_equal(n0, np.full(4, 0.03)) and np.array_equal(depth, [-1.0, 0.0, 0.5, 100.0])  # unchanged


def test_shallow_water_is_shallow_factor_times_rougher() -> None:
    roughness = DepthRoughness(6.0, 2.0, 1.0)

    assert roughness.scale(0.04, 0.0) == pytest.approx(0.08)
    assert roughness.scale(0.04, -1.0) == pytest.approx(0.08)  # dry counts as no depth
    assert roughness.scale(0.04, 1.0) == pytest.approx(0.04 * (1.0 + 1.0 / 7.0))
    assert DepthRoughness(6.0, 2.0, 0.5).scale(0.04, 1e9) == pytest.approx(0.02)  # deep_factor in the limit


def test_a_flat_bed_s_roughness_is_its_n_scaled_for_the_depth() -> None:
    """On a flat bed out to the section's ends every segment is as deep as the water, and so are the end walls the
    water stands against, so the composite n is n0 scaled for that depth."""
    xs = make_section(np.full(21, BED))
    roughness = DepthRoughness(6.0, 2.0, 1.0)

    geometry = hydraulic_geometry(xs, depth=1.5, roughness=roughness)

    assert geometry.wetted_perimeter == pytest.approx(20.0 + 2 * 1.5)  # the walls count
    assert geometry.mannings_n == pytest.approx(roughness.scale(N, 1.5), rel=1e-12)
    assert geometry.area == hydraulic_geometry(xs, depth=1.5).area


def test_each_segment_takes_the_depth_over_its_inner_end() -> None:
    """In a V at 1 m spacing with banks rising 1 m a metre, water 2 m deep covers a segment 2 m deep at its inner
    end and 1 m at its outer end, then meets the ground at the end of the next, over its inner end 1 m deep."""
    xs = trapezoid(0.0, 1.0, ordinates_per_side=5)
    roughness = DepthRoughness(6.0, 2.0, 1.0)
    n_15 = lambda depth: float(roughness.scale(N, depth)) ** 1.5
    length = math.sqrt(2.0)

    geometry = hydraulic_geometry(xs, wse=BED + 2.0, roughness=roughness)

    assert (geometry.area, geometry.wetted_perimeter, geometry.top_width) == pytest.approx((4.0, 4 * length, 4.0),
                                                                                         rel=1e-12)
    weighted = 2 * (length * n_15(2.0) + length * n_15(1.0))
    assert geometry.mannings_n == pytest.approx((weighted / (4 * length)) ** (2 / 3), rel=1e-12)


@pytest.mark.parametrize("shape", SHAPES)
def test_factors_of_one_leave_the_roughness_as_it_is(shape: str) -> None:
    rng = np.random.default_rng(41)
    for left, right in BANKS + [(-1.0, -1.0)]:
        xs = with_banks(SHAPES[shape](), left, right)
        for wse in BED + rng.uniform(-0.5, 8.0, 20):
            roughness = DepthRoughness(float(rng.uniform(0.5, 9.0)), 1.0, 1.0)
            assert compound_geometry(xs, wse=wse, roughness=roughness) == compound_geometry(xs, wse=wse)
            assert discharge(xs, SLOPE, wse=wse, roughness=roughness) == discharge(xs, SLOPE, wse=wse)


def test_the_depth_varying_wse_is_the_lowest_that_carries_the_flow() -> None:
    """Against a dense scan of the water surface, on random valleys with walls, spills and banks."""
    rng = np.random.default_rng(43)
    for _ in range(300):
        size = 2 * int(rng.integers(3, 40)) + 1
        elevations = BED + np.cumsum(rng.normal(0.0, 0.7, size))
        elevations -= elevations[size // 2] - BED
        if rng.random() < 0.3:
            elevations[rng.integers(0, size, 2)] = 9999.0
        xs = make_section(elevations, rng.uniform(0.02, 0.12, size), spacing=3.0)
        if rng.random() < 0.5:
            with_banks(xs, float(rng.uniform(0.0, 30.0)), float(rng.uniform(0.0, 30.0)))
        roughness = random_roughness(rng)
        q = float(10 ** rng.uniform(-1, 3))
        wse = wse_for_discharge(xs, q, SLOPE, roughness=roughness)
        if np.isclose(xs.elevations, wse, rtol=0.0, atol=1e-9).any():
            # Where the discharge jumps past q, at ground the water spills over, it carries q just above
            assert discharge(xs, SLOPE, wse=wse + 1e-9, roughness=roughness) >= q * (1 - 1e-9)
        else:
            assert discharge(xs, SLOPE, wse=wse, roughness=roughness) >= q
        scan = BED + np.linspace(0.0, wse - BED, 2001)[1:-1]
        assert all(discharge(xs, SLOPE, wse=w, roughness=roughness) < q * (1 + 1e-9) for w in scan)


@pytest.mark.parametrize("shape", SHAPES)
def test_with_factors_of_one_the_wse_is_the_constant_roughness_one(shape: str) -> None:
    xs = SHAPES[shape]()
    for left, right in BANKS + [(-1.0, -1.0)]:
        with_banks(xs, left, right)
        for q in (0.05, 1.0, 20.0, 300.0):
            expected = wse_for_discharge(xs, q, SLOPE)
            got = wse_for_discharge(xs, q, SLOPE, roughness=DepthRoughness(6.0, 1.0, 1.0))
            assert got == pytest.approx(expected, abs=1e-8) or (math.isnan(got) and math.isnan(expected))


def test_the_depth_varying_wse_s_special_cases() -> None:
    xs = trapezoid(10.0, 2.0)
    args = (*hydraulics.hydraulic_profile(xs), -1.0, -1.0)
    roughness = (6.0, 2.0, 1.0)

    assert hydraulics.wse_for_profile_conveyance(*args, 0.0, *roughness) == BED
    assert hydraulics.wse_for_profile_conveyance(*args, -5.0, *roughness) == BED
    assert math.isnan(hydraulics.wse_for_profile_conveyance(*args, math.nan, *roughness))
    reach = hydraulics.wse_for_profile_conveyance(*args, 5000.0, *roughness)
    assert math.isnan(hydraulics.wse_for_profile_conveyance(*args, 5000.0, *roughness, reach - 0.01))
    assert hydraulics.wse_for_profile_conveyance(*args, 5000.0, *roughness, reach + 0.01) == reach
    # Water held in by walls rises until it carries the flow, however deep
    walled = walled_bed()
    deep = wse_for_discharge(walled, 500.0, SLOPE, roughness=DepthRoughness())
    assert deep > BED + 5.0
    assert discharge(walled, SLOPE, wse=deep, roughness=DepthRoughness()) == pytest.approx(500.0)


def test_the_slope_factor_multiplies_the_discharge() -> None:
    xs = channel_and_floodplains()

    for factor in (0.5, 1.0, 1.7):
        for roughness in (None, DepthRoughness()):
            q = discharge(xs, SLOPE, depth=2.5, roughness=roughness)
            assert discharge(xs, SLOPE, depth=2.5, roughness=roughness, slope_factor=factor) == \
                pytest.approx(q * factor, rel=1e-12)
            assert wse_for_discharge(xs, q * factor, SLOPE, roughness=roughness, slope_factor=factor) == \
                pytest.approx(wse_for_discharge(xs, q, SLOPE, roughness=roughness), abs=1e-9)


@pytest.mark.parametrize("roughness, message", [
    (DepthRoughness(6.0, 0.9, 1.0), "shallow_factor"), (DepthRoughness(6.0, 2.0, 0.0), "deep_factor"),
    (DepthRoughness(6.0, 2.0, 1.2), "deep_factor"), (DepthRoughness(0.0, 2.0, 1.0), "k_decay"),
    (DepthRoughness(math.nan, 2.0, 1.0), "k_decay")])
def test_roughness_parameters_are_checked(roughness, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        discharge(trapezoid(10.0, 2.0), SLOPE, depth=1.0, roughness=roughness)
    with pytest.raises(ValueError, match=message):
        roughness.scale(0.035, 1.0)


@pytest.mark.parametrize("factor", [0.0, -1.0, math.nan, math.inf])
def test_the_slope_factor_must_be_finite_and_positive(factor: float) -> None:
    with pytest.raises(ValueError, match="slope_factor"):
        discharge(trapezoid(10.0, 2.0), SLOPE, depth=1.0, slope_factor=factor)


# --- Profiles ---------------------------------------------------------------------------------------------------


def random_valley(rng) -> XSection:
    """A random valley with walls, spills and banks sometimes."""
    size = 2 * int(rng.integers(3, 40)) + 1
    elevations = BED + np.cumsum(rng.normal(0.0, 0.7, size))
    elevations -= elevations[size // 2] - BED
    if rng.random() < 0.3:
        elevations[rng.integers(0, size, 2)] = 9999.0
    xs = make_section(elevations, rng.uniform(0.02, 0.12, size), spacing=float(rng.choice([1.0, 3.0, 7.3])))
    if rng.random() < 0.6:
        with_banks(xs, float(rng.uniform(0.0, 30.0)), float(rng.uniform(0.0, 30.0)))
    return xs


def split_profile(rng, profile, extra: int, keep=()):
    """The same ground with extra vertices along its segments, and at the distances in keep. Each new vertex takes
    the Manning's n of the segment it's on, which is its vertex nearer the stream cell's."""
    stations, elevations, n, _ = profile
    new = np.concatenate([rng.uniform(stations[0], stations[-1], extra), np.asarray(keep, dtype=np.float64)])
    new = np.unique(new[(new != 0.0) & ~np.isin(new, stations) & (new > stations[0]) & (new < stations[-1])])
    below = np.searchsorted(stations, new) - 1  # the vertex before each new one
    inner = np.where(new < 0.0, below + 1, below)
    order = np.argsort(np.concatenate([stations, new]), kind="stable")
    all_stations = np.concatenate([stations, new])[order]
    all_n = np.concatenate([n, n[inner]])[order]
    return hydraulics.Profile(all_stations, np.interp(all_stations, stations, elevations), all_n,
                              int(np.flatnonzero(all_stations == 0.0)[0]))


def test_splitting_the_ground_s_segments_changes_nothing() -> None:
    """With n not varying with depth, extra vertices along the segments, at the banks too, leave every
    subsection's hydraulics as they were."""
    rng = np.random.default_rng(61)
    for _ in range(300):
        xs = random_valley(rng)
        profile = hydraulics.hydraulic_profile(xs)
        banks = (xs.left_bank_distance, xs.right_bank_distance)
        split = split_profile(rng, profile, int(rng.integers(1, 40)), keep=[-banks[0], banks[1]])
        for wse in BED + rng.uniform(-0.5, 8.0, 10):
            expected = hydraulics.profile_compound_geometry(*profile, *banks, wse, 1.0, 1.0, 1.0)
            got = hydraulics.profile_compound_geometry(*split, *banks, wse, 1.0, 1.0, 1.0)
            np.testing.assert_allclose(np.array(got), np.array(expected), rtol=1e-10, atol=1e-9)


def test_a_vertical_face_adds_only_its_wetted_height() -> None:
    """A rectangle 10 m wide and 3 m deep, drawn with vertical faces: 2 m of water is 20 m^2, 10 m across, with
    14 m of wetted perimeter."""
    profile = hydraulics.Profile(np.array([-5.0, -5.0, 0.0, 5.0, 5.0]), np.array([BED + 3.0, BED, BED, BED, BED + 3.0]),
                                 np.full(5, N), 2)

    _, channel, _ = hydraulics.profile_compound_geometry(*profile, -1.0, -1.0, BED + 2.0, 1.0, 1.0, 1.0)

    assert channel[:3] == pytest.approx((20.0, 14.0, 10.0))
    assert hydraulics.wse_for_profile_conveyance(*profile, -1.0, -1.0, manning_conveyance(20.0, 14.0), 1.0, 1.0,
                                                 1.0) == pytest.approx(BED + 2.0)


@pytest.mark.parametrize("beyond", [1.0, 3.0])
def test_a_face_at_a_bank_belongs_to_the_side_whose_water_it_holds(beyond: float) -> None:
    """A channel whose banks, 5 m out, are 2 m up, with a face at each bank to ground beside it that is lower (a
    face falling outward, which holds the overbank's water) or higher (rising, which holds the channel's)."""
    stations = np.array([-20.0, -5.0, -5.0, -3.0, 0.0, 3.0, 5.0, 5.0, 20.0])
    elevations = BED + np.array([beyond + 0.5, beyond, 2.0, 0.0, 0.0, 0.0, 2.0, beyond, beyond + 0.5])
    profile = hydraulics.Profile(stations, elevations, np.full(9, N), 4)

    left, channel, right = hydraulics.profile_compound_geometry(*profile, 5.0, 5.0, BED + 4.0, 1.0, 1.0, 1.0)

    face = abs(beyond - 2.0)
    assert left == pytest.approx(right)
    assert channel[:3] == pytest.approx((16.0 + 10.0 * 2.0,
                                         6.0 + 2 * math.hypot(2.0, 2.0) + (2 * face if beyond > 2.0 else 0.0), 10.0))
    assert left[:3] == pytest.approx((15.0 * (4.0 - beyond + 3.5 - beyond) / 2,
                                      math.hypot(15.0, 0.5) + (face if beyond < 2.0 else 0.0) + (3.5 - beyond),
                                      15.0))


def test_the_profile_wse_is_the_lowest_that_carries_the_flow() -> None:
    """Against a dense scan of the water surface, on random profiles with vertices between the ordinates and
    vertical faces."""
    rng = np.random.default_rng(62)
    for _ in range(200):
        xs = random_valley(rng)
        stations, elevations, n, center = split_profile(rng, hydraulics.hydraulic_profile(xs), 10)
        for k in rng.integers(0, stations.size, 3):  # a vertical face: a vertex doubled, at another height
            if k != center:
                stations = np.insert(stations, k, stations[k])
                elevations = np.insert(elevations, k, elevations[k] + rng.normal(0.0, 1.0))
                n = np.insert(n, k, n[k])
                center += int(k < center)
        profile = hydraulics.Profile(stations, elevations, n, center)
        banks = (xs.left_bank_distance, xs.right_bank_distance)
        roughness = tuple(random_roughness(rng))
        target = float(10 ** rng.uniform(0, 4)) / math.sqrt(SLOPE)
        wse = hydraulics.wse_for_profile_conveyance(*profile, *banks, target, *roughness)
        if math.isnan(wse):
            continue
        conveyance = lambda w: hydraulics.profile_conveyance(*profile, *banks, w, *roughness)
        if np.isclose(elevations, wse, rtol=0.0, atol=1e-9).any():
            # Where the conveyance jumps past the target, at ground the water spills over, it's carried just above
            assert conveyance(wse + 1e-9) >= target * (1 - 1e-9)
        else:
            assert conveyance(wse) >= target * (1 - 1e-12)
        scan = elevations[center] + np.linspace(0.0, wse - elevations[center], 2001)[1:-1]
        assert all(conveyance(w) < target * (1 + 1e-9) for w in scan)


def test_the_cross_section_functions_use_its_profile() -> None:
    """With a profile set, the hydraulics are the profile's, not the ordinates'."""
    xs = channel_and_floodplains()
    xs.profile = hydraulics.Profile(np.array([-30.0, -12.0, -6.0, 0.0, 6.0, 12.0, 30.0]),
                                    BED + np.array([2.0, 2.0, 0.0, 0.0, 0.0, 2.0, 2.0]), np.full(7, N), 3)

    parts = hydraulics.profile_compound_geometry(*xs.profile, 12.0, 12.0, BED + 1.0, 1.0, 1.0, 1.0)
    assert compound_geometry(xs, wse=BED + 1.0).channel.area == pytest.approx(parts[1][0])
    assert hydraulic_geometry(xs, depth=1.0).top_width == pytest.approx(12.0 + 6.0)
    q = discharge(xs, SLOPE, wse=BED + 1.5)
    assert wse_for_discharge(xs, q, SLOPE) == pytest.approx(BED + 1.5)
    assert wse_for_discharge(xs, q, SLOPE, roughness=DepthRoughness()) > BED + 1.5  # rougher when shallow


def test_the_hydraulics_at_many_water_surfaces_are_those_at_each() -> None:
    """section_hydraulics_at, which works out ascending water surfaces together, gives what section_hydraulics gives
    at each, bit for bit, as do the standard step's tables, whose levels may not ascend: on random profiles with
    vertices between the ordinates, vertical faces, banks and walls, at water surfaces on the ground's vertices and
    between them, dry and over the walls, with n varying with the depth or not."""
    from arc import rating_curve, standard_step
    rng = np.random.default_rng(63)
    for _ in range(400):
        xs = random_valley(rng)
        stations, elevations, n, center = split_profile(rng, hydraulics.hydraulic_profile(xs),
                                                        int(rng.integers(0, 10)))
        for k in rng.integers(0, stations.size, 2):  # vertical faces
            if k != center:
                stations = np.insert(stations, k, stations[k])
                elevations = np.insert(elevations, k, elevations[k] + rng.normal(0.0, 1.0))
                n = np.insert(n, k, n[k])
                center += int(k < center)
        banks = (float(xs.left_bank_distance), float(xs.right_bank_distance))
        roughness = [(1.0, 1.0, 1.0), tuple(random_roughness(rng)), (float(rng.uniform(0.5, 10.0)),) +
                     (float(rng.uniform(0.2, 2.0)),) * 2][int(rng.integers(3))]
        levels = np.concatenate([elevations[center] + rng.uniform(-1.0, 12.0, int(rng.integers(1, 50))),
                                 rng.choice(elevations, 5), [10_000.0]])
        levels = np.unique(levels)  # ascending
        args = (stations, elevations, n, center, *banks)
        got = rating_curve.section_hydraulics_at(*args, levels, *roughness)
        expected = np.array([rating_curve.section_hydraulics(*args, wse, *roughness) for wse in levels])
        assert np.array_equal(got.view(np.uint64), expected.view(np.uint64))
        table = standard_step._table(*args, levels, *roughness)
        shuffled = rng.permutation(levels)
        table_shuffled = standard_step._table(*args, shuffled, *roughness)
        order = np.argsort(shuffled)
        for column, column_shuffled in zip(table, table_shuffled):
            assert np.array_equal(column.view(np.uint64), column_shuffled[order].view(np.uint64))
