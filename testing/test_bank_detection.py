"""Bank detection (arc.bathymetry.find_banks) on channels whose banks are known: shapes, cell sizes, angles to the grid,
noise, and two real rivers whose water mask shows where the water is.

The width-to-depth ratio finds a channel's bank tops: the stage, a height of one of the cross section's ordinates above
its stream cell, at which the water is about to spill out of the channel. On the ordinates, that is within a spacing of
the true bank top on each side. Its stages are at least MIN_STAGE_STEP (0.1 m) apart, so that centimetres of noise on
the DEM's water or bed don't stop it in the channel's first centimetres; the highest stage is always tried, so a
floodplain too short or too flat to rise 0.1 m still ends the search. Flat water and a water mask (or the land cover)
find the water's edge, not the bank tops.
"""
from __future__ import annotations

import math
from dataclasses import fields
from pathlib import Path

import numpy as np
import pytest
from osgeo import gdal

from arc import pipeline
from arc.bathymetry import (
    banks_by_flat_water,
    banks_by_land_cover,
    banks_by_width_to_depth_ratio,
    find_banks,
)
from arc.bathymetry.banks import MIN_STAGE_STEP
from arc.config import Configs
from arc.xsection.xsection import XSection

BED = 100.0


def section(height, spacing: float, half_width: float = 80.0) -> XSection:
    """A symmetric cross section of the ground height(distance) above the stream cell, with ordinates `spacing`
    apart out to half_width each side."""
    distance = np.arange(-half_width, half_width + spacing / 2, spacing)
    return XSection(BED + height(np.abs(distance)), np.full(distance.size, 0.035), spacing)


def channel(bottom_half: float, side: float, bank: float, floodplain: float = 0.01):
    """A trapezoid bottom_half each side of the stream, sides rising 1 in `side` to banks `bank` high, and
    floodplains beyond rising `floodplain` a metre. Its bank tops are bottom_half + side * bank out."""
    top = bottom_half + side * bank
    return lambda d: np.clip((d - bottom_half) / side if side else np.where(d > bottom_half, bank, 0.0), 0.0, bank) \
        + floodplain * np.clip(d - top, 0.0, None), top


# --- Shapes and cell sizes ------------------------------------------------------------------------------------------

SHAPES = {
    "trapezoid": (10.0, 2.0, 3.0),
    "steep banks": (8.0, 0.5, 4.0),
    "wide and shallow": (30.0, 5.0, 1.5),
    "a V": (0.0, 3.0, 4.0),
}


@pytest.mark.parametrize("spacing", [0.5, 1.0, 2.0, 5.0])
@pytest.mark.parametrize("shape", SHAPES)
def test_width_to_depth_banks_are_within_a_spacing_of_the_bank_tops(shape: str, spacing: float) -> None:
    bottom_half, side, bank = SHAPES[shape]
    height, top = channel(bottom_half, side, bank)

    banks = banks_by_width_to_depth_ratio(section(height, spacing))

    # At the last ordinate below the bank top, or the first beyond it on the floodplain, which rises 1 cm a metre
    assert banks.valid
    assert banks.left == pytest.approx(top, abs=spacing) and banks.right == pytest.approx(top, abs=spacing)
    assert BED + bank - spacing / side - 1e-9 <= banks.left_elevation <= BED + bank + 0.01 * spacing + 1e-9


def test_with_one_bank_lower_the_water_spills_over_it() -> None:
    """The left bank 2 m high, the right 3 m: at the stage the water tops the left bank, the right bank is the water's
    edge part way up its side."""
    left, _ = channel(10.0, 2.0, 2.0)
    right, _ = channel(10.0, 2.0, 3.0)
    distance = np.arange(-60.0, 61.0)
    xs = XSection(BED + np.where(distance < 0, left(-distance), right(distance)), np.full(distance.size, 0.035), 1.0)

    banks = banks_by_width_to_depth_ratio(xs)

    assert banks.left == pytest.approx(14.0) and banks.left_elevation == pytest.approx(BED + 2.0)
    assert banks.right == pytest.approx(14.0) and banks.right_elevation == pytest.approx(BED + 2.0)


def test_a_two_stage_channel_s_banks_are_its_inner_channel_s() -> None:
    """A channel 2 m deep and 24 m wide at the top, a berm 20 m wide each side 2 m up, and outer banks to 5 m: the
    water first spills onto the berms, so the banks are the inner channel's."""
    height = lambda d: (np.clip(d - 10.0, 0.0, 2.0) + np.clip(d - 32.0, 0.0, 3.0)
                        + 0.01 * np.clip(d - 35.0, 0.0, None))

    banks = banks_by_width_to_depth_ratio(section(height, 1.0))

    assert (banks.left, banks.right) == pytest.approx((12.0, 12.0))


def test_a_v_shaped_valley_has_no_width_to_depth_banks() -> None:
    """In a V the ratio of width to depth stays the same at every stage, so it never rises: no banks there. Nor does
    flat water find any, 0.1 m up a V 0.6 m wide, narrower than two ordinates' spacing: find_banks finds none."""
    xs = section(lambda d: d / 3.0, 1.0)

    assert not banks_by_width_to_depth_ratio(xs).valid
    assert find_banks(xs).method == "none"


def test_a_short_floodplain_still_ends_the_search() -> None:
    """Beyond banks 3 m high the floodplain rises 0.05 m before the cross section ends at walls: no stage is
    MIN_STAGE_STEP above the bank top, but the highest is tried, and the water spilling there ends the search at the
    bank tops."""
    height, top = channel(10.0, 2.0, 3.0)
    distance = np.arange(-21.0, 22.0)
    elevations = np.where(np.abs(distance) <= 21.0, BED + height(np.abs(distance)), BED)
    elevations[[0, -1]] = 9999.0  # walls, as off the raster
    assert height(np.array([20.0]))[0] - 3.0 < MIN_STAGE_STEP

    banks = banks_by_width_to_depth_ratio(XSection(elevations, np.full(distance.size, 0.035), 1.0))

    assert (banks.left, banks.right) == pytest.approx((top, top))


# --- The water's edge -----------------------------------------------------------------------------------------------


def flat_water_river(spacing: float):
    """A channel whose DEM shows its water flat 0.6 m above its bed across the 21 m it spans, with banks 2.4 m high
    and 30 m apart: (height, water's half width, bank tops' distance)."""
    height, top = channel(9.0, 2.5, 2.4)
    return (lambda d: np.maximum(height(d), 0.6), 10.5, top)


@pytest.mark.parametrize("spacing", [1.0, 2.0, 3.0])
def test_flat_water_finds_the_water_s_edge_and_width_to_depth_the_bank_tops(spacing: float) -> None:
    height, water, top = flat_water_river(spacing)
    xs = section(height, spacing)

    flat, ratio = banks_by_flat_water(xs), banks_by_width_to_depth_ratio(xs)

    assert flat.left == pytest.approx(water, abs=spacing) and flat.right == pytest.approx(water, abs=spacing)
    assert ratio.left == pytest.approx(top, abs=spacing) and ratio.right == pytest.approx(top, abs=spacing)


@pytest.mark.parametrize("spacing", [1.0, 2.0, 3.0])
def test_a_water_mask_s_banks_are_half_way_to_the_first_dry_ordinate(spacing: float) -> None:
    height, water, _ = flat_water_river(spacing)
    xs = section(height, spacing)
    distance = (np.arange(xs.elevations.size) - xs.elevations.size // 2) * spacing
    mask = (np.abs(distance) <= water).astype(float)
    last_wet = math.floor(water / spacing) * spacing

    banks = banks_by_land_cover(xs, mask, 1.0)

    assert (banks.left, banks.right) == pytest.approx((last_wet + spacing / 2,) * 2)


# --- Noise ----------------------------------------------------------------------------------------------------------


def noisy_banks(sigma: float, *, on_the_water: bool, trials: int = 200) -> np.ndarray:
    """The width-to-depth banks' widths on the flat-water river at 1 m with Gaussian noise sigma on every ordinate but
    the stream cell's, or only on the dry ones."""
    height, water, _ = flat_water_river(1.0)
    rng = np.random.default_rng(3)
    widths = []
    for _ in range(trials):
        xs = section(height, 1.0)
        distance = np.abs(np.arange(xs.elevations.size) - xs.elevations.size // 2)
        noise = rng.normal(0.0, sigma, xs.elevations.size)
        noise[distance == 0] = 0.0
        if not on_the_water:
            noise[distance <= water] = 0.0
        xs.elevations += noise
        banks = banks_by_width_to_depth_ratio(xs)
        widths.append(banks.top_width if banks.valid else math.nan)
    return np.array(widths)


@pytest.mark.parametrize(("sigma", "share"), [(0.02, 0.8), (0.05, 0.6)])
def test_noise_on_the_water_doesn_t_stop_the_search_inside_the_channel(sigma: float, share: float) -> None:
    """With noise of 2 cm (5 cm) on the water and the banks, the banks are within a fifth of the bank tops' 30 m at 92%
    (80%) of the cross sections. Trying every stage, the water spilled from one dip of the noise into the next in the
    first centimetres, the ratio rose, and the banks were a few metres apart at nearly all of them."""
    widths = noisy_banks(sigma, on_the_water=True)

    assert np.mean(np.abs(widths - 30.0) <= 0.2 * 30.0) >= share
    assert np.nanmedian(widths) == pytest.approx(30.0, abs=2.0)


@pytest.mark.parametrize("sigma", [0.05, 0.25])
def test_noise_on_the_banks_alone_leaves_them_where_they_are(sigma: float) -> None:
    """With the water flat, as DEMs level it, even 25 cm of noise on the banks and floodplain leaves the banks within a
    fifth of the bank tops' width at every cross section."""
    widths = noisy_banks(sigma, on_the_water=False)

    assert np.mean(np.abs(widths - 30.0) <= 0.2 * 30.0) >= 0.9


# --- On a DEM: cell sizes and angles --------------------------------------------------------------------------------


@pytest.mark.parametrize("degrees", [0.0, 30.0, 45.0])
@pytest.mark.parametrize("cell", [2.0, 8.0])
def test_on_a_dem_the_banks_are_within_two_spacings_of_the_bank_tops(tmp_path: Path, cell: float,
                                                                     degrees: float) -> None:
    """test_dem_resilience's channel, its banks 32 m apart, on a DEM of cells 2 m or 8 m wide (a quarter of the
    channel's width), crossing the grid at an angle: every cross section's banks, a spacing apart on the slant, are
    within two spacings of the bank tops."""
    import test_dem_resilience as dem

    inputs, along = dem.write_dem(tmp_path, cell, degrees=degrees)
    results = pipeline.run(Configs.from_mapping(inputs), quiet=True, write=False)
    inside = np.abs(along(results.cells.rows, results.cells.cols)) <= dem.LENGTH / 2 - 60.0

    for k in np.flatnonzero(inside):
        banks, xs = results.sections[k].banks, results.sections[k].xs
        assert banks.method == "width_to_depth_ratio"
        assert banks.top_width == pytest.approx(2 * dem.BANKS_AT, abs=2.0 * xs.ordinate_distance)


# --- Real rivers ----------------------------------------------------------------------------------------------------

REAL_SITES = Path(r"C:\Users\lrr43\Documents\masters\fim_sites_nencarta")


def real_site_sections(site: str):
    """A FIM site's cross sections as its Bathy input file samples them, without carving, and the water mask at each
    one's ordinates (the stream cells water). Skipped where the site's data aren't on this machine."""
    import yaml

    path = REAL_SITES / site / "ARC_InputFiles" / "GEOGLOWS_ARC_Input_fabdem_Bathy.yaml"
    if not path.exists():
        pytest.skip(f"{site}'s data aren't on this machine")
    inputs = yaml.safe_load(path.read_text(encoding="utf-8"))
    # The file was shared with curve2flood: ARC's keys only
    arc_keys = {f.name for f in fields(Configs) if f.init}
    inputs = {key: value for key, value in inputs.items() if key.lower() in arc_keys}
    for key in ("BATHY_Out_File", "Print_VDT_Database", "Standard_Step"):
        inputs.pop(key, None)
    if not Path(inputs["LU_Manning_n"]).exists():
        inputs["LU_Manning_n"] = str(REAL_SITES.parent / inputs["LU_Manning_n"])
    if not Path(inputs["LU_Manning_n"]).exists():
        pytest.skip(f"{site}'s Manning's n table isn't on this machine")
    configs = Configs.from_mapping(inputs)
    results = pipeline.run(configs, quiet=True, write=False)
    mask = (np.nan_to_num(gdal.Open(str(configs.bathywatermask)).ReadAsArray()) != 0).astype(float)
    mask[results.cells.rows, results.cells.cols] = 1.0
    for s in results.sections:
        if s is not None:
            on = (s.rows >= 0) & (s.rows < mask.shape[0]) & (s.cols >= 0) & (s.cols < mask.shape[1])
            along = np.full(s.rows.size, np.nan)
            along[on] = mask[s.rows[on], s.cols[on]]
            yield s.xs, along


@pytest.mark.parametrize("site", ["Wabash_River_at_New_Harmony_(2016)", "Susquehanna_River_at_Harrisburg_Pa_(2013)"])
def test_a_real_site_s_width_to_depth_banks_are_outside_its_water(site: str) -> None:
    """Bank tops can't be inside the water. Where both find banks, the width-to-depth ratio's are less than 0.9 of
    the water mask's width at no more than 3% of the cross sections (trying every stage, at 73% on the Wabash, whose
    conditioned DEM dips 10 cm towards the stream across its water, and 72% on the Susquehanna)."""
    inside = found = 0
    for xs, water in real_site_sections(site):
        mask, ratio = banks_by_land_cover(xs, water, 1.0), banks_by_width_to_depth_ratio(xs)
        if mask.valid and ratio.valid:
            found += 1
            inside += ratio.top_width < 0.9 * mask.top_width
    assert found > 200
    assert inside <= 0.03 * found
