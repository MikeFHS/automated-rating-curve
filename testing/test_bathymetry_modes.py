"""What Bathy_Use_Banks means, on a river whose channel is known.

The river's channel is a trapezoid 18 m wide at the bed, its sides sloping 2.4:1 to banks 2.5 m high (30 m apart),
with floodplains rising 1% beyond them, n 0.03 and a slope of 0.001. Its DEM was made at a low flow, as DEMs are: it
shows the water's surface, flat at 0.625 m deep across the 21 m the water spans, and not the bed beneath it. The
water's edge is half way between two cells, 10.5 m out, so a water mask of the cells under water puts the banks there.

Bathy_Use_Banks says which level the DEM shows the channel full to, and so what Flow_File_BF has to be:

- False: the water surface at the stream cell, at the flow when the DEM was made (here the low flow, 8.87 m^3/s).
- True: the bank elevation, at the flow that fills the channel to its banks (here bankfull, 98.2 m^3/s).

Whatever the flow, ARC carves the channel so that the flow fills it to that level. Given the flow that belongs with
the level, the channel is the true one; given the other, it is metres too deep or too shallow. The banks a water mask
gives are the water's edge, not the bank tops, so with Bathy_Use_Banks and a mask the level is the water's.

With Bathy_Use_Banks in a canyon, the banks the DEM shows (a bench in its walls, or its rim) can be far above the
bankfull water, and carving the flow to fill the channel up to them builds the bed up metres above the DEM's water.
The canyon here is 40 m wide at the water, with walls 1:1 up to a bench 10 m above it and 30 m wide, and on up to 60 m.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
from osgeo import gdal, osr
from test_pipeline import _write_raster, write_inputs

from arc import pipeline
from arc.config import Configs
from arc.hydraulics import wse_for_discharge

SLOPE, N, LAND_N = 0.001, 0.03, 0.06
BOTTOM, SIDE, BANK = 18.0, 2.4, 2.5
LOW_DEPTH = 0.625  # the water's depth when the DEM was made
ROWS, COLS, CENTER_ROW = 81, 400, 40
WATER_TRAPEZOID_H = 1.5 / 21.0  # Bathy_Trap_H of the channel below the water: 1.5 m of each side in its 21 m
BANKS_TRAPEZOID_H = 0.2  # and of the channel below the banks: 6 m in 30 m


def manning_flow(depth: float) -> float:
    area = (BOTTOM + SIDE * depth) * depth
    perimeter = BOTTOM + 2.0 * depth * math.sqrt(1.0 + SIDE ** 2)
    return area * (area / perimeter) ** (2 / 3) * math.sqrt(SLOPE) / N


LOW_FLOW, BANKFULL_FLOW = manning_flow(LOW_DEPTH), manning_flow(BANK)


def height(distance: np.ndarray, hill=0.0) -> np.ndarray:
    """The true ground's height above the bed at a distance from the centre line, where a hill stands `hill` metres
    above the bank tops: its walls rise 1:1 from them."""
    distance = np.abs(distance)
    beyond = np.clip(distance - (BOTTOM / 2 + SIDE * BANK), 0.0, None)
    return np.clip((distance - BOTTOM / 2) / SIDE, 0.0, BANK) + np.maximum(0.01 * beyond, np.minimum(beyond, hill))


def hill_height(cols: np.ndarray) -> np.ndarray:
    """A hill 20 m above the bank tops from column 150 to 250, rising and falling over 30 columns either side."""
    return 20.0 * np.clip(np.minimum(cols - 120, 280 - cols) / 30.0, 0.0, 1.0)


def true_bed(col) -> float:
    return 100.0 - SLOPE * col


def write_river(folder: Path, flow: float, *, use_banks: bool, mask: bool = False,
                trapezoid_h: float = BANKS_TRAPEZOID_H, hill: bool = False) -> dict:
    """The river's inputs, in two reaches along row CENTER_ROW, with Flow_File_BF flow and a maximum flow of three
    times bankfull, and with hill, through a canyon in a hill (hill_height)."""
    rows, cols = np.mgrid[0:ROWS, 0:COLS]
    water = np.abs(rows - CENTER_ROW) <= 10
    ground = true_bed(cols) + height(rows - CENTER_ROW, hill_height(cols) if hill else 0.0)
    dem = np.where(water, np.maximum(ground, true_bed(cols) + LOW_DEPTH), ground)
    streams = np.zeros(dem.shape, dtype=np.int32)
    streams[CENTER_ROW, 20:200] = 1
    streams[CENTER_ROW, 200:380] = 2
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(32616)
    geotransform = (500000.0, 1.0, 0.0, 3800000.0, 0.0, -1.0)
    reaches = {1: {"start": (CENTER_ROW, 20), "end": (CENTER_ROW, 199), "downstream": 2, "baseflow": flow,
                   "q_max": 3.0 * BANKFULL_FLOW},
               2: {"start": (CENTER_ROW, 200), "end": (CENTER_ROW, 379), "downstream": -1, "baseflow": flow,
                   "q_max": 3.0 * BANKFULL_FLOW}}
    inputs = write_inputs(folder, dem, streams, srs, geotransform, reaches, water_n=N, land_n=LAND_N)
    inputs.update({"BATHY_Out_File": str(folder / "bathy.tif"), "Bathy_Use_Banks": use_banks,
                   "Bathy_Trap_H": trapezoid_h})
    if mask:
        inputs["BathyWaterMask"] = _write_raster(folder / "mask.tif", water.astype(np.uint8), geotransform, srs,
                                                 gdal.GDT_Byte)
    return inputs


def run(folder: Path, flow: float, **kwargs) -> list:
    """The cross sections a gen_slope_dist or more from the ends of the stream, where the slope is the bed's."""
    inputs = write_river(folder, flow, **kwargs)
    results = pipeline.run(Configs.from_mapping(inputs), quiet=True, write=False)
    return [(section, results.slopes[k]) for k, section in enumerate(results.sections)
            if 30 <= section.col <= 370]


def carved_bed(section) -> float:
    return float(section.xs.elevations[section.xs.elevations.size // 2]) - true_bed(section.col)


# Each mode with each flow: its banks' level, and the bed carved, above the true bed
CASES = {
    "water surface, low flow": (dict(use_banks=False, mask=True, trapezoid_h=WATER_TRAPEZOID_H), LOW_FLOW),
    "water surface, low flow, banks from the DEM": (dict(use_banks=False, trapezoid_h=WATER_TRAPEZOID_H), LOW_FLOW),
    "water surface, bankfull flow": (dict(use_banks=False, mask=True, trapezoid_h=WATER_TRAPEZOID_H), BANKFULL_FLOW),
    "banks, bankfull flow": (dict(use_banks=True), BANKFULL_FLOW),
    "banks, low flow": (dict(use_banks=True), LOW_FLOW),
    "banks from a water mask, bankfull flow": (dict(use_banks=True, mask=True), BANKFULL_FLOW),
}


@pytest.mark.parametrize("case", CASES)
def test_the_flow_fills_the_carved_channel_to_the_mode_s_level(tmp_path: Path, case: str) -> None:
    """The defining rule: Flow_File_BF fills the carved cross section to the DEM's water surface at the stream cell
    without Bathy_Use_Banks, and to the bank elevation with it, whichever flow it is."""
    options, flow = CASES[case]

    for section, slope in run(tmp_path, flow, **options):
        level = section.bank_elevation if options["use_banks"] else section.dem_low_point
        # Either side of the junction the bed's running median is a step (0.5 mm here) off the depth that carries the
        # flow (arc.bathymetry.bed_smoothing), which moves the water up to 1.1 mm
        assert wse_for_discharge(section.xs, flow, slope) == pytest.approx(level, abs=1.5e-3)


@pytest.mark.parametrize("case", ["water surface, low flow", "water surface, low flow, banks from the DEM",
                                  "banks, bankfull flow", "banks from a water mask, bankfull flow"])
def test_each_mode_given_its_own_flow_carves_the_true_channel(tmp_path: Path, case: str) -> None:
    """The low flow below the water surface, its banks the water's edge with a mask or without, and bankfull below
    the bank tops, mask or not: each is the true channel, to a millimetre (the bed's running median, without
    Bathy_Use_Banks, is a cell's fall, 1 mm, off near a reach's ends). The trapezoid has to be the true channel's below
    each level, Bathy_Trap_H."""
    options, flow = CASES[case]

    for section, _ in run(tmp_path, flow, **options):
        assert carved_bed(section) == pytest.approx(0.0, abs=1.5e-3)
        level = section.bank_elevation if options["use_banks"] else section.dem_low_point
        assert level - true_bed(section.col) == pytest.approx(BANK if options["use_banks"] else LOW_DEPTH, abs=1e-3)


def test_without_a_water_mask_the_water_surface_mode_carves_between_the_water_s_edges(tmp_path: Path) -> None:
    """Without Bathy_Use_Banks the channel is carved between the DEM's water's edges, half way from the last cell of
    water to the first of land, 21 m apart, not the bank tops 30 m apart that the width-to-depth ratio finds: the bank
    faces above the water keep their elevations. The hydraulics divide the carved cross section at the bank tops all
    the same: the bank faces are channel."""
    options, flow = CASES["water surface, low flow, banks from the DEM"]

    for section, _ in run(tmp_path, flow, **options):
        assert section.carve_banks.method == "water_edge"
        assert section.carve_banks.top_width == pytest.approx(21.0)
        assert section.hydraulic_banks.method == "width_to_depth_ratio"
        assert section.hydraulic_banks.top_width == pytest.approx(30.0)
        assert (section.xs.left_bank_distance, section.xs.right_bank_distance) == pytest.approx((15.0, 15.0))
        dry = np.abs(np.arange(section.xs.elevations.size) - section.xs.elevations.size // 2) >= 11
        assert np.all(section.xs.elevations[dry] >= section.dem_low_point)


@pytest.mark.parametrize("case, low, high", [("water surface, bankfull flow", -2.5, -2.0),
                                             ("banks, low flow", 1.8, 2.0)])
def test_a_flow_that_doesn_t_belong_with_the_mode_s_level_carves_the_wrong_channel(
        tmp_path: Path, case: str, low: float, high: float) -> None:
    """Bankfull below the water surface carves the bed 2 m too deep. The low flow below the banks fills the channel
    in, its bed nearly 2 m above the true one and 1.3 m above the water the DEM shows."""
    options, flow = CASES[case]

    for section, _ in run(tmp_path, flow, **options):
        assert low < carved_bed(section) < high
        above_the_water = carved_bed(section) > section.dem_low_point - true_bed(section.col)
        assert above_the_water == (case == "banks, low flow")


CANYON_SLOPE, CANYON_CELL, CANYON_WATER = 0.002, 5.0, 1.0  # the canyon's fall, its cells, its water's depth (m)


def canyon_height(distance: np.ndarray) -> np.ndarray:
    """The canyon's ground above its bed: 40 m wide, walls 1:1 to a bench 10 m up and 30 m wide, then 1:1 to 60 m."""
    distance = np.abs(distance)
    return np.clip(distance - 20.0, 0.0, 10.0) + np.clip(distance - 60.0, 0.0, 50.0)


def run_canyon(folder: Path, flow: float) -> tuple:
    """The canyon on 5 m cells, its DEM's water flat 1 m above its bed, with Bathy_Use_Banks and Flow_File_BF flow
    (and three times that the maximum): its cross sections a gen_slope_dist from the stream's ends, their slopes, the
    DEM, and the bathymetry raster."""
    rows, cols = np.mgrid[0:81, 0:160]
    bed = 500.0 - CANYON_SLOPE * CANYON_CELL * cols
    across = (rows - 40) * CANYON_CELL
    ground = bed + canyon_height(across)
    dem = np.where(np.abs(across) <= 21.0, np.maximum(ground, bed + CANYON_WATER), ground)
    streams = np.zeros(dem.shape, dtype=np.int32)
    streams[40, 10:80] = 1
    streams[40, 80:150] = 2
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(32616)
    reaches = {1: {"start": (40, 10), "end": (40, 79), "downstream": 2, "baseflow": flow, "q_max": 3.0 * flow},
               2: {"start": (40, 80), "end": (40, 149), "downstream": -1, "baseflow": flow, "q_max": 3.0 * flow}}
    inputs = write_inputs(folder, dem, streams, srs, (500000.0, CANYON_CELL, 0.0, 3800000.0, 0.0, -CANYON_CELL),
                          reaches, water_n=N, land_n=N)
    inputs.update({"BATHY_Out_File": str(folder / "bathy.tif"), "Bathy_Use_Banks": True,
                   "Stream_Slope_Method": "reach_average"})
    results = pipeline.run(Configs.from_mapping(inputs), quiet=True, write=False)
    sections = [(section, results.slopes[k]) for k, section in enumerate(results.sections)
                if 30 <= section.col <= 130]
    return sections, dem, results.bathymetry


def test_in_a_canyon_no_bank_is_above_the_maximum_flow_s_water(tmp_path: Path) -> None:
    """The bench, 10 m up, is above the water that carries the maximum flow (900 m^3/s, about 5 m up the walls), so
    it isn't taken for the banks, as width-to-depth would take it: the banks are the walls at the highest stage below
    that water, 4 m above the DEM's water, and the bed is built up 2 to 3 m above the DEM's water rather than 6 m.
    They aren't bank tops (max_flow_water), but with none anywhere on the site, they are all there is to go by. A
    hydro-conditioned DEM's channel (a burned line, or a river edited metres below its water) can look the same; where
    its maximum flow tops the trench, its rim is taken and its bed raised."""
    sections, dem, bathymetry = run_canyon(tmp_path, 300.0)

    assert 2.0 < np.nanmax(bathymetry - dem) < 3.0
    for section, _ in sections:
        water = dem[40, section.col]
        assert section.banks.method == "max_flow_water"
        assert section.bank_elevation == pytest.approx(water + 4.0, abs=0.01)


def test_a_canyon_through_a_hill_takes_the_bank_height_and_width_of_the_reach_around_it(tmp_path: Path) -> None:
    """The river runs through a hill 20 m above its bank tops for 100 m, the canyon's walls rising 1:1 from them. The
    maximum flow stands below the hill's top in the canyon, where the width-to-depth ratio is still falling: no bank
    top (max_flow_water). Its bank height and width are those of the bank tops round it, 2.5 m above the bed and 30 m
    apart, and its channel is carved as theirs is. Taking the highest stage below the maximum flow's water for a bank
    top instead carved the canyon 38 m wide, and its bed 0.35 m high."""
    options, flow = CASES["banks, bankfull flow"]

    sections = [section for section, _ in run(tmp_path, flow, hill=True, **options)]

    canyon = [section for section in sections if 150 <= section.col <= 250]
    open_bed = np.median([carved_bed(section) for section in sections if not 100 <= section.col <= 300])
    assert canyon and all(section.banks.method == "max_flow_water" for section in canyon)
    for section in canyon:
        assert section.bank_elevation - true_bed(section.col) == pytest.approx(BANK, abs=0.01)
        assert section.carve_banks.top_width == pytest.approx(30.0, abs=0.01)
        assert carved_bed(section) == pytest.approx(open_bed, abs=0.01)


def test_with_bank_elevations_a_water_mask_s_banks_are_the_bank_tops(tmp_path: Path) -> None:
    """With Bathy_Use_Banks the mask only shows where the water is: the banks are the bank tops above it, 30 m apart
    and 2.5 m above the bed, not the mask's water's edge 21 m apart, whose bank elevation would be the water's level."""
    options, flow = CASES["banks from a water mask, bankfull flow"]

    for section, _ in run(tmp_path, flow, **options):
        assert section.hydraulic_banks.method == "width_to_depth_ratio"
        assert section.hydraulic_banks.top_width == pytest.approx(30.0)
        assert section.bank_elevation - true_bed(section.col) == pytest.approx(BANK, abs=1e-3)


def test_without_bank_elevations_a_water_mask_s_banks_are_the_water_s_edge(tmp_path: Path) -> None:
    """Half way between the last cell of water and the first dry one, 10.5 m out: the DEM's water's edge, which the
    mask agrees with, is where the channel is carved to."""
    options, flow = CASES["water surface, low flow"]

    for section, _ in run(tmp_path, flow, **options):
        assert section.carve_banks.method == "water_edge"
        assert section.carve_banks.top_width == pytest.approx(21.0)
