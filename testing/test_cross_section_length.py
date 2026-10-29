"""Cross sections trimmed to what the steps need and extended when a step needs more (arc.pipeline's notes on cross
section lengths), the Manning's n raster kept as its classes, and the angle search sampling only as far out as it
looks: each gives what the whole cross sections give, bit for bit."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import test_pipeline as channel
from numba import njit

from arc import pipeline
from arc.bathymetry import Banks, carve_channel, ordinate_cells, set_in_bank_roughness
from arc.config import Configs
from arc.hydraulics import top_widths
from arc.xsection.orientation import (
    _SAME_DIRECTION,
    _narrowest_direction,
    angle_offsets,
)
from arc.xsection.sampling import (
    OFF_RASTER_ELEVATION,
    sample_elevations,
)
from arc.xsection.xsection import XSection


def terrain(shape=(90, 110), seed=3) -> np.ndarray:
    """A valley with a floodplain, ridges beyond it and bumps, falling along the columns, with a corner off the
    raster (walls)."""
    rng = np.random.default_rng(seed)
    rows, cols = np.mgrid[0:shape[0], 0:shape[1]]
    across = np.abs(rows - shape[0] / 2.0)
    dem = 100.0 - 0.01 * cols + np.where(across < 4, 0.0, 0.3 * (across - 4)) + 3.0 * np.sin(across / 6.0) ** 2
    dem += rng.uniform(0.0, 0.4, shape)
    dem[:8, :12] = OFF_RASTER_ELEVATION
    return dem


# --- The angle search ------------------------------------------------------------------------------------------------


@njit
def _reach(elevations, center, step, spacing):
    end = elevations.size - 1 if step > 0 else 0
    j = center
    while j != end and elevations[j + step] < OFF_RASTER_ELEVATION:
        j += step
    return spacing * abs(j - center)


@njit
def whole_section_search(dem, row, col, stream_direction, length, dx, dy, offsets, test_depth):
    """The angle search as it was, on whole cross sections."""
    tried = np.empty(offsets.size)
    widths = np.empty((offsets.size, 2))
    reaches = np.empty((offsets.size, 2))
    count = 0
    for offset in offsets:
        offset = offset % np.pi
        if offset > np.pi / 2:
            offset -= np.pi
        repeat = False
        for j in range(count):
            if abs(tried[j] - offset) < _SAME_DIRECTION:
                repeat = True
        if repeat:
            continue
        elevations, spacing = sample_elevations(dem, row, col, stream_direction + offset, length, dx, dy)
        center = elevations.size // 2
        left, right = top_widths(elevations, spacing, elevations[center] + test_depth)
        left_reach, right_reach = _reach(elevations, center, -1, spacing), _reach(elevations, center, 1, spacing)
        tried[count] = offset
        widths[count, 0] = min(left, left_reach)
        widths[count, 1] = min(right, right_reach)
        reaches[count, 0], reaches[count, 1] = left_reach, right_reach
        count += 1
    left_limit, right_limit = reaches[:count, 0].min(), reaches[:count, 1].min()
    best, narrowest = 0, np.inf
    for j in range(count):
        width = min(widths[j, 0], left_limit) + min(widths[j, 1], right_limit)
        if width < narrowest:
            best, narrowest = j, width
    return stream_direction + tried[best]


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_the_angle_search_finds_what_it_finds_on_whole_cross_sections(dtype) -> None:
    """Sampling each direction only out to the water's edge, and to a reach only where it could limit the widths,
    it turns each cross section the same way: in valleys and on floodplains, near the raster's edges and the walls
    beyond them, with water deep and shallow and cross sections short and long."""
    dem = terrain().astype(dtype)
    rng = np.random.default_rng(5)
    for _ in range(3000):
        row, col = int(rng.integers(dem.shape[0])), int(rng.integers(dem.shape[1]))
        direction = float(rng.uniform(0.0, np.pi))
        length = float(rng.choice([10.0, 60.0, 200.0, 1000.0]))
        depth = float(rng.choice([0.2, 1.0, 5.0, 25.0]))
        offsets = angle_offsets(float(rng.choice([13.28, 40.0, 180.0])), float(rng.choice([1.328, 7.5])))
        dx, dy = float(rng.choice([1.0, 0.8])), float(rng.choice([1.0, 1.3]))
        assert _narrowest_direction(dem, row, col, direction, length, dx, dy, offsets, depth) == \
            whole_section_search(dem, row, col, direction, length, dx, dy, offsets, depth)


# --- The Manning's n raster as classes ------------------------------------------------------------------------------


def test_a_manning_s_n_raster_kept_as_classes_samples_as_the_raster_of_numbers() -> None:
    rng = np.random.default_rng(9)
    classes = rng.choice(np.array([10, 20, 30, 80], dtype=np.uint8), size=(70, 90))
    lookup = np.zeros(256)
    lookup[[10, 20, 30, 80]] = np.float32([0.12, 0.05, 0.3, 0.03]).astype(np.float64)
    mannings_n = pipeline.ManningsN(classes, lookup, float(np.float32(0.035)))
    mannings_n.water_between_banks(rng.integers(-5, 75, 900), rng.integers(-5, 95, 900))  # some off the raster
    numbers = np.where(mannings_n.water, mannings_n.water_n, lookup[classes])
    for _ in range(500):
        row, col = int(rng.integers(70)), int(rng.integers(90))
        direction, length = float(rng.uniform(0.0, np.pi)), float(rng.choice([5.0, 40.0, 300.0]))
        expected, _ = sample_elevations(numbers, row, col, direction, length, 1.0, 1.2)
        np.testing.assert_array_equal(mannings_n.sample(row, col, direction, length, 1.0, 1.2), expected)
    rows, cols = rng.integers(-3, 73, 50), rng.integers(-3, 93, 50)
    on = (rows >= 0) & (rows < 70) & (cols >= 0) & (cols < 90)
    np.testing.assert_array_equal(mannings_n.at(rows, cols)[on], numbers[rows[on], cols[on]])
    assert np.all(mannings_n.at(rows, cols)[~on] == OFF_RASTER_ELEVATION)


# --- Trimming and extending ------------------------------------------------------------------------------------------


def section_at(grid: pipeline.Grid, row: int, col: int, direction: float, length: float) -> pipeline.Section:
    elevations, spacing = sample_elevations(grid.dem, row, col, direction, length, grid.dx, grid.dy)
    rows, cols = ordinate_cells(row, col, direction, length, grid.dx, grid.dy)
    center = elevations.size // 2
    return pipeline.Section(XSection(elevations, np.empty(0), float(spacing)), row, col, direction,
                            (direction - np.pi / 2) % np.pi, rows, cols, 1, float(elevations[center]),
                            length=length, full_half=center)


@pytest.mark.parametrize("carved", [False, True])
def test_a_trimmed_cross_section_extended_is_the_untrimmed_one(carved: bool) -> None:
    """Trimmed, given its n and carved, and extended back out, a cross section has the untrimmed one's ordinates,
    cells, n and carved profile."""
    dem = terrain()
    grid = pipeline.Grid(None, dem, np.zeros(dem.shape, np.int64), np.full(dem.shape, 10, np.uint8), 1.0, 1.1)
    lookup = np.zeros(256)
    lookup[10], lookup[80] = 0.1, 0.035
    mannings_n = pipeline.ManningsN(grid.land_cover, lookup, 0.035)
    rng = np.random.default_rng(1)
    mannings_n.water_between_banks(rng.integers(0, 90, 400), rng.integers(0, 110, 400))
    resampler = pipeline.Resampler(grid, mannings_n)
    banks = Banks("width_to_depth_ratio", 3.4, 2.6, 101.0, 101.2, False, True)
    for row, col, direction in ((45, 50, 0.3), (44, 70, 0.05), (46, 90, 3.0)):  # across the valley floor
        full = section_at(grid, row, col, direction, 120.0)
        trimmed = section_at(grid, row, col, direction, 120.0)
        full.banks = trimmed.banks = banks
        pipeline.trim_section(trimmed, full.dem_low_point + 0.5)
        assert trimmed.xs.elevations.size < full.xs.elevations.size
        for section in (full, trimmed):
            pipeline.sample_roughness(section, grid, mannings_n)
            set_in_bank_roughness(section.xs, banks, 0.035)
            if carved:
                carve_channel(section.xs, banks, 1.5, trapezoid_height=0.2, bank_elevation=101.0)
        assert resampler.extend(trimmed, 10**6)
        for name in ("elevations", "mannings_n"):
            np.testing.assert_array_equal(getattr(trimmed.xs, name), getattr(full.xs, name))
        np.testing.assert_array_equal(trimmed.rows, full.rows)
        np.testing.assert_array_equal(trimmed.cols, full.cols)
        assert trimmed.length == pytest.approx(full.length, abs=2.0 * full.xs.ordinate_distance)
        if carved:
            for a, b in zip(trimmed.xs.profile, full.xs.profile):
                np.testing.assert_array_equal(a, b)
        assert not resampler.extend(trimmed, 10**6)  # no further than before it was trimmed


def test_a_cross_section_reaches_above_water_once_it_rises_above_it_on_both_sides() -> None:
    xs = XSection(np.array([104.0, 99.0, 103.0, 100.0, 101.0, 102.0, 105.0]), np.full(7, 0.03), 1.0)
    section = pipeline.Section(xs, 0, 0, 0.0, 0.0, np.zeros(7, np.intp), np.zeros(7, np.intp), 1, 100.0,
                               length=7.0, full_half=10)
    assert pipeline.reaches_above(section, 103.5)  # 104 on the left, 105 on the right
    assert not pipeline.reaches_above(section, 104.0)  # nothing above it on the left
    assert not pipeline.reaches_above(section, math.nan)
    section.full_half = 3  # untrimmed: as far as it can reach
    assert pipeline.reaches_above(section, 200.0)


# --- Whole runs --------------------------------------------------------------------------------------------------


def run_both(inputs: dict) -> tuple[pipeline.Results, pipeline.Results]:
    configs = Configs.from_mapping(inputs)
    return (pipeline.run(configs, quiet=True, write=False, trim=False),
            pipeline.run(configs, quiet=True, write=False))


def assert_same(untrimmed: pipeline.Results, trimmed: pipeline.Results) -> None:
    for a, b in ((untrimmed.curves, trimmed.curves),):
        if a is not None:
            np.testing.assert_array_equal(b.increments, a.increments)
            np.testing.assert_array_equal(b.metadata, a.metadata)
    if untrimmed.bathymetry is not None:
        np.testing.assert_array_equal(trimmed.bathymetry, untrimmed.bathymetry)
    # The cross-section file holds each cross section as far as the run kept it: a trimmed one's halves are the
    # untrimmed one's out to where it ends
    assert len(trimmed.cross_sections) == len(untrimmed.cross_sections)
    for a, b in zip(untrimmed.cross_sections, trimmed.cross_sections):
        assert (b.comid, b.row, b.col, b.ordinate_dist) == (a.comid, a.row, a.col, a.ordinate_dist)
        for name in ("xs1_profile", "manning_n_raster1", "xs2_profile", "manning_n_raster2"):
            x, y = getattr(a, name), getattr(b, name)
            np.testing.assert_array_equal(y, x[:y.size])
    if untrimmed.representative is not None:
        pd.testing.assert_frame_equal(trimmed.representative, untrimmed.representative, check_exact=True)


@pytest.mark.parametrize("use_banks", [False, True])
@pytest.mark.parametrize("margin", [pipeline.TRIM_MARGIN, -50.0], ids=["as estimated", "every step extends"])
def test_trimmed_cross_sections_give_the_untrimmed_results(tmp_path: Path, monkeypatch, margin: float,
                                                           use_banks: bool) -> None:
    """With bathymetry, the standard step, the cross-section file and representative cross sections, on an oblique
    channel with bumps, the results are the untrimmed cross sections', both as the trimming estimates what the steps
    need and when it trims to almost nothing, so that every step has to extend them."""
    monkeypatch.setattr(pipeline, "TRIM_MARGIN", margin)
    monkeypatch.setattr(pipeline, "TRIM_CARVING", 0.0 if margin < 0 else pipeline.TRIM_CARVING)
    inputs = {**channel.write_oblique_channel(tmp_path, roughness=0.3), "Degree_Manip": 20, "Degree_Interval": 5,
              "Standard_Step": True,
              "BATHY_Out_File": str(tmp_path / "bathy.tif"), "Bathy_Use_Banks": use_banks,
              "XS_Out_File": str(tmp_path / "xs.parquet"), "Build_Representative_Cross_Section": True,
              "Representative_Cross_Section_File": str(tmp_path / "representative.csv")}
    untrimmed, trimmed = run_both(inputs)

    assert_same(untrimmed, trimmed)
    lengths = [s.xs.elevations.size for s in trimmed.sections if s is not None]
    assert min(lengths) < max(s.xs.elevations.size for s in untrimmed.sections if s is not None)


def test_without_bathymetry_or_the_standard_step_trimming_changes_nothing(tmp_path: Path) -> None:
    inputs = {**channel.write_channel(tmp_path), "Print_Curve_File": str(tmp_path / "c.csv")}
    assert_same(*run_both(inputs))


def test_a_cross_section_reaches_past_the_raster_s_edge_before_it_s_trimmed(tmp_path: Path) -> None:
    """Untrimmed, every cross section runs off the raster on both sides, into its walls."""
    results = pipeline.run(Configs.from_mapping(channel.write_oblique_channel(tmp_path)), quiet=True, write=False,
                           trim=False)
    for section in results.sections:
        center = section.xs.elevations.size // 2
        assert section.xs.elevations[0] == section.xs.elevations[-1] == OFF_RASTER_ELEVATION
        assert section.full_half == center
