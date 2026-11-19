"""Valley_Sections (arc.xsection.valley): each cross section's overbanks from lines out from its banks square to the
valley, with the valley's slope.

The river is synthetic_valley.meander(): a valley falling 0.001 with a 1.1 km floodplain, 10 m cells, and a channel
40 m wide and 2 m deep winding 200 m either side of the valley's axis every 800 m, n 0.04 everywhere. The reference is
a 2D model of its steady flow (LISFLOOD-FP's local inertial scheme), whose water surfaces at the stream cells are kept
in data/valley_meander_2d.npz. Over the floodplain the 2D water runs straight down the valley. ARC's cross sections,
square to the channel, cut the floodplain at a slant where the channel crosses the valley and take the channel's
gentler slope, so its water over the floodplain is out by 0.15 m on average with normal depth, mostly too low, and
0.06 m with the standard step. With Valley_Sections, the valley's direction taken over 16 channel widths of the stream
either way (Valley_Dir_Widths, the default: 800 m, the channel being 50 m bank to bank on ARC's cross sections), it's
out by 0.012 m and 0.010 m. In the channel little changes.
"""
from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import synthetic_valley
from osgeo import osr
from test_pipeline import write_inputs

from arc import pipeline
from arc.config import Configs
from arc.hydraulics import compound_geometry, discharge, hydraulic_profile
from arc.outputs import vdt_dataframe
from arc.outputs.rating_curves import SLOPE as SLOPE_COLUMN
from arc.pipeline import valley_windows
from arc.xsection.stream_path import along_stream_stations
from arc.xsection.valley import (
    Valley,
    fill_other_water,
    ray_reach,
    slope_ratio,
    turn,
    valley_axes,
    valley_lines,
    valley_overbanks,
)
from arc.xsection.xsection import Profile, XSection

SPACING, N = 5.0, 0.04
SLOPE = 0.001
OVERBANK_FLOWS = slice(2, None)  # 150, 400 and 1000 m^3/s: the water is over the floodplain


def trapezoid(half: int = 60) -> XSection:
    """A cross section with ordinates SPACING apart: a channel 30 m wide at its bed and 2 m deep, sides 2.5:1, between
    flat floodplains at 100 m, banks at its top, 20 m out."""
    x = (np.arange(2 * half + 1) - half) * SPACING
    ground = np.clip(98.0 + (np.abs(x) - 15.0) / 2.5, 98.0, 100.0)
    xs = XSection(ground, np.full(x.size, N), SPACING)
    xs.left_bank_distance = xs.right_bank_distance = 20.0
    return xs


def lines(left: np.ndarray, right: np.ndarray, ratio: float = 1.0, n: float = N) -> Valley:
    """Overbanks whose lines from the banks have this ground, out from the bank, ordinates SPACING apart."""
    elevations = np.r_[np.asarray(left, float)[::-1], 0.0, right]
    return Valley(0.3, ratio, ratio, elevations, np.full(elevations.size, n), SPACING, elevations.size // 2)


def conveyance(*parts) -> float:
    return sum(part.area * part.hydraulic_radius ** (2 / 3) / part.mannings_n for part in parts if part.area > 0.0)


@pytest.mark.parametrize("wse", [99.0, 100.5, 102.0])
def test_the_overbanks_flow_at_the_valley_s_slope(wse: float) -> None:
    """S_v = S x sinuosity, and Manning's discharge with n and S_v is that with n / sqrt(S_v / S) and S: with the
    lines' ground the cross section's own, the splice changes only the overbanks' n, so that their conveyance at the
    channel's slope is that at the valley's, and the channel's is the same."""
    plain, split = trapezoid(), trapezoid()
    split.valley = lines(np.full(56, 100.0), np.full(56, 100.0), ratio=2.25)  # out to 300 m, as the trapezoid

    profile = hydraulic_profile(split)
    parts = compound_geometry(plain, wse=wse)
    expected = conveyance(parts.channel) * math.sqrt(SLOPE) + \
        conveyance(parts.left_overbank, parts.right_overbank) * math.sqrt(2.25 * SLOPE)

    np.testing.assert_array_equal(profile.stations, hydraulic_profile(plain).stations)
    np.testing.assert_array_equal(profile.elevations, hydraulic_profile(plain).elevations)
    assert discharge(split, SLOPE, wse=wse) == pytest.approx(expected, rel=1e-10)
    assert compound_geometry(split, wse=wse).channel == compound_geometry(plain, wse=wse).channel


@pytest.mark.parametrize(("own", "average", "sinuosity", "ratio"), [
    (0.0005, 0.0006, 1.5, 1.8), (0.001, 0.001, 1.0, 1.0), (0.0002, 0.001, 2.0, 3.0), (0.004, 0.001, 1.2, 1 / 3),
    (math.nan, 0.001, 1.4, 1.4), (0.001, math.nan, 1.4, 1.4)])
def test_the_valley_s_slope_is_the_averaged_channel_slope_times_the_sinuosity(own, average, sinuosity, ratio) -> None:
    """Over the cell's own slope, between a third and three; the sinuosity without both slopes."""
    assert slope_ratio(own, average, sinuosity) == pytest.approx(ratio)


def test_the_splice_takes_the_channel_from_the_section_and_the_overbanks_from_the_lines() -> None:
    """Between the banks, the cross section's own profile (here a carved one, its corners between the ordinates);
    a vertex at each bank on its ground; beyond, each line's ground, as far beyond the bank as along the line."""
    xs = trapezoid(half=8)
    stations = np.array([-40.0, -35.0, -30.0, -25.0, -21.0, -17.5, 0.0, 17.5, 21.0, 25.0, 30.0, 35.0, 40.0])
    xs.profile = Profile(stations, np.array([100, 100, 100, 100, 99.6, 97.5, 97.5, 97.5, 99.6, 100, 100, 100, 100.0]),
                         np.full(stations.size, N), 6)
    xs.valley = lines([101.0, 101.5, 102.0, 102.5], [99.5, 99.8, 100.0, 100.2])

    profile = hydraulic_profile(xs)

    np.testing.assert_allclose(profile.stations, [-40, -35, -30, -25, -20, -17.5, 0, 17.5, 20, 25, 30, 35, 40])
    np.testing.assert_allclose(profile.elevations, [102.5, 102, 101.5, 101, 99.0, 97.5, 97.5, 97.5, 99.0, 99.5, 99.8,
                                                    100, 100.2])
    assert profile.stations[profile.center] == 0.0


def test_the_spliced_profile_is_kept_until_what_it_came_from_changes() -> None:
    xs = trapezoid()
    xs.valley = lines(np.full(10, 100.0), np.full(10, 100.0), ratio=1.5)

    first = hydraulic_profile(xs)
    assert hydraulic_profile(xs) is first
    xs.right_bank_distance = 25.0
    moved = hydraulic_profile(xs)
    assert moved is not first and 25.0 in moved.stations
    xs.valley = lines(np.full(10, 100.0), np.full(10, 100.0), ratio=2.0)
    assert hydraulic_profile(xs) is not moved
    xs.valley = None
    np.testing.assert_array_equal(hydraulic_profile(xs).elevations, xs.elevations)


def test_water_along_the_lines_is_filled_to_its_lip() -> None:
    """The river's other bends, other streams and lakes that the lines cross are filled to the lower of the dry ground
    either side, or of the far side's where a stretch starts at the bank; dry ground is left be."""
    elevations = np.array([9999.0, 101.0, 99.0, 98.0, 100.5, 99.0, 97.0, 98.5, 99.5, 99.0, 98.0, 102.0, 103.0])
    wet = np.array([0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1], bool)  # the stream cell's at 6

    filled = fill_other_water(elevations, wet)

    np.testing.assert_array_equal(filled, [9999.0, 101.0, 100.5, 100.5, 100.5, 100.5, 97.0, 99.5, 99.5, 99.5, 99.5,
                                           102.0, 103.0])


def test_a_straight_line_s_valley_is_the_line() -> None:
    """Along a straight line the valley is the line, and the sinuosity 1, up to the reach's ends; another reach's
    cells don't count, and a reach too short for a line has none."""
    k = np.arange(30)
    rows = np.r_[k, np.full(30, 40), [60, 61]]
    cols = np.r_[k, k, [0, 1]]
    reaches = np.r_[np.full(30, 1), np.full(30, 2), [3, 3]]

    direction, sinuosity, _ = valley_axes(rows, cols, reaches, 10.0, 10.0, 5)

    np.testing.assert_allclose(direction[:30], math.pi / 4)
    np.testing.assert_allclose(direction[30:60], 0.0)
    np.testing.assert_allclose(sinuosity[:60], 1.0)
    assert np.isnan(direction[60:]).all() and np.isnan(sinuosity[60:]).all()


def sections_with_banks(widths) -> list:
    """Stand-ins for the pipeline's sections, as valley_windows reads them: NaN for one without banks."""
    out = []
    for width in widths:
        xs = SimpleNamespace(left_bank_distance=width / 2 if width == width else -1.0,
                             right_bank_distance=width / 2 if width == width else -1.0)
        out.append(SimpleNamespace(usable=True, xs=xs))
    return out


@pytest.mark.parametrize(("widths", "dir_dist", "expected"), [
    (16.0, 40, [16 * 55 / 10] * 3 + [16 * 200 / 10] * 2 + [40] * 2),  # each reach's median width (55 m), in cells
    (0.0, 40, [40] * 7),  # Valley_Dir_Dist alone
    (0.1, 40, [2.0] * 5 + [40] * 2),  # at least 2 cells
])
def test_the_window_is_a_number_of_the_reach_s_channel_widths(widths, dir_dist, expected) -> None:
    """Valley_Dir_Widths times the median bank-to-bank width of the reach's cross sections, in cells; Valley_Dir_Dist
    for a reach without banks, or with Valley_Dir_Widths 0."""
    sections = sections_with_banks([40.0, 50.0, 60.0, 200.0, 200.0, math.nan, math.nan])
    sections[0], sections[5] = None, None  # no cross section here, nor here
    configs = SimpleNamespace(valley_dir_widths=widths, valley_dir_dist=dir_dist)
    cells = SimpleNamespace(count=7, reaches=np.array([1, 1, 1, 2, 2, 3, 3]))

    windows = valley_windows(configs, SimpleNamespace(dx=10.0, dy=10.0), cells, sections)

    np.testing.assert_allclose(windows, expected)


def test_over_whole_meanders_the_valley_is_the_valley_s_axis() -> None:
    """120 cells either way along the stream, 2.4 km, is about two of the meanders: the line between its ends runs
    within a few degrees of the valley's axis, and the sinuosity is the stream's."""
    valley = synthetic_valley.meander()
    rows, cols = valley["path"].T
    middle = synthetic_valley.middle(valley, (cols + 0.5) * 10.0)

    direction, sinuosity, _ = valley_axes(rows, cols, np.ones(rows.size, int), 10.0, 10.0, 120)

    stations = along_stream_stations(rows, cols, 10.0, 10.0)
    ends = np.flatnonzero(middle)[[0, -1]]
    stream = abs(stations[ends[1]] - stations[ends[0]]) / ((cols[ends[1]] - cols[ends[0]]) * 10.0)
    assert max(turn(d, 0.0) for d in direction[middle]) < math.radians(3.0)
    assert np.median(sinuosity[middle]) == pytest.approx(stream, rel=0.01)


# --- The meandering valley against the 2D model -----------------------------------------------------------------------


def run_valley(folder: Path, valley: dict, *, valley_sections: bool, standard_step: bool,
               valley_dir_dist: int = 40, valley_dir_widths: float = 16.0) -> pipeline.Results:
    """ARC on a synthetic valley (synthetic_valley.build), with its n, its rating curves up to 1000 m^3/s."""
    path = valley["path"]
    streams = np.zeros(valley["dem"].shape, np.int32)
    streams[path[:, 0], path[:, 1]] = 1
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(32612)
    reaches = {1: {"start": tuple(path[0]), "end": tuple(path[-1]), "downstream": -1, "baseflow": 30.0,
                   "q_max": 1000.0}}
    dx = valley["dx"]
    inputs = write_inputs(folder, valley["dem"], streams, srs, (500000.0, dx, 0.0, 4000000.0, 0.0, -dx), reaches,
                          water_n=synthetic_valley.N, land_n=synthetic_valley.N)
    inputs.update({"VDT_Database_NumIterations": 60, "Gen_Slope_Dist": 25, "Standard_Step": standard_step,
                   "Valley_Sections": valley_sections, "Valley_Dir_Dist": valley_dir_dist,
                   "Valley_Dir_Widths": valley_dir_widths})
    return pipeline.run(Configs.from_mapping(inputs), quiet=True, write=False)


def errors(results: pipeline.Results, valley: dict, reference: dict) -> np.ndarray:
    """ARC's water surface less the 2D model's at each stream cell of the middle two meanders (by its rating curve),
    for each of the reference's flows."""
    wse_2d = {tuple(cell): wse for cell, wse in zip(reference["path"], reference["wse"])}
    out = []
    for k in range(results.cells.count):
        cell = (int(results.cells.rows[k]), int(results.cells.cols[k]))
        if not synthetic_valley.middle(valley, (cell[1] + 0.5) * valley["dx"]) or cell not in wse_2d:
            continue
        increments = results.curves.increments[k]
        q, wse = increments[:, 0], increments[:, 3]
        ok = np.isfinite(q) & np.isfinite(wse)
        out.append(np.interp(reference["flows"], q[ok], wse[ok]) - wse_2d[cell] if ok.sum() >= 2 else
                   np.full(len(reference["flows"]), np.nan))
    return np.array(out)


@pytest.fixture(scope="module")
def meander_errors(tmp_path_factory) -> dict:
    valley, reference = synthetic_valley.meander(), synthetic_valley.reference()
    np.testing.assert_array_equal(reference["path"], valley["path"])
    out = {}
    for step in (False, True):
        for split in (False, True):
            results = run_valley(tmp_path_factory.mktemp("meander"), valley, valley_sections=split, standard_step=step)
            out[step, split] = errors(results, valley, reference)
            out[step, "results" if split else "plain"] = results
    return out


@pytest.mark.parametrize(("standard_step", "before", "after"), [(False, 0.10, 0.02), (True, 0.05, 0.02)],
                         ids=["normal-depth", "standard-step"])
def test_over_the_floodplain_the_water_is_the_2d_model_s(meander_errors, standard_step: bool, before: float,
                                                        after: float) -> None:
    """The mean error over the floodplain, more than `before` without Valley_Sections, is under `after` with it,
    and none of the flows' median errors is more than 2 cm. In the channel little changes (under 3 cm)."""
    plain, split = meander_errors[standard_step, False], meander_errors[standard_step, True]
    assert plain.shape == split.shape and len(split) > 150

    assert np.abs(plain[:, OVERBANK_FLOWS]).mean() > before
    assert np.abs(split[:, OVERBANK_FLOWS]).mean() < after
    assert np.abs(np.median(split[:, OVERBANK_FLOWS], axis=0)).max() < 0.02
    assert np.abs(split[:, 0] - plain[:, 0]).max() < 0.03  # where ARC's water tops the banks


@pytest.mark.parametrize("standard_step", [False, True], ids=["normal-depth", "standard-step"])
def test_the_left_line_is_on_the_section_s_left(meander_errors, standard_step: bool) -> None:
    """The left line runs out on the same side of the stream as the cell's own cross section's left: the two cross
    the channel at less than a right angle."""
    results = meander_errors[standard_step, "results"]
    split = [s for s in results.sections if s is not None and s.xs.valley is not None]
    assert len(split) > 0.75 * len(results.sections)  # not the straight channel above and below the meanders
    for section in split:
        rows, cols = pipeline.valley_cells(section, 10.0, 10.0)
        own = np.array([section.rows[0] - section.row, section.cols[0] - section.col], float)
        theirs = np.array([rows[0] - section.row, cols[0] - section.col], float)
        assert own @ theirs > 0.0


def test_a_shorter_line_goes_on_to_the_end_of_the_water_at_its_end() -> None:
    """A line cut short where its ground first rises above the level goes on to the first ordinate past it that isn't
    water, so that the water there is filled to the lower of its lips, as on the line as long as the longer: the water
    from ordinate 2 to 6 (the rise at 3 in it) is filled to ordinate 7's 0.6, not ordinate 1's 1.0."""
    dem = np.ones((5, 40))
    water = np.zeros((5, 40), np.uint8)
    for k, z in ((2, 0.5), (3, 5.0), (4, 0.5), (5, 0.5), (6, 0.5), (7, 0.6)):
        dem[2, 19 - k] = z
        water[2, 19 - k] = k < 7
    lines = ((19.0, 2.0, -1.0, 0.0), (21.0, 2.0, 1.0, 0.0))
    starts, cols = np.zeros(6, np.int64), np.empty(0, np.int64)
    known = (np.empty(0), np.empty(0))

    def sample(halves):
        return valley_lines(dem, np.full(dem.shape, N), lines, 1.0, halves, 1.0, 1.0, 0.0, N, starts, cols, water,
                            known)

    z_even, n_even, centre = sample((10, 10))
    z, n, left = sample((3, 10))
    assert centre == 10 and left == 7
    np.testing.assert_array_equal(z, z_even[centre - left:])
    np.testing.assert_array_equal(n, n_even[centre - left:])
    assert z[left - 2] == 0.6


@pytest.mark.parametrize("half, uneven", [(-1, False), (-1, True), (4, False), (4, True), (500, False)])
def test_both_lines_sampled_in_one_call_are_the_three_calls_they_replace(half: int, uneven: bool) -> None:
    """valley_overbanks gives what each line's ray_reach and then valley_lines gave, bit for bit, as sample_valley
    took them: out to the farther line's first ground above the level, each to its own (uneven), or `half`, at most
    to the raster's edge."""
    rng = np.random.default_rng(2 * (half + 1) + uneven)
    dem = rng.uniform(0.0, 3.0, (60, 70))
    n_raster = rng.uniform(0.02, 0.1, dem.shape).astype(np.float32)
    water = (rng.random(dem.shape) < 0.2).astype(np.uint8)
    lines = ((30.3, 29.6, -0.8, 0.6), (33.1, 31.2, 0.8, -0.6))
    rows, cols = np.nonzero(rng.random(dem.shape) < 0.05)
    starts = np.searchsorted(rows, np.arange(dem.shape[0] + 1)).astype(np.int64)
    cols = cols.astype(np.int64)
    level = 1.7
    reaches = [ray_reach(dem, *line, 1.3, 1.0, 1.2, level) for line in lines]
    full_half = max(edge for _, edge, _ in reaches)
    if half < 0 and uneven:
        halves = tuple(max(min(above, full_half), 1) for above, _, _ in reaches)
    else:
        out_to = max(above for above, _, _ in reaches) if half < 0 else half
        halves = (max(min(out_to, full_half), 1),) * 2
    expected = valley_lines(dem, n_raster, lines, 1.3, halves, 1.0, 1.2, 0.4, 0.035, starts, cols, water,
                            tuple(ground for _, _, ground in reaches))
    got = valley_overbanks(dem, n_raster, lines, 1.3, 1.0, 1.2, level, half, uneven, 0.4, 0.035, starts, cols, water)
    assert (int(got[2]), int(got[3])) == (int(expected[2]), full_half)
    assert got[0].tobytes() == expected[0].tobytes() and got[1].tobytes() == expected[1].tobytes()


@pytest.mark.parametrize("standard_step", [False, True], ids=["normal-depth", "standard-step"])
def test_lines_each_out_to_their_own_ground_give_the_curves_of_lines_as_long_as_each_other(
        meander_errors, tmp_path: Path, monkeypatch, standard_step: bool) -> None:
    """Each line out only to its own first ground above the level (sample_valley's uneven), the shorter one made as
    long as the longer once a step works above it, gives the rating curves of both lines out as far as the longer,
    bit for bit."""
    uneven = meander_errors[standard_step, "results"]
    even_valley = pipeline.sample_valley
    made = []

    def sample_valley(*args, uneven=False, **kwargs):
        valley = even_valley(*args, **kwargs)
        made.append(valley.even)
        return valley

    monkeypatch.setattr(pipeline, "sample_valley", sample_valley)
    even = run_valley(tmp_path, synthetic_valley.meander(), valley_sections=True, standard_step=standard_step)

    assert all(made) and len(made) > 100
    taken = [s.xs.valley for s in uneven.sections if s is not None and s.xs.valley is not None]
    assert len(taken) > 100 and not all(valley.even for valley in taken)
    np.testing.assert_array_equal(even.curves.increments, uneven.curves.increments)
    np.testing.assert_array_equal(even.curves.metadata, uneven.curves.metadata)
    np.testing.assert_array_equal(even.curves.valley_slopes, uneven.curves.valley_slopes)


def test_the_lines_start_at_the_banks(meander_errors) -> None:
    """Each line's first ordinate is a spacing out from the cell's own bank, so the cross section is one line across
    the valley, kinked at the banks."""
    results = meander_errors[False, "results"]
    for section in (s for s in results.sections if s is not None and s.xs.valley is not None):
        valley, xs = section.xs.valley, section.xs
        rows, cols = pipeline.valley_cells(section, 10.0, 10.0)
        own, h = section.direction - math.pi / 2, valley.centre
        for side, bank, k in ((-1, xs.left_bank_distance, h - 1), (1, xs.right_bank_distance, h + 1)):
            bank_row = section.row + side * math.sin(own) * bank / 10.0
            bank_col = section.col + side * math.cos(own) * bank / 10.0
            gap = math.hypot(rows[k] - bank_row, cols[k] - bank_col) * 10.0
            assert gap <= valley.ordinate_distance + 10.0 * math.sqrt(0.5)  # a spacing, and rounding to a cell


def test_the_overbanks_take_the_valley_s_slope(meander_errors) -> None:
    """The channel's slope times the slope ratio is the valley's, 0.001, within 1% along the middle meanders, where
    the cells' own channel slopes times the sinuosity swing 12% either side of it."""
    results = meander_errors[False, "results"]
    valley = synthetic_valley.meander()
    ks = [k for k, s in enumerate(results.sections) if s is not None and s.xs.valley is not None
          and synthetic_valley.middle(valley, (s.col + 0.5) * 10.0)]
    overbank = np.array([results.slopes[k] * results.sections[k].xs.valley.slope_ratio for k in ks])
    local = np.array([results.slopes[k] * results.sections[k].xs.valley.sinuosity for k in ks])

    np.testing.assert_allclose(overbank, 0.001, rtol=0.01)
    assert np.ptp(local) > 0.0002


def test_the_vdt_database_gives_the_valley_s_slope(meander_errors) -> None:
    """With Valley_Sections, each cell with a rating curve has a valley slope (the VDT's Valley_Slope, for Curve2Flood's
    velocities over the floodplain): its slope times its slope ratio, or its own slope where its overbanks follow the
    stream."""
    results = meander_errors[False, "results"]
    curves = results.curves
    rated = ~np.isnan(curves.metadata[:, SLOPE_COLUMN])
    assert np.array_equal(~np.isnan(curves.valley_slopes), rated)
    for k in np.flatnonzero(rated):
        valley = results.sections[k].xs.valley
        expected = results.slopes[k] * (valley.slope_ratio if valley is not None else 1.0)
        assert curves.valley_slopes[k] == pytest.approx(expected, rel=1e-12)
    split = [k for k, s in enumerate(results.sections) if s is not None and s.xs.valley is not None and rated[k]]
    assert (curves.valley_slopes[split] > curves.metadata[split, SLOPE_COLUMN]).mean() > 0.9  # the valley is shorter
    assert "Valley_Slope" in vdt_dataframe(curves).columns
    plain = meander_errors[False, "plain"].curves  # without Valley_Sections, no column
    assert plain.valley_slopes is None and "Valley_Slope" not in vdt_dataframe(plain).columns


def test_a_straight_valley_is_left_as_it_is(tmp_path: Path) -> None:
    """A straight channel down a straight valley: no cross section turns from the valley or winds, and the rating
    curves are the same."""
    valley = synthetic_valley.build("straight", length=2000.0)

    plain = run_valley(tmp_path / "plain", valley, valley_sections=False, standard_step=False)
    split = run_valley(tmp_path / "split", valley, valley_sections=True, standard_step=False)

    assert all(s is None or s.xs.valley is None for s in split.sections)
    np.testing.assert_array_equal(split.curves.increments, plain.curves.increments)
    np.testing.assert_array_equal(split.curves.valley_slopes, split.curves.metadata[:, SLOPE_COLUMN])  # its own
