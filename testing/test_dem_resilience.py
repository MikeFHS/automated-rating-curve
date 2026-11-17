"""ARC's rating curves on one channel as DEMs of different cell sizes, alignments and noise show it, against the true
channel's.

The channel is 20 m wide at the bed, its sides sloping 2:1 up to banks 3 m high and 32 m apart, with floodplains
rising 1% beyond them and valley walls rising 5% beyond 100 m. Its n is 0.035, its bed falls 0.001 per metre, and its
maximum flow, 300 m^3/s, stands 1.07 m over the banks. Each cell of a DEM is the mean of the true ground over it, as a
DEM resampled from finer data is, and the stream is 800 m long, in two reaches. The true rating curve is the channel's
exact cross section (its corners all fall on a 0.5 m polyline's points), divided at its banks as ARC divides one
(arc.hydraulics), at the bed's slope.

What the DEM does to the water surface at each flow:
- Cell size. Coarse cells fill in the channel's corners, so the water stands higher, most at low flows: within 5% of
  a cell size of the true water surface, from cells a quarter of the bank-to-bank width (8 m: 0.21 m) to 1 m (7 mm).
- A stream at an angle to the grid, likewise. Every slope method gives the bed's slope to within 2%: the stations are
  measured along the stream raster's stair-step path smoothed (arc.xsection.stream_path), which was up to 8% longer
  than the stream and made the slopes that much too gentle.
- Noise across the stream, as from the ground beside it, moves the water at the maximum flow by a centimetre or two
  (5 cm of noise) to a few tenths of a metre (25 cm). The width-to-depth ratio still finds the banks, as it tries
  stages at least 0.1 m apart: on its first stages, a few centimetres up, the water spills from one dip of the noise
  into the next, and trying every stage, the search stopped there, with banks far inside the channel and the water at
  the maximum flow a tenth of a metre high.
- Noise along the stream changes its slope. The methods that count each rise between cells as a fall
  (local_average, local_average_corrected, reach_average) make slopes several times too steep from noise of a couple
  of centimetres (xfail). local_fit and end_points don't.
"""
from __future__ import annotations

import functools
import math
from pathlib import Path

import numpy as np
import pytest
from osgeo import osr
from test_pipeline import write_inputs

from arc import pipeline
from arc.config import Configs
from arc.hydraulics import wse_for_discharge
from arc.outputs.rating_curves import WSE, Q
from arc.xsection.xsection import XSection

SLOPE, N, MAX_FLOW = 0.001, 0.035, 300.0
BOTTOM, SIDE, BANK, FLOODPLAIN, WALLS_FROM, WALL = 20.0, 2.0, 3.0, 0.01, 100.0, 0.05
BANKS_AT = BOTTOM / 2 + SIDE * BANK  # 16 m from the centre line
LENGTH, HALF_WIDTH = 800.0, 160.0  # metres: the stream, and the DEM each side of it
SUBCELLS = 8  # points a side over which each cell's ground is averaged
SLOPE_METHODS = ["local_average", "local_average_corrected", "reach_average", "local_fit", "end_points"]
COUNTS_RISES_AS_FALLS = {"local_average", "local_average_corrected", "reach_average"}


def height(distance: np.ndarray) -> np.ndarray:
    """The true ground's height above the bed at a distance from the centre line."""
    distance = np.abs(distance)
    return (np.clip((distance - BOTTOM / 2) / SIDE, 0.0, BANK) + FLOODPLAIN * np.clip(distance - BANKS_AT, 0.0, None)
            + WALL * np.clip(distance - WALLS_FROM, 0.0, None))


@functools.cache
def _true_depth(q: float) -> float:
    spacing = 0.5
    distance = np.arange(-HALF_WIDTH, HALF_WIDTH + spacing / 2, spacing)
    xs = XSection(height(distance), np.full(distance.size, N), spacing)
    xs.left_bank_distance = xs.right_bank_distance = BANKS_AT
    return wse_for_discharge(xs, q, SLOPE)


def true_depth(q: float) -> float:
    """The true water surface's height above the bed at a flow."""
    return _true_depth(round(float(q), 9))


def write_dem(folder: Path, cell: float, *, degrees: float = 0.0, offset: float = 0.0, noise: float = 0.0,
              noisy_stream: bool = True, seed: int = 0) -> tuple[dict, callable]:
    """The channel on a DEM of cells `cell` metres wide, its centre line at `degrees` to the columns and `offset`
    cells from a cell's centre, with Gaussian noise of standard deviation `noise` metres on every cell or, without
    noisy_stream, on all but the stream's. Returns ARC's inputs, and a function giving the distance along the centre
    line from its middle to a cell's centre."""
    angle = math.radians(degrees)
    cos, sin = math.cos(angle), math.sin(angle)
    if degrees:
        rows = cols = int(round((LENGTH * cos + 2 * HALF_WIDTH * sin) / cell + 40)) | 1
    else:
        rows, cols = 2 * int(round(HALF_WIDTH / cell)) + 1, int(round(LENGTH / cell)) + 40
    middle_row, middle_col = rows // 2, cols // 2
    across = (np.arange(SUBCELLS) + 0.5) / SUBCELLS - 0.5
    r, c = np.mgrid[0:rows, 0:cols]
    dem = np.zeros((rows, cols))
    for dr in across:
        for dc in across:
            x, y = (c + dc - middle_col) * cell, (middle_row - r - dr) * cell
            dem += 100.0 - SLOPE * (x * cos + y * sin) + height(-x * sin + y * cos - offset * cell)
    dem /= SUBCELLS ** 2

    # The stream cells, stepping along the centre line
    stream = []
    for t in np.arange(-LENGTH / 2, LENGTH / 2, cell / 4):
        x, y = t * cos - offset * cell * sin, t * sin + offset * cell * cos
        here = (int(round(middle_row - y / cell)), int(round(middle_col + x / cell)))
        if not stream or stream[-1] != here:
            stream.append(here)
    if noise:
        error = np.random.default_rng(seed).normal(0.0, noise, dem.shape)
        if not noisy_stream:
            error[tuple(np.array(stream).T)] = 0.0
        dem += error
    streams = np.zeros(dem.shape, dtype=np.int32)
    half = len(stream) // 2
    for i, (row, col) in enumerate(stream):
        streams[row, col] = 1 if i < half else 2
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(32616)
    reaches = {1: {"start": stream[0], "end": stream[half - 1], "downstream": 2, "baseflow": 0.0,
                   "q_max": MAX_FLOW},
               2: {"start": stream[half], "end": stream[-1], "downstream": -1, "baseflow": 0.0, "q_max": MAX_FLOW}}
    inputs = write_inputs(folder, dem, streams, srs, (500000.0, cell, 0.0, 3800000.0, 0.0, -cell), reaches,
                          water_n=N, land_n=N)
    # 20 m each way for the stream's direction and its local slopes, whatever the cell size
    reach = max(int(round(20.0 / cell)), 2)
    inputs.update({"Gen_Dir_Dist": reach, "Gen_Slope_Dist": reach})
    along = lambda rows_, cols_: (cols_ - middle_col) * cell * cos + (middle_row - rows_) * cell * sin
    return inputs, along


def water_surface_errors(folder: Path, cell: float, slope_method: str = "local_average", **dem) -> np.ndarray:
    """ARC's water surface less the true one, at each flow of each rating curve (a row each) more than 60 m from the
    ends of the stream, where the slope methods see the whole uniform channel."""
    inputs, along = write_dem(folder, cell, **dem)
    inputs["Stream_Slope_Method"] = slope_method
    results = pipeline.run(Configs.from_mapping(inputs), quiet=True, write=False)
    distance = along(results.cells.rows, results.cells.cols)
    inside = np.abs(distance) <= LENGTH / 2 - 60.0
    increments = results.curves.increments[inside]
    assert np.isfinite(increments[:, :, WSE]).all()
    bed = 100.0 - SLOPE * distance[inside]
    truth = np.vectorize(true_depth)(increments[:, :, Q])
    return increments[:, :, WSE] - bed[:, None] - truth


def slopes(folder: Path, cell: float, slope_method: str, **dem) -> np.ndarray:
    inputs, _ = write_dem(folder, cell, **dem)
    inputs["Stream_Slope_Method"] = slope_method
    return pipeline.run(Configs.from_mapping(inputs), quiet=True, write=False).slopes


# --- Cell size and alignment ----------------------------------------------------------------------------------------


@pytest.mark.parametrize("offset", [0.0, 0.37], ids=["centred", "off centre"])
def test_the_water_surface_nears_the_true_one_as_the_cells_shrink(tmp_path: Path, offset: float) -> None:
    """Within 5% of a cell size everywhere, and closer at every halving of the cells: from 0.21 m (0.29 m with the
    centre line off the cells' centres) at 8 m cells, a quarter of the channel's width, to 7 mm (18 mm) at 1 m."""
    worst = []
    for cell in (8.0, 4.0, 2.0, 1.0):
        errors = water_surface_errors(tmp_path / f"cell_{cell}", cell, offset=offset)
        worst.append(np.abs(errors).max())
        assert worst[-1] < 0.05 * cell
    assert np.all(np.diff(worst) < 0.0)


@pytest.mark.parametrize("degrees", [22.5, 30.0, 45.0])
def test_a_stream_at_an_angle_to_the_grid_nears_the_true_water_surface_too(tmp_path: Path, degrees: float) -> None:
    """9 cells in 10 are within 5% of a cell size of the true water surface (from 0.32–0.40 m at 8 m cells), and
    closer at every halving of the cells."""
    worst = []
    for cell in (8.0, 4.0, 2.0):
        errors = water_surface_errors(tmp_path / f"cell_{cell}", cell, degrees=degrees)
        worst.append(np.percentile(np.abs(errors), 90))
        assert worst[-1] < 0.05 * cell
    assert np.all(np.diff(worst) < 0.0)


@pytest.mark.parametrize("degrees", [22.5, 30.0])
@pytest.mark.parametrize("method", SLOPE_METHODS)
def test_a_stream_at_an_angle_to_the_grid_has_the_bed_s_slope(tmp_path: Path, method: str, degrees: float) -> None:
    """At 22.5° and 30° to the grid, the stream raster's stair-step path is 1.08 and 1.07 times as long as the
    stream, which made the methods that measure along the raster find slopes that much too gentle, and the water 4 cm
    high at the maximum flow. Along the smoothed path (arc.xsection.stream_path), every method is within 2%, as
    end_points, which measures the stream's line, always was."""
    measured = slopes(tmp_path, 2.0, method, degrees=degrees)

    assert np.median(measured) == pytest.approx(SLOPE, rel=0.02)


# --- Noise ----------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(("noise", "at_the_top", "at_the_bottom"), [(0.05, 0.03, 0.05), (0.25, 0.3, 0.2)])
def test_noise_beside_the_stream_moves_the_water_surface_little(tmp_path: Path, noise: float, at_the_top: float,
                                                                at_the_bottom: float) -> None:
    """Noise everywhere but on the stream cells (whose slope stays the bed's, as on a DEM with its streams burned in):
    at 9 cells in 10, the water surface at the maximum flow is within 1.5 cm of the true one (4.07 m deep) with 5 cm of
    noise, and 0.25 m with 25 cm; at the smallest flow, 3.5 cm and 0.14 m."""
    errors = water_surface_errors(tmp_path, 2.0, noise=noise, noisy_stream=False)

    assert np.percentile(np.abs(errors[:, -1]), 90) < at_the_top
    assert np.percentile(np.abs(errors[:, 0]), 90) < at_the_bottom
    assert abs(np.median(errors[:, 0])) < 0.05


@pytest.mark.parametrize(("noise", "share"), [(0.02, 0.95), (0.05, 0.85), (0.1, 0.75)])
def test_noise_beside_the_stream_leaves_the_banks_where_they_are(tmp_path: Path, noise: float, share: float) -> None:
    """Without noise, the width-to-depth ratio finds the banks at their tops, 32 m apart. With noise of 2, 5 and 10 cm
    on the channel's bed and banks, the banks are still a median 32 m apart, and 98%, 92% and 84% of the cross
    sections' within a fifth of that. Trying every stage, the ratio rose and fell from one centimetre to the next and
    the search stopped at its first rise, far inside the channel: with 5 cm, banks a median 14 m apart."""
    inputs, _ = write_dem(tmp_path, 2.0, noise=noise, noisy_stream=False)
    results = pipeline.run(Configs.from_mapping(inputs), quiet=True, write=False)
    widths = np.array([section.banks.top_width if section.banks.valid else np.nan for section in results.sections])

    assert np.nanmedian(widths) == pytest.approx(2 * BANKS_AT, abs=1.0)
    assert np.mean(np.abs(widths - 2 * BANKS_AT) <= 0.2 * 2 * BANKS_AT) >= share


@pytest.mark.parametrize("method", [
    pytest.param(m, marks=pytest.mark.xfail(strict=True, reason="counts each rise between stream cells as a fall"))
    if m in COUNTS_RISES_AS_FALLS else m for m in SLOPE_METHODS])
def test_noise_along_the_stream_leaves_its_slope_alone(tmp_path: Path, method: str) -> None:
    """2 cm of noise on 2 m cells, on a bed falling 2 mm a cell: the median slope is within 15% of the bed's. The
    methods that count each rise between cells as a fall find 0.0019 to 0.0033, and the water at the maximum flow
    0.3 to 0.5 m too low."""
    measured = slopes(tmp_path, 2.0, method, noise=0.02, seed=1)

    assert np.median(measured) == pytest.approx(SLOPE, rel=0.15)
