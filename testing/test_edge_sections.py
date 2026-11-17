"""Drop_Edge_Sections: a cross section whose water reaches the raster's edge loses its rating curve."""
import math
from types import SimpleNamespace

import numpy as np

from arc import pipeline
from arc.outputs.rating_curves import RatingCurves
from arc.rating_curve import WSE
from arc.xsection.sampling import OFF_RASTER_ELEVATION
from arc.xsection.valley import Valley
from arc.xsection.xsection import XSection

ROWS, COLS, CELL = 50, 21, 10.0
GRID = pipeline.Grid(None, np.zeros((ROWS, COLS), np.float32), None, np.zeros((ROWS, COLS), np.uint8), CELL, CELL)


def v_section(half=12, nodata_at=None):
    """A cross section along row 25 centred on column 10, its ground 100 m at the stream cell and rising 0.5 m a cell
    either way, with ordinates off the raster (columns below 0 and above 20) walls; a cell without data at
    nodata_at ordinates right of the centre."""
    k = np.arange(-half, half + 1)
    z = 100.0 + 0.5 * np.abs(k)
    z[(10 + k < 0) | (10 + k > COLS - 1)] = OFF_RASTER_ELEVATION
    if nodata_at is not None:
        z[half + nodata_at] = OFF_RASTER_ELEVATION
    xs = XSection(z, np.full(z.size, 0.035), CELL)
    cells = np.zeros(z.size, np.intp)
    # The stream runs down the column (pi / 2), so its cross section runs along the row
    return pipeline.Section(xs, 25, 10, math.pi / 2, 0.0, cells, cells, 1, 100.0)


def test_water_out_to_the_raster_edge_reaches_it():
    # At 106 m the water covers the ground (105 m at most) out to columns 0 and 20, and meets the walls past them
    assert pipeline.reaches_edge(v_section(), 106.0, GRID)
    # A cross section sampled only out to the raster's edge: the water covers it to its ends
    assert pipeline.reaches_edge(v_section(half=10), 106.0, GRID)


def test_water_inside_the_raster_doesnt():
    assert not pipeline.reaches_edge(v_section(), 103.0, GRID)
    assert not pipeline.reaches_edge(v_section(), 99.0, GRID)


def test_a_cell_without_data_isnt_the_raster_edge():
    # A wall 4 cells right of the stream cell, on the raster: the water stops there, but not at the raster's edge
    assert not pipeline.reaches_edge(v_section(nodata_at=4), 103.0, GRID)


def test_the_margin_reaches_in_from_the_edge():
    # At 104 m the water's edges are at columns 2 and 18, 20 m from the raster's edge
    assert pipeline.reaches_edge(v_section(), 104.0, GRID, margin=25.0)
    assert not pipeline.reaches_edge(v_section(), 104.0, GRID, margin=15.0)


def test_beyond_a_bank_the_water_runs_out_square_to_the_valley():
    section = v_section()
    section.xs.left_bank_distance = section.xs.right_bank_distance = 20.0
    # The valley runs along the row, so the lines out from the banks run down the column
    section.xs.valley = Valley(math.pi, 1.0, 1.0, np.zeros(3), np.zeros(3), CELL, 1)
    assert pipeline._station_position(section, 10.0, CELL, CELL) == (11.0, 25.0)
    assert np.allclose(pipeline._station_position(section, 50.0, CELL, CELL), (12.0, 28.0))
    assert np.allclose(pipeline._station_position(section, -50.0, CELL, CELL), (8.0, 22.0))


def test_only_the_curves_whose_water_reaches_the_edge_are_dropped():
    sections = [v_section(), v_section()]
    curves = RatingCurves.empty(2, 3)
    curves.increments[:, :, :] = 1.0
    curves.metadata[:, :] = 1.0
    curves.increments[0, :, WSE] = [101.0, 103.0, 106.0]  # its water reaches the edge at the top flow
    curves.increments[1, :, WSE] = [101.0, 102.0, 103.0]

    dropped = pipeline.drop_edge_curves(SimpleNamespace(edge_margin=0.0), GRID, sections, curves)

    assert dropped == 1
    assert np.isnan(curves.increments[0]).all() and np.isnan(curves.metadata[0]).all()
    assert np.isfinite(curves.increments[1]).all()
