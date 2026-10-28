"""Outlet_WSE_File: the water surface the standard step starts outlets from, instead of normal depth."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from osgeo import gdal

from arc import pipeline
from arc.config import Configs
from arc.outputs.rating_curves import Q, WSE
from arc.standard_step import OUTLET_REACH, _given_wse
import test_pipeline as channel
from test_pipeline import write_channel
from test_standard_step import normal_wse

RAISED = 0.5  # metres the given water stands above reach 12's normal depth


def outlet_inputs(folder: Path, *, cols_off: float = 0.0, flows=None) -> tuple[dict, int]:
    """The synthetic channel with the standard step, and an Outlet_WSE_File giving reach 12's last cell (or a point
    cols_off columns downstream of it) its normal depth raised by RAISED at its flows (or at these). Also that
    cell's column."""
    inputs = {**write_channel(folder), "Stream_Slope_Method": "reach_average", "Standard_Step": True}
    plain = pipeline.run(Configs.from_mapping(inputs), quiet=True, write=False)
    reach_12 = np.flatnonzero(plain.cells.reaches == 12)
    last = reach_12[np.argmax(plain.cells.cols[reach_12])]
    col, row = int(plain.cells.cols[last]), int(plain.cells.rows[last])
    if flows is None:
        flows = plain.curves.increments[last, :, Q]
    x0, width, _, y0, _, height = gdal.Open(inputs["DEM_File"]).GetGeoTransform()
    table = pd.DataFrame({"reach_id": 12, "x": x0 + (col + cols_off + 0.5) * width, "y": y0 + (row + 0.5) * height,
                          "flow": flows, "wse": [normal_wse(col, q) + RAISED for q in flows]})
    table.to_csv(folder / "outlet_wse.csv", index=False)
    return {**inputs, "Outlet_WSE_File": str(folder / "outlet_wse.csv")}, col


def run(inputs: dict) -> pipeline.Results:
    return pipeline.run(Configs.from_mapping(inputs), quiet=True, write=False)


def by_column(results: pipeline.Results, reach: int) -> np.ndarray:
    ks = np.flatnonzero(results.cells.reaches == reach)
    return ks[np.argsort(results.cells.cols[ks])]


def test_an_outlet_starts_from_the_water_given_for_it_and_backs_it_up(tmp_path: Path) -> None:
    """Reach 12, the outlet, starts at its last cell from the water given there, half a metre over its normal depth,
    instead of normal depth; the water above it backs up, less so further upstream, and so does reach 11's."""
    inputs, col = outlet_inputs(tmp_path)
    given = run(inputs)
    plain = run({k: v for k, v in inputs.items() if k != "Outlet_WSE_File"})

    reach_12 = by_column(given, 12)
    increments = given.curves.increments
    np.testing.assert_allclose(increments[reach_12[-1], :, WSE],
                               [normal_wse(col, q) + RAISED for q in increments[reach_12[-1], :, Q]], atol=1e-6)
    top = channel.INCREMENTS - 1
    excess = increments[reach_12, top, WSE] - plain.curves.increments[reach_12, top, WSE]
    assert excess[-1] == pytest.approx(RAISED, abs=0.01)
    assert np.all(np.diff(excess) >= -1e-6)  # never more backed up further upstream
    reach_11 = by_column(given, 11)
    assert np.all(increments[reach_11, top, WSE] > plain.curves.increments[reach_11, top, WSE])


def test_a_point_too_far_from_the_outlet_is_not_taken(tmp_path: Path) -> None:
    inputs, col = outlet_inputs(tmp_path, cols_off=OUTLET_REACH + 1)
    given = run(inputs)

    last = by_column(given, 12)[-1]
    np.testing.assert_allclose(given.curves.increments[last, :, WSE],
                               [normal_wse(col, q) for q in given.curves.increments[last, :, Q]], atol=0.001)


def test_beyond_the_given_flows_an_outlet_starts_as_far_from_normal_depth_as_at_the_nearest(tmp_path: Path) -> None:
    """Given water only up to half the maximum flow, the higher flows start at normal depth raised as much as the
    given water is at the highest flow given: so that the water rises with the flow, with no step down where the
    given water ends."""
    flows = channel.Q_MAX[12] * np.linspace(0.05, 0.5, 10)
    inputs, col = outlet_inputs(tmp_path, flows=flows)
    given = run(inputs)

    last = by_column(given, 12)[-1]
    q, wse = given.curves.increments[last, :, Q], given.curves.increments[last, :, WSE]
    inside = (q >= flows[0]) & (q <= flows[-1])
    assert inside.any() and (~inside).any()
    over = wse - np.array([normal_wse(col, flow) for flow in q])
    assert np.all(over[inside] > RAISED - 0.1)  # the given water, up to the interpolation between its flows
    np.testing.assert_allclose(over[~inside], over[inside][-1], atol=1e-6)
    assert np.all(np.diff(wse) > 0.0)


def test_the_nearest_point_within_reach_gives_the_water_by_flow() -> None:
    flows = np.array([10.0, 20.0, 30.0])
    near = (5.0, 7.0, np.array([10.0, 30.0]), np.array([1.0, 3.0]))
    nearer = (5.0, 6.0, np.array([10.0, 30.0]), np.array([2.0, 4.0]))
    far = (5.0, 5.0 + OUTLET_REACH + 1.0, np.array([10.0, 30.0]), np.array([9.0, 9.0]))

    np.testing.assert_allclose(_given_wse([near, nearer], 5.0, 5.0, flows, 30.0, 30.0), [2.0, 3.0, 4.0])
    np.testing.assert_allclose(_given_wse([near], 5.0, 5.0, np.array([5.0, 20.0, 40.0]), 30.0, 30.0),
                               [np.nan, 2.0, np.nan])
    assert np.isnan(_given_wse([far], 5.0, 0.0, flows, 30.0, 30.0)).all()


def test_the_file_needs_the_standard_step_and_its_columns(tmp_path: Path) -> None:
    inputs, _ = outlet_inputs(tmp_path)
    with pytest.raises(ValueError, match="needs Standard_Step"):
        Configs.from_mapping({**inputs, "Standard_Step": False})
    pd.read_csv(inputs["Outlet_WSE_File"]).drop(columns="wse").to_csv(inputs["Outlet_WSE_File"], index=False)
    with pytest.raises(ValueError, match="no column wse"):
        run(inputs)
