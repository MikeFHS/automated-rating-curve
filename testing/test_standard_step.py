"""The standard-step method (arc.standard_step): its hydraulic tables and steps against hand-worked answers, and the
water surfaces it gives the synthetic channel (see test_pipeline.py)."""
from __future__ import annotations

import math
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import test_pipeline as channel
from osgeo import gdal
from scipy.optimize import brentq
from test_pipeline import write_channel

from arc import pipeline
from arc.config import Configs
from arc.outputs.rating_curves import WSE, Q
from arc.standard_step import (
    EXPANSION,
    G,
    HydraulicTable,
    _eddy_ends,
    flows_for,
    step_up,
    velocity_head,
)
from arc.xsection.xsection import XSection

WIDTH = 20.0  # the rectangle's bed width, in metres
N = 0.03
SLOPE = 0.001
FLOW = 30.0


def rectangle(bed: float = 100.0) -> XSection:
    """A rectangular channel 20 m wide, walled on both sides, with ordinates a metre apart."""
    offsets = np.arange(-15, 16, dtype=np.float64)
    elevations = np.where(np.abs(offsets) <= WIDTH / 2, bed, 9999.0)
    return XSection(elevations, np.full(offsets.size, N), 1.0)


def manning_depth(q: float = FLOW, slope: float = SLOPE) -> float:
    def discharge(y):
        area, perimeter = WIDTH * y, WIDTH + 2.0 * y
        return area * (area / perimeter) ** (2 / 3) * math.sqrt(slope) / N
    return brentq(lambda y: discharge(y) - q, 1e-6, 20.0, xtol=1e-12)


# --- Tables and steps -----------------------------------------------------------------------------------------------


def test_the_table_s_normal_depth_is_manning_s() -> None:
    table = HydraulicTable(rectangle(), 102.0)

    assert table.normal(FLOW, SLOPE) - table.bed == pytest.approx(manning_depth(), abs=0.005)
    assert table.normal(FLOW, SLOPE, 2.0) - table.bed == pytest.approx(manning_depth(FLOW / 2.0), abs=0.005)


def test_the_table_s_critical_depth_is_a_rectangle_s() -> None:
    """(q^2 / g b^2)^(1/3), to the table's levels, which are about 2 cm apart here."""
    table = HydraulicTable(rectangle(), 102.0)

    assert table.critical(FLOW) - table.bed == pytest.approx((FLOW ** 2 / (G * WIDTH ** 2)) ** (1 / 3), abs=0.025)


def test_uniform_flow_steps_up_at_normal_depth() -> None:
    """Between two cross sections alike but for the bed, which rises with the slope, the water at normal depth below
    is at normal depth above: the friction slope is the bed's, and the velocity heads are equal."""
    length = 50.0
    down, up = HydraulicTable(rectangle(100.0), 102.0), HydraulicTable(rectangle(100.0 + SLOPE * length), 102.0)
    depth = manning_depth()

    wse, critical = step_up(down, 100.0 + depth, FLOW, up, FLOW, length)

    assert not critical
    assert wse - up.bed == pytest.approx(depth, abs=0.005)


def test_deeper_water_below_backs_up_and_fades() -> None:
    """With the water a metre above normal depth below, the cross section above is deeper than normal, but by less."""
    length = 200.0
    down, up = HydraulicTable(rectangle(100.0), 103.0), HydraulicTable(rectangle(100.0 + SLOPE * length), 103.0)
    depth = manning_depth()

    wse, critical = step_up(down, 100.0 + depth + 1.0, FLOW, up, FLOW, length)

    assert not critical
    assert 0.0 < wse - (up.bed + depth) < 1.0


def test_water_standing_above_the_table_backs_up_beyond_it() -> None:
    """With the water below 6 m deep, above the 5 m the cross section above has a table for, the step still finds the
    water there: a little higher than below, by the friction over the 10 m between them."""
    down, up = HydraulicTable(rectangle(100.0), 104.0), HydraulicTable(rectangle(100.01), 101.0)
    assert up.levels[-1] < 106.0

    wse, critical = step_up(down, 106.0, FLOW, up, FLOW, 10.0)

    area, conveyance, _ = up.state(wse)
    friction = (FLOW / conveyance) ** 2
    assert not critical
    assert 106.0 < wse < 106.0 + 10.0 * 2.0 * friction
    assert wse == pytest.approx(106.0 + 10.0 * friction, abs=1e-4)


def test_water_too_low_below_gives_critical_depth_above() -> None:
    """Supercritical water below (5 cm deep, 30 m/s) carries more energy than any subcritical water surface above can
    balance."""
    down, up = HydraulicTable(rectangle(100.0), 102.0), HydraulicTable(rectangle(100.05), 102.0)

    wse, critical = step_up(down, 100.05, FLOW, up, FLOW, 50.0)

    assert critical
    assert wse == up.critical(FLOW)


def with_a_high_point(bed: float) -> XSection:
    """The rectangle's left wall, and on its right, beyond the bank, 10 m of floodplain 0.5 m up, a high point 1.5 m
    up and 300 m of flat 2 cm below it, n 0.06: water topping the high point spreads over the flat at once."""
    offsets = np.arange(-322, 323, dtype=np.float64)
    heights = np.select([offsets < -10, offsets <= 10, offsets <= 20, offsets == 21], [9999.0, 0.0, 0.5, 1.5], 1.48)
    xs = XSection(bed + heights, np.where(np.abs(offsets) <= 10, N, 0.06), 1.0)
    xs.left_bank_distance = xs.right_bank_distance = 10.5
    return xs


def test_where_the_energy_jumps_at_a_high_point_the_step_takes_the_side_with_energy_enough() -> None:
    """Topping the high point, the flat joins the right overbank at once: the cross section's velocity head jumps
    from 0.139 m to 0.179 m and its conveyance drops 9% (the overbank's hydraulic radius with it), so its energy jumps
    across what the water below needs, and for water below over a range of 3 cm the step's water is at the high
    point. It is on the high point's far side, with energy enough there, every time, so that the step up from it gets
    the same hydraulics. (Where the search's last guess fell decided it, before: on the near side at 3% of them,
    scattered, which a change in how closely the water surfaces were found moved, by metres on real rivers.)"""
    flow, length = 60.0, 30.0
    down, up = HydraulicTable(with_a_high_point(100.0), 103.0), HydraulicTable(with_a_high_point(100.03), 103.0)
    high_point = 101.53
    heads = [velocity_head(up, high_point + side, flow) for side in (-1e-9, 1e-9)]
    steps = np.array([step_up(down, wse, flow, up, flow, length)[0] for wse in np.linspace(101.47, 101.50, 301)])
    there = np.abs(steps - high_point) < 2e-6

    assert heads[0] == pytest.approx(0.139, abs=0.001) and heads[1] == pytest.approx(0.179, abs=0.001)
    assert there.all()
    assert np.all(steps > high_point)
    np.testing.assert_allclose([velocity_head(up, wse, flow) for wse in steps], heads[1], rtol=1e-5)


def narrower(width: float, bed: float) -> XSection:
    offsets = np.arange(-15, 16, dtype=np.float64)
    return XSection(np.where(np.abs(offsets) <= width / 2, bed, 9999.0), np.full(offsets.size, N), 1.0)


def test_by_default_the_eddy_loss_is_on_the_change_from_the_cross_section_below() -> None:
    """A narrowing going upstream: the flow slows going downstream, an expansion."""
    down, up = HydraulicTable(rectangle(100.0), 103.0), HydraulicTable(narrower(12.0, 100.05), 103.0)
    head_down = velocity_head(down, 101.2, FLOW)

    assert step_up(down, 101.2, FLOW, up, FLOW, 50.0) == step_up(down, 101.2, FLOW, up, FLOW, 50.0,
                                                                 eddy_head=head_down, eddy_steps=1)


@pytest.mark.parametrize("steps", [1, 4])
def test_the_eddy_loss_is_the_step_s_share_of_the_change_over_its_window(steps: int) -> None:
    """The energy balances with C |h_up - h_far| / steps for the eddy loss: here the flow slows going downstream to
    h_far, so C is the expansion's."""
    down, up = HydraulicTable(rectangle(100.0), 103.0), HydraulicTable(narrower(12.0, 100.05), 103.0)
    far_head, length = 0.001, 50.0

    wse, critical = step_up(down, 101.2, FLOW, up, FLOW, length, eddy_head=far_head, eddy_steps=steps)

    head_up, head_down = velocity_head(up, wse, FLOW), velocity_head(down, 101.2, FLOW)
    friction = (2 * FLOW / (down.state(101.2)[1] + up.state(wse)[1])) ** 2
    assert not critical and head_up > far_head
    assert wse + head_up == pytest.approx(101.2 + head_down + length * friction
                                          + EXPANSION * (head_up - far_head) / steps, abs=1e-6)


def test_each_eddy_window_ends_at_the_first_cross_section_far_enough_below() -> None:
    stations = np.array([0.0, 30.0, 60.0, 90.0, 120.0, 150.0])

    np.testing.assert_array_equal(_eddy_ends(stations, 100.0), [4, 5, 5, 5, 5, 5])
    np.testing.assert_array_equal(_eddy_ends(stations, 0.0), [1, 2, 3, 4, 5, 5])


def test_the_flows_are_shares_of_the_maximum_and_the_baseflow_is_one_of_them() -> None:
    """The first flow at or above the baseflow is set just below it, whichever it is; none is above the maximum flow,
    and a baseflow just over a flow below it leaves the flows as they are rather than out of order."""
    np.testing.assert_allclose(flows_for(10.0, 0.0, 5), [2.0, 4.0, 6.0, 8.0, 10.0])
    np.testing.assert_allclose(flows_for(10.0, 1.5, 5), [1.499, 4.0, 6.0, 8.0, 10.0])
    np.testing.assert_allclose(flows_for(10.0, 3.0, 5), [2.0, 2.999, 6.0, 8.0, 10.0])
    np.testing.assert_allclose(flows_for(10.0, 8.0, 5), [2.0, 4.0, 6.0, 7.999, 10.0])
    np.testing.assert_allclose(flows_for(10.0, 12.0, 5), [2.0, 4.0, 6.0, 8.0, 10.0])
    np.testing.assert_allclose(flows_for(10.0, 2.0005, 5), [2.0, 4.0, 6.0, 8.0, 10.0])


# --- The synthetic channel ------------------------------------------------------------------------------------------


def run_channel(folder: Path) -> pipeline.Results:
    # One slope for each reach, the bed's, so that the water is at normal depth where nothing backs it up
    inputs = {**write_channel(folder), "Stream_Slope_Method": "reach_average", "Standard_Step": True}
    return pipeline.run(Configs.from_mapping(inputs), quiet=True, write=False)


def normal_wse(col: int, q: float) -> float:
    """The synthetic channel's normal-depth water surface at a column."""
    return channel.BED - channel.SLOPE * col + channel.normal_depth(q)


def test_with_nothing_backing_it_up_the_water_is_at_normal_depth(tmp_path: Path) -> None:
    """Reach 12 is the outlet, and its cross sections and slope are all alike: uniform flow. Its rating curves are
    rewritten at tenths of its maximum flow, the second set just below its baseflow, with the water at normal
    depth."""
    results = run_channel(tmp_path)
    flows = channel.Q_MAX[12] * np.arange(1, channel.INCREMENTS + 1) / channel.INCREMENTS
    flows[1] = channel.BASEFLOW[12] - 0.001

    for k in np.flatnonzero(results.cells.reaches == 12):
        increments = results.curves.increments[k]
        np.testing.assert_allclose(increments[:, Q], flows)
        col = int(results.cells.cols[k])
        np.testing.assert_allclose(increments[:, WSE], [normal_wse(col, q) for q in flows], atol=0.001)


def test_the_reach_below_backs_water_up_the_reach_above(tmp_path: Path) -> None:
    """Reach 12 carries more than reach 11 at the same share of its maximum flow, so its water is deeper where they
    meet. Reach 11's last cross section starts there, and the water above it falls towards its own normal depth."""
    results = run_channel(tmp_path)
    cols, reaches, increments = results.cells.cols, results.cells.reaches, results.curves.increments
    reach_11 = np.flatnonzero(reaches == 11)
    reach_11 = reach_11[np.argsort(cols[reach_11])]
    first_of_12 = np.flatnonzero(reaches == 12)[np.argmin(cols[reaches == 12])]
    top = channel.INCREMENTS - 1

    assert increments[reach_11[-1], top, WSE] == pytest.approx(increments[first_of_12, top, WSE])
    excess = np.array([increments[k, top, WSE] - normal_wse(int(cols[k]), channel.Q_MAX[11]) for k in reach_11])
    assert excess[-1] > 0.02
    assert np.all(np.diff(excess) >= -1e-6)  # never more above normal depth further upstream


SILL = 0.5  # metres: how far the sill at reach 11's foot stands above the bed's even fall


def sill_bed(distance: np.ndarray) -> np.ndarray:
    """The channel's bed falling evenly but for a sill over reach 11's last 15 cells, columns 105 to 119."""
    return channel.BED - channel.SLOPE * distance + np.where((distance >= 105.0) & (distance < 120.0), SILL, 0.0)


@pytest.mark.parametrize("reach_12_on_raster", [True, False])
def test_a_reach_whose_neighbours_have_no_flows_is_marched_from_its_downstream_end(tmp_path: Path,
                                                                                  reach_12_on_raster: bool) -> None:
    """Reach 11 alone has flows, and the sill at its foot puts its last ten cross sections higher than its first ten,
    which would turn it round. Reach 12 on the stream raster, though it has no flows, or else the end of reach 11's
    line, says which end is downstream: the march starts at the sill, which backs the water up the reach above it, and
    the water falls all the way down the reach."""
    inputs = {**write_channel(tmp_path, bed=sill_bed), "Stream_Slope_Method": "reach_average", "Standard_Step": True}
    flows = pd.read_csv(inputs["Flow_File"])
    flows[flows.COMID == 11].to_csv(inputs["Flow_File"], index=False)
    if not reach_12_on_raster:
        streams = gdal.Open(inputs["Stream_File"], gdal.GA_Update)
        values = streams.GetRasterBand(1).ReadAsArray()
        streams.GetRasterBand(1).WriteArray(np.where(values == 12, 0, values))
        streams = None

    results = pipeline.run(Configs.from_mapping(inputs), quiet=True, write=False)

    ks = np.flatnonzero(results.cells.reaches == 11)
    ks = ks[np.argsort(results.cells.cols[ks])]
    wse = results.curves.increments[ks, :, WSE]
    top, first_col = channel.INCREMENTS - 1, int(results.cells.cols[ks[0]])
    assert np.all(wse[0] > wse[-1])
    assert wse[0, top] > normal_wse(first_col, channel.Q_MAX[11]) + 0.1


@pytest.mark.parametrize("q_max_12", [channel.Q_MAX[12], 0.6 * channel.Q_MAX[11]])
def test_a_reach_above_one_without_flows_is_an_outlet_no_higher_than_the_water_below(tmp_path: Path,
                                                                                   q_max_12: float) -> None:
    """Reach 13, columns 110 to 119, has no flows and lies between reaches 11 and 12. Reach 11 starts at its own
    normal depth, as an outlet does, though reach 12 carrying more water stands higher. But carrying less, reach 12
    stands lower, and reach 11's water then starts no higher than reach 12's risen by its slope over the 11 m between
    their cells."""
    inputs = {**write_channel(tmp_path), "Stream_Slope_Method": "reach_average", "Standard_Step": True}
    streams = gdal.Open(inputs["Stream_File"], gdal.GA_Update)
    values = streams.GetRasterBand(1).ReadAsArray()
    values[channel.CENTER_ROW, 110:120] = 13
    streams.GetRasterBand(1).WriteArray(values)
    streams = None
    layer = gpd.read_file(inputs["StrmShp_File"])
    gap = layer[layer.LINKNO == 11].assign(LINKNO=13, COMID=13, DSLINKNO=12)
    layer.loc[layer.LINKNO == 11, "DSLINKNO"] = 13
    pd.concat([layer, gap]).to_file(inputs["StrmShp_File"])
    flows = pd.read_csv(inputs["Flow_File"])
    flows.loc[flows.COMID == 12, "qmax"] = q_max_12
    flows.to_csv(inputs["Flow_File"], index=False)

    results = pipeline.run(Configs.from_mapping(inputs), quiet=True, write=False)

    cols, reaches, increments = results.cells.cols, results.cells.reaches, results.curves.increments
    last_of_11 = np.flatnonzero(reaches == 11)[np.argmax(cols[reaches == 11])]
    first_of_12 = np.flatnonzero(reaches == 12)[np.argmin(cols[reaches == 12])]
    assert (cols[last_of_11], cols[first_of_12]) == (109, 120)
    normal = np.array([normal_wse(109, q) for q in increments[last_of_11, :, Q]])
    below = increments[first_of_12, :, WSE] + results.slopes[last_of_11] * 11.0
    np.testing.assert_allclose(increments[last_of_11, :, WSE], np.minimum(normal, below), atol=0.001)
    assert np.all(below > normal) if q_max_12 > channel.Q_MAX[11] else np.all(below < normal)


def held_and_unheld(folder: Path, monkeypatch: pytest.MonkeyPatch, **extra) -> tuple:
    """The synthetic channel carved for its baseflow below its banks, with the standard step: the run as it is, and
    the same run with the step not given the baseflow levels."""
    inputs = {**write_channel(folder), "Stream_Slope_Method": "reach_average", "Standard_Step": True,
              "BATHY_Out_File": str(folder / "bathy.tif"), "Bathy_Use_Banks": True, **extra}
    held = pipeline.run(Configs.from_mapping(inputs), quiet=True, write=False)
    step = pipeline.apply_standard_step
    monkeypatch.setattr(pipeline, "apply_standard_step",
                        lambda *args, **kwargs: step(*args, **{**kwargs, "baseflow_levels": None}))
    unheld = pipeline.run(Configs.from_mapping(inputs), quiet=True, write=False)
    return held, unheld


def test_up_to_its_baseflow_the_water_is_held_at_the_level_its_channel_was_carved_for(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The channels carved without the slope_adjustment_factor of 0.5 that quarters the step's conveyance, too
    shallow for it, so that the step lifts the baseflow over the banks, as a channel carved for a steeper slope than
    its banks fall does on real rivers. Held, the water at every flow up to the baseflow, the baseflow itself among
    them, is at most at the bank elevation; the flows above it are the step's own."""
    depth = pipeline.bathymetry_depth
    monkeypatch.setattr(pipeline, "bathymetry_depth",
                        lambda *args, **kwargs: depth(*args, **{**kwargs, "slope_factor": 1.0}))
    held, unheld = held_and_unheld(tmp_path, monkeypatch, slope_adjustment_factor=0.5)

    over = []
    for k in np.flatnonzero(np.isfinite(held.curves.increments[:, -1, Q])):
        section, baseflow = held.sections[k], channel.BASEFLOW[int(held.cells.comids[k])]
        level = section.baseflow_level
        assert level == section.bank_elevation
        q, wse = held.curves.increments[k, :, Q], held.curves.increments[k, :, WSE]
        at_baseflow = np.isclose(q, baseflow - 0.001, rtol=0.0, atol=1e-9)
        assert at_baseflow.sum() == 1
        assert np.all(wse[q <= baseflow] <= level + 1e-9)
        assert np.all(np.diff(wse) >= 0.0)
        above = q > baseflow
        np.testing.assert_allclose(wse[above], unheld.curves.increments[k, above, WSE], atol=1e-9)
        node = int(np.flatnonzero(at_baseflow)[0])
        over.append(unheld.curves.increments[k, node, WSE] - level)
    assert max(over) > 0.1  # it would have stood over the banks


def test_without_bathymetry_nothing_is_held(tmp_path: Path) -> None:
    results = run_channel(tmp_path)

    assert all(math.isnan(s.baseflow_level) for s in results.sections if s is not None)


def test_the_standard_step_is_off_by_default_and_needs_the_stream_network() -> None:
    inputs = {"DEM_File": "dem.tif", "Flow_File": "flows.csv", "Flow_File_ID": "COMID", "Flow_File_QMax": "qmax"}

    assert Configs.from_mapping(inputs).standard_step is False
    with pytest.raises(ValueError, match="standard step needs the stream network"):
        Configs.from_mapping({**inputs, "Standard_Step": True})
