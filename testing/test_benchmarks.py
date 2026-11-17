"""ARC against textbook answers: the steady-flow tests of HEC-RAS's verification report (USACE, 2018, RD-52) whose
answers are exact. Chow's (1959) trapezoidal channel (test 2.2.1): its normal and critical depths, and its water
surface backed up behind a lake (an M1 profile) and drawn down to critical depth above a steep reach (an M2 profile).
Flow over a bump without friction (test 2.2.2), and subcritical flow over a variable sloping bed (test 4.2.2), each
in its own section below. test_shallow_water.py runs RD-52's 2D tests on the 2D model of the valley tests.

The channel is 20 ft wide at the bed, its sides slope 2:1, n is 0.025, the bed slope 0.0016 and the flow 400 ft^3/s.
In RD-52 a lake holds the water 5.0 ft above the bed at the downstream end, and the cross sections are 25 ft apart for
3000 ft. Here the channel's banks are 8 ft high, with floodplains beyond them. The channel holds the maximum flow,
400 ft^3/s, so ARC's banks are that water's edges (a bank above it could be a valley wall), and the water at normal
depth or drawn down below it is all channel; behind the lake, the water backed up above it is divided there (the DEM
lake test is a strict xfail). ARC works in metres, and the results are compared in feet.

The reference profiles are the gradually varied flow equation, worked out exactly (to 1e-11):

    dy/dx = (S0 - Sf) / (1 - a Q^2 T / (g A^3))

with Sf Manning's friction slope at the depth y, and x along the stream. The standard step's eddy loss, C times the
change in velocity head from one cross section to the next, makes the velocity-head coefficient a 1 - C where the water
slows going downstream (behind the lake, C = 0.3), so the same equation with a = 0.7 is the reference with it. Where
the water speeds up (the drawdown, C = 0.1), the step reckons the loss at the water surface without it, which makes it
C' = C (1 - Fr^2) / (1 - (1 - C) Fr^2) of the gain in velocity head (arc.standard_step), so that the reference has
1 - (1 + C') Q^2 T / (g A^3) below the line, falling to nothing at critical depth as without the loss. RD-52 turned the
eddy loss off (a = 1); spreading it over 10^12 steps (step_up's eddy_steps) does the same here.

The DEM runs (arc.pipeline) put the channel on 1 ft cells, the trapezoid's corners on their centres, so its cross
sections are the trapezoid exactly. A second reach below holds the water: one whose normal depth is 5.0 ft (the lake),
or one too steep for subcritical flow.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
from osgeo import osr
from scipy.integrate import quad, solve_ivp
from scipy.optimize import brentq
from test_pipeline import write_inputs

from arc import pipeline
from arc.config import Configs
from arc.hydraulics import hydraulic_geometry, wse_for_discharge
from arc.outputs.rating_curves import WSE
from arc.outputs.rating_curves import Q as Q_COLUMN
from arc.standard_step import G, HydraulicTable, step_up, velocity_head
from arc.xsection.xsection import XSection

FT = 0.3048  # metres in a foot
BOTTOM, SIDE, SLOPE, N, FLOW = 20.0 * FT, 2.0, 0.0016, 0.025, 400.0 * FT ** 3
LAKE_DEPTH = 5.0 * FT
SPACING, LENGTH = 25.0 * FT, 3000.0 * FT  # RD-52's cross sections
NO_EDDY_LOSS = 10 ** 12  # eddy_steps that spread the eddy loss to nothing


def area(y: float) -> float:
    return (BOTTOM + SIDE * y) * y


def top_width(y: float) -> float:
    return BOTTOM + 2.0 * SIDE * y


def manning_flow(y: float, slope: float = SLOPE) -> float:
    perimeter = BOTTOM + 2.0 * y * math.sqrt(1.0 + SIDE ** 2)
    return area(y) * (area(y) / perimeter) ** (2 / 3) * math.sqrt(slope) / N


def friction_slope(y: float) -> float:
    return SLOPE * (FLOW / manning_flow(y)) ** 2


def froude_squared(y: float, alpha: float = 1.0) -> float:
    return alpha * FLOW ** 2 * top_width(y) / (G * area(y) ** 3)


NORMAL_DEPTH = brentq(lambda y: manning_flow(y) - FLOW, 1e-3, 10.0, xtol=1e-14)
CRITICAL_DEPTH = brentq(lambda y: froude_squared(y) - 1.0, 1e-3, 10.0, xtol=1e-14)


def backwater(x: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """The depths x metres upstream of the lake (M1)."""
    def rise(_, y):
        return [-(SLOPE - friction_slope(y[0])) / (1.0 - froude_squared(y[0], alpha))]
    return solve_ivp(rise, (0.0, float(x[-1])), [LAKE_DEPTH], t_eval=x, method="DOP853", rtol=1e-11, atol=1e-12).y[0]


def tapered(froude_squared: float, contraction: float) -> float:
    """The share of the gain in velocity head the standard step's contraction loss C takes where the water speeds up
    going downstream, the loss reckoned at the water surface without it (see the notes above)."""
    return contraction * (1.0 - froude_squared) / (1.0 - (1.0 - contraction) * froude_squared)


def drawdown(x: np.ndarray, contraction: float = 0.0) -> np.ndarray:
    """The depths x metres upstream of a break to a steep slope, where the water passes through critical depth (M2):
    the distance to each depth, integrated up from critical depth, where it is finite, towards normal depth, which it
    only nears. contraction is the standard step's contraction loss coefficient, 0 for none."""
    critical = CRITICAL_DEPTH
    span = NORMAL_DEPTH - critical
    # Closer together where the depth nears normal depth, and only to within 1e-5 of the way: nearer, the friction
    # slope's difference from the bed's is lost to rounding
    depths = np.concatenate([critical + span * np.linspace(0.0, 0.999, 1000),
                             NORMAL_DEPTH - span * np.geomspace(1e-3, 1e-5, 60)])
    run = lambda y: (1.0 - (1.0 + tapered(froude_squared(y), contraction)) * froude_squared(y)) \
        / (friction_slope(y) - SLOPE)
    steps = [quad(run, a, b, epsabs=1e-10, epsrel=1e-10)[0] for a, b in zip(depths[:-1], depths[1:])]
    return np.interp(x, np.cumsum([0.0] + steps), depths)


def ground(offset_ft: np.ndarray) -> np.ndarray:
    """The channel's height above its bed, in metres, at a distance in feet from its centre line."""
    offset = np.abs(offset_ft)
    return FT * np.where(offset <= 26.0, np.maximum(offset - 10.0, 0.0) / SIDE, 8.0 + 0.01 * (offset - 26.0))


def chow_section(bed: float) -> XSection:
    """The channel's cross section, with points a foot apart."""
    offsets = np.arange(-45.0, 46.0)
    return XSection(bed + ground(offsets), np.full(offsets.size, N), FT)


def march(tables: list[HydraulicTable], first_wse: float, flow: float, spacing: float, *,
          eddy_loss: bool) -> tuple[np.ndarray, np.ndarray]:
    """The water surfaces at cross sections `spacing` metres apart (their tables, the first at the downstream end),
    from the first's, first_wse, upstream by step_up, and whether each is critical depth for want of a subcritical
    one."""
    wse, critical = [first_wse], [False]
    for down, up in zip(tables[:-1], tables[1:]):
        no_loss = {} if eddy_loss else {"eddy_head": velocity_head(down, wse[-1], flow), "eddy_steps": NO_EDDY_LOSS}
        w, at_critical = step_up(down, wse[-1], flow, up, flow, spacing, **no_loss)
        wse.append(w)
        critical.append(at_critical)
    return np.array(wse), np.array(critical)


def march_up(first_wse, *, eddy_loss: bool) -> tuple[np.ndarray, np.ndarray]:
    """The depths at RD-52's cross sections, from the first, at the downstream end, whose water surface first_wse
    gives from its table, upstream by step_up."""
    x = np.arange(0.0, LENGTH + SPACING / 2, SPACING)
    tables = [HydraulicTable(chow_section(SLOPE * d), SLOPE * d + 6.0 * FT) for d in x]
    wse, _ = march(tables, first_wse(tables[0]), FLOW, SPACING, eddy_loss=eddy_loss)
    return x, wse - SLOPE * x


# --- The cross section ----------------------------------------------------------------------------------------------


def test_chow_s_channel_carries_its_flow_at_its_textbook_normal_depth() -> None:
    """3.36 ft deep at 4.45 ft/s (Chow, 1959, Example 10.1)."""
    xs = chow_section(0.0)
    wse = wse_for_discharge(xs, FLOW, SLOPE)

    assert wse == pytest.approx(NORMAL_DEPTH, abs=1e-9)
    assert round(wse / FT, 2) == 3.36
    assert round(FLOW / hydraulic_geometry(xs, wse=wse).area / FT, 2) == 4.45


def test_chow_s_channel_s_critical_depth_is_within_a_level_of_its_table() -> None:
    """The standard step's critical depth is its table's level of least specific energy: 2.15 ft, to the levels'
    spacing."""
    table = HydraulicTable(chow_section(0.0), 6.0 * FT)

    assert round(CRITICAL_DEPTH / FT, 2) == 2.15
    assert table.critical(FLOW) == pytest.approx(CRITICAL_DEPTH, abs=table.levels[1] - table.levels[0])


# --- The standard step ----------------------------------------------------------------------------------------------


@pytest.mark.parametrize("eddy_loss", [False, True], ids=["as RD-52", "with the eddy loss"])
def test_the_standard_step_gives_chow_s_backwater_profile(eddy_loss: bool) -> None:
    """RD-52's test 2.2.1: from 5.0 ft at the lake, the water falls towards normal depth upstream. HEC-RAS came within
    0.01 ft of Chow's hand calculation; the standard step comes within 0.001 ft of the exact profile."""
    x, depths = march_up(lambda table: table.bed + LAKE_DEPTH, eddy_loss=eddy_loss)
    reference = backwater(x, 0.7 if eddy_loss else 1.0)

    np.testing.assert_allclose(depths / FT, reference / FT, atol=0.001)
    assert np.all(np.diff(depths) < 0.0)
    assert depths[-1] / FT == pytest.approx(NORMAL_DEPTH / FT, abs=0.02)  # 3000 ft up, all but at normal depth


def test_the_standard_step_draws_the_water_down_to_critical_depth_above_a_steep_reach() -> None:
    """With critical depth at the break, the water rises upstream towards normal depth, never reaching it: within
    0.03 ft of the exact profile in the 200 ft where it rises steeply, and 0.01 ft beyond, with RD-52's cross
    sections. The step starts at its table's critical depth, which is within a level of the exact one."""
    x, depths = march_up(lambda table: table.critical(FLOW), eddy_loss=False)
    error = np.abs(depths - drawdown(x)) / FT

    assert depths[0] / FT == pytest.approx(CRITICAL_DEPTH / FT, abs=0.06)
    assert error[x < 200.0 * FT].max() < 0.03 and error[x >= 200.0 * FT].max() < 0.01
    assert np.all(np.diff(depths) > 0.0) and depths.max() < NORMAL_DEPTH


# --- On a DEM -------------------------------------------------------------------------------------------------------

HALF_WIDTH, MARGIN, CELLS_BELOW = 45, 20, 400  # cells: the raster's rows each side of the stream, its margins
CHANNEL = int(round(LENGTH / FT))  # the channel's cells, one per foot
STEEP_SLOPE = 0.01  # the steep reach's: its normal depth, 2.0 ft, is below critical depth


def write_chow_channel(folder: Path, below: str) -> dict:
    """The channel on 1 ft cells, 3000 ft long (reach 1), and below it 400 ft of the same channel at another slope
    (reach 2): one whose normal depth is 5.0 ft, the lake (below="lake"), or a steep one ("steep"). The bed is level
    where they meet, 100 m up."""
    lower_slope = brentq(lambda s: manning_flow(LAKE_DEPTH, s) - FLOW, 1e-6, 0.1) if below == "lake" else STEEP_SLOPE
    rows, cols = np.mgrid[0:2 * HALF_WIDTH + 1, 0:MARGIN + CHANNEL + CELLS_BELOW + MARGIN]
    meet = MARGIN + CHANNEL  # reach 2's first column
    feet_down = cols - (meet - 1)  # along the stream from reach 1's last cell
    bed = 100.0 - np.where(feet_down <= 0, SLOPE, lower_slope) * feet_down * FT
    dem = bed + ground(rows - HALF_WIDTH)
    streams = np.zeros(dem.shape, dtype=np.int32)
    streams[HALF_WIDTH, MARGIN:meet] = 1
    streams[HALF_WIDTH, meet:meet + CELLS_BELOW] = 2
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(32616)
    reaches = {1: {"start": (HALF_WIDTH, MARGIN), "end": (HALF_WIDTH, meet - 1), "downstream": 2, "baseflow": 0.0,
                   "q_max": FLOW},
               2: {"start": (HALF_WIDTH, meet), "end": (HALF_WIDTH, meet + CELLS_BELOW - 1), "downstream": -1,
                   "baseflow": 0.0, "q_max": FLOW}}
    inputs = write_inputs(folder, dem, streams, srs, (500000.0, FT, 0.0, 3800000.0, 0.0, -FT), reaches, water_n=N,
                          land_n=N)
    return {**inputs, "Stream_Slope_Method": "reach_average"}  # each reach's own, uniform slope


def depths_at_the_maximum_flow(results: pipeline.Results, reach: int, *, last: int = MARGIN + CHANNEL - 1,
                               cell: float = FT, flow: float = FLOW) -> tuple[np.ndarray, np.ndarray]:
    """A reach's cells' distances in metres upstream of reach 1's last cell (in column `last`, on cells `cell`
    metres wide), in order, and the water's depth at each at the top of its rating curve, which carries `flow`."""
    ks = np.flatnonzero(results.cells.comids == reach)
    x = (last - results.cells.cols[ks]) * cell
    order = np.argsort(x)
    ks, x = ks[order], x[order]
    assert np.allclose(results.curves.increments[ks, -1, Q_COLUMN], flow, rtol=1e-9)
    beds = np.array([results.sections[k].dem_low_point for k in ks])
    return x, results.curves.increments[ks, -1, WSE] - beds


def test_on_a_dem_each_cross_section_alone_is_at_normal_depth(tmp_path: Path) -> None:
    results = pipeline.run(Configs.from_mapping(write_chow_channel(tmp_path, "lake")), quiet=True, write=False)
    _, depths = depths_at_the_maximum_flow(results, 1)

    assert depths.size == CHANNEL
    np.testing.assert_allclose(depths, NORMAL_DEPTH, atol=1e-6)


@pytest.mark.xfail(strict=True, reason="The banks are no higher than the maximum flow's normal-depth water, 3.4 ft "
                   "up in this 8 ft channel, so the hydraulics divide the lake's backwater above it there: up to 0.06 "
                   "ft low (docs/limitations.md). test_the_standard_step_gives_chow_s_backwater_profile checks the "
                   "step itself.")
def test_on_a_dem_the_standard_step_backs_the_water_up_behind_the_lake(tmp_path: Path) -> None:
    """Reach 1 starts from the lake's water, 5.0 ft above the lake's bed a foot downstream, and is then within
    0.01 ft of the exact profile. The eddy loss over each step is reckoned over the next 300 m downstream
    (Standard_Step_Eddy_Length), not on the change from the cross section below, so it isn't exactly the
    reference's."""
    inputs = {**write_chow_channel(tmp_path, "lake"), "Standard_Step": True}
    results = pipeline.run(Configs.from_mapping(inputs), quiet=True, write=False)
    x, depths = depths_at_the_maximum_flow(results, 1)
    _, lake = depths_at_the_maximum_flow(results, 2)

    np.testing.assert_allclose(lake, LAKE_DEPTH, atol=1e-5)
    assert depths[0] / FT == pytest.approx(5.0, abs=0.001)
    np.testing.assert_allclose(depths / FT, backwater(x, 0.7) / FT, atol=0.01)


@pytest.fixture(scope="module")
def drawdown_inputs(tmp_path_factory: pytest.TempPathFactory) -> dict:
    return {**write_chow_channel(tmp_path_factory.mktemp("drawdown"), "steep"), "Standard_Step": True}


def test_on_a_dem_a_steep_reach_is_at_critical_depth(drawdown_inputs: dict) -> None:
    """The standard step is subcritical: a reach too steep for subcritical flow is at critical depth (2.15 ft), to the
    table's levels, where the textbook has it at its supercritical normal depth (2.0 ft)."""
    results = pipeline.run(Configs.from_mapping(drawdown_inputs), quiet=True, write=False)
    _, steep = depths_at_the_maximum_flow(results, 2)

    assert brentq(lambda y: manning_flow(y, STEEP_SLOPE) - FLOW, 1e-3, 10.0) < CRITICAL_DEPTH
    np.testing.assert_allclose(steep / FT, CRITICAL_DEPTH / FT, atol=0.05)


def test_on_a_dem_the_standard_step_draws_the_water_down_to_the_steep_reach(drawdown_inputs: dict) -> None:
    """With the eddy loss on each step's change in velocity head (Standard_Step_Eddy_Length 0), the profile is within
    0.005 ft of the exact one beyond the first 20 ft, and 0.03 ft in them, where the step starts from its table's
    critical depth, and rises all the way up. As ARC runs it by default, the eddy loss is reckoned over the next 300 m
    downstream: the drawdown's loss is spread over the 300 m above it, which raises the water there, up to 0.1 ft
    above the exact profile and 0.01 ft above normal depth, falling back towards normal depth further up."""
    results = pipeline.run(Configs.from_mapping(drawdown_inputs), quiet=True, write=False)
    x, as_run = depths_at_the_maximum_flow(results, 1)
    stepwise_inputs = {**drawdown_inputs, "Standard_Step_Eddy_Length": 0}
    stepwise = depths_at_the_maximum_flow(pipeline.run(Configs.from_mapping(stepwise_inputs), quiet=True,
                                                       write=False), 1)[1]
    reference = drawdown(x, 0.1)

    beyond = x >= 20.0 * FT
    assert np.abs(stepwise - reference)[beyond].max() / FT < 0.005
    assert np.abs(stepwise - reference)[~beyond].max() / FT < 0.03
    assert np.all(np.diff(stepwise) >= 0.0) and stepwise.max() < NORMAL_DEPTH
    assert np.abs(as_run - reference).max() / FT < 0.1
    assert NORMAL_DEPTH < as_run.max() < NORMAL_DEPTH + 0.01 * FT
    for depths in (stepwise, as_run):
        assert depths[0] / FT == pytest.approx(CRITICAL_DEPTH / FT, abs=0.05)


# --- Flow over a bump without friction (RD-52 test 2.2.2) -----------------------------------------------------------
#
# A channel 0.3 m wide, its bed flat but for a bump 0.2 m high between 8 and 12 m (Goutal & Maurel, 1997), carries
# 0.054 m^3/s without friction (n 0.0001) to a tailwater 0.33 m deep at 20.6 m. The water passes through critical
# depth at the crest, runs down the bump supercritical, and jumps up to the tailwater's level. ARC's cross sections are
# RD-52's, 0.2 m apart; their walls rise 0.6 m over the next ordinate, 1 cm out, and the exact water surface is for
# that shape: Bernoulli's equation, from critical depth at the crest upstream and down the bump on the supercritical
# side, and from the tailwater up to the jump, which is where the two sides' specific forces are equal. RD-52 runs
# HEC-RAS in its mixed flow regime, which came within 0.01 m of it throughout, the jump too. The standard step is
# subcritical, as HEC-RAS's subcritical regime is.

BUMP_WIDTH, BUMP_WALL, BUMP_ORDINATE = 0.3, 0.6, 0.01
BUMP_SIDE = BUMP_ORDINATE / BUMP_WALL  # the walls' run for each metre they rise
BUMP_FLOW, BUMP_N, TAILWATER = 0.054, 0.0001, 0.33
BUMP_SPACING, BUMP_LENGTH, CREST, FOOT = 0.2, 20.6, 10.0, 12.0


def bump_bed(x: np.ndarray) -> np.ndarray:
    return np.where((x > 8.0) & (x < FOOT), 0.2 - 0.05 * (x - CREST) ** 2, 0.0)


def bump_area(h: float) -> float:
    return (BUMP_WIDTH + BUMP_SIDE * h) * h


def bump_energy(h: float, bed: float) -> float:
    return bed + h + BUMP_FLOW ** 2 / (2.0 * G * bump_area(h) ** 2)


def bump_froude_squared(h: float) -> float:
    return BUMP_FLOW ** 2 * (BUMP_WIDTH + 2.0 * BUMP_SIDE * h) / (G * bump_area(h) ** 3)


def bump_force(h: float) -> float:
    """The specific force: the flow's momentum and the water's pressure on the cross section, per unit weight."""
    return BUMP_FLOW ** 2 / (G * bump_area(h)) + BUMP_WIDTH * h ** 2 / 2.0 + BUMP_SIDE * h ** 3 / 3.0


BUMP_CRITICAL = brentq(lambda h: bump_froude_squared(h) - 1.0, 1e-3, 1.0, xtol=1e-14)


def bump_depth(energy: float, bed: float, *, supercritical: bool = False) -> float:
    """The depth whose energy over the bed is `energy`, on the subcritical or supercritical side, or NaN for none."""
    if bump_energy(BUMP_CRITICAL, bed) > energy:
        return math.nan
    low, high = (1e-4, BUMP_CRITICAL) if supercritical else (BUMP_CRITICAL, 10.0)
    return brentq(lambda h: bump_energy(h, bed) - energy, low, high, xtol=1e-14)


def bump_exact(x: np.ndarray, contraction: float = 0.0) -> np.ndarray:
    """The exact depths at x metres along the channel. contraction is the standard step's contraction loss
    coefficient upstream of the crest, where the water speeds up towards it: the energy there is the crest's and the
    loss on the way, the integral of C' Fr^2 over the depth from critical depth's up (see tapered)."""
    crest, tail = bump_energy(BUMP_CRITICAL, 0.2), bump_energy(TAILWATER, 0.0)
    loss = lambda h: quad(lambda s: tapered(bump_froude_squared(s), contraction) * bump_froude_squared(s),
                          BUMP_CRITICAL, h, epsabs=1e-13, epsrel=1e-12)[0] if contraction else 0.0
    depths = np.empty(x.size)
    for i, (along, bed) in enumerate(zip(x, bump_bed(x))):
        if along < CREST:
            depths[i] = brentq(lambda h: bump_energy(h, bed) - loss(h) - crest, BUMP_CRITICAL, 10.0, xtol=1e-14)
        elif along == CREST:
            depths[i] = BUMP_CRITICAL
        else:
            fast, slow = bump_depth(crest, bed, supercritical=True), bump_depth(tail, bed)
            # Supercritical until the tailwater's subcritical water pushes harder
            depths[i] = slow if bump_force(slow) >= bump_force(fast) else fast
    return depths


def bump_section(bed: float) -> XSection:
    """The channel's cross section: its bed, 0.3 m across with ordinates 1 cm apart, and its walls 0.6 m high at the
    next ordinates, then ground rising gently."""
    offsets = np.arange(-25, 26) * BUMP_ORDINATE
    beyond = np.maximum(np.abs(offsets) - BUMP_WIDTH / 2 - BUMP_ORDINATE, 0.0)
    heights = np.where(np.abs(offsets) <= BUMP_WIDTH / 2 + 1e-9, 0.0, BUMP_WALL + 0.01 * beyond)
    return XSection(bed + heights, np.full(offsets.size, BUMP_N), BUMP_ORDINATE)


def bump_march(*, eddy_loss: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """RD-52's cross sections' distances along the channel, and the standard step's depths and critical flags at
    them, from the tailwater up."""
    x = np.round(np.arange(0.0, BUMP_LENGTH + BUMP_SPACING / 2, BUMP_SPACING), 9)
    beds = bump_bed(x)
    tables = [HydraulicTable(bump_section(bed), bed + 0.5) for bed in beds[::-1]]
    wse, critical = march(tables, TAILWATER, BUMP_FLOW, BUMP_SPACING, eddy_loss=eddy_loss)
    return x, wse[::-1] - beds, critical[::-1]


@pytest.mark.parametrize("eddy_loss", [False, True], ids=["as RD-52", "with the eddy loss"])
def test_the_standard_step_takes_the_water_over_the_bump_from_critical_depth_at_its_crest(eddy_loss: bool) -> None:
    """Upstream of the crest, within 0.001 m of the exact water surface (HEC-RAS: 0.01 m), from critical depth at the
    crest, to a level of the table: the energy is least there, so the level's error hardly changes it. The eddy loss
    on the water speeding up towards the crest, 0.1 of its gain in velocity head where it is slow and less towards
    critical depth (reckoned at the water without it), stands the water 0.006 m above the frictionless profile
    upstream. From the bump's foot down, the water is the tailwater's 0.33 m."""
    x, depths, _ = bump_march(eddy_loss=eddy_loss)
    exact = bump_exact(x, 0.1 if eddy_loss else 0.0)
    upstream, crest, below = x < CREST, x == CREST, x >= FOOT
    table = HydraulicTable(bump_section(0.2), 0.7)

    np.testing.assert_allclose(depths[upstream], exact[upstream], atol=0.001)
    assert depths[crest][0] == pytest.approx(BUMP_CRITICAL, abs=table.levels[1] - table.levels[0])
    np.testing.assert_allclose(depths[below], TAILWATER, atol=1e-5)
    if eddy_loss:
        assert (depths - bump_exact(x))[upstream].max() == pytest.approx(0.006, abs=0.001)


def test_the_standard_step_holds_the_water_at_critical_depth_or_above_down_the_bump_with_no_jump() -> None:
    """Below the crest the exact water runs down the bump supercritical, and jumps up to the tailwater's level 1.66 m
    on. The standard step is subcritical: it holds the water at critical depth from the crest as far as the
    tailwater's energy can't reach (1.25 m on), and has it at the tailwater's subcritical depth beyond. So its water
    stands above the exact water between the crest and the jump, and on it from the jump down."""
    x, depths, critical = bump_march(eddy_loss=False)
    exact = bump_exact(x)
    tail = bump_energy(TAILWATER, 0.0)
    bed = lambda s: float(bump_bed(np.array(s)))
    reach = brentq(lambda s: bump_energy(BUMP_CRITICAL, bed(s)) - tail, CREST, FOOT)
    jump = brentq(lambda s: bump_force(bump_depth(tail, bed(s)))
                  - bump_force(bump_depth(bump_energy(BUMP_CRITICAL, 0.2), bed(s), supercritical=True)),
                  reach + 1e-9, FOOT)
    held, below_jump = (x >= CREST) & (x < reach), (x > jump) & (x < FOOT)
    table = HydraulicTable(bump_section(0.2), 0.7)

    assert reach - CREST == pytest.approx(1.25, abs=0.01) and jump - CREST == pytest.approx(1.66, abs=0.01)
    np.testing.assert_array_equal(critical, held)
    np.testing.assert_allclose(depths[held], BUMP_CRITICAL, atol=table.levels[1] - table.levels[0])
    assert np.all(depths[(x > CREST) & (x < jump)] > exact[(x > CREST) & (x < jump)])
    np.testing.assert_allclose(depths[below_jump], exact[below_jump], atol=1e-4)


# --- Subcritical flow over a variable sloping bed (RD-52 test 4.2.2) ------------------------------------------------
#
# MacDonald et al. (1997) worked channel beds out backwards from water surfaces chosen for them, so that the exact
# answer is known. RD-52 runs SWASHES's (Delestre et al., 2013) on its 2D model: a reach 1000 m long carrying 2 m^3/s
# per metre of width at n 0.033, its depth h_c (1 + exp(-16 (x/1000 - 1/2)^2) / 2) at x metres down it, with h_c the
# critical depth, so subcritical all the way, the Froude number 0.99 at its ends and 0.54 in its middle. Here the
# channel is a trapezoid 10 m wide at the bed, its walls rising 3 m over the next metre, carrying 20 m^3/s, with the
# same depths for its own critical depth, and its bed worked out for them from the energy equation
#
#     dz/dx = (Fr^2 - 1) dh/dx - Sf,    Fr^2 = Q^2 T / (g A^3)
#
# integrated exactly. The bed falls 8.1 m, at slopes from 0.0034 to 0.013.

MAC_LENGTH, MAC_N, MAC_FLOW = 1000.0, 0.033, 20.0
MAC_BOTTOM, MAC_WALL, MAC_ORDINATE = 10.0, 3.0, 1.0
MAC_SIDE = MAC_ORDINATE / MAC_WALL  # the walls' run for each metre they rise


def mac_area(h):
    return (MAC_BOTTOM + MAC_SIDE * h) * h


def mac_froude_squared(h):
    return MAC_FLOW ** 2 * (MAC_BOTTOM + 2.0 * MAC_SIDE * h) / (G * mac_area(h) ** 3)


def mac_friction(h):
    perimeter = MAC_BOTTOM + 2.0 * h * math.sqrt(1.0 + MAC_SIDE ** 2)
    return (MAC_N * MAC_FLOW) ** 2 * perimeter ** (4 / 3) / mac_area(h) ** (10 / 3)


MAC_CRITICAL = brentq(lambda h: mac_froude_squared(h) - 1.0, 1e-3, 5.0, xtol=1e-14)


def mac_depth(x):
    """The exact depths x metres down the reach."""
    return MAC_CRITICAL * (1.0 + 0.5 * np.exp(-16.0 * (np.asarray(x) / MAC_LENGTH - 0.5) ** 2))


def mac_bed_slope(x):
    """dz/dx, x metres down the reach."""
    s = np.asarray(x) / MAC_LENGTH - 0.5
    rise = -16.0 * MAC_CRITICAL * np.exp(-16.0 * s ** 2) * s / MAC_LENGTH  # dh/dx
    return (mac_froude_squared(mac_depth(x)) - 1.0) * rise - mac_friction(mac_depth(x))


_MAC_BED = solve_ivp(lambda x, z: [mac_bed_slope(x)], (MAC_LENGTH, 0.0), [0.0], method="DOP853", dense_output=True,
                     rtol=1e-12, atol=1e-13).sol


def mac_bed(x) -> np.ndarray:
    """The bed's elevation x metres down the reach, above its end's."""
    return _MAC_BED(np.asarray(x, dtype=float))[0]


def mac_heights(offset: np.ndarray) -> np.ndarray:
    """The channel's height above its bed, metres across from its centre line."""
    beyond = np.maximum(np.abs(offset) - MAC_BOTTOM / 2 - MAC_ORDINATE, 0.0)
    return np.where(np.abs(offset) <= MAC_BOTTOM / 2, 0.0, MAC_WALL + 0.01 * beyond)


def mac_section(bed: float) -> XSection:
    offsets = np.arange(-25, 26) * MAC_ORDINATE
    return XSection(bed + mac_heights(offsets), np.full(offsets.size, MAC_N), MAC_ORDINATE)


def mac_march(spacing: float, *, eddy_loss: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cross sections `spacing` metres apart up from the reach's end, their distances down it, and the standard step's
    depths and critical flags at them, from the exact depth at the end."""
    x = np.arange(MAC_LENGTH, -1e-9, -spacing)
    beds = mac_bed(x)
    tables = [HydraulicTable(mac_section(bed), bed + 1.5) for bed in beds]
    wse, critical = march(tables, beds[0] + mac_depth(MAC_LENGTH), MAC_FLOW, spacing, eddy_loss=eddy_loss)
    return x, wse - beds, critical


def test_the_standard_step_gives_macdonald_s_profile() -> None:
    """Without the eddy loss, the step's energy equation is the one the bed was worked out from: with cross sections
    32 m apart, the water is within 0.001 m of the exact depths everywhere (0.0005 m on average), the error falling as
    the spacing's square, to 0.00002 m at 4 m apart. (HEC-RAS's 2D model, on cells of 32 m to 1 m: within 0.0033 m to
    0.0005 m on average.)"""
    errors = {}
    for spacing in (32.0, 16.0, 8.0, 4.0):
        x, depths, critical = mac_march(spacing, eddy_loss=False)
        assert not critical.any()
        errors[spacing] = np.abs(depths - mac_depth(x)).max()

    assert errors[32.0] < 0.001 and errors[4.0] < 0.00003
    assert errors[32.0] / errors[16.0] > 3.5 and errors[16.0] / errors[8.0] > 3.5


MAC_CELLS = int(round(MAC_LENGTH / MAC_ORDINATE))  # the channel's cells, one per metre


def write_macdonald_channel(folder: Path) -> dict:
    """MacDonald's channel on 1 m cells, its 1000 cells 1 to 1000 m down the reach (reach 1), and below them 400 m of
    the same channel at the slope whose normal depth is the depth at the end (reach 2), the tailwater. The bed's
    slope at either end carries on into the raster's margins."""
    tail_slope = mac_friction(mac_depth(MAC_LENGTH))
    rows, cols = np.mgrid[0:2 * HALF_WIDTH + 1, 0:MARGIN + MAC_CELLS + CELLS_BELOW + MARGIN]
    meet = MARGIN + MAC_CELLS  # reach 2's first column
    x = MAC_LENGTH + (cols[0] - (meet - 1)) * MAC_ORDINATE  # metres down reach 1
    bed = np.where(x > MAC_LENGTH, -tail_slope * (x - MAC_LENGTH),
                   np.where(x < 0.0, mac_bed(0.0) + mac_bed_slope(0.0) * x, mac_bed(np.clip(x, 0.0, MAC_LENGTH))))
    dem = 100.0 + bed[None, :] + mac_heights((rows - HALF_WIDTH) * MAC_ORDINATE)
    streams = np.zeros(dem.shape, dtype=np.int32)
    streams[HALF_WIDTH, MARGIN:meet] = 1
    streams[HALF_WIDTH, meet:meet + CELLS_BELOW] = 2
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(32616)
    reaches = {1: {"start": (HALF_WIDTH, MARGIN), "end": (HALF_WIDTH, meet - 1), "downstream": 2, "baseflow": 0.0,
                   "q_max": MAC_FLOW},
               2: {"start": (HALF_WIDTH, meet), "end": (HALF_WIDTH, meet + CELLS_BELOW - 1), "downstream": -1,
                   "baseflow": 0.0, "q_max": MAC_FLOW}}
    return write_inputs(folder, dem, streams, srs, (500000.0, MAC_ORDINATE, 0.0, 3800000.0, 0.0, -MAC_ORDINATE),
                        reaches, water_n=MAC_N, land_n=MAC_N)


def test_on_a_dem_the_standard_step_follows_macdonald_s_bed(tmp_path: Path) -> None:
    """Each cross section alone has the water at normal depth for its own slope, 0.019 m out on average and up to
    0.056 m. With Standard_Step, 0.0023 m out on average, and within 0.005 m everywhere, the last 5 m above the
    tailwater's reach too, where the Froude number is 0.99 (see the next test)."""
    inputs = write_macdonald_channel(tmp_path)
    errors = {}
    for step in (False, True):
        results = pipeline.run(Configs.from_mapping({**inputs, "Standard_Step": step}), quiet=True, write=False)
        up, depths = depths_at_the_maximum_flow(results, 1, last=MARGIN + MAC_CELLS - 1, cell=MAC_ORDINATE,
                                                flow=MAC_FLOW)
        assert depths.size == MAC_CELLS
        errors[step] = depths - mac_depth(MAC_LENGTH - up)

    assert np.abs(errors[False]).mean() == pytest.approx(0.019, abs=0.001)
    assert np.abs(errors[True]).mean() < 0.003 and np.abs(errors[True]).max() < 0.005


def test_near_critical_depth_uniform_flow_stays_at_normal_depth() -> None:
    """MacDonald's channel at its depth at the end, a Froude number of 0.986, as uniform flow: a step up 1 m of the
    same channel, its bed higher by the friction slope, keeps the water at normal depth, with the eddy loss or
    without. Reckoned at the water's own velocity head, the contraction loss (0.1 of the gain in velocity head going
    downstream) would grow with the depth faster than the specific energy does (1.1 Fr^2 > 1), so that the energy
    equation's root at normal depth only touched zero, and the step found the next one up, 0.02 m higher; reckoned at
    the water without it, the loss is nothing here."""
    depth = mac_depth(MAC_LENGTH)
    rise = mac_friction(depth)
    down, up = HydraulicTable(mac_section(0.0), depth), HydraulicTable(mac_section(rise), rise + depth)
    no_loss = {"eddy_head": velocity_head(down, depth, MAC_FLOW), "eddy_steps": NO_EDDY_LOSS}

    assert step_up(down, depth, MAC_FLOW, up, MAC_FLOW, 1.0, **no_loss)[0] - rise == pytest.approx(depth, abs=1e-5)
    assert step_up(down, depth, MAC_FLOW, up, MAC_FLOW, 1.0)[0] - rise == pytest.approx(depth, abs=1e-5)
