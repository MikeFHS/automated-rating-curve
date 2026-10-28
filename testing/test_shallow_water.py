"""The 2D model that test_valley_sections.py checks ARC against (shallow_water.py), on the analytic tests of HEC-RAS's
verification report for its own 2D model (USACE, 2018, RD-52, chapter 4):

- 4.2.4, a flood wave spreading over a flat bed (Hunter et al., 2005);
- 4.2.2, subcritical flow over a variable sloping bed (MacDonald et al., 1997), on RD-52's six grids;
- 4.2.1, Chow's (1959) channel backed up behind a lake, test_benchmarks.py's;
- 4.2.3, flow over a bump without friction, test_benchmarks.py's, with a hydraulic jump;
- 4.2.5, a dam break over a flat frictionless bed, onto still water and onto a dry bed (Stoker, 1957);
- 4.2.6, sloshing in a rectangular basin without friction (Lamb, 1945);
- 4.2.8, runoff from rain on a plane, subcritical and supercritical;
- 4.2.9, the Coriolis force holding a channel's water surface tilted across it (Gill, 1982), and turning water in a
  wide basin (an inertial oscillation, not RD-52's);
- and, not RD-52's, the meandering valley's channel crossing the grid at an angle, a known weakness of a raster (an
  expected failure, xfail): its stair steps cost it conveyance, and with advection momentum at each, so that the
  valley tests keep to the local inertial scheme (synthetic_valley.py).

Where the answer is a steady flow, the model is checked against the exact solution of its own equations, which says
it solves them right, and against the textbook's, which says how far they are from it. The schemes are first order:
the faces take their upwind cell's depth, and the momentum its upwind faces' velocities, so the errors halve with the
cells. Without advection (LISFLOOD-FP's local inertial scheme, advection=False), the steady water surface has no
velocity head: on MacDonald's bed that puts the water 0.017 m out on every grid.

RD-52's test 4.2.7, long-wave run-up on a planar slope (Carrier et al., 2003), isn't here: its exact solution is an
integral transform that has to be worked out numerically, and RD-52 gives it in figures. Nor are its laboratory and
field tests, which it gives in figures too.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
from shallow_water import G, steady, unsteady
from test_benchmarks import (
    BUMP_FLOW,
    BUMP_WIDTH,
    FLOW,
    FT,
    LAKE_DEPTH,
    LENGTH,
    SLOPE,
    TAILWATER,
    N,
    backwater,
    bump_bed,
    ground,
)

# --- A flood wave over a flat bed (RD-52 test 4.2.4) ----------------------------------------------------------------
#
# Water comes in at x = 0 onto a flat, dry bed (n 0.01) at 1 m/s everywhere, its depth there rising so that the wave
# keeps its shape: h = (7/3 n^2 u^2 (u t - x))^(3/7) behind its front, which is at x = u t (Hunter et al., 2005).
# RD-52's cells are 25 m.

WAVE_N, WAVE_SPEED, WAVE_LENGTH = 0.01, 1.0, 5000.0
WAVE_TIMES = (900.0, 1800.0, 2700.0, 3600.0)


def wave_depth(x, t: float):
    return (7.0 / 3.0 * WAVE_N ** 2 * WAVE_SPEED ** 2 * np.clip(WAVE_SPEED * t - x, 0.0, None)) ** (3.0 / 7.0)


@pytest.mark.parametrize("cell", [25.0, 10.0])
def test_a_flood_wave_over_a_flat_bed_keeps_its_water_and_hunter_s_shape(cell: float) -> None:
    """All the water that comes in is there (to 1e-12), and more than 500 m behind the front it is within 0.007 m of
    the exact depth on 25 m cells and 0.003 m on 10 m cells, up to an hour on, when it is 0.93 m deep at the inflow.
    The front, where the water is 5 cm deep, runs up to two cells ahead of the exact one, as HEC-RAS's did. (Without
    advection, the water was 1.5 to 2 cm too deep on either, and its front lagged by up to 170 m.)"""
    cells = int(WAVE_LENGTH / cell)
    x = (np.arange(cells) + 0.5) * cell
    out = unsteady(np.zeros((1, cells)), WAVE_N, cell, WAVE_TIMES, inflow_rows=[0],
                   inflow=lambda t: WAVE_SPEED * wave_depth(0.0, t) * cell)
    for k, t in enumerate(WAVE_TIMES):
        depths, exact = out["h"][k, 0], wave_depth(x, t)
        behind = x < WAVE_SPEED * t - 500.0
        lag = x[np.argmax(exact < 0.05)] - x[np.argmax(depths < 0.05)]

        assert depths.sum() * cell * cell == pytest.approx(out["volume_in"][k], rel=1e-12)
        assert np.abs(depths - exact)[behind].max() < (0.007 if cell == 25.0 else 0.003)
        assert -2.0 * cell <= lag <= 0.0


# --- Subcritical flow over a variable sloping bed (RD-52 test 4.2.2) ------------------------------------------------
#
# SWASHES's case (Delestre et al., 2013), as RD-52 runs it: 1000 m long and two cells wide, 2 m^3/s per metre of width
# coming in, n 0.033, and the exact depth at the end held at the downstream edge. test_benchmarks.py has the depths
# and how the bed comes from them; here the channel is a strip of one infinitely wide, its friction slope
# n^2 q^2 / h^(10/3). The cells are the reach divided into 31, 62, 125, 250, 500 and 1000: RD-52's 32 m to 1 m, on
# which HEC-RAS was within 0.0033, 0.0015, 0.00096, 0.00072, 0.00062 and 0.0005 m on average.

SLOPED_Q, SLOPED_N, SLOPED_LENGTH = 2.0, 0.033, 1000.0
SLOPED_CRITICAL = (SLOPED_Q ** 2 / G) ** (1.0 / 3.0)
SLOPED_GRIDS = (31, 62, 125, 250, 500, 1000)
HEC_RAS_SLOPED_MAE = {31: 0.00334, 62: 0.00150, 125: 0.00096, 250: 0.00072, 500: 0.00062, 1000: 0.00050}


def sloped_depth(x):
    return SLOPED_CRITICAL * (1.0 + 0.5 * np.exp(-16.0 * (np.asarray(x) / SLOPED_LENGTH - 0.5) ** 2))


def sloped_friction(h):
    return SLOPED_N ** 2 * SLOPED_Q ** 2 / h ** (10.0 / 3.0)


def sloped_bed_slope(x):
    """dz/dx, x metres down the reach: (Fr^2 - 1) dh/dx - Sf."""
    s = np.asarray(x) / SLOPED_LENGTH - 0.5
    rise = -16.0 * SLOPED_CRITICAL * np.exp(-16.0 * s ** 2) * s / SLOPED_LENGTH
    h = sloped_depth(x)
    return (SLOPED_Q ** 2 / (G * h ** 3) - 1.0) * rise - sloped_friction(h)


_SLOPED_BED = solve_ivp(lambda x, z: [sloped_bed_slope(x)], (SLOPED_LENGTH, 0.0), [0.0], method="DOP853",
                        dense_output=True, rtol=1e-12, atol=1e-13).sol


def local_inertial_profile(x: np.ndarray) -> np.ndarray:
    """The local inertial equations' exact steady depths at x (ascending), their water surface falling at the
    friction slope alone, d(z + h)/dx = -Sf, from the exact depth at the end."""
    profile = solve_ivp(lambda s, h: [-sloped_bed_slope(s) - sloped_friction(h[0])], (SLOPED_LENGTH, float(x[0])),
                        [float(sloped_depth(SLOPED_LENGTH))], t_eval=x[::-1], method="DOP853", rtol=1e-11,
                        atol=1e-12)
    return profile.y[0][::-1]


def sloped_runs(advection: bool) -> dict:
    """The model's steady depths down the reach on each grid, from dry: {cells: (their centres, depths)}."""
    runs = {}
    for cells in SLOPED_GRIDS:
        cell = SLOPED_LENGTH / cells
        x = (np.arange(cells) + 0.5) * cell
        z = np.tile(_SLOPED_BED(x)[0], (2, 1))
        out = steady(z, SLOPED_N, cell, [0, 1], SLOPED_Q * 2 * cell, stage_out=float(sloped_depth(SLOPED_LENGTH)),
                     advection=advection, tol=1e-4, check=500)
        assert out["outflow"] == pytest.approx(SLOPED_Q * 2 * cell, rel=1e-4)
        runs[cells] = x, out["h"][0]
    return runs


def test_on_macdonald_s_bed_the_model_converges_to_macdonald_s_profile() -> None:
    """The steady depths are the exact ones to within 0.4 mm for each metre of the cells' size on average (0.012 m on
    32 m cells, 0.0004 m on 1 m), the error halving with the cells. HEC-RAS's model, whose order of convergence RD-52
    puts at 1.32, is closer on cells of 4 m and more (0.0033 m on 32 m cells), and this one on 1 m cells."""
    for cells, (x, depths) in sloped_runs(advection=True).items():
        error = np.abs(depths - sloped_depth(x)).mean()

        assert error < 0.0004 * SLOPED_LENGTH / cells
        if cells == 1000:
            assert error < HEC_RAS_SLOPED_MAE[cells]


def test_without_advection_the_model_converges_to_the_local_inertial_profile() -> None:
    """With advection=False, the steady depths are the local inertial equations' exact profile to within 0.5 mm for
    each metre of the cells' size on average, to 0.3 mm on 1 m cells, and so 0.017 m from MacDonald's on every grid:
    their water surface falls at the friction slope, where the exact energy does, d(z + h + V^2/2g)/dx = -Sf."""
    for cells, (x, depths) in sloped_runs(advection=False).items():
        assert np.abs(depths - local_inertial_profile(x)).mean() < 0.0005 * SLOPED_LENGTH / cells
        assert np.abs(depths - sloped_depth(x)).mean() > 0.015


# --- Chow's channel behind a lake (RD-52 test 4.2.1) ----------------------------------------------------------------
#
# test_benchmarks.py's channel: 20 ft wide at the bed, its sides sloping 2:1 to banks 8 ft high, n 0.025, the bed
# slope 0.0016, 400 ft^3/s coming in across it, and the lake holding the water 5.0 ft deep at the downstream edge,
# 3000 ft down. RD-52 ran it one 40 ft cell wide, each cell carrying the trapezoid inside it (HEC-RAS's subgrid
# terrain); a raster has to draw the trapezoid in cells, here 4 ft square, each the mean of the ground over it. The
# model starts dry.

CHOW_CELL_FT = 4.0
CHOW_OFFSETS_FT = (np.arange(-8, 8) + 0.5) * CHOW_CELL_FT  # the cells' centres across the channel
CHOW_HEIGHTS = np.array([np.mean(ground(offset + CHOW_CELL_FT * np.linspace(-0.5, 0.5, 401)))
                         for offset in CHOW_OFFSETS_FT])  # each cell's mean height above the bed, in metres
CHOW_WIDTH = CHOW_CELL_FT * FT


def chow_strips(depth: float) -> tuple[np.ndarray, np.ndarray]:
    """Each cell's area and conveyance across the channel with the water `depth` over the bed: its depth^(5/3) / n,
    times its width."""
    depths = np.clip(depth - CHOW_HEIGHTS, 0.0, None)
    return CHOW_WIDTH * depths, CHOW_WIDTH * depths ** (5.0 / 3.0) / N


def chow_profile(up: np.ndarray) -> np.ndarray:
    """The model's own equations' exact steady depths at `up` metres up the channel from the lake (ascending): each
    cell's friction its own depth's, the water level across the channel, and the momentum flux of the cells' flows,
    beta Q^2 / A with beta = A sum(K_i^2 / A_i) / K^2 (the cells' conveyances K_i and areas A_i):

        dy/ds = (Sf - S0) / (1 + Q^2 / (g A) d(beta / A)/dy)."""
    def momentum(y):
        a, k = chow_strips(y)
        wet = a > 0.0
        return a.sum() * np.sum(k[wet] ** 2 / a[wet]) / k.sum() ** 2 / a.sum()

    def rise(_, y):
        a, k = chow_strips(y[0])
        change = (momentum(y[0] + 1e-6) - momentum(y[0] - 1e-6)) / 2e-6
        return [((FLOW / k.sum()) ** 2 - SLOPE) / (1.0 + FLOW ** 2 / (G * a.sum()) * change)]

    return solve_ivp(rise, (0.0, float(up[-1])), [LAKE_DEPTH], t_eval=up, method="DOP853", rtol=1e-10,
                     atol=1e-12).y[0]


def test_chow_s_channel_in_2d_is_its_own_equations_profile_not_chow_s() -> None:
    """From dry, the water settles to the exact profile of the model's own equations, each cell's friction its own
    depth's, within 0.001 ft. That isn't Chow's: the cells carry the flow 3.18 ft deep, where Manning's equation with
    the trapezoid's hydraulic radius needs 3.36 ft, as each cell's friction leaves the walls out. So the water stands
    below Chow's profile but at the lake, by 0.14 ft on average and 0.2 ft at most, where it nears normal depth.
    HEC-RAS's 2D model, its cells' faces taking the hydraulic radius from the terrain, was 0.007 ft below on average.
    (Without advection, the water started dry never settled; the inflow is shared across the channel by the cells'
    conveyance under their mean water surface, as HEC-RAS shares it, else the shallow cells take too much of it, and
    the water at the inflow stood 0.18 ft low.)"""
    cell = CHOW_WIDTH
    columns = int(round(LENGTH / cell))
    up = LENGTH - (np.arange(columns) + 0.5) * cell
    z = (SLOPE * up)[None, :] + CHOW_HEIGHTS[:, None]
    out = steady(z, N, cell, np.flatnonzero(CHOW_HEIGHTS < 8.0 * FT), FLOW, stage_out=LAKE_DEPTH, tol=1e-4,
                 check=1000)
    bed_row = int(np.argmin(np.abs(CHOW_OFFSETS_FT)))
    up, depths = up[::-1], out["h"][bed_row, ::-1]
    own = chow_profile(up)
    normal = brentq(lambda y: chow_strips(y)[1].sum() * math.sqrt(SLOPE) - FLOW, 0.1, 3.0)
    below = (backwater(up) - depths) / FT

    assert out["outflow"] == pytest.approx(FLOW, rel=1e-4)
    assert np.abs(depths - own).max() / FT < 0.001
    assert normal / FT == pytest.approx(3.18, abs=0.005)
    assert below.min() > -0.001
    assert below.mean() == pytest.approx(0.14, abs=0.01) and below.max() == pytest.approx(0.20, abs=0.01)


# --- Flow over a bump without friction (RD-52 test 4.2.3) -----------------------------------------------------------
#
# test_benchmarks.py's bump, as RD-52 runs it on its 2D model: a channel 0.3 m wide in 0.05 m cells, six across, 21 m
# long, n 0.00001, 0.054 m^3/s coming in, and the tailwater 0.33 m deep at the downstream edge. The exact water comes
# from critical depth at the crest by Bernoulli's equation, up the channel and down the bump supercritical, and from
# the tailwater up to where the momentum each side balances: there it jumps.

BUMP_CELL, BUMP_LENGTH_2D = 0.05, 21.0
BUMP_Q = BUMP_FLOW / BUMP_WIDTH  # m^2/s
BUMP_CRITICAL_2D = (BUMP_Q ** 2 / G) ** (1.0 / 3.0)


def bump_energy_2d(h, bed):
    return bed + h + BUMP_Q ** 2 / (2.0 * G * h ** 2)


def bump_force_2d(h):
    return BUMP_Q ** 2 / (G * h) + h ** 2 / 2.0


def bump_exact_2d(x: np.ndarray) -> np.ndarray:
    """The exact depths along the channel, at x metres."""
    crest, tail = bump_energy_2d(BUMP_CRITICAL_2D, 0.2), bump_energy_2d(TAILWATER, 0.0)
    depths = np.empty(x.size)
    for i, (along, bed) in enumerate(zip(x, bump_bed(x))):
        if along < 10.0:
            depths[i] = brentq(lambda h: bump_energy_2d(h, bed) - crest, BUMP_CRITICAL_2D, 5.0)
        else:
            fast = brentq(lambda h: bump_energy_2d(h, bed) - crest, 1e-4, BUMP_CRITICAL_2D) \
                if bump_energy_2d(BUMP_CRITICAL_2D, bed) < crest else BUMP_CRITICAL_2D
            slow = brentq(lambda h: bump_energy_2d(h, bed) - tail, BUMP_CRITICAL_2D, 5.0) \
                if bump_energy_2d(BUMP_CRITICAL_2D, bed) < tail else math.nan
            depths[i] = slow if slow == slow and bump_force_2d(slow) >= bump_force_2d(fast) else fast
    return depths


def test_flow_over_a_bump_passes_critical_depth_at_the_crest_and_jumps_where_bernoulli_and_momentum_say() -> None:
    """From dry, the water settles 0.411 m deep upstream of the bump, 0.003 m below the depth from which Bernoulli's
    equation takes it through critical depth at the crest, 0.414 m (HEC-RAS: within 0.01 m), and still to 0.1 mm
    from 100 s to 120 s. It runs down the bump supercritical and jumps up to the tailwater 1.63 m past the crest, in
    the cell the exact jump is in; beyond 13 m it is within 0.001 m of the tailwater's level, and within 0.002 m of
    the exact water on average all along (0.055 m at most, at the jump, a cell's width of its 0.15 m rise). (Without
    advection, nothing made the water critical at the crest, and it sloshed by tenths of a metre.)"""
    x = (np.arange(int(round(BUMP_LENGTH_2D / BUMP_CELL))) + 0.5) * BUMP_CELL
    z = np.tile(bump_bed(x), (int(round(BUMP_WIDTH / BUMP_CELL)), 1))
    bernoulli = brentq(lambda h: bump_energy_2d(h, 0.0) - bump_energy_2d(BUMP_CRITICAL_2D, 0.2), BUMP_CRITICAL_2D, 1.0)
    out = unsteady(z, 1e-5, BUMP_CELL, (100.0, 110.0, 120.0), inflow_rows=np.arange(z.shape[0]),
                   inflow=lambda t: BUMP_FLOW, stage_out=TAILWATER)
    upstream = (z + out["h"])[:, :, x < 7.0].mean(axis=(1, 2))
    depths, exact = out["h"][-1].mean(axis=0), bump_exact_2d(x)
    rise = np.where(x[:-1] > 10.0, np.diff(depths), 0.0)

    assert bernoulli == pytest.approx(0.414, abs=0.001)
    assert np.ptp(upstream) < 0.0001 and upstream[-1] == pytest.approx(bernoulli, abs=0.005)
    assert out["h"].min() > 0.0 and np.allclose(out["h"][-1], out["h"][-1][0])
    assert np.argmax(rise) == np.argmax(np.where(x[:-1] > 10.0, np.diff(exact), 0.0))
    assert np.abs(depths - exact)[x > 13.0].max() < 0.001 and np.abs(depths - exact).mean() < 0.0025


# --- A dam break over a flat frictionless bed (RD-52 test 4.2.5) ----------------------------------------------------
#
# RD-52's: 1200 m long and two 1 m cells wide, the water 10 m deep above the dam, at 600 m, and 5 m deep or dry below
# it, still, n 0.000001. The exact answer (Stoker, 1957) is the Riemann problem's: a rarefaction runs up the water
# above the dam, and down it a shock onto the still water, or a thinning front onto the dry bed (Ritter's solution).

DAM, DAM_DEPTH, DAM_CELLS = 600.0, 10.0, 1200
DAM_TIMES = (10.0, 20.0, 30.0)


def _riemann_side(h: float, side: float) -> float:
    """The velocity change across a rarefaction (h below side's depth) or a shock (above it), from depth side to h."""
    if h <= side:
        return 2.0 * (math.sqrt(G * h) - math.sqrt(G * side))
    return (h - side) * math.sqrt(0.5 * G * (h + side) / (h * side))


def dam_break(x: np.ndarray, t: float, below: float) -> tuple[np.ndarray, np.ndarray]:
    """The exact depths and velocities at x, t seconds after the dam goes, the water `below` deep below it."""
    xi, celerity = (x - DAM) / t, math.sqrt(G * DAM_DEPTH)
    depth, speed = np.where(xi < 0.0, DAM_DEPTH, below), np.zeros(x.size)
    if below == 0.0:
        fan = (xi >= -celerity) & (xi <= 2.0 * celerity)
        depth[fan] = (2.0 * celerity - xi[fan]) ** 2 / (9.0 * G)
        speed[fan] = 2.0 / 3.0 * (xi[fan] + celerity)
        return depth, speed
    star = brentq(lambda h: _riemann_side(h, DAM_DEPTH) + _riemann_side(h, below), 1e-9, DAM_DEPTH, xtol=1e-14)
    star_speed = -_riemann_side(star, DAM_DEPTH)
    shock = star * star_speed / (star - below)
    fan = (xi >= -celerity) & (xi < star_speed - math.sqrt(G * star))
    depth[fan] = (2.0 * celerity - xi[fan]) ** 2 / (9.0 * G)
    speed[fan] = 2.0 / 3.0 * (xi[fan] + celerity)
    middle = (xi >= star_speed - math.sqrt(G * star)) & (xi < shock)
    depth[middle], speed[middle] = star, star_speed
    return depth, speed


@pytest.mark.parametrize("below", [5.0, 0.0], ids=["onto still water", "onto a dry bed"])
def test_a_dam_break_runs_as_stoker_s(below: float) -> None:
    """At 10, 20 and 30 s, all the water is there, and the depths are within 0.03 to 0.04 m of the exact ones by
    their root mean square, 0.006 to 0.02 m on average (HEC-RAS: 0.13 to 0.19 m and 0.02 to 0.04 m, its shock running
    slow). Onto still water the shock, the water jumping from 5 m to 7.27 m deep, runs at the exact 9.35 m/s, a cell
    of where it should be, as the momentum each side of it balances, and the velocities are within 0.06 m/s by their
    root mean square. Onto the dry bed the front's thin tip, under 1 cm, lags Ritter's by 30 to 70 m: the faces'
    upwind depths spread it."""
    x = (np.arange(DAM_CELLS) + 0.5)
    h0 = np.tile(np.where(x < DAM, DAM_DEPTH, below), (2, 1))
    out = unsteady(np.zeros((2, DAM_CELLS)), 1e-6, 1.0, DAM_TIMES, h0=h0)
    for k, t in enumerate(DAM_TIMES):
        depth, speed = dam_break(x, t, below)
        assert out["h"][k].sum() == pytest.approx(h0.sum(), rel=1e-12)
        assert np.sqrt(np.mean((out["h"][k, 0] - depth) ** 2)) < 0.05
        if below > 0.0:
            velocity = 0.5 * (out["ux"][k][0, :-1] + out["ux"][k][0, 1:])
            middle = 0.5 * (below + depth.max(where=x > DAM, initial=0.0))
            shock = x[np.flatnonzero((x > DAM) & (out["h"][k, 0] < middle))[0]]
            assert np.sqrt(np.mean((velocity - speed) ** 2)) < 0.06
            assert shock == pytest.approx(x[np.flatnonzero((x > DAM) & (depth < middle))[0]], abs=1.0)


# --- Sloshing in a rectangular basin (RD-52 test 4.2.6) -------------------------------------------------------------
#
# RD-52's basin: 10 km long, 300 m wide and 10 m deep, flat, in 100 m cells, n 0.00001, its water surface set to the
# fundamental standing wave, 1 cm high: eta = A cos(pi x / L) cos(omega t), omega = pi sqrt(g h) / L, a period of
# 2022 s. RD-52 compares the water surface along it at 34 min, 2 h 54, 7 h 29 and 9 h 49, and at its first cell's
# centre every minute for 10 hours.

SLOSH_LENGTH, SLOSH_WIDTH, SLOSH_CELL, SLOSH_DEPTH, SLOSH_HEIGHT = 10_000.0, 300.0, 100.0, 10.0, 0.01
SLOSH_PROFILES = (34 * 60.0, 174 * 60.0, 449 * 60.0, 589 * 60.0)


def test_sloshing_keeps_its_period_and_height() -> None:
    """The water surface is within 0.000004 to 0.0002 m of the exact standing wave along the basin by its root mean
    square at RD-52's times, and 0.00018 m at the first cell all ten hours, with the wave's height kept to within
    0.2% (HEC-RAS: 0.000008 to 0.00018 m along the basin, and 0.00016 m at the first cell with 0.5 s time steps,
    0.0014 m with 5 s ones; this model's are 5 s). The error is in the wave's phase, mostly from the advection:
    without it, 0.00008 m."""
    x = (np.arange(int(SLOSH_LENGTH / SLOSH_CELL)) + 0.5) * SLOSH_CELL
    wavenumber = math.pi / SLOSH_LENGTH
    omega = wavenumber * math.sqrt(G * SLOSH_DEPTH)
    rows = int(SLOSH_WIDTH / SLOSH_CELL)
    series = tuple(np.arange(60.0, 36_000.0 + 1.0, 60.0))
    times = sorted(set(series) | set(SLOSH_PROFILES))
    out = unsteady(np.zeros((rows, x.size)), 1e-5, SLOSH_CELL, times,
                   h0=SLOSH_DEPTH + np.tile(SLOSH_HEIGHT * np.cos(wavenumber * x), (rows, 1)))
    surface = out["h"][:, 0, :] - SLOSH_DEPTH
    exact = SLOSH_HEIGHT * np.cos(wavenumber * x)[None, :] * np.cos(omega * np.array(times))[:, None]
    at = [times.index(t) for t in series]

    for t in SLOSH_PROFILES:
        k = times.index(t)
        assert np.sqrt(np.mean((surface[k] - exact[k]) ** 2)) < 0.0003
    assert np.sqrt(np.mean((surface[at, 0] - exact[at, 0]) ** 2)) < 0.0002
    assert np.abs(surface[at[-40:], 0]).max() == pytest.approx(SLOSH_HEIGHT * math.cos(wavenumber * x[0]), rel=0.005)


# --- Runoff from rain on a plane (RD-52 test 4.2.8) -----------------------------------------------------------------
#
# RD-52's two cases (Govindaraju et al., 1988): a plane 100 m long, dry, rained on steadily, its slope and n making
# the steady flow at its foot Froude number F0 and kinematic wave number K = S0 L / (h0 F0^2): F0 0.5 and K 10 (4 mm/s,
# n 0.0548), and F0 1.5 and K 3 (2.7 mm/s, n 0.0155), the water leaving at normal depth. On 1 m cells. RD-52 compares
# the outflow with other models' in figures. The kinematic wave is the textbook's answer for the outflow: the water
# deepening everywhere at the rain's rate until the time it takes to reach the foot, t_e = h0 / i, so Q / Q0 =
# (t / t_e)^(5/3) until then, and Q0 after. The steady depths are the shallow water equations' exactly, rain bringing no
# momentum with it:
#
#     dh/dx = (g (S0 - Sf) - 2 q i / h^2) / (g (1 - q^2 / (g h^3))),    q = i x,
#
# subcritical all the way down in the first case, from normal depth at the foot; in the second, subcritical down to
# where the numerator and denominator are both zero, 44 m down, and supercritical below.

RUNOFF_LENGTH = 100.0
RUNOFF_CASES = {"F0 0.5, K 10": (0.5, 0.004, 0.0548), "F0 1.5, K 3": (1.5, 0.0027, 0.0155)}


def runoff_steady(x: np.ndarray, rain: float, n: float, slope: float, foot: float) -> np.ndarray:
    """The exact steady depths at x (ascending), the depth at the foot `foot` where the flow is subcritical there."""
    numerator = lambda s, h: G * (slope - n ** 2 * (rain * s) ** 2 / h ** (10.0 / 3.0)) - 2.0 * rain * rain * s / h ** 2
    denominator = lambda s, h: G * (1.0 - (rain * s) ** 2 / (G * h ** 3))
    rise = lambda s, h: [numerator(s, h[0]) / denominator(s, h[0])]
    critical = lambda s: ((rain * s) ** 2 / G) ** (1.0 / 3.0)
    if numerator(RUNOFF_LENGTH, critical(RUNOFF_LENGTH)) < 0.0:
        return solve_ivp(rise, (RUNOFF_LENGTH, float(x[0])), [foot], t_eval=x[::-1], method="DOP853", rtol=1e-10,
                         atol=1e-12).y[0][::-1]
    # Through critical depth where the numerator is zero there too, with the slope of the supercritical side
    through = brentq(lambda s: numerator(s, critical(s)), 1.0, RUNOFF_LENGTH)
    point, e = (through, critical(through)), 1e-6
    partial = lambda f, i: (f(point[0] + e * (i == 0), point[1] + e * (i == 1))
                            - f(point[0] - e * (i == 0), point[1] - e * (i == 1))) / (2.0 * e)
    slope_there = min(np.roots([partial(denominator, 1), partial(denominator, 0) - partial(numerator, 1),
                                -partial(numerator, 0)]).real)
    step, above, below = 0.05, x[x < through], x[x >= through]
    upper = solve_ivp(rise, (through - step, float(above[0])), [point[1] - slope_there * step], t_eval=above[::-1],
                      method="LSODA", rtol=1e-10, atol=1e-12).y[0][::-1]
    lower = solve_ivp(rise, (through + step, float(below[-1])), [point[1] + slope_there * step], t_eval=below,
                      method="LSODA", rtol=1e-10, atol=1e-12).y[0]
    return np.concatenate([upper, lower])


@pytest.mark.parametrize("case", list(RUNOFF_CASES))
def test_runoff_from_rain_on_a_plane_rises_to_its_steady_flow(case: str) -> None:
    """All the rain is there or gone out (to 1e-12 m^3). In the first case the outflow rises within 0.04 of the
    kinematic wave's share of the steady flow until 0.6 t_e; in the second, whose K is smaller, the outflow rounds off
    sooner, 0.1 under it by then. It reaches 0.79 and 0.75 of the steady flow at t_e, where the kinematic wave's turns
    sharply to it, as the other models' in RD-52 do, and is within 0.1% of it by 3 t_e. The steady depths are within
    0.0022 and 0.0012 m of the exact ones more than 5 m from either end; within 0.03 m at the foot, where the
    supercritical water of the second case is drawn down by its leaving at normal depth there."""
    froude, rain, n = RUNOFF_CASES[case]
    steady_flow = rain * RUNOFF_LENGTH
    foot = (steady_flow / (froude * math.sqrt(G))) ** (2.0 / 3.0)
    slope = (steady_flow * n / foot ** (5.0 / 3.0)) ** 2
    assert slope * RUNOFF_LENGTH / (foot * froude ** 2) == pytest.approx(float(case.split("K ")[1]), rel=0.01)
    rise_time = foot / rain
    x = np.arange(RUNOFF_LENGTH) + 0.5
    times = np.arange(0.05, 3.0 + 1e-9, 0.05) * rise_time
    out = unsteady((slope * (RUNOFF_LENGTH - x))[None, :], n, 1.0, times, s_out=slope, rain=rain)
    share = out["outflow"] / steady_flow
    kinematic = np.minimum(times / rise_time, 1.0) ** (5.0 / 3.0)
    exact = runoff_steady(x, rain, n, slope, foot)
    inside = (x > 5.0) & (x < RUNOFF_LENGTH - 5.0)

    assert out["h"][-1].sum() + out["volume_out"][-1] == pytest.approx(out["volume_rain"][-1], abs=1e-12)
    if froude < 1.0:
        assert np.abs(share - kinematic)[times <= 0.6 * rise_time].max() < 0.04
    assert share[19] == pytest.approx(0.8 if froude < 1.0 else 0.75, abs=0.05)
    assert share[-1] == pytest.approx(1.0, abs=0.001)
    assert np.abs(out["h"][-1, 0] - exact)[inside].max() < 0.003 and np.abs(out["h"][-1, 0] - exact).max() < 0.03


# --- The Coriolis force (RD-52 test 4.2.9) --------------------------------------------------------------------------
#
# RD-52's channel is 100 km long and 5 km wide, flat, 100 m deep, n 0.01, carrying 500,000 m^3/s (1 m/s) on 200 m
# cells at 45 degrees north and south, its steady water surface held at the outlet. Across it, the Coriolis force on
# the flow along it is balanced by the water surface's slope, f U / g = 1.0512e-5 (geostrophic balance), the water
# higher on the flow's right in the north. Here the channel is 20 km long, the water starting level at 1 m/s, its
# slope across the middle averaged over the day's last hour (the water sloshing across it every 320 s).

CORIOLIS_LENGTH, CORIOLIS_WIDTH, CORIOLIS_CELL, CORIOLIS_DEPTH, CORIOLIS_SPEED = 20_000.0, 5000.0, 200.0, 100.0, 1.0
EARTH_ROTATION = 7.2921159e-5  # rad/s


@pytest.mark.parametrize("latitude", [45.0, -45.0])
def test_the_coriolis_force_tilts_a_channel_s_water_across_it(latitude: float) -> None:
    """Across the channel's middle, the water surface slopes at f U / g to within 0.5% (0.17% under; HEC-RAS: 0.01%
    to 0.11% under), up towards the flow's right in the north and its left in the south (the rows run south)."""
    coriolis = 2.0 * EARTH_ROTATION * math.sin(math.radians(latitude))
    rows, columns = int(CORIOLIS_WIDTH / CORIOLIS_CELL), int(CORIOLIS_LENGTH / CORIOLIS_CELL)
    across = (np.arange(rows) + 0.5) * CORIOLIS_CELL
    times = 86_400.0 - 3600.0 + 20.0 * np.arange(181)
    out = unsteady(np.zeros((rows, columns)), 0.01, CORIOLIS_CELL, times, inflow_rows=np.arange(rows),
                   inflow=lambda t: CORIOLIS_SPEED * CORIOLIS_DEPTH * CORIOLIS_WIDTH, stage_out=CORIOLIS_DEPTH,
                   h0=np.full((rows, columns), CORIOLIS_DEPTH), u0=np.full((rows, columns + 1), CORIOLIS_SPEED),
                   coriolis=coriolis)
    tilt = np.mean([np.polyfit(across, depths[:, columns // 2], 1)[0] for depths in out["h"]])

    assert tilt == pytest.approx(coriolis * CORIOLIS_SPEED / G, rel=0.005)


def test_the_coriolis_force_turns_water_in_a_wide_basin_as_an_inertial_oscillation() -> None:
    """Water set moving at 0.1 m/s in a basin 200 km across, 10 m deep, flat and frictionless, at 45 degrees north,
    turns to its right at the Coriolis parameter's rate in the basin's middle, out of reach of the walls' waves for
    the first two hours: u = U cos(f t), v = U sin(f t) along the rows, to within 0.0005 m/s. Each component turns
    the other, so this checks both terms."""
    coriolis = 2.0 * EARTH_ROTATION * math.sin(math.radians(45.0))
    cells, speed = 100, 0.1
    u0 = np.zeros((cells, cells + 1))
    u0[:, 1:-1] = speed
    times = (1800.0, 3600.0, 7200.0)
    out = unsteady(np.zeros((cells, cells)), 1e-6, 2000.0, times, h0=np.full((cells, cells), 10.0), u0=u0,
                   coriolis=coriolis)
    middle = cells // 2
    for k, t in enumerate(times):
        assert out["ux"][k][middle, middle] == pytest.approx(speed * math.cos(coriolis * t), abs=0.0005)
        assert out["uy"][k][middle, middle] == pytest.approx(speed * math.sin(coriolis * t), abs=0.0005)


# --- A channel across the grid (not RD-52's) ------------------------------------------------------------------------
#
# The meandering valley's channel (synthetic_valley.py), straight: 30 m wide at the bed, 2 m deep, its sides 2.5:1,
# n 0.04, its bed falling 0.001 along it, on 10 m cells, carrying 30 m^3/s. It comes in along the grid, crosses it at an
# angle for 1400 m, and leaves along it.

def channel_across_the_grid(angle: float, cell: float = 10.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The DEM, and each cell's distance along the channel and from its centreline."""
    turn = math.radians(angle)
    entry, across = 300.0, 1400.0
    corners = [(0.0, 0.0), (entry, 0.0), (entry + across * math.cos(turn), across * math.sin(turn))]
    corners.append((corners[2][0] + entry, corners[2][1]))
    points = np.vstack([np.c_[np.linspace(a[0], b[0], int(math.dist(a, b) / 0.5), endpoint=False),
                              np.linspace(a[1], b[1], int(math.dist(a, b) / 0.5), endpoint=False)]
                        for a, b in zip(corners[:-1], corners[1:])])
    along = np.r_[0.0, np.cumsum(np.hypot(*np.diff(points, axis=0).T))]
    columns, rows = int(round(corners[3][0] / cell)), int(round((corners[3][1] + 300.0) / cell))
    x, y = (np.arange(columns) + 0.5) * cell, (np.arange(rows) + 0.5) * cell - 150.0
    distance, station = np.empty((rows, columns)), np.empty((rows, columns))
    for c in range(columns):
        near = np.flatnonzero(np.abs(points[:, 0] - x[c]) < 400.0)
        apart = np.hypot(y[:, None] - points[near, 1][None, :], x[c] - points[near, 0][None, :])
        nearest = apart.argmin(axis=1)
        distance[:, c], station[:, c] = apart[np.arange(rows), nearest], along[near][nearest]
    cut = np.clip(2.0 - np.clip(distance - 15.0, 0.0, None) / 2.5, 0.0, None)
    return 103.0 - 0.001 * station - cut, station, distance


@pytest.mark.xfail(strict=True, reason="a channel a few cells wide across the grid carries less than along it: its "
                                       "stair steps cost conveyance, and with advection momentum at each")
@pytest.mark.parametrize("advection", [True, False], ids=["shallow water", "local inertial"])
def test_a_channel_across_the_grid_carries_its_flow_as_one_along_it(advection: bool) -> None:
    """Uniform flow down the channel should stand at the same depth whichever way it crosses the grid: 0.97 m along
    it. At 45 degrees it stands 0.46 m deeper, its water falling 22% faster than its bed in the middle of the
    stretch, still deepening upstream (at 26.6 degrees, 0.44 m and 14%): water turning at each stair step of its banks
    loses momentum. Without advection, 0.10 m deeper (0.18 m at 26.6 degrees), as each cell's water crosses its faces
    along the grid only. So the meandering valley's channel stands 0.41 m higher at 30 m^3/s with advection than
    without, and synthetic_valley.py keeps to the local inertial scheme."""
    depths = []
    for angle in (0.0, 45.0):
        z, station, distance = channel_across_the_grid(angle)
        out = steady(z, 0.04, 10.0, np.flatnonzero(distance[:, 0] < 20.0), 30.0, s_out=0.001, advection=advection)
        middle = (distance < 7.1) & (station > 300.0 + 420.0) & (station < 300.0 + 980.0)
        depths.append(out["h"][middle].mean())

    assert depths[0] == pytest.approx(0.97, abs=0.005)
    assert depths[1] == pytest.approx(depths[0], abs=0.02)
