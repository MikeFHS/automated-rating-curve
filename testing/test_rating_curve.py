from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.optimize import brentq

from arc.hydraulics import DepthRoughness, discharge, wse_for_discharge
from arc.rating_curve import WSE, Q, V, max_flow_wse, rating_curve
from arc.xsection.xsection import XSection

BED = 100.0
N = 0.035
SLOPE = 0.001


def make_section(elevations, mannings_n=N, spacing=1.0) -> XSection:
    elevations = np.asarray(elevations, dtype=np.float64)
    return XSection(elevations, np.full(elevations.size, mannings_n), spacing)


def trapezoid(bottom_width=10.0, side_slope=2.0, half=60, spacing=1.0) -> XSection:
    x = spacing * np.arange(-half, half + 1)
    return make_section(BED + np.maximum(np.abs(x) - bottom_width / 2, 0.0) / side_slope, spacing=spacing)


def trapezoid_discharge(depth, bottom=10.0, side_slope=2.0, n=N, slope=SLOPE) -> float:
    top = bottom + 2 * side_slope * depth
    area = depth * (bottom + top) / 2
    perimeter = bottom + 2 * depth * math.hypot(1.0, side_slope)
    return area * (area / perimeter) ** (2 / 3) * math.sqrt(slope) / n


def channel_and_hollow() -> XSection:
    """A 10 m wide channel 2 m deep, whose right bank tops out at 102 m in front of a hollow 80 m wide whose floor
    is 1 m lower, so that at 102 m the water spills into it."""
    distance = np.arange(0.0, 101.0)
    left = BED + np.clip(distance - 5.0, 0.0, 2.0) + 0.1 * np.maximum(distance - 7.0, 0.0)
    right = BED + np.clip(distance - 5.0, 0.0, 2.0)
    right[8:88] = BED + 1.0
    right[88:] = BED + 10.0
    return make_section(np.concatenate([left[:0:-1], right]))


def test_the_last_increment_carries_the_maximum_flow_at_the_manning_depth() -> None:
    """A prismatic trapezoid, where the sampled cross section is exact."""
    xs = trapezoid()
    q_max = 30.0
    curve = rating_curve(xs, q_max, SLOPE, 10)
    depth = brentq(lambda y: trapezoid_discharge(y) - q_max, 1e-6, 20.0, xtol=1e-14)

    assert curve.valid and (curve.start, curve.last) == (0, 10)
    assert curve.max_wse == pytest.approx(BED + depth, abs=1e-9)
    assert curve.increments[-1, Q] == pytest.approx(q_max, rel=1e-9)
    assert np.allclose(np.diff(curve.increments[:, WSE]), depth / 10, atol=1e-12)
    for q, v, t, wse, p in curve.increments:
        y = wse - BED
        assert q == pytest.approx(trapezoid_discharge(y), rel=1e-9)
        assert t == pytest.approx(10.0 + 4.0 * y, rel=1e-12)
        assert p == pytest.approx(10.0 + 2 * y * math.hypot(1.0, 2.0), rel=1e-12)
        assert v == pytest.approx(q / (y * (10.0 + 2.0 * y)), rel=1e-12)


def test_the_slope_factor_and_roughness_reach_the_rating_curve() -> None:
    xs = trapezoid()
    roughness = DepthRoughness()
    curve = rating_curve(xs, 30.0, SLOPE, 8, roughness=roughness, slope_factor=1.3)

    assert curve.max_wse == pytest.approx(wse_for_discharge(xs, 30.0, SLOPE, roughness=roughness, slope_factor=1.3))
    for q, _, _, wse, _ in curve.increments:
        assert q == pytest.approx(discharge(xs, SLOPE, wse=wse, roughness=roughness, slope_factor=1.3), rel=1e-12)
    # Rougher shallow water needs a deeper channel for the same flow
    assert curve.max_wse > rating_curve(xs, 30.0, SLOPE, 8, slope_factor=1.3).max_wse


def test_the_first_increment_carries_just_under_the_baseflow() -> None:
    """At its own water surface, with its velocity that discharge over its area there."""
    xs = trapezoid()
    plain = rating_curve(xs, 30.0, SLOPE, 10)
    first = plain.increments[0, Q]
    area = first / plain.increments[0, V]

    above = rating_curve(xs, 30.0, SLOPE, 10, baseflow=first * 0.5)
    below = rating_curve(xs, 30.0, SLOPE, 10, baseflow=first * 2.0)

    assert above.increments[0, Q] == pytest.approx(first * 0.5 - 0.001)
    assert above.increments[0, V] == pytest.approx(above.increments[0, Q] / area, rel=1e-12)
    assert np.array_equal(np.delete(above.increments[0], [Q, V]), np.delete(plain.increments[0], [Q, V]))
    assert np.array_equal(above.increments[1:], plain.increments[1:])
    assert np.array_equal(below.increments, plain.increments)
    assert np.array_equal(rating_curve(xs, 30.0, SLOPE, 10, baseflow=0.0005).increments, plain.increments)


def test_no_flow_or_no_increments_give_a_curve_without_any() -> None:
    xs = trapezoid()

    for curve in (rating_curve(xs, 0.0, SLOPE, 10), rating_curve(xs, math.nan, SLOPE, 10),
                  rating_curve(xs, 30.0, SLOPE, 0)):
        assert (curve.start, curve.last) == (-1, 0) and not curve.valid
        assert np.isnan(curve.increments).all()


def test_a_section_that_can_t_carry_the_flow_has_no_rating_curve() -> None:
    walls = make_section([9999.0, BED, 9999.0])
    walls.mannings_n[[0, 2]] = 9999.0

    assert rating_curve(trapezoid(), 30.0, 0.0, 10) is None  # no slope
    assert math.isnan(max_flow_wse(trapezoid(), 30.0, -1.0))
    assert rating_curve(make_section([BED]), 30.0, SLOPE, 10) is None  # no width at all


def test_a_spill_far_below_the_maximum_flow_isn_t_acceptable() -> None:
    """Just below the bank top the channel carries less than half the maximum flow, so the cross section gets no
    rating curve."""
    xs = channel_and_hollow()
    below = discharge(xs, SLOPE, wse=102.0)
    above = discharge(xs, SLOPE, wse=102.0 + 1e-9)
    assert above > 3 * below

    assert wse_for_discharge(xs, 2.5 * below, SLOPE) == 102.0
    assert rating_curve(xs, 2.5 * below, SLOPE, 10) is None
    curve = rating_curve(xs, 1.5 * below, SLOPE, 10)
    assert curve is not None and curve.max_wse == 102.0
    assert curve.increments[-1, Q] == pytest.approx(below, rel=1e-12)  # the channel's, just below the spill


def test_where_discharge_falls_the_last_increment_is_repeated_or_the_water_raised_a_centimetre() -> None:
    """Water spreading over a flat floodplain beside a small V channel carries less as it first rises above it."""
    distance = np.abs(np.arange(-60.0, 61.0))
    xs = make_section(np.where(distance > 52, BED + 2.0 + (distance - 52),
                               np.where(distance <= 2, BED + distance, BED + 2.0)))
    assert discharge(xs, SLOPE, wse=BED + 2.05) < discharge(xs, SLOPE, wse=BED + 2.0)
    q_max = discharge(xs, SLOPE, wse=BED + 2.5)
    outcomes = set()
    for count in range(5, 80):
        curve = rating_curve(xs, q_max, SLOPE, count)
        increments = curve.increments
        assert np.all(np.diff(increments[:, Q]) >= 0.0)
        assert increments[-1, Q] == pytest.approx(q_max, rel=1e-9)
        step = (curve.max_wse - BED) / count
        for i in range(1, count):
            if np.array_equal(increments[i], increments[i - 1]):
                outcomes.add("repeated")
            elif not math.isclose(increments[i, WSE], BED + step * (i + 1), abs_tol=1e-9):
                outcomes.add("raised")
                assert increments[i, WSE] < BED + step * (i + 2)  # no higher than the next increment
    assert outcomes == {"repeated", "raised"}
