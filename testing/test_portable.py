from __future__ import annotations

import hashlib
import math

import numpy as np
import pytest
from numba import njit

from arc.portable import DIRECTION_DECIMALS, cos_sin, exp, length, log, rounded, sin_cos

ULP = np.finfo(np.float64).eps


@njit(cache=False)
def _each(f, x):
    out = np.empty_like(x)
    for i in range(x.size):
        out[i] = f(x[i])
    return out


@njit(cache=False)
def _sines_cosines(x):
    s, c = np.empty_like(x), np.empty_like(x)
    for i in range(x.size):
        s[i], c[i] = sin_cos(x[i])
    return s, c


def _ulps(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.abs(a.view(np.int64) - b.view(np.int64))


# Inputs made by IEEE arithmetic alone (linspace, ldexp), so the same floats on every system
ANGLES = np.linspace(-10.0, 10.0, 200_001)
POWERS = np.linspace(-700.0, 700.0, 200_001)
VALUES = np.concatenate([np.ldexp(np.linspace(0.5, 1.0, 4_001), k) for k in range(-40, 41)])


def _digest(*arrays: np.ndarray) -> str:
    return hashlib.sha256(b"".join(np.ascontiguousarray(a).tobytes() for a in arrays)).hexdigest()[:16]


def test_each_function_gives_the_same_bits_on_every_system() -> None:
    """The point of arc.portable: these digests were worked out on Windows and match Linux's (WSL, numba 0.65 and
    0.68). A system whose digest differs works some of them out differently, and ARC's outputs could differ there."""
    sines, cosines = _sines_cosines(ANGLES)
    rounded_cosines, rounded_sines = zip(*(cos_sin(float(a)) for a in ANGLES[::50]))
    assert _digest(sines, cosines) == "eb8e551a55c204eb"
    assert _digest(np.array(rounded_cosines), np.array(rounded_sines)) == "fb821c728b695211"
    assert _digest(_each(exp, POWERS)) == "73bafe0009efcfdf"
    assert _digest(_each(log, VALUES)) == "19fbe50d03edd6c4"


def test_sines_and_cosines_are_within_an_ulp_or_so_of_the_c_library_s() -> None:
    """Both are within an ulp of the true value, so within two of each other (mpmath, at 300 bits, found ours within
    one everywhere on 20,000 angles and every multiple of pi / 4 up to 200)."""
    sines, cosines = _sines_cosines(ANGLES)
    assert _ulps(sines, np.sin(ANGLES)).max() <= 2
    assert _ulps(cosines, np.cos(ANGLES)).max() <= 2
    assert sin_cos(0.0) == (0.0, 1.0) and math.copysign(1.0, sin_cos(-0.0)[0]) == -1.0
    assert sin_cos(math.pi / 2)[1] == pytest.approx(6.123233995736766e-17, abs=1e-30)
    assert all(math.isnan(v) for v in sin_cos(math.nan))


def test_exponentials_and_logarithms_are_within_an_ulp_or_so_of_the_c_library_s() -> None:
    assert _ulps(_each(exp, POWERS), np.exp(POWERS)).max() <= 2
    assert _ulps(_each(log, VALUES), np.log(VALUES)).max() <= 2
    assert exp(0.0) == 1.0 and exp(-math.inf) == 0.0 and exp(math.inf) == math.inf and exp(710.0) == math.inf
    assert exp(-746.0) == 0.0 and 0.0 < exp(-745.0) < 1e-320 and math.isnan(exp(math.nan))
    assert log(1.0) == 0.0 and log(0.0) == -math.inf and math.isnan(log(-1.0)) and log(math.inf) == math.inf
    assert log(2.0) == math.log(2.0) and log(1e-310) == pytest.approx(math.log(1e-310), rel=ULP)


def test_a_logarithm_near_1_keeps_its_digits() -> None:
    """1 is in the middle of its stretch of the table, whose c is 1, so log(1 + d) is d's series exactly worked."""
    for d in (1e-15, -1e-15, 1e-10, -3e-8, 2.0 ** -20, -(2.0 ** -21), 1e-3, -1e-3):
        assert log(1.0 + d) == pytest.approx(math.log1p((1.0 + d) - 1.0), rel=2 * ULP)


def test_a_value_is_rounded_to_the_direction_decimals() -> None:
    unit = 10.0 ** -DIRECTION_DECIMALS
    for value in np.concatenate([np.cos(ANGLES[::100]), [0.0, -0.0, 1.0, -1.0, 0.5e-6, -2.5e-6, 0.4999999e-6]]):
        result = rounded(float(value))
        assert abs(result - value) <= 0.5 * unit * (1 + 1e-9), value  # to the nearest millionth
        assert abs(result / unit - round(result / unit)) < 1e-6, value
    assert math.isnan(rounded(math.nan)) and rounded(math.inf) == math.inf


def test_an_angle_s_cosine_and_sine_are_its_sine_and_cosine_rounded() -> None:
    for angle in ANGLES[::100]:
        s, c = sin_cos(float(angle))
        assert cos_sin(float(angle)) == (rounded(c), rounded(s))


def test_an_angle_a_last_digit_or_two_off_has_the_same_rounded_cosine_and_sine() -> None:
    """Angles come from the C library's atan2, whose last digit differs from system to system: rounded, their cosine
    and sine don't, but where the true value lies within those digits of a rounding boundary (none of these)."""
    for angle in ANGLES[::20]:
        expected = cos_sin(float(angle))
        for ulps in (-2, -1, 1, 2):
            off = float(angle)
            for _ in range(abs(ulps)):
                off = float(np.nextafter(off, np.inf if ulps > 0 else -np.inf))
            assert cos_sin(off) == expected


def test_a_length_is_the_square_root_of_its_squares() -> None:
    rng = np.random.default_rng(1)
    for x, y in rng.uniform(-100.0, 100.0, (5000, 2)):
        assert length(float(x), float(y)) == math.sqrt(x * x + y * y)
        assert length(float(x), float(y)) == pytest.approx(math.hypot(x, y), rel=3e-16)
    assert length(3.0, 4.0) == 5.0
