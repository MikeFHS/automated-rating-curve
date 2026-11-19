"""
Sines, cosines, exponentials, logarithms and lengths worked out the same, to the last digit, on every system.

The C math library's sin, cos, atan2, exp, log and hypot (numpy's float64 ones and numba's call it too) differ in
their last digit from one system's to another's: between Windows' and Linux's, in about 1 in 30 sines and cosines,
1 in 750 angles, 1 in 200 exponentials, 1 in 4,000 logarithms and 1 in 6 hypot's. A cross section's ordinates step
exactly a cell along the axis it crosses fastest, so such a digit can put an ordinate on the raster's edge row on one
system and off it on the other; and on flat ground, where a hair's change in a water surface floods over a rim, a last
digit in a bed's depth or a valley's smoothing moved a top width by up to 100 m.

So ARC has its own: sin_cos, exp and log, ported from fdlibm (Sun's freely distributable libm, which FreeBSD's and many
others grew from) and fast C libraries' tables, with additions, multiplications and bit operations alone, which IEEE
arithmetic rounds the same everywhere (numba doesn't fuse a multiplication and an addition into one rounding unless
asked to, with fastmath). Each is within an ulp of the true value and at least as fast as the C library's, compiled, on
Windows and on Linux (an i7-14700, ns a call: sin and cos together 12 and 11 against 18 and 19, exp 3.1 and 2.6
against 3.8 and 3.7, log 3.5 and 2.8 against 3.5 and 3.2). Lengths are sqrt(x*x + y*y), which every system rounds
correctly (1.2 to 1.5 ns against hypot's 3 on Windows and 12 on Linux).

Angles still come from the C library's atan2. So a cross section's cosine and sine (cos_sin) are rounded to
DIRECTION_DECIMALS decimals (a millionth: under a millimetre across a kilometre), which takes up an angle a last digit
off: of 4 million directions made by atan2 as ARC makes them, 5,350 angles differed between Windows and Linux, but
none of their cosines and sines from 9 decimals on (about one in 10^10 is left at 6). Directions themselves aren't
rounded, which keeps them exact where they can be (pi / 2 down a column; pi / 2 isn't a whole number of millionths).
"""
import math
from decimal import Decimal, localcontext

import numpy as np
from numba import njit

# The decimals a direction's cosine and sine are rounded to
DIRECTION_DECIMALS = 6
_SCALE = 1e6  # 10 ** DIRECTION_DECIMALS
_UNIT = 1e-6  # 1 / _SCALE, as near as a float gets
# Added and taken away again, it rounds a float below 2 ** 51 to the nearest whole number (ties to even): IEEE
# addition rounds the sum to a whole number, as no float this large has a fraction. Two additions, faster than rint.
_ROUNDER = 6755399441055744.0  # 1.5 * 2 ** 52


@njit(cache=True, error_model="numpy")
def rounded(value):
    """value to DIRECTION_DECIMALS decimals, alike on every system (for |value| < 2e9): a product, two additions and
    a product, which IEEE arithmetic rounds correctly everywhere. NaN and infinities stay as they are."""
    return ((value * _SCALE + _ROUNDER) - _ROUNDER) * _UNIT


@njit(cache=True, error_model="numpy")
def cos_sin(angle):
    """The cosine and sine of an angle (radians), rounded (see the notes above). From plain Python, a call costs
    about what a Python function calling math.cos and math.sin does."""
    s, c = sin_cos(angle)
    return rounded(c), rounded(s)


@njit(cache=True, error_model="numpy")
def length(x, y):
    """The length of the vector (x, y): sqrt(x*x + y*y), alike everywhere, where the C library's hypot isn't."""
    return math.sqrt(x * x + y * y)


# --- exp and log ----------------------------------------------------------------------------------------------------
# Additions, multiplications and bit operations alone, which IEEE arithmetic rounds the same everywhere (numba doesn't
# fuse a multiplication and an addition into one rounding unless asked to, with fastmath).
#
# exp(x) = 2 ** (k / 128) * exp(r): k the whole number nearest x * 128 / ln 2, so |r| <= ln 2 / 256, and exp(r) - 1
# by Taylor's series to r ** 5 (the next term is under 1e-18 of it). 2 ** (j / 128) is a table, each a float and the
# float's error, for j = k mod 128, then scaled by 2 ** (k // 128).
#
# log(x) = e ln 2 + log(c) + log(1 + r): x = 2 ** e * z with z from 0.69 up to 1.38, found without a branch by taking
# 0.689453125's bits from x's (the difference's top bits are e, the next eight the stretch of z's 256 it's in). c is
# the stretch's middle, and r = (z - c) / c, worked out without dividing as (z - c) * (1 / c), the reciprocal a float
# and its error (z - c is exact, as z and c are within a factor 2 of each other). |r| < 1 / 512, and log(1 + r) by its
# series to r ** 6. log(c) is a table too. 1 is in the middle of its stretch, whose c is 1, so that near 1 r is z - 1
# exactly and the logarithm keeps its digits.
#
# The tables are worked out once, here, by Python's decimal module (exact to 40 digits, and its own code on every
# system), and rounded to floats as Python rounds them, correctly. Outside the tables' range (exp's arguments past
# +-708, log's below 2 ** -1022, at or below 0, infinite or NaN) both fall back to fdlibm's (Sun's freely distributable
# libm, which FreeBSD's and many others grew from), line for line, as are its constants below.
_EXP_TABLE_BITS = 7
_LOG_TABLE_BITS = 8
_LOG_OFFSET = np.int64(0x3fe6080000000000)  # 0.689453125's bits: 1 falls in the middle of a stretch
_TOP_12_BITS = np.int64(-(1 << 52))  # 0xfff0000000000000


def _log_middles() -> list[float]:
    """The middle of each of log's 256 stretches of z, exactly, as floats: 1 for the stretch 1 is in, which spans
    two binades, and so isn't its middle."""
    middles = []
    for i in range(1 << _LOG_TABLE_BITS):
        start = float(np.int64(_LOG_OFFSET + (np.int64(i) << (52 - _LOG_TABLE_BITS))).view(np.float64))
        end = float(np.int64(_LOG_OFFSET + (np.int64(i + 1) << (52 - _LOG_TABLE_BITS))).view(np.float64))
        middles.append(1.0 if start <= 1.0 < end else (start + end) / 2.0)
    return middles


def _tables():
    with localcontext() as ctx:
        ctx.prec = 40
        ln2 = Decimal(2).ln()
        n = 1 << _EXP_TABLE_BITS
        powers = [(ln2 * j / n).exp() for j in range(n)]
        exp_hi = np.array([float(p) for p in powers])
        exp_lo = np.array([float(p - Decimal(float(p))) for p in powers])
        # ln 2 / 128 in two: a high part of 36 bits, so that k times it is exact for |k| < 2 ** 17, and the rest
        step = ln2 / n
        bits = np.float64(float(step)).view(np.int64) & ~np.int64((1 << 17) - 1)
        step_hi = float(np.int64(bits).view(np.float64))
        step_lo = float(step - Decimal(step_hi))
        # Each of log's stretches' middle c, 1 / c in two parts, and log(c) in two: a high part a whole number of
        # 2 ** -32, as fdlibm's ln 2 high part is, so that e times that plus it is exact, and the rest
        middles = _log_middles()
        log_table = np.empty((len(middles), 5))
        for j, c in enumerate(middles):
            inverse, logc = 1 / Decimal(c), Decimal(c).ln()
            logc_hi = float((logc * 2 ** 32).to_integral_value()) / 2 ** 32
            log_table[j] = (c, float(inverse), float(inverse - Decimal(float(inverse))), logc_hi,
                            float(logc - Decimal(logc_hi)))
    return exp_hi, exp_lo, step_hi, step_lo, log_table.ravel()


# log's table: for each stretch of z in turn, c, 1 / c's two parts and log(c)'s two parts
_EXP_HI, _EXP_LO, _STEP_HI, _STEP_LO, _LOG_TABLE = _tables()
_EXP_N_OVER_LN2 = float.fromhex("0x1.71547652b82fep7")  # 128 / ln 2, as fdlibm's 1 / ln 2 (0x3ff71547 652b82fe) times 128
_EXP_LIMIT = 708.0  # within it, exp(x) and its table's 2 ** (k // 128) are normal floats
# A positive normal float's bits, less 2 ** -1022's, are below this; any other float's (0, negatives, subnormals,
# infinity, NaN), taken as unsigned, aren't
_LOWEST_NORMAL_BITS = np.uint64(0x0010000000000000)
_NORMAL_SPAN = np.uint64(0x7fe0000000000000)


@njit(cache=True, error_model="numpy")
def exp(x):
    """e ** x, the same on every system (see the notes above)."""
    if not abs(x) < _EXP_LIMIT:
        return _exp_fdlibm(x)  # infinities, NaN, overflow, results below the normal floats

    shifted = x * _EXP_N_OVER_LN2 + _ROUNDER
    k = np.float64(shifted).view(np.int64) - np.float64(_ROUNDER).view(np.int64)  # the whole number nearest
    kd = shifted - _ROUNDER
    r = (x - kd * _STEP_HI) - kd * _STEP_LO
    r2 = r * r
    p = r + r2 * (0.5 + r * (1.0 / 6.0) + r2 * (1.0 / 24.0 + r * (1.0 / 120.0)))  # exp(r) - 1
    j = k & 127
    hi, lo = _EXP_HI[j], _EXP_LO[j]
    y = hi + (lo + (hi * p + lo * p))
    return y * np.int64((0x3ff + (k >> _EXP_TABLE_BITS)) << 52).view(np.float64)


@njit(cache=True, error_model="numpy")
def log(x):
    """The natural logarithm of x, the same on every system (see the notes above)."""
    bits = np.float64(x).view(np.int64)
    if not np.uint64(bits) - _LOWEST_NORMAL_BITS < _NORMAL_SPAN:  # one comparison, unsigned
        return _log_fdlibm(x)  # 0, negatives, subnormals, infinity, NaN

    offset = bits - _LOG_OFFSET
    k = ((offset >> (52 - _LOG_TABLE_BITS)) & ((1 << _LOG_TABLE_BITS) - 1)) * 5  # z's stretch's row
    e = offset >> 52
    z = np.int64(bits - (offset & _TOP_12_BITS)).view(np.float64)  # x / 2 ** e, from 0.69 up to 1.38
    d = z - _LOG_TABLE[k]  # exact
    r = d * _LOG_TABLE[k + 1] + d * _LOG_TABLE[k + 2]
    r2 = r * r
    r4 = r2 * r2
    # log(1 + r) - r, by its series to r ** 6 (Estrin's scheme; the next term is under 1e-17 of r)
    tail = r2 * ((-0.5 + r * (1.0 / 3.0)) + r2 * (-0.25 + r * 0.2) + r4 * (-1.0 / 6.0))
    ed = float(e)
    hi = ed * _LN2_HI + _LOG_TABLE[k + 3]  # exact
    return hi + (r + (tail + (ed * _LN2_LO + _LOG_TABLE[k + 4])))


# fdlibm's constants
_LN2_HI = 6.93147180369123816490e-01  # 0x3fe62e42 fee00000
_LN2_LO = 1.90821492927058770002e-10  # 0x3dea39ef 35793c76
_INV_LN2 = 1.44269504088896338700e+00  # 0x3ff71547 652b82fe
_P1 = 1.66666666666666019037e-01  # 0x3FC55555 5555553E
_P2 = -2.77777777770155933842e-03  # 0xBF66C16C 16BEBD93
_P3 = 6.61375632143793436117e-05  # 0x3F11566A AF25DE2C
_P4 = -1.65339022054652515390e-06  # 0xBEBBBD41 C5D26BF1
_P5 = 4.13813679705723846039e-08  # 0x3E663769 72BEA4D0
_O_THRESHOLD = 7.09782712893383973096e+02  # 0x40862E42 FEFA39EF
_U_THRESHOLD = -7.45133219101941108420e+02  # 0xc0874910 D52D3051
_TWO54 = 1.80143985094819840000e+16  # 0x43500000 00000000
_LG1 = 6.666666666666735130e-01  # 3FE55555 55555593
_LG2 = 3.999999999940941908e-01  # 3FD99999 9997FA04
_LG3 = 2.857142874366239149e-01  # 3FD24924 94229359
_LG4 = 2.222219843214978396e-01  # 3FCC71C5 1D8E78AF
_LG5 = 1.818357216161805012e-01  # 3FC74664 96CB03DE
_LG6 = 1.531383769920937332e-01  # 3FC39A09 D078C69F
_LG7 = 1.479819860511658591e-01  # 3FC2F112 DF3E5244


@njit(cache=True, error_model="numpy")
def _high_word(x):
    return np.int64(np.float64(x).view(np.int64) >> 32)


@njit(cache=True, error_model="numpy")
def _power_of_two(k):
    """2 ** k for -1022 <= k <= 1023, exactly."""
    return np.int64((0x3ff + k) << 52).view(np.float64)


@njit(cache=True, error_model="numpy")
def _exp_fdlibm(x):
    """e ** x, the same on every system (fdlibm's __ieee754_exp)."""
    hx = _high_word(x)
    negative = hx < 0
    hx &= 0x7fffffff
    if hx >= 0x40862E42:  # |x| >= 709.78..., or not a number
        if hx >= 0x7ff00000:
            if x != x:  # noqa: PLR0124
                return x
            return 0.0 if negative else x  # exp(-inf) = 0, exp(inf) = inf
        if x > _O_THRESHOLD:
            return np.inf
        if x < _U_THRESHOLD:
            return 0.0

    k = 0
    hi = lo = 0.0
    if hx > 0x3fd62e42:  # |x| > ln2 / 2
        if hx < 0x3FF0A2B2:  # and |x| < 1.5 ln2
            if negative:
                hi, lo, k = x + _LN2_HI, -_LN2_LO, -1
            else:
                hi, lo, k = x - _LN2_HI, _LN2_LO, 1
        else:
            k = int(_INV_LN2 * x + (-0.5 if negative else 0.5))
            t = float(k)
            hi = x - t * _LN2_HI  # exact: t * _LN2_HI is
            lo = t * _LN2_LO
        x = hi - lo
    elif hx < 0x3e300000:  # |x| < 2 ** -28
        return 1.0 + x

    t = x * x
    c = x - t * (_P1 + t * (_P2 + t * (_P3 + t * (_P4 + t * _P5))))
    if k == 0:
        return 1.0 - ((x * c) / (c - 2.0) - x)

    y = 1.0 - ((lo - (x * c) / (2.0 - c)) - hi)
    if k >= -1021:
        if k == 1024:
            return y * 2.0 * _power_of_two(1023)
        return y * _power_of_two(k)
    return y * _power_of_two(k + 1000) * _power_of_two(-1000)


@njit(cache=True, error_model="numpy")
def _log_fdlibm(x):
    """The natural logarithm of x, the same on every system (fdlibm's __ieee754_log)."""
    bits = np.float64(x).view(np.int64)
    hx = np.int64(bits >> 32)
    lx = bits & 0xffffffff
    k = 0
    if hx < 0x00100000:  # x < 2 ** -1022
        if ((hx & 0x7fffffff) | lx) == 0:
            return -np.inf  # log(+-0)
        if hx < 0:
            return np.nan  # log of a negative number
        k -= 54
        x *= _TWO54  # a subnormal number, scaled up
        bits = np.float64(x).view(np.int64)
        hx = np.int64(bits >> 32)
        lx = bits & 0xffffffff
    if hx >= 0x7ff00000:
        return x + x
    k += (hx >> 20) - 1023
    hx &= 0x000fffff
    i = (hx + 0x95f64) & 0x100000
    x = np.int64(((hx | (i ^ 0x3ff00000)) << 32) | lx).view(np.float64)  # x or x / 2, normalised
    k += i >> 20
    f = x - 1.0
    if (0x000fffff & (2 + hx)) < 3:  # -2 ** -20 <= f < 2 ** -20
        if f == 0.0:
            if k == 0:
                return 0.0
            dk = float(k)
            return dk * _LN2_HI + dk * _LN2_LO
        r = f * f * (0.5 - 0.33333333333333333 * f)
        if k == 0:
            return f - r
        dk = float(k)
        return dk * _LN2_HI - ((r - dk * _LN2_LO) - f)

    s = f / (2.0 + f)
    dk = float(k)
    z = s * s
    i = hx - 0x6147a
    w = z * z
    j = 0x6b851 - hx
    t1 = w * (_LG2 + w * (_LG4 + w * _LG6))
    t2 = z * (_LG1 + w * (_LG3 + w * (_LG5 + w * _LG7)))
    i |= j
    r = t2 + t1
    if i > 0:
        hfsq = 0.5 * f * f
        if k == 0:
            return f - (hfsq - s * (hfsq + r))
        return dk * _LN2_HI - ((hfsq - (s * (hfsq + r) + dk * _LN2_LO)) - f)
    if k == 0:
        return f - s * (f - r)
    return dk * _LN2_HI - ((s * (f - r) - dk * _LN2_LO) - f)


# --- sin and cos ----------------------------------------------------------------------------------------------------
# fdlibm's kernels for the sine and cosine on [-pi / 4, pi / 4], and its reduction of an angle by a whole number of
# quarter turns (Cody and Waite's, pi / 2 in three parts, exact for up to 2 ** 19 quarter turns), line for line. Both
# come from one reduction, and the quarter turns' swap and signs are chosen without a branch: faster than the C
# library's sin and cos together. Past 2 ** 19 quarter turns (or for infinities and NaN), the C library's, which no
# angle ARC makes reaches.
_INV_PIO2 = 6.36619772367581382433e-01  # 0x3FE45F30 6DC9C883
_PIO2_1 = 1.57079632673412561417e+00  # 0x3FF921FB 54400000
_PIO2_1T = 6.07710050650619224932e-11  # 0x3DD0B461 1A626331
_PIO2_2 = 6.07710050630396597660e-11  # 0x3DD0B461 1A600000
_PIO2_2T = 2.02226624879595063154e-21  # 0x3BA3198A 2E037073
_PIO2_3 = 2.02226624871116645580e-21  # 0x3BA3198A 2E000000
_PIO2_3T = 8.47842766036889956997e-32  # 0x397B839A 252049C1
_S1 = -1.66666666666666324348e-01  # 0xBFC55555 55555549
_S2 = 8.33333333332248946124e-03  # 0x3F811111 1110F8A6
_S3 = -1.98412698298579493134e-04  # 0xBF2A01A0 19C161D5
_S4 = 2.75573137070700676789e-06  # 0x3EC71DE3 57B1FE7D
_S5 = -2.50507602534068634195e-08  # 0xBE5AE5E6 8A2B9CEB
_S6 = 1.58969099521155010221e-10  # 0x3DE5D93A 5ACFD57C
_C1 = 4.16666666666666019037e-02  # 0x3FA55555 5555554C
_C2 = -1.38888888888741095749e-03  # 0xBF56C16C 16C15177
_C3 = 2.48015872894767294178e-05  # 0x3EFA01A0 19CB1590
_C4 = -2.75573143513906633035e-07  # 0xBE927E4F 809C52AD
_C5 = 2.08757232129817482790e-09  # 0x3E21EE9E BDB4B1C4
_C6 = -1.13596475577881948265e-11  # 0xBDA8FAE9 BE8838D4
_SINCOS_LIMIT = 823549.6  # 2 ** 19 quarter turns


@njit(cache=True, error_model="numpy")
def sin_cos(x):
    """The sine and cosine of x (radians), the same on every system, unrounded (see the notes above)."""
    if not abs(x) < _SINCOS_LIMIT:
        return math.sin(x), math.cos(x)

    # x less n quarter turns, as y0 + y1 (fdlibm's __ieee754_rem_pio2, for medium-sized angles)
    fn = (x * _INV_PIO2 + _ROUNDER) - _ROUNDER
    n = np.int64(fn)
    r = x - fn * _PIO2_1
    w = fn * _PIO2_1T
    y0 = r - w
    j = (_high_word(x) >> 20) & 0x7ff
    if j - ((_high_word(y0) >> 20) & 0x7ff) > 16:  # cancellation: a second round
        t = r
        w = fn * _PIO2_2
        r = t - w
        w = fn * _PIO2_2T - ((t - r) - w)
        y0 = r - w
        if j - ((_high_word(y0) >> 20) & 0x7ff) > 49:  # and a third
            t = r
            w = fn * _PIO2_3
            r = t - w
            w = fn * _PIO2_3T - ((t - r) - w)
            y0 = r - w
    y1 = (r - y0) - w

    # fdlibm's __kernel_sin(y0, y1, 1) and __kernel_cos(y0, y1)
    z = y0 * y0
    zz = z * z
    rs = _S2 + z * (_S3 + z * _S4) + z * zz * (_S5 + z * _S6)
    v = z * y0
    s = y0 - ((z * (0.5 * y1 - v * rs) - y1) - v * _S1)
    rc = z * (_C1 + z * (_C2 + z * _C3)) + zz * zz * (_C4 + z * (_C5 + z * _C6))
    hz = 0.5 * z
    one_less = 1.0 - hz
    c = one_less + (((1.0 - one_less) - hz) + (z * rc - y0 * y1))

    # Back by the n quarter turns: an odd n swaps them, and n mod 4 of 2 or 3 turns both over
    odd = (n & 1) == 1
    sign = 1.0 - float(n & 2)
    return sign * (c if odd else s), sign * (-s if odd else c)
