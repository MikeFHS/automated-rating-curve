"""How deep a cross section's channel is below its reference level, from baseflow or a drainage-area power law.

The reference level is the water surface the DEM shows at the stream cell, or with bank elevations (Bathy_Use_Banks)
the smoothed bank elevation. The channel is a trapezoid as wide as its banks at the reference level, whose sloping
sides each take up trapezoid_height (Bathy_Trap_H) of that width, however wide it is, narrower than a cell included.
A channel without valid banks is one cell wide (single_cell_banks). Its depth is the one at which the cross section,
carved that deep (arc.bathymetry.channel), carries the baseflow in its channel with its water at the reference level,
by the same hydraulics as its rating curve: Manning's n varying with depth as Depth_Varying_N says, the n the channel
has between its banks (arc.pipeline gives the water class's n from the Manning's n table), and the
slope_adjustment_factor. So at normal depth the rating curve carries the baseflow exactly at the reference level,
whatever those settings. A drainage-area depth, where there is one, is used as it is instead.
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
from numba import njit

from arc.bathymetry.banks import Banks, single_cell_banks
from arc.xsection.xsection import XSection

if TYPE_CHECKING:
    from arc.hydraulics import DepthRoughness

DEFAULT_MANNINGS_N = 0.03  # the channel's roughness when none is given


class BathymetryDepth(NamedTuple):
    depth: float
    apply: bool  # whether to carve the channel at all
    source: str  # "target_depth" or "baseflow_manning"


@njit(cache=True, error_model="numpy")
def _newton_or_bisect(u, step, lo, hi):
    """The next u after a Newton step from u, bisecting if the step leaves the bracket [lo, hi], and whether that
    is close enough. Newton's method converges quadratically, so a step under 1e-9 leaves an error of about 1e-18; a
    bisection needs the bracket itself small. A step too small to move u stays in the bracket."""
    newton = u - step
    if lo <= newton <= hi:
        return newton, abs(step) <= 1e-9 * (1.0 + abs(u))
    middle = 0.5 * (lo + hi)
    return middle, hi - lo <= 1e-12 * (1.0 + abs(middle))


@njit(cache=True, error_model="numpy")
def _log_conveyance(y, bottom, top):
    """ln(A**(5/3) / P**(2/3)) of the trapezoid at depth y, and its derivative with respect to y."""
    side = 0.5 * (top - bottom)
    area = 0.5 * y * (bottom + top)
    slant = math.sqrt(side * side + y * y)
    perimeter = bottom + 2.0 * slant
    return (5.0 / 3.0) * math.log(area) - (2.0 / 3.0) * math.log(perimeter), \
        (5.0 / 3.0) / y - (4.0 / 3.0) * y / (slant * perimeter)


@njit(cache=True, error_model="numpy")
def trapezoid_depth(q, bottom_width, top_width, slope, mannings_n):
    """The depth at which a trapezoid with these bottom and top widths carries q by Manning's equation: 0 for no
    flow, NaN for no slope or widths that aren't a trapezoid's."""
    if not top_width > 0.0 or not 0.0 <= bottom_width <= top_width:
        return math.nan
    if not q > 0.0:
        return 0.0
    if not slope > 0.0 or not mannings_n > 0.0:
        return math.nan
    
    target = math.log(q * mannings_n / math.sqrt(slope))

    # Start from shallow water across the whole top width W, where A = W_mean * y and P is about W, so that
    # ln K = 5/3 * ln(W_mean * y) - 2/3 * ln W
    u = 0.6 * target + 0.4 * math.log(top_width) - math.log(0.5 * (bottom_width + top_width))

    # ln K rises with ln y at a rate between 1 and 8/3, so Newton's method on ln y converges in a few steps. It
    # bisects if a step would leave the bracket of the steps so far, which rounding might cause.
    u_lo, u_hi = -np.inf, np.inf
    for _ in range(100):
        y = math.exp(u)
        value, slope_in_y = _log_conveyance(y, bottom_width, top_width)
        error = value - target
        if error == 0.0:
            break
        if error > 0.0:
            u_hi = u
        else:
            u_lo = u
        u, converged = _newton_or_bisect(u, error / (y * slope_in_y), u_lo, u_hi)
        if converged or not math.isfinite(u):
            break

    return math.exp(u)


def channel_depth(xs: XSection, banks: Banks, q: float, slope: float, *, trapezoid_height: float,
                  mannings_n: float = DEFAULT_MANNINGS_N, roughness: DepthRoughness | None = None,
                  slope_factor: float = 1.0, bank_elevation: float | None = None,
                  divide: Banks | None = None) -> float:
    """How deep the channel is carved below its reference level, bank_elevation or else the stream cell's elevation,
    for the cross section to carry the baseflow q there by its rating curve's hydraulics (see the notes above and
    arc.bathymetry.channel.carved_depth): with roughness, the rating curves' DepthRoughness (None for a constant n),
    and slope_factor, divided at `divide` if the hydraulics don't divide it at the banks it's carved between. The
    trapezoid's depth with a constant mannings_n starts the search, and is the depth of a cross section without
    Manning's n to carve with.

    Zero for no baseflow, and for a channel without valid banks where the cross section isn't on the raster either
    side of the stream cell. NaN for a slope that isn't positive.
    """
    trapezoid_height = _check_trapezoid_height(trapezoid_height)
    carved = banks
    if not banks.valid:
        carved = single_cell_banks(xs)
        if not carved.valid:
            return 0.0
        
    top_width = carved.top_width
    guess = float(trapezoid_depth(q, top_width * (1.0 - 2.0 * trapezoid_height), top_width,
                                  slope * float(slope_factor) ** 2, mannings_n))
    if not 0.0 < guess < math.inf:
        return guess
    
    from arc.bathymetry.channel import carved_depth  # which imports this module
    return carved_depth(xs, banks, q, slope, guess, trapezoid_height=trapezoid_height, roughness=roughness,
                        slope_factor=slope_factor, bank_elevation=bank_elevation, divide=divide)


def bathymetry_depth(xs: XSection, banks: Banks, q: float, slope: float, *, trapezoid_height: float,
                     target_depth: float | None = None, mannings_n: float = DEFAULT_MANNINGS_N,
                     roughness: DepthRoughness | None = None, slope_factor: float = 1.0,
                     bank_elevation: float | None = None, divide: Banks | None = None) -> BathymetryDepth:
    """The depth to carve the channel to: a drainage-area target depth if there's a valid one, which always applies,
    or otherwise channel_depth for the baseflow, which applies only with baseflow (`divide` as it takes it)."""
    if target_depth is not None and math.isfinite(target_depth) and target_depth > 0.0:
        return BathymetryDepth(float(target_depth), True, "target_depth")
    
    depth = channel_depth(xs, banks, q, slope, trapezoid_height=trapezoid_height, mannings_n=mannings_n,
                          roughness=roughness, slope_factor=slope_factor, bank_elevation=bank_elevation,
                          divide=divide)
    return BathymetryDepth(depth, bool(q > 0.0), "baseflow_manning")


def power_law_geometry(drainage_area: float, coefficient_depth: float | None, exponent_depth: float | None,
                       coefficient_width: float | None, exponent_width: float | None
                       ) -> tuple[float | None, float | None]:
    """The bankfull depth and width that the drainage-area power laws give, each None without its coefficients."""
    depth = None
    if coefficient_depth is not None and exponent_depth is not None:
        depth = float(coefficient_depth * drainage_area ** exponent_depth)
    width = None
    if coefficient_width is not None and exponent_width is not None:
        width = float(coefficient_width * drainage_area ** exponent_width)

    return depth, width


def _check_trapezoid_height(trapezoid_height: float) -> float:
    """The share of the top width each sloping side takes: at most a half, when the trapezoid becomes a triangle."""
    if not 0.0 <= trapezoid_height <= 0.5:
        raise ValueError(f"trapezoid_height must be between 0 and 0.5, not {trapezoid_height!r}.")
    
    return float(trapezoid_height)
