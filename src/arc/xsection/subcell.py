"""Narrow channels below the cell scale (Subcell_Channels).

A DEM's cell holds the mean of the ground over it, and ARC joins its cross sections' ordinates, the cells' means, with
straight lines. Across a channel only a few cells wide that turns its corners into ramps: a 60 m wide bed in 30 m cells
becomes a V with two thirds of its area, so the water that carries a flow stands too high, and most of all at a
constriction, the narrowest channel, which controls the backwater above it.

Where a cross section's channel is under NARROW_CELLS ordinate spacings across, its ordinates from MARGIN beyond its
left bank to MARGIN beyond its right are taken as the means of the ground over their cells, and the ground there is
reconstructed as the polyline, FINE points a cell, whose cell means are the ordinates and whose slope changes least in
total (L1 trend filtering, a small linear program). Each point is kept between the lowest and the highest of its own
cell's and its neighbours' means, so that the ground never stands above or below them: unlimited, the reconstruction
draws levees and tents between the cells of real terrain. Outside the window the ground is the ordinates, as before.

A carved channel (arc.bathymetry.carve_channel) keeps its exact profile between its banks, and the reconstruction of
the ground as it was before the carve replaces the straight lines from its bank tops out to the ordinates. A cross
section that wasn't carved and has no banks is reconstructed MARGIN either side of its stream cell: the bank search
finds none on a channel a cell or two wide. Manual cross sections are left as they are.

The reconstruction becomes the cross section's profile (XSection.profile), as a carve's does, so the rating curves,
the standard step and Valley_Sections all use it. On the synthetic constriction it brings the backwater's error from
+1.05 m to +0.07 m in 20 m cells and from +0.30 m to +0.004 m in 10 m cells.

The programs are solved BATCH cross sections at a time, as one program whose blocks don't share a variable (each
block's solution is its own program's), on up to THREADS threads, about a third of a millisecond a cross section.
Every program has a solution (the cells' own means, flat across each cell, meet its constraints), but not always only
one: the solver's method picks which, so it stays HiGHS's default, and the batches are always the same size, which
makes a run's reconstructions the same every time.
"""
from __future__ import annotations

import math
import os
from concurrent.futures import ThreadPoolExecutor
from typing import NamedTuple

import numpy as np

from arc.hydraulics import uniform_profile
from arc.xsection.sampling import OFF_RASTER_ELEVATION
from arc.xsection.xsection import Profile, XSection

NARROW_CELLS = 10  # a channel under this many ordinate spacings across is narrow
MARGIN = 2  # ordinates beyond each bank that are reconstructed too
FINE = 5  # points a cell, odd, so that each cell's middle point is at its centre
BATCH = 100  # cross sections whose programs are solved together
THREADS = min(4, os.cpu_count() or 1)  # the solver holds Python's lock part of the time: more threads don't help


class _Window(NamedTuple):
    """The ordinates of a cross section to reconstruct: their cells' means z, from lo to hi ordinates from the stream
    cell, around a channel with these banks, carved or not."""
    z: np.ndarray
    lo: int
    hi: int
    left: float
    right: float
    carved: bool


def _window(xs: XSection) -> _Window | None:
    """The window of a narrow channel's cross section to reconstruct (see the notes above), or None."""
    if xs.subcell or xs.mannings_n.size != xs.elevations.size:
        return None

    spacing = float(xs.ordinate_distance)
    carved = xs.profile is not None
    if carved:
        if xs.ground is None or xs.carved_banks is None:
            return None
        ground, (left, right) = xs.ground, xs.carved_banks
    else:
        ground, left, right = xs.elevations, float(xs.left_bank_distance), float(xs.right_bank_distance)
        if not (left > 0.0 and right > 0.0):
            left = right = 0.5 * spacing

    if not (left > 0.0 and right > 0.0 and left + right < NARROW_CELLS * spacing):
        return None

    center, ground_center = xs.elevations.size // 2, ground.size // 2
    tolerance = 1e-9 * spacing
    lo = max(-math.floor((left + tolerance) / spacing) - MARGIN, -ground_center, -center)
    hi = min(math.floor((right + tolerance) / spacing) + MARGIN, ground.size - 1 - ground_center,
             xs.elevations.size - 1 - center)
    z = np.asarray(ground[ground_center + lo:ground_center + hi + 1], dtype=np.float64)
    usable = np.isfinite(z) & (z < OFF_RASTER_ELEVATION)
    if not usable[-lo]:
        return None

    # The window shrunk to the ordinates on the raster around the stream cell
    while not usable[0]:
        z, usable, lo = z[1:], usable[1:], lo + 1
    while not usable[-1]:
        z, usable, hi = z[:-1], usable[:-1], hi - 1

    if not usable.all() or z.size < 3:
        return None

    return _Window(z, lo, hi, float(left), float(right), carved)


def _l1_trends(windows: list[np.ndarray], spacings: list[float]) -> list[np.ndarray] | None:
    """For each window of cell means z, the FINE points a cell whose cell means are z and whose slope changes least
    in total, each between the lowest and the highest of its own cell's and its neighbours' means; None if the
    solver fails. One program: the points x of every window, then the slope changes' bounds t, with -t <= the change
    in slope <= t at each interior point of each window, minimising the sum of t."""
    # Imported here, as only Subcell_Channels needs them: scipy.optimize takes a run a fifth of a second to import
    from scipy.optimize import linprog
    from scipy.sparse import coo_matrix

    cells = np.array([z.size for z in windows])
    points = cells * FINE
    x_start = np.concatenate([[0], np.cumsum(points)])
    n_x = int(x_start[-1])
    # The interior points of each window: x[p], x[p + 1] and x[p + 2] are one of its turns
    turn_start = np.concatenate([[0], np.cumsum(points - 2)])
    n_t = int(turn_start[-1])
    first = np.concatenate([x_start[b] + np.arange(points[b] - 2) for b in range(len(windows))])
    scale = np.concatenate([np.full(points[b] - 2, FINE / spacings[b]) for b in range(len(windows))])
    t = np.arange(n_t)

    # Inequalities: (x[p] - 2 x[p + 1] + x[p + 2]) / h - t <= 0, and its negative - t <= 0
    rows = np.concatenate([t, t, t, t, n_t + t, n_t + t, n_t + t, n_t + t])
    cols = np.concatenate([first, first + 1, first + 2, n_x + t] * 2)
    values = np.concatenate([scale, -2.0 * scale, scale, -np.ones(n_t), -scale, 2.0 * scale, -scale, -np.ones(n_t)])
    a_ub = coo_matrix((values, (rows, cols)), shape=(2 * n_t, n_x + n_t)).tocsr()
    # Equalities: each cell's FINE points average to its mean
    a_eq = coo_matrix((np.full(n_x, 1.0 / FINE), (np.arange(n_x) // FINE, np.arange(n_x))),
                      shape=(int(cells.sum()), n_x + n_t)).tocsr()
    z = np.concatenate(windows)
    below, above = [], []
    for w in windows:
        neighbours = [np.r_[w[:1], w[:-1]], w, np.r_[w[1:], w[-1:]]]
        below.append(np.repeat(np.minimum.reduce(neighbours), FINE))
        above.append(np.repeat(np.maximum.reduce(neighbours), FINE))

    bounds = np.concatenate([np.column_stack([np.concatenate(below), np.concatenate(above)]),
                             np.column_stack([np.zeros(n_t), np.full(n_t, np.inf)])])
    result = linprog(np.concatenate([np.zeros(n_x), np.ones(n_t)]), A_ub=a_ub, b_ub=np.zeros(2 * n_t),
                     A_eq=a_eq, b_eq=z, bounds=bounds, method="highs")
    if not result.success:
        return None
    
    return [result.x[x_start[b]:x_start[b + 1]] for b in range(len(windows))]


def _apply(xs: XSection, window: _Window, fine_z: np.ndarray) -> bool:
    """Give the cross section its reconstructed profile, from its window's points, and say whether it did."""
    spacing = float(xs.ordinate_distance)
    z, lo, hi, left, right, carved = window
    center = xs.elevations.size // 2
    tolerance = 1e-9 * spacing
    h = spacing / FINE
    fine_s = (np.arange(z.size * FINE) + 0.5) * h + lo * spacing - spacing / 2
    fine_s[np.abs(fine_s) < 0.5 * h] = 0.0  # the stream cell's middle point
    fine_n = np.repeat(np.asarray(xs.mannings_n[center + lo:center + hi + 1], dtype=np.float64), FINE)
    window_lo, window_hi = lo * spacing - spacing / 2 - tolerance, hi * spacing + spacing / 2 + tolerance
    base = xs.profile if carved else uniform_profile(xs.elevations, xs.mannings_n, spacing)
    s, e, n = base.stations, base.elevations, base.mannings_n
    outer_left, outer_right = s < window_lo, s > window_hi
    if carved:
        # The channel between its bank tops as carved, the reconstruction out from them, the ordinates beyond
        channel = (s >= -left - tolerance) & (s <= right + tolerance)
        first = int(np.flatnonzero(channel)[0])
        in_left, in_right = fine_s < -left - tolerance, fine_s > right + tolerance
        stations = np.concatenate([s[outer_left], fine_s[in_left], s[channel], fine_s[in_right], s[outer_right]])
        elevations = np.concatenate([e[outer_left], fine_z[in_left], e[channel], fine_z[in_right], e[outer_right]])
        mannings_n = np.concatenate([n[outer_left], fine_n[in_left], n[channel], fine_n[in_right], n[outer_right]])
        stream_cell = int(outer_left.sum() + in_left.sum() + (base.center - first))
    else:
        stations = np.concatenate([s[outer_left], fine_s, s[outer_right]])
        elevations = np.concatenate([e[outer_left], fine_z, e[outer_right]])
        mannings_n = np.concatenate([n[outer_left], fine_n, n[outer_right]])
        stream_cell = int(np.flatnonzero(stations == 0.0)[0])

    if not (np.all(np.diff(stations) >= 0.0) and stations[stream_cell] == 0.0):
        return False

    xs.profile = Profile(stations, elevations, mannings_n, stream_cell)
    xs.subcell = True
    return True


def reconstruct_all(sections: list[XSection]) -> int:
    """Give each narrow channel's cross section its reconstructed profile (see the notes above), and say how many it
    did. Needs the banks the hydraulics use set (arc.bathymetry.set_bank_distances), and for a carved channel, the
    ground before the carve (XSection.ground)."""
    found = [(xs, w) for xs, w in ((xs, _window(xs)) for xs in sections) if w is not None]
    batches = [found[start:start + BATCH] for start in range(0, len(found), BATCH)]

    def solve(batch):
        return _l1_trends([w.z for _, w in batch], [float(xs.ordinate_distance) for xs, _ in batch])

    if len(batches) > 1 and THREADS > 1:
        with ThreadPoolExecutor(THREADS) as pool:
            solutions = list(pool.map(solve, batches))
    else:
        solutions = [solve(batch) for batch in batches]

    made = 0
    for batch, solved in zip(batches, solutions):
        if solved is not None:
            made += sum(_apply(xs, w, fine_z) for (xs, w), fine_z in zip(batch, solved))
    return made


def reconstruct(xs: XSection) -> bool:
    """reconstruct_all for one cross section."""
    return reconstruct_all([xs]) == 1
