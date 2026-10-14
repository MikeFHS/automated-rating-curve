"""Linking a cross section's ordinates to raster cells, to read land cover and to write bathymetry, and filling the
gaps the cross sections leave in the bathymetry raster.

Each ordinate belongs to the cell nearest to it. Sampling steps one cell along whichever axis the cross section
crosses fastest, so consecutive ordinates are in different cells.

Once every cross section is burned in, fill_bathymetry_gaps fills each cell without bathymetry that has some in at
least four of the eight cells around it with their mean. Without Bathy_Use_Banks the channel should only lower the
ground, so given the DEM it first drops cells whose bathymetry is above the ground (on a cross section at an angle the
ordinates are interpolated, so a cell's own ground can be lower than the ordinate), and doesn't fill a gap above the
ground either.
"""
from __future__ import annotations

import numpy as np
from numba import njit

from arc.xsection.sampling import _compute_dem_coordinates

GAP_NEIGHBOURS = 4  # a gap is filled if at least this many of the eight cells around it have bathymetry


def ordinate_cells(row: int, col: int, stream_direction: float, cross_section_length: float, dx: float, dy: float
                   ) -> tuple[np.ndarray, np.ndarray]:
    """The row and column of the cell nearest each ordinate of the cross section that sample_cross_section takes
    with these arguments. Ordinates off the raster get cells off it too."""
    columns, rows, _ = _compute_dem_coordinates(row, col, stream_direction, cross_section_length, dx, dy)
    return np.rint(rows).astype(np.intp), np.rint(columns).astype(np.intp)


def sample_land_cover(land_cover: np.ndarray, row: int, col: int, stream_direction: float,
                      cross_section_length: float, dx: float, dy: float) -> np.ndarray:
    """The land cover class at each ordinate, from its nearest cell, as floats that are NaN off the raster."""
    rows, cols = ordinate_cells(row, col, stream_direction, cross_section_length, dx, dy)
    on_raster = (rows >= 0) & (rows < land_cover.shape[0]) & (cols >= 0) & (cols < land_cover.shape[1])
    values = np.full(rows.size, np.nan)
    values[on_raster] = land_cover[rows[on_raster], cols[on_raster]]
    return values


@njit(cache=True, error_model="numpy")
def burn_into_raster(raster, rows, cols, elevations, changed):
    """Write the changed ordinates' elevations into their cells of an output raster. A cell that already has a value
    from another cross section takes the average of that and the new one."""
    for k in range(elevations.size):
        if not changed[k]:
            continue
        r, c = rows[k], cols[k]
        if 0 <= r < raster.shape[0] and 0 <= c < raster.shape[1]:
            current = raster[r, c]
            raster[r, c] = elevations[k] if np.isnan(current) else current + (elevations[k] - current) * 0.5


@njit(cache=True, error_model="numpy")
def _drop_above(bathymetry, ground):
    rows, cols = bathymetry.shape
    for r in range(rows):
        for c in range(cols):
            if bathymetry[r, c] > ground[r, c]:
                bathymetry[r, c] = np.nan


@njit(cache=True, error_model="numpy")
def _count_row(bathymetry, r, counts, change):
    """Add change to each column's count where row r has bathymetry."""
    if 0 <= r < bathymetry.shape[0]:
        for c in range(bathymetry.shape[1]):
            if not np.isnan(bathymetry[r, c]):
                counts[c + 1] += change


@njit(cache=True, error_model="numpy")
def _write_row(bathymetry, r, fills):
    if 0 <= r < bathymetry.shape[0]:
        for c in range(bathymetry.shape[1]):
            if not np.isnan(fills[c]):
                bathymetry[r, c] = fills[c]


@njit(cache=True, error_model="numpy")
def _fill_gaps(bathymetry, ground, below_ground):
    rows, cols = bathymetry.shape
    # counts[c + 1] is how many cells of column c have bathymetry, from the row above the one being filled to the row
    # below, so most cells without enough neighbours are passed over without looking at them
    counts = np.zeros(cols + 2, np.int64)
    _count_row(bathymetry, 0, counts, 1)
    # A row's fills are written once the row below it is done, so that no fill helps fill another
    fills = np.empty(cols, bathymetry.dtype)
    pending = np.full(cols, np.nan, bathymetry.dtype)
    for r in range(rows):
        _count_row(bathymetry, r + 1, counts, 1)
        for c in range(cols):
            fills[c] = np.nan
            if not np.isnan(bathymetry[r, c]) or counts[c] + counts[c + 1] + counts[c + 2] < GAP_NEIGHBOURS:
                continue
            count = 0
            total = 0.0
            # The neighbours summed row by row
            for rr in range(max(r - 1, 0), min(r + 2, rows)):
                for cc in range(max(c - 1, 0), min(c + 2, cols)):
                    value = bathymetry[rr, cc]
                    if not np.isnan(value):
                        count += 1
                        total += value
            fills[c] = total / count
            if below_ground and fills[c] > ground[r, c]:
                fills[c] = np.nan
        _count_row(bathymetry, r - 1, counts, -1)  # as it was, before it takes its fills
        _write_row(bathymetry, r - 1, pending)
        fills, pending = pending, fills
    _write_row(bathymetry, rows - 1, pending)


def fill_bathymetry_gaps(bathymetry: np.ndarray, ground: np.ndarray | None = None) -> np.ndarray:
    """Fill the gaps in a bathymetry raster in place, and return it.

    A cell without bathymetry (NaN) that has it in at least four of the eight cells around it gets their mean. Every
    gap is filled from the raster as it was, so a gap filled doesn't help fill another. Beyond the raster's edge
    counts as no bathymetry.

    Without Bathy_Use_Banks, pass the DEM as ground. The channel then only lowers the ground, so cells above it are
    dropped first, and a gap isn't filled above the ground there.
    """
    bathymetry = np.asarray(bathymetry)
    if bathymetry.ndim != 2:
        raise ValueError("Bathymetry NaN filling requires a 2-D array.")
    if not np.issubdtype(bathymetry.dtype, np.floating):
        raise TypeError("Bathymetry NaN filling requires a floating array.")
    if ground is None:
        _fill_gaps(bathymetry, bathymetry, False)
        return bathymetry
    ground = np.asarray(ground)
    if ground.shape != bathymetry.shape:
        raise ValueError(f"The ground is {ground.shape} but the bathymetry {bathymetry.shape}.")
    _drop_above(bathymetry, ground)
    _fill_gaps(bathymetry, ground, True)
    return bathymetry
