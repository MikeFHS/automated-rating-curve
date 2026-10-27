"""Linking a cross section's ordinates to raster cells, to read land cover and to write bathymetry, and filling the
gaps the cross sections leave in the bathymetry raster.

Each ordinate belongs to the cell nearest to it. Sampling steps one cell along whichever axis the cross section
crosses fastest, so consecutive ordinates are in different cells.

Where cross sections overlap, a cell takes the mean of the elevations burned into it, whatever order they come in.
The cell at a cross section's centre is the exception: it takes that centre's elevation, which is the bed its rating
curve was worked out on (the VDT database's Elev), whatever its neighbours' channels burn there. A cell at the centre
of more than one cross section (the low spot can move two onto one cell) takes the mean of their centres. Only the
ordinates whose ground the carve changed, and the centre, are burned, and nothing of a cross section the carve left as
it was (or that wasn't carved): its centre takes whatever its neighbours burn there, or the gap fill.

Once every cross section is burned in, fill_sparse_gaps fills each cell without bathymetry that has some in at least four of the eight cells around it
with their mean. Without Bathy_Use_Banks the channel should only lower the ground, so given the DEM it first drops
cells whose bathymetry is above the ground (on a cross section at an angle the ordinates are interpolated, so a cell's
own ground can be lower than the ordinate), and doesn't fill a gap above the ground either.
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np
from numba import njit

from arc.xsection.sampling import OFF_RASTER_ELEVATION

GAP_NEIGHBOURS = 4  # a gap is filled if at least this many of the eight cells around it have bathymetry


def ordinate_cells(row: int, col: int, stream_direction: float, cross_section_length: float, dx: float, dy: float
                   ) -> tuple[np.ndarray, np.ndarray]:
    """The row and column of the cell nearest each ordinate of the cross section that sample_cross_section takes
    with these arguments. Ordinates off the raster get cells off it too. The cells _compute_dem_coordinates gives,
    worked out the same way, compiled."""
    xs_direction = stream_direction - np.pi / 2
    cos, sin = np.cos(xs_direction), np.sin(xs_direction)
    spacing = 1 / max(abs(cos) / dx, abs(sin) / dy)
    return _ordinate_cells(float(row), float(col), float(cos), float(sin), float(spacing),
                           int(cross_section_length / 2 / spacing), float(dx), float(dy))


@njit(cache=True, error_model="numpy")
def _ordinate_cells(row, col, cos, sin, spacing, half, dx, dy):
    rows = np.empty(2 * half + 1, np.intp)
    cols = np.empty(2 * half + 1, np.intp)
    for k in range(-half, half + 1):
        x = spacing * k
        cols[k + half] = np.intp(np.rint(col + cos * x / dx))
        rows[k + half] = np.intp(np.rint(row + sin * x / dy))

    return rows, cols


def sample_land_cover(land_cover: np.ndarray, row: int, col: int, stream_direction: float,
                      cross_section_length: float, dx: float, dy: float) -> np.ndarray:
    """The land cover class at each ordinate, from its nearest cell, as floats that are NaN off the raster."""
    rows, cols = ordinate_cells(row, col, stream_direction, cross_section_length, dx, dy)
    on_raster = (rows >= 0) & (rows < land_cover.shape[0]) & (cols >= 0) & (cols < land_cover.shape[1])
    values = np.full(rows.size, np.nan)
    values[on_raster] = land_cover[rows[on_raster], cols[on_raster]]
    return values


@njit(cache=True, error_model="numpy")
def _burn(raster, burns, r, c, elevation, centre):
    """Burn one elevation into a cell. burns counts what the cell's mean is of: the ordinates burned into it, or
    once a centre has been, minus the number of centres."""
    burns[r, c] = _burn_cell(raster, r, c, elevation, centre, burns[r, c])


@njit(cache=True, error_model="numpy")
def _burn_cell(raster, r, c, elevation, centre, count):
    """_burn, given the cell's count, returning its new count."""
    if centre:
        if count >= 0:  # the first centre replaces the ordinates burned before it
            raster[r, c] = elevation
            return -1
        count -= 1
    elif count < 0:
        return count  # a centre's cell
    else:
        count += 1

    current = raster[r, c]
    raster[r, c] = elevation if abs(count) == 1 else current + (elevation - current) / abs(count)
    return count


@njit(cache=True, error_model="numpy")
def burn_into_raster(raster, burns, rows, cols, elevations, changed):
    """Burn a carved cross section into the bathymetry raster: its changed ordinates' elevations, and its centre's,
    changed or not (see the notes above), unless the carve changed nothing. `burns`, an int32 array the shape of the
    raster that starts at 0, keeps each cell's count for the means, so pass the same one for every cross section."""
    if not changed.any():
        return
    
    centre = elevations.size // 2
    for k in range(elevations.size):
        if not (changed[k] or k == centre):
            continue
        r, c = rows[k], cols[k]
        if 0 <= r < raster.shape[0] and 0 <= c < raster.shape[1]:
            _burn(raster, burns, r, c, elevations[k], k == centre)


class SparseRaster(NamedTuple):
    """A float32 raster that is NaN but at a few cells, as those cells and their values: a bathymetry raster without
    the memory of the whole raster until it's needed (to_array)."""
    shape: tuple[int, int]
    cells: np.ndarray  # each cell's row * ncols + col, ascending
    values: np.ndarray  # float32

    def to_array(self) -> np.ndarray:
        raster = np.full(self.shape, np.nan, dtype=np.float32)
        raster.ravel()[self.cells] = self.values
        return raster


class Burns:
    """Carved cross sections gathered to be burned into a bathymetry raster at once, giving what burn_into_raster would
    have burned into a float32 raster of NaN one by one: each cell's ordinates are burned in the order they came, and a
    cell's mean depends on its own ordinates alone. It takes memory only for the ordinates burned, not for a raster or
    the counts burn_into_raster keeps for every cell."""

    def __init__(self, shape: tuple[int, int]):
        self.shape = (int(shape[0]), int(shape[1]))
        self._cells, self._elevations, self._centres = [], [], []

    def add(self, rows: np.ndarray, cols: np.ndarray, elevations: np.ndarray, changed: np.ndarray) -> None:
        """A carved cross section, as burn_into_raster takes it."""
        cells, values, centres = _ordinates_to_burn(rows, cols, elevations, changed, *self.shape)
        if cells.size:
            self._cells.append(cells)
            self._elevations.append(values)
            self._centres.append(centres)

    def burned(self) -> SparseRaster:
        """The raster with them all burned in."""
        if not self._cells:
            return SparseRaster(self.shape, np.empty(0, np.int64), np.empty(0, np.float32))
        
        cells = np.concatenate(self._cells)
        return SparseRaster(self.shape, *_burned_values(cells, np.concatenate(self._elevations),
                                                        np.concatenate(self._centres),
                                                        np.argsort(cells, kind="stable")))


@njit(cache=True, error_model="numpy")
def _ordinates_to_burn(rows, cols, elevations, changed, nrows, ncols):
    """The cells (row * ncols + col) burn_into_raster burns, their elevations, and which are the centre."""
    size = elevations.size
    cells = np.empty(size, np.int64)
    values = np.empty(size, np.float64)
    centres = np.zeros(size, np.bool_)
    if not changed.any():
        return cells[:0], values[:0], centres[:0]
    
    centre = size // 2
    count = 0
    for k in range(size):
        if not (changed[k] or k == centre):
            continue
        r, c = rows[k], cols[k]
        if 0 <= r < nrows and 0 <= c < ncols:
            cells[count] = r * ncols + c
            values[count] = elevations[k]
            centres[count] = k == centre
            count += 1

    return cells[:count], values[:count], centres[:count]


@njit(cache=True, error_model="numpy")
def _burned_values(cells, elevations, centres, order):
    """Each cell the ordinates burn, ascending, and its value once they're burned into a float32 raster of NaN, each
    cell's in their order, which order (the ordinates sorted by cell, stably) keeps."""
    burned = np.empty(order.size, np.int64)
    values = np.empty(order.size, np.float32)
    cell = np.empty((1, 1), np.float32)  # one cell of the raster at a time
    count = n = 0
    for i in range(order.size):
        j = order[i]
        if i == 0 or cells[j] != cells[order[i - 1]]:
            if i > 0:
                values[n] = cell[0, 0]
                n += 1
            burned[n] = cells[j]
            cell[0, 0] = np.nan
            count = 0  # a cell not yet burned
        count = _burn_cell(cell, 0, 0, elevations[j], centres[j], count)

    if order.size:
        values[n] = cell[0, 0]
        n += 1

    return burned[:n], values[:n]


def fill_sparse_gaps(raster: SparseRaster, dem: np.ndarray | None = None) -> SparseRaster:
    """
    Without Bathy_Use_Banks, pass the DEM, whose cells at or above OFF_RASTER_ELEVATION have no ground and others the
    ground their elevation as float32 holds it"""
    rows, cols = raster.shape
    below_ground = dem is not None
    cells, values = _sparse_fill(rows, cols, raster.cells, raster.values,
                                 dem if below_ground else np.zeros((1, 1), np.float32), below_ground)
    return SparseRaster(raster.shape, cells, values)


@njit(cache=True, error_model="numpy")
def _ground(dem, r, c):
    """The ground at a cell, as fill_sparse_gaps takes it from the DEM: float32, NaN at a wall."""
    if dem[r, c] >= OFF_RASTER_ELEVATION:
        return np.float32(np.nan)
    return np.float32(dem[r, c])


@njit(cache=True, error_model="numpy")
def _load_row(buffer, cells, values, first, last, cols):
    """Put a row's cells (cells[first:last]) into a row buffer of NaN."""
    for i in range(first, last):
        buffer[cells[i] % cols] = values[i]


@njit(cache=True, error_model="numpy")
def _clear_row(buffer, cells, first, last, cols):
    for i in range(first, last):
        buffer[cells[i] % cols] = np.nan


@njit(cache=True, error_model="numpy")
def _sparse_fill(rows, cols, cells, values, dem, below_ground):
    """_drop_above and _fill_gaps on the cells that can change: those with bathymetry, and the gaps beside them, a
    row at a time, with the rows above and below it held as whole rows."""
    # Cells without a value, or above the ground, are gaps
    keep = np.empty(cells.size, np.bool_)
    for i in range(cells.size):
        keep[i] = not np.isnan(values[i]) and not (
            below_ground and values[i] > _ground(dem, cells[i] // cols, cells[i] % cols))
        
    cells, values = cells[keep], values[keep]
    n = cells.size
    # Where each row's cells start among them (they're in order), for the rows that have any
    row_of = cells // cols
    starts = np.empty(n + 1, np.int64)
    present = np.empty(n, np.int64)
    count = 0
    for i in range(n):
        if i == 0 or row_of[i] != row_of[i - 1]:
            present[count] = row_of[i]
            starts[count] = i
            count += 1

    starts[count] = n
    present = present[:count]
    # The rows above, at and below the one being filled, as whole rows of NaN but at their cells
    above, here, below = np.full(cols, np.float32(np.nan)), np.full(cols, np.float32(np.nan)), \
        np.full(cols, np.float32(np.nan))
    seen = np.full(cols, -1, np.int64)  # the row each column was last looked at in, as a gap
    filled = np.empty(8 * n, np.int64)
    fills = np.empty(8 * n, np.float32)
    nfills = 0
    p = 0  # the first of the present rows at or below the row before the one being filled
    loaded = np.full(3, -1, np.int64)  # the present-row index in each buffer (above, here, below), or -1
    r = present[0] - 1 if count else 0
    last_row = present[count - 1] + 1 if count else -1
    while r <= last_row:
        if r < 0 or r >= rows:
            r += 1
            continue

        # Move the buffers down a row: each holds row r - 1, r and r + 1's cells
        while p < count and present[p] < r - 1:
            p += 1

        wanted = np.full(3, -1, np.int64)
        for q in range(p, min(p + 3, count)):
            if r - 1 <= present[q] <= r + 1:
                wanted[present[q] - (r - 1)] = q

        if loaded[0] >= 0:
            _clear_row(above, cells, starts[loaded[0]], starts[loaded[0] + 1], cols)

        above, here, below = here, below, above
        loaded[0], loaded[1], loaded[2] = loaded[1], loaded[2], -1
        for slot in range(3):
            if loaded[slot] != wanted[slot]:
                buffer = above if slot == 0 else (here if slot == 1 else below)
                if loaded[slot] >= 0:
                    _clear_row(buffer, cells, starts[loaded[slot]], starts[loaded[slot] + 1], cols)
                if wanted[slot] >= 0:
                    _load_row(buffer, cells, values, starts[wanted[slot]], starts[wanted[slot] + 1], cols)
                loaded[slot] = wanted[slot]

        if loaded[0] < 0 and loaded[1] < 0 and loaded[2] < 0:
            # No cell near this row: skip ahead to the row before the next one with cells
            r = max(r + 1, present[p] - 1) if p < count else last_row + 1
            continue

        # The gaps in row r beside a cell with bathymetry
        for slot in range(3):
            q = loaded[slot]
            if q < 0:
                continue

            for i in range(starts[q], starts[q + 1]):
                c0 = cells[i] % cols
                for c in range(max(c0 - 1, 0), min(c0 + 2, cols)):
                    if seen[c] == r or not np.isnan(here[c]):
                        continue

                    seen[c] = r
                    # As _fill_gaps: the neighbours summed row by row
                    neighbours = 0
                    total = 0.0
                    for rr in range(3):
                        if not 0 <= r - 1 + rr < rows:
                            continue

                        buffer = above if rr == 0 else (here if rr == 1 else below)
                        for cc in range(max(c - 1, 0), min(c + 2, cols)):
                            value = buffer[cc]
                            if not np.isnan(value):
                                neighbours += 1
                                total += value

                    if neighbours < GAP_NEIGHBOURS:
                        continue

                    fill = np.float32(total / neighbours)
                    if below_ground and fill > _ground(dem, r, c):
                        continue

                    filled[nfills] = r * cols + c
                    fills[nfills] = fill
                    nfills += 1
        r += 1
    # Both, in order
    every = np.concatenate((cells, filled[:nfills]))
    order = np.argsort(every)
    return every[order], np.concatenate((values, fills[:nfills]))[order]


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
