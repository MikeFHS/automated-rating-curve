"""Distances along a reach's stream cells.

A reach's stream cells join their eight neighbours: dx apart along a row, dy along a column, and hypot(dx, dy) across
a diagonal. The path along the stream between two cells is the shortest through the reach's cells. Where the cells
don't all join up, as where a reach leaves the raster and comes back or its rasterisation skips a cell, the path jumps
each gap in a straight line, across the narrowest part of the gap.

A station is a cell's distance along the stream from one end of the reach. It isn't the path's own length: a stream at
an angle to the grid crosses it in steps along a row or column and across a diagonal, a stair-step path up to 8% longer
than the stream (at 22.5 degrees), which made its slopes that much too gentle. So the stations are measured along the
path from the far end of the reach back to its start (stepping each time to the neighbour on a shortest path nearest
the start, and the lowest row and then column of any that tie), with each of its cells moved towards the mean of the
cells SMOOTHING_CELLS either side of it along the path (fewer near its ends, as many each side), but no more than half
a cell along a row or a column: about as far as a stair step strays from the stream it follows, so that a tight bend,
whose mean lies further inside it, keeps most of its length. A straight stream at any angle then comes out within 0.3%
of its length, a meander of 60 cells or more from crest to crest within 1%, one of 20 cells 5% short (its cells' path
is 5% long), and a stream along a row or a diagonal exactly. Across a gap the station still takes the path's straight
jump, and the means don't reach across a jump longer than one and a half diagonals. A cell off that path, such as the
corner cell where a rasterised line steps along a row and then a column rather than across the diagonal, takes the
station at its own distance along the shortest path.
"""
from __future__ import annotations

import math

import numpy as np
from numba import njit

SMOOTHING_CELLS = 3  # how many cells either side along the path each cell's centre is averaged with for the stations


@njit(cache=True, error_model="numpy")
def _row_index(rows, cols):
    """The cells in order of row and then column, their columns in that order, and where each row's cells start in
    it: row r's are at positions row_start[r - first_row] to row_start[r - first_row + 1]."""
    first_row, first_col = rows.min(), cols.min()
    order = np.argsort((rows - first_row) * (cols.max() - first_col + 1) + (cols - first_col))
    row_start = np.zeros(rows.max() - first_row + 2, np.int64)
    for r in rows:
        row_start[r - first_row + 1] += 1
    for k in range(1, row_start.size):
        row_start[k] += row_start[k - 1]

    return order, cols[order], row_start, first_row


@njit(cache=True, error_model="numpy")
def _columns_between(sorted_cols, first, last, low, high):
    """The positions, from first to last, of one row's cells whose columns are from low to high."""
    if last - first <= 16:  # a stream crosses most rows in a cell or two
        lo = first
        while lo < last and sorted_cols[lo] < low:
            lo += 1
        hi = lo
        while hi < last and sorted_cols[hi] <= high:
            hi += 1
        return lo, hi
    
    row = sorted_cols[first:last]
    return first + np.searchsorted(row, low), first + np.searchsorted(row, high, side="right")


@njit(cache=True, error_model="numpy")
def _adjacency(rows, cols, dx, dy):
    """Each cell's neighbours among the cells, and how far away each is, as offsets into flat arrays: cell i's
    neighbours are neighbours[start[i]:start[i + 1]]. A cell at the same place as another is its neighbour at 0 m."""
    n = rows.size
    order, sorted_cols, row_start, first_row = _row_index(rows, cols)
    last_row = first_row + row_start.size - 2
    start = np.zeros(n + 1, np.int64)
    for i in range(n):
        count = -1  # not itself
        for row in range(max(rows[i] - 1, first_row), min(rows[i] + 1, last_row) + 1):
            lo, hi = _columns_between(sorted_cols, row_start[row - first_row], row_start[row - first_row + 1],
                                      cols[i] - 1, cols[i] + 1)
            count += hi - lo
        start[i + 1] = start[i] + count

    neighbours = np.empty(start[n], np.int64)
    lengths = np.empty(start[n])
    for i in range(n):
        k = start[i]
        for row in range(max(rows[i] - 1, first_row), min(rows[i] + 1, last_row) + 1):
            lo, hi = _columns_between(sorted_cols, row_start[row - first_row], row_start[row - first_row + 1],
                                      cols[i] - 1, cols[i] + 1)
            for p in range(lo, hi):
                j = order[p]
                if j != i:
                    neighbours[k] = j
                    lengths[k] = math.hypot((cols[j] - cols[i]) * dx, (rows[j] - rows[i]) * dy)
                    k += 1

    return start, neighbours, lengths


@njit(cache=True, error_model="numpy")
def _push(heap_distance, heap_cell, size, distance, cell):
    """Add a cell to a binary min-heap on distance, returning the new size."""
    i = size
    while i > 0:
        parent = (i - 1) // 2
        if not distance < heap_distance[parent]:
            break
        heap_distance[i] = heap_distance[parent]
        heap_cell[i] = heap_cell[parent]
        i = parent

    heap_distance[i] = distance
    heap_cell[i] = cell
    return size + 1


@njit(cache=True, error_model="numpy")
def _pop(heap_distance, heap_cell, size):
    """Remove the nearest cell from the heap, returning its distance, the cell and the new size."""
    distance, cell = heap_distance[0], heap_cell[0]
    size -= 1
    last_distance, last_cell = heap_distance[size], heap_cell[size]
    i = 0
    while True:
        child = 2 * i + 1
        if child >= size:
            break
        if child + 1 < size and heap_distance[child + 1] < heap_distance[child]:
            child += 1
        if not heap_distance[child] < last_distance:
            break
        heap_distance[i] = heap_distance[child]
        heap_cell[i] = heap_cell[child]
        i = child

    heap_distance[i] = last_distance
    heap_cell[i] = last_cell
    return distance, cell, size


@njit(cache=True, error_model="numpy")
def _distances_from(rows, cols, dx, dy, source, start, neighbours, lengths):
    """Dijkstra's shortest paths from one cell through the adjacency, jumping gaps (see the module notes)."""
    n = rows.size
    distance = np.full(n, np.inf)
    settled = np.zeros(n, np.bool_)
    heap_distance = np.empty(neighbours.size + n + 1)
    heap_cell = np.empty(neighbours.size + n + 1, np.int64)
    distance[source] = 0.0
    size = _push(heap_distance, heap_cell, 0, 0.0, source)
    remaining = n
    while remaining > 0:
        if size == 0:
            # The cells left don't join the ones reached. Jump the narrowest gap to one of them in a straight line.
            gap, near, far = np.inf, -1, -1
            for u in range(n):
                if settled[u]:
                    for v in range(n):
                        if not settled[v]:
                            length = math.hypot((cols[v] - cols[u]) * dx, (rows[v] - rows[u]) * dy)
                            if length < gap:
                                gap, near, far = length, u, v
            distance[far] = distance[near] + gap
            size = _push(heap_distance, heap_cell, size, distance[far], far)

        d, u, size = _pop(heap_distance, heap_cell, size)
        if settled[u]:
            continue

        settled[u] = True
        remaining -= 1
        for k in range(start[u], start[u + 1]):
            v = neighbours[k]
            if d + lengths[k] < distance[v]:
                distance[v] = d + lengths[k]
                size = _push(heap_distance, heap_cell, size, distance[v], v)

    return distance


@njit(cache=True, error_model="numpy")
def _along_stream_distances(rows, cols, dx, dy, source):
    start, neighbours, lengths = _adjacency(rows, cols, dx, dy)
    return _distances_from(rows, cols, dx, dy, source, start, neighbours, lengths)


@njit(cache=True, error_model="numpy")
def _along_stream_stations(rows, cols, dx, dy):
    """Stations from one end of the reach: the cell farthest along the stream from the first cell."""
    start, neighbours, lengths = _adjacency(rows, cols, dx, dy)
    end = np.argmax(_distances_from(rows, cols, dx, dy, 0, start, neighbours, lengths))
    return _stations_from(rows, cols, dx, dy, end, SMOOTHING_CELLS)[1]


@njit(cache=True, error_model="numpy")
def _stations_from(rows, cols, dx, dy, source, cells):
    """Each cell's distance from cell `source` along the shortest path through the cells, and its station from there
    (see the notes above), with the path's cells moved to the means of `cells` either side."""
    start, neighbours, lengths = _adjacency(rows, cols, dx, dy)
    distances = _distances_from(rows, cols, dx, dy, source, start, neighbours, lengths)
    n = rows.size
    # The path back from the farthest cell to the source
    path = np.empty(n, np.int64)
    current = np.argmax(distances)
    path[0] = current
    count = 1
    while distances[current] > 0.0 and count < n:
        tolerance = 1e-9 * (1.0 + distances[current])
        best = -1
        for k in range(start[current], start[current + 1]):
            v = neighbours[k]
            if distances[v] < distances[current] and abs(distances[v] + lengths[k] - distances[current]) <= tolerance:
                if best < 0 or distances[v] < distances[best] or (distances[v] == distances[best] and (
                        rows[v] < rows[best] or (rows[v] == rows[best] and cols[v] < cols[best]))):
                    best = v

        if best < 0:
            # A gap: back to the cell the path jumped it from
            error = np.inf
            for v in range(n):
                if distances[v] < distances[current]:
                    jump = math.hypot((cols[v] - cols[current]) * dx, (rows[v] - rows[current]) * dy)
                    if abs(distances[v] + jump - distances[current]) < error:
                        error, best = abs(distances[v] + jump - distances[current]), v

        path[count] = best
        count += 1
        current = best

    path = path[:count][::-1]
    d = distances[path]
    x = cols[path] * dx
    y = rows[path] * dy
    gap = 1.5 * math.hypot(dx, dy)
    # Where the run of joined-up cells each of the path's cells is in starts and ends
    run_start = np.zeros(count, np.int64)
    run_end = np.zeros(count, np.int64)
    first = 0
    for i in range(1, count + 1):
        if i == count or d[i] - d[i - 1] > gap:
            for k in range(first, i):
                run_start[k], run_end[k] = first, i - 1
            first = i

    mx = np.empty(count)
    my = np.empty(count)
    for i in range(count):
        h = min(cells, i - run_start[i], run_end[i] - i)
        sx = sy = 0.0
        for k in range(i - h, i + h + 1):
            sx += x[k]
            sy += y[k]
        mx[i] = x[i] + min(max(sx / (2 * h + 1) - x[i], -0.5 * dx), 0.5 * dx)
        my[i] = y[i] + min(max(sy / (2 * h + 1) - y[i], -0.5 * dy), 0.5 * dy)

    along = np.zeros(count)
    for i in range(1, count):
        step = d[i] - d[i - 1] if run_start[i] == i else math.hypot(mx[i] - mx[i - 1], my[i] - my[i - 1])
        along[i] = along[i - 1] + step

    return distances, np.interp(distances, d, along)


@njit(cache=True, error_model="numpy")
def _nearest_cell(rows, cols, dx, dy, other_rows, other_cols):
    """The cell nearest in a straight line to any of the other cells (the first of any that tie)."""
    best, nearest = np.inf, -1
    for i in range(rows.size):
        for j in range(other_rows.size):
            length = math.hypot((cols[i] - other_cols[j]) * dx, (rows[i] - other_rows[j]) * dy)
            if length < best:
                best, nearest = length, i

    return nearest


def _cell_arrays(rows, cols) -> tuple[np.ndarray, np.ndarray]:
    rows = np.asarray(rows, dtype=np.int64)
    cols = np.asarray(cols, dtype=np.int64)
    if rows.shape != cols.shape or rows.ndim != 1:
        raise ValueError(f"rows and cols must be 1-D and the same length, not {rows.shape} and {cols.shape}.")
    
    return rows, cols


def along_stream_distances(rows, cols, dx: float, dy: float, source: int) -> np.ndarray:
    """Each cell's distance along the shortest path through the reach's cells from cell `source`, in metres. Not
    smoothed, as stations are (see the notes above): the source can be anywhere along the reach."""
    rows, cols = _cell_arrays(rows, cols)
    return _along_stream_distances(rows, cols, float(dx), float(dy), int(source))


def along_stream_stations(rows, cols, dx: float, dy: float) -> np.ndarray:
    """Each cell's station: its distance along the stream from one end of the reach, in metres (see the notes above).
    Which end isn't defined."""
    rows, cols = _cell_arrays(rows, cols)
    if rows.size == 0:
        return np.empty(0)
    return _along_stream_stations(rows, cols, float(dx), float(dy))


def downstream_order(rows, cols, dx: float, dy: float, *, downstream_cells=None, upstream_cells=None
                     ) -> tuple[np.ndarray, np.ndarray] | None:
    """The reach's cells in order from upstream to downstream, and their stations, which start at 0 upstream.

    The downstream end is the cell nearest the reach downstream (downstream_cells, a (rows, cols) pair). Without
    that, the upstream end is the cell nearest the reach upstream (upstream_cells), and the downstream end is the
    cell farthest along the stream from it. None if neither is given. Cells equally far along stay in the order given.
    A reach of one cell, or of none, is in order already.
    """
    rows, cols = _cell_arrays(rows, cols)
    dx, dy = float(dx), float(dy)
    if rows.size <= 1:
        return np.arange(rows.size), np.zeros(rows.size)
    
    downstream = _neighbour_cell(rows, cols, dx, dy, downstream_cells)
    if downstream < 0:
        upstream = _neighbour_cell(rows, cols, dx, dy, upstream_cells)
        if upstream < 0:
            return None
        downstream = int(np.argmax(_along_stream_distances(rows, cols, dx, dy, upstream)))

    to_downstream, stations = _stations_from(rows, cols, dx, dy, downstream, SMOOTHING_CELLS)
    order = np.argsort(-to_downstream, kind="stable")
    return order, stations[order[0]] - stations[order]


def _neighbour_cell(rows, cols, dx, dy, cells) -> int:
    if cells is None:
        return -1
    
    other_rows, other_cols = _cell_arrays(*cells)
    if other_rows.size == 0:
        return -1
    
    return int(_nearest_cell(rows, cols, dx, dy, other_rows, other_cols))


def stream_raster_cells(streams: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The rows and columns of a stream raster's cells above 0, row by row, and the IDs they hold."""
    flat = np.flatnonzero(streams > 0)
    rows, cols = np.divmod(flat, streams.shape[1])
    return rows, cols, streams.ravel()[flat]


def stream_cells_by_reach(streams: np.ndarray | None = None, *, cells=None) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """The rows and columns of each reach's cells in a stream raster, whose cells above 0 hold their reach's ID, or
    in its cells as stream_raster_cells gives them."""
    rows, cols, ids = stream_raster_cells(streams) if cells is None else cells
    order = np.argsort(ids, kind="stable")
    ids = ids[order]
    starts = np.flatnonzero(np.r_[True, ids[1:] != ids[:-1]])
    ends = np.r_[starts[1:], ids.size]
    return {int(ids[s]): (rows[order[s:e]].astype(np.int64), cols[order[s:e]].astype(np.int64))
            for s, e in zip(starts, ends)}
