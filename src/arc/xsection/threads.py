"""Flow-through threads (Flow_Through_Threads): the other threads of a river that bars split.

A cross section's water spreads out from its stream cell until it meets higher ground (arc.hydraulics). Where a bar
splits the river, the threads the stream line doesn't follow stay dry until the water tops the bar, though the river
fills them from upstream: below the bar's top the cross section puts all the flow in one thread, which stands too
high, and the bathymetry carves that thread too deep to carry it.

A thread is a stretch of the water mask (BathyWaterMask) along the cross section, apart from the stream cell's own,
that joins the river both upstream and downstream of the cross section: with the water mask's other stretches along
the cross section cut out of it, and the cells beside them, its water still reaches the river's stream cells (its own
reach's, the reach it flows into and those flowing into it) more than THREAD_MARGIN metres up and down the river, and
within THREAD_REACH metres of the stream cell across and along the grid. That leaves out water that joins the river
on one side only, as a slough, an oxbow lake open at one end or a backwater does, whose water stands still at the
river's level. It also leaves out any stretch with a stream cell on or beside it, which is a stream of its own, or the
river itself again where it bends back across the cross section. A thread's lowest ground is at most MAX_THREAD_RISE
above the stream cell's: water higher up is a terrace's pond, not the river's. The limits are in metres, not cells,
so they mean the same on any DEM.

A thread is kept as the stations of its water's edges, half way between its last water ordinate and the first dry one
either side, as the water's edge banks are (arc.bathymetry.banks). The hydraulics spread water from every vertex
between them, the cross section's hydraulic banks reach out to its outermost threads, so their water is the channel's
and takes the water's Manning's n, and the bathymetry carves each to the channel's depth (arc.bathymetry.channel).
"""
from __future__ import annotations

import math

import numpy as np
from numba import njit

from arc.portable import cos_sin

THREAD_REACH = 3000.0  # metres: how far from the stream cell a thread's water may go to join the river
THREAD_MARGIN = 60.0  # metres along the river beyond the cross section that a thread's water must reach either way
MAX_THREAD_RISE = 2.0  # metres: a thread's lowest ground is at most this far above the stream cell's

_UNSEEN, _CUT, _SEEN = 0, 1, 2


@njit(cache=True, error_model="numpy")
def _stream_id(starts, stream_cols, stream_ids, row, col):
    """The reach ID of the stream cell at (row, col), listed row by row as arc.xsection.stream_path.is_stream_cell
    takes them, or -1 if it isn't one."""
    lo, hi = starts[row], starts[row + 1]
    while lo < hi:
        middle = (lo + hi) // 2
        if stream_cols[middle] < col:
            lo = middle + 1
        else:
            hi = middle

    if lo < starts[row + 1] and stream_cols[lo] == col:
        return stream_ids[lo]
    return -1


@njit(cache=True, error_model="numpy")
def _runs(water):
    """The first and last index of each stretch of True."""
    out = np.empty((water.size, 2), np.int64)
    count = 0
    k = 0
    while k < water.size:
        if water[k]:
            start = k
            while k + 1 < water.size and water[k + 1]:
                k += 1
            out[count, 0], out[count, 1] = start, k
            count += 1
        k += 1

    return out[:count]


@njit(cache=True, error_model="numpy")
def _room(queue, used):
    """The queue, or a copy twice its size, with room for one cell after the first used."""
    if used < queue.size:
        return queue
    grown = np.empty(2 * queue.size, queue.dtype)
    grown[:used] = queue[:used]
    return grown


@njit(cache=True, error_model="numpy")
def _beside_a_stream(starts, stream_cols, stream_ids, rows, cols, first, last, nrows, ncols):
    """Whether any cell of ordinates first to last, or a cell beside one, is a stream cell."""
    for i in range(first, last + 1):
        for r in range(rows[i] - 1, rows[i] + 2):
            for c in range(cols[i] - 1, cols[i] + 2):
                if 0 <= r < nrows and 0 <= c < ncols and _stream_id(starts, stream_cols, stream_ids, r, c) >= 0:
                    return True
    return False


@njit(cache=True, error_model="numpy")
def _joins_both_ways(mask, starts, stream_cols, stream_ids, river, rows, cols, runs, which, row, col, cos, sin, dx,
                     dy, reach_rows, reach_cols, margin, state, queue):
    """Whether run `which`'s water, with the other runs' cells and those beside them cut out of the water mask,
    reaches a stream cell of the river (one of the reach IDs in river) more than margin metres up the river from the
    cross section's stream cell (row, col), and one more than margin down it, within reach_rows rows and reach_cols
    columns of it; and the queue. state is work space at least the window's size, all _UNSEEN, and queue work space
    of any size, which grows as the search needs: the queue returned is the one to pass next."""
    nrows, ncols = mask.shape
    r0, r1 = max(row - reach_rows, 0), min(row + reach_rows + 1, nrows)
    c0, c1 = max(col - reach_cols, 0), min(col + reach_cols + 1, ncols)
    width = c1 - c0
    touched = 0  # the cells set, to clear afterwards: those cut and those queued
    for k in range(runs.shape[0]):
        if k == which:
            continue
        for i in range(runs[k, 0], runs[k, 1] + 1):
            for r in range(rows[i] - 1, rows[i] + 2):
                for c in range(cols[i] - 1, cols[i] + 2):
                    if r0 <= r < r1 and c0 <= c < c1:
                        cell = (r - r0) * width + (c - c0)
                        if state[cell] == _UNSEEN:
                            state[cell] = _CUT
                            queue = _room(queue, touched)
                            queue[touched] = cell
                            touched += 1

    head = touched
    for i in range(runs[which, 0], runs[which, 1] + 1):
        r, c = rows[i], cols[i]
        if r0 <= r < r1 and c0 <= c < c1:
            cell = (r - r0) * width + (c - c0)
            if state[cell] != _SEEN:
                state[cell] = _SEEN  # its own cells, even beside another run's
                queue = _room(queue, touched)
                queue[touched] = cell
                touched += 1

    up = down = False
    while head < touched and not (up and down):
        cell = queue[head]
        head += 1
        r, c = r0 + cell // width, c0 + cell % width
        reach_id = _stream_id(starts, stream_cols, stream_ids, r, c)
        if reach_id >= 0:
            for j in range(river.size):
                if river[j] == reach_id:
                    along = (c - col) * dx * cos + (r - row) * dy * sin
                    if along > margin:
                        down = True
                    elif along < -margin:
                        up = True
                    break

        for rr in range(r - 1, r + 2):
            for cc in range(c - 1, c + 2):
                if r0 <= rr < r1 and c0 <= cc < c1 and mask[rr, cc] > 0:
                    near = (rr - r0) * width + (cc - c0)
                    if state[near] == _UNSEEN:
                        state[near] = _SEEN
                        queue = _room(queue, touched)
                        queue[touched] = near
                        touched += 1

    for k in range(touched):
        state[queue[k]] = _UNSEEN
    return up and down, queue


@njit(cache=True, error_model="numpy")
def section_threads(mask, starts, stream_cols, stream_ids, river, rows, cols, elevations, spacing, row, col,
                    direction, dx, dy, reach_rows, reach_cols, margin, max_rise, state, queue):
    """A cross section's threads (see the notes above), a row for each, the stations of its water's two edges, those
    within THREAD_REACH of the stream cell; and the queue (see _joins_both_ways). The cross section's ordinates fall in
    cells rows and cols, its stream cell's (row, col) in the middle, with these elevations, spacing metres apart;
    direction is the stream's (arc.xsection.sampling's convention), and river the reach IDs whose stream cells are the
    river's. The search looks reach_rows rows and reach_cols columns either way, margin is the metres along the river
    beyond the cross section a thread's water must reach, and state and queue are work space for _joins_both_ways."""
    nrows, ncols = mask.shape
    # The ordinates within THREAD_REACH of the stream cell: a thread farther out couldn't join the river within it
    full = rows.size // 2
    half = min(full, int(THREAD_REACH / spacing))
    rows, cols = rows[full - half:full + half + 1], cols[full - half:full + half + 1]
    elevations = elevations[full - half:full + half + 1]
    size = rows.size
    center = size // 2
    water = np.zeros(size, np.bool_)
    for i in range(size):
        r, c = rows[i], cols[i]
        water[i] = 0 <= r < nrows and 0 <= c < ncols and mask[r, c] > 0
    water[center] = True
    runs = _runs(water)
    out = np.empty((runs.shape[0], 2))
    count = 0
    cos, sin = cos_sin(direction)
    highest = elevations[center] + max_rise
    for k in range(runs.shape[0]):
        first, last = runs[k, 0], runs[k, 1]
        if first <= center <= last:
            continue

        lowest = np.inf
        for i in range(first, last + 1):
            lowest = min(lowest, elevations[i])
        if not lowest <= highest:
            continue

        if _beside_a_stream(starts, stream_cols, stream_ids, rows, cols, first, last, nrows, ncols):
            continue

        joins, queue = _joins_both_ways(mask, starts, stream_cols, stream_ids, river, rows, cols, runs, k, row, col, cos,
                                        sin, dx, dy, reach_rows, reach_cols, margin, state, queue)
        if joins:
            out[count, 0] = (first - center - 0.5) * spacing
            out[count, 1] = (last - center + 0.5) * spacing
            count += 1

    return out[:count].copy(), queue


class ThreadFinder:
    """Finds cross sections' threads on a grid's water mask, keeping the work space between them."""

    def __init__(self, mask: np.ndarray, stream_index: tuple[np.ndarray, np.ndarray], stream_ids: np.ndarray,
                 dx: float, dy: float, network=None):
        self.mask = mask
        self.starts, self.stream_cols = stream_index
        self.stream_ids = np.asarray(stream_ids, np.int64)
        self.dx, self.dy = abs(float(dx)), abs(float(dy))
        # The search's window: THREAD_REACH metres either way across the grid and down it
        self.reach_rows = math.ceil(THREAD_REACH / self.dy)
        self.reach_cols = math.ceil(THREAD_REACH / self.dx)
        window = min(2 * self.reach_rows + 1, mask.shape[0]) * min(2 * self.reach_cols + 1, mask.shape[1])
        self.state = np.zeros(window, np.uint8)  # never more than the water mask's own cells
        self.queue = np.empty(min(window, 1 << 16), np.int64)  # cut cells, then the search's; grows as it needs
        self.network = network  # the reach network (arc.bathymetry.reach_network), or None
        self.rivers: dict[int, np.ndarray] = {}

    def river(self, reach: int) -> np.ndarray:
        """The reach IDs whose stream cells are the river at a reach: its own, the one it flows into and those that
        flow into it."""
        ids = self.rivers.get(reach)
        if ids is None:
            near = {int(reach)}
            if self.network is not None and reach in self.network:
                near.update(int(r) for r in self.network.successors(reach))
                near.update(int(r) for r in self.network.predecessors(reach))
            ids = np.array(sorted(near), dtype=np.int64)
            self.rivers[reach] = ids
        return ids

    def threads(self, rows: np.ndarray, cols: np.ndarray, elevations: np.ndarray, spacing: float, row: int, col: int,
                direction: float, reach: int) -> np.ndarray:
        """A cross section's threads (section_threads), a row for each: the stations of its water's edges."""
        threads, self.queue = section_threads(
            self.mask, self.starts, self.stream_cols, self.stream_ids, self.river(int(reach)), rows, cols,
            np.asarray(elevations, np.float64), float(spacing), int(row), int(col), float(direction), self.dx, self.dy,
            self.reach_rows, self.reach_cols, THREAD_MARGIN, MAX_THREAD_RISE, self.state, self.queue)
        return threads


def outside(threads: np.ndarray | None, left: float, right: float) -> np.ndarray | None:
    """The threads' parts beyond a channel's banks (left and right, distances from the stream cell), with any that
    lie within them dropped; None for none."""
    if threads is None or len(threads) == 0 or not (left >= 0.0 and right >= 0.0):
        return threads if threads is not None and len(threads) else None

    kept = []
    for low, high in np.asarray(threads, np.float64):
        if high <= -left or low >= right:
            kept.append((low, high))
        elif low < -left:
            kept.append((low, -left))
        elif high > right:
            kept.append((right, high))

    kept = [(low, high) for low, high in kept if high > low]
    return np.array(kept, dtype=np.float64) if kept else None


def reach_out(threads: np.ndarray | None) -> tuple[float, float]:
    """How far from the stream cell the outermost threads' outer edges are, left and right (0 for none)."""
    if threads is None or len(threads) == 0:
        return 0.0, 0.0
    threads = np.asarray(threads, np.float64)
    return float(max(-threads[:, 0].min(), 0.0)), float(max(threads[:, 1].max(), 0.0))


def in_threads(xs) -> np.ndarray:
    """Which of a cross section's ordinates are in its threads (XSection.threads), the edges included."""
    size = xs.elevations.size
    inside = np.zeros(size, dtype=bool)
    if xs.threads is None:
        return inside
    offsets = (np.arange(size) - size // 2) * float(xs.ordinate_distance)
    for low, high in np.asarray(xs.threads, np.float64):
        inside |= (offsets >= low) & (offsets <= high)
    return inside
