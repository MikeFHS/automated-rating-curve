"""A synthetic river valley and a 2D model of the steady flow through it, for test_valley_sections.py and the
documentation's figures of Valley_Sections (docs/figures/make_valley_figures.py).

The valley falls S_V down its length, along the raster's columns, with a flat floodplain between gently rising walls.
A trapezoidal channel is cut into the floodplain along a centreline that is straight, or that winds as a sine wave
between ENTRY and ENTRY + meander_length metres down the valley. Manning's n is the same everywhere.

The 2D model is LISFLOOD-FP's local inertial scheme (shallow_water.py, advection=False), run until the flow is
steady: Q enters the channel at the upstream edge and leaves the downstream edge at normal depth. It gives each cell's
water depth and its unit discharges, which show the direction the water moves there. It leaves out the velocity head
(3 to 12 cm in the channel, 0.4 to 3 cm over the floodplain). With it, solving the full shallow water equations
(advection=True), the channel's water stands higher, 0.41 m at 30 m^3/s, but that is the raster's doing: a channel
four cells wide crossing the grid at an angle loses momentum at each of its stair steps, and uniform flow down a
straight one at 26.6 or 45 degrees to the grid stands 0.44 m deeper than along it (0.18 and 0.10 m without advection;
test_shallow_water.py). It takes a few minutes for the
meandering valley, so the water surfaces it gives at the stream cells are kept in data/valley_meander_2d.npz, and

    python testing/synthetic_valley.py

makes them again.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from shallow_water import steady

ENTRY = 600.0  # metres of straight channel above the meanders
N = 0.04  # Manning's n, everywhere
FLOWS = (30.0, 65.0, 150.0, 400.0, 1000.0)  # m^3/s: 30 in the channel, 65 about bankfull, then over the floodplain
# Four meanders each, as far either side of the valley's axis as `amplitude`, every `wavelength` metres down it, in a
# channel 40 m wide at the top (`bottom` 30 m) unless given. The first four are long for their channel, 20 and 40
# widths a meander; natural meanders are 10 to 14 widths long (Leopold & Wolman, 1960), as the last three are
CASES = {
    "gentle": {"amplitude": 100.0, "wavelength": 800.0, "meander_length": 3200.0, "length": 4800.0},
    "meander": {"amplitude": 200.0, "wavelength": 800.0, "meander_length": 3200.0, "length": 4800.0},
    "strong": {"amplitude": 300.0, "wavelength": 800.0, "meander_length": 3200.0, "length": 4800.0},
    "long": {"amplitude": 400.0, "wavelength": 1600.0, "meander_length": 6400.0, "length": 8000.0},
    "natural": {"amplitude": 110.0, "wavelength": 440.0, "meander_length": 1760.0, "length": 3360.0},
    "natural_tight": {"amplitude": 165.0, "wavelength": 440.0, "meander_length": 1760.0, "length": 3360.0},
    "natural_wide": {"amplitude": 220.0, "wavelength": 880.0, "meander_length": 3520.0, "length": 5120.0,
                     "bottom": 70.0},
}
REFERENCE = Path(__file__).resolve().parent / "data" / "valley_meander_2d.npz"


def centreline(x: np.ndarray, kind: str, amplitude: float = 200.0, wavelength: float = 800.0,
               meander_length: float = 3200.0) -> np.ndarray:
    """The channel's centreline's distance across the valley from its axis, at x metres down it."""
    x = np.asarray(x, float)
    if kind == "straight":
        return np.zeros_like(x)
    s = np.clip(x - ENTRY, 0.0, meander_length)
    return amplitude * np.sin(2.0 * np.pi * s / wavelength)


def build(kind: str = "meander", *, dx: float = 10.0, length: float = 4800.0, width: float = 1500.0,
          floodplain: float = 1100.0, s_v: float = 0.001, z0: float = 100.0, bottom: float = 30.0, depth: float = 2.0,
          side: float = 2.5, wall: float = 0.05, **meander) -> dict:
    """The valley: its DEM (rows across the valley, columns down it), the ground without the channel, the centreline
    as points 0.5 m apart down the valley, and the stream cells in order along the centreline, 8-connected."""
    nx, ny = int(round(length / dx)), int(round(width / dx))
    x = (np.arange(nx) + 0.5) * dx
    y = (np.arange(ny) + 0.5) * dx - width / 2
    X, Y = np.meshgrid(x, y)
    ground = z0 - s_v * X + wall * np.clip(np.abs(Y) - floodplain / 2, 0, None)
    xs = np.arange(0.0, length + 1e-9, 0.5)
    ys = centreline(xs, kind, **meander)
    distance = np.full(X.shape, np.inf)  # from each cell to the centreline, column by column
    for j in range(nx):
        lo, hi = np.searchsorted(xs, x[j] - 300.0), np.searchsorted(xs, x[j] + 300.0)
        distance[:, j] = np.hypot(Y[:, j][:, None] - ys[None, lo:hi], X[:, j][:, None] - xs[None, lo:hi]).min(axis=1)
    cut = np.where(distance <= bottom / 2, depth, np.clip(depth - (distance - bottom / 2) / side, 0.0, None))
    cells = []  # the cells the centreline passes through
    for px, py in zip(xs, ys):
        cell = (min(max(int((py + width / 2) // dx), 0), ny - 1), min(int(px // dx), nx - 1))
        if not cells or cells[-1] != cell:
            cells.append(cell)
    path = [cells[0]]  # without the cells it only clips the corners of
    for k in range(1, len(cells) - 1):
        a, b = path[-1], cells[k + 1]
        if max(abs(a[0] - b[0]), abs(a[1] - b[1])) > 1:
            path.append(cells[k])
    path.append(cells[-1])
    return {"dem": ground - cut, "ground": ground, "dx": dx, "x": x, "y": y, "xs": xs, "ys": ys, "path": np.array(path),
            "s_v": s_v, "depth": depth, "wavelength": meander.get("wavelength", 800.0)}


def solve(valley: dict, q: float, n: float = N, *, h0: np.ndarray | None = None, advection: bool = False,
          alpha: float = 0.5, t_max: float = 200_000.0, tol: float = 0.002, check: int = 2000) -> dict:
    """The steady 2D flow of q m^3/s through the valley, from the depths h0 (dry without them): each cell's depth and
    its unit discharges down (ux) and across (uy) the valley, and the outflow it stopped at, within tol of q. The
    flow comes in across the channel at the upstream edge, and leaves the downstream edge at normal depth for the
    valley's slope. advection=True solves the full shallow water equations (see the notes above)."""
    inflow_rows = np.flatnonzero(valley["ground"][:, 0] - valley["dem"][:, 0] > 0.5 * valley["depth"])
    return steady(valley["dem"], n, valley["dx"], inflow_rows, q, s_out=valley["s_v"], h0=h0, advection=advection,
                  alpha=alpha, t_max=t_max, tol=tol, check=check)


def solve_flows(valley: dict, flows=FLOWS) -> dict:
    """solve for each flow in turn, each starting from the last one's depths: {flow: solve's result}."""
    out, h = {}, None
    for q in flows:
        out[q] = solve(valley, q, h0=h)
        h = out[q]["h"]
    return out


def meander(case: str = "meander") -> dict:
    """One of the CASES: 10 m cells, a channel 2 m deep winding four times across the valley. The tests use "meander",
    40 m wide, 200 m either side of the valley's axis every 800 m."""
    return build("meander", **CASES[case])


def middle(valley: dict, x: np.ndarray) -> np.ndarray:
    """Whether each distance down the valley is in its middle two meanders, clear of where they start and end."""
    wavelength = valley["wavelength"]
    return (x >= ENTRY + wavelength) & (x <= ENTRY + 3.0 * wavelength)


def reference() -> dict:
    """The 2D model's water surfaces at the meandering valley's stream cells, for each of FLOWS: {"path": the stream
    cells, "flows": FLOWS, "wse": stream cells by flows}."""
    with np.load(REFERENCE) as data:
        return {key: data[key] for key in data.files}


if __name__ == "__main__":
    valley = meander()
    flows = solve_flows(valley)
    rows, cols = valley["path"].T
    wse = np.stack([valley["dem"][rows, cols] + flows[q]["h"][rows, cols] for q in FLOWS], axis=1)
    REFERENCE.parent.mkdir(exist_ok=True)
    np.savez_compressed(REFERENCE, path=valley["path"], flows=np.array(FLOWS), wse=wse)
    print(f"wrote {REFERENCE}: outflows {[round(flows[q]['outflow'], 1) for q in FLOWS]}")
