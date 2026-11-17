"""A 2D model of shallow water flowing over a raster, for synthetic_valley.py and test_shallow_water.py: the shallow
water equations on a staggered grid (Stelling & Duinmeijer 2003, Int. J. Numer. Meth. Fluids 43), or with
advection=False the local inertial scheme of LISFLOOD-FP (Bates, Horritt & Fewtrell 2010, J. Hydrol. 387; de Almeida &
Bates 2013, Water Resour. Res. 49).

Each cell holds a water depth h, and each face between two cells a velocity u (between columns, along them) or v
(between rows, across them), worked out each time step from the slope of the water surface across it, the advection
of the water's momentum, and the Coriolis force, with Manning's friction semi-implicit:

    u' = (u - dt (g dη/dx + u du/dx + v du/dy + f v)) / (1 + g dt n^2 |u| / h_f^(4/3))
    v' = (v - dt (g dη/dy + u dv/dx + v dv/dy - f u)) / (1 + g dt n^2 |v| / h_f^(4/3))

with η = z + h the water surface, h_f the depth the water flows across the face at (the higher water surface less the
higher bed), and y along the rows (f > 0 turns the flow to its right, as in the northern hemisphere, with the rows
running south). The water across a face is the velocity times the depth upwind of it, its water surface less the higher
bed, which keeps each depth positive (the outflow over a time step is at most the water there). The advection is
first-order upwind, in Stelling & Duinmeijer's momentum-conservative form: each face's velocity takes up its
neighbours' as the water bringing it flows in,

    u du/dx + v du/dy = [max(q_w, 0) (u - u_w) + min(q_e, 0) (u_e - u) + max(q_s, 0) (u - u_s) + min(q_n, 0) (u_n - u)]
                        / (h_bar dx)

with q_w and q_e the discharges at the centres of the cells either side, q_s and q_n those across the rows at the face's
ends, u_w, u_e, u_s and u_n the neighbouring faces' velocities, and h_bar the mean of the two cells' depths, so that a
hydraulic jump stands where the momentum each side balances. Supercritical flow, jumps and wet fronts are stable with
it. The steady water surface falls at the friction slope less the change in velocity head, d(z + h + u^2/2g)/dx = -Sf,
as the 1D energy equation has it. Rain adds to each cell's depth. The time step is alpha dx / (sqrt(g h_max) +
max |u| + max |v|).

With advection=False the faces hold unit discharges q, updated as

    q' = (q - g h_f dt dη/dx) / (1 + g dt n^2 |q| / h_f^(7/3)),

the momentum equation with its local acceleration but no advection (the local inertial approximation): the steady
water surface falls at the friction slope alone, d(z + h)/dx = -Sf, with no velocity head, and supercritical flow is
unstable. Its time step is alpha dx / sqrt(g h_max). Each cell's friction is its own depth's, as in a channel infinitely
wide, with either.

The water comes in across the upstream edge (the first column's west faces) on the rows given, and leaves across the
downstream edge (the last column's east faces) at normal depth for a slope (a wall for a slope of 0), or towards a
fixed water level at the edge, half a cell beyond the last column's centres. The other edges are walls.
"""
from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
from numba import njit

G = 9.81
DRY = 1e-4  # metres: no water crosses a face shallower than this
ADVECTION_DEPTH = 1e-3  # metres: nor is momentum advected where the water is shallower than this


@njit
def _step(z, h, qx, qy, n, dx, dt, inflow_rows, inflow_q, s_out, stage_out):
    """One time step of the local inertial scheme: the unit discharges across the cells' faces, then the depths.
    stage_out is the water level at the downstream edge, or NaN for the water to leave at normal depth for the slope
    s_out."""
    ny, nx = z.shape
    eta = z + h
    for r in range(ny):
        for c in range(1, nx):
            flow_depth = max(eta[r, c - 1], eta[r, c]) - max(z[r, c - 1], z[r, c])
            if flow_depth > 1e-4:
                slope = (eta[r, c] - eta[r, c - 1]) / dx
                q = qx[r, c]
                qx[r, c] = (q - G * flow_depth * dt * slope) / (1.0 + G * dt * n * n * abs(q)
                                                           / flow_depth ** (7.0 / 3.0))
            else:
                qx[r, c] = 0.0
        if stage_out != stage_out:
            depth = h[r, nx - 1]
            qx[r, nx] = depth ** (5.0 / 3.0) * math.sqrt(s_out) / n if depth > 1e-4 else 0.0
        else:
            flow_depth = max(eta[r, nx - 1], stage_out) - z[r, nx - 1]
            if flow_depth > 1e-4:
                slope = (stage_out - eta[r, nx - 1]) / (0.5 * dx)
                q = qx[r, nx]
                qx[r, nx] = (q - G * flow_depth * dt * slope) / (1.0 + G * dt * n * n * abs(q)
                                                             / flow_depth ** (7.0 / 3.0))
            else:
                qx[r, nx] = 0.0
        qx[r, 0] = 0.0
    for k in range(inflow_rows.size):
        qx[inflow_rows[k], 0] = inflow_q[k]
    for r in range(1, ny):
        for c in range(nx):
            flow_depth = max(eta[r - 1, c], eta[r, c]) - max(z[r - 1, c], z[r, c])
            if flow_depth > 1e-4:
                slope = (eta[r, c] - eta[r - 1, c]) / dx
                q = qy[r, c]
                qy[r, c] = (q - G * flow_depth * dt * slope) / (1.0 + G * dt * n * n * abs(q)
                                                           / flow_depth ** (7.0 / 3.0))
            else:
                qy[r, c] = 0.0
    for r in range(ny):
        for c in range(nx):
            h[r, c] = max(h[r, c] + dt * (qx[r, c] - qx[r, c + 1] + qy[r, c] - qy[r + 1, c]) / dx, 0.0)


@njit
def _fluxes(z, h, ux, uy, qx, qy, inflow_rows, inflow_q, s_out, stage_out, n):
    """The water across each face: its velocity times the depth upwind of it (the shallow water equations' step), and
    the velocities at the edges' faces that go with the inflow and outflow."""
    ny, nx = z.shape
    for r in range(ny):
        for c in range(1, nx):
            bed = max(z[r, c - 1], z[r, c])
            u = ux[r, c]
            upwind = z[r, c - 1] + h[r, c - 1] if u > 0.0 else z[r, c] + h[r, c]
            qx[r, c] = u * max(upwind - bed, 0.0)
        depth = h[r, nx - 1]
        if stage_out != stage_out:
            q = depth ** (5.0 / 3.0) * math.sqrt(s_out) / n if depth > DRY else 0.0
            qx[r, nx] = q
            ux[r, nx] = q / depth if depth > DRY else 0.0
        else:
            u = ux[r, nx]
            qx[r, nx] = u * max(depth if u > 0.0 else stage_out - z[r, nx - 1], 0.0)
        qx[r, 0] = 0.0
        ux[r, 0] = 0.0
    for k in range(inflow_rows.size):
        r = inflow_rows[k]
        qx[r, 0] = inflow_q[k]
        ux[r, 0] = inflow_q[k] / h[r, 0] if h[r, 0] > ADVECTION_DEPTH else 0.0
    for r in range(1, ny):
        for c in range(nx):
            bed = max(z[r - 1, c], z[r, c])
            v = uy[r, c]
            upwind = z[r - 1, c] + h[r - 1, c] if v > 0.0 else z[r, c] + h[r, c]
            qy[r, c] = v * max(upwind - bed, 0.0)


@njit
def _step_swe(z, h, ux, uy, qx, qy, n, dx, dt, inflow_rows, inflow_q, s_out, stage_out, rain, coriolis):
    """One time step of the shallow water equations: the velocities across the cells' faces (along the columns, then
    across them, with the new velocities along them in the Coriolis force), the water across them, then the depths.
    qx and qy come in as the last step's water across the faces, which the advection takes."""
    ny, nx = z.shape
    eta = z + h
    friction = G * dt * n * n
    u_old = ux.copy()
    for r in range(ny):
        for c in range(1, nx + 1):
            u = u_old[r, c]
            if c == nx:
                if stage_out != stage_out:
                    continue  # set by _fluxes
                flow_depth = max(eta[r, nx - 1], stage_out) - z[r, nx - 1]
                slope = (stage_out - eta[r, nx - 1]) / (0.5 * dx)
                mean_depth = h[r, nx - 1]
            else:
                flow_depth = max(eta[r, c - 1], eta[r, c]) - max(z[r, c - 1], z[r, c])
                slope = (eta[r, c] - eta[r, c - 1]) / dx
                mean_depth = 0.5 * (h[r, c - 1] + h[r, c])
            if flow_depth <= DRY:
                ux[r, c] = 0.0
                continue
            force = -G * slope
            if mean_depth > ADVECTION_DEPTH:
                west = 0.5 * (qx[r, c - 1] + qx[r, c])
                brought = max(west, 0.0) * (u - u_old[r, c - 1])
                if c < nx:
                    east = 0.5 * (qx[r, c] + qx[r, c + 1])
                    brought += min(east, 0.0) * (u_old[r, c + 1] - u)
                    if r > 0:
                        brought += max(0.5 * (qy[r, c - 1] + qy[r, c]), 0.0) * (u - u_old[r - 1, c])
                    if r < ny - 1:
                        brought += min(0.5 * (qy[r + 1, c - 1] + qy[r + 1, c]), 0.0) * (u_old[r + 1, c] - u)
                # and rain falls with none, so the water it adds has to be brought up to speed
                force -= brought / (mean_depth * dx) + rain * u / mean_depth
            if coriolis != 0.0 and c < nx:
                force -= coriolis * 0.25 * (uy[r, c - 1] + uy[r, c] + uy[r + 1, c - 1] + uy[r + 1, c])
            ux[r, c] = (u + dt * force) / (1.0 + friction * abs(u) / flow_depth ** (4.0 / 3.0))
    v_old = uy.copy()
    for r in range(1, ny):
        for c in range(nx):
            v = v_old[r, c]
            flow_depth = max(eta[r - 1, c], eta[r, c]) - max(z[r - 1, c], z[r, c])
            if flow_depth <= DRY:
                uy[r, c] = 0.0
                continue
            force = -G * (eta[r, c] - eta[r - 1, c]) / dx
            mean_depth = 0.5 * (h[r - 1, c] + h[r, c])
            if mean_depth > ADVECTION_DEPTH:
                brought = max(0.5 * (qy[r - 1, c] + qy[r, c]), 0.0) * (v - v_old[r - 1, c])
                brought += min(0.5 * (qy[r, c] + qy[r + 1, c]), 0.0) * (v_old[r + 1, c] - v)
                # Water coming in across the edges brings no velocity across the rows
                brought += max(0.5 * (qx[r - 1, c] + qx[r, c]), 0.0) * (v - (v_old[r, c - 1] if c > 0 else 0.0))
                brought += min(0.5 * (qx[r - 1, c + 1] + qx[r, c + 1]), 0.0) * \
                    ((v_old[r, c + 1] if c < nx - 1 else 0.0) - v)
                force -= brought / (mean_depth * dx) + rain * v / mean_depth
            if coriolis != 0.0:
                force += coriolis * 0.25 * (ux[r - 1, c] + ux[r - 1, c + 1] + ux[r, c] + ux[r, c + 1])
            uy[r, c] = (v + dt * force) / (1.0 + friction * abs(v) / flow_depth ** (4.0 / 3.0))
    _fluxes(z, h, ux, uy, qx, qy, inflow_rows, inflow_q, s_out, stage_out, n)
    for r in range(ny):
        for c in range(nx):
            h[r, c] = max(h[r, c] + dt * ((qx[r, c] - qx[r, c + 1] + qy[r, c] - qy[r + 1, c]) / dx + rain), 0.0)


class _Model:
    """The state of a run and its time steps, for steady and unsteady."""

    def __init__(self, z, n, dx, inflow_rows, s_out, stage_out, h0, u0, v0, rain, coriolis, advection, alpha):
        self.z, self.n, self.dx, self.alpha = np.asarray(z, dtype=np.float64), float(n), float(dx), float(alpha)
        ny, nx = self.z.shape
        self.h = np.zeros((ny, nx)) if h0 is None else np.array(h0, dtype=np.float64)
        self.qx, self.qy = np.zeros((ny, nx + 1)), np.zeros((ny + 1, nx))
        self.inflow_rows = np.asarray(inflow_rows, dtype=np.int64)
        self.s_out, self.stage = float(s_out), math.nan if stage_out is None else float(stage_out)
        self.rain, self.coriolis, self.advection = float(rain), float(coriolis), bool(advection)
        if not self.advection and (rain or coriolis or u0 is not None or v0 is not None):
            raise ValueError("The local inertial scheme (advection=False) has no rain, Coriolis force or velocities "
                             "to start from.")
        if self.advection:
            self.ux = np.zeros((ny, nx + 1)) if u0 is None else np.array(u0, dtype=np.float64)
            self.uy = np.zeros((ny + 1, nx)) if v0 is None else np.array(v0, dtype=np.float64)
            _fluxes(self.z, self.h, self.ux, self.uy, self.qx, self.qy, self.inflow_rows,
                    np.zeros(self.inflow_rows.size), self.s_out, self.stage, self.n)

    def time_step(self) -> float:
        celerity = math.sqrt(G * max(self.h.max(), 0.1))
        if not self.advection:
            return self.alpha * self.dx / celerity
        return self.alpha * self.dx / (celerity + np.abs(self.ux).max() + np.abs(self.uy).max())

    def step(self, dt: float, q: float) -> None:
        inflow_q = np.full(self.inflow_rows.size, q / (max(self.inflow_rows.size, 1) * self.dx))
        if self.advection:
            # Shared by the cells' conveyance, depth^(5/3), under their mean water surface, as HEC-RAS shares a flow
            # boundary, once they're wet
            depths = self.h[self.inflow_rows, 0]
            wet = depths > DRY
            level = float(np.mean((self.z[self.inflow_rows, 0] + depths)[wet])) if wet.any() else -math.inf
            conveyance = np.clip(level - self.z[self.inflow_rows, 0], 0.0, None) ** (5.0 / 3.0)
            if conveyance.sum() > 0.0:
                inflow_q = q * conveyance / (conveyance.sum() * self.dx)
            _step_swe(self.z, self.h, self.ux, self.uy, self.qx, self.qy, self.n, self.dx, dt, self.inflow_rows,
                      inflow_q, self.s_out, self.stage, self.rain, self.coriolis)
        else:
            _step(self.z, self.h, self.qx, self.qy, self.n, self.dx, dt, self.inflow_rows, inflow_q, self.s_out,
                  self.stage)

    def outflow(self) -> float:
        return float(self.qx[:, -1].sum() * self.dx)


def steady(z: np.ndarray, n: float, dx: float, inflow_rows: np.ndarray, q: float, *, s_out: float = 0.0,
           stage_out: float | None = None, h0: np.ndarray | None = None, advection: bool = True, alpha: float = 0.5,
           t_max: float = 200_000.0, tol: float = 0.002, check: int = 2000) -> dict:
    """The steady flow of q m^3/s, coming in evenly across inflow_rows, over the bed z (rows by columns, dx metres
    square), from the depths h0 (dry without them), still: each cell's depth and its unit discharges along (ux) and
    across (uy) the columns, and the outflow it stopped at, within tol of q and steady over three checks."""
    model = _Model(z, n, dx, inflow_rows, s_out, stage_out, h0, None, None, 0.0, 0.0, advection, alpha)
    t, steps, last, stable = 0.0, 0, None, 0
    while t < t_max:
        dt = model.time_step()
        model.step(dt, q)
        t += dt
        steps += 1
        if steps % check == 0:
            out = model.outflow()
            stable = stable + 1 if last is not None and abs(out - q) < tol * q and abs(out - last) < 0.2 * tol * q \
                else 0
            if stable >= 3:
                break
            last = out
    return {"h": model.h, "ux": 0.5 * (model.qx[:, :-1] + model.qx[:, 1:]),
            "uy": 0.5 * (model.qy[:-1, :] + model.qy[1:, :]), "outflow": model.outflow()}


def unsteady(z: np.ndarray, n: float, dx: float, times, *, inflow_rows=(),
             inflow: Callable[[float], float] | None = None, s_out: float = 0.0, stage_out: float | None = None,
             h0: np.ndarray | None = None, u0: np.ndarray | None = None, v0: np.ndarray | None = None, rain: float = 0.0, coriolis: float = 0.0,
             advection: bool = True, alpha: float = 0.5) -> dict:
    """The flow over the bed z from the depths h0 and face velocities u0 (rows by columns + 1) and v0 (rows + 1 by
    columns), still and dry without them, with inflow(t) m^3/s coming in evenly across inflow_rows at t seconds
    (taken at the middle of each time step), rain m/s on every cell and the Coriolis parameter coriolis (1/s): each
    cell's depth and its faces' velocities at each of times (ascending; the velocities only with advection), and the
    outflow (m^3/s) then, and the water that came in, rained and went out by then (m^3)."""
    model = _Model(z, n, dx, inflow_rows, s_out, stage_out, h0, u0, v0, rain, coriolis, advection, alpha)
    area = model.z.size * model.dx ** 2
    t, came_in, rained, went_out = 0.0, 0.0, 0.0, 0.0
    out = {"h": [], "ux": [], "uy": [], "outflow": [], "volume_in": [], "volume_rain": [], "volume_out": []}
    for end in times:
        while t < end - 1e-9:
            dt = min(model.time_step(), end - t)
            q = inflow(t + 0.5 * dt) if inflow is not None else 0.0
            model.step(dt, q)
            came_in += q * dt
            rained += model.rain * area * dt
            went_out += model.outflow() * dt
            t += dt
        out["h"].append(model.h.copy())
        if model.advection:
            out["ux"].append(model.ux.copy())
            out["uy"].append(model.uy.copy())
        out["outflow"].append(model.outflow())
        out["volume_in"].append(came_in)
        out["volume_rain"].append(rained)
        out["volume_out"].append(went_out)
    return {key: np.array(value) for key, value in out.items()}
