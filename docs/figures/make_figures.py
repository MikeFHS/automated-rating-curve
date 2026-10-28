"""Draw the documentation's figures with ARC itself, into docs/images.

    python docs/figures/make_figures.py

A made-up valley is sampled, its banks found, its channel carved and its rating curve worked out by the same
functions arc.pipeline uses, so the figures show what ARC does. It needs matplotlib.
"""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch
from scipy.integrate import quad
from scipy.optimize import brentq

from arc.bathymetry import (
    bathymetry_depth,
    carve_channel,
    find_banks,
    set_bank_distances,
    set_in_bank_roughness,
)
from arc.hydraulics import DepthRoughness, hydraulic_profile, wse_for_discharge
from arc.outputs.curve_files import (
    DEPTH_GUESS,
    TOP_WIDTH_GUESS,
    VELOCITY_GUESS,
    fit_power_law,
)
from arc.outputs.rating_curves import WSE, Q, T, V
from arc.rating_curve import rating_curve
from arc.standard_step import EDDY_LENGTH, G, HydraulicTable, _eddy_ends, _march, _packed, step_up
from arc.xsection.xsection import XSection

IMAGES = Path(__file__).resolve().parent.parent / "images"

# Colours: one blue for water and the rating curve's points, orange for the fitted power law, inks for the ground
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, SECONDARY, MUTED, GRID = "#0b0b0b", "#52514e", "#a8a7a1", "#e6e5e0"

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"], "font.size": 10,
    "svg.fonttype": "none", "axes.edgecolor": MUTED, "axes.labelcolor": SECONDARY, "xtick.color": SECONDARY,
    "ytick.color": SECONDARY, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True, "figure.facecolor": "white",
    "axes.facecolor": "white", "savefig.facecolor": "white", "svg.hashsalt": "arc",
})
SVG_METADATA = {"Date": None}  # no date, so drawing the figures again changes the files only if the figures change

SPACING = 5.0  # metres between the cross section's points
SLOPE = 0.001
WATER_N, LAND_N = 0.03, 0.05
BASEFLOW, MAX_FLOW = 12.0, 260.0
TRAPEZOID_HEIGHT = 0.2  # Bathy_Trap_H
INCREMENTS = 15  # VDT_Database_NumIterations
ROUGHNESS = DepthRoughness(6.0, 2.0, 1.0)  # the defaults of k_decay, shallow_factor and deep_factor


def valley_ground(x: np.ndarray) -> np.ndarray:
    """A valley as a DEM shows it: the water surface flat at 100 m across a channel 30 m wide, steep banks to 102 m,
    floodplains rising gently (the left one lower), and valley walls."""
    distance = np.abs(x)
    ground = np.where(distance <= 15.0, 100.0, 100.0 + np.clip(distance - 15.0, 0.0, 5.0) * 0.4)
    floodplain = np.clip(distance - 20.0, 0.0, None)
    ground = ground + np.where(x < 0, 0.006, 0.012) * floodplain
    ground = ground + 0.08 * np.clip(distance - 140.0, 0.0, None)
    return ground


def carved_section() -> tuple[XSection, np.ndarray, object, float]:
    """The valley's cross section, as arc.pipeline makes it: banks, the water's n between them, the channel carved
    for the baseflow, and the banks set for the hydraulics. Also the ground before carving, the banks and the depth."""
    x = SPACING * np.arange(-40, 41)
    ground = valley_ground(x)
    xs = XSection(ground.copy(), np.full(x.size, LAND_N), SPACING)
    banks = find_banks(xs)
    set_in_bank_roughness(xs, banks, WATER_N)
    depth = bathymetry_depth(xs, banks, BASEFLOW, SLOPE, trapezoid_height=TRAPEZOID_HEIGHT, mannings_n=WATER_N)
    carve_channel(xs, banks, depth.depth, trapezoid_height=TRAPEZOID_HEIGHT)
    set_bank_distances(xs, banks)
    return xs, ground, banks, depth.depth


def water_edges(profile, wse: float) -> tuple[float, float]:
    """The distances from the stream cell to the water's edges, left and right, on a cross section's profile."""
    stations, elevations, center = profile.stations, profile.elevations, profile.center
    edges = []
    for step in (-1, 1):
        j = center
        while 0 <= j + step < stations.size and elevations[j + step] < wse:
            j += step
        if not 0 <= j + step < stations.size:
            edges.append(abs(float(stations[j])))
            continue
        share = (wse - elevations[j]) / (elevations[j + step] - elevations[j])
        edges.append(abs(float(stations[j] + share * (stations[j + step] - stations[j]))))
    return edges[0], edges[1]


def arrow(ax, start, end, text=None, text_xy=None, **text_kwargs) -> None:
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="<|-|>", mutation_scale=9, color=SECONDARY, lw=1.0,
                                 shrinkA=0, shrinkB=0))
    if text:
        ax.annotate(text, text_xy, color=INK, fontsize=9, **text_kwargs)


def cross_section_figure() -> None:
    xs, ground, banks, depth = carved_section()
    profile = hydraulic_profile(xs)
    stations, elevations = profile.stations, profile.elevations
    bed = float(elevations[profile.center])
    curve = rating_curve(xs, MAX_FLOW, SLOPE, INCREMENTS, baseflow=BASEFLOW, roughness=ROUGHNESS)
    flood = float(curve.increments[-1, WSE])
    x = SPACING * np.arange(-40, 41)

    fig, (whole, zoom) = plt.subplots(2, 1, figsize=(8.0, 7.2), gridspec_kw={"height_ratios": [1.15, 1.0]})

    # The whole cross section at the maximum flow
    left_edge, right_edge = water_edges(profile, flood)
    water_x = np.linspace(-left_edge, right_edge, 400)
    whole.fill_between(water_x, np.interp(water_x, stations, elevations), flood, color=BLUE, alpha=0.12, lw=0)
    whole.plot([-left_edge, right_edge], [flood, flood], color=BLUE, lw=2, solid_capstyle="round")
    whole.plot(stations, elevations, color=INK, lw=1.6)
    for bank in (-banks.left, banks.right):
        whole.plot([bank, bank], [float(np.interp(bank, stations, elevations)), flood], color=SECONDARY, lw=0.9)
    whole.annotate("overbank", (-0.5 * (left_edge + banks.left), flood + 0.1), ha="center", va="bottom",
                   color=SECONDARY, fontsize=9)
    whole.annotate("overbank", (0.5 * (right_edge + banks.right) - 5.0, flood + 0.1), ha="center", va="bottom",
                   color=SECONDARY, fontsize=9)
    whole.annotate("channel", (0.0, 100.6), ha="center", va="center", color=SECONDARY, fontsize=9)
    arrow(whole, (-left_edge, flood + 0.9), (right_edge, flood + 0.9), "top width (t)",
          (0.5 * (right_edge - left_edge), flood + 1.05), ha="center", va="bottom")
    whole.annotate("water surface at the maximum flow (wse)", (95.0, flood + 1.75), color=INK, fontsize=9,
                   ha="center", va="bottom")
    whole.annotate("", (right_edge - 3.0, flood), (95.0, flood + 1.7),
                   arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    whole.annotate("ground", (170.0, float(valley_ground(np.array([170.0]))[0]) + 0.2), color=SECONDARY,
                   fontsize=9, ha="right", va="bottom")
    whole.set_xlim(-205, 205)
    whole.set_ylim(98.5, flood + 2.6)
    whole.set_ylabel("Elevation (m)")
    whole.set_title("The cross section at its maximum flow", loc="left", color=INK, fontsize=11)
    whole.set_xlabel("Distance from the stream cell (m)")

    # The channel, carved below the DEM's water surface
    inside = np.abs(x) <= 40.0
    zoom.plot(x[inside], ground[inside], color=MUTED, lw=2, label="DEM")
    near = np.abs(stations) <= 40.0
    zoom.plot(stations[near], elevations[near], color=INK, lw=1.6, label="Cross section ARC rates (carved)")
    baseflow_wse = 100.0
    left_bf, right_bf = water_edges(profile, baseflow_wse)
    zoom.fill_between(np.linspace(-left_bf, right_bf, 200),
                      np.interp(np.linspace(-left_bf, right_bf, 200), stations, elevations), baseflow_wse,
                      color=BLUE, alpha=0.12, lw=0)
    zoom.plot([-left_bf, right_bf], [baseflow_wse, baseflow_wse], color=BLUE, lw=2, solid_capstyle="round",
              label="Water surface at the baseflow (the DEM's)")
    zoom.scatter([0.0], [bed], s=46, color=INK, zorder=5, edgecolor="white", linewidth=1.5)
    zoom.annotate("Elev (VDT database) = BaseElev (curve file):\nthe bed at the centre of the cross section",
                  (1.0, bed - 0.12), color=INK, fontsize=9, va="top")
    arrow(zoom, (-4.0, bed), (-4.0, baseflow_wse), "depth = wse − Elev", (-5.0, 0.5 * (bed + baseflow_wse)),
          ha="right", va="center")
    width = banks.left + banks.right
    side = TRAPEZOID_HEIGHT * width
    for sign, edge in ((-1, banks.left), (1, banks.right)):
        zoom.scatter([sign * edge], [float(np.interp(sign * edge, stations, elevations))], s=46, color=SECONDARY,
                     zorder=5, edgecolor="white", linewidth=1.5)
    zoom.annotate("bank", (-banks.left - 0.8, 100.15), color=SECONDARY, fontsize=9, ha="right", va="bottom")
    zoom.annotate("bank", (banks.right + 0.8, 100.15), color=SECONDARY, fontsize=9, ha="left", va="bottom")
    arrow(zoom, (-banks.left, bed - 0.12), (-banks.left + side, bed - 0.12), "Bathy_Trap_H × width",
          (-banks.left - 0.8, bed - 0.12), ha="right", va="center")
    zoom.set_xlim(-40, 40)
    zoom.set_ylim(bed - 1.0, 102.6)
    zoom.set_xlabel("Distance from the stream cell (m)")
    zoom.set_ylabel("Elevation (m)")
    zoom.set_title("The channel, carved below the DEM's water surface", loc="left", color=INK, fontsize=11)
    zoom.legend(loc="upper center", frameon=False, fontsize=9)

    fig.tight_layout(h_pad=2.0)
    fig.savefig(IMAGES / "cross_section.svg", metadata=SVG_METADATA)
    plt.close(fig)
    print(f"cross_section.svg: bed {bed:.2f} m, channel depth {depth:.2f} m, banks {banks.left:.1f} and "
          f"{banks.right:.1f} m ({banks.method}), flood WSE {flood:.2f} m")


def rating_curve_figure() -> None:
    xs, _, _, _ = carved_section()
    profile = hydraulic_profile(xs)
    bed = float(profile.elevations[profile.center])
    curve = rating_curve(xs, MAX_FLOW, SLOPE, INCREMENTS, baseflow=BASEFLOW, roughness=ROUGHNESS)
    q, v, t, wse = (curve.increments[:, field] for field in (Q, V, T, WSE))
    depth = wse - bed
    fits = {"depth": fit_power_law(q, depth, DEPTH_GUESS), "top width": fit_power_law(q, t, TOP_WIDTH_GUESS),
            "velocity": fit_power_law(q, v, VELOCITY_GUESS)}
    smooth = np.linspace(q.min(), q.max(), 200)

    fig, axes = plt.subplots(1, 3, figsize=(9.0, 3.3))
    panels = (("depth", depth, "Depth (m)", "wse − Elev"), ("top width", t, "Top width (m)", "t"),
              ("velocity", v, "Velocity (m/s)", "v"))
    for ax, (name, values, label, column) in zip(axes, panels):
        a, b, _ = fits[name]
        ax.plot(smooth, a * smooth ** b, color=ORANGE, lw=2, label="curve file: a·Qᵇ")
        ax.scatter(q, values, s=40, color=BLUE, edgecolor="white", linewidth=1.2, zorder=3,
                   label="VDT database: the flows")
        ax.set_title(f"{label.split(' (')[0]} ({column})", loc="left", color=INK, fontsize=10)
        ax.set_xlabel("Discharge, q (m³/s)")
        ax.set_ylabel(label)
        ax.set_xlim(0, q.max() * 1.05)
        ax.set_ylim(0, values.max() * 1.12)
    axes[0].legend(loc="lower right", frameon=False, fontsize=8.5)
    fig.tight_layout(w_pad=1.6)
    fig.savefig(IMAGES / "rating_curve.svg", metadata=SVG_METADATA)
    plt.close(fig)
    print("rating_curve.svg:", {name: (round(a, 3), round(b, 3)) for name, (a, b, _) in fits.items()})


def directions_figure() -> None:
    """The direction convention on a north-up grid: a stream running 30 degrees clockwise from east."""
    flow = math.radians(30.0)
    xs_angle = flow + math.pi / 2  # the cross section's direction, towards the higher rows
    centre = np.array([4.5, 4.5])  # (column, row) of the centre cell's middle

    fig, ax = plt.subplots(figsize=(6.4, 6.0))
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for k in range(10):
        ax.plot([k, k], [0, 9], color=GRID, lw=0.8, zorder=0)
        ax.plot([0, 9], [k, k], color=GRID, lw=0.8, zorder=0)
    # The stream's cells, along its line, and the centre cell darker
    cells = set()
    for s in np.linspace(-6.0, 6.0, 120):
        col, row = centre + s * np.array([math.cos(flow), math.sin(flow)])
        if 0 <= col < 9 and 0 <= row < 9:
            cells.add((math.floor(col), math.floor(row)))
    for col, row in cells:
        alpha = 0.45 if (col, row) == (4, 4) else 0.18
        ax.add_patch(plt.Rectangle((col, row), 1, 1, color=BLUE, alpha=alpha, lw=0))
    # The stream's direction
    along = np.array([math.cos(flow), math.sin(flow)])
    ax.plot(*np.array([centre - 4.2 * along, centre + 4.2 * along]).T, color=BLUE, lw=2, solid_capstyle="round")
    ax.annotate("stream direction: 30°", centre + 4.3 * along + np.array([-0.3, 0.45]), color=INK, fontsize=10,
                ha="center", va="top")
    # The cross section, side 1 towards the higher rows
    direction = np.array([math.cos(xs_angle), math.sin(xs_angle)])
    side1, side2 = centre + 3.9 * direction, centre - 3.9 * direction
    ax.plot(*np.array([side2, side1]).T, color=INK, lw=2, solid_capstyle="round")
    ax.annotate("side 1: towards the higher rows", side1 + np.array([0.15, 0.05]), color=INK, fontsize=9, va="top")
    ax.annotate("side 2", side2 + np.array([0.2, -0.05]), color=INK, fontsize=9, va="bottom")
    # XS_Angle, from the +column axis towards the +row axis
    ax.plot([centre[0], centre[0] + 2.4], [centre[1], centre[1]], color=SECONDARY, lw=1, ls=(0, (2, 2)))
    angles = np.linspace(0.0, xs_angle, 60)
    ax.plot(centre[0] + 1.3 * np.cos(angles), centre[1] + 1.3 * np.sin(angles), color=ORANGE, lw=2)
    ax.annotate("XS_Angle", centre + 1.55 * np.array([math.cos(0.55 * xs_angle), math.sin(0.55 * xs_angle)]),
                color=INK, fontsize=10, ha="left", va="top")
    ax.annotate("0: along a row,\ntowards higher columns", (centre[0] + 2.5, centre[1] - 0.1), color=SECONDARY,
                fontsize=8.5, va="bottom")
    # Axes: rows grow downwards
    ax.set_xlim(-0.6, 9.4)
    ax.set_ylim(9.6, -0.9)
    ax.set_xticks(np.arange(9) + 0.5, [str(k) for k in range(9)])
    ax.set_yticks(np.arange(9) + 0.5, [str(k) for k in range(9)])
    ax.tick_params(length=0)
    ax.set_xlabel("Column  →  (east on a north-up raster)")
    ax.set_ylabel("Row  ↓  (south on a north-up raster)")
    ax.xaxis.set_label_position("top")
    ax.xaxis.tick_top()
    ax.set_aspect("equal")
    ax.set_title("XS_Angle = 120° = 2.094 rad for a stream running 30° clockwise from east", color=INK,
                 fontsize=10, y=-0.08)
    fig.tight_layout()
    fig.savefig(IMAGES / "directions.svg", metadata=SVG_METADATA)
    plt.close(fig)
    print("directions.svg")


def reach_section(bed: float, bottom_width: float, spacing: float = SPACING) -> XSection:
    """A trapezoidal channel 4 m deep, its sides sloping 1 in 2, in a floodplain rising 1 in 100, with points
    `spacing` apart."""
    x = np.arange(-300.0, 300.0 + spacing / 2, spacing)
    edge = np.abs(x) - bottom_width / 2
    ground = bed + np.clip(edge / 2.0, 0.0, 4.0) + 0.01 * np.maximum(edge - 8.0, 0.0)
    return XSection(ground, np.full(x.size, 0.035), spacing)


def standard_step_figure() -> None:
    """A reach 3 km long, its channel 40 m wide but for a stretch 20 m wide in the middle: each cross section's own
    normal depth, and the standard step's water surface (Standard_Step)."""
    flow, slope, spacing = 150.0, 0.0005, 30.0
    stations = np.arange(0.0, 3000.0 + spacing / 2, spacing)
    narrow = (stations >= 1350.0) & (stations <= 1650.0)
    beds = 100.0 - slope * stations
    sections = [reach_section(bed, 20.0 if n else 40.0) for bed, n in zip(beds, narrow)]
    normal = np.array([wse_for_discharge(xs, flow, slope) for xs in sections])
    tables = [HydraulicTable(xs, float(w)) for xs, w in zip(sections, normal)]
    linked = np.empty(stations.size)
    linked[-1] = max(normal[-1], tables[-1].critical(flow))
    for m in range(stations.size - 2, -1, -1):
        linked[m], _ = step_up(tables[m + 1], linked[m + 1], flow, tables[m], flow, spacing)

    km = stations / 1000.0
    fig, ax = plt.subplots(figsize=(9.0, 3.6))
    ax.axvspan(1.35, 1.65, color=GRID, lw=0)
    ax.text(1.5, beds.min() + 0.15, "channel 20 m wide", ha="center", va="bottom", color=SECONDARY, fontsize=8.5)
    ax.text(0.3, beds.min() + 0.15, "channel 40 m wide", ha="center", va="bottom", color=SECONDARY, fontsize=8.5)
    ax.plot(km, beds, color=INK, lw=1.5, label="bed")
    ax.plot(km, normal, color=ORANGE, lw=2, marker="o", ms=3, label="each cross section on its own (normal depth)")
    ax.plot(km, linked, color=BLUE, lw=2, label="standard step (Standard_Step)")
    ax.set_title(f"The water surface along a reach at {flow:.0f} m³/s, flowing left to right", loc="left", color=INK,
                 fontsize=10)
    ax.set_xlabel("Distance downstream (km)")
    ax.set_ylabel("Elevation (m)")
    ax.set_xlim(0, km[-1])
    ax.set_ylim(beds.min() - 0.3, max(normal.max(), linked.max()) + 0.6)
    ax.legend(loc="upper right", frameon=False, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(IMAGES / "standard_step.svg", metadata=SVG_METADATA)
    plt.close(fig)
    print("standard_step.svg: depth wide %.2f m, narrow %.2f m; backed up %.2f m above normal depth at the top"
          % (normal[0] - beds[0], (normal - beds)[narrow].mean(), linked[0] - normal[0]))


def baseflow_hold_figure() -> None:
    """A reach 4 km long whose bank elevation is level for a kilometre at a time and then 0.5 m lower, as banks made
    never to rise downstream often are, while each channel is carved for the stream's even slope, so that at normal
    depth its baseflow fills it to its banks: the baseflow by the standard step, held at the bank elevation and not."""
    slope, spacing, terrace, drop, bottom_width = 0.0005, 30.0, 1000.0, 0.5, 30.0
    stations = np.arange(0.0, 4000.0, spacing)
    banks = 104.0 - drop * np.floor(stations / terrace)
    sections = []
    for bank in banks:
        xs = reach_section(bank - 4.0, bottom_width)
        xs.left_bank_distance = xs.right_bank_distance = bottom_width / 2 + 8.0  # the tops of the channel's sides
        sections.append(xs)

    tables = [HydraulicTable(xs, float(bank) + 2.0) for xs, bank in zip(sections, banks)]
    baseflow = float(tables[0].state(float(banks[0]))[1]) * math.sqrt(slope)  # what fills the channel to its banks
    normal = np.array([wse_for_discharge(xs, baseflow, slope) for xs in sections])

    # The march Standard_Step runs on each reach (arc.standard_step.apply_standard_step), from normal depth at the outlet
    flows = np.full((stations.size, 1), baseflow)
    linked = {}
    for held in (False, True):
        wse, *_ = _march(*_packed(tables), *tables[-1].parameters, flows, normal[-1:], np.full(flows.shape, held),
                         banks, np.diff(stations), _eddy_ends(stations, EDDY_LENGTH), 1.0)
        linked[held] = wse[:, 0]

    km = stations / 1000.0
    fig, ax = plt.subplots(figsize=(9.0, 3.6))
    ax.step(km, banks, where="post", color=INK, lw=1.5,
            label="bank elevation: each cross section on its own (normal depth)")
    ax.plot(km, linked[False], color=ORANGE, lw=2, label="standard step, not held")
    ax.plot(km, linked[True], color=BLUE, lw=2, label="standard step, held at the bank elevation (as ARC does)")
    ax.set_title(f"The baseflow ({baseflow:.0f} m³/s) along a reach whose banks are level for 1 km at a time, "
                 "flowing left to right", loc="left", color=INK, fontsize=10)
    ax.set_xlabel("Distance downstream (km)")
    ax.set_ylabel("Elevation (m)")
    ax.set_xlim(0, 4.0)
    ax.set_ylim(banks.min() - 0.3, linked[False].max() + 0.5)
    ax.legend(loc="upper right", frameon=False, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(IMAGES / "baseflow_hold.svg", metadata=SVG_METADATA)
    plt.close(fig)
    print("baseflow_hold.svg: normal depth %.4f m off the banks at most; not held, up to %.2f m over them; held, "
          "%.4f m at most" % (np.abs(normal - banks).max(), (linked[False] - banks).max(),
                              (linked[True] - banks).max()))


class Trapezoid:
    """The channel of the slope-change figures, reach_section's 20 m wide at the bed, worked out exactly while the
    water is below its banks, carrying 100 m³/s."""
    bottom, side, n, flow = 20.0, 2.0, 0.035, 100.0

    def area(self, y: float) -> float:
        return (self.bottom + self.side * y) * y

    def friction(self, y: float) -> float:
        """Manning's friction slope at the depth y."""
        perimeter = self.bottom + 2.0 * y * math.sqrt(1.0 + self.side ** 2)
        conveyance = self.area(y) * (self.area(y) / perimeter) ** (2 / 3) / self.n
        return (self.flow / conveyance) ** 2

    def froude_squared(self, y: float) -> float:
        return self.flow ** 2 * (self.bottom + 2.0 * self.side * y) / (G * self.area(y) ** 3)

    def force(self, y: float) -> float:
        """The specific force, the flow's momentum and the water's pressure per unit weight: the same either side of
        a hydraulic jump."""
        return self.flow ** 2 / (G * self.area(y)) + self.bottom * y ** 2 / 2.0 + self.side * y ** 3 / 3.0

    def normal(self, slope: float) -> float:
        return brentq(lambda y: self.friction(y) - slope, 1e-3, 30.0, xtol=1e-12)

    def critical(self) -> float:
        return brentq(lambda y: self.froude_squared(y) - 1.0, 1e-3, 30.0, xtol=1e-12)

    def sequent(self, y: float) -> float:
        """The subcritical depth a jump from the supercritical depth y rises to."""
        return brentq(lambda d: self.force(d) - self.force(y), self.critical(), 30.0, xtol=1e-12)

    def profile(self, depths: np.ndarray, slope: float) -> np.ndarray:
        """The distance downstream from the first of the depths to each, along gradually varied flow on a bed of
        this slope: dx/dy = (1 - Fr^2) / (S0 - Sf)."""
        run = lambda y: (1.0 - self.froude_squared(y)) / (slope - self.friction(y))
        steps = [quad(run, a, b, epsabs=1e-10, epsrel=1e-10)[0] for a, b in zip(depths[:-1], depths[1:])]
        return np.cumsum([0.0] + steps)


TRAPEZOID = Trapezoid()
STEEP_SLOPE, MILD_SLOPE = 0.02, 0.0005


def slope_change_reach(slopes: tuple[float, float], lengths: tuple[float, float],
                       spacing: float = 5.0) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Cross sections `spacing` metres apart along a reach of the trapezoid, its bed slope slopes[0] for its first
    lengths[0] metres and slopes[1] for the next lengths[1]: their stations and beds, each one's water surface on its
    own (normal depth), and the standard step's, marched from normal depth at the outlet as Standard_Step marches it."""
    stations = np.arange(0.0, sum(lengths) + spacing / 2, spacing)
    upper = stations < lengths[0]
    beds = 100.0 - np.where(upper, slopes[0] * stations, slopes[0] * lengths[0] + slopes[1] * (stations - lengths[0]))
    sections = []
    for bed in beds:
        # Points 2 m apart, so that the trapezoid's corners are on them and its cross sections are it exactly
        xs = reach_section(bed, TRAPEZOID.bottom, spacing=2.0)
        xs.left_bank_distance = xs.right_bank_distance = TRAPEZOID.bottom / 2 + 8.0
        sections.append(xs)

    flow = TRAPEZOID.flow
    normal = np.array([wse_for_discharge(xs, flow, slope)
                       for xs, slope in zip(sections, np.where(upper, slopes[0], slopes[1]))])
    tables = [HydraulicTable(xs, float(bed) + 3.5) for xs, bed in zip(sections, beds)]
    flows = np.full((stations.size, 1), flow)
    wse, *_ = _march(*_packed(tables), *tables[-1].parameters, flows, normal[-1:], np.zeros(flows.shape, dtype=bool),
                     beds, np.diff(stations), _eddy_ends(stations, EDDY_LENGTH), 1.0)
    return stations, beds, normal, wse[:, 0]


def slope_change_axes(stations: np.ndarray, beds: np.ndarray, normal: np.ndarray, linked: np.ndarray,
                      exact_x: np.ndarray, exact: np.ndarray, steep: tuple[float, float], scale: float, title: str):
    """The water surfaces along a slope-change reach, its distances in metres over scale: the steep reach shaded, the
    bed and critical depth over it, each cross section's own normal depth, the standard step's and the exact water
    surface (exact, a depth at each of exact_x)."""
    x = stations / scale
    fig, ax = plt.subplots(figsize=(9.0, 3.8))
    ax.axvspan(steep[0] / scale, steep[1] / scale, color=GRID, lw=0)
    ax.plot(x, beds, color=INK, lw=1.5, label="bed")
    ax.plot(x, beds + TRAPEZOID.critical(), color=SECONDARY, lw=1.2, ls=(0, (1, 2)), label="critical depth")
    ax.plot(x, normal, color=ORANGE, lw=2, label="each cross section on its own (normal depth)")
    ax.plot(x, linked, color=BLUE, lw=2, label="standard step (Standard_Step)")
    ax.plot(exact_x / scale, np.interp(exact_x, stations, beds) + exact, color=INK, lw=1.2, ls=(0, (4, 2)),
            label="exact")
    # The steep reach named at the top of its band, the mild one under its bed
    ax.text((steep[0] + steep[1]) / 2 / scale, 0.97, f"steep, {STEEP_SLOPE:.0%}", transform=ax.get_xaxis_transform(),
            ha="center", va="top", color=SECONDARY, fontsize=8.5)
    middle = (stations[-1] + steep[1]) / 2 if steep[0] == stations[0] else (stations[0] + steep[0]) / 2
    ax.text(middle / scale, np.interp(middle, stations, beds) - 0.3, f"mild, {MILD_SLOPE:.2%}", ha="center", va="top",
            color=SECONDARY, fontsize=8.5)
    ax.set_title(title, loc="left", color=INK, fontsize=10)
    ax.set_xlabel("Distance downstream (%s)" % ("km" if scale == 1000.0 else "m"))
    ax.set_ylabel("Elevation (m)")
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(beds.min() - 0.9, max(normal.max(), linked.max()) + 1.3)
    return fig, ax


def steep_to_mild_figure() -> None:
    """A steep reach running into a mild one: each cross section's own normal depth, the standard step's water
    surface, and the exact one: the steep reach's supercritical normal depth, a hydraulic jump where the specific
    forces either side are the same, and the backwater (S1) from the mild reach's normal depth."""
    steep_length, mild_length = 150.0, 250.0
    stations, beds, normal, linked = slope_change_reach((STEEP_SLOPE, MILD_SLOPE), (steep_length, mild_length))
    supercritical, mild = TRAPEZOID.normal(STEEP_SLOPE), TRAPEZOID.normal(MILD_SLOPE)
    sequent = TRAPEZOID.sequent(supercritical)
    depths = np.linspace(mild, sequent, 400)
    backwater = steep_length + TRAPEZOID.profile(depths, STEEP_SLOPE)
    jump = backwater[-1]
    exact_x = np.concatenate([[0.0, jump], backwater[::-1], [stations[-1]]])
    exact = np.concatenate([[supercritical, supercritical], depths[::-1], [mild]])

    fig, ax = slope_change_axes(stations, beds, normal, linked, exact_x, exact, (0.0, steep_length), 1.0,
                                f"The water surface along a steep reach running into a mild one, at "
                                f"{TRAPEZOID.flow:.0f} m³/s, flowing left to right")
    jump_bed = float(np.interp(jump, stations, beds))
    ax.annotate("hydraulic jump", (jump, jump_bed + (supercritical + sequent) / 2), (jump + 12.0, jump_bed + 2.5),
                color=INK, fontsize=8.5, va="bottom", arrowprops={"arrowstyle": "-", "color": SECONDARY, "lw": 0.8})
    ax.legend(loc="upper right", frameon=False, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(IMAGES / "steep_to_mild.svg", metadata=SVG_METADATA)
    plt.close(fig)
    depth = linked - beds
    rises = stations[np.flatnonzero(depth > TRAPEZOID.critical() + 0.01)[0]]
    print("steep_to_mild.svg: critical %.3f m, steep %.3f m, mild %.3f m, sequent %.3f m; exact jump %.0f m above "
          "the break, the step's from %.0f m; the step's backwater off the exact by %.3f m at most; on its own, %.2f m "
          "too low above the break"
          % (TRAPEZOID.critical(), supercritical, mild, sequent, steep_length - jump, steep_length - rises + 5.0,
             np.abs(depth - np.interp(stations, exact_x[2:], exact[2:]))[stations >= jump].max(),
             (linked - normal)[stations < steep_length][-1]))


def mild_to_steep_figure() -> None:
    """A mild reach running onto a steep one: each cross section's own normal depth, the standard step's water
    surface, and the exact one, through critical depth at the break: drawn down to it on the mild reach (M2), and on
    down to the steep reach's supercritical normal depth below it (S2)."""
    mild_length, steep_length = 3000.0, 300.0
    stations, beds, normal, linked = slope_change_reach((MILD_SLOPE, STEEP_SLOPE), (mild_length, steep_length))
    critical, mild, supercritical = TRAPEZOID.critical(), TRAPEZOID.normal(MILD_SLOPE), TRAPEZOID.normal(STEEP_SLOPE)
    # Closer together where the depths near normal depth, which they only near
    up = mild - critical
    drawdown = np.concatenate([critical + up * np.linspace(0.0, 0.999, 400),
                               mild - up * np.geomspace(1e-3, 1e-5, 40)[1:]])
    drawdown_x = mild_length + TRAPEZOID.profile(drawdown, MILD_SLOPE)
    down = critical - supercritical
    fall = np.concatenate([critical - down * np.linspace(0.0, 0.999, 200),
                           supercritical + down * np.geomspace(1e-3, 1e-5, 40)[1:]])
    fall_x = mild_length + TRAPEZOID.profile(fall, STEEP_SLOPE)
    exact_x = np.linspace(0.0, stations[-1], 6401)
    exact = np.where(exact_x < mild_length, np.interp(exact_x, drawdown_x[::-1], drawdown[::-1], left=mild),
                     np.interp(exact_x, fall_x, fall, right=supercritical))

    fig, ax = slope_change_axes(stations, beds, normal, linked, exact_x, exact, (mild_length, stations[-1]), 1000.0,
                                f"The water surface along a mild reach running onto a steep one, at "
                                f"{TRAPEZOID.flow:.0f} m³/s, flowing left to right")
    ax.legend(loc="lower left", frameon=False, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(IMAGES / "mild_to_steep.svg", metadata=SVG_METADATA)
    plt.close(fig)
    upper = stations < mild_length
    error = linked - beds - np.interp(stations, exact_x, exact)
    near = upper & (stations >= mild_length - 200.0)
    print("mild_to_steep.svg: critical %.3f m, mild %.3f m, steep %.3f m; the drawdown 95%% of the way to normal depth "
          "%.0f m above the break, S2 95%% of the way down %.0f m below it; the step off the exact on the mild reach "
          "by %+.3f m at most in the 200 m above the break and %+.3f m beyond; on its own, %.2f m too high above the "
          "break" % (critical, mild, supercritical, mild_length - np.interp(critical + 0.95 * up, drawdown, drawdown_x),
                     np.interp(-(critical - 0.95 * down), -fall, fall_x) - mild_length, error[near].max(),
                     np.abs(error[upper & ~near]).max(), (normal - linked)[upper][-1]))

if __name__ == "__main__":
    IMAGES.mkdir(exist_ok=True)
    cross_section_figure()
    rating_curve_figure()
    directions_figure()
    standard_step_figure()
    baseflow_hold_figure()
    steep_to_mild_figure()
    mild_to_steep_figure()
