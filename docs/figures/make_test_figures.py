"""Draw the documentation's figures of Bathy_Use_Banks, the benchmark tests, the standard step's eddy loss, the
stream's stations and the width-to-depth bank search, into docs/images.

    python docs/figures/make_test_figures.py

ARC runs on the synthetic rivers that the tests in testing/ build (test_bathymetry_modes.py, test_benchmarks.py,
test_dem_resilience.py and test_bank_detection.py), and the figures show what those tests check. It needs matplotlib,
like make_figures.py, whose style it takes.
"""
from __future__ import annotations

import math
import sys
import tempfile
from pathlib import Path

import numpy as np

TESTING = Path(__file__).resolve().parents[2] / "testing"
sys.path.insert(0, str(TESTING))

import test_bank_detection as detection
import test_bathymetry_modes as modes
import test_benchmarks as chow
import test_dem_resilience as resilience
from make_figures import (
    BLUE,
    GRID,
    IMAGES,
    INK,
    MUTED,
    ORANGE,
    SECONDARY,
    SVG_METADATA,
    plt,
    water_edges,
)

from arc import pipeline
from arc.bathymetry import banks_by_width_to_depth_ratio, set_bank_distances
from arc.bathymetry.banks import _held
from arc.config import Configs
from arc.hydraulics import hydraulic_profile, top_widths, wse_for_discharge
from arc.standard_step import HydraulicTable, step_up, velocity_head
from arc.xsection.stream_path import along_stream_distances, along_stream_stations

# --- Bathy_Use_Banks -------------------------------------------------------------------------------------------------


def river_case(folder: Path, case: str) -> dict:
    """One of test_bathymetry_modes' cases at the cross section 150 m down the river: its carved profile, the DEM
    and the true channel, the level the mode fills the channel to and the water surface of Flow_File_BF, all in
    metres above the true bed."""
    options, flow = modes.CASES[case]
    section, slope = next((s, slope) for s, slope in modes.run(folder, flow, **options) if s.col == 150)
    bed = modes.true_bed(section.col)
    profile = hydraulic_profile(section.xs)
    x = np.linspace(-40.0, 40.0, 801)
    true = modes.height(x)
    dem = np.where(np.abs(x) <= 10.5, np.maximum(true, modes.LOW_DEPTH), true)
    level = section.bank_elevation if options["use_banks"] else section.dem_low_point
    return {"flow": flow, "use_banks": options["use_banks"], "mask": options.get("mask", False), "x": x, "true": true,
            "dem": dem, "stations": profile.stations, "carved": profile.elevations - bed, "level": level - bed,
            "wse": wse_for_discharge(section.xs, flow, slope) - bed, "banks": section.hydraulic_banks,
            "profile": profile._replace(elevations=profile.elevations - bed)}


def draw_river_case(ax, data: dict, title: str) -> None:
    x, stations, carved = data["x"], data["stations"], data["carved"]
    shown = np.abs(stations) <= 40.0  # not the cross section's ends, which are walls
    left, right = water_edges(data["profile"], data["wse"])
    water_x = np.linspace(-left, right, 300)
    ax.fill_between(water_x, np.interp(water_x, stations, carved), data["wse"], color=BLUE, alpha=0.14, lw=0)
    ax.plot(x, data["dem"], color=MUTED, lw=3.0, alpha=0.55, label="DEM: water flat at the low flow", zorder=2)
    ax.plot(stations[shown], carved[shown], color=INK, lw=1.5, label="cross section ARC carves", zorder=3)
    ax.plot(x, data["true"], color=ORANGE, lw=1.2, ls=(0, (3, 3)), label="true channel", zorder=4)
    ax.plot([-left, right], [data["wse"]] * 2, color=BLUE, lw=2, solid_capstyle="round", zorder=4,
            label="water at Flow_File_BF")
    banks = data["banks"]
    ax.scatter([-banks.left, banks.right], np.interp([-banks.left, banks.right], stations, carved), s=34,
               color=SECONDARY, zorder=5, edgecolor="white", linewidth=1.2, label="banks")
    ax.axhline(data["level"], color=BLUE, lw=0.8, ls=(0, (1, 2)), zorder=1)
    level = "the bank\nelevation" if data["use_banks"] else "the DEM's water\nsurface at the stream cell"
    ax.annotate(f"level: {level}", (-39.0, data["level"] - 0.08), color=BLUE, fontsize=8, ha="left", va="top")
    error = float(np.interp(0.0, stations, carved))
    bed = "bed: the true bed" if abs(error) < 0.005 else \
        f"bed {abs(error):.2f} m {'above' if error > 0 else 'below'}\nthe true bed"
    ax.annotate(bed, (-39.0, -2.7), color=INK, fontsize=8.5, ha="left", va="bottom")
    ax.set_title(title, loc="left", color=INK, fontsize=9.5)
    ax.set_xlim(-40, 40)
    ax.set_ylim(-2.8, 3.4)
    ax.set_xlabel("Distance from the stream cell (m)")


def river_figure(folder: Path, cases: list, name: str, width: float) -> None:
    fig, axes = plt.subplots(1, len(cases), figsize=(width, 4.2), sharey=True)
    for ax, (case, title) in zip(axes, cases):
        data = river_case(folder / case.replace(",", "").replace(" ", "_"), case)
        draw_river_case(ax, data, f"{title}: {data['flow']:.1f} m³/s")
    axes[0].set_ylabel("Height above the true bed (m)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(IMAGES / name, metadata=SVG_METADATA)
    plt.close(fig)
    print(name)


def bathy_use_banks_figures(folder: Path) -> None:
    river_figure(folder, [("water surface, low flow", "Bathy_Use_Banks: False, water mask\nFlow_File_BF the flow "
                                                      "when the DEM was made"),
                          ("water surface, low flow, banks from the DEM", "Bathy_Use_Banks: False, no mask\n"
                                                                          "the same flow"),
                          ("banks, bankfull flow", "Bathy_Use_Banks: True, no mask\nFlow_File_BF bankfull"),
                          ("banks from a water mask, bankfull flow", "Bathy_Use_Banks: True, water mask\n"
                                                                     "the same flow")],
                 "bathy_use_banks.svg", 17.5)
    river_figure(folder, [("water surface, bankfull flow", "False, with the bankfull flow"),
                          ("banks, low flow", "True, with the low flow")],
                 "bathy_use_banks_wrong_flow.svg", 9.2)


# --- Chow's channel --------------------------------------------------------------------------------------------------


def chow_figure(folder: Path) -> None:
    """The backwater behind the lake and the drawdown to the steep reach: ARC against the exact profiles, and the
    differences."""
    ft = chow.FT
    x_kernel, backwater_kernel = chow.march_up(lambda table: table.bed + chow.LAKE_DEPTH, eddy_loss=False)
    _, drawdown_kernel = chow.march_up(lambda table: table.critical(chow.FLOW), eddy_loss=False)

    lake = {**chow.write_chow_channel(folder / "lake", "lake"), "Standard_Step": True}
    x_dem, backwater_dem = chow.depths_at_the_maximum_flow(
        pipeline.run(Configs.from_mapping(lake), quiet=True, write=False), 1)
    steep = {**chow.write_chow_channel(folder / "steep", "steep"), "Standard_Step": True}
    _, drawdown_dem = chow.depths_at_the_maximum_flow(
        pipeline.run(Configs.from_mapping(steep), quiet=True, write=False), 1)
    _, drawdown_stepwise = chow.depths_at_the_maximum_flow(
        pipeline.run(Configs.from_mapping({**steep, "Standard_Step_Eddy_Length": 0}), quiet=True, write=False), 1)

    fig, axes = plt.subplots(2, 2, figsize=(10.0, 6.4), sharex=True, gridspec_kw={"height_ratios": [1.5, 1.0]})
    xf, xk = x_dem / ft, x_kernel / ft
    panels = [
        (0, "Backwater behind a lake 5.0 ft deep (M1)", chow.backwater(x_dem), chow.backwater(x_dem, 0.7),
         [(xk, backwater_kernel, chow.backwater(x_kernel), "standard step, RD-52's cross sections, no eddy loss"),
          (xf, backwater_dem, chow.backwater(x_dem, 0.7), "ARC on the DEM (Standard_Step)")]),
        (1, "Drawdown to critical depth above a steep reach (M2)", chow.drawdown(x_dem), chow.drawdown(x_dem, 0.1),
         [(xk, drawdown_kernel, chow.drawdown(x_kernel), "standard step, RD-52's cross sections, no eddy loss"),
          (xf, drawdown_stepwise, chow.drawdown(x_dem, 0.1), "ARC on the DEM, eddy loss step by step"),
          (xf, drawdown_dem, chow.drawdown(x_dem, 0.1), "ARC on the DEM (Standard_Step)")]),
    ]
    colours = [SECONDARY, BLUE, ORANGE]
    for column, title, exact, with_eddy, runs in panels:
        top, bottom = axes[0, column], axes[1, column]
        top.plot(xf, exact / ft, color=MUTED, lw=3.0, alpha=0.6, label="exact (gradually varied flow)")
        top.plot(xf, with_eddy / ft, color=MUTED, lw=1.2, ls=(0, (4, 3)), label="exact, with the eddy loss")
        top.axhline(chow.NORMAL_DEPTH / ft, color=GRID, lw=1.2, zorder=0)
        top.axhline(chow.CRITICAL_DEPTH / ft, color=GRID, lw=1.2, zorder=0)
        top.annotate("normal depth", (2950, chow.NORMAL_DEPTH / ft + 0.03), color=SECONDARY, fontsize=8, ha="right")
        top.annotate("critical depth", (2950, chow.CRITICAL_DEPTH / ft + 0.03), color=SECONDARY, fontsize=8,
                     ha="right")
        for (x, depths, reference, label), colour in zip(runs, colours):
            style = dict(marker="o", ms=2.5, lw=0) if x is xk else dict(lw=1.4)
            top.plot(x, depths / ft, color=colour, label=label, **style)
            bottom.plot(x, (depths - reference) / ft, color=colour, **style)
        bottom.axhline(0.0, color=MUTED, lw=0.8)
        top.set_title(title, loc="left", color=INK, fontsize=10)
        top.set_ylim(2.0, 5.2)
        bottom.set_xlabel("Distance upstream (ft)")
        top.legend(loc="upper right", frameon=False, fontsize=7.5)
        bottom.annotate("each against the exact profile with its own eddy loss", (0.98, 0.95),
                        xycoords="axes fraction", color=SECONDARY, fontsize=7.5, ha="right", va="top")
    axes[0, 0].set_ylabel("Depth (ft)")
    axes[1, 0].set_ylabel("ARC − exact (ft)")
    axes[1, 0].set_xlim(0, 3000)
    fig.tight_layout()
    fig.savefig(IMAGES / "benchmark_chow.svg", metadata=SVG_METADATA)
    plt.close(fig)
    print("benchmark_chow.svg: M1 on the DEM %.4f ft, M2 as run %.3f ft, step by step %.4f ft beyond 20 ft"
          % (np.abs(backwater_dem - chow.backwater(x_dem, 0.7)).max() / ft,
             np.abs(drawdown_dem - chow.drawdown(x_dem, 0.1)).max() / ft,
             np.abs(drawdown_stepwise - chow.drawdown(x_dem, 0.1))[x_dem >= 20 * ft].max() / ft))


# --- The eddy loss's window -----------------------------------------------------------------------------------------


def flickering_reach(folder: Path, amplitude: float, windows, seed: int = 0) -> dict:
    """A river on a DEM of 30 m cells, its channel 60 m wide at the bed on average but each column's width drawn at
    random within `amplitude` of that, sides 2:1 to banks 4 m high, floodplains rising 1%, falling 0.0005, at
    600 m^3/s. ARC samples its 400 cross sections, a cell apart, and the standard step marches up them from normal
    depth, with no eddy loss and with each window (step_up's eddy_head and eddy_steps, as arc.standard_step charges
    it). The water's mean height above the profile without the eddy loss, over all but the bottom kilometre, for each
    window."""
    from osgeo import osr
    from test_pipeline import write_inputs

    rng = np.random.default_rng(seed)
    spacing, slope, flow, half_rows, count = 30.0, 0.0005, 600.0, 60, 400
    rows, cols = np.mgrid[0:2 * half_rows + 1, 0:count + 40]
    half = 30.0 * (1.0 + amplitude * rng.uniform(-1.0, 1.0, cols.shape[1]))[cols]
    distance = np.abs(rows - half_rows) * spacing
    dem = (100.0 - slope * spacing * cols + np.clip((distance - half) / 2.0, 0.0, 4.0)
           + 0.01 * np.clip(distance - half - 8.0, 0.0, None))
    streams = np.zeros(dem.shape, dtype=np.int32)
    streams[half_rows, 20:20 + count] = 1
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(32616)
    reaches = {1: {"start": (half_rows, 20), "end": (half_rows, 19 + count), "downstream": -1, "baseflow": 0.0,
                   "q_max": flow}}
    inputs = write_inputs(folder, dem, streams, srs, (500000.0, spacing, 0.0, 3800000.0, 0.0, -spacing), reaches,
                          water_n=0.035, land_n=0.035)
    results = pipeline.run(Configs.from_mapping({**inputs, "Stream_Slope_Method": "reach_average"}), quiet=True,
                           write=False)
    sections = sorted((s for s in results.sections if s is not None), key=lambda s: s.col)  # downstream
    stations = spacing * np.arange(len(sections))
    count = len(sections)
    tables = []
    for s in sections:
        set_bank_distances(s.xs, s.hydraulic_banks)
        tables.append(HydraulicTable(s.xs, float(s.xs.elevations[s.xs.elevations.size // 2]) + 4.0))
    outlet = wse_for_discharge(sections[-1].xs, flow, slope)

    def march(window):
        wse = np.empty(count)
        heads = np.empty(count)
        wse[-1] = outlet
        heads[-1] = velocity_head(tables[-1], outlet, flow)
        for m in range(count - 2, -1, -1):
            if window is None:
                eddy_head, steps = heads[m + 1], 10 ** 12  # no eddy loss
            else:
                far = next((j for j in range(m + 1, count) if stations[j] - stations[m] >= window), count - 1)
                eddy_head, steps = heads[far], far - m
            wse[m] = step_up(tables[m + 1], wse[m + 1], flow, tables[m], flow, spacing, eddy_head=eddy_head,
                             eddy_steps=steps)[0]
            heads[m] = velocity_head(tables[m], wse[m], flow)
        return wse

    upstream = stations <= stations[-1] - 1000.0
    without = march(None)
    return {window: float(np.mean((march(window) - without)[upstream])) for window in windows}


def eddy_figure(folder: Path) -> None:
    """Standard_Step_Eddy_Length on Chow's drawdown to a steep reach, and on a reach whose width flickers."""
    ft = chow.FT
    steep = {**chow.write_chow_channel(folder / "steep", "steep"), "Standard_Step": True}
    windows = {0: SECONDARY, 300: BLUE, 1000: ORANGE}
    profiles = {}
    for window in windows:
        x, profiles[window] = chow.depths_at_the_maximum_flow(pipeline.run(Configs.from_mapping(
            {**steep, "Standard_Step_Eddy_Length": window}), quiet=True, write=False), 1)
    exact = chow.drawdown(x, 0.1)
    lengths = [0.0, 30.0, 100.0, 300.0, 1000.0, 3000.0]
    flicker = {amplitude: flickering_reach(folder / f"flicker_{amplitude}", amplitude, lengths)
               for amplitude in (0.2, 0.4)}

    fig, (drawdown_ax, flicker_ax) = plt.subplots(1, 2, figsize=(11.0, 3.9), gridspec_kw={"width_ratios": [1.4, 1]})
    drawdown_ax.plot(x / ft, exact / ft, color=MUTED, lw=3.0, alpha=0.6, label="exact, with the eddy loss")
    for window, colour in windows.items():
        label = "each step's own (0 m)" if window == 0 else f"over {window:,} m" + (" (the default)" if window == 300
                                                                                     else "")
        drawdown_ax.plot(x / ft, profiles[window] / ft, color=colour, lw=1.4, label=label)
    drawdown_ax.axhline(chow.NORMAL_DEPTH / ft, color=GRID, lw=1.2, zorder=0)
    drawdown_ax.annotate("normal depth", (2950, chow.NORMAL_DEPTH / ft + 0.01), color=SECONDARY, fontsize=8,
                         ha="right", va="bottom")
    drawdown_ax.set_xlim(0, 3000)
    drawdown_ax.set_ylim(2.1, 3.45)
    drawdown_ax.set_xlabel("Distance upstream of the break (ft)")
    drawdown_ax.set_ylabel("Depth (ft)")
    drawdown_ax.set_title("Chow's channel drawn down to critical depth: the eddy loss reckoned", loc="left",
                          color=INK, fontsize=9.5)
    drawdown_ax.legend(frameon=False, fontsize=8, loc="lower right")
    for (amplitude, rises), colour in zip(flicker.items(), [BLUE, ORANGE]):
        shown = [max(length, 10.0) for length in rises]
        flicker_ax.plot(shown, list(rises.values()), color=colour, marker="o", ms=4, lw=1.5,
                        label=f"widths within {amplitude:.0%} of the mean")
    flicker_ax.axvline(300.0, color=GRID, lw=1.2, zorder=0)
    flicker_ax.set_xscale("log")
    flicker_ax.set_xticks([10, 30, 100, 300, 1000, 3000], ["0", "30", "100", "300", "1,000", "3,000"])
    flicker_ax.set_xlabel("Standard_Step_Eddy_Length (m)")
    flicker_ax.set_ylabel("Water raised by the eddy loss (m)")
    flicker_ax.set_title("A river whose width flickers, on 30 m cells", loc="left", color=INK, fontsize=9.5)
    flicker_ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(IMAGES / "eddy_window.svg", metadata=SVG_METADATA)
    plt.close(fig)
    print("eddy_window.svg: drawdown max error", {w: round(float(np.abs(p - exact).max() / ft), 3)
                                                   for w, p in profiles.items()},
          "flicker", {a: {w: round(v, 3) for w, v in r.items()} for a, r in flicker.items()})


# --- The stream's stations ------------------------------------------------------------------------------------------


def line_cells(degrees: float, length: float):
    """A straight stream's cells (rows, cols), stepping along it a twentieth of a cell at a time from the origin."""
    angle = math.radians(degrees)
    cells = []
    for t in np.arange(0.0, length, 0.05):
        cell = (int(round(-t * math.sin(angle))), int(round(t * math.cos(angle))))
        if not cells or cells[-1] != cell:
            cells.append(cell)
    return np.array([r for r, _ in cells]), np.array([c for _, c in cells])


def stations_figure(folder: Path) -> None:
    """The stream raster's stair-step path and the smoothed one the stations follow, the length each gives a straight
    stream at every angle, and the slopes each method then finds."""
    rows, cols = line_cells(30.0, 24.0)
    angles = np.arange(0.0, 90.1, 2.5)
    path, smoothed = [], []
    for degrees in angles:
        r, c = line_cells(degrees, 300.0)
        true = math.hypot(r[-1] - r[0], c[-1] - c[0])
        path.append(along_stream_distances(r, c, 1.0, 1.0, 0).max() / true)
        smoothed.append(along_stream_stations(r, c, 1.0, 1.0).max() / true)
    slope_angles = [0.0, 10.0, 22.5, 30.0, 45.0]
    slopes = {method: [float(np.median(resilience.slopes(folder / f"{method}_{degrees}", 2.0, method, degrees=degrees)))
                       / resilience.SLOPE for degrees in slope_angles] for method in resilience.SLOPE_METHODS}

    fig, (cells_ax, length_ax, slope_ax) = plt.subplots(1, 3, figsize=(12.0, 3.8),
                                                        gridspec_kw={"width_ratios": [1.15, 1, 1]})
    for r, c in zip(rows, cols):
        cells_ax.add_patch(plt.Rectangle((c - 0.5, -r - 0.5), 1, 1, facecolor=GRID, edgecolor="white", lw=1))
    cells_ax.plot([0, 24 * math.cos(math.radians(30))], [0, 24 * math.sin(math.radians(30))], color=MUTED, lw=1,
                  ls=(0, (4, 3)), label="the stream")
    cells_ax.plot(cols, -rows, color=ORANGE, lw=1.2, marker="o", ms=2.5, label="the stream raster's path")
    order = np.argsort(along_stream_distances(rows, cols, 1.0, 1.0, 0))
    x, y = cols[order].astype(float), -rows[order].astype(float)
    mx, my = x.copy(), y.copy()
    for i in range(x.size):
        h = min(3, i, x.size - 1 - i)
        mx[i] = x[i] + np.clip(x[i - h:i + h + 1].mean() - x[i], -0.5, 0.5)
        my[i] = y[i] + np.clip(y[i - h:i + h + 1].mean() - y[i], -0.5, 0.5)
    cells_ax.plot(mx, my, color=BLUE, lw=1.6, label="the path the stations follow")
    cells_ax.set_aspect("equal")
    cells_ax.set_xlim(-1, 22)
    cells_ax.set_ylim(-1, 13)
    cells_ax.set_xticks([])
    cells_ax.set_yticks([])
    cells_ax.grid(False)
    cells_ax.set_title("A stream at 30° to the grid", loc="left", color=INK, fontsize=9.5)
    cells_ax.legend(frameon=False, fontsize=7.5, loc="upper left")
    length_ax.plot(angles, path, color=ORANGE, lw=1.6, label="along the stream raster's path (before)")
    length_ax.plot(angles, smoothed, color=BLUE, lw=1.6, label="the stations")
    length_ax.axhline(1.0, color=MUTED, lw=0.8)
    length_ax.set_xlabel("Angle of the stream to the grid (degrees)")
    length_ax.set_ylabel("Length found ÷ the stream's")
    length_ax.set_xticks([0, 15, 30, 45, 60, 75, 90])
    length_ax.set_title("A straight stream 300 cells long", loc="left", color=INK, fontsize=9.5)
    length_ax.legend(frameon=False, fontsize=7.5)
    styles = [(ORANGE, "-"), ("#c99a2e", (0, (3, 2))), (SECONDARY, "-"), (BLUE, "-"), (INK, (0, (1, 2)))]
    for (method, ratios), (colour, line) in zip(slopes.items(), styles):
        slope_ax.plot(slope_angles, ratios, color=colour, ls=line, marker="o", ms=4, lw=1.4, label=method)
    slope_ax.plot(angles[angles <= 45], 1.0 / np.array(path)[angles <= 45], color=MUTED, lw=1, ls=(0, (4, 3)),
                  label="along the raster's path (before)")
    slope_ax.axhline(1.0, color=MUTED, lw=0.8)
    slope_ax.set_ylim(0.9, 1.03)
    slope_ax.set_xlabel("Angle of the stream to the grid (degrees)")
    slope_ax.set_ylabel("Median slope ÷ the bed's")
    slope_ax.set_title("The slope methods on test_dem_resilience's channel", loc="left", color=INK, fontsize=9.5)
    slope_ax.legend(frameon=False, fontsize=7, loc="lower left")
    fig.tight_layout()
    fig.savefig(IMAGES / "stream_stations.svg", metadata=SVG_METADATA)
    plt.close(fig)
    print("stream_stations.svg: path length worst", round(max(path), 4), "stations worst",
          round(max(abs(s - 1) for s in smoothed), 4), "slopes",
          {m: [round(v, 3) for v in r] for m, r in slopes.items()})


# --- The width-to-depth search --------------------------------------------------------------------------------------


def every_stage_banks(elevations: np.ndarray, spacing: float):
    """The width-to-depth search as it was, trying every stage: the banks, and the stages and ratios it tried."""
    thalweg = float(elevations[elevations.size // 2])
    heights = np.unique([z - thalweg for z in elevations if 0.0 < z - thalweg <= 25.0])
    last, last_ratio, tried = None, np.inf, []
    for height in heights:
        left, right = top_widths(elevations, spacing, thalweg + height)
        ratio = round((round(left, 3) + round(right, 3)) / height, 3)
        tried.append((height, ratio))
        if ratio > last_ratio:
            return _held(elevations, spacing, *last), tried
        last, last_ratio = (left, right), ratio
    return (math.nan, math.nan), tried


def stepped_stages(elevations: np.ndarray, spacing: float):
    """The stages ARC's search tries, at least 0.1 m apart, and their ratios, up to where it stops."""
    thalweg = float(elevations[elevations.size // 2])
    heights = np.unique([z - thalweg for z in elevations if 0.0 < z - thalweg <= 25.0])
    tried, last_tried, last_ratio = [], -np.inf, np.inf
    for i, height in enumerate(heights):
        if height - last_tried < 0.1 - 1e-9 and i < heights.size - 1:
            continue
        last_tried = height
        left, right = top_widths(elevations, spacing, thalweg + height)
        ratio = round((round(left, 3) + round(right, 3)) / height, 3)
        tried.append((height, ratio))
        if ratio > last_ratio:
            break
        last_ratio = ratio
    return tried


def width_to_depth_figure(folder: Path) -> None:
    """A noisy cross section of test_bank_detection's flat-water river, the search's ratios on it as it was and as it
    is, and how often each finds the bank tops as the noise grows."""
    height, _, top = detection.flat_water_river(1.0)
    rng = np.random.default_rng(5)
    xs = detection.section(height, 1.0)
    noise = rng.normal(0.0, 0.02, xs.elevations.size)
    noise[xs.elevations.size // 2] = 0.0
    xs.elevations += noise
    distance = (np.arange(xs.elevations.size) - xs.elevations.size // 2) * 1.0
    (old_left, old_right), old_tried = every_stage_banks(xs.elevations, 1.0)
    new = banks_by_width_to_depth_ratio(xs)
    new_tried = stepped_stages(xs.elevations, 1.0)

    sigmas = [0.0, 0.01, 0.02, 0.05, 0.1]
    shares = {"as it was: every stage": [], "as it is: stages 0.1 m apart": []}
    for sigma in sigmas:
        old, now = [], []
        for _ in range(200):
            trial = detection.section(height, 1.0)
            n = rng.normal(0.0, sigma, trial.elevations.size)
            n[trial.elevations.size // 2] = 0.0
            trial.elevations += n
            (l, r), _ = every_stage_banks(trial.elevations, 1.0)
            old.append(l + r)
            b = banks_by_width_to_depth_ratio(trial)
            now.append(b.top_width if b.valid else math.nan)
        for name, widths in zip(shares, (old, now)):
            shares[name].append(float(np.mean(np.abs(np.array(widths) - 2 * top) <= 0.2 * 2 * top)))

    fig, (section_ax, ratio_ax, share_ax) = plt.subplots(1, 3, figsize=(12.0, 3.8))
    keep = np.abs(distance) <= 25.0
    section_ax.plot(distance[keep], xs.elevations[keep] - 100.0, color=INK, lw=1.2, label="cross section, 2 cm noise")
    section_ax.plot(distance[keep], height(np.abs(distance[keep])), color=MUTED, lw=1, ls=(0, (4, 3)),
                    label="without the noise")
    for (left, right), colour, label in (((old_left, old_right), ORANGE, "banks, every stage"),
                                         ((new.left, new.right), BLUE, "banks, stages 0.1 m apart")):
        if left >= 0 and right >= 0:
            section_ax.scatter([-left, right], np.interp([-left, right], distance, xs.elevations - 100.0), s=40,
                               color=colour, zorder=5, edgecolor="white", label=label)
    section_ax.set_xlabel("Distance from the stream cell (m)")
    section_ax.set_ylabel("Height above the bed (m)")
    section_ax.set_title("The DEM's water flat at 0.6 m, banks 2.4 m up", loc="left", color=INK, fontsize=9.5)
    section_ax.legend(frameon=False, fontsize=7.5, loc="upper center")
    ratio_ax.plot(*zip(*old_tried), color=ORANGE, marker="o", ms=3, lw=1, label="every stage: stops at the first rise")
    ratio_ax.plot(*zip(*new_tried), color=BLUE, marker="o", ms=3, lw=1.4, label="stages at least 0.1 m apart")
    ratio_ax.set_yscale("log")
    ratio_ax.set_xlabel("Stage above the stream cell (m)")
    ratio_ax.set_ylabel("Top width ÷ depth")
    ratio_ax.set_title("The ratio the search follows", loc="left", color=INK, fontsize=9.5)
    ratio_ax.legend(frameon=False, fontsize=7.5)
    for (name, values), colour in zip(shares.items(), [ORANGE, BLUE]):
        share_ax.plot(sigmas, values, color=colour, marker="o", ms=4, lw=1.5, label=name)
    share_ax.set_ylim(0, 1.05)
    share_ax.set_xlabel("Noise on the water and banks, standard deviation (m)")
    share_ax.set_ylabel("Banks within a fifth of the bank tops")
    share_ax.set_title("200 noisy cross sections at each", loc="left", color=INK, fontsize=9.5)
    share_ax.legend(frameon=False, fontsize=7.5, loc="lower left")
    fig.tight_layout()
    fig.savefig(IMAGES / "width_to_depth_noise.svg", metadata=SVG_METADATA)
    plt.close(fig)
    print("width_to_depth_noise.svg: example banks", (round(old_left + old_right, 1), round(new.top_width, 1)),
          "shares", {k: [round(v, 2) for v in vals] for k, vals in shares.items()})


# --- The DEM ---------------------------------------------------------------------------------------------------------


def dem_figure(folder: Path) -> None:
    """test_dem_resilience's channel: the water surface's error against the cell size, and against noise beside the
    stream, and the slope against noise along it."""
    cells = [8.0, 4.0, 2.0, 1.0]
    sizes = {"stream along the columns": [], "off the cells' centres": [], "at 30° to the grid": []}
    for cell in cells:
        for name, options in (("stream along the columns", {}), ("off the cells' centres", {"offset": 0.37}),
                              ("at 30° to the grid", {"degrees": 30.0})):
            errors = resilience.water_surface_errors(folder / f"{name[:6]}_{cell}", cell, **options)
            sizes[name].append(np.percentile(np.abs(errors), 90))

    noises = [0.0, 0.05, 0.1, 0.25, 0.5]
    beside = {"maximum flow (300 m³/s)": [], "smallest flow": []}
    for noise in noises:
        errors = resilience.water_surface_errors(folder / f"beside_{noise}", 2.0, noise=noise, noisy_stream=False)
        beside["maximum flow (300 m³/s)"].append(np.median(errors[:, -1]))
        beside["smallest flow"].append(np.median(errors[:, 0]))

    along_noises = [0.0, 0.01, 0.02, 0.05, 0.1]
    along = {method: [np.median(resilience.slopes(folder / f"along_{method}_{noise}", 2.0, method, noise=noise,
                                                  seed=1)) / resilience.SLOPE for noise in along_noises]
             for method in resilience.SLOPE_METHODS}

    fig, (size_ax, beside_ax, along_ax) = plt.subplots(1, 3, figsize=(11.5, 3.7))
    for (name, values), colour in zip(sizes.items(), [INK, SECONDARY, BLUE]):
        size_ax.plot(cells, values, color=colour, marker="o", ms=4, lw=1.5, label=name)
    size_ax.plot(cells, [0.05 * c for c in cells], color=MUTED, lw=1, ls=(0, (4, 3)), label="5% of a cell")
    size_ax.set_xscale("log", base=2)
    size_ax.set_yscale("log")
    size_ax.set_xticks(cells, [f"{c:g}" for c in cells])
    size_ax.set_xlabel("Cell size (m); the channel is 32 m wide")
    size_ax.set_ylabel("Water surface error, 9 cells in 10 (m)")
    size_ax.set_title("Cell size", loc="left", color=INK, fontsize=10)
    size_ax.legend(frameon=False, fontsize=7.5, loc="lower right")

    for (name, values), colour in zip(beside.items(), [BLUE, SECONDARY]):
        beside_ax.plot(noises, values, color=colour, marker="o", ms=4, lw=1.5, label=name)
    beside_ax.axhline(0.0, color=MUTED, lw=0.8)
    beside_ax.set_xlabel("Noise beside the stream, standard deviation (m)")
    beside_ax.set_ylabel("Median water surface error (m)")
    beside_ax.set_title("Noise across the stream (2 m cells)", loc="left", color=INK, fontsize=10)
    beside_ax.legend(frameon=False, fontsize=7.5, loc="upper left")

    # local_average_corrected's percentiles grow with the noise too, so it is local_average's line: dashed over it
    styles = [(ORANGE, "-"), ("#c99a2e", (0, (3, 2))), (SECONDARY, "-"), (BLUE, "-"), (INK, "-")]
    for (method, ratios), (colour, line) in zip(along.items(), styles):
        along_ax.plot(along_noises, ratios, color=colour, ls=line, marker="o", ms=4, lw=1.5, label=method)
    along_ax.axhline(1.0, color=MUTED, lw=0.8)
    along_ax.set_yscale("log")
    along_ax.set_xlabel("Noise along the stream, standard deviation (m)")
    along_ax.set_ylabel("Median slope ÷ the bed's")
    along_ax.set_title("Noise along the stream (2 m cells, slope 0.001)", loc="left", color=INK, fontsize=10)
    along_ax.legend(frameon=False, fontsize=7.5, loc="upper left")
    fig.tight_layout()
    fig.savefig(IMAGES / "benchmark_dem.svg", metadata=SVG_METADATA)
    plt.close(fig)
    print("benchmark_dem.svg: 90th percentile errors", {k: [round(v, 3) for v in vals] for k, vals in sizes.items()},
          "beside", {k: [round(v, 3) for v in vals] for k, vals in beside.items()},
          "slopes", {k: [round(v, 2) for v in vals] for k, vals in along.items()})


if __name__ == "__main__":
    IMAGES.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory() as temporary:
        folder = Path(temporary)
        bathy_use_banks_figures(folder / "rivers")
        chow_figure(folder / "chow")
        dem_figure(folder / "dem")
        eddy_figure(folder / "eddy")
        stations_figure(folder / "stations")
        width_to_depth_figure(folder / "width_to_depth")
