"""Draw the documentation's figures of Valley_Sections (docs/valley_sections.md), into docs/images, and print its table.

    python docs/figures/make_valley_figures.py

ARC runs on the synthetic meandering valleys of testing/synthetic_valley.py, Valley_Sections off and on, with normal
depth and with the standard step, against the 2D model there (LISFLOOD-FP's local inertial scheme), which this solves
again for each valley: it takes about a quarter of an hour. It needs matplotlib, like make_figures.py, whose style it
takes.
"""
from __future__ import annotations

import math
import sys
import tempfile
from pathlib import Path

import numpy as np

TESTING = Path(__file__).resolve().parents[2] / "testing"
sys.path.insert(0, str(TESTING))

import synthetic_valley
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
)
from test_valley_sections import errors, run_valley

from arc import pipeline
from arc.hydraulics import hydraulic_profile
from arc.xsection.valley import turn

SHOWN_FLOW = 400.0  # m^3/s, about a metre over the floodplain
OVERBANK = slice(2, None)  # the flows over the floodplain: 150, 400 and 1000 m^3/s
FIXED = (40, 80)  # Valley_Dir_Dist, in cells, with Valley_Dir_Widths 0
DEFAULT = "16 widths"  # Valley_Dir_Widths' default, the figures' window
CASES = ("gentle", "meander", "strong", "long", "natural", "natural_tight", "natural_wide")


def reference(valley: dict, flows: dict) -> dict:
    """The 2D model's water surfaces at the stream cells, as synthetic_valley.reference gives them."""
    rows, cols = valley["path"].T
    return {"path": valley["path"], "flows": np.array(synthetic_valley.FLOWS),
            "wse": np.stack([valley["dem"][rows, cols] + flows[q]["h"][rows, cols] for q in synthetic_valley.FLOWS], 1)}


def runs(folder: Path, valley: dict, fixed=FIXED) -> dict:
    """ARC's results with the standard step off and on: Valley_Sections off (None), on with fixed windows (in cells),
    and on by default (DEFAULT)."""
    out = {}
    for step in (False, True):
        out[step, None] = run_valley(folder / f"{step}_off", valley, valley_sections=False, standard_step=step)
        for window in fixed:
            out[step, window] = run_valley(folder / f"{step}_{window}", valley, valley_sections=True,
                                           standard_step=step, valley_dir_dist=window, valley_dir_widths=0.0)
        out[step, DEFAULT] = run_valley(folder / f"{step}_default", valley, valley_sections=True, standard_step=step)
    return out


def xy(valley: dict, rows, cols):
    """Metres down the valley and across it from its axis, at cells' centres."""
    dx = valley["dx"]
    return (np.asarray(cols) + 0.5) * dx, (np.asarray(rows) + 0.5) * dx - valley["y"].size * dx / 2


# --- The flow over the floodplain and the cross sections -------------------------------------------------------------


def kinked(valley: dict, section) -> tuple[np.ndarray, np.ndarray]:
    """A cell's cross section with Valley_Sections, in metres down and across the valley: the left line from its far
    end in to the left bank, across the channel on the cell's own cross section, and the right line out."""
    dx = valley["dx"]
    lines = section.xs.valley
    own, along = section.direction - math.pi / 2, lines.direction - math.pi / 2
    distances = np.arange(1, lines.elevations.size // 2 + 1) * lines.ordinate_distance
    banks = [(section.col + side * math.cos(own) * bank / dx, section.row + side * math.sin(own) * bank / dx)
             for side, bank in ((-1, section.xs.left_bank_distance), (1, section.xs.right_bank_distance))]
    cols = np.r_[(banks[0][0] - math.cos(along) * distances / dx)[::-1], banks[0][0], banks[1][0],
                 banks[1][0] + math.cos(along) * distances / dx]
    rows = np.r_[(banks[0][1] - math.sin(along) * distances / dx)[::-1], banks[0][1], banks[1][1],
                 banks[1][1] + math.sin(along) * distances / dx]
    return xy(valley, rows, cols)


def pick_cells(valley: dict, results) -> dict:
    """A cell where the channel crosses the valley most steeply and one at the top of a bend, in the middle meanders:
    their sections (Valley_Sections on)."""
    sections = [s for s in results.sections if s is not None and s.xs.valley is not None
                and synthetic_valley.middle(valley, xy(valley, s.row, s.col)[0])]
    crossing = max(sections, key=lambda s: turn(s.direction, 0.0))
    bend = min(sections, key=lambda s: turn(s.direction, 0.0) - 1e-6 * abs(xy(valley, s.row, s.col)[1]))
    return {"crossing": crossing, "bend": bend}


def flow_figure(valley: dict, flows: dict, results) -> dict:
    """The 2D model's water at SHOWN_FLOW, the way it moves over the floodplain, and two cells' cross sections."""
    flow = flows[SHOWN_FLOW]
    x, y = valley["x"], valley["y"]
    lo, hi = synthetic_valley.ENTRY + valley["wavelength"], synthetic_valley.ENTRY + 3 * valley["wavelength"]
    columns = (x >= lo - 100) & (x <= hi + 100)
    rows = np.abs(y) <= 520
    depth = np.where(flow["h"] > 0.01, flow["h"], np.nan)[np.ix_(rows, columns)]
    cells = pick_cells(valley, results)

    fig, ax = plt.subplots(figsize=(11.0, 5.4))
    image = ax.imshow(depth, cmap="Blues", vmin=0, vmax=3.5, origin="lower", interpolation="nearest",
                      extent=(x[columns][0] - 5, x[columns][-1] + 5, y[rows][0] - 5, y[rows][-1] + 5))
    ax.plot(valley["xs"], valley["ys"], color=BLUE, lw=1.0, alpha=0.9)
    step = 5  # cells between arrows
    yy, xx = np.meshgrid(y[rows][::step], x[columns][::step], indexing="ij")
    ux = flow["ux"][np.ix_(rows, columns)][::step, ::step]
    uy = flow["uy"][np.ix_(rows, columns)][::step, ::step]
    speed = np.hypot(ux, uy)
    off_channel = (valley["ground"] - valley["dem"])[np.ix_(rows, columns)][::step, ::step] < 0.05
    shown = off_channel & (speed > 0.02)
    ax.quiver(xx[shown], yy[shown], (ux / speed)[shown], (uy / speed)[shown], color=SECONDARY, scale=70, width=0.0016,
              headwidth=3.5, alpha=0.7)
    for name, section in cells.items():
        for (px, py), colour, style, label in (
                (xy(valley, section.rows, section.cols), INK, (0, (4, 2)), "the cell's own cross section"),
                (kinked(valley, section), ORANGE, "-",
                 "with Valley_Sections: the channel, and from each bank a line square to the valley")):
            inside = (np.abs(py) <= 520) & (px >= lo - 100) & (px <= hi + 100)
            ax.plot(px[inside], py[inside], color=colour, lw=1.6, ls=style, label=label if name == "crossing" else None)
        cx, cy = xy(valley, section.row, section.col)
        ax.plot(cx, cy, "o", color=INK, ms=4.5, zorder=5)
        ax.annotate("where the channel\ncrosses the valley" if name == "crossing" else "at a bend", (cx, cy),
                    (cx + 40, cy + (60 if cy < 0 else -60)), color=INK, fontsize=8.5,
                    va="bottom" if cy < 0 else "top")
    ax.set_xlim(lo - 100, hi + 100)
    ax.set_ylim(-520, 520)
    ax.set_aspect("equal")
    ax.grid(False)
    ax.set_xlabel("Down the valley (m)")
    ax.set_ylabel("Across the valley (m)")
    ax.set_title(f"The 2D model at {SHOWN_FLOW:.0f} m³/s: over the floodplain the water runs down the valley "
                 f"(arrows), not along the channel. Valley_Dir_Widths 16 (800 m).", loc="left", color=INK,
                 fontsize=9.5)
    bar = fig.colorbar(image, ax=ax, shrink=0.6, pad=0.01)
    bar.set_label("Water depth (m)", color=SECONDARY)
    bar.outline.set_edgecolor(MUTED)
    ax.legend(frameon=False, fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.11), ncol=2)
    fig.tight_layout()
    fig.savefig(IMAGES / "valley_flow.svg", metadata=SVG_METADATA)
    plt.close(fig)
    return cells


def section_figure(valley: dict, reference_wse: dict, plain, split, cells: dict) -> None:
    """The two cells: their own cross sections, with Valley_Sections, and the water at SHOWN_FLOW."""
    q = list(synthetic_valley.FLOWS).index(SHOWN_FLOW)
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 3.8), sharey=True)
    for ax, (name, section) in zip(axes, cells.items()):
        k = next(k for k, s in enumerate(split.sections) if s is section)
        own = plain.sections[k].xs
        valley_xs = section.xs.valley
        bed = float(own.elevations[own.elevations.size // 2])
        own_profile = hydraulic_profile(own)
        spliced = hydraulic_profile(section.xs)
        ax.plot(own_profile.stations, own_profile.elevations - bed, color=MUTED, lw=3.0, alpha=0.6,
                label="the cell's own cross section")
        ax.plot(spliced.stations, spliced.elevations - bed, color=ORANGE, lw=1.3,
                label="with Valley_Sections (kinked at the banks)")
        for k_run, colour, style, label in ((plain, SECONDARY, (0, (4, 2)), "ARC, Valley_Sections off"),
                                           (split, BLUE, "-", "ARC, Valley_Sections on")):
            increments = k_run.curves.increments[k]
            ok = np.isfinite(increments[:, 0]) & np.isfinite(increments[:, 3])
            wse = np.interp(SHOWN_FLOW, increments[ok, 0], increments[ok, 3])
            ax.axhline(wse - bed, color=colour, lw=1.2, ls=style, label=label)
        wse_2d = reference_wse[(section.row, section.col)][q]
        ax.axhline(wse_2d - bed, color=BLUE, lw=3.0, alpha=0.25, label="the 2D model")
        for bank in (-own.left_bank_distance, own.right_bank_distance):
            ax.axvline(bank, color=GRID, lw=1.0, zorder=0)
        ax.set_xlim(-700, 700)
        ax.set_ylim(-0.3, 4.2)
        ax.set_xlabel("Distance from the stream cell (m)")
        sinuosity, angle = valley_xs.sinuosity, math.degrees(turn(section.direction, valley_xs.direction))
        title = "Where the channel crosses the valley" if name == "crossing" else "At a bend"
        ax.set_title(f"{title}: {angle:.0f}° between the two, sinuosity {sinuosity:.2f}", loc="left", color=INK,
                     fontsize=9.5)
    axes[0].set_ylabel(f"Height above the bed (m), and the water at {SHOWN_FLOW:.0f} m³/s")
    axes[0].legend(frameon=False, fontsize=7.5, loc="lower left")
    fig.tight_layout()
    fig.savefig(IMAGES / "valley_section.svg", metadata=SVG_METADATA)
    plt.close(fig)


# --- The errors -----------------------------------------------------------------------------------------------------


def errors_figure(valley: dict, reference_ref: dict, results: dict) -> None:
    """ARC's water less the 2D model's along the middle meanders at SHOWN_FLOW, and its mean size at each flow."""
    q = list(synthetic_valley.FLOWS).index(SHOWN_FLOW)
    styles = {(False, None): (MUTED, "-", "normal depth, Valley_Sections off"),
              (False, DEFAULT): (BLUE, "-", "normal depth, Valley_Sections on"),
              (True, None): (SECONDARY, (0, (4, 2)), "standard step, Valley_Sections off"),
              (True, DEFAULT): (ORANGE, (0, (4, 2)), "standard step, Valley_Sections on")}
    fig, (along, by_flow) = plt.subplots(1, 2, figsize=(11.0, 3.8), gridspec_kw={"width_ratios": [1.7, 1]})
    for key, (colour, style, label) in styles.items():
        res = results[key]
        e = errors(res, valley, reference_ref)
        x = np.array([(res.cells.cols[k] + 0.5) * valley["dx"] for k in range(res.cells.count)
                      if synthetic_valley.middle(valley, (res.cells.cols[k] + 0.5) * valley["dx"])])
        order = np.argsort(x)
        along.plot(x[order], e[order, q], color=colour, ls=style, lw=1.3, label=label)
        by_flow.plot(np.arange(len(synthetic_valley.FLOWS)), np.nanmean(np.abs(e), axis=0), color=colour, ls=style,
                     lw=1.5, marker="o", ms=4)
    along.axhline(0.0, color=INK, lw=0.8)
    along.set_xlabel("Down the valley (m)")
    along.set_ylabel("ARC's water less the 2D model's (m)")
    along.set_title(f"At {SHOWN_FLOW:.0f} m³/s, along the middle two meanders", loc="left", color=INK, fontsize=9.5)
    along.legend(frameon=False, fontsize=7.5, ncol=2, loc="lower left")
    by_flow.set_xticks(range(len(synthetic_valley.FLOWS)), [f"{q:.0f}" for q in synthetic_valley.FLOWS])
    by_flow.axvspan(-0.3, 1.5, color=GRID, alpha=0.5, lw=0, zorder=0)
    by_flow.annotate("up to bankfull", (0.0, 0.0), xycoords=("data", "axes fraction"), xytext=(0, 4),
                     textcoords="offset points", color=SECONDARY, fontsize=8)
    by_flow.set_xlim(-0.3, len(synthetic_valley.FLOWS) - 0.7)
    by_flow.set_ylim(0, None)
    by_flow.set_xlabel("Flow (m³/s)")
    by_flow.set_ylabel("Mean size of the error (m)")
    by_flow.set_title("At each flow", loc="left", color=INK, fontsize=9.5)
    fig.tight_layout()
    fig.savefig(IMAGES / "valley_errors.svg", metadata=SVG_METADATA)
    plt.close(fig)


# --- The slope -------------------------------------------------------------------------------------------------------


def slope_case(folder: Path, valley: dict, reference_ref: dict, results: dict) -> dict:
    """The slopes along the middle meanders with Valley_Sections (by default, normal depth): each cell's channel
    slope, times the sinuosity, and the overbanks' (the channel's times the slope ratio); and the median errors at
    each flow, off, with the overbanks at the channel's own slope, and as built."""
    split = results[False, DEFAULT]
    ks = [k for k, s in enumerate(split.sections) if s is not None and s.xs.valley is not None
          and synthetic_valley.middle(valley, xy(valley, s.row, s.col)[0])]
    built = pipeline.slope_ratio
    pipeline.slope_ratio = lambda channel_slope, mean_slope, sinuosity: 1.0  # the overbanks at the channel's slope
    try:
        at_channel_slope = run_valley(folder / "channel_slope", valley, valley_sections=True, standard_step=False)
    finally:
        pipeline.slope_ratio = built
    return {"x": np.array([xy(valley, split.sections[k].row, split.sections[k].col)[0] for k in ks]),
            "slope": np.array([split.slopes[k] for k in ks]),
            "sinuosity": np.array([split.sections[k].xs.valley.sinuosity for k in ks]),
            "ratio": np.array([split.sections[k].xs.valley.slope_ratio for k in ks]),
            "errors": {label: np.median(errors(res, valley, reference_ref), axis=0) for label, res in (
                ("Valley_Sections off", results[False, None]),
                ("the overbanks at the channel's slope", at_channel_slope), ("Valley_Sections", split))}}


def slope_figure(cases: dict) -> None:
    """The slope rule on two of the synthetic valleys: along the middle meanders, and its effect on the water."""
    fig, axes = plt.subplots(2, len(cases), figsize=(11.0, 6.6))
    for col, (name, data) in enumerate(cases.items()):
        case = synthetic_valley.CASES[name]
        order = np.argsort(data["x"])
        x, slope = data["x"][order], data["slope"][order] * 1000
        ax = axes[0, col]
        ax.axhline(1.0, color=BLUE, lw=3.0, alpha=0.3, label="the valley's slope")
        ax.plot(x, slope, color=MUTED, lw=1.2, label="each cell's channel slope, S")
        ax.plot(x, slope * data["sinuosity"][order], color=SECONDARY, lw=1.0, ls=(0, (4, 2)),
                label="S × σ, each cell's own S")
        ax.plot(x, slope * data["ratio"][order], color=ORANGE, lw=1.6, label="S̄ × σ, the averaged S: the overbanks'")
        ax.set_ylim(0, 1.5)
        ax.set_xlabel("Down the valley (m)")
        ax.set_ylabel("Slope (m/km)")
        ax.set_title(f"{case['amplitude']:.0f} m either side every {case['wavelength']:.0f} m", loc="left", color=INK,
                     fontsize=9.5)
        ax = axes[1, col]
        for (label, median), colour in zip(data["errors"].items(), (MUTED, SECONDARY, ORANGE)):
            ax.plot(range(len(synthetic_valley.FLOWS)), median, color=colour, marker="o", ms=4, lw=1.4, label=label)
        ax.axhline(0.0, color=INK, lw=0.8)
        ax.set_xticks(range(len(synthetic_valley.FLOWS)), [f"{q:.0f}" for q in synthetic_valley.FLOWS])
        ax.set_xlabel("Flow (m³/s)")
        ax.set_ylabel("ARC's water less the 2D model's (median, m)")
    axes[0, 0].legend(frameon=False, fontsize=7.5, loc="lower left")
    axes[1, 0].legend(frameon=False, fontsize=7.5, loc="lower left")
    fig.tight_layout()
    fig.savefig(IMAGES / "valley_slope.svg", metadata=SVG_METADATA)
    plt.close(fig)
    for name, data in cases.items():
        own, overbank = data["slope"] * data["sinuosity"] * 1000, data["slope"] * data["ratio"] * 1000
        print(f"valley_slope.svg {name}: S x sinuosity p10-p90 {np.percentile(own, 10):.3f}-"
              f"{np.percentile(own, 90):.3f}, averaged {np.percentile(overbank, 10):.3f}-"
              f"{np.percentile(overbank, 90):.3f} (median {np.median(overbank):.3f}) m/km; median errors "
              f"{ {k: np.round(v, 3).tolist() for k, v in data['errors'].items()} }")


def table_rows(name: str, valley: dict, reference_ref: dict, results: dict) -> list[str]:
    """The docs' table's row: the mean size of the error over the floodplain, each way, and ARC's channel width (the
    median bank-to-bank width of the cross sections)."""
    def mean(key):
        return np.nanmean(np.abs(errors(results[key], valley, reference_ref)[:, OVERBANK]))

    case = synthetic_valley.CASES[name]
    rows, cols = valley["path"].T
    x = (cols + 0.5) * valley["dx"]
    middle = synthetic_valley.middle(valley, x)
    from arc.xsection.stream_path import along_stream_stations
    stations = along_stream_stations(rows, cols, valley["dx"], valley["dx"])
    ends = np.flatnonzero(middle)[[0, -1]]
    sinuosity = abs(stations[ends[1]] - stations[ends[0]]) / (x[ends[1]] - x[ends[0]])
    width = float(np.median([s.xs.left_bank_distance + s.xs.right_bank_distance for s in results[False, None].sections
                             if s is not None and s.xs.left_bank_distance > 0 and s.xs.right_bank_distance > 0]))
    keys = (None, *FIXED, DEFAULT)
    return [f"| {case['amplitude']:.0f} m every {case['wavelength']:.0f} m ({sinuosity:.2f}) | {width:.0f}, "
            f"{case['wavelength'] / width:.0f} | " + " | ".join(f"{mean((False, key)):.3f}" for key in keys) + " | "
            + " | ".join(f"{mean((True, key)):.3f}" for key in keys) + " |"]


if __name__ == "__main__":
    IMAGES.mkdir(exist_ok=True)
    table = ["| Meanders (sinuosity) | Channel width (m), widths a meander | Normal depth: off | 40 cells | 80 cells | "
             "16 widths | Standard step: off | 40 cells | 80 cells | 16 widths |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    slopes = {}
    with tempfile.TemporaryDirectory() as temporary:
        folder = Path(temporary)
        for name in CASES:
            valley = synthetic_valley.meander(name)
            flows = synthetic_valley.solve_flows(valley)
            ref = reference(valley, flows)
            results = runs(folder / name, valley)
            if name == "meander":
                stored = synthetic_valley.reference()
                print("2D water surfaces against the stored reference: largest difference",
                      round(float(np.abs(stored["wse"] - ref["wse"]).max()), 4), "m")
                cells = flow_figure(valley, flows, results[False, DEFAULT])
                by_cell = {tuple(c): w for c, w in zip(ref["path"], ref["wse"])}
                section_figure(valley, by_cell, results[False, None], results[False, DEFAULT], cells)
                errors_figure(valley, ref, results)
            if name in ("meander", "strong"):
                slopes[name] = slope_case(folder / name, valley, ref, results)
            table += table_rows(name, valley, ref, results)
            print(name, "done", flush=True)
    slope_figure(slopes)
    print("\n".join(table))
