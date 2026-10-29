"""Draw the documentation's figures of the standard step and of the bank methods on real rivers, into docs/images.

    python docs/figures/make_site_figures.py

The rivers are in docs/figures/data (see its README.md), extracted from their full inputs by extract_site_data.py. For
the standard step, it runs ARC on three FIM sites' inputs, clipped to the water of a flood the USGS mapped, with
Standard_Step off and on, and draws the water surface along one reach at that flood's flow. For the bank methods, it
finds the banks of cross sections of the Karnali and of the N40W112 tile where the methods find different ones.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from make_figures import (
    BLUE,
    IMAGES,
    INK,
    MUTED,
    ORANGE,
    SECONDARY,
    SVG_METADATA,
    plt,
)

from arc import pipeline
from arc.bathymetry.smoothing import order_reach
from arc.config import Configs
from arc.outputs.rating_curves import WSE, Q
from arc.xsection.xsection import XSection

DATA = Path(__file__).resolve().parent / "data"

# The tile, the cross section's stream cell (row, col), and what it shows
BANK_PANELS = [
    ("karnali", (1574, 934), "Karnali: the water mask stops on the channel's flat bed on the left"),
    ("karnali", (1458, 523), "Karnali: the land cover runs out over the floodplain on both sides"),
    ("karnali", (1548, 882), "Karnali: a braidplain 1.6 km wide, all water in the mask"),
    ("n40w112", (1046, 274), "Salt Lake City (N40W112): the mask stops inside the DEM's channel, the land cover past it"),
]

# The site's folder in DATA, the reach, and the panel's title, given the length of the reach drawn
PANELS = [
    ("wabash", 760686524, "Wabash River at New Harmony, Indiana: {km:.1f} km of the main stem"),
    ("flatrock", 760663139, "Flatrock River at Columbus, Indiana: {km:.1f} km of the main stem"),
    ("leaf", 750243718, "Hattiesburg, Mississippi: {km:.1f} km of the smaller of two rivers, up from where they meet"),
]


def site_configs(name: str) -> Configs:
    """A site's inputs in DATA, its paths joined to its folder."""
    folder = DATA / name
    inputs = yaml.safe_load((folder / "inputs.yaml").read_text(encoding="utf-8"))
    return Configs.from_mapping({key: str(folder / value) if isinstance(value, str) and (folder / value).is_file()
                                 else value for key, value in inputs.items()})


def profile(configs: Configs, reach: int, flood: pd.Series | None = None) -> dict:
    """The reach's bed and water surfaces at the flood's flow (by flow-file ID; the flow file's flood column if not
    given), without and with the standard step, from upstream to downstream."""
    runs = {flag: pipeline.run(dataclasses.replace(configs, standard_step=flag), quiet=True, write=False)
            for flag in (False, True)}
    results = runs[False]
    grid = pipeline.read_grid(configs)
    network = pipeline.stream_network(configs, pipeline.read_streamfile(configs))
    rated = [k for k, s in enumerate(results.sections)
             if s is not None and s.usable and np.isfinite(results.curves.increments[k, -1, Q])]
    by_reach: dict[int, list[int]] = {}
    for k in rated:
        by_reach.setdefault(int(results.cells.reaches[k]), []).append(k)
    cells = {r: (results.cells.rows[ks].astype(np.int64), results.cells.cols[ks].astype(np.int64))
             for r, ks in by_reach.items()}
    ks = np.array(by_reach[reach])
    beds = np.array([float(results.sections[k].xs.elevations[results.sections[k].xs.elevations.size // 2])
                     for k in ks])
    order, stations = order_reach(network, reach, *cells[reach], grid.dx, grid.dy, cells, beds)
    ks, beds = ks[order], beds[order]
    if flood is None:
        flood = pd.read_csv(configs.flow_file).set_index(configs.flow_file_id)["flood"]
    flow = float(flood[int(results.cells.comids[ks[0]])])

    def water(run) -> np.ndarray:
        out = []
        for k in ks:
            increments = run.curves.increments[k]
            ok = np.isfinite(increments[:, Q])
            q, w = increments[ok, Q], increments[ok, WSE]
            out.append(np.interp(flow, q, w) if q.size > 1 and q[0] <= flow <= q[-1] else np.nan)
        return np.array(out)

    return {"rows": results.cells.rows[ks], "cols": results.cells.cols[ks], "km": np.asarray(stations) / 1000.0,
            "bed": beds, "flow": flow, "default": water(runs[False]),
            "linked": water(runs[True])}


def main() -> None:
    fig, axes = plt.subplots(len(PANELS), 1, figsize=(9.0, 3.0 * len(PANELS)))
    for ax, (site, reach, title) in zip(axes, PANELS):
        p = profile(site_configs(site), reach)
        ax.plot(p["km"], p["bed"], color=MUTED, lw=1, label="the rating curves' bed (Elev)")
        ax.plot(p["km"], p["default"], color=ORANGE, lw=1.6, label="each cross section on its own (the default)")
        ax.plot(p["km"], p["linked"], color=BLUE, lw=2, label="standard step (Standard_Step)")
        ax.set_title(f"{title.format(km=p['km'][-1])}, at {p['flow']:,.0f} m³/s", loc="left", color=INK, fontsize=10)
        ax.set_xlim(0, p["km"][-1])
        top = np.nanmax([p["default"], p["linked"]])
        bottom = np.nanmin(p["bed"])
        ax.set_ylim(bottom - 0.05 * (top - bottom), top + 0.25 * (top - bottom))
        ax.set_ylabel("Elevation (m)")
        print(f"{site}: {p['km'][-1]:.1f} km, standard step - default median "
              f"{np.nanmedian(p['linked'] - p['default']):+.2f} m")
    axes[-1].set_xlabel("Distance downstream (km)")
    axes[0].legend(loc="upper right", frameon=False, fontsize=8.5, ncol=3)
    fig.tight_layout(h_pad=1.5)
    fig.savefig(IMAGES / "standard_step_sites.svg", metadata=SVG_METADATA)
    plt.close(fig)


def section_from(ordinates: pd.DataFrame) -> tuple[XSection, dict]:
    """A cross section from its rows of DATA/bank_sections.csv, and whether the water mask and the land cover have
    water at its ordinates (NaN off the raster)."""
    distance = ordinates["distance"].to_numpy()
    elevations = ordinates["elevation"].to_numpy(np.float64)
    xs = XSection(elevations, np.zeros_like(elevations), (distance[-1] - distance[0]) / (distance.size - 1))
    return xs, {"mask": ordinates["water_mask"].to_numpy(), "land": ordinates["land_cover_water"].to_numpy()}


def bank_sections() -> dict:
    """The cross sections in DATA/bank_sections.csv (section_from), by tile and stream cell."""
    return {(tile, (row, col)): section_from(ordinates)
            for (tile, row, col), ordinates in pd.read_csv(DATA / "bank_sections.csv").groupby(["tile", "row", "col"])}


def banks_figure() -> None:
    """Cross sections of real rivers where the water mask, the land cover, the width-to-depth ratio and flat water
    find different banks."""
    from arc.bathymetry import (
        banks_by_flat_water,
        banks_by_land_cover,
        banks_by_width_to_depth_ratio,
    )

    sections = bank_sections()
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 7.0))
    methods = [("mask", BLUE, "v", "water mask"), ("land", "#2a9d8f", "D", "land cover"),
               ("wd", ORANGE, "^", "width-to-depth ratio"), ("flat", SECONDARY, "o", "flat water")]
    for ax, (tile, cell, title) in zip(axes.ravel(), BANK_PANELS):
        xs, along = sections[tile, cell]
        found = {"mask": banks_by_land_cover(xs, along["mask"], 1.0),
                 "land": banks_by_land_cover(xs, along["land"], 1.0),
                 "wd": banks_by_width_to_depth_ratio(xs), "flat": banks_by_flat_water(xs)}
        spacing = float(xs.ordinate_distance)
        x = (np.arange(xs.elevations.size) - xs.elevations.size // 2) * spacing
        z = np.where(xs.elevations < 9000.0, xs.elevations, np.nan)
        valid = [b for b in found.values() if b.valid]
        reach = max(max(b.left for b in valid), max(b.right for b in valid))
        margin = 0.15 * reach + 2 * spacing
        lo, hi = -max(b.left for b in valid) - margin, max(b.right for b in valid) + margin
        keep = (x >= lo) & (x <= hi)
        low = float(np.nanmin(z[keep]))
        ax.plot(x[keep], z[keep], color=INK, lw=1.3, label="DEM")
        wet = keep & (along["mask"] == 1.0)
        ax.scatter(x[wet], np.full(wet.sum(), low - 0.6), marker="s", s=9, color=BLUE, alpha=0.5,
                   label="water in the mask")
        for (name, colour, marker, label), offset in zip(methods, (0.0, 0.25, 0.5, 0.75)):
            banks = found[name]
            if banks.valid:
                ax.scatter([-banks.left, banks.right], [banks.left_elevation, banks.right_elevation], s=46, color=colour,
                           marker=marker, zorder=6, edgecolor="white", linewidth=1.0, label=label)
        ax.set_title(title, loc="left", color=INK, fontsize=9)
        ax.set_xlim(lo, hi)
        ax.set_xlabel("Distance from the stream cell (m)")
        ax.set_ylabel("Elevation (m)")
        print(f"{title}: " + ", ".join(f"{name} {b.top_width:.0f} m" if b.valid else f"{name} none"
                                       for name, b in found.items()))
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=6, frameon=False, fontsize=8.5)
    fig.tight_layout(rect=(0, 0.05, 1, 1), h_pad=1.5)
    fig.savefig(IMAGES / "bank_methods_sites.svg", metadata=SVG_METADATA)
    plt.close(fig)


if __name__ == "__main__":
    main()
    banks_figure()
