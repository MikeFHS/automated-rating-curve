"""Draw the documentation's figures of the standard step and of the bank methods on real rivers, into docs/images.

    python docs/figures/make_site_figures.py

Unlike make_figures.py, this needs data that isn't in the repository: three FIM sites' ARC inputs, as nencarta makes
them (FIM_SITES_NENCARTA, each site's ARC_InputFiles/GEOGLOWS_ARC_Input_fabdem_Bathy.yaml), and the USGS flood flows
(FIM_SITES, each site's flow_files/stage=*.csv). It runs ARC on each site with Standard_Step off and on, and draws the
water surface along one reach at one of the USGS flows. For the bank methods, it needs nencarta's inputs for the
Karnali (KARNALI) and the N40W112 tile (N40W112), and draws cross sections where the methods find different banks.
"""
from __future__ import annotations

import dataclasses
import os
from pathlib import Path

import numpy as np
import pandas as pd
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

FIM_SITES_NENCARTA = Path(os.environ.get("FIM_SITES_NENCARTA", r"C:\Users\lrr43\Documents\masters\fim_sites_nencarta"))
FIM_SITES = Path(os.environ.get("FIM_SITES", r"C:\Users\lrr43\Documents\masters\fim_sites"))
KARNALI = Path(os.environ.get("KARNALI", r"C:\Users\lrr43\Downloads\karnali_newest\ARC_InputFiles"
                              r"\GEOGLOWS_ARC_Input_dem_80.94070_28.32125_81.38632_28.72232_buffered_Bathy.yaml"))
N40W112 = Path(os.environ.get("N40W112", r"C:\Users\lrr43\Downloads\n40w112\ARC_InputFiles"
                              r"\GEOGLOWS_ARC_Input_N40W112_FABDEM_V1-2_Bathy.yaml"))

# The tile, the cross section's stream cell (row, col), and what it shows
BANK_PANELS = [
    (KARNALI, (857, 142), "Karnali: the water mask stops on the channel's flat bed on the left"),
    (KARNALI, (1169, 455), "Karnali: the mask stops short on the left, the land cover runs onto the floodplain"),
    (KARNALI, (1639, 1036), "Karnali: a braidplain 2.6 km wide, all water in the mask"),
    (N40W112, (2236, 328), "Salt Lake City (N40W112): the mask and the land cover either side of the DEM's channel"),
]

# The site, the reach, the USGS stage whose flow is drawn, and the panel's title
PANELS = [
    ("Wabash_River_at_New_Harmony_(2016)", 760686524, "18",
     "Wabash River at New Harmony, Indiana: 6.9 km of the main stem"),
    ("Flatrock_River_at_Columbus_(2012)", 760663139, "624_7",
     "Flatrock River at Columbus, Indiana: 5.5 km of the main stem"),
    ("Leaf_River_at_Hattiesburg_(2013)", 750243718, "3",
     "Hattiesburg, Mississippi: 4.2 km of the smaller of two rivers, up from where they meet"),
]


def profiles(site: str, reach: int, stage: str) -> dict:
    """The reach's bed and water surfaces at the stage's flow, without and with the standard step, from upstream to
    downstream."""
    configs = Configs.from_file(FIM_SITES_NENCARTA / site / "ARC_InputFiles" / "GEOGLOWS_ARC_Input_fabdem_Bathy.yaml")
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
    flows = pd.read_csv(FIM_SITES / site / "flow_files" / f"stage={stage}.csv").set_index("COMID")["cms"]
    flow = float(flows[int(results.cells.comids[ks[0]])])

    def water(run) -> np.ndarray:
        out = []
        for k in ks:
            increments = run.curves.increments[k]
            ok = np.isfinite(increments[:, Q])
            q, w = increments[ok, Q], increments[ok, WSE]
            out.append(np.interp(flow, q, w) if q.size > 1 and q[0] <= flow <= q[-1] else np.nan)
        return np.array(out)

    return {"km": np.asarray(stations) / 1000.0, "bed": beds, "flow": flow, "default": water(runs[False]),
            "linked": water(runs[True])}


def main() -> None:
    fig, axes = plt.subplots(len(PANELS), 1, figsize=(9.0, 3.0 * len(PANELS)))
    for ax, (site, reach, stage, title) in zip(axes, PANELS):
        p = profiles(site, reach, stage)
        ax.plot(p["km"], p["bed"], color=MUTED, lw=1, label="the rating curves' bed (Elev)")
        ax.plot(p["km"], p["default"], color=ORANGE, lw=1.6, label="each cross section on its own (the default)")
        ax.plot(p["km"], p["linked"], color=BLUE, lw=2, label="standard step (Standard_Step)")
        ax.set_title(f"{title}, at {p['flow']:,.0f} m³/s", loc="left", color=INK, fontsize=10)
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


def tile_sections(config: Path) -> dict:
    """A tile's cross sections as its Bathy input file samples them (without carving), by stream cell, each with the
    water mask and the land cover at its ordinates (the stream cells water in both)."""
    import dataclasses

    from osgeo import gdal

    previous = os.getcwd()
    os.chdir(config.parent.parent)  # the input file's relative paths are from the tile's folder
    try:
        configs = dataclasses.replace(Configs.from_file(config), bathy_out_file=None, standard_step=False,
                                      print_vdt_database=None, print_curve_file=None)
        results = pipeline.run(configs, quiet=True, write=False)
        mask = (np.nan_to_num(gdal.Open(str(configs.bathywatermask)).ReadAsArray()) != 0).astype(float)
        land = gdal.Open(str(configs.lu_raster_sameres)).ReadAsArray().astype(float)
    finally:
        os.chdir(previous)
    mask[results.cells.rows, results.cells.cols] = 1.0
    land[results.cells.rows, results.cells.cols] = configs.lc_water_value
    out = {}
    for s in results.sections:
        if s is not None:
            on = (s.rows >= 0) & (s.rows < mask.shape[0]) & (s.cols >= 0) & (s.cols < mask.shape[1])
            along = {}
            for name, raster in (("mask", mask), ("land", land)):
                along[name] = np.full(s.rows.size, np.nan)
                along[name][on] = raster[s.rows[on], s.cols[on]]
            out[(s.row, s.col)] = (s.xs, along, configs.lc_water_value)
    return out


def banks_figure() -> None:
    """Cross sections of real rivers where the water mask, the land cover, the width-to-depth ratio and flat water
    find different banks."""
    from arc.bathymetry import (
        banks_by_flat_water,
        banks_by_land_cover,
        banks_by_width_to_depth_ratio,
    )

    tiles = {}
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 7.0))
    methods = [("mask", BLUE, "v", "water mask"), ("land", "#2a9d8f", "D", "land cover"),
               ("wd", ORANGE, "^", "width-to-depth ratio"), ("flat", SECONDARY, "o", "flat water")]
    for ax, (config, cell, title) in zip(axes.ravel(), BANK_PANELS):
        if config not in tiles:
            tiles[config] = tile_sections(config)
        xs, along, water = tiles[config][cell]
        found = {"mask": banks_by_land_cover(xs, along["mask"], 1.0),
                 "land": banks_by_land_cover(xs, along["land"], water),
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
