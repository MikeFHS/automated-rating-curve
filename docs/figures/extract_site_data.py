"""Extract the data make_site_figures.py draws from, into docs/figures/data, from the full inputs.

    python docs/figures/extract_site_data.py

Only needed to change what the figures show: the extracts are in the repository, and make_site_figures.py draws from
them alone. This needs the full inputs that nencarta made: three FIM sites' (FIM_SITES_NENCARTA, each site's
ARC_InputFiles/GEOGLOWS_ARC_Input_fabdem_Bathy.yaml) and their USGS flood flows (FIM_SITES, each site's
flow_files/stage=*.csv), and the Karnali's (KARNALI) and the N40W112 tile's (N40W112) Bathy input files.

- For the standard step's figure, each site's ARC inputs, as Sweetwater Creek's are in testing/data, clipped to the
  stream cells and the water of the USGS flood drawn, without and with the standard step, of the reach drawn and those
  below it, which set its water (MARGIN cells around), the DEM rounded to DEM_STEP; and its input file with ARC's keys
  alone. The flood's flow is the flow file's flood column. The clipped site is run again, and the reach drawn compared
  with the full site's: the clip moves its slopes a little, as its neighbours' stream cells off the clip aren't in
  their local averages.
- For the bank methods' figure, the four cross sections as the Bathy input files sample them to find their banks
  (without carving), with the water mask and the land cover at their ordinates, cut down to what the figure shows or
  as much more as keeps each method's banks.
"""
from __future__ import annotations

import dataclasses
import os
import shutil
from dataclasses import fields
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import yaml
from make_site_figures import (
    BANK_PANELS,
    DATA,
    PANELS,
    profile,
    section_from,
    site_configs,
)
from osgeo import gdal

from arc import pipeline
from arc.bathymetry import (
    banks_by_flat_water,
    banks_by_land_cover,
    banks_by_width_to_depth_ratio,
)
from arc.config import Configs
from arc.io.vector import Vector
from arc.outputs.rating_curves import WSE, Q
from arc.xsection.orientation import angle_offsets
from arc.xsection.stream_path import stream_cells_by_reach

gdal.UseExceptions()

FIM_SITES_NENCARTA = Path(os.environ.get("FIM_SITES_NENCARTA", r"C:\Users\lrr43\Documents\masters\fim_sites_nencarta"))
FIM_SITES = Path(os.environ.get("FIM_SITES", r"C:\Users\lrr43\Documents\masters\fim_sites"))
TILES = {
    "karnali": Path(os.environ.get("KARNALI", r"C:\Users\lrr43\Downloads\karnali_newest\ARC_InputFiles"
                                   r"\GEOGLOWS_ARC_Input_dem_80.94070_28.32125_81.38632_28.72232_buffered_Bathy.yaml")),
    "n40w112": Path(os.environ.get("N40W112", r"C:\Users\lrr43\Downloads\n40w112\ARC_InputFiles"
                                   r"\GEOGLOWS_ARC_Input_N40W112_FABDEM_V1-2_Bathy.yaml")),
}

# Each folder in docs/figures/data: the FIM site it's from, and the USGS stage whose flow is drawn
SITES = {
    "wabash": ("Wabash_River_at_New_Harmony_(2016)", "18"),
    "flatrock": ("Flatrock_River_at_Columbus_(2012)", "624_7"),
    "leaf": ("Leaf_River_at_Hattiesburg_(2013)", "3"),
}
MARGIN = 10  # cells kept beyond the stream cells and the flood's water
DEM_STEP = 1 / 256  # metres: the DEM is rounded to multiples of this (2 mm at most), which compress better
# The inputs that are files, each written to the clipped site as this
FILES = {"dem_file": "dem.tif", "stream_file": "streams.tif", "lu_raster_sameres": "land_cover.tif",
         "bathywatermask": "water_mask.tif", "strmshp_file": "streams.parquet", "lu_manning_n": "mannings_n.txt",
         "flow_file": "flows.csv"}
FLOAT = ["COMPRESS=DEFLATE", "PREDICTOR=3", "ZLEVEL=9", "TILED=YES"]
INTEGER = ["COMPRESS=DEFLATE", "PREDICTOR=2", "ZLEVEL=9"]


def arc_inputs(path: Path) -> dict:
    """An input file's keys that ARC reads, without the files: those shared with curve2flood are left out."""
    inputs = yaml.safe_load(path.read_text(encoding="utf-8"))
    files = {f.name for f in fields(Configs) if "PathLike" in str(f.type)}
    keys = {f.name for f in fields(Configs) if f.init} - files
    return {key: value for key, value in inputs.items() if key.lower() in keys}


def flood_water(configs: Configs, flood: pd.Series, reaches: set[int]) -> tuple[np.ndarray, np.ndarray]:
    """The stream cells of these reaches, and the cells at the ends of the water at each of their cross sections, out
    from its stream cell, at its reach's flood flow, without and with the standard step."""
    rows, cols = [], []
    for flag in (False, True):
        results = pipeline.run(dataclasses.replace(configs, standard_step=flag), quiet=True, write=False)
        on = np.isin(results.cells.reaches, list(reaches))
        rows.append(results.cells.rows[on])
        cols.append(results.cells.cols[on])
        for k in np.flatnonzero(on):
            section = results.sections[k]
            flow = flood.get(int(results.cells.comids[k]), np.nan)
            increments = results.curves.increments[k]
            ok = np.isfinite(increments[:, Q])
            if section is None or not section.usable or ok.sum() < 2 or not np.isfinite(flow):
                continue
            wet = section.xs.elevations <= np.interp(flow, increments[ok, Q], increments[ok, WSE])
            centre = wet.size // 2
            left = centre - np.argmin(np.append(wet[centre::-1], False))
            right = centre + np.argmin(np.append(wet[centre:], False))
            ends = [max(left, 0), min(right, wet.size - 1)]
            rows.append(section.rows[ends])
            cols.append(section.cols[ends])
    return np.concatenate(rows), np.concatenate(cols)


def clip(source: os.PathLike, target: Path, window: list[int], step: float | None = None) -> None:
    """A raster clipped to the window (its left column, top row, columns and rows), its values rounded to multiples
    of the step if one is given."""
    clipped = gdal.Translate("", str(source), srcWin=window, format="MEM")
    band = clipped.GetRasterBand(1)
    if step:
        values = band.ReadAsArray()
        band.WriteArray(np.where(values == band.GetNoDataValue(), values, np.round(values / step) * step))
    floating = band.DataType in (gdal.GDT_Float32, gdal.GDT_Float64)
    gdal.GetDriverByName("GTiff").CreateCopy(str(target), clipped, options=FLOAT if floating else INTEGER)


def clip_site(name: str, site: str, stage: str) -> None:
    """A FIM site's inputs, clipped to the stream cells and the flood's water of the reach drawn and those below it,
    into docs/figures/data/name."""
    path = FIM_SITES_NENCARTA / site / "ARC_InputFiles" / "GEOGLOWS_ARC_Input_fabdem_Bathy.yaml"
    original = yaml.safe_load(path.read_text(encoding="utf-8"))
    paths = {key.lower(): value for key, value in original.items() if key.lower() in FILES}
    configs = Configs.from_mapping({**arc_inputs(path), **paths})
    flood = pd.read_csv(FIM_SITES / site / "flow_files" / f"stage={stage}.csv").set_index("COMID")["cms"]
    reach = next(reach for folder_name, reach, _ in PANELS if folder_name == name)
    network = pipeline.stream_network(configs, pipeline.read_streamfile(configs))

    rows, cols = flood_water(configs, flood, {reach} | nx.descendants(network, reach))
    reference = gdal.Open(str(configs.dem_file))
    top, left = max(int(rows.min()) - MARGIN, 0), max(int(cols.min()) - MARGIN, 0)
    bottom = min(int(rows.max()) + MARGIN, reference.RasterYSize - 1)
    right = min(int(cols.max()) + MARGIN, reference.RasterXSize - 1)
    window = [left, top, right - left + 1, bottom - top + 1]

    folder = DATA / name
    folder.mkdir(parents=True, exist_ok=True)
    clip(configs.dem_file, folder / FILES["dem_file"], window, DEM_STEP)
    for key in ("stream_file", "lu_raster_sameres", "bathywatermask"):
        clip(getattr(configs, key), folder / FILES[key], window)

    # The stream lines and flows of the reaches on the clipped stream raster
    streams = gdal.Open(str(folder / FILES["stream_file"])).ReadAsArray()
    reaches = np.unique(streams[streams > 0]).astype(np.int64)
    lines = Vector(configs.strmshp_file).to_geopandas()
    columns = [configs.reach_id, configs.downstream_reach_id] + \
        ([configs.drainage_area_field] if configs.drainage_area_field else [])
    lines = lines[lines[configs.reach_id].isin(reaches)][columns + ["geometry"]]
    lines.to_parquet(folder / FILES["strmshp_file"], index=False)
    flows = pd.read_csv(configs.flow_file)
    flows = flows[flows[configs.flow_file_id].isin(reaches)]
    flows = flows[[configs.flow_file_id, configs.flow_file_bf, configs.flow_file_qmax]].assign(
        flood=flows[configs.flow_file_id].map(flood).round(3).to_numpy())
    flows.to_csv(folder / FILES["flow_file"], index=False)
    shutil.copyfile(configs.lu_manning_n, folder / FILES["lu_manning_n"])

    inputs = {key: FILES[key.lower()] for key in original if key.lower() in FILES} | arc_inputs(path)
    with open(folder / "inputs.yaml", "w", encoding="utf-8") as file:
        file.write(f"# {site.replace('_', ' ')}: ARC's inputs, clipped to the stream cells and the water of the USGS\n"
                   f"# flood drawn (see docs/figures/data/README.md). The paths are relative to this folder.\n")
        yaml.safe_dump(inputs, file, sort_keys=False)

    # The reach drawn, clipped against in full, at the cross sections both rate
    full, clipped = profile(configs, reach, flood), profile(site_configs(name), reach)
    at = {(row - top, col - left): k for k, (row, col) in enumerate(zip(full["rows"], full["cols"]))}
    i, j = np.array([(at[cell], k) for k, cell in enumerate(zip(clipped["rows"], clipped["cols"])) if cell in at]).T
    print(f"{name}: {window[2]} x {window[3]} cells of {reference.RasterXSize} x {reference.RasterYSize}, "
          f"{len(reaches)} reaches; {len(clipped['rows'])} cross sections drawn, {len(full['rows'])} in full; "
          "clipped - full at the flood, median and largest: " + ", ".join(
              f"{key} {np.nanmedian(np.abs(d)):.3f}, {np.nanmax(np.abs(d)):.3f} m"
              for key in ("bed", "default", "linked") for d in [clipped[key][j] - full[key][i]]))


def tile_sections(config: Path, cells: list[tuple[int, int]]) -> pd.DataFrame:
    """These stream cells' cross sections as a tile's Bathy input file samples them to find their banks, out to the
    raster's edge, each with whether the water mask and the land cover have water at its ordinates (the stream cells
    water in both), NaN off the raster."""
    configs = Configs.from_file(config)
    grid = pipeline.read_grid(configs)
    stream = pipeline.stream_cells(grid, pipeline.read_flows(configs), None)
    water = {"water_mask": grid.water_mask.astype(float),
             "land_cover_water": (grid.land_cover == configs.lc_water_value).astype(float)}
    for raster in water.values():
        raster[stream.rows, stream.cols] = 1.0
    reaches = stream_cells_by_reach(cells=grid.stream_raster_cells())
    offsets = angle_offsets(configs.degree_manip, configs.degree_interval)
    out = []
    for row, col in cells:
        reach = int(stream.reaches[(stream.rows == row) & (stream.cols == col)][0])
        s = pipeline.sample_section(configs, grid, row, col, offsets, whole=True, reach_cells=reaches[reach])
        on = (s.rows >= 0) & (s.rows < grid.dem.shape[0]) & (s.cols >= 0) & (s.cols < grid.dem.shape[1])
        along = {name: np.where(on, raster[np.clip(s.rows, 0, raster.shape[0] - 1),
                                           np.clip(s.cols, 0, raster.shape[1] - 1)], np.nan)
                 for name, raster in water.items()}
        size = s.xs.elevations.size
        out.append(pd.DataFrame({"row": row, "col": col,
                                 "distance": (np.arange(size) - size // 2) * float(s.xs.ordinate_distance),
                                 "elevation": s.xs.elevations, **along}))
    return pd.concat(out)


def all_banks(ordinates: pd.DataFrame) -> list:
    """A cross section's banks by each of the methods the figure draws."""
    xs, along = section_from(ordinates)
    return [banks_by_land_cover(xs, along["mask"], 1.0), banks_by_land_cover(xs, along["land"], 1.0),
            banks_by_width_to_depth_ratio(xs), banks_by_flat_water(xs)]


def trimmed(ordinates: pd.DataFrame) -> pd.DataFrame:
    """A cross section cut down to as many ordinates either side of its stream cell as the figure shows, or as many
    more as keep each method's banks where they are on the whole of it."""
    centre = len(ordinates) // 2
    banks = repr(all_banks(ordinates))  # repr, so that banks a method doesn't find (NaN) count as the same
    reach = np.nanmax([[b.left, b.right] for b in all_banks(ordinates) if b.valid]) / \
        (ordinates["distance"].iloc[centre + 1] - ordinates["distance"].iloc[centre])
    half = int(1.2 * reach) + 3  # banks_figure shows 15% and two ordinates beyond the farthest bank
    while half < centre and repr(all_banks(ordinates.iloc[centre - half:centre + half + 1])) != banks:
        half += max(half // 4, 1)
    return ordinates.iloc[max(centre - half, 0):centre + half + 1]


def bank_sections() -> None:
    """The bank methods' figure's cross sections, into docs/figures/data/bank_sections.csv."""
    tables = []
    for tile, config in TILES.items():
        cells = [cell for name, cell, _ in BANK_PANELS if name == tile]
        for _, ordinates in tile_sections(config, cells).groupby(["row", "col"], sort=False):
            tables.append(trimmed(ordinates).assign(tile=tile))
    table = pd.concat(tables)[["tile", "row", "col", "distance", "elevation", "water_mask", "land_cover_water"]]
    table.to_csv(DATA / "bank_sections.csv", index=False, float_format="%.9g")  # float32 elevations exactly
    print(f"bank_sections.csv: {len(table)} ordinates")


if __name__ == "__main__":
    for name, (site, stage) in SITES.items():
        clip_site(name, site, stage)
    bank_sections()
