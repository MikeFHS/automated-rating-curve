"""ARC's main workflow, from its input rasters to its rating curves, cross sections and bathymetry.

    from arc.config import Configs
    from arc.pipeline import run
    results = run(Configs.from_file("ARC_Input_File.yaml"))

or from the command line, ``arc ARC_Input_File.yaml`` (or ``arc ARC_Input_File.yaml``).

The steps
---------
1. Read Inputs: the flow file, and the DEM, stream and land cover rasters and the water mask (BathyWaterMask), which must
   share one grid (north-up or south-up, not rotated, in metres or degrees). The stream network (StrmShp_File), if
   given, serves the bathymetry, the slopes and the drainage-area power laws. DEM cells without data are walls.
2. Stream cells: the stream raster's cells whose ID is in the flow file (every stream cell for representative cross
   sections built without flows), or the manual cross sections' cells.
3. Slopes, by Stream_Slope_Method (arc.xsection.slope). The local averages near a reach's ends reach into its
   main-stem neighbours' cells, where there's a stream network. A reach the method can't give a slope takes its
   neighbours' in the stream network, and the cells of a reach left without one are skipped.
4. Roughness: the stream cells become water in the land cover (and in the water mask), and each land cover class
   takes its n from the Manning's n table.
5. Cross sections (arc.xsection): the stream's direction at each stream cell, the low spot, the direction where the
   channel is narrowest, and the cross section there, as far out as finding its banks looks (see below).
6. Banks (arc.bathymetry.find_banks, banks_at): with bathymetry, the DEM's water's edge without Bathy_Use_Banks and
   the bank tops, below the water that carries the maximum flow (bank_ceiling), with it; without bathymetry, the bank
   tops below that water too, or its edges where the width-to-depth ratio is still falling there. The hydraulics
   divide every cross section at its bank tops: without Bathy_Use_Banks, those found as without bathymetry, while the
   channel is carved between the water's edges (dividing_banks). The water mask (or the land cover's water, with
   FindBanksBasedOnLandCover) gives the water level, holds the DEM's water, and is the banks where neither finds any.
   Where nothing resolves a channel, a single cell as wide as a drainage-area width prior no wider than one ordinate
   spacing. Then each cross section is trimmed to what the later steps are expected to need (see below), and the
   water's n goes between the hydraulics' banks.
7. Bathymetry, if BATHY_Out_File is set (arc.bathymetry): each reach's banks and bank elevations smoothed, and the
   bank elevations ramped to meet at each junction on a main stem (unless Bathy_Smooth_Banks is off), each channel's depth found, the depths filled in and the beds smoothed along the network
   (unless Bathy_Smooth_Bed is off, as it is by default with Standard_Step or Bathy_Use_Banks), and the channel carved
   into each cross section and burned into the bathymetry raster, whose gaps are then filled. Each cross section's
   centre cell gets its own carved bed, the Elev of its rating curve, and other cells where cross sections overlap get
   their mean (see arc.bathymetry.raster). A channel whose banks nothing resolved is carved as wide as its width
   prior, where it has one, and otherwise one cell wide. The water's n goes between the hydraulics' banks: those the
   channel is carved to, which the width filter may have moved, or without Bathy_Use_Banks, its bank tops if wider. A bank from the DEM above the water that carries the cell's maximum
   flow is a valley wall, and isn't an observation of the bank elevation (bank_ceilings).
8. Rating curves (arc.rating_curve), linked along the stream network by the standard step with Standard_Step
   (arc.standard_step), and the representative cross sections (arc.outputs.representative) when they're asked for.
9. Outputs (arc.outputs), on the DEM's grid.

Representative cross sections and rating curves
-----------------------------------------------
A run asked for both makes them from the same cross sections, carved as they would be for the rating curves alone, so
that asking for the representative cross sections doesn't change the rating curves. With Standard_Step, each cross
section with a rating curve carries, in the representative cross sections, the flows its rating curve has at its water
surfaces, rather than normal depth's (see arc.outputs.representative). A run that only builds
representative cross sections (one that asks for no output of the rating curves) makes no rating curves, and needs a
flow file only to carve its channels for the baseflow (see arc.config).

Cross section lengths
---------------------
A cross section reaches as far as its water needs, which can
be out to the raster's edge, beyond which the ordinates are walls (OFF_RASTER_ELEVATION).
Each one is first sampled only as far as finding its banks looks: on each side out to the first ground more than
MAX_BANK_HEIGHT (25 m) above its centre, where the width-to-depth search's stages stop (ordinates_above), and with
water at its centre in the water mask or land cover, out to where that water ends. The angle search before it samples
each direction it tries only out to its own water's edge. The cross section is then fitted to the highest water surface
the later steps are expected to work at (fit_section, trim_level): cut to the first ordinates above it, or extended to
reach them. Any step that needs it longer extends it first, from the same rasters (Resampler). A step works on a cross
section at a water surface only once it rises above it on both sides, or reaches the raster's edge on both, which is
what makes the results the same as on cross sections sampled out to the raster's edge, bit for bit. The trim_level
estimate only decides how often a step has to extend a cross section, and how far the cross sections in the
cross-section file reach.
"""
from __future__ import annotations

import argparse
import ast
import json
import logging
import math
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import NamedTuple

import geopandas as gpd
import numpy as np
import pandas as pd
import tqdm
from numba import njit

from arc import LOG
from arc._log import logging_format
from arc.bathymetry import (
    Banks,
    Burns,
    ReachSections,
    SparseRaster,
    banks_for_width,
    bathymetry_depth,
    carve_channel,
    fill_sparse_gaps,
    find_banks,
    in_bank,
    ordinate_cells,
    reach_network,
    set_bank_distances,
    set_in_bank_roughness,
    single_cell_banks,
    smooth_bank_elevations,
    smooth_channel_depths,
)
from arc.bathymetry.banks import MAX_BANK_HEIGHT, _on_raster_ordinates
from arc.bathymetry.bed_smoothing import MAX_BED_GRADE, MEDIAN_WINDOW
from arc.bathymetry.depth import power_law_geometry
from arc.bathymetry.smoothing import WATER_BANKS, reach_widths
from arc.config import Configs
from arc.hydraulics import DepthRoughness, hydraulic_profile, wse_for_discharge
from arc.io import Raster, Vector
from arc.outputs import (
    CrossSectionRecord,
    RatingCurves,
    RepresentativeSample,
    representative_cross_section_dataframe,
    vdt_dataframe,
    write_ap,
    write_cross_sections,
    write_curve_file,
    write_reach_average_curve_file,
    write_representative_cross_sections,
    write_vdt,
)
from arc.outputs.decisions import SLOPE_SOURCES, qa_flags, summary, write_decision_log
from arc.outputs.rating_curves import COL, COMID, ELEV, ROW, SLOPE, XS_ANGLE
from arc.outputs.representative import highest_stage
from arc.outputs.tables import run_metadata
from arc.rating_curve import WSE, Q, RatingCurve, max_flow_wse, rating_curve
from arc.standard_step import StepSummary, apply_standard_step
from arc.xsection.low_spot import low_spot_cell
from arc.xsection.orientation import (
    angle_offsets,
    narrowest_direction,
    reach_direction,
    stream_direction,
)
from arc.xsection.sampling import (
    OFF_RASTER_ELEVATION,
    _ordinates,
    ordinates_above,
    sample_classes,
    sample_elevations,
)
from arc.xsection.slope import (
    corrected_local_slopes,
    end_point_slope,
    fill_unresolved_reach_slopes,
    local_average_slopes,
    local_fit_slopes,
    main_stem_above,
    reach_median_slope,
)
from arc.xsection.stream_path import (
    along_stream_stations,
    stream_cells_by_reach,
    stream_raster_cells,
)
from arc.xsection.subcell import reconstruct_all
from arc.xsection.valley import (
    Valley,
    fill_other_water,
    ray_classes,
    ray_reach,
    ray_values,
    slope_ratio,
    valley_axes,
    worth_splitting,
)
from arc.xsection.xsection import Profile, XSection

_REQUIRED_MANUAL_COLUMNS = ("Row", "Col", "Ordinate_Dist", "XS1_Profile", "XS2_Profile", "LC1_Profile",
                            "LC2_Profile", "XS1_Row", "XS1_Col", "XS2_Row", "XS2_Col")
_ALONG_A_ROW = 1e-12  # a cross section whose direction's sine is smaller than this runs along a row
_METADATA_BUT_BASEFLOW = [COMID, ROW, COL, ELEV, SLOPE, XS_ANGLE]
# How far above the water surface that carries a cell's maximum flow over the DEM's cross section to trim it, which
# only decides how often a step has to extend it (see trim_level): this much above it, and with the standard step,
# its table's top over a bed down to TRIM_CARVING below the stream cell
TRIM_MARGIN = 3.0  # metres
TRIM_CARVING = 8.0  # metres


# --- What a run works with ------------------------------------------------------------------------------------------


class Grid(NamedTuple):
    """The input rasters, on one grid."""
    reference: Raster  # the DEM, whose grid the outputs are written on
    # float32 where the DEM's values fit it exactly (a float32 or 8- or 16-bit integer DEM), and otherwise float64,
    # with cells without data as walls. Its elevations are always worked with as float64, as if it were (elevations,
    # and the sampling functions given like=ELEVATIONS), which gives the same values without twice the memory.
    dem: np.ndarray
    # The stream raster: int32 if its integer type fits it, and otherwise int64, each stream cell holding its reach's ID
    # and every other cell 0. None where stream_cells holds its stream cells instead, which take a fraction of the
    # memory, as read_grid gives it
    streams: np.ndarray | None
    land_cover: np.ndarray  # uint8
    dx: float  # the cells' width and height in metres
    dy: float
    water_mask: np.ndarray | None = None  # uint8, 1 where BathyWaterMask has water and 0 elsewhere; None without one
    # The stream raster's cells, as arc.xsection.stream_path.stream_raster_cells gives them, if known, and each one's
    # row * ncols + col, which ascend
    stream_cells: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
    stream_flat: np.ndarray | None = None

    def elevations(self, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
        """The DEM at these cells, as float64."""
        return self.dem[rows, cols].astype(np.float64)

    def stream_raster_cells(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The rows and columns of the stream cells, row by row, and their reaches' IDs."""
        return stream_raster_cells(self.streams) if self.stream_cells is None else self.stream_cells

    def id_raster(self) -> np.ndarray:
        """The stream raster, as streams holds it."""
        if self.streams is not None:
            return self.streams
        rows, cols, ids = self.stream_cells
        raster = np.zeros(self.dem.shape, dtype=ids.dtype)
        raster[rows, cols] = ids
        return raster

    def on_streams(self, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
        """Whether each of these cells, all on the raster, is a stream cell."""
        if self.streams is not None:
            return self.streams[rows, cols] > 0
        flat = self.stream_flat
        if flat is None:
            flat = self.stream_cells[0] * self.dem.shape[1] + self.stream_cells[1]  # ascending, as the cells are
        cells = rows * self.dem.shape[1] + cols
        at = np.minimum(np.searchsorted(flat, cells), max(flat.size - 1, 0))
        return (flat[at] == cells) if flat.size else np.zeros(cells.size, dtype=bool)


# The type the DEM's elevations are sampled in (see Grid.dem)
ELEVATIONS = np.empty(0, np.float64)
# The types a DEM is held in as float32: those whose every value float32 holds exactly
_FLOAT32_DEMS = {np.dtype(t) for t in (np.float32, np.int8, np.uint8, np.int16, np.uint16)}
_WALL_ROWS = 64  # rows of the DEM looked over at a time for cells without data


@dataclass(frozen=True)
class Cells:
    """The stream cells a run works on, in order."""
    rows: np.ndarray
    cols: np.ndarray
    comids: np.ndarray  # the flow file's ID for each cell: its reach's, or a manual cross section's
    reaches: np.ndarray  # the reach each cell is on (a manual cross section's source stream)

    @property
    def count(self) -> int:
        return int(self.rows.size)

    def subset(self, keep: np.ndarray) -> Cells:
        return Cells(self.rows[keep], self.cols[keep], self.comids[keep], self.reaches[keep])


@dataclass
class Section:
    """A stream cell's cross section and what the run finds out about it."""
    xs: XSection
    row: int  # the cell it's centred on, which the low spot may have moved from the stream cell
    col: int
    direction: float  # the stream direction it was sampled in (see arc.xsection.sampling for the convention)
    xs_angle: float  # XS_Angle: the cross section's own direction, from 0 up to pi, towards the higher rows
    rows: np.ndarray  # each ordinate's cell
    cols: np.ndarray
    side_one: int  # which half of the cross section, +1 or -1, is side 1 in the cross-section file
    dem_low_point: float  # the elevation at its centre before any bathymetry, from the DEM or a manual profile
    land_cover: np.ndarray | None = None  # a manual cross section's land cover
    banks: Banks | None = None  # as found
    # With bathymetry and without Bathy_Use_Banks, the bank tops, where the hydraulics divide the channel carved below
    # the DEM's water (dividing_banks)
    bank_tops: Banks | None = None
    carve_banks: Banks | None = None  # with bathymetry, those the channel is carved between: after the width filter
    hydraulic_banks: Banks | None = None  # as the hydraulics use them: the carve's, or its bank tops (dividing_banks)
    bank_elevation: float = math.nan  # the smoothed bank elevation, with bathymetry
    # The water surface its channel was carved to carry the baseflow at (its bank elevation with Bathy_Use_Banks, its
    # stream cell's otherwise), where bathymetry carved it for the baseflow; the standard step holds the baseflow there
    baseflow_level: float = math.nan
    usable: bool = True  # False once the bathymetry drops its reach
    manual: bool = False
    # How the bathymetry carved its channel: "none", "baseflow" (the depth that carries the baseflow), "depth_law"
    # (the drainage-area depth) or "stream_level" (the baseflow at its stream cell's level), and how deep; "" without
    # bathymetry (see arc.outputs.decisions)
    carve: str = ""
    carve_depth: float = math.nan
    dropped_at_edge: bool = False  # its rating curve dropped by Drop_Edge_Sections
    length: float = math.nan  # the length it's sampled to, which grows as steps extend it
    full_half: int = 0  # how many ordinates either side of its centre it has out to the raster's edge (0 if manual)


@dataclass
class Results:
    """What a run made. The per-cell arrays and sections are in the order of cells, a section None where its
    centre had no data.

    Attributes
    ----------
    configs : Configs
        The run's inputs.
    cells : Cells
        The stream cells the run worked on: their rows and columns, flow-file IDs (comids) and reaches.
    slopes : numpy.ndarray
        Each cell's stream slope, in m/m.
    baseflow, max_flow : numpy.ndarray
        Each cell's baseflow and maximum flow from the flow file, in m³/s (0 without them).
    sections : list
        Each cell's cross section (Section), or None.
    curves : RatingCurves or None
        The rating curves, from which the VDT database, curve file and AP database are written: ``output_array()``
        gives a row per cell, in the columns ``columns()`` names. None for a run that makes none.
    vdt : pandas.DataFrame or None
        The VDT database, as written to Print_VDT_Database.
    cross_sections : list
        The cross-section file's rows, with XS_Out_File.
    representative : pandas.DataFrame or None
        The representative cross sections, with Build_Representative_Cross_Section.
    bathymetry : numpy.ndarray or None
        The bathymetry raster, with BATHY_Out_File.
    seconds : float
        How long the run took.
    slope_sources : numpy.ndarray or None
        Where each cell's slope came from, as an index into arc.outputs.decisions.SLOPE_SOURCES.
    skipped : Cells or None
        The stream cells skipped for want of a slope.
    step : StepSummary or None
        What the standard step did, with Standard_Step.
    qa_flags : numpy.ndarray or None
        Each cell's QA_Flag (arc.outputs.decisions).
    """
    configs: Configs
    cells: Cells
    slopes: np.ndarray
    baseflow: np.ndarray
    max_flow: np.ndarray
    sections: list[Section | None]
    curves: RatingCurves | None = None
    vdt: pd.DataFrame | None = None
    cross_sections: list[Section] = field(default_factory=list)
    representative: pd.DataFrame | None = None
    bathymetry: np.ndarray | None = None
    seconds: float = 0.0
    slope_sources: np.ndarray | None = None
    skipped: Cells | None = None
    step: StepSummary | None = None
    qa_flags: np.ndarray | None = None


# --- 1. Inputs --------------------------------------------------------------------------------------------------------


def read_flows(configs: Configs) -> pd.DataFrame | None:
    """The flow file's baseflow and maximum-flow columns, indexed by its IDs. None for a run that only builds
    representative cross sections without baseflow bathymetry, which needs no flows."""
    if not configs.makes_rating_curves and not configs.use_representative_baseflow_bathymetry:
        return None
    
    if not configs.flow_file or not configs.flow_file_id:
        raise ValueError("A Flow_File and its Flow_File_ID are needed for the rating curves.")
    
    path = str(configs.flow_file)
    df = pd.read_parquet(path) if path.endswith('.parquet') else pd.read_csv(path)
    columns = [configs.flow_file_bf] if configs.use_representative_baseflow_bathymetry else \
        [c for c in (configs.flow_file_bf, configs.flow_file_qmax) if c]
    missing = [c for c in [configs.flow_file_id, *columns] if c not in df.columns]
    if missing:
        raise KeyError(f"The flow file {path} has no column {', '.join(missing)}.")
    
    df = df.set_index(configs.flow_file_id)[columns]
    if not df.index.is_unique:
        raise ValueError(f"The flow file {path} repeats IDs in {configs.flow_file_id}.")
    
    return df


def read_grid(configs: Configs) -> Grid:
    """The DEM, stream and land cover rasters, and the water mask if there is one, checked to be on one grid in
    metres or degrees."""
    paths = [("DEM", configs.dem_file), ("stream", configs.stream_file), ("land cover", configs.lu_raster_sameres)]
    if configs.bathywatermask:
        paths.append(("water mask", configs.bathywatermask))

    rasters = {name: Raster(path) for name, path in paths}
    reference = rasters["DEM"]
    for name, raster in rasters.items():
        if raster.is_rotated:
            raise ValueError(f"The {name} raster {raster.filepath} is rotated, which ARC can't use.")
        
        units = raster.axis_units()
        if not units or not units <= {"metre", "metres", "meter", "meters", "degree", "degrees"}:
            raise ValueError(f"The {name} raster {raster.filepath}'s coordinates are in "
                             f"{', '.join(sorted(units)) or 'no known units'}, not metres or degrees.")
        
        if not raster.same_grid(reference):
            raise ValueError(f"The {name} raster {raster.filepath} isn't on the DEM's grid: {raster.shape} cells at "
                             f"{raster.geotransform}, where the DEM has {reference.shape} at {reference.geotransform}.")

    stream_cells = read_stream_cells(rasters["stream"])

    dem = reference.read_array(np.float32 if reference.dtype in _FLOAT32_DEMS else np.float64)
    for top in range(0, dem.shape[0], _WALL_ROWS):
        # A block of rows at a time, which keeps the masks small and in the cache
        block = dem[top:top + _WALL_ROWS]
        no_data = ~np.isfinite(block)
        if reference.nodata_value is not None:
            no_data |= block == np.float64(reference.nodata_value)  # compared as float64, whatever the DEM is held in
        np.copyto(block, OFF_RASTER_ELEVATION, where=no_data)

    land_cover = rasters["land cover"].read_array(np.uint8)

    water_mask = None
    if "water mask" in rasters:
        mask = rasters["water mask"].read_array()
        # Compared as it is, without a float64 copy, but with the nodata value as float64, as a float64 copy would be
        water = mask != 0 if np.issubdtype(mask.dtype, np.integer) else np.isfinite(mask) & (mask != 0.0)
        if rasters["water mask"].nodata_value is not None:
            water &= mask != np.float64(rasters["water mask"].nodata_value)
        del mask
        water_mask = water.view(np.uint8)  # 1 where True and 0 where False, without a copy

    dx, dy = reference.cell_size_in_metres()
    LOG.info(f"Cell size {dx} m by {dy} m")
    return Grid(reference, dem, None, land_cover, dx, dy, water_mask, stream_cells,
                stream_cells[0] * dem.shape[1] + stream_cells[1])


def read_stream_cells(raster: Raster) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The stream raster's cells (Grid.stream_cells): those that hold an ID above 0 that isn't the raster's nodata
    value, as int32 if the raster's integer type fits it and otherwise int64. The raster is read before the DEM, so
    that it's gone before the DEM, which takes as much memory, comes in."""
    narrow = raster.dtype in (np.int8, np.int16, np.int32, np.uint8, np.uint16)
    streams = raster.read_array().astype(np.int32 if narrow else np.int64, copy=False)
    if raster.nodata_value is not None and math.isfinite(raster.nodata_value):
        np.copyto(streams, 0, where=streams == int(raster.nodata_value))

    return stream_raster_cells(streams)


def read_streamfile(configs: Configs):
    """The stream network's features, or None if there's no StrmShp_File."""
    if not configs.strmshp_file:
        return None
    
    return Vector(configs.strmshp_file).to_geopandas()


def stream_network(configs: Configs, gdf: gpd.GeoDataFrame | None) -> dict[int, int] | None:
    """The reach network from the stream layer. None without the layer, or without its reach_id and
    downstream_reach_id fields when neither the bathymetry nor the standard step needs them (they then only give
    slopes to the reaches the slope method can't)."""
    if gdf is None:
        return None
    
    for column in (configs.reach_id, configs.downstream_reach_id):
        if not column or column not in gdf.columns:
            if configs.bathy_out_file or configs.standard_step:
                raise ValueError(f"The stream network {configs.strmshp_file} has no field {column!r}.")
            LOG.warning(f"The stream network {configs.strmshp_file} has no field {column!r}, so a reach without a "
                        "slope can't take its neighbours'.")
            return None
        
    return reach_network(gdf[configs.reach_id].tolist(), gdf[configs.downstream_reach_id].tolist())



def line_outlets(configs: Configs, grid: Grid, gdf: gpd.GeoDataFrame | None, reaches) -> dict[int, tuple[int, int]]:
    """The cell (row, col) holding the end of each of these reaches' lines in the stream layer, which may be off the
    raster. Stream networks are drawn from upstream to downstream, so it's the downstream end, which orders a reach
    none of whose neighbours is on the stream raster (see order_reach). A reach's first feature counts, and of a
    multi-part line, the last part."""
    if gdf is None or configs.reach_id not in gdf.columns:
        return {}
    
    ids = pd.to_numeric(gdf[configs.reach_id], errors="coerce")
    first = (ids.isin(np.unique(np.asarray(reaches, dtype=np.int64))) & ~ids.duplicated()).to_numpy()
    features = gdf[first]
    if features.crs is not None:
        features = features.to_crs(grid.reference.crs)

    x0, width, _, y0, _, height = grid.reference.geotransform
    outlets = {}
    for reach, line in zip(ids[first], features.geometry):
        if line is None or line.is_empty:
            continue

        x, y = (line.geoms[-1] if hasattr(line, "geoms") else line).coords[-1][:2]
        outlets[int(reach)] = (math.floor((y - y0) / height), math.floor((x - x0) / width))

    return outlets



def outlet_levels(configs: Configs, grid: Grid) -> dict | None:
    """The water surfaces Outlet_WSE_File gives to start outlets from, by reach: each point's row and column on the
    raster (fractional, from its x and y in the DEM's coordinates) and its water surface at each of its flows, in
    increasing order (see apply_standard_step). None without the file."""
    if not configs.outlet_wse_file:
        return None
    
    path = str(configs.outlet_wse_file)
    table = pd.read_parquet(path) if path.lower().endswith(".parquet") else pd.read_csv(path)
    missing = sorted({"reach_id", "x", "y", "flow", "wse"} - set(table.columns))
    if missing:
        raise ValueError(f"Outlet_WSE_File {path} has no column {', '.join(missing)}: it needs reach_id, x, y, flow and wse.")
    
    table = table.dropna(subset=["reach_id", "x", "y", "flow", "wse"]).sort_values(["reach_id", "x", "y", "flow"])
    x0, width, _, y0, _, height = grid.reference.geotransform
    levels: dict[int, list] = {}
    for (reach, x, y), point in table.groupby(["reach_id", "x", "y"], sort=False):
        levels.setdefault(int(reach), []).append(((y - y0) / height - 0.5, (x - x0) / width - 0.5,
                                                  point["flow"].to_numpy(np.float64), point["wse"].to_numpy(np.float64)))
        
    return levels



def power_law_targets(configs: Configs, gdf: gpd.GeoDataFrame | None) -> dict[int, tuple[float | None, float | None]]:
    """Each reach's drainage-area bankfull depth and width, where configured."""
    if not (configs.use_bathymetry_powerlaw or configs.use_bathymetry_powerlaw_width):
        return {}
    
    if gdf is None:
        raise ValueError("The drainage-area power laws need a StrmShp_File.")
    
    field_name, reach_field = configs.drainage_area_field, configs.reach_id
    missing = sorted({reach_field, field_name} - set(gdf.columns))
    if missing:
        raise KeyError("The stream vector dataset is missing the columns required for drainage-area bathymetry: "
                       + ", ".join(missing))
    
    table = gdf[[reach_field, field_name]].dropna(subset=[reach_field, field_name])
    reach_ids = pd.to_numeric(table[reach_field], errors='raise').astype(np.int64)
    areas = pd.to_numeric(table[field_name], errors='raise')
    depth_pair = (configs.coefficient_depth, configs.exponent_depth) if configs.use_bathymetry_powerlaw \
        else (None, None)
    targets = {}
    for reach, area in zip(reach_ids, areas):
        reach = int(reach)
        if reach in targets:
            continue  # a reach's first feature counts
        area = float(area)
        if not math.isfinite(area) or area <= 0.0:
            raise ValueError(f"Drainage area for reach {reach} must be positive in field {field_name}.")
        depth, width = power_law_geometry(area, *depth_pair, configs.coefficient_width, configs.exponent_width)
        if depth is not None and not (math.isfinite(depth) and depth > 0.0):
            raise ValueError(f"Estimated depth for reach {reach} was not positive. Check coefficient_depth, "
                             "exponent_depth, and the drainage area values.")
        if width is not None and not (math.isfinite(width) and width > 0.0):
            raise ValueError(f"Estimated width for reach {reach} was not positive. Check coefficient_width, "
                             "exponent_width, and the drainage area values.")
        targets[reach] = (depth, width)

    return targets


# --- Manual cross sections ------------------------------------------------------------------------------------------


def _parse_array(value, dtype) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value.astype(dtype, copy=False)
    
    if value is None or (isinstance(value, float) and math.isnan(value)) or value in ("", "[]"):
        return np.array([], dtype=dtype)
    
    if isinstance(value, (list, tuple)):
        return np.asarray(value, dtype=dtype)
    
    try:
        parsed = json.loads(value)
    except (json.JSONDecodeError, TypeError):
        parsed = ast.literal_eval(value)

    return np.asarray(parsed, dtype=dtype)


def read_manual_sections(configs: Configs) -> dict[int, dict]:
    """The manual cross sections, by the ID they share with the flow file."""
    path = str(configs.manual_cross_sections_file)
    if path.endswith(".parquet"):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path, sep="\t" if path.lower().endswith((".tsv", ".txt")) else ",")

    id_field = configs.flow_file_id if configs.makes_rating_curves else configs.reach_id
    missing = sorted({id_field, *_REQUIRED_MANUAL_COLUMNS} - set(df.columns))
    if missing:
        raise KeyError("Manual cross-section file is missing required columns: " + ", ".join(missing))
    
    records = {}
    for _, row in df.iterrows():
        manual_id = int(row[id_field])
        record = {"row": int(row["Row"]), "col": int(row["Col"]), "xs_angle": float(row.get("XS_Angle", 0.0) or 0.0),
                  "ordinate_dist": float(row["Ordinate_Dist"])}
        for side in (1, 2):
            record[f"xs{side}_profile"] = _parse_array(row[f"XS{side}_Profile"], np.float64)
            record[f"lc{side}_profile"] = _parse_array(row[f"LC{side}_Profile"], np.uint8)
            record[f"xs{side}_row"] = _parse_array(row[f"XS{side}_Row"], np.int64)
            record[f"xs{side}_col"] = _parse_array(row[f"XS{side}_Col"], np.int64)

        if record["xs1_profile"].size == 0 or record["xs2_profile"].size == 0:
            raise ValueError(f"Manual cross section {manual_id} did not contain profile values on both sides.")
        
        for side in (1, 2):
            if not (record[f"xs{side}_profile"].size == record[f"lc{side}_profile"].size
                    == record[f"xs{side}_row"].size == record[f"xs{side}_col"].size):
                raise ValueError(f"Manual cross section {manual_id} has mismatched side-{side} array lengths.")
            
        source = row.get("Source_Stream_ID", manual_id)
        record["source_stream_id"] = manual_id if pd.isna(source) else int(source)
        records[manual_id] = record

    return records


def manual_section(record: dict) -> Section:
    """A manual cross section as a Section: side 2 on the left, side 1 on the right, the shorter padded with
    walls."""
    half = max(record["xs1_profile"].size, record["xs2_profile"].size) - 1

    def both_sides(key, fill, dtype):
        sides = []
        for side in (2, 1):
            values = np.full(half + 1, fill, dtype=dtype)
            values[:record[f"{key.format(side)}"].size] = record[f"{key.format(side)}"]
            sides.append(values)
        return np.concatenate([sides[0][:0:-1], sides[1]])

    elevations = both_sides("xs{}_profile", OFF_RASTER_ELEVATION, np.float64)
    rows = both_sides("xs{}_row", -1, np.int64)
    cols = both_sides("xs{}_col", -1, np.int64)
    land_cover = both_sides("lc{}_profile", np.nan, np.float64)
    xs = XSection(elevations, np.empty(0), float(record["ordinate_dist"]))
    xs_angle = float(record["xs_angle"])
    return Section(xs, int(record["row"]), int(record["col"]), xs_angle + np.pi / 2, xs_angle, rows, cols, 1,
                   float(record["xs1_profile"][0]), land_cover, manual=True)


# --- 2. Stream cells and 3. slopes ---------------------------------------------------------------------------------


def stream_cells(grid: Grid, flows: pd.DataFrame | None, manual: dict | None) -> Cells:
    """The cells to work on, in the stream raster's row order or the manual cross sections': those of the flow
    file's reaches, or without flows, every stream cell."""
    if flows is None:
        ids = np.unique(grid.stream_raster_cells()[2])
    else:
        ids = np.asarray(flows.index, dtype=np.int64)

    if manual:
        shared = [int(i) for i in ids if int(i) in manual]
        if not shared:
            raise ValueError("No IDs were shared between the ARC flow file and the manual cross-section file.")
        
        return Cells(np.array([manual[i]["row"] for i in shared], dtype=np.int64),
                     np.array([manual[i]["col"] for i in shared], dtype=np.int64), np.array(shared, dtype=np.int64),
                     np.array([manual[i]["source_stream_id"] for i in shared], dtype=np.int64))
    
    if (ids > 0).all():
        # Only the stream cells can hold these IDs
        rows, cols, values = grid.stream_raster_cells()
        keep = np.isin(values, ids)
        rows, cols, comids = rows[keep], cols[keep], values[keep].astype(np.int64)
    else:
        streams = grid.id_raster()
        rows, cols = np.nonzero(np.isin(streams, ids))
        comids = streams[rows, cols].astype(np.int64)

    if rows.size == 0:
        raise ValueError("No stream cell has an ID in the flow file, so there is nothing to work on.")
    
    return Cells(rows.astype(np.int64), cols.astype(np.int64), comids, comids.copy())


def _cells_by_reach(cells: Cells) -> dict[int, list[int]]:
    by_reach: dict[int, list[int]] = {}
    for k, reach in enumerate(cells.reaches):
        by_reach.setdefault(int(reach), []).append(k)

    return by_reach


def _reach_cells(rows, cols, cells: Cells, ks) -> tuple[np.ndarray, np.ndarray, dict]:
    """A reach's stream cells, rows and cols, with any of cells ks off them (as a manual cross section's can be),
    and where each cell is among them."""
    have = {(int(r), int(c)): i for i, (r, c) in enumerate(zip(rows, cols))}
    extra = list(dict.fromkeys((int(cells.rows[k]), int(cells.cols[k])) for k in ks
                               if (int(cells.rows[k]), int(cells.cols[k])) not in have))
    if extra:
        rows = np.concatenate([rows, np.array([r for r, _ in extra], dtype=np.int64)])
        cols = np.concatenate([cols, np.array([c for _, c in extra], dtype=np.int64)])
        have = {(int(r), int(c)): i for i, (r, c) in enumerate(zip(rows, cols))}

    return rows, cols, have


def _at_cells(values, have: dict, cells: Cells, ks) -> np.ndarray:
    return np.array([values[have[(int(cells.rows[k]), int(cells.cols[k]))]] for k in ks], dtype=np.float64)


def _local_slopes(grid: Grid, rows, cols, cells: Cells, ks, distance, neighbours=None) -> tuple[np.ndarray, np.ndarray]:
    """The local average slope at each of cells ks, and at every cell of their reach (see _reach_cells), averaged
    over the reach's cells and its neighbours' (neighbours, their rows and cols) within the distance."""
    own = len(rows)
    extra = 0 if neighbours is None else len(neighbours[0])
    if extra:
        rows, cols = np.concatenate([rows, neighbours[0]]), np.concatenate([cols, neighbours[1]])

    rows, cols, have = _reach_cells(rows, cols, cells, ks)
    stations = along_stream_stations(rows, cols, grid.dx, grid.dy)
    values = local_average_slopes(grid.elevations(rows, cols), rows, cols, stations, grid.dx, grid.dy, distance)
    return _at_cells(values, have, cells, ks), np.r_[values[:own], values[own + extra:]]


def _main_stem_cells(network, reach: int, stream_reaches: dict) -> tuple[np.ndarray, np.ndarray] | None:
    """The stream cells of the reach's neighbours on its main stem, the reach above it (main_stem_above) and the one
    below, or None without a stream network."""
    if network is None or reach not in network:
        return None
    
    above = main_stem_above(network, reach, stream_reaches)
    parts = [stream_reaches[n] for n in [above, *network.successors(reach)] if n in stream_reaches]
    if not parts:
        return None
    
    return np.concatenate([r for r, _ in parts]), np.concatenate([c for _, c in parts])


def _fitted_slopes(grid: Grid, rows, cols, cells: Cells, ks, distance) -> tuple[np.ndarray, float]:
    """The fitted-line slope at each of cells ks, and their reach's, from all of its cells (see _reach_cells and
    local_fit_slopes)."""
    rows, cols, have = _reach_cells(rows, cols, cells, ks)
    if rows.size == 0:
        return np.full(len(ks), math.nan), math.nan
    
    stations = along_stream_stations(rows, cols, grid.dx, grid.dy)
    z = grid.elevations(rows, cols)
    values, reach_slope = local_fit_slopes(np.where(z >= OFF_RASTER_ELEVATION, np.nan, z), stations, distance)
    return _at_cells(values, have, cells, ks), reach_slope


def _dem_slopes(configs: Configs, grid: Grid, cells: Cells, by_reach: dict, network=None, stream_reaches=None
                ) -> tuple[dict, dict]:
    """By a method that measures slopes between stream cells: each reach's slope, NaN where the DEM gives none, and
    the slopes of the run's cells, by reach, NaN where a cell has none. The run's reaches get a slope, and if one of them
    gets none, which it then takes from its neighbours (cell_slopes), so does every other reach on the stream raster.

    With a stream network, the local averages take in the cells of the reach's main-stem neighbours too, so that a
    cell near a junction is averaged over both sides of it; the reach's percentiles and median slope are still its
    own cells'. The fitted lines (`local_fit`) stay within the reach. `stream_reaches` are each reach's cells on the stream
    raster (`stream_cells_by_reach`), if they're known."""
    method = configs.stream_slope_method
    distance = configs.gen_slope_dist
    if stream_reaches is None:
        stream_reaches = stream_cells_by_reach(cells=grid.stream_raster_cells())

    empty = (np.empty(0, np.int64), np.empty(0, np.int64))
    reach_slopes, cell_values = {}, {}

    def measure(reach: int) -> None:
        rows, cols = stream_reaches.get(reach, empty)
        ks = by_reach.get(reach, [])
        if method == "local_fit":
            cell_values[reach], reach_slopes[reach] = _fitted_slopes(grid, rows, cols, cells, ks, distance)
            return
        
        if method in ("local_average", "local_average_corrected"):
            local, everywhere = _local_slopes(grid, rows, cols, cells, ks, distance,
                                              _main_stem_cells(network, reach, stream_reaches))
            
        if method == "local_average":
            measured = everywhere[~np.isnan(everywhere)]
            reach_slopes[reach] = float(np.median(measured)) if measured.size else math.nan
            cell_values[reach] = local
            return
        
        stations = along_stream_stations(rows, cols, grid.dx, grid.dy)
        median, lower, upper = reach_median_slope(grid.elevations(rows, cols), rows, cols, stations, grid.dx, grid.dy,
                                                  distance, configs.slope_low_percentile,
                                                  configs.slope_high_percentile)
        reach_slopes[reach] = median
        cell_values[reach] = np.full(len(ks), median) if method == "reach_average" else corrected_local_slopes(local, lower, upper)

    for reach in sorted(by_reach):
        measure(reach)

    if any(math.isnan(reach_slopes[reach]) for reach in by_reach):
        for reach in sorted(set(stream_reaches) - set(by_reach)):
            measure(reach)

    return reach_slopes, {reach: cell_values[reach] for reach in by_reach}



def _end_point_slopes(configs: Configs, grid: Grid, gdf: gpd.GeoDataFrame, by_reach: dict) -> dict:
    """Each of the run's reaches' end-point slope, NaN where it can't be measured, and if one of them has none, which it
    then takes from its neighbours, every other reach's on the stream raster."""
    if gdf is None:
        raise ValueError("The 'end_points' stream slope method requires a strmshp_file.")
    
    id_field = configs.flow_file_id if configs.makes_rating_curves else configs.reach_id
    if id_field not in gdf.columns:
        raise KeyError(f"The stream network {configs.strmshp_file} has no field {id_field!r}.")
    
    dem = grid.dem.astype(np.float64)
    dem[grid.dem >= OFF_RASTER_ELEVATION] = np.nan
    ids = gdf[id_field]
    slopes = {}

    def measure(reach: int) -> None:
        matches = np.flatnonzero((ids == reach).to_numpy())
        slope = math.nan
        if matches.size:
            feature = gdf.iloc[matches[:1]]  # a reach's first feature counts
            length = float(feature.to_crs(feature.estimate_utm_crs()).length.iloc[0])
            slope = end_point_slope(feature.to_crs(grid.reference.crs).geometry.iloc[0], length, dem,
                                    grid.reference.geotransform)
        slopes[int(reach)] = slope

    for reach in sorted(by_reach):
        measure(reach)
    if any(math.isnan(slopes[int(reach)]) for reach in by_reach):
        for reach in sorted(set(np.unique(grid.stream_raster_cells()[2]).tolist()) - set(by_reach)):
            measure(reach)
    return slopes


def _listed(reaches) -> str:
    reaches = sorted(reaches)
    return ", ".join(map(str, reaches[:10])) + (" and others" if len(reaches) > 10 else "")



def cell_slopes(configs: Configs, grid: Grid, cells: Cells, network, gdf: gpd.GeoDataFrame | None, stream_reaches=None
                ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Each cell's slope by Stream_Slope_Method, which cells to keep (see arc.xsection.slope), and where each slope
    came from (an index into arc.outputs.decisions.SLOPE_SOURCES). Whatever the method, a reach it can't give a slope
    takes its neighbours' in the stream network, and a cell without one takes its reach's. The cells of a reach left
    without a slope are skipped. stream_reaches are as _dem_slopes takes them."""
    by_reach = _cells_by_reach(cells)
    if configs.stream_slope_method == "end_points":
        reach_slopes = _end_point_slopes(configs, grid, gdf, by_reach)
        cell_values = {reach: np.full(len(ks), reach_slopes[reach]) for reach, ks in by_reach.items()}
    else:
        reach_slopes, cell_values = _dem_slopes(configs, grid, cells, by_reach, network, stream_reaches)

    measured = {reach for reach, slope in reach_slopes.items() if not math.isnan(slope)}
    if network is not None:
        left = fill_unresolved_reach_slopes(network, reach_slopes)
    else:
        left = set(reach_slopes) - measured

    slopes = np.full(cells.count, np.nan)
    keep = np.ones(cells.count, dtype=bool)
    sources = np.full(cells.count, SLOPE_SOURCES.index("neighbours"), dtype=np.int8)
    for reach, ks in by_reach.items():
        if reach in left:
            keep[ks] = False
        elif reach in measured:
            own = ~np.isnan(cell_values[reach])
            slopes[ks] = np.where(own, cell_values[reach], reach_slopes[reach])
            sources[ks] = np.where(own, SLOPE_SOURCES.index("cell"), SLOPE_SOURCES.index("reach"))
        else:
            slopes[ks] = reach_slopes[reach]  # its neighbours'

    filled = (set(by_reach) - measured) - left
    if filled:
        LOG.info(f"Reaches {_listed(filled)} have no slope of their own, and take their neighbours' in the stream network.")
        
    skipped = left & set(by_reach)
    if skipped:
        hint = "" if network is not None else " (give StrmShp_File, with reach_id and downstream_reach_id, to take them from the stream network)"
        LOG.warning(f"Skipping the {int(np.count_nonzero(~keep))} stream cells of reaches {_listed(skipped)}: neither "
                    f"the DEM nor their neighbours give them a slope{hint}.")
        
    return slopes, keep, sources


def cell_flows(configs: Configs, flows: pd.DataFrame | None, cells: Cells) -> tuple[np.ndarray, np.ndarray]:
    """Each cell's baseflow and maximum flow, 0 where there are none."""
    baseflow = np.zeros(cells.count)
    max_flow = np.zeros(cells.count)
    if flows is None:
        return baseflow, max_flow
    
    missing = sorted(set(cells.comids.tolist()) - set(flows.index.tolist()))
    if missing:
        raise KeyError("The flow file has no flows for stream IDs " + ", ".join(map(str, missing[:10]))
                       + ("..." if len(missing) > 10 else ""))
    
    if configs.flow_file_bf:
        baseflow = flows[configs.flow_file_bf].reindex(cells.comids).to_numpy(np.float64)
    if configs.flow_file_qmax and configs.flow_file_qmax in flows.columns:
        max_flow = flows[configs.flow_file_qmax].reindex(cells.comids).to_numpy(np.float64)

    return baseflow, max_flow


def cell_targets(targets: dict, cells: Cells) -> tuple[np.ndarray, np.ndarray]:
    """Each cell's drainage-area bankfull depth and width, NaN where there's none."""
    depth = np.full(cells.count, np.nan)
    width = np.full(cells.count, np.nan)
    if not targets:
        return depth, width
    
    missing = sorted({int(r) for r in cells.reaches} - set(targets))
    if missing:
        raise KeyError("The stream network has no drainage area for reach_id " + ", ".join(map(str, missing[:10]))
                       + ("..." if len(missing) > 10 else ""))
    
    for k, reach in enumerate(cells.reaches):
        d, w = targets[int(reach)]
        depth[k] = math.nan if d is None else d
        width[k] = math.nan if w is None else w

    return depth, width


# --- 4. Roughness ----------------------------------------------------------------------------------------------------


def _bounded(n: np.ndarray) -> np.ndarray:
    # n above 10 is taken as 0.035, and n at or below 0 as 0.005
    n[n > 10] = 0.035
    n[n <= 0.0] = 0.005
    return n


def mannings_n_table(configs: Configs) -> tuple[np.ndarray, float]:
    """Each land cover class's Manning's n (float64, by class number), and the water class's n, both bounded (see
    _bounded) and at float32's precision."""
    path = str(configs.lu_manning_n)
    table = pd.read_parquet(path) if path.endswith('.parquet') else pd.read_csv(path, sep='\t')
    lookup = np.zeros(256, dtype=np.float32)
    lookup[table.iloc[:, 0].astype(np.uint8).values] = table.iloc[:, 2].values
    water_rows = table.loc[table.iloc[:, 0].astype(int) == configs.lc_water_value]
    if water_rows.empty:
        raise ValueError("The Manning's n table has no entry for the configured water class.")
    
    water_n = float(water_rows.iloc[-1, 2])
    if not math.isfinite(water_n):
        raise ValueError("Water-class Manning's n must be finite.")
    
    return _bounded(lookup).astype(np.float64), float(_bounded(np.array([water_n], dtype=np.float32))[0])


def mannings_n_raster(configs: Configs, land_cover: np.ndarray) -> tuple[np.ndarray, float]:
    """Each cell's Manning's n from its land cover class, and the water class's n (see mannings_n_table)."""
    lookup, water_n = mannings_n_table(configs)
    return lookup[land_cover], water_n


class ManningsN:
    """The Manning's n raster, kept as its land cover classes rather than as numbers, a byte a cell instead of eight:
    each cell's n is its class's, or the water's where water_between_banks marks it. Cross sections are sampled from
    it as from the raster of numbers, bit for bit."""

    def __init__(self, classes: np.ndarray, lookup: np.ndarray, water_n: float):
        self.classes = classes  # uint8
        self.lookup = np.asarray(lookup, dtype=np.float64)  # each class's n
        self.water_n = float(water_n)
        self.water = np.zeros(classes.shape, dtype=np.bool_)

    def water_between_banks(self, rows: np.ndarray, cols: np.ndarray) -> None:
        """Give these cells (any off the raster are passed over) the water's n."""
        on_raster = (rows >= 0) & (rows < self.classes.shape[0]) & (cols >= 0) & (cols < self.classes.shape[1])
        self.water[rows[on_raster], cols[on_raster]] = True

    def sample(self, row: int, col: int, direction: float, length: float, dx: float, dy: float) -> np.ndarray:
        """Sampled as sample_elevations samples a raster."""
        return sample_classes(self.classes, self.water, self.lookup, self.water_n, int(row), int(col),
                              float(direction), float(length), float(dx), float(dy))

    def at(self, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
        """Each cell's n, OFF_RASTER_ELEVATION off the raster."""
        on_raster = (rows >= 0) & (rows < self.classes.shape[0]) & (cols >= 0) & (cols < self.classes.shape[1])
        n = np.full(rows.size, OFF_RASTER_ELEVATION)
        r, c = rows[on_raster], cols[on_raster]
        n[on_raster] = np.where(self.water[r, c], self.water_n, self.lookup[self.classes[r, c]])
        return n


# --- 5. Cross sections --------------------------------------------------------------------------------------------


def _side_one(direction: float) -> int:
    """Which half of a cross section sampled in this stream direction is side 1: the half towards the higher rows,
    or along a row towards the higher columns."""
    xs_direction = direction - np.pi / 2
    sin, cos = math.sin(xs_direction), math.cos(xs_direction)
    return 1 if sin > _ALONG_A_ROW or (abs(sin) <= _ALONG_A_ROW and cos > 0.0) else -1


def reach_length(grid: Grid, row: int, col: int) -> float:
    """A cross section length that takes any cross section centred on a cell past the raster's edge on both sides:
    each ordinate is a cell further along the axis the cross section crosses fastest, and no more than hypot(dx, dy)
    from the last."""
    rows, cols = grid.dem.shape
    cells = max(row, rows - 1 - row, col, cols - 1 - col) + 2
    return 2.0 * cells * math.hypot(grid.dx, grid.dy)


def sample_section(configs: Configs, grid: Grid, row: int, col: int, offsets: np.ndarray,
                   whole: bool = False, reach_cells=None) -> Section | None:
    """A stream cell's cross section, with its elevations but not yet its roughness: centred on the low spot and turned
    to where the channel is narrowest, and out on each side to the first ground more than MAX_BANK_HEIGHT above its
    centre, or with whole, to the raster's edge on both sides. None if its centre has no data. reach_cells are the
    rows and columns of its reach's stream cells (stream_cells_by_reach), which the stream's direction is taken from,
    or without them, grid.streams, which a grid from read_grid doesn't hold (it keeps only the stream cells)."""
    if reach_cells is None:
        direction = stream_direction(grid.streams, row, col, configs.gen_dir_dist, grid.dx, grid.dy)
    else:
        direction = reach_direction(*reach_cells, row, col, configs.gen_dir_dist, grid.dx, grid.dy)

    if configs.low_spot_range > 0:
        row, col = low_spot_cell(grid.dem, row, col, direction, reach_length(grid, row, col), grid.dx, grid.dy,
                                 configs.low_spot_range, like=ELEVATIONS)
        
    length = reach_length(grid, row, col)
    direction = narrowest_direction(grid.dem, row, col, direction, length, grid.dx, grid.dy, offsets, like=ELEVATIONS)
    _, _, spacing, full_half = _ordinates(direction, length, grid.dx, grid.dy)
    # Out to the first ground more than MAX_BANK_HEIGHT above the centre on each side, as far as finding banks looks,
    # or with whole, to the raster's edge
    half = full_half if whole else ordinates_above(grid.dem, row, col, direction, length, grid.dx, grid.dy, MAX_BANK_HEIGHT, ELEVATIONS)
    length = _length_for(max(half, 1), spacing)
    elevations, spacing = sample_elevations(grid.dem, row, col, direction, length, grid.dx, grid.dy, ELEVATIONS)
    center = elevations.size // 2
    if not elevations[center] < OFF_RASTER_ELEVATION:
        return None
    
    rows, cols = ordinate_cells(row, col, direction, length, grid.dx, grid.dy)
    return Section(XSection(elevations, np.empty(0), float(spacing)), int(row), int(col), float(direction),
                   float((direction - np.pi / 2) % np.pi), rows, cols, _side_one(direction), float(elevations[center]),
                   length=float(length), full_half=int(full_half))


def reaches_its_banks(along: np.ndarray | None, water_value) -> bool:
    """Whether a cross section sampled as far as ground MAX_BANK_HEIGHT up (sample_section) also reaches, with water
    at its centre in the water raster (along), to where the water ends on each side, as finding its banks looks."""
    if along is None:
        return True
    
    center = along.size // 2
    if along[center] != water_value:
        return True
    
    return bool((along[:center] != water_value).any() and (along[center + 1:] != water_value).any())


def banks_at(configs: Configs) -> str:
    """What the banks are (arc.bathymetry.find_banks' `at`): with bathymetry, the bank tops with Bathy_Use_Banks, the
    level the channel is rebuilt below, and the DEM's water's edge without it, as the carve fills only the DEM's water;
    without bathymetry, the bank tops too, where the hydraulics divide the channel from its overbanks, and where the
    channel holds the maximum flow's water, that water's edges."""
    if not configs.bathy_out_file:
        return "divide"
    return "tops" if configs.bathy_use_banks else "water"


def dividing_banks(section: Section, banks: Banks | None) -> Banks | None:
    """The banks the hydraulics divide a cross section at, with the water's n between them: with bathymetry and
    without Bathy_Use_Banks, its bank tops (bank_tops), never inside the banks given, the carve's, which are the DEM's
    water's edge; otherwise the banks given. The carve fills only the DEM's water, but the channel the water flows in
    reaches its bank tops: the bank faces above the water are channel, not overbank."""
    tops = section.bank_tops
    if tops is None or not tops.valid or tops.single_cell or banks is None or not banks.valid:
        return banks
    
    left = (tops.left, tops.left_elevation) if tops.left > banks.left else (banks.left, banks.left_elevation)
    right = (tops.right, tops.right_elevation) if tops.right > banks.right else (banks.right, banks.right_elevation)
    return tops._replace(left=left[0], right=right[0], left_elevation=left[1], right_elevation=right[1])


def bank_ceiling(configs: Configs, section: Section, grid: Grid, mannings_n: ManningsN, resampler: Resampler | None,
                 max_flow: float, slope: float, roughness: DepthRoughness | None) -> float:
    """How far above the stream cell the water that carries the cross section's maximum flow over the DEM's cross
    section stands (trim_wse), or NaN: with Bathy_Use_Banks the bank tops are no higher, as a bank above it is a valley
    wall, and without bathymetry too, where the banks are its edges if the ratio is still falling there. The cross
    section has no Manning's n yet, so it is sampled for this (a manual one's, at its ordinates' cells). A resampler
    extends a trimmed cross section past that water first (worked out again on it extended, a few times at most), so
    that the bank search sees the ground up to it."""
    thalweg = float(section.xs.elevations[section.xs.elevations.size // 2])
    wse = math.nan
    for _ in range(3):
        n = mannings_n.at(section.rows, section.cols) if section.manual else \
            mannings_n.sample(section.row, section.col, section.direction, section.length, grid.dx, grid.dy)
        wse = trim_wse(configs, section, n, max_flow, slope, roughness)
        if not math.isfinite(wse) or resampler is None or not resampler.reach_above(section, wse + TRIM_MARGIN):
            break
    return wse - thalweg


def bank_water(configs: Configs, grid: Grid) -> tuple[np.ndarray | None, int | None]:
    """The raster whose water's end gives the banks, and the water's value in it: the water mask if there is one,
    whatever FindBanksBasedOnLandCover says, or else with FindBanksBasedOnLandCover, the land cover. None and None if
    neither, when the banks come from the DEM alone."""
    if grid.water_mask is not None:
        return grid.water_mask, 1
    
    if configs.findbanksbasedonlandcover:
        return grid.land_cover, configs.lc_water_value
    
    return None, None


def water_along(section: Section, water: np.ndarray | None, grid: Grid) -> np.ndarray | None:
    """The bank_water raster at each ordinate's cell (as arc.bathymetry.sample_land_cover reads it), NaN off the
    raster, or None without one. A manual cross section takes the water mask at its ordinates' cells, or its own land
    cover."""
    if water is None:
        return None
    
    if section.manual and water is not grid.water_mask:
        return section.land_cover
    
    rows, cols = section.rows, section.cols
    on_raster = (rows >= 0) & (rows < water.shape[0]) & (cols >= 0) & (cols < water.shape[1])
    values = np.full(rows.size, np.nan)
    values[on_raster] = water[rows[on_raster], cols[on_raster]]
    return values


def sample_roughness(section: Section, grid: Grid, mannings_n: ManningsN) -> None:
    """Give a section its Manning's n: sampled like its elevations, or for a manual one, each ordinate's cell's."""
    if section.manual:
        section.xs.mannings_n = mannings_n.at(section.rows, section.cols)
    else:
        section.xs.mannings_n = mannings_n.sample(section.row, section.col, section.direction, section.length,
                                                  grid.dx, grid.dy)


# --- 6. Trimming the cross sections (see the notes above) -----------------------------------------------------------


def trim_level(configs: Configs, section: Section, wse: float) -> float:
    """The highest water surface the later steps are expected to work at on a cross section, its banks and Manning's
    n as found: TRIM_MARGIN above the water that carries its maximum flow over the DEM's cross section (wse, from
    trim_wse), and with the standard step, above its table's top over a bed TRIM_CARVING below the stream cell; or, for
    representative cross sections, their highest stage. Without either, the stream cell's elevation."""
    xs = section.xs
    thalweg = float(xs.elevations[xs.elevations.size // 2])
    level = highest_stage(thalweg) if configs.build_representative_cross_section else thalweg
    if math.isfinite(wse):
        if configs.standard_step:
            wse += max(3.0, 0.5 * (wse - thalweg + TRIM_CARVING))  # as HydraulicTable reaches above the top
        level = max(level, wse + TRIM_MARGIN)

    return level


def trim_wse(configs: Configs, section: Section, n: np.ndarray, max_flow: float, slope: float,
             roughness: DepthRoughness | None) -> float:
    """The water surface that carries a cross section's maximum flow over the DEM's cross section, n its Manning's n
    before the water's goes between banks, for trim_level: NaN where there's none, or no flow or slope."""
    if not (max_flow > 0.0 and slope > 0.0):
        return math.nan
    
    xs = section.xs
    return wse_for_discharge(XSection(xs.elevations, n, xs.ordinate_distance), max_flow, slope, roughness=roughness,
                             slope_factor=configs.slope_adjustment_factor)


def fit_section(configs: Configs, section: Section, grid: Grid, mannings_n: ManningsN, resampler: Resampler,
                max_flow: float, slope: float, roughness: DepthRoughness | None, target_width: float) -> None:
    """Cut a cross section to just above the highest water surface the later steps are expected to work at
    (trim_level), extending it first where it doesn't reach that high, and working the estimate out again on it
    extended. The water that carries its maximum flow is worked out again only if it reached the cross section's end:
    extending a cross section that rises above it on both sides adds ground only beyond where the water stopped, which
    leaves it where it was, bit for bit."""
    known = False  # whether wse is the cross section's as it is
    while True:
        if not known:
            n = mannings_n.sample(section.row, section.col, section.direction, section.length, grid.dx, grid.dy)
            wse = trim_wse(configs, section, n, max_flow, slope, roughness)

        level = trim_level(configs, section, wse)
        if reaches_above(section, level):
            break

        known = reaches_above(section, wse)
        thalweg = float(section.xs.elevations[section.xs.elevations.size // 2])
        spacing = float(section.xs.ordinate_distance)
        needed = ordinates_above(grid.dem, section.row, section.col, section.direction,
                                 _length_for(section.full_half, spacing), grid.dx, grid.dy, level - thalweg,
                                 ELEVATIONS)
        if not resampler.extend(section, max(needed, 2 * (section.xs.elevations.size // 2))):
            break

    trim_section(section, level, (target_width,))


def _ordinates_to(elevations: np.ndarray, step: int, level: float) -> int:
    """How many ordinates out from the centre on one side the first above level is, or more than the side has."""
    center = elevations.size // 2
    side = elevations[center + 1:] if step > 0 else elevations[center - 1::-1]
    above = np.flatnonzero(side > level)
    return int(above[0]) + 1 if above.size else elevations.size


def _ordinates_for(distance: float, spacing: float) -> int:
    """Ordinates enough either side for a bank or a channel this far from the centre, with one to spare (0 for
    none)."""
    return math.ceil(distance / spacing) + 2 if math.isfinite(distance) and distance > 0.0 else 0


def trim_section(section: Section, level: float, widths=()) -> None:
    """Keep only a cross section's ordinates out to the first above level on each side, and enough for its banks and
    for channels these widths (any of them on one side). Its Manning's n isn't sampled yet."""
    xs = section.xs
    center, spacing = xs.elevations.size // 2, float(xs.ordinate_distance)
    half = max(_ordinates_to(xs.elevations, -1, level), _ordinates_to(xs.elevations, 1, level),
               *(_ordinates_for(distance, spacing) for distance in (section.banks.left, section.banks.right, *widths)))
    if half >= center:
        return
    
    keep = slice(center - half, center + half + 1)
    xs.elevations = xs.elevations[keep].copy()
    section.rows, section.cols = section.rows[keep].copy(), section.cols[keep].copy()
    section.length = _length_for(half, spacing)


def _length_for(half: int, spacing: float) -> float:
    """A cross section length that samples `half` ordinates either side of the centre at this spacing."""
    return 2.0 * spacing * (half + 0.5)


def reaches_above(section: Section, level: float) -> bool:
    """Whether the hydraulics can work a cross section out at a water surface as they would on it untrimmed: it rises
    above the water on both sides, or it's untrimmed (or manual). Only the latter for a level that isn't a number."""
    valley = section.xs.valley
    if (valley.elevations.size // 2 >= valley.full_half) if valley is not None else \
            (section.xs.elevations.size // 2 >= section.full_half):
        return True
    
    profile = hydraulic_profile(section.xs)
    return bool(_rises_above(profile.elevations, profile.center, float(level)))


@njit(cache=True, error_model="numpy")
def _rises_above(elevations, center, level):
    """Whether a profile has a vertex above level on both sides of its centre."""
    left = right = False
    for j in range(center):
        if elevations[j] > level:
            left = True
            break

    for j in range(center + 1, elevations.size):
        if elevations[j] > level:
            right = True
            break

    return left and right


class Resampler:
    """Extends trimmed cross sections, sampling the ordinates beyond their ends from the DEM and the Manning's n raster
    as they were before they were trimmed."""

    def __init__(self, grid: Grid, mannings_n: ManningsN):
        self.grid = grid
        self.mannings_n = mannings_n

    def extend(self, section: Section, half: int) -> bool:
        """Sample a cross section out to `half` ordinates either side of its centre (as far as it was before it was
        trimmed at most), and say whether it grew. The arrays are replaced, never changed in place."""
        xs, grid = section.xs, self.grid
        old = xs.elevations.size // 2
        half = min(int(half), section.full_half)
        if half <= old:
            return False
        
        spacing = float(xs.ordinate_distance)
        length = _length_for(half, spacing)
        z, _ = sample_elevations(grid.dem, section.row, section.col, section.direction, length, grid.dx, grid.dy, ELEVATIONS)
        added = half - old
        xs.elevations = np.concatenate([z[:added], xs.elevations, z[-added:]])
        n = None
        if xs.mannings_n.size:
            n = self.mannings_n.sample(section.row, section.col, section.direction, length, grid.dx, grid.dy)
            xs.mannings_n = np.concatenate([n[:added], xs.mannings_n, n[-added:]])

        section.rows, section.cols = ordinate_cells(section.row, section.col, section.direction, length, grid.dx, grid.dy)
        if xs.profile is not None:
            # A carved channel's profile: the ground beyond its banks is the ordinates
            p = xs.profile
            stations = np.concatenate([np.arange(-half, -old) * spacing, p.stations, np.arange(old + 1, half + 1) * spacing])
            xs.profile = Profile(stations, np.concatenate([z[:added], p.elevations, z[-added:]]),
                                 np.concatenate([n[:added], p.mannings_n, n[-added:]]), p.center + added)
            
        section.length = length
        return True

    def reach_above(self, section: Section, level: float) -> bool:
        """Extend a cross section, doubling it, until it reaches above a water surface (reaches_above), or is as
        long as before it was trimmed, and say whether it grew."""
        grew = False
        while not reaches_above(section, level):
            valley = section.xs.valley
            if valley is not None:
                # The overbanks are the lines square to the valley (arc.xsection.valley): it's they that grow
                if not self.extend_valley(section, 2 * (valley.elevations.size // 2)):
                    break
            elif not self.extend(section, 2 * (section.xs.elevations.size // 2)):
                break

            grew = True

        return grew

    def extend_valley(self, section: Section, half: int) -> bool:
        """Sample a cross section's overbanks square to the valley out to `half` ordinates from each bank (to the
        raster's edge at most), and say whether they grew."""
        valley = section.xs.valley
        if min(int(half), valley.full_half) <= valley.elevations.size // 2:
            return False
        
        section.xs.valley = sample_valley(self.grid, self.mannings_n, section, valley.direction, valley.sinuosity,
                                          valley.slope_ratio, int(half))
        return True

    def hold_width(self, section: Section, width: float) -> bool:
        """Extend a cross section to hold a channel this wide on either side, and say whether it grew."""
        return self.extend(section, _ordinates_for(width, float(section.xs.ordinate_distance)))


# --- Valley overbanks (Valley_Sections) ------------------------------------------------------------------------------


def sample_valley(grid: Grid, mannings_n: ManningsN, section: Section, direction: float, sinuosity: float,
                  ratio: float, half: int | None = None) -> Valley:
    """A cross section's overbanks square to the valley (arc.xsection.valley): from each of its banks, a line in the
    valley's direction, out `half` ordinates, or without it, to the first ground above the lower of the ends of the
    cross section itself (as far as it was trimmed to reach), with the water they cross filled
    (fill_other_water)."""
    xs = section.xs
    row, col, dx, dy = section.row, section.col, grid.dx, grid.dy
    own = section.direction - math.pi / 2
    left, right = float(xs.left_bank_distance), float(xs.right_bank_distance)
    across = direction - math.pi / 2
    cos, sin = math.cos(across), math.sin(across)
    spacing = 1.0 / max(abs(cos) / dx, abs(sin) / dy)  # a cell a step along the axis it crosses fastest
    lines = (((col - math.cos(own) * left / dx, row - math.sin(own) * left / dy), -cos, -sin),
             ((col + math.cos(own) * right / dx, row + math.sin(own) * right / dy), cos, sin))
    z = xs.elevations
    center = float(z[z.size // 2])
    ends = [float(e) for e in (z[0], z[-1]) if e < OFF_RASTER_ELEVATION]
    level = center + max(min(ends) - center if ends else MAX_BANK_HEIGHT, 0.0)
    reaches = [ray_reach(grid.dem, x0, y0, c, s, spacing, dx, dy, level) for (x0, y0), c, s in lines]
    full_half = max(edge for _, edge in reaches)
    half = max(above for above, _ in reaches) if half is None else int(half)
    half = max(min(half, full_half), 1)
    elevations, n, rows, cols = [], [], [], []
    distances = np.arange(1, half + 1) * spacing
    for (x0, y0), c, s in lines:
        elevations.append(ray_values(grid.dem, x0, y0, c, s, spacing, half, dx, dy))
        n.append(ray_classes(mannings_n.classes, mannings_n.water, mannings_n.lookup, mannings_n.water_n, x0, y0, c,
                             s, spacing, half, dx, dy))
        cols.append(np.rint(x0 + c * distances / dx).astype(np.intp))
        rows.append(np.rint(y0 + s * distances / dy).astype(np.intp))

    center_n = float(xs.mannings_n[xs.mannings_n.size // 2]) if xs.mannings_n.size else 0.0
    rows = np.concatenate([rows[0][::-1], (row,), rows[1]])
    cols = np.concatenate([cols[0][::-1], (col,), cols[1]])
    on_raster = (rows >= 0) & (rows < grid.dem.shape[0]) & (cols >= 0) & (cols < grid.dem.shape[1])
    wet = np.zeros(rows.size, bool)
    wet[on_raster] = grid.on_streams(rows[on_raster], cols[on_raster])
    if grid.water_mask is not None:
        wet[on_raster] |= grid.water_mask[rows[on_raster], cols[on_raster]] > 0

    wet[half] = False
    elevations = fill_other_water(np.concatenate([elevations[0][::-1], (center,), elevations[1]]), wet)
    return Valley(float(direction), float(sinuosity), float(ratio), elevations,
                  np.concatenate([n[0][::-1], (center_n,), n[1]]), float(spacing), int(full_half), rows, cols)


def valley_windows(configs: Configs, grid: Grid, cells: Cells, sections: list) -> np.ndarray:
    """How many cells each way along the stream each cell's valley is taken over (arc.xsection.valley.valley_axes):
    Valley_Dir_Widths of its reach's channel widths, the median of the bank-to-bank widths of the reach's usable
    cross sections with banks, in cells along the stream (at least 2); or Valley_Dir_Dist, for a reach without banks
    or with Valley_Dir_Widths 0. A meander runs down the valley about 10 to 14 channel widths, so a window a fixed
    number of cells long is a whole meander on a small river and part of one on a big river: on synthetic meanders of
    natural proportions, 12 to 16 widths did as well as the best fixed window for each."""
    windows = np.full(cells.count, float(configs.valley_dir_dist))
    if not configs.valley_dir_widths > 0.0:
        return windows
    
    widths = np.array([s.xs.left_bank_distance + s.xs.right_bank_distance
                       if s is not None and s.usable and s.xs.left_bank_distance > 0.0
                       and s.xs.right_bank_distance > 0.0 else np.nan for s in sections])
    cell = math.sqrt((grid.dx ** 2 + grid.dy ** 2) / 2.0)
    order = np.argsort(cells.reaches, kind="stable")
    starts = np.flatnonzero(np.r_[True, cells.reaches[order][1:] != cells.reaches[order][:-1]])
    for begin, end in zip(starts, np.r_[starts[1:], order.size]):
        ks = order[begin:end]
        known = widths[ks][np.isfinite(widths[ks])]
        if known.size:
            windows[ks] = max(configs.valley_dir_widths * float(np.median(known)) / cell, 2.0)

    return windows


def attach_valleys(configs: Configs, grid: Grid, cells: Cells, sections: list[Section], mannings_n: ManningsN, slopes: np.ndarray) -> int:
    """With Valley_Sections, give each usable cross section with banks (not a manual one) overbanks square to its
    valley (sample_valley), where the valley turns from it or the stream winds (arc.xsection.valley.worth_splitting),
    with the valley's slope over its own (slope_ratio), and say how many. The left line runs out on the cross
    section's left side."""
    windows = valley_windows(configs, grid, cells, sections)
    directions, sinuosities, mean_slopes = valley_axes(cells.rows, cells.cols, cells.reaches, grid.dx, grid.dy,
                                                       windows, slopes)
    count = 0
    for k, section in enumerate(sections):
        if section is None or not section.usable or section.manual:
            continue

        xs = section.xs
        if not (xs.left_bank_distance > 0.0 and xs.right_bank_distance > 0.0):
            continue

        direction, sinuosity = float(directions[k]), float(sinuosities[k])
        if not worth_splitting(section.direction, direction, sinuosity):
            continue

        if math.cos(direction - section.direction) < 0.0:
            direction += math.pi

        xs.valley = sample_valley(grid, mannings_n, section, direction, sinuosity,
                                  slope_ratio(float(slopes[k]), float(mean_slopes[k]), sinuosity))
        count += 1

    return count


# --- 7. Bathymetry ---------------------------------------------------------------------------------------------------


def unresolved_banks(xs: XSection, target_width: float) -> Banks:
    """The banks of a channel that neither the DEM nor its reach resolved: as wide as its width prior if it has
    one, and otherwise one cell wide."""
    if math.isfinite(target_width) and target_width > 0.0:
        banks = banks_for_width(xs, target_width)
        if banks.valid:
            return banks
        
    return single_cell_banks(xs)


def bank_ceilings(sections: list[Section], ks, max_flow, slopes, roughness: DepthRoughness | None,
                  slope_factor: float, resampler: Resampler | None = None) -> np.ndarray:
    """The ceilings of cross sections ks (see arc.bathymetry.smoothing.below_ceilings): the water surface that carries
    each one's maximum flow over its cross section as the DEM has it, before any carving and without banks. NaN, for
    no ceiling, where the banks are the water's (water mask or land cover) or not valid, or where there's no maximum
    flow or slope to carry, or the cross section never carries it. A resampler extends a trimmed cross section that
    doesn't reach above its ceiling, and it's worked out again."""
    ceilings = np.full(len(ks), math.nan)
    for position, k in enumerate(ks):
        section = sections[k]
        q, slope = float(max_flow[k]), float(slopes[k])
        if section.banks.valid and section.banks.method not in WATER_BANKS and q > 0.0 and slope > 0.0:
            while True:
                ceilings[position] = wse_for_discharge(section.xs, q, slope, roughness=roughness,
                                                       slope_factor=slope_factor)
                if resampler is None or not resampler.reach_above(section, ceilings[position]):
                    break
    return ceilings


def apply_bathymetry(configs: Configs, grid: Grid, cells: Cells, sections: list[Section], network, slopes, baseflow,
                     target_depth, target_width, water_n: float, *, stream_cells=None, outlets=None, max_flow=None,
                     roughness: DepthRoughness | None = None, resampler: Resampler | None = None) -> SparseRaster:
    """Smooth the banks, find, smooth and carve each channel, and return the filled bathymetry raster (as a
    SparseRaster).

    With `Bathy_Use_Banks`, the sections of a reach given no bank elevation become unusable (which happens only where
    the whole site has no bank observation). A channel whose banks are still unresolved after the width filter is
    carved at its width prior, or one cell wide (unresolved_banks). The depth that carries the baseflow is solved on
    the carved cross section with the rating curves' roughness and `slope_adjustment_factor` (bathymetry_depth), and
    water_n, the water class's Manning's n, is the n the channel has between its banks, so the cross section is
    given it there: between the banks the hydraulics use, which the width filter may have moved from the banks
    found (run() gives it between those). `stream_cells` and outlets help order the cross sections of a reach none of
    whose neighbours has any (see order_reach). With max_flow, each cell's maximum flow, and the rating curves'
    roughness, a bank from the DEM above the water that carries the maximum flow isn't an observation of the bank
    elevation (bank_ceilings). A resampler extends trimmed cross sections where they're too short for a step.

    With `Bathy_Use_Banks`, a cross section whose banks weren't found (none, or a single cell from the width prior) and
    whose baseflow is below Bathy_Bankless_Max_Baseflow is carved to carry the baseflow at its stream cell's level,
    not at its reach's bank elevation. It has no bank of its own: the elevation is its neighbours' bank height above
    its stream, and where a creek is too narrow for the DEM to show, the ground beside it can lie below that, so the
    flood maps spread the baseflow away from it.
    """
    if network is None:
        raise ValueError("Bathymetry needs the stream network, from StrmShp_File, reach_id and downstream_reach_id.")
    
    use_banks = configs.bathy_use_banks
    by_reach: dict[int, list[int]] = {}
    for k, section in enumerate(sections):
        if section is not None:
            by_reach.setdefault(int(cells.reaches[k]), []).append(k)

    if resampler is not None and configs.bathy_smooth_banks:
        # The width filter may give a cross section banks for its reach's median width
        for ks in by_reach.values():
            widths = reach_widths([sections[k].banks for k in ks])
            if widths is not None:
                for k in ks:
                    resampler.hold_width(sections[k], widths.median)

    reaches = {reach: ReachSections(cells.rows[ks], cells.cols[ks], [sections[k].xs for k in ks],
                                    [sections[k].banks for k in ks],
                                    None if max_flow is None else bank_ceilings(
                                        sections, ks, max_flow, slopes, roughness, configs.slope_adjustment_factor,
                                        resampler))
               for reach, ks in by_reach.items()}
    
    smoothed = smooth_bank_elevations(network, reaches, grid.dx, grid.dy, smooth=configs.bathy_smooth_banks,
                                      stream_cells=stream_cells, outlets=outlets)

    for reach, ks in list(by_reach.items()):
        result = smoothed[reach]
        if use_banks and not np.isfinite(result.bank_elevations).any():
            LOG.warning(f"Marking reach_id {reach} as unusable because it has no finite bank observations and no "
                        f"network-smoothed fallback. Removed {len(ks)} sampled records from downstream bathymetry "
                        "and rating-curve generation.")
            for k in ks:
                sections[k].usable = False
            del by_reach[reach], reaches[reach]
            continue

        for position, k in enumerate(ks):
            banks = result.banks[position]
            if not banks.valid and resampler is not None:
                resampler.hold_width(sections[k], float(target_width[k]))

            sections[k].carve_banks = banks if banks.valid else unresolved_banks(sections[k].xs, float(target_width[k]))
            sections[k].hydraulic_banks = dividing_banks(sections[k], sections[k].carve_banks)
            if use_banks and _carved_to_its_stream(sections[k], float(baseflow[k]), configs.bathy_bankless_max_baseflow):
                result.bank_elevations[position] = _centre(sections[k])

            sections[k].bank_elevation = float(result.bank_elevations[position])
            set_in_bank_roughness(sections[k].xs, sections[k].hydraulic_banks, water_n)

    depths, applies = {}, {}
    for reach, ks in by_reach.items():
        depths[reach], applies[reach] = [], []
        for k in ks:
            section = sections[k]
            section.carve = "none"
            depth = bathymetry_depth(section.xs, section.carve_banks, float(baseflow[k]), float(slopes[k]),
                                     trapezoid_height=configs.bathy_trap_h,
                                     target_depth=None if math.isnan(target_depth[k]) else float(target_depth[k]),
                                     mannings_n=water_n, roughness=roughness,
                                     slope_factor=configs.slope_adjustment_factor,
                                     bank_elevation=section.bank_elevation if use_banks else None,
                                     divide=section.hydraulic_banks)
            depths[reach].append(depth.depth)
            applies[reach].append(depth.apply)
            if depth.apply:
                section.carve = "depth_law" if depth.source == "target_depth" else "baseflow"
                if use_banks and _carved_to_its_stream(section, float(baseflow[k]), configs.bathy_bankless_max_baseflow):
                    section.carve = "stream_level"

            if depth.apply and depth.source == "baseflow_manning" and 0.0 < depth.depth < math.inf:
                # The level the baseflow fills the channel to: the carve's reference (see smooth_channel_depths)
                section.baseflow_level = section.bank_elevation if use_banks else _centre(section)

    # Without bed smoothing, a running median of one cross section and no cap leave each bed as it is
    smooth_bed = configs.bathy_smooth_bed
    channels = smooth_channel_depths(network, reaches, {reach: smoothed[reach] for reach in reaches}, depths,
                                     grid.dx, grid.dy, use_banks=use_banks,
                                     max_bed_grade=MAX_BED_GRADE if configs.bathy_bed_cap and smooth_bed else None,
                                     window=MEDIAN_WINDOW if smooth_bed else 1)

    carve = {}
    for reach, ks in by_reach.items():
        for position, k in enumerate(ks):
            if applies[reach][position]:
                carve[k] = float(channels[reach].depths[position])
                sections[k].carve_depth = carve[k]

    burns = Burns(grid.dem.shape)
    for k in sorted(carve):
        section = sections[k]
        changed = carve_channel(section.xs, section.carve_banks, carve[k], trapezoid_height=configs.bathy_trap_h,
                                bank_elevation=section.bank_elevation if use_banks else None,
                                keep_ground=configs.subcell_channels)
        burns.add(section.rows, section.cols, section.xs.elevations, changed)

    # The raster held as the cells it has bathymetry at until the run is done with the DEM (see run)
    return fill_sparse_gaps(burns.burned(), None if use_banks else grid.dem)


def _carved_to_its_stream(section: Section, baseflow: float, max_baseflow: float) -> bool:
    """Whether a cross section carries its baseflow at its stream cell's level (see apply_bathymetry): its banks
    weren't found, and its baseflow is below max_baseflow."""
    return section.banks is not None and section.banks.method in ("none", "single_cell") and baseflow < max_baseflow


# --- 8. Rating curves ------------------------------------------------------------------------------------------------


def _curve_top(curve: RatingCurve | None, xs: XSection, q_max: float, slope: float, increments: int,
               roughness: DepthRoughness | None, slope_factor: float) -> float:
    """The highest water surface rating_curve worked a cross section's hydraulics out at: the one that carries its
    maximum flow, or where the discharge can fall, an increment above it (see arc.rating_curve)."""
    if not (q_max > 0.0 and slope > 0.0):
        return -math.inf
    
    if curve is None:
        return max_flow_wse(xs, q_max, slope, roughness=roughness, slope_factor=slope_factor)
    
    if increments <= 0:
        return curve.max_wse
    
    profile = hydraulic_profile(xs)
    thalweg = float(profile.elevations[profile.center])
    return thalweg + (curve.max_wse - thalweg) / increments * (increments + 1)


def rating_curves(configs: Configs, grid: Grid, cells: Cells, sections: list, slopes, baseflow, max_flow,
                  roughness: DepthRoughness | None, quiet: bool = True,
                  resampler: Resampler | None = None) -> RatingCurves:
    """Each cell's rating curve and metadata, NaN where it has none. The elevation is that of the centre of the cross
    section the rating curve is worked out on: the carved bed with bathymetry, and otherwise the DEM's (or a manual
    cross section's own). A resampler extends a trimmed cross section that doesn't reach above the water its rating
    curve was worked out at, and it's worked out again."""
    increments = int(configs.vdt_database_numiterations)
    # With Valley_Sections, each cell's valley slope too (see arc.outputs.rating_curves)
    curves = RatingCurves.empty(cells.count, max(increments, 0), valley_slopes=configs.valley_sections)
    reach_average = configs.reach_average_curve_file
    for k in tqdm.tqdm(range(cells.count), disable=quiet):
        section = sections[k]
        comid = float(cells.comids[k])
        if section is None or not section.usable:
            if reach_average:
                # A cell without a usable cross section still gets metadata for the reach-average curve file
                if section is None:
                    row, col, xs_angle = int(cells.rows[k]), int(cells.cols[k]), np.nan
                    elevation = float(grid.dem[row, col])
                else:
                    row, col, xs_angle, elevation = section.row, section.col, section.xs_angle, _centre(section)

                curves.metadata[k, _METADATA_BUT_BASEFLOW] = [comid, row, col, elevation, 0.0, xs_angle]

            continue

        elevation = _centre(section)
        while True:
            curve = rating_curve(section.xs, float(max_flow[k]), float(slopes[k]), increments,
                                 baseflow=float(baseflow[k]), roughness=roughness,
                                 slope_factor=configs.slope_adjustment_factor)
            if resampler is None or not resampler.reach_above(section, _curve_top(
                    curve, section.xs, float(max_flow[k]), float(slopes[k]), increments, roughness,
                    configs.slope_adjustment_factor)):
                break

        if curve is not None and curve.valid:
            curves.increments[k] = curve.increments
            curves.metadata[k] = [comid, section.row, section.col, elevation, baseflow[k], slopes[k],
                                  section.xs_angle]
            if curves.valley_slopes is not None:
                curves.valley_slopes[k] = valley_slope(section, float(slopes[k]))
        elif reach_average:
            curves.metadata[k, _METADATA_BUT_BASEFLOW] = \
                [comid, section.row, section.col, elevation, slopes[k], section.xs_angle]
            
    return curves


def valley_slope(section: Section, slope: float) -> float:
    """The slope a section's overbanks flow at, given its stream's: times its slope ratio where Valley_Sections took
    them square to the valley (arc.xsection.valley), and the stream's own where they follow the stream."""
    valley = section.xs.valley
    return slope * valley.slope_ratio if valley is not None else slope


def _centre(section: Section) -> float:
    """The elevation of the centre of a section's cross section, as the hydraulics use it."""
    return float(section.xs.elevations[section.xs.elevations.size // 2])


# --- Cross sections the raster's edge cuts off (Drop_Edge_Sections) ------------------------------------------------


def drop_edge_curves(configs: Configs, grid: Grid, sections: list[Section], curves: RatingCurves,
                     resampler: Resampler | None = None) -> int:
    """
    Drop the rating curve of each cell whose water, at the top of its rating curve, reaches the raster's edge, or
    comes within Edge_Margin metres of it (reaches_edge), and say how many it dropped. The edge cuts such a cross
    section off, a wall where the ground would go on, so its water depends on where the raster ends: the overlapping
    rasters of two neighbouring tiles give the same cell different water, which shows as a seam where their maps meet.
    The water surface rises with the flow, so it's at the top that it spreads widest.
    """
    wse = curves.increments[:, :, WSE]
    rated = np.flatnonzero(np.isfinite(wse[:, -1]))
    tops = np.nanmax(wse[rated], axis=1)  # the top of each rated cell's curve: its last water is a number
    margin = float(configs.edge_margin)
    dropped = 0
    for k, top in zip(rated.tolist(), tops.tolist()):
        section = sections[k]
        if section is None or not section.usable:
            continue

        if resampler is not None:
            resampler.reach_above(section, top)

        if reaches_edge(section, top, grid, margin):
            section.dropped_at_edge = True
            curves.increments[k] = np.nan
            curves.metadata[k] = np.nan
            if curves.valley_slopes is not None:
                curves.valley_slopes[k] = np.nan

            dropped += 1

    return dropped


def reaches_edge(section: Section, wse: float, grid: Grid, margin: float = 0.0) -> bool:
    """
    Whether a cross section's water at a water surface reaches the raster's edge, or comes within margin metres of
    it on the map. Walking out from its centre on each side, the water reaches the edge where it covers the profile to
    its end (a cross section reaches the raster's edge at most) or meets a wall off the raster; a wall on the raster
    is a cell without data, not the edge.
    """
    profile = hydraulic_profile(section.xs)
    stations, elevations, center = profile.stations, profile.elevations, profile.center
    if not wse > elevations[center]:
        return False
    
    rows, cols = grid.dem.shape
    for step in (-1, 1):
        end = elevations.size - 1 if step > 0 else 0
        j = _water_end(elevations, center, float(wse), step)
        if j == end:
            return True
        
        z_in, z_out = float(elevations[j]), float(elevations[j + step])
        if z_out >= OFF_RASTER_ELEVATION:
            x, y = _station_position(section, float(stations[j + step]), grid.dx, grid.dy)
            if not (0.0 <= x <= cols - 1 and 0.0 <= y <= rows - 1):
                return True
            
        if margin > 0.0:
            # The water's edge, part way along the segment, on the map
            s0, s1 = float(stations[j]), float(stations[j + step])
            x, y = _station_position(section, s0 + (s1 - s0) * (wse - z_in) / (z_out - z_in), grid.dx, grid.dy)
            if min(min(x, cols - 1 - x) * grid.dx, min(y, rows - 1 - y) * grid.dy) <= margin:
                return True
            
    return False


@njit(cache=True, error_model="numpy")
def _water_end(elevations, center, level, step):
    """The last vertex out from a profile's centre, one way (step -1 or 1), that water at a level covers to."""
    end = elevations.size - 1 if step > 0 else 0
    j = center
    while j != end and level > elevations[j + step]:
        j += step

    return j


def _station_position(section: Section, station: float, dx: float, dy: float) -> tuple[float, float]:
    """The column and row of a point a station (metres, negative to the left) along a section's hydraulic profile:
    on its own cross section, or beyond a bank on the line out from it square to the valley, where Valley_Sections
    gave it one (see hydraulic_profile and arc.xsection.valley.splice)."""
    xs = section.xs
    own = section.direction - math.pi / 2
    cos, sin = math.cos(own), math.sin(own)
    left, right = float(xs.left_bank_distance), float(xs.right_bank_distance)
    valley = xs.valley
    if valley is not None and left > 0.0 and right > 0.0 and (station < -left or station > right):
        across = valley.direction - math.pi / 2
        cos_v, sin_v = math.cos(across), math.sin(across)
        if station < -left:
            beyond = -station - left
            return (section.col - cos * left / dx - cos_v * beyond / dx, section.row - sin * left / dy - sin_v * beyond / dy)
        
        beyond = station - right
        return (section.col + cos * right / dx + cos_v * beyond / dx, section.row + sin * right / dy + sin_v * beyond / dy)
    
    return section.col + cos * station / dx, section.row + sin * station / dy


# --- 9. Outputs -----------------------------------------------------------------------------------------------------


def cross_section_record(section: Section, comid: int) -> CrossSectionRecord:
    """A section's row of the cross-section file: each side from its centre out to its last ordinate on the raster,
    side 1 towards the higher rows (see _side_one)."""
    elevations, n = section.xs.elevations, section.xs.mannings_n
    center = elevations.size // 2
    sides = []
    for step in (section.side_one, -section.side_one):
        index = center + step * np.arange(1 + _on_raster_ordinates(elevations, center, step))
        sides.append((elevations[index].astype(np.float64), n[index].astype(np.float64), int(section.rows[index[-1]]),
                      int(section.cols[index[-1]])))
        
    (xs1, n1, r1, c1), (xs2, n2, r2, c2) = sides
    return CrossSectionRecord(int(comid), section.row, section.col, xs1, float(section.xs.ordinate_distance), n1, xs2,
                              n2, r1, c1, r2, c2)


def write_outputs(configs: Configs, grid: Grid, results: Results, flows: pd.DataFrame | None) -> None:
    """Write the outputs the configuration names: the rasters on a thread of their own while the tables are written,
    since both wait mostly on compression outside Python."""
    with ThreadPoolExecutor(1) as pool:
        rasters = pool.submit(write_rasters, configs, grid, results)
        write_tables(configs, results, flows)
        rasters.result()


def write_tables(configs: Configs, results: Results, flows: pd.DataFrame | None) -> None:
    """Write the tables the configuration names, each Parquet file with the run's metadata (run_metadata)."""
    metadata = run_metadata(configs)
    curves = results.curves
    if curves is not None:
        if not curves.has_vdt_data() and configs.vdt_database_numiterations > 0:
            LOG.warning('No VDT data was generated, so no hydraulic output files will be created.')
        else:
            if configs.print_vdt_database:
                results.vdt = write_vdt(curves, configs.print_vdt_database, metadata)

            if configs.print_ap_database:
                write_ap(curves, configs.print_ap_database, metadata)

            qmax = flows[configs.flow_file_qmax] if flows is not None and configs.flow_file_qmax in flows.columns else pd.Series(dtype=np.float64)
            if configs.reach_average_curve_file:
                vdt = results.vdt if results.vdt is not None else vdt_dataframe(curves)
                write_reach_average_curve_file(curves, vdt, qmax, configs.print_curve_file, metadata)
            elif configs.print_curve_file:
                write_curve_file(curves, qmax, configs.print_curve_file, metadata)

    if configs.xs_out_file:
        write_cross_sections(results.cross_sections, configs.xs_out_file, metadata)

    if configs.build_representative_cross_section and configs.representative_cross_section_file:
        write_representative_cross_sections(results.representative, configs.representative_cross_section_file,
                                            metadata)

    if configs.print_decision_log:
        write_decision_log(results, configs.print_decision_log, metadata)


def write_rasters(configs: Configs, grid: Grid, results: Results) -> None:
    """Write the bathymetry raster, if the configuration names it."""
    if results.bathymetry is not None:
        Raster.write_array(results.bathymetry, grid.reference, configs.bathy_out_file, dtype=np.float32,
                           compression=configs.compression, metadata=run_metadata(configs))


# --- The whole run -------------------------------------------------------------------------------------------------


@contextmanager
def run_log(configs: Configs):
    """With Log_File, ARC's messages at the info level and above also go to that file while the run lasts, whatever
    the console shows, after the ARC version, the inputs (as JSON) and what was worked out from them."""
    if not configs.log_file:
        yield
        return

    from arc import __version__
    path = Path(configs.log_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(path, mode="w", encoding="utf-8")
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging_format)
    level = LOG.level
    LOG.addHandler(handler)
    LOG.setLevel(min(level, logging.INFO) if level else logging.INFO)
    try:
        LOG.info(f"ARC {__version__}")
        LOG.info("Inputs: " + json.dumps(configs.as_mapping(), default=str))
        LOG.info("Worked out from the inputs: " + ", ".join(f"{f.name} {getattr(configs, f.name)}"
                                                            for f in fields(configs) if not f.init))
        yield
    finally:
        LOG.removeHandler(handler)
        handler.close()
        LOG.setLevel(level)


def run(configs: Configs, *, quiet: bool = False, write: bool = True, trim: bool = True) -> Results:
    """Run ARC for a configuration, write the outputs it names, and return what it made. With Log_File, the run's
    messages also go to that file (run_log).

    Parameters
    ----------
    configs : Configs
        The inputs.
    quiet : bool
        Hide the progress bars.
    write : bool
        Write the outputs the inputs name. With False, the run only returns them.
    trim : bool
        Fit each cross section to the water the later steps are expected to work at, and extend it when a step needs
        it longer (see the module's notes on cross section lengths). With False, every cross section reaches the
        raster's edge on both sides: the results are the same, bit for bit, but slower.

    Returns
    -------
    Results
        What the run made.
    """
    with run_log(configs):
        return _run(configs, quiet=quiet, write=write, trim=trim)


def _run(configs: Configs, *, quiet: bool, write: bool, trim: bool) -> Results:
    """run, without its log file."""
    started = time.perf_counter()
    flows = read_flows(configs)
    grid = read_grid(configs)
    gdf = read_streamfile(configs)
    network = stream_network(configs, gdf)
    targets = power_law_targets(configs, gdf)
    manual = read_manual_sections(configs) if configs.manual_cross_sections_file else None

    # Each reach's stream cells, found once for the steps that look for them
    stream_reaches = stream_cells_by_reach(cells=grid.stream_raster_cells())
    cells = stream_cells(grid, flows, manual)
    slopes, keep, slope_sources = cell_slopes(configs, grid, cells, network, gdf, stream_reaches)
    if not keep.any():
        raise ValueError("No stream cells remain: no reach has a slope (see the warning above).")

    skipped = cells.subset(~keep) if not keep.all() else None
    cells, slopes, slope_sources = cells.subset(keep), slopes[keep], slope_sources[keep]
    baseflow, max_flow = cell_flows(configs, flows, cells)
    target_depth, target_width = cell_targets(targets, cells)

    # The stream cells are water, whatever the land cover and the water mask say
    grid.land_cover[cells.rows, cells.cols] = configs.lc_water_value
    if grid.water_mask is not None:
        grid.water_mask[cells.rows, cells.cols] = 1

    lookup, water_n = mannings_n_table(configs)
    mannings_n = ManningsN(grid.land_cover, lookup, water_n)
    roughness = DepthRoughness(configs.k_decay, configs.shallow_factor, configs.deep_factor) \
        if configs.depth_varying_n else None

    resampler = Resampler(grid, mannings_n)
    # Each cell's cross section and banks, and then the cross section trimmed
    offsets = angle_offsets(configs.degree_manip, configs.degree_interval)
    water, water_value = bank_water(configs, grid)
    at = banks_at(configs)
    sections: list[Section | None] = []
    for k in tqdm.tqdm(range(cells.count), disable=quiet):
        if manual:
            section = manual_section(manual[int(cells.comids[k])])
        else:
            section = sample_section(configs, grid, int(cells.rows[k]), int(cells.cols[k]), offsets, whole=not trim,
                                     reach_cells=stream_reaches[int(cells.reaches[k])])
            
        if section is not None:
            width = None if math.isnan(target_width[k]) else float(target_width[k])
            # The bank tops are no higher than the water that carries the maximum flow; every mode finds them, without
            # Bathy_Use_Banks for the hydraulics alone
            ceiling = bank_ceiling(configs, section, grid, mannings_n, resampler if trim and not section.manual else None,
                                   float(max_flow[k]), float(slopes[k]), roughness)
            along = water_along(section, water, grid)
            while not reaches_its_banks(along, water_value) and \
                    resampler.extend(section, 2 * (section.xs.elevations.size // 2)):
                along = water_along(section, water, grid)

            section.banks = find_banks(section.xs, target_width=width, land_cover=along,
                                       water_value=water_value if along is not None else None, at=at,
                                       ceiling=ceiling)
            if at == "water":
                section.bank_tops = find_banks(section.xs, target_width=width, land_cover=along,
                                               water_value=water_value if along is not None else None, at="divide",
                                               ceiling=ceiling)
            section.hydraulic_banks = dividing_banks(section, section.banks)
            if trim and not section.manual:
                fit_section(configs, section, grid, mannings_n, resampler, float(max_flow[k]), float(slopes[k]),
                            roughness, float(target_width[k]))
                
        sections.append(section)

    # The water's n between each cross section's hydraulic banks, put in the raster so that the cross sections crossing
    # it get it too, and then each cross section's n
    for section in sections:
        if section is not None:
            wet = in_bank(section.xs, section.hydraulic_banks)
            mannings_n.water_between_banks(section.rows[wet], section.cols[wet])

    for section in sections:
        if section is not None:
            sample_roughness(section, grid, mannings_n)
            set_in_bank_roughness(section.xs, section.hydraulic_banks, water_n)

    if not configs.valley_sections:
        # Only the valleys' overbanks read the water mask again: it goes before the steps that take the most memory
        grid = grid._replace(water_mask=None)
        resampler.grid, water = grid, None

    resampler = resampler if trim else None

    # Where every reach lies, which orders the cross sections of a reach none of whose neighbours has any
    outlets = None
    if network is not None and (configs.bathy_out_file or configs.standard_step):
        outlets = line_outlets(configs, grid, gdf, cells.reaches)
    else:
        stream_reaches = None

    results = Results(configs, cells, slopes, baseflow, max_flow, sections, slope_sources=slope_sources,
                      skipped=skipped)
    bathymetry = None
    if configs.bathy_out_file:
        bathymetry = apply_bathymetry(configs, grid, cells, sections, network, slopes, baseflow, target_depth,
                                      target_width, water_n, stream_cells=stream_reaches, outlets=outlets,
                                      max_flow=max_flow, roughness=roughness, resampler=resampler)
        
    usable = [(k, s) for k, s in enumerate(sections) if s is not None and s.usable]
    for _, section in usable:
        set_bank_distances(section.xs, section.hydraulic_banks)

    if configs.subcell_channels:
        made = reconstruct_all([section.xs for _, section in usable if not section.manual])
        LOG.info(f"Subcell_Channels: {made} of {len(usable)} cross sections have their narrow channel reconstructed "
                 "below the cell scale")

    if configs.valley_sections:
        split = attach_valleys(configs, grid, cells, sections, mannings_n, slopes)
        LOG.info(f"Valley_Sections: {split} of {len(usable)} cross sections take their overbanks square to the valley")

    if configs.makes_rating_curves:
        results.curves = rating_curves(configs, grid, cells, sections, slopes, baseflow, max_flow, roughness, quiet,
                                       resampler)
        dropped = 0
        if configs.drop_edge_sections:
            # Before the standard step too, which would otherwise march the water through them
            dropped = drop_edge_curves(configs, grid, sections, results.curves, resampler)

        if configs.standard_step:
            elevations = grid.elevations(cells.rows, cells.cols)
            levels = np.array([math.nan if s is None else s.baseflow_level for s in sections])
            step = apply_standard_step(network, cells.rows, cells.cols, cells.reaches, sections, slopes, max_flow,
                                          baseflow, results.curves, grid.dx, grid.dy, roughness=roughness,
                                          slope_factor=configs.slope_adjustment_factor,
                                          eddy_length=configs.standard_step_eddy_length, stream_cells=stream_reaches,
                                          outlets=outlets, elevations=np.where(elevations >= OFF_RASTER_ELEVATION,
                                                                               np.nan, elevations),
                                          baseflow_levels=levels,
                                          ensure_level=None if resampler is None else
                                          lambda k, level, sampler=resampler: sampler.reach_above(sections[k], level),
                                          outlet_wse=outlet_levels(configs, grid))
            results.step = step
            LOG.info(f"Standard step: {step.cells} rating curves linked, {step.critical} of {step.steps} "
                     f"steps at critical depth, {step.held} water surfaces held at the baseflow level"
                     + (f", {step.given} outlets started from Outlet_WSE_File" if configs.outlet_wse_file else ""))
            if configs.drop_edge_sections:
                # The step's water can spread wider than the normal depths'
                dropped += drop_edge_curves(configs, grid, sections, results.curves, resampler)

        if configs.drop_edge_sections:
            LOG.info(f"Drop_Edge_Sections: {dropped} rating curves dropped, their water reaching the raster's edge"
                     + (f" or within {configs.edge_margin:g} m of it" if configs.edge_margin > 0 else ""))
            
    if configs.xs_out_file:
        results.cross_sections = [cross_section_record(s, cells.comids[k]) for k, s in usable]

    if configs.build_representative_cross_section:
        if resampler is not None:
            for _, s in usable:
                resampler.reach_above(s, highest_stage(_centre(s)))

        # With the standard step, each rated cross section's discharges are its rating curve's
        stepped = results.curves.increments if configs.standard_step and results.curves is not None else None
        samples = [RepresentativeSample(int(cells.comids[k]), s.xs, _centre(s), float(slopes[k]),
                                        stepped[k][:, [WSE, Q]] if stepped is not None
                                        and np.isfinite(stepped[k, -1, Q]) else None) for k, s in usable]
        results.representative = representative_cross_section_dataframe(
            samples, roughness=roughness, slope_factor=configs.slope_adjustment_factor)

    # Done with the input rasters: they go before the bathymetry raster is made, so that it isn't in memory with them
    del resampler, mannings_n, water
    grid = grid._replace(dem=None, streams=None, land_cover=None, water_mask=None, stream_cells=None, stream_flat=None)
    if bathymetry is not None:
        results.bathymetry = bathymetry.to_array()

    results.qa_flags = qa_flags(results)
    if results.curves is not None:
        results.curves = results.curves._replace(qa_flags=results.qa_flags)
        
    LOG.info(summary(results))

    if write:
        write_outputs(configs, grid, results, flows)

    results.seconds = time.perf_counter() - started
    LOG.info(f"Simulation Took {results.seconds:.1f} seconds")
    return results


def main(argv: list[str] | None = None) -> Results:
    """The command line: python -m arc.pipeline ARC_Input_File.yaml [--quiet]."""
    parser = argparse.ArgumentParser(description="Run ARC")
    parser.add_argument("config", help="ARC's input file, YAML or tab-separated text")
    parser.add_argument("-q", "--quiet", action="store_true", help="Hide the progress bars")
    args = parser.parse_args(argv)
    return run(Configs.from_file(args.config), quiet=args.quiet)


if __name__ == "__main__":
    main()
