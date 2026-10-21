"""ARC's main workflow, from its input rasters to its rating curves, cross sections and bathymetry.

    from arc.config import Configs
    from arc.pipeline import run
    results = run(Configs.from_file("ARC_Input_File.yaml"))

or from the command line, ``python -m arc.pipeline ARC_Input_File.yaml`` (or ``arc ARC_Input_File.yaml``).

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
6. Banks (arc.bathymetry.find_banks): where the water mask's water ends, if there is a mask, or where the land cover's
   does, with FindBanksBasedOnLandCover, and otherwise from the DEM. Then each cross section is trimmed to what the
   later steps are expected to need (see below), and the water's n goes between the banks.
7. Bathymetry, if BATHY_Out_File is set (arc.bathymetry): each reach's banks and bank elevations smoothed, and the
   bank elevations ramped to meet at each junction on a main stem (unless Bathy_Smooth_Banks is off), each channel's depth found, the depths filled in and the beds smoothed along the network
   (unless Bathy_Smooth_Bed is off, as it is by default with Standard_Step or Bathy_Use_Banks), and the channel carved
   into each cross section and burned into the bathymetry raster, whose gaps are then filled. Each cross section's
   centre cell gets its own carved bed, the Elev of its rating curve, and other cells where cross sections overlap get
   their mean (see arc.bathymetry.raster). A channel whose banks nothing resolved is carved as wide as its width
   prior, where it has one, and otherwise one cell wide. The water's n goes between the banks the channel is carved
   to, which the width filter may have moved. A bank from the DEM above the water that carries the cell's maximum
   flow is a valley wall, and isn't an observation of the bank elevation (bank_ceilings).
8. Rating curves (arc.rating_curve), linked along the stream network by the standard step with Standard_Step
   (arc.standard_step), and the representative cross sections (arc.outputs.representative) when they're asked for.
9. Outputs (arc.outputs), on the DEM's grid.

Representative cross sections and rating curves
-----------------------------------------------
A run asked for both makes them from the same cross sections, carved as they would be for the rating curves alone, so
that asking for the representative cross sections doesn't change the rating curves. A run that only builds
representative cross sections (one that asks for no output of the rating curves) makes no rating curves, and needs a
flow file only to carve its channels for the baseflow (see arc.config).

Cross section lengths
---------------------
A cross section has no set length (X_Section_Dist is no longer used): it reaches as far as its water needs, which can
be out to the raster's edge, beyond which the ordinates are walls (OFF_RASTER_ELEVATION). The hydraulics only look at
a cross section out to where its water ends, the first ground above the water on each side, so most of a cross section
that long is never used, and holding all of it for every cell would take memory in proportion to the raster's size.
So each one is first sampled only as far as finding its banks looks: on each side out to the first ground more than
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
import math
import time
from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np
import pandas as pd
import tqdm
from numba import njit

from arc import LOG
from arc.bathymetry import (Banks, ReachSections, banks_for_width, bathymetry_depth, burn_into_raster, carve_channel,
                            fill_bathymetry_gaps, find_banks, in_bank, ordinate_cells, reach_network,
                            set_bank_distances, set_in_bank_roughness, single_cell_banks, smooth_bank_elevations,
                            smooth_channel_depths)
from arc.bathymetry.banks import MAX_BANK_HEIGHT, _on_raster_ordinates
from arc.bathymetry.bed_smoothing import MAX_BED_GRADE, MEDIAN_WINDOW
from arc.bathymetry.depth import power_law_geometry
from arc.bathymetry.smoothing import WATER_BANKS, reach_widths
from arc.config import Configs
from arc.hydraulics import DepthRoughness, hydraulic_profile, wse_for_discharge
from arc.io import Raster, Vector
from arc.outputs import (CrossSectionRecord, RatingCurves, RepresentativeSample,
                         representative_cross_section_dataframe, vdt_dataframe, write_ap, write_cross_sections,
                         write_curve_file, write_reach_average_curve_file, write_representative_cross_sections,
                         write_vdt)
from arc.outputs.rating_curves import COL, COMID, ELEV, ROW, SLOPE, XS_ANGLE
from arc.outputs.representative import highest_stage
from arc.rating_curve import RatingCurve, max_flow_wse, rating_curve
from arc.standard_step import apply_standard_step
from arc.xsection.low_spot import low_spot_cell
from arc.xsection.orientation import angle_offsets, narrowest_direction, stream_direction
from arc.xsection.sampling import (OFF_RASTER_ELEVATION, _ordinates, ordinates_above, sample_classes,
                                   sample_elevations)
from arc.xsection.slope import (corrected_local_slopes, end_point_slope, fill_unresolved_reach_slopes,
                                local_average_slopes, local_fit_slopes, main_stem_above, reach_median_slope)
from arc.xsection.stream_path import along_stream_stations, stream_cells_by_reach
from arc.xsection.xsection import Profile, XSection

FLOODED_CELL = 3  # the flood raster's value at the centre cell of each cross section used
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
    dem: np.ndarray  # float64, with cells without data as walls
    # int32 if the raster's integer type fits it, and otherwise int64, each stream cell holding its reach's ID and
    # every other cell 0
    streams: np.ndarray
    land_cover: np.ndarray  # uint8
    dx: float  # the cells' width and height in metres
    dy: float
    water_mask: np.ndarray | None = None  # uint8, 1 where BathyWaterMask has water and 0 elsewhere; None without one


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
    hydraulic_banks: Banks | None = None  # as the hydraulics use them: after the width filter with bathymetry
    bank_elevation: float = math.nan  # the smoothed bank elevation, with bathymetry
    # The water surface its channel was carved to carry the baseflow at (its bank elevation with Bathy_Use_Banks, its
    # stream cell's otherwise), where bathymetry carved it for the baseflow; the standard step holds the baseflow there
    baseflow_level: float = math.nan
    usable: bool = True  # False once the bathymetry drops its reach
    manual: bool = False
    length: float = math.nan  # the length it's sampled to, which grows as steps extend it
    full_half: int = 0  # how many ordinates either side of its centre it has out to the raster's edge (0 if manual)


@dataclass
class Results:
    """What a run made. The per-cell arrays and sections are in the order of cells, a section None where its
    centre had no data."""
    configs: Configs
    cells: Cells
    slopes: np.ndarray
    baseflow: np.ndarray
    max_flow: np.ndarray
    sections: list
    curves: RatingCurves | None = None
    vdt: pd.DataFrame | None = None
    cross_sections: list = field(default_factory=list)
    representative: pd.DataFrame | None = None
    bathymetry: np.ndarray | None = None
    seconds: float = 0.0


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

    dem = reference.read_array(np.float64)
    no_data = ~np.isfinite(dem)
    if reference.nodata_value is not None:
        no_data |= dem == reference.nodata_value
    dem[no_data] = OFF_RASTER_ELEVATION

    stream_raster = rasters["stream"]
    streams = stream_raster.read_array()
    narrow = streams.dtype in (np.int8, np.int16, np.int32, np.uint8, np.uint16)
    streams = streams.astype(np.int32 if narrow else np.int64, copy=False)
    if stream_raster.nodata_value is not None and math.isfinite(stream_raster.nodata_value):
        streams[streams == int(stream_raster.nodata_value)] = 0
    streams[streams < 0] = 0

    land_cover = rasters["land cover"].read_array().astype(np.uint8)

    water_mask = None
    if "water mask" in rasters:
        mask = rasters["water mask"].read_array()
        if np.issubdtype(mask.dtype, np.integer):
            water = mask != 0  # compared as it is, without a float64 copy
        else:
            mask = mask.astype(np.float64, copy=False)
            water = np.isfinite(mask) & (mask != 0.0)
        if rasters["water mask"].nodata_value is not None:
            water &= mask != rasters["water mask"].nodata_value
        water_mask = water.astype(np.uint8)

    dx, dy = reference.cell_size_in_metres()
    LOG.info(f"Cell size {dx} m by {dy} m")
    return Grid(reference, dem, streams, land_cover, dx, dy, water_mask)


def read_stream_layer(configs: Configs):
    """The stream network's features, or None if there's no StrmShp_File."""
    if not configs.strmshp_file:
        return None
    return Vector(configs.strmshp_file).to_geopandas()


def stream_network(configs: Configs, layer):
    """The reach network from the stream layer. None without the layer, or without its reach_id and
    downstream_reach_id fields when neither the bathymetry nor the standard step needs them (they then only give
    slopes to the reaches the slope method can't)."""
    if layer is None:
        return None
    for column in (configs.reach_id, configs.downstream_reach_id):
        if not column or column not in layer.columns:
            if configs.bathy_out_file or configs.standard_step:
                raise ValueError(f"The stream network {configs.strmshp_file} has no field {column!r}.")
            LOG.warning(f"The stream network {configs.strmshp_file} has no field {column!r}, so a reach without a "
                        "slope can't take its neighbours'.")
            return None
    return reach_network(layer[configs.reach_id].tolist(), layer[configs.downstream_reach_id].tolist())


def line_outlets(configs: Configs, grid: Grid, layer, reaches) -> dict[int, tuple[int, int]]:
    """The cell (row, col) holding the end of each of these reaches' lines in the stream layer, which may be off the
    raster. Stream networks are drawn from upstream to downstream, so it's the downstream end, which orders a reach
    none of whose neighbours is on the stream raster (see order_reach). A reach's first feature counts, and of a
    multi-part line, the last part."""
    if layer is None or configs.reach_id not in layer.columns:
        return {}
    ids = pd.to_numeric(layer[configs.reach_id], errors="coerce")
    first = (ids.isin(np.unique(np.asarray(reaches, dtype=np.int64))) & ~ids.duplicated()).to_numpy()
    features = layer[first]
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


def power_law_targets(configs: Configs, layer) -> dict[int, tuple[float | None, float | None]]:
    """Each reach's drainage-area bankfull depth and width, where configured."""
    if not (configs.use_bathymetry_powerlaw or configs.use_bathymetry_powerlaw_width):
        return {}
    if layer is None:
        raise ValueError("The drainage-area power laws need a StrmShp_File.")
    field_name, reach_field = configs.drainage_area_field, configs.reach_id
    missing = sorted({reach_field, field_name} - set(layer.columns))
    if missing:
        raise KeyError("The stream vector dataset is missing the columns required for drainage-area bathymetry: "
                       + ", ".join(missing))
    table = layer[[reach_field, field_name]].dropna(subset=[reach_field, field_name])
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
    except Exception:
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


def stream_cells(configs: Configs, grid: Grid, flows: pd.DataFrame | None, manual: dict | None) -> Cells:
    """The cells to work on, in the stream raster's row order or the manual cross sections': those of the flow
    file's reaches, or without flows, every stream cell."""
    if flows is None:
        ids = np.unique(grid.streams)
        ids = ids[ids > 0]
    else:
        ids = np.asarray(flows.index, dtype=np.int64)
    if manual:
        shared = [int(i) for i in ids if int(i) in manual]
        if not shared:
            raise ValueError("No IDs were shared between the ARC flow file and the manual cross-section file.")
        return Cells(np.array([manual[i]["row"] for i in shared], dtype=np.int64),
                     np.array([manual[i]["col"] for i in shared], dtype=np.int64), np.array(shared, dtype=np.int64),
                     np.array([manual[i]["source_stream_id"] for i in shared], dtype=np.int64))
    rows, cols = np.nonzero(np.isin(grid.streams, ids))
    if rows.size == 0:
        raise ValueError("No stream cell has an ID in the flow file, so there is nothing to work on.")
    comids = grid.streams[rows, cols].astype(np.int64)
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
    values = local_average_slopes(grid.dem[rows, cols], rows, cols, stations, grid.dx, grid.dy, distance)
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
    z = grid.dem[rows, cols]
    values, reach_slope = local_fit_slopes(np.where(z >= OFF_RASTER_ELEVATION, np.nan, z), stations, distance)
    return _at_cells(values, have, cells, ks), reach_slope


def _dem_slopes(configs: Configs, grid: Grid, cells: Cells, by_reach: dict, network=None) -> tuple[dict, dict]:
    """By a method that measures slopes between stream cells: each reach's slope, NaN where the DEM gives none, and
    the slopes of the run's cells, by reach, NaN where a cell has none. The run's reaches get a slope, and if one of them
    gets none, which it then takes from its neighbours (cell_slopes), so does every other reach on the stream raster.

    With a stream network, the local averages take in the cells of the reach's main-stem neighbours too, so that a
    cell near a junction is averaged over both sides of it; the reach's percentiles and median slope are still its
    own cells'. The fitted lines (local_fit) stay within the reach."""
    method = configs.stream_slope_method
    distance = configs.gen_slope_dist
    stream_reaches = stream_cells_by_reach(grid.streams)
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
        median, lower, upper = reach_median_slope(grid.dem[rows, cols], rows, cols, stations, grid.dx, grid.dy,
                                                  distance, configs.slope_low_percentile,
                                                  configs.slope_high_percentile)
        reach_slopes[reach] = median
        cell_values[reach] = np.full(len(ks), median) if method == "reach_average" \
            else corrected_local_slopes(local, lower, upper)

    for reach in sorted(by_reach):
        measure(reach)
    if any(math.isnan(reach_slopes[reach]) for reach in by_reach):
        for reach in sorted(set(stream_reaches) - set(by_reach)):
            measure(reach)
    return reach_slopes, {reach: cell_values[reach] for reach in by_reach}


def _end_point_slopes(configs: Configs, grid: Grid, layer, by_reach: dict) -> dict:
    """Each of the run's reaches' end-point slope, NaN where it can't be measured, and if one of them has none, which it
    then takes from its neighbours, every other reach's on the stream raster."""
    if layer is None:
        raise ValueError("The 'end_points' stream slope method requires a strmshp_file.")
    id_field = configs.flow_file_id if configs.makes_rating_curves else configs.reach_id
    if id_field not in layer.columns:
        raise KeyError(f"The stream network {configs.strmshp_file} has no field {id_field!r}.")
    dem = np.where(grid.dem >= OFF_RASTER_ELEVATION, np.nan, grid.dem)
    ids = layer[id_field]
    slopes = {}

    def measure(reach: int) -> None:
        matches = np.flatnonzero((ids == reach).to_numpy())
        slope = math.nan
        if matches.size:
            feature = layer.iloc[matches[:1]]  # a reach's first feature counts
            length = float(feature.to_crs(feature.estimate_utm_crs()).length.iloc[0])
            slope = end_point_slope(feature.to_crs(grid.reference.crs).geometry.iloc[0], length, dem,
                                    grid.reference.geotransform)
        slopes[int(reach)] = slope

    for reach in sorted(by_reach):
        measure(reach)
    if any(math.isnan(slopes[int(reach)]) for reach in by_reach):
        for reach in sorted(set(np.unique(grid.streams[grid.streams > 0]).tolist()) - set(by_reach)):
            measure(reach)
    return slopes


def _listed(reaches) -> str:
    reaches = sorted(reaches)
    return ", ".join(map(str, reaches[:10])) + (" and others" if len(reaches) > 10 else "")


def cell_slopes(configs: Configs, grid: Grid, cells: Cells, network, layer) -> tuple[np.ndarray, np.ndarray]:
    """Each cell's slope by Stream_Slope_Method, and which cells to keep (see arc.xsection.slope). Whatever the
    method, a reach it can't give a slope takes its neighbours' in the stream network, and a cell without one takes
    its reach's. The cells of a reach left without a slope are skipped."""
    by_reach = _cells_by_reach(cells)
    if configs.stream_slope_method == "end_points":
        reach_slopes = _end_point_slopes(configs, grid, layer, by_reach)
        cell_values = {reach: np.full(len(ks), reach_slopes[reach]) for reach, ks in by_reach.items()}
    else:
        reach_slopes, cell_values = _dem_slopes(configs, grid, cells, by_reach, network)
    measured = {reach for reach, slope in reach_slopes.items() if not math.isnan(slope)}
    if network is not None:
        left = fill_unresolved_reach_slopes(network, reach_slopes)
    else:
        left = set(reach_slopes) - measured

    slopes = np.full(cells.count, np.nan)
    keep = np.ones(cells.count, dtype=bool)
    for reach, ks in by_reach.items():
        if reach in left:
            keep[ks] = False
        elif reach in measured:
            slopes[ks] = np.where(np.isnan(cell_values[reach]), reach_slopes[reach], cell_values[reach])
        else:
            slopes[ks] = reach_slopes[reach]  # its neighbours'
    filled = (set(by_reach) - measured) - left
    if filled:
        LOG.info(f"Reaches {_listed(filled)} have no slope of their own, and take their neighbours' in the stream "
                 "network.")
    skipped = left & set(by_reach)
    if skipped:
        hint = "" if network is not None else \
            " (give StrmShp_File, with reach_id and downstream_reach_id, to take them from the stream network)"
        LOG.warning(f"Skipping the {int(np.count_nonzero(~keep))} stream cells of reaches {_listed(skipped)}: neither "
                    f"the DEM nor their neighbours give them a slope{hint}.")
    return slopes, keep


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
                   whole: bool = False) -> Section | None:
    """A stream cell's cross section, with its elevations but not yet its roughness: centred on the low spot and turned
    to where the channel is narrowest, and out on each side to the first ground more than MAX_BANK_HEIGHT above its
    centre, or with whole, to the raster's edge on both sides. None if its centre has no data."""
    direction = stream_direction(grid.streams, row, col, configs.gen_dir_dist, grid.dx, grid.dy)
    if configs.low_spot_range > 0:
        row, col = low_spot_cell(grid.dem, row, col, direction, reach_length(grid, row, col), grid.dx, grid.dy,
                                 configs.low_spot_range)
    length = reach_length(grid, row, col)
    direction = narrowest_direction(grid.dem, row, col, direction, length, grid.dx, grid.dy, offsets)
    _, _, spacing, full_half = _ordinates(direction, length, grid.dx, grid.dy)
    # Out to the first ground more than MAX_BANK_HEIGHT above the centre on each side, as far as finding banks looks,
    # or with whole, to the raster's edge
    half = full_half if whole else ordinates_above(grid.dem, row, col, direction, length, grid.dx, grid.dy,
                                                   MAX_BANK_HEIGHT)
    length = _length_for(max(half, 1), spacing)
    elevations, spacing = sample_elevations(grid.dem, row, col, direction, length, grid.dx, grid.dy)
    center = elevations.size // 2
    if not elevations[center] < OFF_RASTER_ELEVATION:
        return None
    rows, cols = ordinate_cells(row, col, direction, length, grid.dx, grid.dy)
    return Section(XSection(elevations, np.empty(0), float(spacing)), int(row), int(col), float(direction),
                   float((direction - np.pi / 2) % np.pi), rows, cols, _side_one(direction), float(elevations[center]),
                   length=float(length), full_half=int(full_half))


def reaches_its_banks(section: Section, along: np.ndarray | None, water_value) -> bool:
    """Whether a cross section sampled as far as ground MAX_BANK_HEIGHT up (sample_section) also reaches, with water
    at its centre in the water raster (along), to where the water ends on each side, as finding its banks looks."""
    if along is None:
        return True
    center = along.size // 2
    if not along[center] == water_value:
        return True
    return bool((along[:center] != water_value).any() and (along[center + 1:] != water_value).any())


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


def trim_level(configs: Configs, section: Section, n: np.ndarray, max_flow: float, slope: float,
               roughness: DepthRoughness | None) -> float:
    """The highest water surface the later steps are expected to work at on a cross section, its banks and Manning's
    n as found: TRIM_MARGIN above the water that carries its maximum flow over the DEM's cross section, and with the
    standard step, above its table's top over a bed TRIM_CARVING below the stream cell; or, for representative cross
    sections, their highest stage. Without either, the stream cell's elevation. n is its Manning's n before the water's
    goes between banks."""
    xs = section.xs
    thalweg = float(xs.elevations[xs.elevations.size // 2])
    level = highest_stage(thalweg) if configs.build_representative_cross_section else thalweg
    if max_flow > 0.0 and slope > 0.0:
        wse = wse_for_discharge(XSection(xs.elevations, n, xs.ordinate_distance), max_flow, slope,
                                roughness=roughness, slope_factor=configs.slope_adjustment_factor)
        if math.isfinite(wse):
            if configs.standard_step:
                wse += max(3.0, 0.5 * (wse - thalweg + TRIM_CARVING))  # as HydraulicTable reaches above the top
            level = max(level, wse + TRIM_MARGIN)
    return level


def fit_section(configs: Configs, section: Section, grid: Grid, mannings_n: ManningsN, resampler: Resampler,
                max_flow: float, slope: float, roughness: DepthRoughness | None, target_width: float) -> None:
    """Cut a cross section to just above the highest water surface the later steps are expected to work at
    (trim_level), extending it first where it doesn't reach that high, and working the estimate out again on it
    extended."""
    while True:
        n = mannings_n.sample(section.row, section.col, section.direction, section.length, grid.dx, grid.dy)
        level = trim_level(configs, section, n, max_flow, slope, roughness)
        if reaches_above(section, level):
            break
        thalweg = float(section.xs.elevations[section.xs.elevations.size // 2])
        spacing = float(section.xs.ordinate_distance)
        needed = ordinates_above(grid.dem, section.row, section.col, section.direction,
                                 _length_for(section.full_half, spacing), grid.dx, grid.dy, level - thalweg)
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
    return int(math.ceil(distance / spacing)) + 2 if math.isfinite(distance) and distance > 0.0 else 0


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
    if section.xs.elevations.size // 2 >= section.full_half:
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
        z, _ = sample_elevations(grid.dem, section.row, section.col, section.direction, length, grid.dx, grid.dy)
        added = half - old
        xs.elevations = np.concatenate([z[:added], xs.elevations, z[-added:]])
        n = None
        if xs.mannings_n.size:
            n = self.mannings_n.sample(section.row, section.col, section.direction, length, grid.dx, grid.dy)
            xs.mannings_n = np.concatenate([n[:added], xs.mannings_n, n[-added:]])
        section.rows, section.cols = ordinate_cells(section.row, section.col, section.direction, length, grid.dx,
                                                    grid.dy)
        if xs.profile is not None:
            # A carved channel's profile: the ground beyond its banks is the ordinates
            p = xs.profile
            stations = np.concatenate([np.arange(-half, -old) * spacing, p.stations,
                                       np.arange(old + 1, half + 1) * spacing])
            xs.profile = Profile(stations, np.concatenate([z[:added], p.elevations, z[-added:]]),
                                 np.concatenate([n[:added], p.mannings_n, n[-added:]]), p.center + added)
        section.length = length
        return True

    def reach_above(self, section: Section, level: float) -> bool:
        """Extend a cross section, doubling it, until it reaches above a water surface (reaches_above), or is as
        long as before it was trimmed, and say whether it grew."""
        grew = False
        while not reaches_above(section, level) and self.extend(section, 2 * (section.xs.elevations.size // 2)):
            grew = True
        return grew

    def hold_width(self, section: Section, width: float) -> bool:
        """Extend a cross section to hold a channel this wide on either side, and say whether it grew."""
        return self.extend(section, _ordinates_for(width, float(section.xs.ordinate_distance)))


# --- 7. Bathymetry ---------------------------------------------------------------------------------------------------


def unresolved_banks(xs: XSection, target_width: float) -> Banks:
    """The banks of a channel that neither the DEM nor its reach resolved: as wide as its width prior if it has
    one, and otherwise one cell wide."""
    if math.isfinite(target_width) and target_width > 0.0:
        banks = banks_for_width(xs, target_width)
        if banks.valid:
            return banks
    return single_cell_banks(xs)


def bank_ceilings(sections: list, ks, max_flow, slopes, roughness: DepthRoughness | None,
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
        if section.banks.valid and section.banks.method != WATER_BANKS and q > 0.0 and slope > 0.0:
            while True:
                ceilings[position] = wse_for_discharge(section.xs, q, slope, roughness=roughness,
                                                       slope_factor=slope_factor)
                if resampler is None or not resampler.reach_above(section, ceilings[position]):
                    break
    return ceilings


def apply_bathymetry(configs: Configs, grid: Grid, cells: Cells, sections: list, network, slopes, baseflow,
                     target_depth, target_width, water_n: float, *, stream_cells=None, outlets=None, max_flow=None,
                     roughness: DepthRoughness | None = None, resampler: Resampler | None = None) -> np.ndarray:
    """Smooth the banks, find, smooth and carve each channel, and return the filled bathymetry raster.

    With Bathy_Use_Banks, the sections of a reach given no bank elevation become unusable (which happens only where
    the whole site has no bank observation). A channel whose banks are still unresolved after the width filter is
    carved at its width prior, or one cell wide (unresolved_banks). The depth that carries the baseflow is solved with
    water_n, the water class's Manning's n, which is the n the channel has between its banks, so the cross section
    is given it there: between the banks the hydraulics use, which the width filter may have moved from the banks
    found (run() gives it between those). stream_cells and outlets help order the cross sections of a reach none of
    whose neighbours has any (see order_reach). With max_flow, each cell's maximum flow, and the rating curves'
    roughness, a bank from the DEM above the water that carries the maximum flow isn't an observation of the bank
    elevation (bank_ceilings). A resampler extends trimmed cross sections where they're too short for a step.

    With Bathy_Use_Banks, a cross section whose banks weren't found (none, or a single cell from the width prior) and
    whose baseflow is below Bathy_Bankless_Max_Baseflow is carved to carry the baseflow at its stream cell's level,
    not at its reach's bank elevation. It has no bank of its own: the elevation is its neighbours' bank height above
    its stream, and where a creek is too narrow for the DEM to show, the ground beside it can lie below that, so the
    flood maps spread the baseflow away from it. On the N40W112 tile (Salt Lake City) that flooded 2,226 cells beside
    one creek at the 2-year flow; carved to its stream level, 290. Below 20 m³/s it leaves 56 of the 68 US test sites
    as they were, and their mean MCC the same (0.5609 against 0.5617); for every bank-less cross section, the maps
    flood too little and it falls to 0.5523.
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
            sections[k].hydraulic_banks = banks if banks.valid else unresolved_banks(sections[k].xs,
                                                                                     float(target_width[k]))
            if use_banks and _carved_to_its_stream(sections[k], float(baseflow[k]),
                                                   configs.bathy_bankless_max_baseflow):
                result.bank_elevations[position] = _centre(sections[k])
            sections[k].bank_elevation = float(result.bank_elevations[position])
            set_in_bank_roughness(sections[k].xs, sections[k].hydraulic_banks, water_n)

    depths, applies = {}, {}
    for reach, ks in by_reach.items():
        depths[reach], applies[reach] = [], []
        for k in ks:
            section = sections[k]
            depth = bathymetry_depth(section.xs, section.hydraulic_banks, float(baseflow[k]), float(slopes[k]),
                                     trapezoid_height=configs.bathy_trap_h,
                                     target_depth=None if math.isnan(target_depth[k]) else float(target_depth[k]),
                                     mannings_n=water_n)
            depths[reach].append(depth.depth)
            applies[reach].append(depth.apply)
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
    bathymetry = np.full(grid.dem.shape, np.nan, dtype=np.float32)
    burns = np.zeros(grid.dem.shape, dtype=np.int32)
    for k in sorted(carve):
        section = sections[k]
        changed = carve_channel(section.xs, section.hydraulic_banks, carve[k], trapezoid_height=configs.bathy_trap_h,
                                bank_elevation=section.bank_elevation if use_banks else None)
        burn_into_raster(bathymetry, burns, section.rows, section.cols, section.xs.elevations, changed)
    del burns
    ground = None
    if not use_banks:
        ground = grid.dem.astype(np.float32)
        ground[grid.dem >= OFF_RASTER_ELEVATION] = np.nan
    return fill_bathymetry_gaps(bathymetry, ground)


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
    curves = RatingCurves.empty(cells.count, max(increments, 0))
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
        elif reach_average:
            curves.metadata[k, _METADATA_BUT_BASEFLOW] = \
                [comid, section.row, section.col, elevation, slopes[k], section.xs_angle]
    return curves


def _centre(section: Section) -> float:
    """The elevation of the centre of a section's cross section, as the hydraulics use it."""
    return float(section.xs.elevations[section.xs.elevations.size // 2])


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
    """Write the outputs the configuration names."""
    curves = results.curves
    if curves is not None:
        if not curves.has_vdt_data() and configs.vdt_database_numiterations > 0:
            LOG.warning('No VDT data was generated, so no hydraulic output files will be created.')
        else:
            if configs.print_vdt_database:
                results.vdt = write_vdt(curves, configs.print_vdt_database)
            if configs.print_ap_database:
                write_ap(curves, configs.print_ap_database)
            qmax = flows[configs.flow_file_qmax] if flows is not None and configs.flow_file_qmax in flows.columns \
                else pd.Series(dtype=np.float64)
            if configs.reach_average_curve_file:
                vdt = results.vdt if results.vdt is not None else vdt_dataframe(curves)
                write_reach_average_curve_file(curves, vdt, qmax, configs.print_curve_file)
            elif configs.print_curve_file:
                write_curve_file(curves, qmax, configs.print_curve_file)
    if configs.xs_out_file:
        write_cross_sections(results.cross_sections, configs.xs_out_file)
    if configs.build_representative_cross_section and configs.representative_cross_section_file:
        write_representative_cross_sections(results.representative, configs.representative_cross_section_file)
    if results.bathymetry is not None:
        Raster.write_array(results.bathymetry, grid.reference, configs.bathy_out_file, dtype=np.float32,
                           compression=configs.compression)
    if configs.aroutflood:
        flood = np.zeros(grid.dem.shape, dtype=np.uint8)
        if curves is not None:
            for section in results.sections:
                if section is not None and section.usable:
                    flood[section.row, section.col] = FLOODED_CELL
        Raster.write_array(flood, grid.reference, configs.aroutflood, dtype=np.uint8,
                           compression=configs.compression)


# --- The whole run -------------------------------------------------------------------------------------------------


def run(configs: Configs, *, quiet: bool = False, write: bool = True, trim: bool = True) -> Results:
    """Run ARC for a configuration, write the outputs it names (unless write is False), and return what it made.
    With trim False, the cross sections are all kept out to the raster's edge, which gives the same results with
    more memory, and cross sections in the cross-section file that reach that far (see the notes above)."""
    started = time.perf_counter()
    flows = read_flows(configs)
    grid = read_grid(configs)
    layer = read_stream_layer(configs)
    network = stream_network(configs, layer)
    targets = power_law_targets(configs, layer)
    manual = read_manual_sections(configs) if configs.manual_cross_sections_file else None

    cells = stream_cells(configs, grid, flows, manual)
    slopes, keep = cell_slopes(configs, grid, cells, network, layer)
    if not keep.any():
        raise ValueError("No stream cells remain: no reach has a slope (see the warning above).")
    cells, slopes = cells.subset(keep), slopes[keep]
    baseflow, max_flow = cell_flows(configs, flows, cells)
    target_depth, target_width = cell_targets(targets, cells)

    # The stream cells are water, whatever the land cover and the water mask say
    land_cover = grid.land_cover.copy()
    land_cover[cells.rows, cells.cols] = configs.lc_water_value
    water_mask = None
    if grid.water_mask is not None:
        water_mask = grid.water_mask.copy()
        water_mask[cells.rows, cells.cols] = 1
    grid = grid._replace(land_cover=land_cover, water_mask=water_mask)
    lookup, water_n = mannings_n_table(configs)
    mannings_n = ManningsN(land_cover, lookup, water_n)
    roughness = DepthRoughness(configs.k_decay, configs.shallow_factor, configs.deep_factor) \
        if configs.depth_varying_n else None

    resampler = Resampler(grid, mannings_n)
    # Each cell's cross section and banks, and then the cross section trimmed (see the notes above)
    offsets = angle_offsets(configs.degree_manip, configs.degree_interval)
    water, water_value = bank_water(configs, grid)
    sections: list[Section | None] = []
    for k in tqdm.tqdm(range(cells.count), disable=quiet):
        if manual:
            section = manual_section(manual[int(cells.comids[k])])
        else:
            section = sample_section(configs, grid, int(cells.rows[k]), int(cells.cols[k]), offsets, whole=not trim)
        if section is not None:
            width = None if math.isnan(target_width[k]) else float(target_width[k])
            along = water_along(section, water, grid)
            while not reaches_its_banks(section, along, water_value) and \
                    resampler.extend(section, 2 * (section.xs.elevations.size // 2)):
                along = water_along(section, water, grid)
            section.banks = find_banks(section.xs, target_width=width, land_cover=along,
                                       water_value=water_value if along is not None else None)
            section.hydraulic_banks = section.banks
            if trim and not section.manual:
                fit_section(configs, section, grid, mannings_n, resampler, float(max_flow[k]), float(slopes[k]),
                            roughness, float(target_width[k]))
        sections.append(section)

    # The water's n between each cross section's banks, put in the raster so that the cross sections crossing it get
    # it too, and then each cross section's n
    for section in sections:
        if section is not None:
            wet = in_bank(section.xs, section.banks)
            mannings_n.water_between_banks(section.rows[wet], section.cols[wet])
    for section in sections:
        if section is not None:
            sample_roughness(section, grid, mannings_n)
            set_in_bank_roughness(section.xs, section.banks, water_n)
    resampler = resampler if trim else None

    # Where every reach lies, which orders the cross sections of a reach none of whose neighbours has any
    stream_reaches = outlets = None
    if network is not None and (configs.bathy_out_file or configs.standard_step):
        stream_reaches = stream_cells_by_reach(grid.streams)
        outlets = line_outlets(configs, grid, layer, cells.reaches)

    results = Results(configs, cells, slopes, baseflow, max_flow, sections)
    if configs.bathy_out_file:
        results.bathymetry = apply_bathymetry(configs, grid, cells, sections, network, slopes, baseflow,
                                              target_depth, target_width, water_n, stream_cells=stream_reaches,
                                              outlets=outlets, max_flow=max_flow, roughness=roughness,
                                              resampler=resampler)
    usable = [(k, s) for k, s in enumerate(sections) if s is not None and s.usable]
    for _, section in usable:
        set_bank_distances(section.xs, section.hydraulic_banks)

    if configs.makes_rating_curves:
        results.curves = rating_curves(configs, grid, cells, sections, slopes, baseflow, max_flow, roughness, quiet,
                                       resampler)
        if configs.standard_step:
            elevations = grid.dem[cells.rows, cells.cols]
            levels = np.array([math.nan if s is None else s.baseflow_level for s in sections])
            summary = apply_standard_step(network, cells.rows, cells.cols, cells.reaches, sections, slopes, max_flow,
                                          baseflow, results.curves, grid.dx, grid.dy, roughness=roughness,
                                          slope_factor=configs.slope_adjustment_factor, stream_cells=stream_reaches,
                                          outlets=outlets, elevations=np.where(elevations >= OFF_RASTER_ELEVATION,
                                                                               np.nan, elevations),
                                          baseflow_levels=levels,
                                          ensure_level=None if resampler is None else
                                          lambda k, level: resampler.reach_above(sections[k], level))
            LOG.info(f"Standard step: {summary.cells} rating curves linked, {summary.critical} of {summary.steps} "
                     f"steps at critical depth, {summary.held} water surfaces held at the baseflow level")
    if configs.xs_out_file:
        results.cross_sections = [cross_section_record(s, cells.comids[k]) for k, s in usable]
    if configs.build_representative_cross_section:
        if resampler is not None:
            for _, s in usable:
                resampler.reach_above(s, highest_stage(_centre(s)))
        samples = [RepresentativeSample(int(cells.comids[k]), s.xs, float(s.xs.elevations[s.xs.elevations.size // 2]),
                                        float(slopes[k])) for k, s in usable]
        results.representative = representative_cross_section_dataframe(
            samples, roughness=roughness, slope_factor=configs.slope_adjustment_factor)

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
