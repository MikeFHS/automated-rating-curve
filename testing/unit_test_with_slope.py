"""
Synthetic ARC test that isolates stream-slope-method behavior on one triangle channel.

This module is intentionally structured like the other synthetic testing
scripts in ``automated-rating-curve/testing`` so it can be used in two ways:

1. Automated test mode
   ``pytest automated-rating-curve/testing/unit_test_with_slope.py``

   Pytest imports the module, runs the ``test_*`` function, writes outputs into
   a temporary directory, and checks that ARC can process one fixed triangle
   channel using three different stream-slope methods.

2. Manual diagnostic mode
   ``python automated-rating-curve/testing/unit_test_with_slope.py``

   Manual mode writes persistent artifacts under
   ``testing/slope_method_case_outputs/``, prints a compact summary table, and
   opens a multi-panel diagnostic figure with one row per slope method.

Why this file exists
--------------------
``unit_test_with_shapes.py`` varies cross-section geometry while keeping the
overall workflow fixed. This file does the opposite: it fixes the synthetic
geometry to one triangular channel and varies the slope source used by ARC.
That makes it easier to diagnose whether differences in output are caused by:

- per-cell local stream-slope estimation,
- reach-average slope reuse, or
- endpoint-derived slope estimation from a vector stream network.

Slope methods covered here
--------------------------
This script exercises the three primary ARC stream-slope methods:

- ``local_average``
- ``reach_average``
- ``end_points``

ARC also contains ``local_average_corrected``. That method is intentionally not
included here because it is a correction workflow layered on top of
``local_average`` rather than one of the three distinct slope-source options
requested for this comparison.
"""

from __future__ import annotations

from pathlib import Path
import stat
import time

from arc import Arc
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from osgeo import gdal, osr
from shapely.geometry import LineString

gdal.UseExceptions()


# -----------------------------------------------------------------------------
# Universal synthetic-channel parameters
# -----------------------------------------------------------------------------
# The purpose of this file is to compare stream-slope methods, so every other
# hydraulic and geometric assumption is held constant across runs.
length_m = 1000
cellsize = 1.0
bed_slope = 0.001
mannings_discharge_m3s = 10.0
roughness = 0.025
floodplain_offset = 20.0

# Triangle geometry only. The side slope is expressed as horizontal-to-vertical
# ratio so an offset of ``side_slope`` meters increases elevation by 1 meter.
side_slope = 2.0

# Raster padding keeps the synthetic reach away from the array edge so ARC's
# neighborhood-based calculations do not operate directly on the domain border.
longitudinal_padding_cells = 25
lateral_padding_cells = 25

channel_nx = int(length_m / cellsize)
channel_ny = 100
channel_start_col = longitudinal_padding_cells
channel_end_col = channel_start_col + channel_nx
channel_mid_col = channel_start_col + (channel_nx // 2)
nx = channel_nx + 2 * longitudinal_padding_cells
ny = channel_ny + 2 * lateral_padding_cells
center_row = ny // 2

# Coordinate system and georeferencing for the synthetic rasters.
epsg = 26912
origin_x = 444000.0
origin_y = 4447000.0
geotransform = (
    origin_x,
    cellsize,
    0.0,
    origin_y,
    0.0,
    -cellsize,
)

# Manual runs place outputs here so the user can inspect all generated inputs
# and ARC outputs after the script finishes.
script_dir = Path(__file__).resolve().parent
manual_output_root = script_dir / "slope_method_case_outputs"

# ARC case configuration for the three requested slope methods. Colors are only
# used by the manual plotting function.
slope_method_configs: dict[str, dict[str, str | bool]] = {
    "local_average": {
        "arc_value": "local_average",
        "color": "tab:blue",
        "needs_stream_vector": False,
    },
    "reach_average": {
        "arc_value": "reach_average",
        "color": "tab:orange",
        "needs_stream_vector": False,
    },
    "end_points": {
        "arc_value": "end_points",
        "color": "tab:green",
        "needs_stream_vector": True,
    },
}


def build_longitudinal_thalweg(station_m: float) -> float:
    """
    Return thalweg elevation at a given downstream station.

    The synthetic reach is uniform, so the thalweg follows one consistent slope
    across the full modeled length.
    """
    return -(station_m * bed_slope)


def lateral_elevation_above_thalweg(offset_m: float) -> float:
    """
    Return triangle-channel elevation above the thalweg at a lateral offset.

    The returned value is capped at ``floodplain_offset`` so the channel ties
    into one common floodplain elevation across the domain.
    """
    return min(floodplain_offset, offset_m / side_slope)


def create_dem() -> np.ndarray:
    """
    Build the DEM for one synthetic triangular channel.

    The active stream occupies only the interior channel columns. The padded
    upstream and downstream columns remain at floodplain elevation so the
    synthetic stream does not touch the raster boundary.
    """
    dem = np.zeros((ny, nx), dtype=np.float32)

    for x_index in range(nx):
        station_m = np.clip((x_index - channel_start_col) * cellsize, 0.0, length_m)
        thalweg_z = build_longitudinal_thalweg(station_m)
        in_active_channel = channel_start_col <= x_index < channel_end_col

        for y_index in range(ny):
            if in_active_channel:
                offset_m = abs(y_index - center_row) * cellsize
                dem[y_index, x_index] = thalweg_z + lateral_elevation_above_thalweg(offset_m)
            else:
                dem[y_index, x_index] = thalweg_z + floodplain_offset

    return dem


def create_stream_raster() -> np.ndarray:
    """
    Create a one-cell-wide stream raster split into two synthetic COMIDs.

    Keeping two stream IDs is useful in this file because the reach-based slope
    methods should then produce one slope value per COMID rather than one value
    for the entire domain.
    """
    stream = np.zeros((ny, nx), dtype=np.uint8)
    stream[center_row, channel_start_col:channel_mid_col] = 1
    stream[center_row, channel_mid_col:channel_end_col] = 2
    return stream


def create_land_cover_raster() -> np.ndarray:
    """Create a uniform land-cover raster so Manning n is constant everywhere."""
    return np.ones((ny, nx), dtype=np.uint8)


def write_raster(path: Path, array: np.ndarray, gdal_dtype: int, wkt: str) -> None:
    """Write one numpy array to GeoTIFF using the shared georeferencing."""
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(str(path), nx, ny, 1, gdal_dtype)
    dataset.SetGeoTransform(geotransform)
    dataset.SetProjection(wkt)
    dataset.GetRasterBand(1).WriteArray(array)
    dataset = None


def _unlink_with_retries(path: Path, retries: int = 5, delay_s: float = 0.2) -> None:
    """Delete one file with Windows-friendly retries and permission fixes."""
    for attempt in range(retries):
        try:
            if path.exists():
                path.chmod(stat.S_IWRITE | stat.S_IREAD)
                path.unlink()
            return
        except FileNotFoundError:
            return
        except PermissionError:
            if attempt == retries - 1:
                raise
            time.sleep(delay_s * (attempt + 1))


def _rmdir_with_retries(path: Path, retries: int = 5, delay_s: float = 0.2) -> None:
    """Remove one empty directory with Windows-friendly retries."""
    for attempt in range(retries):
        try:
            if path.exists():
                path.chmod(stat.S_IWRITE | stat.S_IREAD)
                path.rmdir()
            return
        except FileNotFoundError:
            return
        except PermissionError:
            if attempt == retries - 1:
                raise
            time.sleep(delay_s * (attempt + 1))


def clear_case_directory(case_dir: Path) -> None:
    """
    Remove all files previously created for one synthetic case.

    The manual workflow writes into persistent folders under
    ``testing/slope_method_case_outputs``. Clearing each case directory before
    writing new inputs prevents stale rasters, vector files, and VDT outputs
    from being reused accidentally across runs.
    """
    if not case_dir.exists():
        return

    for child in case_dir.iterdir():
        if child.is_dir():
            clear_case_directory(child)
            _rmdir_with_retries(child)
        else:
            _unlink_with_retries(child)


def _remove_existing_geopackage(gpkg_path: Path) -> None:
    """
    Delete a GeoPackage and its SQLite sidecars before rewriting it.

    Manual reruns reuse stable filenames. Removing the main ``.gpkg`` file plus
    any ``-wal`` and ``-shm`` companions keeps vector outputs deterministic.
    """
    if gpkg_path.exists():
        _unlink_with_retries(gpkg_path)

    gpkg_wal_path = gpkg_path.with_name(gpkg_path.name + "-wal")
    gpkg_shm_path = gpkg_path.with_name(gpkg_path.name + "-shm")
    if gpkg_wal_path.exists():
        _unlink_with_retries(gpkg_wal_path)
    if gpkg_shm_path.exists():
        _unlink_with_retries(gpkg_shm_path)


def _row_col_to_xy(row_idx: float, col_idx: float, raster_geotransform: tuple[float, ...]) -> tuple[float, float]:
    """Convert one raster row/column index to the center coordinates of that cell."""
    x_coord = raster_geotransform[0] + (col_idx + 0.5) * raster_geotransform[1] + (row_idx + 0.5) * raster_geotransform[2]
    y_coord = raster_geotransform[3] + (col_idx + 0.5) * raster_geotransform[4] + (row_idx + 0.5) * raster_geotransform[5]
    return x_coord, y_coord


def export_stream_lines(stream: np.ndarray, stream_lines_path: Path) -> None:
    """
    Export one stream polyline per synthetic COMID from the raster stream mask.

    The ``end_points`` ARC slope method requires a vector stream network. This
    helper converts each nonzero integer stream ID into one ordered line so the
    vector geometry matches the rasterized stream exactly.
    """
    _remove_existing_geopackage(stream_lines_path)

    line_records = []
    unique_stream_ids = sorted(int(stream_id) for stream_id in np.unique(stream) if stream_id > 0)

    for stream_id in unique_stream_ids:
        stream_rows, stream_cols = np.where(stream == stream_id)
        if len(stream_rows) < 2:
            raise ValueError(f"Stream ID {stream_id} must contain at least two cells to export a line.")

        order = np.lexsort((stream_rows, stream_cols))
        line_points: list[tuple[float, float]] = []
        for row_idx, col_idx in zip(stream_rows[order], stream_cols[order]):
            x_coord, y_coord = _row_col_to_xy(float(row_idx), float(col_idx), geotransform)
            if not line_points or line_points[-1] != (x_coord, y_coord):
                line_points.append((x_coord, y_coord))

        if len(line_points) < 2:
            raise ValueError(f"Stream ID {stream_id} collapsed to fewer than two unique vertices.")

        line_records.append({"COMID": stream_id, "geometry": LineString(line_points)})

    stream_lines_gdf = gpd.GeoDataFrame(line_records, geometry="geometry", crs=f"EPSG:{epsg}")
    stream_lines_gdf.to_file(stream_lines_path, driver="GPKG", layer="streams")


def export_vdt_points(vdt_df: pd.DataFrame, vdt_points_path: Path) -> None:
    """Export VDT rows as point features for quick GIS inspection."""
    col_idx = vdt_df["Col"].astype(float)
    row_idx = vdt_df["Row"].astype(float)
    x_coords = geotransform[0] + (col_idx + 0.5) * geotransform[1] + (row_idx + 0.5) * geotransform[2]
    y_coords = geotransform[3] + (col_idx + 0.5) * geotransform[4] + (row_idx + 0.5) * geotransform[5]

    vdt_points_gdf = gpd.GeoDataFrame(
        vdt_df.copy(),
        geometry=gpd.points_from_xy(x_coords, y_coords),
        crs=f"EPSG:{epsg}",
    )
    vdt_points_gdf.to_file(vdt_points_path, driver="GeoJSON")


def export_cross_section_lines(xs_df: pd.DataFrame, raster_path: Path, xs_lines_path: Path) -> None:
    """
    Export one polyline feature per ARC cross section.

    The cross-section export text file stores the center cell and the far
    endpoint on each side. This helper reconstructs those samples as three-
    vertex line features using the DEM geotransform.
    """
    _remove_existing_geopackage(xs_lines_path)

    raster_dataset = gdal.Open(str(raster_path), gdal.GA_ReadOnly)
    if raster_dataset is None:
        raise FileNotFoundError(f"Cannot open raster for cross-section export: {raster_path}")

    raster_geotransform = raster_dataset.GetGeoTransform()
    raster_projection = raster_dataset.GetProjection()
    raster_dataset = None

    line_records = []
    for _, row in xs_df.iterrows():
        endpoint_1 = _row_col_to_xy(float(row["r1"]), float(row["c1"]), raster_geotransform)
        center_point = _row_col_to_xy(float(row["Row"]), float(row["Col"]), raster_geotransform)
        endpoint_2 = _row_col_to_xy(float(row["r2"]), float(row["c2"]), raster_geotransform)

        line_records.append(
            {
                "comid": int(row["COMID"]),
                "row_ctr": int(row["Row"]),
                "col_ctr": int(row["Col"]),
                "r1": int(row["r1"]),
                "c1": int(row["c1"]),
                "r2": int(row["r2"]),
                "c2": int(row["c2"]),
                "ord_dist": float(row["Ordinate_Dist"]),
                "geometry": LineString([endpoint_1, center_point, endpoint_2]),
            }
        )

    xs_lines_gdf = gpd.GeoDataFrame(line_records, geometry="geometry", crs=raster_projection)
    xs_lines_gdf.to_file(xs_lines_path, driver="GPKG", layer="cross_sections")


def write_case_inputs(case_dir: Path, dem: np.ndarray) -> dict[str, Path]:
    """
    Write all ARC inputs needed for one synthetic slope-method case.

    Every slope method gets its own case directory so each run has isolated
    inputs and outputs. A stream GeoPackage is always written because the
    ``end_points`` case requires it and the shared output structure stays
    simpler if every case directory contains the same artifact set.
    """
    clear_case_directory(case_dir)
    case_dir.mkdir(parents=True, exist_ok=True)

    stream = create_stream_raster()
    land_cover = create_land_cover_raster()

    srs = osr.SpatialReference()
    srs.ImportFromEPSG(epsg)
    wkt = srs.ExportToWkt()

    dem_path = case_dir / "dem.tif"
    stream_path = case_dir / "stream.tif"
    land_cover_path = case_dir / "land_cover.tif"
    flow_path = case_dir / "base_max_flow.csv"
    max_flow_only_path = case_dir / "max_flow_only.csv"
    mannings_path = case_dir / "mannings.txt"
    vdt_path = case_dir / "vdt.csv"
    vdt_points_path = case_dir / "vdt_points.geojson"
    xs_output_path = case_dir / "cross_sections.txt"
    xs_lines_path = case_dir / "cross_sections.gpkg"
    stream_lines_path = case_dir / "stream_network.gpkg"

    write_raster(dem_path, dem, gdal.GDT_Float32, wkt)
    write_raster(stream_path, stream, gdal.GDT_Byte, wkt)
    write_raster(land_cover_path, land_cover, gdal.GDT_Byte, wkt)
    export_stream_lines(stream, stream_lines_path)

    # Both synthetic stream IDs use the same discharge so differences among the
    # cases reflect slope-method behavior rather than flow forcing.
    flow_df = pd.DataFrame(
        {
            "COMID": [1, 2],
            "baseflow": [0.0, 0.0],
            "maxflow": [mannings_discharge_m3s, mannings_discharge_m3s],
        }
    )
    flow_df.to_csv(flow_path, index=False)
    flow_df.to_csv(max_flow_only_path, index=False, columns=["COMID", "maxflow"])

    with mannings_path.open("w", encoding="utf-8") as mannings_file:
        mannings_file.write("lu\tdesc\troughness\n")
        mannings_file.write(f"1\tland\t{roughness}\n")

    return {
        "dem_path": dem_path,
        "stream_path": stream_path,
        "land_cover_path": land_cover_path,
        "flow_path": flow_path,
        "max_flow_only_path": max_flow_only_path,
        "mannings_path": mannings_path,
        "vdt_path": vdt_path,
        "vdt_points_path": vdt_points_path,
        "xs_output_path": xs_output_path,
        "xs_lines_path": xs_lines_path,
        "stream_lines_path": stream_lines_path,
    }


def calculate_mannings_discharge(depth_m: float) -> tuple[float, float, float]:
    """
    Calculate discharge, top width, and velocity for the triangle section.

    This analytical reference does not depend on ARC's stream-slope method. It
    therefore provides one stable baseline for all three ARC cases.
    """
    area = side_slope * depth_m ** 2
    wetted_perimeter = 2.0 * depth_m * np.sqrt(1.0 + side_slope ** 2)
    top_width = 2.0 * side_slope * depth_m

    hydraulic_radius = area / wetted_perimeter
    discharge_estimate = (1.0 / roughness) * area * hydraulic_radius ** (2.0 / 3.0) * np.sqrt(bed_slope)
    velocity = discharge_estimate / area
    return discharge_estimate, top_width, velocity


def solve_mannings_depth() -> float:
    """Solve Manning's equation for normal depth using bisection."""
    lower_depth = 1.0e-6
    upper_depth = max(floodplain_offset, 1.0)

    while calculate_mannings_discharge(upper_depth)[0] < mannings_discharge_m3s:
        upper_depth *= 2.0

    for _ in range(60):
        trial_depth = 0.5 * (lower_depth + upper_depth)
        if calculate_mannings_discharge(trial_depth)[0] < mannings_discharge_m3s:
            lower_depth = trial_depth
        else:
            upper_depth = trial_depth

    return 0.5 * (lower_depth + upper_depth)


def compute_station_from_col(col_idx: pd.Series | np.ndarray) -> pd.Series | np.ndarray:
    """
    Convert raster columns back to physical station along the active channel.

    The synthetic rasters include upstream and downstream padding, so VDT
    column values must be offset back into the modeled reach before plots and
    residual calculations are made.
    """
    station_m = (col_idx - channel_start_col) * cellsize
    return np.clip(station_m, 0.0, length_m)


def _build_arc_args(case_paths: dict[str, Path], slope_method_config: dict[str, str | bool]) -> dict[str, str | int]:
    """
    Build the ARC input dictionary for one slope-method case.

    Only the stream-slope-method arguments change among cases. Every other ARC
    setting remains fixed so the comparison isolates that one dimension.
    """
    args: dict[str, str | int] = {
        "DEM_File": str(case_paths["dem_path"]),
        "Stream_File": str(case_paths["stream_path"]),
        "LU_Raster_SameRes": str(case_paths["land_cover_path"]),
        "LU_Manning_n": str(case_paths["mannings_path"]),
        "Flow_File": str(case_paths["flow_path"]),
        "Degree_Manip": 0,
        "Degree_Interval": 0,
        "Low_Spot_Range": 0,
        "Gen_Slope_Dist": 10,
        "Gen_Dir_Dist": 10,
        "X_Section_Dist": 40,
        "VDT_Database_NumIterations": 2,
        "Print_VDT_Database": str(case_paths["vdt_path"]),
        "XS_Out_File": str(case_paths["xs_output_path"]),
        "Flow_File_ID": "COMID",
        "Flow_File_BF": "baseflow",
        "Flow_File_QMax": "maxflow",
        "Stream_Slope_Method": str(slope_method_config["arc_value"]),
    }

    if bool(slope_method_config["needs_stream_vector"]):
        args["StrmShp_File"] = str(case_paths["stream_lines_path"])

    return args


def run_slope_method_case(method_name: str, slope_method_config: dict[str, str | bool], output_root: Path) -> pd.DataFrame:
    """
    Create one synthetic case, run ARC, and attach the Manning reference.

    ``method_name`` is the user-facing case key, while ``arc_value`` is the
    exact string ARC expects in the model input dictionary.
    """
    case_dir = output_root / method_name
    dem = create_dem()
    case_paths = write_case_inputs(case_dir, dem)

    Arc(args=_build_arc_args(case_paths, slope_method_config)).set_log_level("info").run()

    vdt_df = pd.read_csv(case_paths["vdt_path"])
    xs_df = pd.read_csv(case_paths["xs_output_path"], sep="\t")
    vdt_df["Station (m)"] = compute_station_from_col(vdt_df["Col"])

    depth_y = solve_mannings_depth()
    _, top_width_y, velocity_y = calculate_mannings_discharge(depth_y)
    vdt_df["Manning Depth (m)"] = depth_y
    vdt_df["Manning Top Width (m)"] = top_width_y
    vdt_df["Manning Velocity (m/s)"] = velocity_y
    vdt_df["Manning WSE (m)"] = depth_y - bed_slope * vdt_df["Station (m)"]
    vdt_df["Slope Method"] = method_name

    export_vdt_points(vdt_df, case_paths["vdt_points_path"])
    export_cross_section_lines(xs_df, case_paths["dem_path"], case_paths["xs_lines_path"])
    return vdt_df


def run_all_slope_method_cases(output_root: Path) -> dict[str, pd.DataFrame]:
    """
    Run the triangular synthetic reach with every configured slope method.

    This wrapper is shared by automated and manual execution paths so both
    workflows use the same geometry generation and ARC configuration logic.
    """
    output_root.mkdir(parents=True, exist_ok=True)
    return {
        method_name: run_slope_method_case(method_name, method_config, output_root)
        for method_name, method_config in slope_method_configs.items()
    }


def _select_upstream_normal_flow_slice(vdt_df: pd.DataFrame) -> pd.Series:
    """
    Return the regression-comparison reach.

    The synthetic reach is uniform, so the main reason to trim the domain is to
    reduce the influence of boundary effects near the downstream end.
    """
    return vdt_df["Station (m)"] <= (length_m - 25.0)


def assert_slope_method_case_is_valid(method_name: str, vdt_df: pd.DataFrame) -> None:
    """
    Perform regression-style assertions for one slope-method case.

    The purpose of these checks is to catch broken ARC runs, missing output
    columns, or very large departures from the analytical Manning reference.
    The tolerances are deliberately moderate because ARC operates on rasterized
    geometry rather than the exact analytical section.
    """
    assert not vdt_df.empty, f"{method_name} produced an empty VDT table."
    assert "wse_2" in vdt_df.columns, f"{method_name} VDT output is missing wse_2."
    assert "t_2" in vdt_df.columns, f"{method_name} VDT output is missing t_2."
    assert "v_2" in vdt_df.columns, f"{method_name} VDT output is missing v_2."
    assert "Slope" in vdt_df.columns, f"{method_name} VDT output is missing Slope."

    assert np.isfinite(vdt_df["wse_2"]).all(), f"{method_name} ARC WSE contains non-finite values."
    assert np.isfinite(vdt_df["t_2"]).all(), f"{method_name} ARC top width contains non-finite values."
    assert np.isfinite(vdt_df["v_2"]).all(), f"{method_name} ARC velocity contains non-finite values."
    assert np.isfinite(vdt_df["Slope"]).all(), f"{method_name} ARC slope contains non-finite values."
    assert np.isfinite(vdt_df["Manning WSE (m)"]).all(), f"{method_name} Manning WSE contains non-finite values."
    assert np.isfinite(vdt_df["Manning Top Width (m)"]).all(), f"{method_name} Manning top width contains non-finite values."
    assert np.isfinite(vdt_df["Manning Velocity (m/s)"]).all(), f"{method_name} Manning velocity contains non-finite values."

    assert (vdt_df["Manning Depth (m)"] > 0.0).all(), f"{method_name} Manning depth must be positive."
    assert (vdt_df["Manning Top Width (m)"] > 0.0).all(), f"{method_name} Manning top width must be positive."
    assert (vdt_df["Manning Velocity (m/s)"] > 0.0).all(), f"{method_name} Manning velocity must be positive."
    assert (vdt_df["Slope"] > 0.0).all(), f"{method_name} ARC slope must be positive."

    upstream_df = vdt_df.loc[_select_upstream_normal_flow_slice(vdt_df)].copy()
    assert not upstream_df.empty, f"{method_name} upstream comparison window is empty."

    wse_residual = (upstream_df["wse_2"] - upstream_df["Manning WSE (m)"]).abs()
    top_width_residual = (upstream_df["t_2"] - upstream_df["Manning Top Width (m)"]).abs()
    velocity_residual = (upstream_df["v_2"] - upstream_df["Manning Velocity (m/s)"]).abs()

    assert wse_residual.median() < 0.90, f"{method_name} median upstream WSE residual is too large: {wse_residual.median():.3f} m"
    assert wse_residual.max() < 1.75, f"{method_name} maximum upstream WSE residual is too large: {wse_residual.max():.3f} m"
    assert top_width_residual.median() < 3.00, f"{method_name} median upstream top-width residual is too large: {top_width_residual.median():.3f} m"
    assert top_width_residual.max() < 6.00, f"{method_name} maximum upstream top-width residual is too large: {top_width_residual.max():.3f} m"
    assert velocity_residual.median() < 0.75, f"{method_name} median upstream velocity residual is too large: {velocity_residual.median():.3f} m/s"
    assert velocity_residual.max() < 1.50, f"{method_name} maximum upstream velocity residual is too large: {velocity_residual.max():.3f} m/s"

    # The reach-based methods should reuse one slope per COMID. A small
    # tolerance is used instead of exact equality to avoid brittle failures if
    # the output writer changes floating-point formatting in the future.
    if method_name in {"reach_average", "end_points"}:
        for comid, comid_df in vdt_df.groupby("COMID"):
            slope_span = float(comid_df["Slope"].max() - comid_df["Slope"].min())
            assert slope_span < 1.0e-10, (
                f"{method_name} should use one slope per COMID, but COMID {comid} "
                f"has slope span {slope_span:.3e}"
            )


def summarize_results(results: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Build a compact summary table for manual runs.

    The summary keeps the most important information visible without opening the
    VDT files manually: analytical reference values, upstream residuals, and
    the central tendency of ARC's reported stream slope for each method.
    """
    summary_rows = []

    for method_name, vdt_df in results.items():
        upstream_df = vdt_df.loc[_select_upstream_normal_flow_slice(vdt_df)].copy()
        wse_residual = (upstream_df["wse_2"] - upstream_df["Manning WSE (m)"]).abs()
        top_width_residual = (upstream_df["t_2"] - upstream_df["Manning Top Width (m)"]).abs()
        velocity_residual = (upstream_df["v_2"] - upstream_df["Manning Velocity (m/s)"]).abs()

        summary_rows.append(
            {
                "slope_method": method_name,
                "manning_depth_m": float(vdt_df["Manning Depth (m)"].iloc[0]),
                "manning_top_width_m": float(vdt_df["Manning Top Width (m)"].iloc[0]),
                "manning_velocity_mps": float(vdt_df["Manning Velocity (m/s)"].iloc[0]),
                "median_vdt_slope": float(vdt_df["Slope"].median()),
                "max_abs_slope_residual": float((vdt_df["Slope"] - bed_slope).abs().max()),
                "median_upstream_wse_residual_m": float(wse_residual.median()),
                "max_upstream_wse_residual_m": float(wse_residual.max()),
                "median_upstream_top_width_residual_m": float(top_width_residual.median()),
                "max_upstream_top_width_residual_m": float(top_width_residual.max()),
                "median_upstream_velocity_residual_mps": float(velocity_residual.median()),
                "max_upstream_velocity_residual_mps": float(velocity_residual.max()),
                "num_vdt_rows": int(len(vdt_df)),
            }
        )

    return pd.DataFrame(summary_rows).sort_values("slope_method").reset_index(drop=True)


def _compute_fixed_ylim(series_list: list[pd.Series | np.ndarray], pad_fraction: float = 0.05) -> tuple[float, float]:
    """Return one fixed y-limit pair for a full plot column."""
    finite_chunks = []
    for values in series_list:
        array = np.asarray(values, dtype=float)
        finite = array[np.isfinite(array)]
        if finite.size > 0:
            finite_chunks.append(finite)

    if not finite_chunks:
        return (0.0, 1.0)

    combined = np.concatenate(finite_chunks)
    y_min = float(combined.min())
    y_max = float(combined.max())
    span = y_max - y_min
    if span <= 0.0:
        pad = max(abs(y_min) * pad_fraction, 1.0)
    else:
        pad = span * pad_fraction
    return (y_min - pad, y_max + pad)


def plot_results(results: dict[str, pd.DataFrame]) -> None:
    """
    Plot one row per slope method for manual inspection.

    The first three columns compare ARC output against the analytical Manning
    reference. The fourth column shows the row-specific slope values stored in
    the VDT output together with the synthetic bed slope used to build the DEM.
    """
    title_fontsize = 11
    label_fontsize = 8
    tick_fontsize = 8
    legend_fontsize = 8

    figure, axes = plt.subplots(len(slope_method_configs), 4, figsize=(16, 9), sharex="col")
    axes_array = np.atleast_2d(axes)
    all_vdt_frames = list(results.values())
    wse_ylim = _compute_fixed_ylim([df["wse_2"] for df in all_vdt_frames] + [df["Manning WSE (m)"] for df in all_vdt_frames])
    top_width_ylim = _compute_fixed_ylim([df["t_2"] for df in all_vdt_frames] + [df["Manning Top Width (m)"] for df in all_vdt_frames])
    velocity_ylim = _compute_fixed_ylim([df["v_2"] for df in all_vdt_frames] + [df["Manning Velocity (m/s)"] for df in all_vdt_frames])
    slope_ylim = _compute_fixed_ylim(
        [df["Slope"] * 100.0 for df in all_vdt_frames] + [np.array([bed_slope * 100.0])]
    )

    for axis_row, (method_name, method_config) in zip(axes_array, slope_method_configs.items()):
        vdt_df = results[method_name]
        method_color = str(method_config["color"])

        wse_axis = axis_row[0]
        top_width_axis = axis_row[1]
        velocity_axis = axis_row[2]
        slope_axis = axis_row[3]

        wse_axis.plot(vdt_df["Station (m)"], vdt_df["wse_2"], color=method_color, label="ARC")
        wse_axis.plot(vdt_df["Station (m)"], vdt_df["Manning WSE (m)"], color=method_color, linestyle="--", label="Manning")
        wse_axis.set_ylabel("Elevation (m)", fontsize=label_fontsize)
        wse_axis.set_title(f"{method_name} WSE", fontsize=title_fontsize)
        wse_axis.set_ylim(*wse_ylim)
        wse_axis.legend(fontsize=legend_fontsize)
        wse_axis.tick_params(axis="both", labelsize=tick_fontsize)
        wse_axis.grid()

        top_width_axis.plot(vdt_df["Station (m)"], vdt_df["t_2"], color=method_color, label="ARC")
        top_width_axis.plot(vdt_df["Station (m)"], vdt_df["Manning Top Width (m)"], color=method_color, linestyle="--", label="Manning")
        top_width_axis.set_ylabel("Top Width (m)", fontsize=label_fontsize)
        top_width_axis.set_title(f"{method_name} Top Width", fontsize=title_fontsize)
        top_width_axis.set_ylim(*top_width_ylim)
        top_width_axis.legend(fontsize=legend_fontsize)
        top_width_axis.tick_params(axis="both", labelsize=tick_fontsize)
        top_width_axis.grid()

        velocity_axis.plot(vdt_df["Station (m)"], vdt_df["v_2"], color=method_color, label="ARC")
        velocity_axis.plot(vdt_df["Station (m)"], vdt_df["Manning Velocity (m/s)"], color=method_color, linestyle="--", label="Manning")
        velocity_axis.set_ylabel("Velocity (m/s)", fontsize=label_fontsize)
        velocity_axis.set_title(f"{method_name} Velocity", fontsize=title_fontsize)
        velocity_axis.set_ylim(*velocity_ylim)
        velocity_axis.legend(fontsize=legend_fontsize)
        velocity_axis.tick_params(axis="both", labelsize=tick_fontsize)
        velocity_axis.grid()

        slope_axis.plot(vdt_df["Station (m)"], vdt_df["Slope"] * 100.0, color=method_color, label="VDT Slope")
        slope_axis.axhline(bed_slope * 100.0, color=method_color, linestyle="--", label="Synthetic Bed Slope")
        slope_axis.set_ylabel("Slope (%)", fontsize=label_fontsize)
        slope_axis.set_title(f"{method_name} Slope", fontsize=title_fontsize)
        slope_axis.set_ylim(*slope_ylim)
        slope_axis.legend(fontsize=legend_fontsize)
        slope_axis.tick_params(axis="both", labelsize=tick_fontsize)
        slope_axis.grid()

    axes_array[-1, 0].set_xlabel("Station (m)", fontsize=label_fontsize)
    axes_array[-1, 1].set_xlabel("Station (m)", fontsize=label_fontsize)
    axes_array[-1, 2].set_xlabel("Station (m)", fontsize=label_fontsize)
    axes_array[-1, 3].set_xlabel("Station (m)", fontsize=label_fontsize)
    figure.tight_layout(pad=2.5, w_pad=2.0, h_pad=2.5)
    plt.show()


def test_slope_method_cases(tmp_path: Path) -> None:
    """
    Automated regression test for all three synthetic slope-method cases.

    Pytest provides ``tmp_path`` so the run is isolated and does not interfere
    with any persistent manual outputs from earlier exploratory runs.
    """
    results = run_all_slope_method_cases(tmp_path / "slope_method_cases")

    for method_name, vdt_df in results.items():
        assert_slope_method_case_is_valid(method_name, vdt_df)


def main() -> None:
    """
    Run the slope-method comparison manually and show diagnostic plots.

    This function is kept separate from the automated test so importing the
    module under pytest never triggers ARC runs, file writes, or figures.
    """
    manual_output_root.mkdir(parents=True, exist_ok=True)
    results = run_all_slope_method_cases(manual_output_root)
    summary_df = summarize_results(results)

    print("Synthetic slope-method test summary:")
    print(summary_df.to_string(index=False))
    print(f"\nOutputs written to: {manual_output_root}")

    plot_results(results)


if __name__ == "__main__":
    # Manual execution path:
    # - writes persistent outputs under testing/slope_method_case_outputs/
    # - prints a summary table
    # - opens diagnostic plots
    #
    # Pytest never enters this block because it imports the module rather than
    # running it as a script.
    main()
