"""
Synthetic ARC test for trapezoidal, rectangular, and triangular channels using
geographic coordinates instead of a projected CRS.

This module mirrors ``unit_test_with_shapes.py`` closely, but the rasters are
written in NAD83 geographic coordinates (EPSG:4269). That makes the file useful
for validating ARC's code path that detects a geographic CRS and converts raster
cell sizes from degrees to meters internally.

Execution modes
---------------
1. Automated test mode
   ``pytest automated-rating-curve/testing/unit_test_with_shapes_geographic.py``

   Pytest imports the module, runs the ``test_*`` functions, and writes outputs
   to a temporary directory.

2. Manual diagnostic mode
   ``python automated-rating-curve/testing/unit_test_with_shapes_geographic.py``

   The script writes geographic test artifacts into
   ``testing/shape_case_outputs_geographic/``, prints a compact summary, and
   opens the same diagnostic plots used by the projected-coordinate companion
   test. It also exports a stream GeoPackage that ARC can read through
   ``StrmShp_File`` when the ``end_points`` slope method is enabled.

Important geometry note
-----------------------
The synthetic channel geometry is still defined in meters so the Manning
comparison remains easy to interpret. Only the raster georeferencing is
geographic. A helper converts a 1 m synthetic cell size into degree increments
at a representative NAD83 latitude so ARC receives a geographic raster whose
physical spacing is still approximately 1 m.
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
from pyproj import Geod
from shapely.geometry import LineString

gdal.UseExceptions()


# -----------------------------------------------------------------------------
# Universal hydraulic and raster parameters shared by every synthetic case.
# -----------------------------------------------------------------------------
length_m = 1000
cellsize_m = 1.0
bed_slope = 0.001
mannings_discharge_m3s = 10.0
roughness = 0.025
floodplain_offset = 20.0

bottom_width = 10
side_slope = 2.0

# Raster padding keeps the synthetic reach away from the DEM edges. This is the
# same test-domain concept used in the projected-coordinate version.
longitudinal_padding_cells = 25
lateral_padding_cells = 25

channel_nx = int(length_m / cellsize_m)
channel_ny = 100
channel_start_col = longitudinal_padding_cells
channel_end_col = channel_start_col + channel_nx
channel_mid_col = channel_start_col + (channel_nx // 2)
nx = channel_nx + 2 * longitudinal_padding_cells
ny = channel_ny + 2 * lateral_padding_cells
center_row = ny // 2

# -----------------------------------------------------------------------------
# Geographic CRS configuration.
# -----------------------------------------------------------------------------
# EPSG:4269 is NAD83 geographic coordinates. The synthetic raster is centered at
# a representative mid-latitude location so a 1 m east-west pixel remains well
# behaved in degree space.
epsg = 4269
reference_longitude_deg = -105.0
reference_latitude_deg = 40.0
geod = Geod(ellps="GRS80")


def _build_degree_cell_sizes(cellsize_meters: float) -> tuple[float, float]:
    """
    Convert a target physical cell size in meters into degree increments.

    The x and y degree steps are derived independently because one degree of
    longitude and one degree of latitude do not represent the same physical
    distance at a given latitude.
    """
    lon_east, lat_east, _ = geod.fwd(reference_longitude_deg, reference_latitude_deg, 90.0, cellsize_meters)
    lon_north, lat_north, _ = geod.fwd(reference_longitude_deg, reference_latitude_deg, 0.0, cellsize_meters)
    cellsize_x_deg = abs(lon_east - reference_longitude_deg)
    cellsize_y_deg = abs(lat_north - reference_latitude_deg)
    return cellsize_x_deg, cellsize_y_deg


cellsize_x_deg, cellsize_y_deg = _build_degree_cell_sizes(cellsize_m)

# Center the raster around the chosen reference location so the synthetic test
# remains geographically compact and symmetric.
origin_x = reference_longitude_deg - 0.5 * nx * cellsize_x_deg
origin_y = reference_latitude_deg + 0.5 * ny * cellsize_y_deg
geotransform = (
    origin_x,
    cellsize_x_deg,
    0.0,
    origin_y,
    0.0,
    -cellsize_y_deg,
)

# Manual outputs are kept separate from the projected-coordinate test so the
# two workflows do not overwrite one another.
script_dir = Path(__file__).resolve().parent
manual_output_root = script_dir / "geographic_case_outputs"

shape_configs: dict[str, dict[str, float | str]] = {
    "trapezoid": {
        "kind": "trapezoid",
        "bottom_width": bottom_width,
        "side_slope": side_slope,
        "color": "tab:blue",
    },
    "rectangle": {
        "kind": "rectangle",
        "width": bottom_width,
        "color": "tab:orange",
    },
    "triangle": {
        "kind": "triangle",
        "side_slope": side_slope,
        "color": "tab:green",
    },
}


def build_longitudinal_thalweg(station_m: float) -> float:
    """Return thalweg elevation at a downstream station in meters."""
    return -(station_m * bed_slope)


def lateral_elevation_above_thalweg(offset_m: float, shape_config: dict[str, float | str]) -> float:
    """
    Return channel elevation above the thalweg for a lateral offset in meters.
    """
    kind = shape_config["kind"]

    if kind == "trapezoid":
        half_bottom_width = float(shape_config["bottom_width"]) / 2.0
        if offset_m <= half_bottom_width:
            return 0.0
        return min(floodplain_offset, (offset_m - half_bottom_width) / float(shape_config["side_slope"]))

    if kind == "rectangle":
        if offset_m <= float(shape_config["width"]) / 2.0:
            return 0.0
        return floodplain_offset

    if kind == "triangle":
        return min(floodplain_offset, offset_m / float(shape_config["side_slope"]))

    raise ValueError(f"Unsupported shape kind: {kind}")


def create_dem(shape_config: dict[str, float | str]) -> np.ndarray:
    """
    Build a geographic DEM whose synthetic channel geometry is still meter-based.

    The raster georeferencing is geographic, but the cross section and thalweg
    are constructed from cell indices using ``cellsize_m`` so the synthetic
    hydraulics remain directly comparable to the projected test.
    """
    dem = np.zeros((ny, nx), dtype=np.float32)

    for x_index in range(nx):
        station_m = np.clip((x_index - channel_start_col) * cellsize_m, 0.0, length_m)
        thalweg_z = build_longitudinal_thalweg(station_m)
        in_active_channel = channel_start_col <= x_index < channel_end_col
        for y_index in range(ny):
            if in_active_channel:
                offset_m = abs(y_index - center_row) * cellsize_m
                dem[y_index, x_index] = thalweg_z + lateral_elevation_above_thalweg(offset_m, shape_config)
            else:
                dem[y_index, x_index] = thalweg_z + floodplain_offset

    return dem


def create_stream_raster() -> np.ndarray:
    """
    Create a one-cell-wide stream centerline split into two synthetic COMIDs.
    """
    stream = np.zeros((ny, nx), dtype=np.uint8)
    stream[center_row, channel_start_col:channel_mid_col] = 1
    stream[center_row, channel_mid_col:channel_end_col] = 2
    return stream


def create_land_cover_raster() -> np.ndarray:
    """Create a uniform land-cover raster so Manning n is spatially constant."""
    return np.ones((ny, nx), dtype=np.uint8)


def write_raster(path: Path, array: np.ndarray, gdal_dtype: int, wkt: str) -> None:
    """Write a numpy array to GeoTIFF using the shared geographic geotransform."""
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
    """
    if not case_dir.exists():
        return

    for child in case_dir.iterdir():
        if child.is_dir():
            clear_case_directory(child)
            _rmdir_with_retries(child)
        else:
            _unlink_with_retries(child)


def write_case_inputs(case_dir: Path, dem: np.ndarray) -> dict[str, Path]:
    """
    Write all ARC inputs needed for one geographic synthetic case.
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


def calculate_mannings_discharge(depth_m: float, shape_config: dict[str, float | str]) -> tuple[float, float, float]:
    """
    Calculate discharge, top width, and velocity for the requested depth.
    """
    kind = shape_config["kind"]

    if kind == "trapezoid":
        area = (float(shape_config["bottom_width"]) + float(shape_config["side_slope"]) * depth_m) * depth_m
        wetted_perimeter = float(shape_config["bottom_width"]) + (
            2.0 * depth_m * np.sqrt(1.0 + float(shape_config["side_slope"]) ** 2)
        )
        top_width = float(shape_config["bottom_width"]) + 2.0 * float(shape_config["side_slope"]) * depth_m
    elif kind == "rectangle":
        area = float(shape_config["width"]) * depth_m
        wetted_perimeter = float(shape_config["width"]) + 2.0 * depth_m
        top_width = float(shape_config["width"])
    elif kind == "triangle":
        area = float(shape_config["side_slope"]) * depth_m ** 2
        wetted_perimeter = 2.0 * depth_m * np.sqrt(1.0 + float(shape_config["side_slope"]) ** 2)
        top_width = 2.0 * float(shape_config["side_slope"]) * depth_m
    else:
        raise ValueError(f"Unsupported shape kind: {kind}")

    hydraulic_radius = area / wetted_perimeter
    discharge_estimate = (1.0 / roughness) * area * hydraulic_radius ** (2.0 / 3.0) * np.sqrt(bed_slope)
    velocity = discharge_estimate / area
    return discharge_estimate, top_width, velocity


def solve_mannings_depth(shape_config: dict[str, float | str]) -> float:
    """Solve Manning's equation for normal depth using bisection."""
    lower_depth = 1.0e-6
    upper_depth = max(floodplain_offset, 1.0)

    while calculate_mannings_discharge(upper_depth, shape_config)[0] < mannings_discharge_m3s:
        upper_depth *= 2.0

    for _ in range(60):
        trial_depth = 0.5 * (lower_depth + upper_depth)
        if calculate_mannings_discharge(trial_depth, shape_config)[0] < mannings_discharge_m3s:
            lower_depth = trial_depth
        else:
            upper_depth = trial_depth

    return 0.5 * (lower_depth + upper_depth)


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


def _row_col_to_xy(row_idx: float, col_idx: float, raster_geotransform: tuple[float, ...]) -> tuple[float, float]:
    """Convert one raster row/column index to the center coordinates of that cell."""
    x_coord = raster_geotransform[0] + (col_idx + 0.5) * raster_geotransform[1] + (row_idx + 0.5) * raster_geotransform[2]
    y_coord = raster_geotransform[3] + (col_idx + 0.5) * raster_geotransform[4] + (row_idx + 0.5) * raster_geotransform[5]
    return x_coord, y_coord


def _remove_existing_geopackage(gpkg_path: Path) -> None:
    """
    Delete a GeoPackage and its SQLite sidecars before rewriting it.

    Manual reruns reuse stable filenames. Removing the main ``.gpkg`` plus any
    ``-wal`` and ``-shm`` companions avoids stale SQLite state and guarantees
    that each run regenerates the vector outputs from scratch.
    """
    if gpkg_path.exists():
        _unlink_with_retries(gpkg_path)

    gpkg_wal_path = gpkg_path.with_name(gpkg_path.name + "-wal")
    gpkg_shm_path = gpkg_path.with_name(gpkg_path.name + "-shm")
    if gpkg_wal_path.exists():
        _unlink_with_retries(gpkg_wal_path)
    if gpkg_shm_path.exists():
        _unlink_with_retries(gpkg_shm_path)


def export_stream_lines(stream: np.ndarray, stream_lines_path: Path) -> None:
    """
    Export one polyline per synthetic COMID from the stream raster.

    ARC's ``end_points`` stream-slope method requires a vector stream network.
    This helper converts each nonzero integer ID in the raster into one ordered
    centerline feature so the vector input uses the same geometry and COMIDs as
    the synthetic raster input.
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


def export_cross_section_lines(xs_df: pd.DataFrame, raster_path: Path, xs_lines_path: Path) -> None:
    """
    Export one polyline feature per ARC cross section using the DEM georeference.

    The geometry is reconstructed from the ARC cross-section export text file
    using the center row/column and the two far endpoint indices stored for
    each sampled section. Any existing GeoPackage at the output path is deleted
    first so reruns always regenerate the file cleanly.
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


def compute_station_from_col(col_idx: pd.Series | np.ndarray) -> pd.Series | np.ndarray:
    """
    Convert raster columns back to reach station along the active channel.
    """
    station_m = (col_idx - channel_start_col) * cellsize_m
    return np.clip(station_m, 0.0, length_m)


def run_shape_case(shape_name: str, shape_config: dict[str, float | str], output_root: Path) -> pd.DataFrame:
    """
    Create one synthetic geographic case, run ARC, and attach Manning metrics.
    """
    case_dir = output_root / shape_name
    dem = create_dem(shape_config)
    case_paths = write_case_inputs(case_dir, dem)

    Arc(
        args={
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
            "Stream_Slope_Method": "end_points",
            "StrmShp_File": str(case_paths["stream_lines_path"]),
        }
    ).set_log_level("info").run()

    vdt_df = pd.read_csv(case_paths["vdt_path"])
    xs_df = pd.read_csv(case_paths["xs_output_path"], sep="\t")
    vdt_df["Station (m)"] = compute_station_from_col(vdt_df["Col"])

    depth_y = solve_mannings_depth(shape_config)
    _, top_width_y, velocity_y = calculate_mannings_discharge(depth_y, shape_config)
    vdt_df["Manning Depth (m)"] = depth_y
    vdt_df["Manning Top Width (m)"] = top_width_y
    vdt_df["Manning Velocity (m/s)"] = velocity_y
    vdt_df["Manning WSE (m)"] = depth_y - bed_slope * vdt_df["Station (m)"]
    vdt_df["Shape"] = shape_name

    export_vdt_points(vdt_df, case_paths["vdt_points_path"])
    export_cross_section_lines(xs_df, case_paths["dem_path"], case_paths["xs_lines_path"])
    return vdt_df


def run_all_shape_cases(output_root: Path) -> dict[str, pd.DataFrame]:
    """Run every configured shape case and collect the resulting VDT tables."""
    output_root.mkdir(parents=True, exist_ok=True)
    return {
        shape_name: run_shape_case(shape_name, shape_config, output_root)
        for shape_name, shape_config in shape_configs.items()
    }


def _select_upstream_normal_flow_slice(vdt_df: pd.DataFrame) -> pd.Series:
    """Return the upstream comparison window used by the regression checks."""
    return vdt_df["Station (m)"] <= (length_m - 25.0)


def assert_shape_case_is_valid(shape_name: str, vdt_df: pd.DataFrame) -> None:
    """
    Perform regression-style assertions for one geographic shape case.
    """
    assert not vdt_df.empty, f"{shape_name} produced an empty VDT table."
    assert "wse_2" in vdt_df.columns, f"{shape_name} VDT output is missing wse_2."
    assert np.isfinite(vdt_df["wse_2"]).all(), f"{shape_name} ARC WSE contains non-finite values."
    assert np.isfinite(vdt_df["Manning WSE (m)"]).all(), f"{shape_name} Manning WSE contains non-finite values."
    assert np.isfinite(vdt_df["Manning Depth (m)"]).all(), f"{shape_name} Manning depth contains non-finite values."
    assert np.isfinite(vdt_df["Manning Top Width (m)"]).all(), f"{shape_name} Manning top width contains non-finite values."
    assert np.isfinite(vdt_df["Manning Velocity (m/s)"]).all(), f"{shape_name} Manning velocity contains non-finite values."
    assert (vdt_df["Manning Depth (m)"] > 0.0).all(), f"{shape_name} Manning depth must be positive."
    assert (vdt_df["Manning Top Width (m)"] > 0.0).all(), f"{shape_name} Manning top width must be positive."
    assert (vdt_df["Manning Velocity (m/s)"] > 0.0).all(), f"{shape_name} Manning velocity must be positive."

    upstream_df = vdt_df.loc[_select_upstream_normal_flow_slice(vdt_df)].copy()
    assert not upstream_df.empty, f"{shape_name} upstream comparison window is empty."

    wse_residual = (upstream_df["wse_2"] - upstream_df["Manning WSE (m)"]).abs()
    assert wse_residual.median() < 0.75, f"{shape_name} median upstream WSE residual is too large: {wse_residual.median():.3f} m"
    assert wse_residual.max() < 1.50, f"{shape_name} maximum upstream WSE residual is too large: {wse_residual.max():.3f} m"

    assert "t_2" in vdt_df.columns, f"{shape_name} VDT output is missing t_2 top width values."
    top_width_residual = (upstream_df["t_2"] - upstream_df["Manning Top Width (m)"]).abs()
    assert top_width_residual.median() < 2.50, (
        f"{shape_name} median upstream top-width residual is too large: "
        f"{top_width_residual.median():.3f} m"
    )
    assert top_width_residual.max() < 5.00, (
        f"{shape_name} maximum upstream top-width residual is too large: "
        f"{top_width_residual.max():.3f} m"
    )

    assert "v_2" in vdt_df.columns, f"{shape_name} VDT output is missing v_2 velocity values."
    velocity_residual = (upstream_df["v_2"] - upstream_df["Manning Velocity (m/s)"]).abs()
    assert velocity_residual.median() < 0.60, (
        f"{shape_name} median upstream velocity residual is too large: "
        f"{velocity_residual.median():.3f} m/s"
    )
    assert velocity_residual.max() < 1.25, (
        f"{shape_name} maximum upstream velocity residual is too large: "
        f"{velocity_residual.max():.3f} m/s"
    )


def summarize_results(results: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Build a compact summary table for manual runs."""
    summary_rows = []

    for shape_name, vdt_df in results.items():
        upstream_df = vdt_df.loc[_select_upstream_normal_flow_slice(vdt_df)].copy()
        wse_residual = (upstream_df["wse_2"] - upstream_df["Manning WSE (m)"]).abs()
        top_width_residual = (upstream_df["t_2"] - upstream_df["Manning Top Width (m)"]).abs()
        velocity_residual = (upstream_df["v_2"] - upstream_df["Manning Velocity (m/s)"]).abs()
        summary_rows.append(
            {
                "shape": shape_name,
                "manning_depth_m": float(vdt_df["Manning Depth (m)"].iloc[0]),
                "manning_top_width_m": float(vdt_df["Manning Top Width (m)"].iloc[0]),
                "manning_velocity_mps": float(vdt_df["Manning Velocity (m/s)"].iloc[0]),
                "median_upstream_wse_residual_m": float(wse_residual.median()),
                "max_upstream_wse_residual_m": float(wse_residual.max()),
                "median_upstream_top_width_residual_m": float(top_width_residual.median()),
                "max_upstream_top_width_residual_m": float(top_width_residual.max()),
                "median_upstream_velocity_residual_mps": float(velocity_residual.median()),
                "max_upstream_velocity_residual_mps": float(velocity_residual.max()),
                "num_vdt_rows": int(len(vdt_df)),
            }
        )

    return pd.DataFrame(summary_rows).sort_values("manning_depth_m", ascending=False).reset_index(drop=True)


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
    Plot one row per shape for manual visual inspection.
    """
    title_fontsize = 11
    label_fontsize = 8
    tick_fontsize = 8
    legend_fontsize = 8

    figure, axes = plt.subplots(len(shape_configs), 4, figsize=(16, 9), sharex="col")
    axes_array = np.atleast_2d(axes)
    all_vdt_frames = list(results.values())
    wse_ylim = _compute_fixed_ylim([df["wse_2"] for df in all_vdt_frames] + [df["Manning WSE (m)"] for df in all_vdt_frames])
    top_width_ylim = _compute_fixed_ylim([df["t_2"] for df in all_vdt_frames] + [df["Manning Top Width (m)"] for df in all_vdt_frames])
    velocity_ylim = _compute_fixed_ylim([df["v_2"] for df in all_vdt_frames] + [df["Manning Velocity (m/s)"] for df in all_vdt_frames])
    slope_ylim = _compute_fixed_ylim([df["Slope"] * 100.0 for df in all_vdt_frames])

    for axis_row, (shape_name, shape_config) in zip(axes_array, shape_configs.items()):
        vdt_df = results[shape_name]
        wse_axis = axis_row[0]
        top_width_axis = axis_row[1]
        velocity_axis = axis_row[2]
        slope_axis = axis_row[3]

        wse_axis.plot(vdt_df["Station (m)"], vdt_df["wse_2"], color=str(shape_config["color"]), label="ARC")
        wse_axis.plot(
            vdt_df["Station (m)"],
            vdt_df["Manning WSE (m)"],
            color=str(shape_config["color"]),
            linestyle="--",
            label="Manning",
        )
        wse_axis.set_ylabel("Elevation (m)", fontsize=label_fontsize)
        wse_axis.set_title(f"{shape_name.title()} Channel WSE", fontsize=title_fontsize)
        wse_axis.set_ylim(*wse_ylim)
        wse_axis.legend(fontsize=legend_fontsize)
        wse_axis.tick_params(axis="both", labelsize=tick_fontsize)
        wse_axis.grid()

        top_width_axis.plot(vdt_df["Station (m)"], vdt_df["t_2"], color=str(shape_config["color"]), label="ARC")
        top_width_axis.plot(
            vdt_df["Station (m)"],
            vdt_df["Manning Top Width (m)"],
            color=str(shape_config["color"]),
            linestyle="--",
            label="Manning",
        )
        top_width_axis.set_ylabel("Top Width (m)", fontsize=label_fontsize)
        top_width_axis.set_title(f"{shape_name.title()} Channel Top Width", fontsize=title_fontsize)
        top_width_axis.set_ylim(*top_width_ylim)
        top_width_axis.legend(fontsize=legend_fontsize)
        top_width_axis.tick_params(axis="both", labelsize=tick_fontsize)
        top_width_axis.grid()

        velocity_axis.plot(vdt_df["Station (m)"], vdt_df["v_2"], color=str(shape_config["color"]), label="ARC")
        velocity_axis.plot(
            vdt_df["Station (m)"],
            vdt_df["Manning Velocity (m/s)"],
            color=str(shape_config["color"]),
            linestyle="--",
            label="Manning",
        )
        velocity_axis.set_ylabel("Velocity (m/s)", fontsize=label_fontsize)
        velocity_axis.set_title(f"{shape_name.title()} Channel Velocity", fontsize=title_fontsize)
        velocity_axis.set_ylim(*velocity_ylim)
        velocity_axis.legend(fontsize=legend_fontsize)
        velocity_axis.tick_params(axis="both", labelsize=tick_fontsize)
        velocity_axis.grid()

        slope_axis.plot(vdt_df["Station (m)"], vdt_df["Slope"] * 100.0, color=str(shape_config["color"]), label="VDT Slope")
        slope_axis.set_ylabel("Slope (%)", fontsize=label_fontsize)
        slope_axis.set_title(f"{shape_name.title()} Channel Slope", fontsize=title_fontsize)
        slope_axis.set_ylim(*slope_ylim)
        slope_axis.legend(fontsize=legend_fontsize)
        slope_axis.tick_params(axis="both", labelsize=tick_fontsize)
        slope_axis.grid()

    axes_array[-1, 0].set_xlabel("Station (m)", fontsize=label_fontsize)
    axes_array[-1, 1].set_xlabel("Station (m)", fontsize=label_fontsize)
    axes_array[-1, 2].set_xlabel("Station (m)", fontsize=label_fontsize)
    axes_array[-1, 3].set_xlabel("Station (m)", fontsize=label_fontsize)
    figure.suptitle("Synthetic Shape Comparison in NAD83 Geographic Coordinates", fontsize=12)
    figure.tight_layout(pad=2.5, w_pad=2.0, h_pad=2.5)
    plt.show()


def test_shape_cases_against_manning(tmp_path: Path) -> None:
    """
    Automated regression test for all synthetic shapes in geographic space.
    """
    results = run_all_shape_cases(tmp_path / "shape_cases_geographic")
    for shape_name, vdt_df in results.items():
        assert_shape_case_is_valid(shape_name, vdt_df)


def test_manning_depth_ordering() -> None:
    """
    Verify the expected analytical depth ordering for the shared hydraulics.
    """
    trapezoid_depth = solve_mannings_depth(shape_configs["trapezoid"])
    rectangle_depth = solve_mannings_depth(shape_configs["rectangle"])
    triangle_depth = solve_mannings_depth(shape_configs["triangle"])
    assert triangle_depth > rectangle_depth > trapezoid_depth


def main() -> None:
    """
    Run the synthetic geographic shape workflow manually and show plots.
    """
    manual_output_root.mkdir(parents=True, exist_ok=True)
    results = run_all_shape_cases(manual_output_root)
    summary_df = summarize_results(results)

    print("Synthetic geographic shape test summary:")
    print(summary_df.to_string(index=False))
    print(f"\nOutputs written to: {manual_output_root}")
    print(
        "\nRaster CRS: EPSG:4269 (NAD83 geographic)"
        f"\nApproximate cell size: {cellsize_m:.1f} m"
        f"\nLongitude pixel width: {cellsize_x_deg:.10f} degrees"
        f"\nLatitude pixel height: {cellsize_y_deg:.10f} degrees"
    )

    plot_results(results)


if __name__ == "__main__":
    main()
