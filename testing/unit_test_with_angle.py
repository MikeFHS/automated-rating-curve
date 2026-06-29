"""
Synthetic ARC test for trapezoidal, rectangular, and triangular channels at a
user-specified ARC cross-section angle.

This module supports two execution modes:

1. Automated test mode
   ``pytest automated-rating-curve/testing/unit_test_with_angle.py``

   Pytest imports the module, uses the default channel angle, writes outputs
   into a temporary directory, and evaluates regression-style assertions.

2. Manual diagnostic mode
   ``python automated-rating-curve/testing/unit_test_with_angle.py``

   Manual mode reads the requested ``XS_Angle`` from the configuration block in
   this script. The script then:

   - rebuilds the synthetic domain so the angled reach fits inside the raster,
   - creates DEM, stream, land-cover, and flow inputs aligned to that angle,
   - exports a stream GeoPackage that ARC can read as ``StrmShp_File``,
   - runs ARC for trapezoidal, rectangular, and triangular channels,
   - summarizes the resulting hydraulics, and
   - plots one row per shape for visual inspection.

Why this file exists separately from ``unit_test_with_shapes.py``
---------------------------------------------------------------
The companion ``unit_test_with_shapes.py`` file exercises the same synthetic
channel shapes when the stream follows the raster x direction. This file keeps
the same hydraulic assumptions but rotates the stream in plan view, which is a
useful stress test for:

- ARC's local stream-direction estimation,
- cross-section orientation away from cardinal directions, and
- how rasterized geometry behaves when the thalweg is not aligned to rows or
  columns.

Angle convention
----------------
``xs_angle_rad`` follows the raw ``XS_Angle`` values written by ARC into the
VDT table. In practice that means:

- the user-facing angle is the cross-section orientation, not the stream
  orientation,
- the units are radians so the requested value numerically matches the VDT
  output, and
- the angle is measured from the positive x direction in the raster plane.

The synthetic thalweg is rotated internally so ARC should recover an
``XS_Angle`` close to the requested value throughout the reach interior.
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
# Universal hydraulic parameters shared by every synthetic case.
# -----------------------------------------------------------------------------
# The shape comparison is only meaningful when these remain the same across the
# three geometries. Angle changes the plan-view orientation of the reach, but it
# does not change the target discharge, bed slope, roughness, or cross-section
# definitions.
length_m = 1000.0
cellsize = 1.0
bed_slope = 0.001
mannings_discharge_m3s = 10.0
roughness = 0.025
floodplain_offset = 20.0

# Channel geometry shared across shapes whenever the concept applies.
bottom_width = 10.0
side_slope = 2.0

# Automated tests always use the default value so they remain deterministic.
# Manual runs use ``MANUAL_XS_ANGLE_RAD`` below, which can be edited directly in
# the script without relying on command-line arguments.
XS_ANGLE_DEG = float(10.2)
XS_ANGLE_RAD = float(np.deg2rad(10.2))


# Coordinate system and georeferencing for the synthetic rasters.
epsg = 26912
origin_x = 444000.0
origin_y = 4447000.0

# Output root for manual runs. A subdirectory that includes the requested
# ``XS_Angle`` is created beneath this root so outputs from different manual
# runs do not overwrite one another.
script_dir = Path(__file__).resolve().parent
manual_output_root = script_dir / "angle_case_outputs"

# Shape configuration dictionary shared by manual and automated workflows.
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


# -----------------------------------------------------------------------------
# Angle-dependent domain state
# -----------------------------------------------------------------------------
# These globals are configured through ``configure_xs_angle``. The requested
# user angle is stored in ARC's ``XS_Angle`` convention, while the derived
# channel-direction angle is stored separately because DEM construction follows
# the stream centerline rather than the cross section.
xs_angle_rad = XS_ANGLE_RAD
xs_angle_deg = XS_ANGLE_DEG
channel_angle_rad = np.mod((np.pi / 2.0) - xs_angle_rad, np.pi)
channel_angle_deg = float(np.rad2deg(channel_angle_rad))
channel_unit_x = np.cos(channel_angle_rad)
channel_unit_y = np.sin(channel_angle_rad)
channel_normal_x = -channel_unit_y
channel_normal_y = channel_unit_x
nx = 0
ny = 0
center_row = 0
center_col = 0
geotransform = (origin_x, cellsize, 0.0, origin_y, 0.0, -cellsize)


def configure_xs_angle(requested_xs_angle_rad: float) -> None:
    """
    Configure raster dimensions and directional vectors for a requested ``XS_Angle``.

    The user provides the same angle convention that ARC writes into the VDT
    database. Because the DEM must still be built around the stream centerline,
    this function derives the corresponding channel-direction angle first and
    then sizes the raster so the rotated reach, plus floodplain margin, fits
    inside the domain.
    """
    global xs_angle_rad
    global xs_angle_deg
    global channel_angle_deg
    global channel_angle_rad
    global channel_unit_x
    global channel_unit_y
    global channel_normal_x
    global channel_normal_y
    global nx
    global ny
    global center_row
    global center_col
    global geotransform

    if not np.isfinite(requested_xs_angle_rad):
        raise ValueError("requested_xs_angle_rad must be finite.")

    # Cross-section orientation is a line direction, so values that differ by
    # integer multiples of pi represent the same geometry. Normalize into
    # ``[0, pi)`` so directory names, summaries, and assertions are stable.
    xs_angle_rad = float(np.mod(requested_xs_angle_rad, np.pi))
    xs_angle_deg = float(np.rad2deg(xs_angle_rad))

    # ARC's ``XS_Angle`` is perpendicular to the stream centerline. Converting
    # to a conventional mathematical stream angle lets the synthetic thalweg be
    # generated in the DEM's x/y coordinate system.
    channel_angle_rad = float(np.mod((np.pi / 2.0) - xs_angle_rad, np.pi))
    channel_angle_deg = float(np.rad2deg(channel_angle_rad))
    channel_unit_x = float(np.cos(channel_angle_rad))
    channel_unit_y = float(np.sin(channel_angle_rad))
    channel_normal_x = -channel_unit_y
    channel_normal_y = channel_unit_x

    # Reserve enough room for the rotated reach plus floodplain on both sides.
    half_length_m = length_m / 2.0
    max_channel_half_width_m = max(
        bottom_width / 2.0 + side_slope * floodplain_offset,
        bottom_width / 2.0,
        side_slope * floodplain_offset,
    )
    lateral_support_m = max_channel_half_width_m + 10.0
    edge_padding_m = 20.0

    half_extent_x = (
        abs(channel_unit_x) * half_length_m
        + abs(channel_normal_x) * lateral_support_m
        + edge_padding_m
    )
    half_extent_y = (
        abs(channel_unit_y) * half_length_m
        + abs(channel_normal_y) * lateral_support_m
        + edge_padding_m
    )

    nx = 5 * int(np.ceil((2.0 * half_extent_x) / cellsize)) + 1
    ny = 2 * int(np.ceil((2.0 * half_extent_y) / cellsize)) + 1
    center_col = nx // 2
    center_row = ny // 2

    geotransform = (
        origin_x,
        cellsize,
        0.0,
        origin_y,
        0.0,
        -cellsize,
    )


def build_longitudinal_thalweg(distance_from_upstream_m: np.ndarray | float) -> np.ndarray | float:
    """
    Return the thalweg elevation at a given distance downstream from the start.

    The bed follows one consistent slope across the entire synthetic reach.
    """
    return -(distance_from_upstream_m * bed_slope)


def local_xy_from_row_col(rows: np.ndarray | pd.Series | float, cols: np.ndarray | pd.Series | float) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert raster row/column indices into local x/y coordinates in meters.

    The local origin is placed at the center of the raster. Positive x points
    toward increasing columns; positive y points upward, opposite the raster-row
    direction.
    """
    row_arr = np.asarray(rows, dtype=float)
    col_arr = np.asarray(cols, dtype=float)
    x_local = (col_arr + 0.5 - (nx / 2.0)) * cellsize
    y_local = ((ny / 2.0) - (row_arr + 0.5)) * cellsize
    return x_local, y_local


def project_row_col_to_channel(rows: np.ndarray | pd.Series | float, cols: np.ndarray | pd.Series | float) -> tuple[np.ndarray, np.ndarray]:
    """
    Project raster indices onto the synthetic stream centerline.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        ``distance_from_upstream_m`` and ``distance_from_thalweg_m``.

    The distance-to-thalweg is computed against the finite reach segment rather
    than an infinite line. That gives the synthetic reach sensible rounded end
    caps instead of extending the channel geometry indefinitely beyond the
    upstream and downstream endpoints.
    """
    x_local, y_local = local_xy_from_row_col(rows, cols)

    along_from_center = x_local * channel_unit_x + y_local * channel_unit_y
    along_from_center_clamped = np.clip(along_from_center, -(length_m / 2.0), length_m / 2.0)

    closest_x = along_from_center_clamped * channel_unit_x
    closest_y = along_from_center_clamped * channel_unit_y

    dx = x_local - closest_x
    dy = y_local - closest_y

    distance_from_upstream_m = along_from_center_clamped + (length_m / 2.0)
    distance_from_thalweg_m = np.hypot(dx, dy)
    return distance_from_upstream_m, distance_from_thalweg_m


def lateral_elevation_above_thalweg(offset_m: np.ndarray | float, shape_config: dict[str, float | str]) -> np.ndarray | float:
    """
    Return channel elevation above the thalweg for a lateral offset.

    The return value is capped at ``floodplain_offset`` so every synthetic
    channel ties into the same surrounding floodplain.
    """
    offset_arr = np.asarray(offset_m, dtype=float)
    kind = shape_config["kind"]

    if kind == "trapezoid":
        half_bottom_width = float(shape_config["bottom_width"]) / 2.0
        return np.where(
            offset_arr <= half_bottom_width,
            0.0,
            np.minimum(
                floodplain_offset,
                (offset_arr - half_bottom_width) / float(shape_config["side_slope"]),
            ),
        )

    if kind == "rectangle":
        return np.where(offset_arr <= (float(shape_config["width"]) / 2.0), 0.0, floodplain_offset)

    if kind == "triangle":
        return np.minimum(floodplain_offset, offset_arr / float(shape_config["side_slope"]))

    raise ValueError(f"Unsupported shape kind: {kind}")


def create_dem(
    shape_config: dict[str, float | str],
    station_grid_m: np.ndarray,
    distance_to_stream_m: np.ndarray,
) -> np.ndarray:
    """
    Build a DEM whose thalweg follows the rasterized stream path.

    The stream raster and the DEM now share the same centerline support. Bed
    elevation is assigned from ``station_grid_m``, which is derived from the
    rasterized stream path, and cross-section elevation is assigned from
    ``distance_to_stream_m``, which measures distance to the nearest rasterized
    stream cell center. This keeps stream cells on the DEM thalweg instead of
    allowing rasterization offsets to create artificial cross-stream elevation
    differences inside the stream network itself.
    """
    thalweg_z = build_longitudinal_thalweg(station_grid_m)
    lateral_z = lateral_elevation_above_thalweg(distance_to_stream_m, shape_config)
    return (thalweg_z + lateral_z).astype(np.float32)


def clean_stream_raster_like_nencarta(stream_raster: np.ndarray) -> np.ndarray:
    """
    Remove isolated and redundant stream cells using Nencarta's cleanup logic.

    This mirrors the two-pass cleanup used in ``nencarta.main._clean_stream_raster``.
    The intent is to thin a rasterized stream centerline by removing cells that
    are redundant artifacts of rasterization while preserving connectivity.
    """
    nrows, ncols = stream_raster.shape
    cleaned = np.pad(stream_raster.astype(np.int64), pad_width=1, mode="constant")
    cleaned = np.where(cleaned > 0, cleaned, 0)

    row_indices, col_indices = np.where(cleaned > 0)
    num_nonzero = len(row_indices)

    for _ in range(2):
        for index in range(num_nonzero):
            row_idx = row_indices[index]
            col_idx = col_indices[index]
            cell_value = cleaned[row_idx, col_idx]
            if cell_value <= 0:
                continue

            # Left and right are empty, so check whether this is a dangling
            # vertical artifact cell that does not contribute to connectivity.
            if cleaned[row_idx, col_idx + 1] == 0 and cleaned[row_idx, col_idx - 1] == 0:
                if (
                    cleaned[row_idx + 1, col_idx - 1]
                    + cleaned[row_idx + 1, col_idx]
                    + cleaned[row_idx + 1, col_idx + 1]
                ) == 0 and cleaned[row_idx - 1, col_idx] > 0:
                    cleaned[row_idx, col_idx] = 0
                elif (
                    cleaned[row_idx - 1, col_idx - 1]
                    + cleaned[row_idx - 1, col_idx]
                    + cleaned[row_idx - 1, col_idx + 1]
                ) == 0 and cleaned[row_idx + 1, col_idx] > 0:
                    cleaned[row_idx, col_idx] = 0

            # Top and bottom are empty, so check for horizontal dangling cells.
            if cleaned[row_idx, col_idx] > 0 and cleaned[row_idx + 1, col_idx] == 0 and cleaned[row_idx - 1, col_idx] == 0:
                if (
                    cleaned[row_idx + 1, col_idx + 1]
                    + cleaned[row_idx, col_idx + 1]
                    + cleaned[row_idx - 1, col_idx + 1]
                ) == 0 and cleaned[row_idx, col_idx - 1] > 0:
                    cleaned[row_idx, col_idx] = 0
                elif (
                    cleaned[row_idx + 1, col_idx - 1]
                    + cleaned[row_idx, col_idx - 1]
                    + cleaned[row_idx - 1, col_idx - 1]
                ) == 0 and cleaned[row_idx, col_idx + 1] > 0:
                    cleaned[row_idx, col_idx] = 0

        for index in range(num_nonzero):
            row_idx = row_indices[index]
            col_idx = col_indices[index]
            cell_value = cleaned[row_idx, col_idx]
            if cell_value <= 0:
                continue

            # Remove one of two adjacent equal-valued cells when a diagonal
            # neighbor already preserves the local path shape.
            if cleaned[row_idx + 1, col_idx] == cell_value and (
                cleaned[row_idx + 1, col_idx + 1] == cell_value or cleaned[row_idx + 1, col_idx - 1] == cell_value
            ):
                if np.sum(cleaned[row_idx + 1, col_idx - 1:col_idx + 2]) == 0:
                    cleaned[row_idx + 1, col_idx] = 0
            elif cleaned[row_idx - 1, col_idx] == cell_value and (
                cleaned[row_idx - 1, col_idx + 1] == cell_value or cleaned[row_idx - 1, col_idx - 1] == cell_value
            ):
                if np.sum(cleaned[row_idx - 1, col_idx - 1:col_idx + 2]) == 0:
                    cleaned[row_idx - 1, col_idx] = 0
            elif cleaned[row_idx, col_idx + 1] == cell_value and (
                cleaned[row_idx + 1, col_idx + 1] == cell_value or cleaned[row_idx - 1, col_idx + 1] == cell_value
            ):
                if np.sum(cleaned[row_idx - 1:row_idx + 1, col_idx + 2]) == 0:
                    cleaned[row_idx, col_idx + 1] = 0
            elif cleaned[row_idx, col_idx - 1] == cell_value and (
                cleaned[row_idx + 1, col_idx - 1] == cell_value or cleaned[row_idx - 1, col_idx - 1] == cell_value
            ):
                if np.sum(cleaned[row_idx - 1:row_idx + 1, col_idx - 2]) == 0:
                    cleaned[row_idx, col_idx - 1] = 0

    return cleaned[1:nrows + 1, 1:ncols + 1].astype(np.uint8)


def create_stream_raster() -> np.ndarray:
    """
    Create a one-cell-wide stream centerline raster for ARC at the current angle.

    The centerline is sampled at sub-cell spacing along the synthetic thalweg so
    diagonal reaches remain continuous in raster form. After rasterization, the
    stream mask is cleaned with the same redundant-cell thinning logic used in
    ``nencarta.main`` so the synthetic test better matches production behavior.
    """
    stream = np.zeros((ny, nx), dtype=np.uint8)

    sample_spacing_m = cellsize * 0.25
    sample_distances = np.arange(0.0, length_m + sample_spacing_m, sample_spacing_m)
    along_from_center = sample_distances - (length_m / 2.0)

    x_local = along_from_center * channel_unit_x
    y_local = along_from_center * channel_unit_y

    col_idx = np.rint((x_local / cellsize) + (nx / 2.0) - 0.5).astype(int)
    row_idx = np.rint((ny / 2.0) - (y_local / cellsize) - 0.5).astype(int)

    valid = (
        (row_idx >= 0)
        & (row_idx < ny)
        & (col_idx >= 0)
        & (col_idx < nx)
    )
    stream[row_idx[valid], col_idx[valid]] = 1
    return clean_stream_raster_like_nencarta(stream)


def _build_ordered_stream_support(stream: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Order rasterized stream cells from upstream to downstream.

    The ordering begins with the analytical rotated-centerline projection only
    to recover upstream/downstream sequence. Once ordered, cumulative distance
    is computed along the rasterized cell centers themselves and normalized to
    ``length_m`` so the DEM and Manning reference preserve the intended nominal
    reach length while sharing the rasterized stream path.
    """
    stream_rows, stream_cols = np.where(stream > 0)
    if len(stream_rows) < 2:
        raise ValueError("The stream raster must contain at least two stream cells.")

    approx_station_m, _ = project_row_col_to_channel(stream_rows.astype(float), stream_cols.astype(float))
    order = np.lexsort((stream_cols, stream_rows, approx_station_m))
    ordered_rows = stream_rows[order]
    ordered_cols = stream_cols[order]

    keep = np.ones(len(ordered_rows), dtype=bool)
    if len(ordered_rows) > 1:
        keep[1:] = (ordered_rows[1:] != ordered_rows[:-1]) | (ordered_cols[1:] != ordered_cols[:-1])
    ordered_rows = ordered_rows[keep]
    ordered_cols = ordered_cols[keep]

    ordered_x_m, ordered_y_m = local_xy_from_row_col(ordered_rows.astype(float), ordered_cols.astype(float))
    ordered_station_m = np.zeros(len(ordered_rows), dtype=float)
    if len(ordered_rows) > 1:
        segment_lengths_m = np.hypot(np.diff(ordered_x_m), np.diff(ordered_y_m))
        ordered_station_m[1:] = np.cumsum(segment_lengths_m)
        if ordered_station_m[-1] > 0.0:
            ordered_station_m *= length_m / ordered_station_m[-1]

    return ordered_rows, ordered_cols, ordered_x_m, ordered_y_m, ordered_station_m


def build_stream_reference_grids(stream: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Build station and distance grids from the rasterized stream path.

    ``station_grid_m`` stores the along-stream station of the nearest stream
    cell for every raster cell. ``distance_to_stream_m`` stores the Euclidean
    distance to that same nearest stream cell center. Together they let the DEM
    be carved around the rasterized path that ARC will actually use.
    """
    _, _, stream_x_m, stream_y_m, stream_station_m = _build_ordered_stream_support(stream)

    row_grid, col_grid = np.indices((ny, nx), dtype=float)
    query_x_m, query_y_m = local_xy_from_row_col(row_grid.ravel(), col_grid.ravel())

    nearest_station_m = np.empty(query_x_m.size, dtype=np.float32)
    nearest_dist2_m = np.empty(query_x_m.size, dtype=np.float64)

    chunk_size = 2048
    for start in range(0, query_x_m.size, chunk_size):
        end = min(start + chunk_size, query_x_m.size)
        dx_m = query_x_m[start:end, None] - stream_x_m[None, :]
        dy_m = query_y_m[start:end, None] - stream_y_m[None, :]
        dist2_m = dx_m * dx_m + dy_m * dy_m
        nearest_idx = np.argmin(dist2_m, axis=1)
        nearest_station_m[start:end] = stream_station_m[nearest_idx]
        nearest_dist2_m[start:end] = dist2_m[np.arange(end - start), nearest_idx]

    station_grid_m = nearest_station_m.reshape((ny, nx))
    distance_to_stream_m = np.sqrt(nearest_dist2_m).reshape((ny, nx)).astype(np.float32)
    return station_grid_m, distance_to_stream_m


def create_land_cover_raster() -> np.ndarray:
    """Create a uniform land-cover raster so Manning n is constant everywhere."""
    return np.ones((ny, nx), dtype=np.uint8)


def write_raster(path: Path, array: np.ndarray, gdal_dtype: int, wkt: str) -> None:
    """Write a numpy array to GeoTIFF using the shared georeferencing."""
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
    ``testing/angle_case_outputs``. Clearing the case directory before writing
    new inputs prevents stale rasters, vector files, and VDT outputs from being
    reused accidentally across runs.
    """
    if not case_dir.exists():
        return

    for child in case_dir.iterdir():
        if child.is_dir():
            clear_case_directory(child)
            _rmdir_with_retries(child)
        else:
            _unlink_with_retries(child)


def write_case_inputs(case_dir: Path, dem: np.ndarray, stream: np.ndarray) -> dict[str, Path]:
    """
    Write all ARC inputs needed for one synthetic case.

    Before writing new files, the case directory contents are cleared so every
    simulation starts from a clean set of inputs and outputs.
    """
    clear_case_directory(case_dir)
    case_dir.mkdir(parents=True, exist_ok=True)

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
            "COMID": [1],
            "baseflow": [0.0],
            "maxflow": [mannings_discharge_m3s],
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


def calculate_mannings_discharge(
    depth_m: float, shape_config: dict[str, float | str]
) -> tuple[float, float, float]:
    """
    Calculate discharge, top width, and velocity for the requested depth.

    The plan-view angle does not enter this calculation directly. Once a
    cross-section shape is defined, the Manning reference remains a function of
    depth, section geometry, roughness, and slope.
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


def compute_station_from_row_col(rows: pd.Series, cols: pd.Series, station_grid_m: np.ndarray) -> np.ndarray:
    """
    Convert VDT row/column positions into streamwise station coordinates.

    Station is looked up from the precomputed raster-path station grid so the
    Manning reference and the DEM use the same streamwise support.
    """
    row_idx = np.clip(np.rint(np.asarray(rows, dtype=float)).astype(int), 0, ny - 1)
    col_idx = np.clip(np.rint(np.asarray(cols, dtype=float)).astype(int), 0, nx - 1)
    return station_grid_m[row_idx, col_idx]


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

    Repeated manual runs reuse the same filenames. Removing the main ``.gpkg``
    file plus any ``-wal`` and ``-shm`` sidecars prevents stale SQLite state
    from leaking into a later run.
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
    Export the cleaned stream raster as a line GeoPackage for ARC.

    ARC's ``end_points`` slope option reads vector stream lines from
    ``StrmShp_File``. This export is intentionally derived from the cleaned
    raster stream cells, not from the idealized analytic centerline, so the
    vector path and raster path describe the same synthetic geometry. The
    single exported feature carries ``COMID = 1`` to match the synthetic flow
    table used in this test.
    """
    _remove_existing_geopackage(stream_lines_path)

    stream_rows, stream_cols, _, _, _ = _build_ordered_stream_support(stream)
    line_points: list[tuple[float, float]] = []
    for row_idx, col_idx in zip(stream_rows, stream_cols):
        x_coord, y_coord = _row_col_to_xy(float(row_idx), float(col_idx), geotransform)
        if not line_points or line_points[-1] != (x_coord, y_coord):
            line_points.append((x_coord, y_coord))

    if len(line_points) < 2:
        raise ValueError("The ordered stream line must contain at least two unique vertices.")

    stream_lines_gdf = gpd.GeoDataFrame(
        [{"COMID": 1, "geometry": LineString(line_points)}],
        geometry="geometry",
        crs=f"EPSG:{epsg}",
    )
    stream_lines_gdf.to_file(stream_lines_path, driver="GPKG", layer="streams")


def export_cross_section_lines(xs_df: pd.DataFrame, raster_path: Path, xs_lines_path: Path) -> None:
    """
    Export one polyline feature per ARC cross section using the DEM georeference.

    The cross-section text export stores the center row/column plus the far
    endpoint on each side. Those indices are converted to map coordinates with
    the raster geotransform so every feature represents the sampled section as
    endpoint 1 -> center -> endpoint 2. Any existing GeoPackage at the output
    path is removed first so reruns always start from a clean vector file.
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


def run_shape_case(shape_name: str, shape_config: dict[str, float | str], output_root: Path) -> pd.DataFrame:
    """
    Create one synthetic case, run ARC, and attach the Manning reference.
    """
    case_dir = output_root / shape_name
    stream = create_stream_raster()
    station_grid_m, distance_to_stream_m = build_stream_reference_grids(stream)
    dem = create_dem(shape_config, station_grid_m, distance_to_stream_m)
    case_paths = write_case_inputs(case_dir, dem, stream)

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
    vdt_df["Station (m)"] = compute_station_from_row_col(vdt_df["Row"], vdt_df["Col"], station_grid_m)

    depth_y = solve_mannings_depth(shape_config)
    _, top_width_y, velocity_y = calculate_mannings_discharge(depth_y, shape_config)
    vdt_df["Manning Depth (m)"] = depth_y
    vdt_df["Manning Top Width (m)"] = top_width_y
    vdt_df["Manning Velocity (m/s)"] = velocity_y
    vdt_df["Manning WSE (m)"] = depth_y - bed_slope * vdt_df["Station (m)"]
    vdt_df["Shape"] = shape_name
    vdt_df["Requested XS_Angle (rad)"] = xs_angle_rad
    vdt_df["Requested XS_Angle (deg)"] = xs_angle_deg
    vdt_df["Channel Angle (deg)"] = channel_angle_deg

    export_vdt_points(vdt_df, case_paths["vdt_points_path"])
    export_cross_section_lines(xs_df, case_paths["dem_path"], case_paths["xs_lines_path"])
    return vdt_df


def run_all_shape_cases(output_root: Path, requested_xs_angle_rad: float = XS_ANGLE_RAD) -> dict[str, pd.DataFrame]:
    """
    Run every configured shape case at the requested ``XS_Angle``.

    The requested ARC-style cross-section angle is applied once before any
    shape case is generated so DEM, stream raster, VDT stationing, and plots
    all share the same geometry.
    """
    configure_xs_angle(requested_xs_angle_rad)
    output_root.mkdir(parents=True, exist_ok=True)
    return {
        shape_name: run_shape_case(shape_name, shape_config, output_root)
        for shape_name, shape_config in shape_configs.items()
    }


def _select_core_reach_slice(vdt_df: pd.DataFrame) -> pd.Series:
    """
    Return a boolean mask for the regression-comparison reach.

    The mask trims a small margin from each end of the stream so the automated
    checks focus on the interior of the synthetic reach rather than endpoint
    cells, which are more sensitive to rasterized geometry and neighborhood
    effects.
    """
    return (vdt_df["Station (m)"] >= 25.0) & (vdt_df["Station (m)"] <= (length_m - 25.0))


def assert_shape_case_is_valid(shape_name: str, vdt_df: pd.DataFrame) -> None:
    """
    Perform regression-style assertions for one angled shape case.
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
    assert "XS_Angle" in vdt_df.columns, f"{shape_name} VDT output is missing XS_Angle."

    core_df = vdt_df.loc[_select_core_reach_slice(vdt_df)].copy()
    assert not core_df.empty, f"{shape_name} core comparison window is empty."

    xs_angle_residual = (core_df["XS_Angle"] - xs_angle_rad).abs()
    assert xs_angle_residual.median() < 0.08, (
        f"{shape_name} median core XS_Angle residual is too large: "
        f"{xs_angle_residual.median():.3f} rad"
    )
    assert xs_angle_residual.max() < 0.18, (
        f"{shape_name} maximum core XS_Angle residual is too large: "
        f"{xs_angle_residual.max():.3f} rad"
    )

    wse_residual = (core_df["wse_2"] - core_df["Manning WSE (m)"]).abs()
    assert wse_residual.median() < 0.90, f"{shape_name} median core WSE residual is too large: {wse_residual.median():.3f} m"
    assert wse_residual.max() < 1.75, f"{shape_name} maximum core WSE residual is too large: {wse_residual.max():.3f} m"

    assert "t_2" in vdt_df.columns, f"{shape_name} VDT output is missing t_2 top width values."
    top_width_residual = (core_df["t_2"] - core_df["Manning Top Width (m)"]).abs()
    assert top_width_residual.median() < 3.00, (
        f"{shape_name} median core top-width residual is too large: "
        f"{top_width_residual.median():.3f} m"
    )
    assert top_width_residual.max() < 6.00, (
        f"{shape_name} maximum core top-width residual is too large: "
        f"{top_width_residual.max():.3f} m"
    )

    assert "v_2" in vdt_df.columns, f"{shape_name} VDT output is missing v_2 velocity values."
    velocity_residual = (core_df["v_2"] - core_df["Manning Velocity (m/s)"]).abs()
    assert velocity_residual.median() < 0.75, (
        f"{shape_name} median core velocity residual is too large: "
        f"{velocity_residual.median():.3f} m/s"
    )
    assert velocity_residual.max() < 1.50, (
        f"{shape_name} maximum core velocity residual is too large: "
        f"{velocity_residual.max():.3f} m/s"
    )


def summarize_results(results: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Build a compact summary table for manual runs."""
    summary_rows = []

    for shape_name, vdt_df in results.items():
        core_df = vdt_df.loc[_select_core_reach_slice(vdt_df)].copy()
        wse_residual = (core_df["wse_2"] - core_df["Manning WSE (m)"]).abs()
        top_width_residual = (core_df["t_2"] - core_df["Manning Top Width (m)"]).abs()
        velocity_residual = (core_df["v_2"] - core_df["Manning Velocity (m/s)"]).abs()
        summary_rows.append(
            {
                "shape": shape_name,
                "requested_xs_angle_rad": xs_angle_rad,
                "requested_xs_angle_deg": xs_angle_deg,
                "derived_channel_angle_deg": channel_angle_deg,
                "median_xs_angle_rad": float(core_df["XS_Angle"].median()),
                "max_abs_xs_angle_residual_rad": float((core_df["XS_Angle"] - xs_angle_rad).abs().max()),
                "manning_depth_m": float(vdt_df["Manning Depth (m)"].iloc[0]),
                "manning_top_width_m": float(vdt_df["Manning Top Width (m)"].iloc[0]),
                "manning_velocity_mps": float(vdt_df["Manning Velocity (m/s)"].iloc[0]),
                "median_core_wse_residual_m": float(wse_residual.median()),
                "max_core_wse_residual_m": float(wse_residual.max()),
                "median_core_top_width_residual_m": float(top_width_residual.median()),
                "max_core_top_width_residual_m": float(top_width_residual.max()),
                "median_core_velocity_residual_mps": float(velocity_residual.median()),
                "max_core_velocity_residual_mps": float(velocity_residual.max()),
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

    The first column compares ARC WSE against the analytical Manning WSE. The
    second column compares ARC top width against the analytical Manning top
    width. The third column compares ARC velocity against the analytical Manning
    velocity. The fourth column shows the per-row slope values stored in the
    VDT output. The fifth column shows ARC's reported ``XS_Angle`` values from
    the VDT output against the requested synthetic ``XS_Angle``.
    """
    title_fontsize = 11
    label_fontsize = 8
    tick_fontsize = 8
    legend_fontsize = 8

    figure, axes = plt.subplots(len(shape_configs), 5, figsize=(20, 9), sharex="col")
    axes_array = np.atleast_2d(axes)
    all_vdt_frames = list(results.values())
    wse_ylim = _compute_fixed_ylim([df["wse_2"] for df in all_vdt_frames] + [df["Manning WSE (m)"] for df in all_vdt_frames])
    top_width_ylim = _compute_fixed_ylim([df["t_2"] for df in all_vdt_frames] + [df["Manning Top Width (m)"] for df in all_vdt_frames])
    velocity_ylim = _compute_fixed_ylim([df["v_2"] for df in all_vdt_frames] + [df["Manning Velocity (m/s)"] for df in all_vdt_frames])
    slope_ylim = _compute_fixed_ylim([df["Slope"] * 100.0 for df in all_vdt_frames])
    xs_angle_ylim = _compute_fixed_ylim([df["XS_Angle"] for df in all_vdt_frames] + [np.array([xs_angle_rad])])

    for axis_row, (shape_name, shape_config) in zip(axes_array, shape_configs.items()):
        vdt_df = results[shape_name]
        wse_axis = axis_row[0]
        top_width_axis = axis_row[1]
        velocity_axis = axis_row[2]
        slope_axis = axis_row[3]
        xs_angle_axis = axis_row[4]

        wse_axis.plot(vdt_df["Station (m)"], vdt_df["wse_2"], color=str(shape_config["color"]), label="ARC")
        wse_axis.plot(vdt_df["Station (m)"], vdt_df["Manning WSE (m)"], color=str(shape_config["color"]), linestyle="--", label="Manning")
        wse_axis.set_ylabel("Elevation (m)", fontsize=label_fontsize)
        wse_axis.set_title(f"{shape_name.title()} WSE", fontsize=title_fontsize)
        wse_axis.set_ylim(*wse_ylim)
        wse_axis.legend(fontsize=legend_fontsize)
        wse_axis.tick_params(axis="both", labelsize=tick_fontsize)
        wse_axis.grid()

        top_width_axis.plot(vdt_df["Station (m)"], vdt_df["t_2"], color=str(shape_config["color"]), label="ARC")
        top_width_axis.plot(vdt_df["Station (m)"], vdt_df["Manning Top Width (m)"], color=str(shape_config["color"]), linestyle="--", label="Manning")
        top_width_axis.set_ylabel("Top Width (m)", fontsize=label_fontsize)
        top_width_axis.set_title(f"{shape_name.title()} Top Width", fontsize=title_fontsize)
        top_width_axis.set_ylim(*top_width_ylim)
        top_width_axis.legend(fontsize=legend_fontsize)
        top_width_axis.tick_params(axis="both", labelsize=tick_fontsize)
        top_width_axis.grid()

        velocity_axis.plot(vdt_df["Station (m)"], vdt_df["v_2"], color=str(shape_config["color"]), label="ARC")
        velocity_axis.plot(vdt_df["Station (m)"], vdt_df["Manning Velocity (m/s)"], color=str(shape_config["color"]), linestyle="--", label="Manning")
        velocity_axis.set_ylabel("Velocity (m/s)", fontsize=label_fontsize)
        velocity_axis.set_title(f"{shape_name.title()} Velocity", fontsize=title_fontsize)
        velocity_axis.set_ylim(*velocity_ylim)
        velocity_axis.legend(fontsize=legend_fontsize)
        velocity_axis.tick_params(axis="both", labelsize=tick_fontsize)
        velocity_axis.grid()

        slope_axis.plot(vdt_df["Station (m)"], vdt_df["Slope"] * 100.0, color=str(shape_config["color"]), label="VDT Slope")
        slope_axis.set_ylabel("Slope (%)", fontsize=label_fontsize)
        slope_axis.set_title(f"{shape_name.title()} Slope", fontsize=title_fontsize)
        slope_axis.set_ylim(*slope_ylim)
        slope_axis.legend(fontsize=legend_fontsize)
        slope_axis.tick_params(axis="both", labelsize=tick_fontsize)
        slope_axis.grid()

        xs_angle_axis.plot(
            vdt_df["Station (m)"],
            vdt_df["XS_Angle"],
            color=str(shape_config["color"]),
            label="VDT XS_Angle",
        )
        xs_angle_axis.axhline(
            xs_angle_rad,
            color=str(shape_config["color"]),
            linestyle="--",
            label="Requested XS_Angle",
        )
        xs_angle_axis.set_ylabel("Angle (rad)", fontsize=label_fontsize)
        xs_angle_axis.set_title(f"{shape_name.title()} XS_Angle", fontsize=title_fontsize)
        xs_angle_axis.set_ylim(*xs_angle_ylim)
        xs_angle_axis.legend(fontsize=legend_fontsize)
        xs_angle_axis.tick_params(axis="both", labelsize=tick_fontsize)
        xs_angle_axis.grid()

    axes_array[-1, 0].set_xlabel("Station (m)", fontsize=label_fontsize)
    axes_array[-1, 1].set_xlabel("Station (m)", fontsize=label_fontsize)
    axes_array[-1, 2].set_xlabel("Station (m)", fontsize=label_fontsize)
    axes_array[-1, 3].set_xlabel("Station (m)", fontsize=label_fontsize)
    axes_array[-1, 4].set_xlabel("Station (m)", fontsize=label_fontsize)
    figure.suptitle(
        (
            "Synthetic Shape Comparison "
            f"(requested XS_Angle={xs_angle_rad:.3f} rad, "
            f"derived stream angle={channel_angle_deg:.1f} deg)"
        ),
        fontsize=12,
    )
    figure.tight_layout(pad=2.5, w_pad=2.0, h_pad=2.5)
    plt.show()


def test_shape_cases_against_manning(tmp_path: Path) -> None:
    """Automated regression test for all synthetic shapes at the default ``XS_Angle``."""
    results = run_all_shape_cases(tmp_path / "angle_cases", requested_xs_angle_rad=XS_ANGLE_RAD)
    for shape_name, vdt_df in results.items():
        assert_shape_case_is_valid(shape_name, vdt_df)


def test_manning_depth_ordering() -> None:
    """
    Verify the expected analytical depth ordering for the shared hydraulics.

    Angle does not change the one-dimensional Manning depth ordering for the
    idealized cross sections, so this check is independent of the plan-view
    channel angle.
    """
    trapezoid_depth = solve_mannings_depth(shape_configs["trapezoid"])
    rectangle_depth = solve_mannings_depth(shape_configs["rectangle"])
    triangle_depth = solve_mannings_depth(shape_configs["triangle"])
    assert triangle_depth > rectangle_depth > trapezoid_depth


def build_manual_output_root(requested_xs_angle_rad: float) -> Path:
    """
    Return a persistent manual-output directory name for the requested ``XS_Angle``.
    """
    angle_label = f"{requested_xs_angle_rad:.3f}".replace(".", "p")
    return manual_output_root / f"xs_angle_{angle_label}_rad"


def main(requested_xs_angle_rad: float = XS_ANGLE_RAD) -> None:
    """
    Run the synthetic angled-shape workflow manually and show diagnostic plots.

    To change the manual-run angle, update ``MANUAL_XS_ANGLE_RAD`` in the
    configuration section near the top of this file.
    """
    output_root = build_manual_output_root(requested_xs_angle_rad)
    output_root.mkdir(parents=True, exist_ok=True)

    results = run_all_shape_cases(output_root, requested_xs_angle_rad=requested_xs_angle_rad)
    summary_df = summarize_results(results)

    print(
        "Synthetic angle test summary for "
        f"requested XS_Angle={xs_angle_rad:.3f} rad "
        f"({xs_angle_deg:.1f} deg; derived stream angle={channel_angle_deg:.1f} deg):"
    )
    print(summary_df.to_string(index=False))
    print(f"\nOutputs written to: {output_root}")

    plot_results(results)


configure_xs_angle(XS_ANGLE_RAD)


if __name__ == "__main__":
    main(requested_xs_angle_rad=XS_ANGLE_RAD)
