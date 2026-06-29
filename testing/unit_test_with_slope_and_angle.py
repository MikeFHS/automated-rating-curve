"""
Synthetic ARC test that compares stream-slope methods on one angled triangle channel.

This file combines the two main synthetic-testing ideas already used in this
repository:

- ``unit_test_with_slope.py``: vary ARC's stream-slope method while holding
  the channel geometry fixed.
- ``unit_test_with_angle.py``: rotate the synthetic thalweg so the reach is
  not aligned with the raster rows or columns.

The result is a focused test that keeps the cross section fixed to one
triangular shape, keeps the hydraulic forcing fixed, and only varies:

- ``local_average``
- ``reach_average``
- ``end_points``

while the stream itself is rotated in plan view.

Execution modes
---------------
1. Automated test mode
   ``pytest automated-rating-curve/testing/unit_test_with_slope_and_angle.py``

   Pytest imports the module, runs the test function, writes outputs into a
   temporary directory, and checks that ARC can process the angled triangle
   reach under all three requested slope methods.

2. Manual diagnostic mode
   ``python automated-rating-curve/testing/unit_test_with_slope_and_angle.py``

   Manual mode writes persistent artifacts under
   ``testing/slope_method_angle_case_outputs/``, prints a compact summary
   table, and opens a diagnostic figure with one row per slope method.

Angle convention
----------------
The configurable angle in this file follows the same convention used by ARC's
``XS_Angle`` output in the VDT table. That means the user-facing angle is the
cross-section orientation, expressed in radians, and the stream centerline is
derived internally as the perpendicular direction.
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
# Universal hydraulic parameters
# -----------------------------------------------------------------------------
# The only intended differences among cases in this file are the ARC
# stream-slope method and the fact that the stream is rotated in plan view. All
# other geometry and hydraulic assumptions remain fixed.
length_m = 1000.0
cellsize = 1.0
bed_slope = 0.001
mannings_discharge_m3s = 10.0
roughness = 0.025
floodplain_offset = 20.0

# Triangle-only geometry. ``side_slope`` is the horizontal-to-vertical ratio.
side_slope = 2.0

# Automated tests always use this default angle so they remain deterministic.
# Manual runs can change the value directly in the script and rerun the file.
XS_ANGLE_DEG = float(10.2)
XS_ANGLE_RAD = float(np.deg2rad(10.2))

# Coordinate system and georeferencing for the synthetic rasters.
epsg = 26912
origin_x = 444000.0
origin_y = 4447000.0

# Manual outputs are written here for later inspection.
script_dir = Path(__file__).resolve().parent
manual_output_root = script_dir / "slope_method_angle_case_outputs"

# ARC case configuration for the three requested slope methods.
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


# -----------------------------------------------------------------------------
# Angle-dependent domain state
# -----------------------------------------------------------------------------
# ``configure_xs_angle`` updates these globals so the synthetic domain can be
# rebuilt for the requested ARC-style cross-section angle.
xs_angle_rad = XS_ANGLE_RAD
xs_angle_deg = XS_ANGLE_DEG
channel_angle_rad = np.mod((np.pi / 2.0) - xs_angle_rad, np.pi)
channel_angle_deg = float(np.rad2deg(channel_angle_rad))
channel_unit_x = float(np.cos(channel_angle_rad))
channel_unit_y = float(np.sin(channel_angle_rad))
channel_normal_x = -channel_unit_y
channel_normal_y = channel_unit_x
nx = 0
ny = 0
center_row = 0
center_col = 0
geotransform = (origin_x, cellsize, 0.0, origin_y, 0.0, -cellsize)


def configure_xs_angle(requested_xs_angle_rad: float) -> None:
    """
    Configure raster dimensions and directional vectors for one ``XS_Angle``.

    The user provides ARC's cross-section-angle convention. This function
    converts that value to the stream-centerline direction, then sizes the
    raster so the rotated reach plus its floodplain support fit comfortably
    inside the domain.
    """
    global xs_angle_rad
    global xs_angle_deg
    global channel_angle_rad
    global channel_angle_deg
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

    # Line orientations are periodic over pi rather than 2*pi. Normalizing the
    # angle here keeps output paths, summaries, and assertions stable.
    xs_angle_rad = float(np.mod(requested_xs_angle_rad, np.pi))
    xs_angle_deg = float(np.rad2deg(xs_angle_rad))

    # ARC's ``XS_Angle`` is perpendicular to the stream centerline, so derive
    # the corresponding mathematical stream direction for DEM construction.
    channel_angle_rad = float(np.mod((np.pi / 2.0) - xs_angle_rad, np.pi))
    channel_angle_deg = float(np.rad2deg(channel_angle_rad))
    channel_unit_x = float(np.cos(channel_angle_rad))
    channel_unit_y = float(np.sin(channel_angle_rad))
    channel_normal_x = -channel_unit_y
    channel_normal_y = channel_unit_x

    # Size the raster based on the rotated reach and the lateral floodplain
    # extent implied by the triangular channel.
    half_length_m = length_m / 2.0
    max_channel_half_width_m = side_slope * floodplain_offset
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
    Return the thalweg elevation at a given downstream distance.

    The synthetic reach is uniform, so the bed follows one consistent slope.
    """
    return -(distance_from_upstream_m * bed_slope)


def local_xy_from_row_col(rows: np.ndarray | pd.Series | float, cols: np.ndarray | pd.Series | float) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert raster row/column indices into local x/y coordinates in meters.

    The local origin is centered in the raster. Positive x points toward
    increasing columns; positive y points upward, opposite the raster-row
    direction.
    """
    row_arr = np.asarray(rows, dtype=float)
    col_arr = np.asarray(cols, dtype=float)
    x_local = (col_arr + 0.5 - (nx / 2.0)) * cellsize
    y_local = ((ny / 2.0) - (row_arr + 0.5)) * cellsize
    return x_local, y_local


def project_row_col_to_channel(rows: np.ndarray | pd.Series | float, cols: np.ndarray | pd.Series | float) -> tuple[np.ndarray, np.ndarray]:
    """
    Project raster indices onto the rotated stream centerline.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        ``distance_from_upstream_m`` and ``distance_from_thalweg_m``.

    The lateral distance is measured to the finite reach segment rather than an
    infinite line so the synthetic channel has sensible rounded ends.
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


def lateral_elevation_above_thalweg(offset_m: np.ndarray | float) -> np.ndarray | float:
    """
    Return triangle-channel elevation above the thalweg for a lateral offset.

    The returned value is capped at ``floodplain_offset`` so the channel ties
    into one common floodplain.
    """
    offset_arr = np.asarray(offset_m, dtype=float)
    return np.minimum(floodplain_offset, offset_arr / side_slope)


def create_dem(station_grid_m: np.ndarray, distance_to_stream_m: np.ndarray) -> np.ndarray:
    """
    Build a DEM whose triangle thalweg follows the rasterized stream path.

    The stream raster and DEM now share the same centerline support. Bed
    elevation is assigned from ``station_grid_m``, which is derived from the
    rasterized stream path, and cross-section elevation is assigned from
    ``distance_to_stream_m``, which measures distance to the nearest rasterized
    stream cell center. This keeps the raster stream cells on the DEM thalweg
    instead of letting rasterization offsets introduce artificial cross-stream
    elevation differences inside the channel.
    """
    thalweg_z = build_longitudinal_thalweg(station_grid_m)
    lateral_z = lateral_elevation_above_thalweg(distance_to_stream_m)
    return (thalweg_z + lateral_z).astype(np.float32)


def clean_stream_raster_like_nencarta(stream_raster: np.ndarray) -> np.ndarray:
    """
    Remove isolated and redundant stream cells using Nencarta's cleanup logic.

    The angled synthetic stream is rasterized from a continuous centerline, so
    diagonal artifacts are possible. This two-pass cleanup matches the logic
    used in production and helps keep the test input realistic.
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
    Rasterize the rotated stream centerline and split it into two COMIDs.

    The centerline is sampled at sub-cell spacing so the diagonal reach remains
    connected. The raster is cleaned using the same redundant-cell logic as the
    production workflow, then the cleaned cells are split into COMID 1 and
    COMID 2 based on projected station along the stream.
    """
    raw_stream = np.zeros((ny, nx), dtype=np.uint8)

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
    raw_stream[row_idx[valid], col_idx[valid]] = 1

    cleaned_stream = clean_stream_raster_like_nencarta(raw_stream)
    split_stream = np.zeros_like(cleaned_stream, dtype=np.uint8)

    stream_rows, stream_cols = np.where(cleaned_stream > 0)
    station_m, _ = project_row_col_to_channel(stream_rows.astype(float), stream_cols.astype(float))
    split_stream[stream_rows, stream_cols] = np.where(station_m < (length_m / 2.0), 1, 2).astype(np.uint8)
    return split_stream


def _build_ordered_stream_support(stream: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Order rasterized stream cells from upstream to downstream.

    The analytical rotated-centerline projection is used only to recover
    upstream/downstream sequence. Once ordered, cumulative distance is computed
    along the rasterized stream-cell centers themselves and normalized to
    ``length_m`` so the DEM and the Manning reference preserve the intended
    nominal reach length while sharing the rasterized stream path.
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


def _project_points_to_stream_polyline(
    query_x_m: np.ndarray,
    query_y_m: np.ndarray,
    stream_x_m: np.ndarray,
    stream_y_m: np.ndarray,
    stream_station_m: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Project points onto the piecewise-linear stream path between ordered cells.

    The rasterized stream cells are treated as vertices of a polyline. Each
    query point is projected to the nearest point on that polyline, not merely
    to the nearest vertex. This reduces the repeated oscillation that comes
    from carving the DEM around a stair-step raster path as if it were only a
    set of isolated cell centers.
    """
    if len(stream_x_m) < 2:
        raise ValueError("At least two ordered stream vertices are required to build a stream polyline.")

    seg_start_x_m = stream_x_m[:-1]
    seg_start_y_m = stream_y_m[:-1]
    seg_end_x_m = stream_x_m[1:]
    seg_end_y_m = stream_y_m[1:]
    seg_dx_m = seg_end_x_m - seg_start_x_m
    seg_dy_m = seg_end_y_m - seg_start_y_m
    seg_len_m = np.diff(stream_station_m)
    seg_len2_m = seg_dx_m * seg_dx_m + seg_dy_m * seg_dy_m

    valid_segments = seg_len2_m > 0.0
    if not np.any(valid_segments):
        raise ValueError("The ordered stream polyline collapsed to zero-length segments.")

    seg_start_x_m = seg_start_x_m[valid_segments]
    seg_start_y_m = seg_start_y_m[valid_segments]
    seg_dx_m = seg_dx_m[valid_segments]
    seg_dy_m = seg_dy_m[valid_segments]
    seg_len_m = seg_len_m[valid_segments]
    seg_len2_m = seg_len2_m[valid_segments]
    seg_start_station_m = stream_station_m[:-1][valid_segments]

    projected_station_m = np.empty(query_x_m.size, dtype=np.float32)
    projected_dist2_m = np.empty(query_x_m.size, dtype=np.float64)

    chunk_size = 1024
    for start in range(0, query_x_m.size, chunk_size):
        end = min(start + chunk_size, query_x_m.size)
        qx_m = query_x_m[start:end, None]
        qy_m = query_y_m[start:end, None]

        rel_x_m = qx_m - seg_start_x_m[None, :]
        rel_y_m = qy_m - seg_start_y_m[None, :]
        t = (rel_x_m * seg_dx_m[None, :] + rel_y_m * seg_dy_m[None, :]) / seg_len2_m[None, :]
        t = np.clip(t, 0.0, 1.0)

        proj_x_m = seg_start_x_m[None, :] + t * seg_dx_m[None, :]
        proj_y_m = seg_start_y_m[None, :] + t * seg_dy_m[None, :]
        dist2_m = (qx_m - proj_x_m) ** 2 + (qy_m - proj_y_m) ** 2

        nearest_seg_idx = np.argmin(dist2_m, axis=1)
        row_idx = np.arange(end - start)
        nearest_t = t[row_idx, nearest_seg_idx]
        projected_station_m[start:end] = (
            seg_start_station_m[nearest_seg_idx] + nearest_t * seg_len_m[nearest_seg_idx]
        ).astype(np.float32)
        projected_dist2_m[start:end] = dist2_m[row_idx, nearest_seg_idx]

    return projected_station_m, projected_dist2_m


def build_stream_reference_grids(stream: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Build station and distance grids from the rasterized stream polyline.

    ``station_grid_m`` stores the along-stream station of the nearest point on
    the piecewise-linear stream path for every raster cell.
    ``distance_to_stream_m`` stores the Euclidean distance to that same nearest
    polyline point. Those grids let the DEM be carved around the exact raster
    path ARC will use while reducing the repeated stair-step oscillation that
    comes from using nearest cell centers alone.
    """
    _, _, stream_x_m, stream_y_m, stream_station_m = _build_ordered_stream_support(stream)

    row_grid, col_grid = np.indices((ny, nx), dtype=float)
    query_x_m, query_y_m = local_xy_from_row_col(row_grid.ravel(), col_grid.ravel())
    nearest_station_m, nearest_dist2_m = _project_points_to_stream_polyline(
        query_x_m,
        query_y_m,
        stream_x_m,
        stream_y_m,
        stream_station_m,
    )

    station_grid_m = nearest_station_m.reshape((ny, nx))
    distance_to_stream_m = np.sqrt(nearest_dist2_m).reshape((ny, nx)).astype(np.float32)
    return station_grid_m, distance_to_stream_m


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
    ``testing/slope_method_angle_case_outputs``. Clearing a case directory
    before writing prevents stale rasters, vectors, and VDT outputs from being
    reused across runs.
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

    Manual reruns reuse stable filenames, so this cleanup keeps vector outputs
    deterministic and avoids stale SQLite sidecar state.
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
    Export one stream polyline per synthetic COMID from the angled stream raster.

    Each nonzero stream ID becomes one ordered line feature. Ordering is based
    on projected station along the rotated centerline rather than raw row/col
    sorting so the output line follows the true upstream-to-downstream path.
    """
    _remove_existing_geopackage(stream_lines_path)

    line_records = []
    unique_stream_ids = sorted(int(stream_id) for stream_id in np.unique(stream) if stream_id > 0)

    for stream_id in unique_stream_ids:
        stream_rows, stream_cols, _, _, _ = _build_ordered_stream_support((stream == stream_id).astype(np.uint8))
        line_points: list[tuple[float, float]] = []
        for row_idx, col_idx in zip(stream_rows, stream_cols):
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


def write_case_inputs(case_dir: Path, dem: np.ndarray, stream: np.ndarray) -> dict[str, Path]:
    """
    Write all ARC inputs needed for one angled slope-method case.

    Every slope method gets its own directory. A stream GeoPackage is always
    written so the shared output structure stays consistent and the
    ``end_points`` case always has the vector input it needs.
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

    The analytical reference does not depend on plan-view angle or ARC's
    slope-method choice once the cross-section geometry is fixed.
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


def compute_station_from_row_col(rows: pd.Series | np.ndarray, cols: pd.Series | np.ndarray, station_grid_m: np.ndarray) -> np.ndarray:
    """
    Convert VDT row/column positions into streamwise station coordinates.

    Station is looked up from the precomputed raster-path station grid so the
    Manning reference and the DEM use the same streamwise support.
    """
    row_idx = np.clip(np.rint(np.asarray(rows, dtype=float)).astype(int), 0, ny - 1)
    col_idx = np.clip(np.rint(np.asarray(cols, dtype=float)).astype(int), 0, nx - 1)
    return station_grid_m[row_idx, col_idx]


def _build_arc_args(case_paths: dict[str, Path], slope_method_config: dict[str, str | bool]) -> dict[str, str | int]:
    """
    Build the ARC input dictionary for one angled slope-method case.

    Only the stream-slope-method arguments change among cases. All other ARC
    settings stay fixed so the comparison isolates slope-source behavior.
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


def run_slope_method_case(
    method_name: str,
    slope_method_config: dict[str, str | bool],
    output_root: Path,
    stream: np.ndarray,
    station_grid_m: np.ndarray,
    distance_to_stream_m: np.ndarray,
) -> pd.DataFrame:
    """
    Create one angled synthetic case, run ARC, and attach Manning references.

    ``method_name`` is the user-facing case key, while ``arc_value`` is the
    exact string ARC expects for ``Stream_Slope_Method``.
    """
    case_dir = output_root / method_name
    dem = create_dem(station_grid_m, distance_to_stream_m)
    case_paths = write_case_inputs(case_dir, dem, stream)

    Arc(args=_build_arc_args(case_paths, slope_method_config)).set_log_level("info").run()

    vdt_df = pd.read_csv(case_paths["vdt_path"])
    xs_df = pd.read_csv(case_paths["xs_output_path"], sep="\t")
    vdt_df["Station (m)"] = compute_station_from_row_col(vdt_df["Row"], vdt_df["Col"], station_grid_m)

    depth_y = solve_mannings_depth()
    _, top_width_y, velocity_y = calculate_mannings_discharge(depth_y)
    vdt_df["Manning Depth (m)"] = depth_y
    vdt_df["Manning Top Width (m)"] = top_width_y
    vdt_df["Manning Velocity (m/s)"] = velocity_y
    vdt_df["Manning WSE (m)"] = depth_y - bed_slope * vdt_df["Station (m)"]
    vdt_df["Slope Method"] = method_name
    vdt_df["Requested XS_Angle (rad)"] = xs_angle_rad
    vdt_df["Requested XS_Angle (deg)"] = xs_angle_deg
    vdt_df["Channel Angle (deg)"] = channel_angle_deg

    export_vdt_points(vdt_df, case_paths["vdt_points_path"])
    export_cross_section_lines(xs_df, case_paths["dem_path"], case_paths["xs_lines_path"])
    return vdt_df


def run_all_slope_method_cases(output_root: Path, requested_xs_angle_rad: float = XS_ANGLE_RAD) -> dict[str, pd.DataFrame]:
    """
    Run the angled triangle reach with every configured slope method.

    The requested ARC-style cross-section angle is applied once before any case
    is generated so the DEM, stream raster, stationing, and plots share one
    common geometry.
    """
    configure_xs_angle(requested_xs_angle_rad)
    output_root.mkdir(parents=True, exist_ok=True)
    stream = create_stream_raster()
    station_grid_m, distance_to_stream_m = build_stream_reference_grids(stream)
    return {
        method_name: run_slope_method_case(
            method_name,
            method_config,
            output_root,
            stream,
            station_grid_m,
            distance_to_stream_m,
        )
        for method_name, method_config in slope_method_configs.items()
    }


def _select_core_reach_slice(vdt_df: pd.DataFrame) -> pd.Series:
    """
    Return the regression-comparison reach.

    The angled synthetic reach is trimmed at both ends so the checks focus on
    the channel interior rather than endpoint cells, which are more sensitive
    to rasterized geometry and neighborhood effects.
    """
    return (vdt_df["Station (m)"] >= 25.0) & (vdt_df["Station (m)"] <= (length_m - 25.0))


def assert_slope_method_case_is_valid(method_name: str, vdt_df: pd.DataFrame) -> None:
    """
    Perform regression-style assertions for one angled slope-method case.

    The tolerances are moderate because ARC is operating on rasterized geometry
    and an angled stream path. The goal is to catch real regressions rather
    than force exact analytical agreement.
    """
    assert not vdt_df.empty, f"{method_name} produced an empty VDT table."
    assert "wse_2" in vdt_df.columns, f"{method_name} VDT output is missing wse_2."
    assert "t_2" in vdt_df.columns, f"{method_name} VDT output is missing t_2."
    assert "v_2" in vdt_df.columns, f"{method_name} VDT output is missing v_2."
    assert "Slope" in vdt_df.columns, f"{method_name} VDT output is missing Slope."
    assert "XS_Angle" in vdt_df.columns, f"{method_name} VDT output is missing XS_Angle."

    assert np.isfinite(vdt_df["wse_2"]).all(), f"{method_name} ARC WSE contains non-finite values."
    assert np.isfinite(vdt_df["t_2"]).all(), f"{method_name} ARC top width contains non-finite values."
    assert np.isfinite(vdt_df["v_2"]).all(), f"{method_name} ARC velocity contains non-finite values."
    assert np.isfinite(vdt_df["Slope"]).all(), f"{method_name} ARC slope contains non-finite values."
    assert np.isfinite(vdt_df["XS_Angle"]).all(), f"{method_name} ARC XS_Angle contains non-finite values."
    assert np.isfinite(vdt_df["Manning WSE (m)"]).all(), f"{method_name} Manning WSE contains non-finite values."
    assert np.isfinite(vdt_df["Manning Top Width (m)"]).all(), f"{method_name} Manning top width contains non-finite values."
    assert np.isfinite(vdt_df["Manning Velocity (m/s)"]).all(), f"{method_name} Manning velocity contains non-finite values."

    assert (vdt_df["Manning Depth (m)"] > 0.0).all(), f"{method_name} Manning depth must be positive."
    assert (vdt_df["Manning Top Width (m)"] > 0.0).all(), f"{method_name} Manning top width must be positive."
    assert (vdt_df["Manning Velocity (m/s)"] > 0.0).all(), f"{method_name} Manning velocity must be positive."
    assert (vdt_df["Slope"] > 0.0).all(), f"{method_name} ARC slope must be positive."

    core_df = vdt_df.loc[_select_core_reach_slice(vdt_df)].copy()
    assert not core_df.empty, f"{method_name} core comparison window is empty."

    xs_angle_residual = (core_df["XS_Angle"] - xs_angle_rad).abs()
    assert xs_angle_residual.median() < 0.08, (
        f"{method_name} median core XS_Angle residual is too large: "
        f"{xs_angle_residual.median():.3f} rad"
    )
    assert xs_angle_residual.max() < 0.18, (
        f"{method_name} maximum core XS_Angle residual is too large: "
        f"{xs_angle_residual.max():.3f} rad"
    )

    wse_residual = (core_df["wse_2"] - core_df["Manning WSE (m)"]).abs()
    top_width_residual = (core_df["t_2"] - core_df["Manning Top Width (m)"]).abs()
    velocity_residual = (core_df["v_2"] - core_df["Manning Velocity (m/s)"]).abs()

    assert wse_residual.median() < 0.90, f"{method_name} median core WSE residual is too large: {wse_residual.median():.3f} m"
    assert wse_residual.max() < 1.75, f"{method_name} maximum core WSE residual is too large: {wse_residual.max():.3f} m"
    assert top_width_residual.median() < 3.00, f"{method_name} median core top-width residual is too large: {top_width_residual.median():.3f} m"
    assert top_width_residual.max() < 6.00, f"{method_name} maximum core top-width residual is too large: {top_width_residual.max():.3f} m"
    assert velocity_residual.median() < 0.75, f"{method_name} median core velocity residual is too large: {velocity_residual.median():.3f} m/s"
    assert velocity_residual.max() < 1.50, f"{method_name} maximum core velocity residual is too large: {velocity_residual.max():.3f} m/s"

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

    The summary keeps the key diagnostics visible without opening the VDT files
    manually: angle recovery, slope behavior, and upstream residuals versus the
    analytical Manning reference.
    """
    summary_rows = []

    for method_name, vdt_df in results.items():
        core_df = vdt_df.loc[_select_core_reach_slice(vdt_df)].copy()
        wse_residual = (core_df["wse_2"] - core_df["Manning WSE (m)"]).abs()
        top_width_residual = (core_df["t_2"] - core_df["Manning Top Width (m)"]).abs()
        velocity_residual = (core_df["v_2"] - core_df["Manning Velocity (m/s)"]).abs()

        summary_rows.append(
            {
                "slope_method": method_name,
                "requested_xs_angle_rad": xs_angle_rad,
                "requested_xs_angle_deg": xs_angle_deg,
                "derived_channel_angle_deg": channel_angle_deg,
                "median_xs_angle_rad": float(core_df["XS_Angle"].median()),
                "max_abs_xs_angle_residual_rad": float((core_df["XS_Angle"] - xs_angle_rad).abs().max()),
                "manning_depth_m": float(vdt_df["Manning Depth (m)"].iloc[0]),
                "manning_top_width_m": float(vdt_df["Manning Top Width (m)"].iloc[0]),
                "manning_velocity_mps": float(vdt_df["Manning Velocity (m/s)"].iloc[0]),
                "median_vdt_slope": float(vdt_df["Slope"].median()),
                "max_abs_slope_residual": float((vdt_df["Slope"] - bed_slope).abs().max()),
                "median_core_wse_residual_m": float(wse_residual.median()),
                "max_core_wse_residual_m": float(wse_residual.max()),
                "median_core_top_width_residual_m": float(top_width_residual.median()),
                "max_core_top_width_residual_m": float(top_width_residual.max()),
                "median_core_velocity_residual_mps": float(velocity_residual.median()),
                "max_core_velocity_residual_mps": float(velocity_residual.max()),
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
    reference. The fourth column shows the row-specific VDT slope values
    together with the synthetic bed slope. The fifth column shows ARC's
    recovered ``XS_Angle`` against the requested synthetic angle.
    """
    title_fontsize = 11
    label_fontsize = 8
    tick_fontsize = 8
    legend_fontsize = 8

    figure, axes = plt.subplots(len(slope_method_configs), 5, figsize=(20, 9), sharex="col")
    axes_array = np.atleast_2d(axes)
    all_vdt_frames = list(results.values())
    wse_ylim = _compute_fixed_ylim([df["wse_2"] for df in all_vdt_frames] + [df["Manning WSE (m)"] for df in all_vdt_frames])
    top_width_ylim = _compute_fixed_ylim([df["t_2"] for df in all_vdt_frames] + [df["Manning Top Width (m)"] for df in all_vdt_frames])
    velocity_ylim = _compute_fixed_ylim([df["v_2"] for df in all_vdt_frames] + [df["Manning Velocity (m/s)"] for df in all_vdt_frames])
    slope_ylim = _compute_fixed_ylim(
        [df["Slope"] * 100.0 for df in all_vdt_frames] + [np.array([bed_slope * 100.0])]
    )
    xs_angle_ylim = _compute_fixed_ylim([df["XS_Angle"] for df in all_vdt_frames] + [np.array([xs_angle_rad])])

    for axis_row, (method_name, method_config) in zip(axes_array, slope_method_configs.items()):
        vdt_df = results[method_name]
        method_color = str(method_config["color"])

        wse_axis = axis_row[0]
        top_width_axis = axis_row[1]
        velocity_axis = axis_row[2]
        slope_axis = axis_row[3]
        xs_angle_axis = axis_row[4]

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

        xs_angle_axis.plot(vdt_df["Station (m)"], vdt_df["XS_Angle"], color=method_color, label="VDT XS_Angle")
        xs_angle_axis.axhline(xs_angle_rad, color=method_color, linestyle="--", label="Requested XS_Angle")
        xs_angle_axis.set_ylabel("Angle (rad)", fontsize=label_fontsize)
        xs_angle_axis.set_title(f"{method_name} XS_Angle", fontsize=title_fontsize)
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
            "Angled Triangle Slope-Method Comparison "
            f"(requested XS_Angle={xs_angle_rad:.3f} rad, "
            f"derived stream angle={channel_angle_deg:.1f} deg)"
        ),
        fontsize=12,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.97), pad=2.5, w_pad=2.0, h_pad=2.5)
    plt.show()


def test_slope_method_angle_cases(tmp_path: Path) -> None:
    """
    Automated regression test for the three angled synthetic slope-method cases.

    Pytest provides ``tmp_path`` so the run is isolated and does not interfere
    with persistent manual outputs from earlier exploratory runs.
    """
    results = run_all_slope_method_cases(tmp_path / "slope_method_angle_cases")

    for method_name, vdt_df in results.items():
        assert_slope_method_case_is_valid(method_name, vdt_df)


def main() -> None:
    """
    Run the angled slope-method comparison manually and show diagnostic plots.

    This function is kept separate from the automated test so importing the
    module under pytest never triggers ARC runs, file writes, or figures.
    """
    manual_output_root.mkdir(parents=True, exist_ok=True)
    results = run_all_slope_method_cases(manual_output_root)
    summary_df = summarize_results(results)

    print("Synthetic angled slope-method test summary:")
    print(summary_df.to_string(index=False))
    print(f"\nOutputs written to: {manual_output_root}")

    plot_results(results)


if __name__ == "__main__":
    # Manual execution path:
    # - writes persistent outputs under testing/slope_method_angle_case_outputs/
    # - prints a summary table
    # - opens diagnostic plots
    #
    # Pytest never enters this block because it imports the module rather than
    # running it as a script.
    main()
