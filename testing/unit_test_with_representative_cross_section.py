"""
Synthetic ARC test for generating the representative cross-section output.

This file follows the same pattern as the other stand-alone synthetic tests in
``automated-rating-curve/testing``:

1. Automated test mode
   ``pytest automated-rating-curve/testing/unit_test_with_representative_cross_section.py``

2. Manual diagnostic mode
   ``python automated-rating-curve/testing/unit_test_with_representative_cross_section.py``

The automated test runs ARC end to end on one synthetic reach, writes the
representative cross-section CSV, and validates the exported table. Manual
mode writes the same artifacts into a persistent folder under
``testing/representative_cross_section_outputs/``.
"""

from __future__ import annotations

from pathlib import Path
import stat
import time

import numpy as np
import pandas as pd
from osgeo import gdal, osr

from arc import Arc
from arc.hydraulic_data import REPRESENTATIVE_CROSS_SECTION_COLUMNS

gdal.UseExceptions()


# ---------------------------------------------------------------------------
# Synthetic case configuration
# ---------------------------------------------------------------------------
length_m = 120.0
cellsize = 1.0
bed_slope = 0.001
mannings_discharge_m3s = 10.0
roughness = 0.025
floodplain_offset = 20.0
side_slope = 2.0

longitudinal_padding_cells = 20
lateral_padding_cells = 20

channel_nx = int(length_m / cellsize)
channel_ny = 80
channel_start_col = longitudinal_padding_cells
channel_end_col = channel_start_col + channel_nx
center_row = channel_ny // 2 + lateral_padding_cells
nx = channel_nx + 2 * longitudinal_padding_cells
ny = channel_ny + 2 * lateral_padding_cells

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

script_dir = Path(__file__).resolve().parent
manual_output_root = script_dir / "representative_cross_section_outputs"


def build_longitudinal_thalweg(station_m: float) -> float:
    """Return the synthetic thalweg elevation at one downstream station."""
    return -(station_m * bed_slope)


def lateral_elevation_above_thalweg(offset_m: float) -> float:
    """Return the synthetic triangular channel elevation above thalweg."""
    return min(floodplain_offset, offset_m / side_slope)


def create_dem() -> np.ndarray:
    """Build the DEM for one synthetic triangular reach."""
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
    """Create one synthetic stream reach with a single COMID."""
    stream = np.zeros((ny, nx), dtype=np.uint8)
    stream[center_row, channel_start_col:channel_end_col] = 1
    return stream


def create_land_cover_raster() -> np.ndarray:
    """Create a uniform land-cover raster so Manning n stays constant."""
    return np.ones((ny, nx), dtype=np.uint8)


def write_raster(path: Path, array: np.ndarray, gdal_dtype: int, wkt: str) -> None:
    """Write one raster array as GeoTIFF using the shared georeferencing."""
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(str(path), nx, ny, 1, gdal_dtype)
    dataset.SetGeoTransform(geotransform)
    dataset.SetProjection(wkt)
    dataset.GetRasterBand(1).WriteArray(array)
    dataset = None


def _unlink_with_retries(path: Path, retries: int = 5, delay_s: float = 0.2) -> None:
    """Delete one file with Windows-friendly retries."""
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
    """Remove previously generated artifacts for one synthetic case."""
    if not case_dir.exists():
        return

    for child in case_dir.iterdir():
        if child.is_dir():
            clear_case_directory(child)
            _rmdir_with_retries(child)
        else:
            _unlink_with_retries(child)


def write_case_inputs(case_dir: Path, dem: np.ndarray, stream: np.ndarray) -> dict[str, Path]:
    """Write the raster, flow, and Manning inputs required by ARC."""
    clear_case_directory(case_dir)
    case_dir.mkdir(parents=True, exist_ok=True)

    land_cover = create_land_cover_raster()

    srs = osr.SpatialReference()
    srs.ImportFromEPSG(epsg)
    wkt = srs.ExportToWkt()

    dem_path = case_dir / "dem.tif"
    stream_path = case_dir / "stream.tif"
    land_cover_path = case_dir / "land_cover.tif"
    flow_path = case_dir / "flow.csv"
    mannings_path = case_dir / "mannings.txt"
    representative_output_path = case_dir / "representative_cross_section.csv"

    write_raster(dem_path, dem, gdal.GDT_Float32, wkt)
    write_raster(stream_path, stream, gdal.GDT_Byte, wkt)
    write_raster(land_cover_path, land_cover, gdal.GDT_Byte, wkt)

    flow_df = pd.DataFrame(
        {
            "COMID": [1],
            "baseflow": [0.0],
            "maxflow": [mannings_discharge_m3s],
        }
    )
    flow_df.to_csv(flow_path, index=False)

    with mannings_path.open("w", encoding="utf-8") as mannings_file:
        mannings_file.write("lu\tdesc\troughness\n")
        mannings_file.write(f"1\tland\t{roughness}\n")

    return {
        "dem_path": dem_path,
        "stream_path": stream_path,
        "land_cover_path": land_cover_path,
        "flow_path": flow_path,
        "mannings_path": mannings_path,
        "representative_output_path": representative_output_path,
    }


def run_representative_case(output_root: Path) -> pd.DataFrame:
    """Run ARC and return the generated representative cross-section table."""
    case_dir = output_root / "single_reach"
    dem = create_dem()
    stream = create_stream_raster()
    case_paths = write_case_inputs(case_dir, dem, stream)

    Arc(
        args={
            "DEM_File": str(case_paths["dem_path"]),
            "Stream_File": str(case_paths["stream_path"]),
            "LU_Raster_SameRes": str(case_paths["land_cover_path"]),
            "LU_Manning_n": str(case_paths["mannings_path"]),
            "Degree_Manip": 0,
            "Degree_Interval": 0,
            "Low_Spot_Range": 0,
            "Gen_Slope_Dist": 10,
            "Gen_Dir_Dist": 10,
            "X_Section_Dist": 40,
            "Build_Representative_Cross_Section": True,
            "Representative_Cross_Section_File": str(case_paths["representative_output_path"]),
            "Print_VDT_Database": "",
            "Print_AP_Database": "",
            "Print_Curve_File": "",
            "AROutBATHY": "",
            "AROutFLOOD": "",
            "Stream_Slope_Method": "local_average",
            "LC_Water_Value": 80,
            "VDT_Database_NumIterations": 2,
        },
        quiet=True,
        processes=1,
    ).run()

    return pd.read_csv(case_paths["representative_output_path"])


def assert_representative_output_is_valid(df: pd.DataFrame) -> None:
    """Validate the generated representative cross-section table."""
    assert not df.empty
    assert list(df.columns) == REPRESENTATIVE_CROSS_SECTION_COLUMNS
    assert df["COMID"].nunique() == 1
    assert df["COMID"].iloc[0] == 1
    assert df["Depth_Stage_Index"].iloc[0] == 1
    assert df["Depth_Stage_Meters"].min() >= 0.0
    assert df["Depth_Stage_Meters"].max() <= 25.0
    assert np.allclose(
        df["Depth_Stage_Meters"].to_numpy(dtype=np.float64),
        df["Depth_Stage_Index"].to_numpy(dtype=np.float64) * 0.10,
    )
    assert np.isfinite(df.to_numpy(dtype=np.float64, copy=False)).all()
    assert (df["Cross_Section_Count"] > 0).all()
    assert (df["Hydraulic_Sample_Count"] > 0).all()
    assert (df["Representative_Top_Width"] > 0.0).all()
    assert (df["Representative_Cross_Sectional_Area"] > 0.0).all()


def test_generate_representative_cross_section_output(tmp_path: Path) -> None:
    """ARC should write a valid representative cross-section CSV."""
    df = run_representative_case(tmp_path)
    assert_representative_output_is_valid(df)


def main() -> None:
    """Run the stand-alone representative-cross-section case manually."""
    manual_output_root.mkdir(parents=True, exist_ok=True)
    df = run_representative_case(manual_output_root)
    print(df.head(10).to_string(index=False))
    print(f"Wrote {len(df)} representative rows to {manual_output_root}")


if __name__ == "__main__":
    main()
