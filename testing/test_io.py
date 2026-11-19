from __future__ import annotations

import math
from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
from osgeo import gdal, osr
from pyproj import CRS, Geod
from shapely.geometry import LineString, MultiLineString

from arc.io import Raster, Vector
from arc.io.raster import cell_size_on_ellipsoid

gdal.UseExceptions()


def write_raster(path: Path, array: np.ndarray, geotransform, epsg: int, nodata=None) -> Path:
    driver = gdal.GetDriverByName("GTiff")
    types = {np.dtype(np.float32): gdal.GDT_Float32, np.dtype(np.int32): gdal.GDT_Int32,
             np.dtype(np.uint8): gdal.GDT_Byte, np.dtype(np.float64): gdal.GDT_Float64}
    ds = driver.Create(str(path), array.shape[1], array.shape[0], 1, types[array.dtype])
    ds.SetGeoTransform(geotransform)
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(epsg)
    ds.SetProjection(srs.ExportToWkt())
    band = ds.GetRasterBand(1)
    if nodata is not None:
        band.SetNoDataValue(nodata)
    band.WriteArray(array)
    ds = None
    return path


GEOGRAPHIC = (-84.5, 0.000277777777, 0.0, 34.2, 0.0, -0.000277777777)  # 1 arcsecond cells near Atlanta
PROJECTED = (500000.0, 10.0, 0.0, 3800000.0, 0.0, -10.0)


# --- Rasters --------------------------------------------------------------------------------------------------------


def test_a_raster_s_grid_and_values(tmp_path: Path) -> None:
    values = np.arange(12, dtype=np.float32).reshape(3, 4)
    raster = Raster(write_raster(tmp_path / "dem.tif", values, PROJECTED, 32616, nodata=-9999.0))

    assert raster.shape == (3, 4) and (raster.nrows, raster.ncols) == (3, 4)
    assert raster.geotransform == PROJECTED
    assert raster.resolution == (10.0, 10.0)
    assert raster.bbox == (500000.0, 3799970.0, 500040.0, 3800000.0)
    assert raster.nodata_value == -9999.0
    assert raster.dtype == np.float32
    assert not raster.is_rotated
    assert raster.axis_units() == {"metre"}
    assert np.array_equal(raster.read_array(), values)
    assert raster.read_array(np.float64).dtype == np.float64


def test_rasters_are_decoded_on_the_threads_gdal_num_threads_says(tmp_path: Path, monkeypatch) -> None:
    """GDAL_NUM_THREADS, as an environment variable or a GDAL configuration option (which wins), is the number of
    threads a raster is read on, and every CPU without it. Reading leaves it as it was."""
    values = np.arange(12, dtype=np.float32).reshape(3, 4)
    raster = Raster(write_raster(tmp_path / "dem.tif", values, PROJECTED, 32616))
    used = []
    config_option = gdal.config_option

    def recorded(key, value, *args, **kwargs):
        used.append((key, value))
        return config_option(key, value, *args, **kwargs)

    monkeypatch.setattr(gdal, "config_option", recorded)
    monkeypatch.delenv("GDAL_NUM_THREADS", raising=False)
    assert np.array_equal(raster.read_array(), values)
    monkeypatch.setenv("GDAL_NUM_THREADS", "2")
    assert np.array_equal(raster.read_array(np.float64), values)
    with config_option("GDAL_NUM_THREADS", "1"):
        assert np.array_equal(raster.read_array(), values)
        assert gdal.GetConfigOption("GDAL_NUM_THREADS") == "1"
    assert gdal.GetConfigOption("GDAL_NUM_THREADS") == "2"

    assert used == [("GDAL_NUM_THREADS", "ALL_CPUS"), ("GDAL_NUM_THREADS", "2"), ("GDAL_NUM_THREADS", "1")]


def test_a_missing_raster_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        Raster(tmp_path / "missing.tif")


def test_the_file_isn_t_kept_open(tmp_path: Path) -> None:
    """So an output can be written over, or deleted, while a Raster of it is around."""
    path = write_raster(tmp_path / "dem.tif", np.ones((2, 2), np.float32), PROJECTED, 32616)
    raster = Raster(path)
    raster.read_array()
    Raster.write_array(np.zeros((2, 2), np.float32), raster, path)
    path.unlink()
    assert not path.exists()


@pytest.mark.parametrize("geotransform, epsg, ellipsoid", [(GEOGRAPHIC, 4326, "WGS84"),
                                                            ((-120.0, 0.001, 0.0, 45.0, 0.0, -0.0005), 4269, "GRS80")])
def test_a_geographic_raster_s_cells_are_measured_on_its_ellipsoid_at_its_middle_latitude(
        tmp_path: Path, geotransform, epsg: int, ellipsoid: str) -> None:
    raster = Raster(write_raster(tmp_path / "dem.tif", np.ones((1234, 567), np.float32), geotransform, epsg))
    width, height = geotransform[1], -geotransform[5]
    latitude = geotransform[3] - 1234 * height / 2
    geod = Geod(ellps=ellipsoid)

    expected = (geod.inv(0.0, latitude, width, latitude)[2], geod.inv(0.0, latitude, 0.0, latitude + height)[2])
    assert raster.cell_size_in_metres() == pytest.approx(expected, abs=5e-7)  # to the micrometre
    assert all(round(size, 6) == size for size in raster.cell_size_in_metres())


WGS84 = Geod(ellps="WGS84")
ARCSECOND = 1.0 / 3600.0


def _tile_latitude(south: int) -> float:
    """The latitude a 1-degree, 1-arcsecond tile is measured at, as nencarta buffers it, by 0.1 degrees: its middle."""
    top = south + 1.1 + ARCSECOND / 2.0
    return (top + (top - 4320 * ARCSECOND)) / 2.0


def _geodesic_size(latitude: float, size: float = ARCSECOND) -> tuple[float, float]:
    return abs(WGS84.inv(0.0, latitude, size, latitude)[2]), abs(WGS84.inv(0.0, latitude, 0.0, latitude + size)[2])


@pytest.mark.parametrize("arcseconds", [1.0 / 3.0, 1.0, 3.0, 30.0])
def test_a_cell_s_size_on_its_ellipsoid_is_its_geodesics(arcseconds: float) -> None:
    """Within 3e-9 m, about the geodesics' own error, from the equator to 80 degrees."""
    size = arcseconds * ARCSECOND
    for latitude in np.linspace(-80.0, 80.0, 161):
        expected = _geodesic_size(float(latitude), size)
        assert cell_size_on_ellipsoid(WGS84.a, WGS84.es, float(latitude), size, size) == pytest.approx(expected,
                                                                                                         abs=3e-9)


def test_a_tile_s_cell_size_rounds_as_its_geodesics_did() -> None:
    """At every 1-degree tile from 60 S to 75 N, to the micrometre: the cell size, and with it ARC's outputs, are what
    the geodesics gave them on Linux."""
    for south in range(-60, 75):
        latitude = _tile_latitude(south)
        size = cell_size_on_ellipsoid(WGS84.a, WGS84.es, latitude, ARCSECOND, ARCSECOND)
        assert [round(s, 6) for s in size] == [round(s, 6) for s in _geodesic_size(latitude)], south


def test_a_cell_s_size_is_the_same_to_the_micrometre_whatever_the_last_digits_of_sin_and_cos(monkeypatch) -> None:
    """Another system's C math library may give sin and cos a last digit off. At the tile at 50 N, the geodesic's
    height of a cell is 4e-10 m from a micrometre's half, nearer than the geodesic moved between Windows and Linux;
    the cell size doesn't move."""
    import arc.io.raster as raster_module

    latitude = _tile_latitude(50)
    height = _geodesic_size(latitude)[1]
    assert abs((height * 1e6) % 1.0 - 0.5) < 1e-3  # within a nanometre of rounding either way

    class OffByOne:
        """math, with sin and cos `ulps` last digits off."""
        def __init__(self, ulps: int):
            self.ulps = ulps

        def __getattr__(self, name):
            return getattr(math, name)

        def _off(self, value: float) -> float:
            for _ in range(abs(self.ulps)):
                value = float(np.nextafter(value, np.inf if self.ulps > 0 else -np.inf))
            return value

        def sin(self, x: float) -> float:
            return self._off(math.sin(x))

        def cos(self, x: float) -> float:
            return self._off(math.cos(x))

    sizes = set()
    for ulps in (-2, -1, 0, 1, 2):
        monkeypatch.setattr(raster_module, "math", OffByOne(ulps))
        width, height = cell_size_on_ellipsoid(WGS84.a, WGS84.es, latitude, ARCSECOND, ARCSECOND)
        sizes.add((round(width, 6), round(height, 6)))
    assert len(sizes) == 1


def test_a_projected_raster_s_cells_are_its_resolution(tmp_path: Path) -> None:
    raster = Raster(write_raster(tmp_path / "dem.tif", np.ones((3, 4), np.float32), PROJECTED, 32616))

    assert raster.cell_size_in_metres() == (10.0, 10.0)


def test_a_raster_projected_in_feet_has_no_cell_size_in_metres(tmp_path: Path) -> None:
    raster = Raster(write_raster(tmp_path / "dem.tif", np.ones((2, 2), np.float32), PROJECTED, 2240))  # US feet

    with pytest.raises(ValueError, match="not metres"):
        raster.cell_size_in_metres()


def test_the_same_grid(tmp_path: Path) -> None:
    dem = Raster(write_raster(tmp_path / "dem.tif", np.ones((3, 4), np.float32), PROJECTED, 32616))
    streams = Raster(write_raster(tmp_path / "streams.tif", np.ones((3, 4), np.int32), PROJECTED, 32616))
    shifted = Raster(write_raster(tmp_path / "shifted.tif", np.ones((3, 4), np.int32),
                                  (PROJECTED[0] + 5.0,) + PROJECTED[1:], 32616))
    smaller = Raster(write_raster(tmp_path / "smaller.tif", np.ones((3, 3), np.int32), PROJECTED, 32616))

    assert dem.same_grid(streams)
    assert not dem.same_grid(shifted) and not dem.same_grid(smaller)


def test_a_raster_is_written_on_the_reference_grid_with_zstd_in_tiles(tmp_path: Path) -> None:
    """The values on the reference raster's grid, compressed with ZSTD in 512 × 512 tiles without a predictor, and
    without a nodata value unless one is given."""
    reference = Raster(write_raster(tmp_path / "dem.tif", np.ones((40, 30), np.float32), PROJECTED, 32616))
    values = np.random.default_rng(1).normal(100, 5, (40, 30)).astype(np.float32)
    values[3:7, 4:9] = np.nan

    written = Raster.write_array(values, reference, tmp_path / "out" / "bathymetry.tif", dtype=np.float32)

    ds = gdal.Open(str(written.filepath))
    assert np.array_equal(ds.ReadAsArray(), values, equal_nan=True)
    assert ds.GetGeoTransform() == reference.geotransform
    assert ds.GetProjection() == reference.projection
    structure = ds.GetMetadata("IMAGE_STRUCTURE")
    assert structure.get("COMPRESSION") == "ZSTD" and "PREDICTOR" not in structure
    assert ds.GetRasterBand(1).GetBlockSize() == [512, 512]
    assert ds.GetRasterBand(1).GetNoDataValue() is None
    ds = None


@pytest.mark.parametrize("compression", ["LZW", "DEFLATE", "LZMA"])
def test_any_compressed_raster_is_tiled_without_a_predictor(tmp_path: Path, compression: str) -> None:
    reference = Raster(write_raster(tmp_path / "dem.tif", np.ones((40, 30), np.float32), PROJECTED, 32616))
    values = np.full((40, 30), np.nan, np.float32)
    values[10:12, 5:20] = 97.25

    written = Raster.write_array(values, reference, tmp_path / "bathymetry.tif", compression=compression.lower())

    ds = gdal.Open(str(written.filepath))
    structure = ds.GetMetadata("IMAGE_STRUCTURE")
    assert structure.get("COMPRESSION") == compression and "PREDICTOR" not in structure
    assert ds.GetRasterBand(1).GetBlockSize() == [512, 512]
    ds = None
    assert np.array_equal(written.read_array(), values, equal_nan=True)


def test_writing_a_raster_with_nodata_and_other_compressions(tmp_path: Path) -> None:
    reference = Raster(write_raster(tmp_path / "dem.tif", np.ones((4, 5), np.float32), PROJECTED, 32616))
    flood = np.zeros((4, 5), np.uint8)
    flood[1, 2] = 3
    written = Raster.write_array(flood, reference, tmp_path / "flood.tif", compression="NONE", nodata_value=255)

    assert written.dtype == np.uint8 and written.nodata_value == 255
    assert np.array_equal(written.read_array(), flood)
    ds = gdal.Open(str(written.filepath))
    assert ds.GetRasterBand(1).GetBlockSize()[0] == 5  # uncompressed rasters stay in strips, unpadded
    ds = None
    with pytest.raises(ValueError, match="reference raster"):
        Raster.write_array(np.zeros((3, 3)), reference, tmp_path / "wrong.tif")


# --- Vectors --------------------------------------------------------------------------------------------------------


def stream_lines(crs) -> gpd.GeoDataFrame:
    lines = [LineString([(-87.0, 34.0), (-86.99, 34.005), (-86.98, 34.0)]),
             MultiLineString([[(-86.98, 34.0), (-86.97, 34.0)], [(-86.97, 34.0), (-86.97, 34.02)]]),
             LineString([(-86.97, 34.02), (-86.95, 34.03)])]
    return gpd.GeoDataFrame({"LINKNO": [1, 2, 3], "DSLINKNO": [2, 3, -1], "area": [1.0, 2.0, 3.5]},
                            geometry=lines, crs=crs)


def test_a_vector_s_metadata_and_features(tmp_path: Path) -> None:
    path = tmp_path / "streams.gpkg"
    stream_lines("EPSG:4326").to_file(path)
    vector = Vector(path)

    assert vector.crs == CRS.from_epsg(4326)
    assert vector.bbox == pytest.approx((-87.0, 34.0, -86.95, 34.03))
    assert vector.columns == ("LINKNO", "DSLINKNO", "area")
    assert len(vector) == 3
    assert vector.to_geopandas()["LINKNO"].tolist() == [1, 2, 3]


def test_a_geoparquet_vector_s_metadata_comes_from_its_own_metadata(tmp_path: Path) -> None:
    path = tmp_path / "streams.parquet"
    stream_lines("EPSG:4326").to_parquet(path)
    vector = Vector(path)

    assert vector.crs == CRS.from_epsg(4326)
    assert vector.bbox == pytest.approx((-87.0, 34.0, -86.95, 34.03))
    assert vector.columns == ("LINKNO", "DSLINKNO", "area")
    assert len(vector) == 3
    assert vector.to_geopandas()["LINKNO"].tolist() == [1, 2, 3]
