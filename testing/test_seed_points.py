"""Which stream cells the FIST GeoJSON marks as SEED locations: the upstream ends of the streams that have cells."""
from __future__ import annotations

import os
from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
from osgeo import gdal, osr
from shapely.geometry import LineString, Point

from arc.create_geojson import FindClosestSEEDPoints, Read_Or_Find_SEED_Locations

CRS = "EPSG:32615"
# Two headwater streams, 1 and 2, join and flow into stream 3
STREAMS = gpd.GeoDataFrame(
    {"LINKNO": [1, 2, 3], "DSLINKNO": [3, 3, -1]},
    geometry=[LineString([(0, 1000), (1000, 500)]), LineString([(0, 0), (1000, 500)]),
              LineString([(1000, 500), (3000, 500)])],
    crs=CRS,
)
SEEDS = gpd.GeoDataFrame({"LINKNO": [1, 2]}, geometry=[Point(0, 1000), Point(0, 0)], crs=CRS)


def stream_cells(*streams: int) -> gpd.GeoDataFrame:
    """A cell every 100 m along each of these streams, from 50 m below its upstream end."""
    rows = []
    for stream in streams:
        line = STREAMS.geometry[STREAMS.LINKNO == stream].iloc[0]
        for along in np.arange(50.0, line.length, 100.0):
            rows.append({"COMID": stream, "geometry": line.interpolate(along)})
    return gpd.GeoDataFrame(rows, crs=CRS)


def seed_cells(cells: gpd.GeoDataFrame, with_streams: bool = True) -> list[tuple[int, float, float]]:
    streams = (STREAMS, "LINKNO", "DSLINKNO") if with_streams else ()
    marked = FindClosestSEEDPoints(SEEDS, cells, CRS, *streams)
    marked = marked[marked.SEED == "1"]
    return sorted((comid, round(point.x), round(point.y)) for comid, point in zip(marked.COMID, marked.geometry))


def first_cell(stream: int) -> tuple[int, float, float]:
    point = stream_cells(stream).geometry.iloc[0]
    return (stream, round(point.x), round(point.y))


def test_each_headwater_stream_is_marked_at_its_upstream_end():
    assert seed_cells(stream_cells(1, 2, 3)) == [first_cell(1), first_cell(2)]


def test_a_seed_whose_stream_has_no_cells_marks_no_other_stream_s_cell():
    # Stream 2 still flows into stream 3, so stream 3's upstream end isn't an end of the domain
    assert seed_cells(stream_cells(2, 3)) == [first_cell(2)]
    assert seed_cells(stream_cells(2, 3), with_streams=False) == [first_cell(2)]


def test_a_stream_whose_inflowing_streams_have_no_cells_is_marked_at_its_upstream_end():
    assert seed_cells(stream_cells(3)) == [first_cell(3)]
    assert seed_cells(stream_cells(3), with_streams=False) == []


# --- Reusing a SEED file ------------------------------------------------------------------------------------------


def write_streams_and_dem(folder: Path) -> tuple[Path, Path]:
    streams = folder / "streams.gpkg"
    STREAMS.to_file(streams, driver="GPKG")
    dem = folder / "dem.tif"
    ds = gdal.GetDriverByName("GTiff").Create(str(dem), 31, 11, 1, gdal.GDT_Float32)
    ds.SetGeoTransform((-50.0, 100.0, 0.0, 1050.0, 0.0, -100.0))
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(32615)
    ds.SetProjection(srs.ExportToWkt())
    ds.GetRasterBand(1).WriteArray(np.zeros((11, 31), dtype=np.float32))
    ds = None
    return streams, dem


def seeds_from(streams: Path, dem: Path, seed_file: Path) -> list[tuple[int, float, float]]:
    seeds = Read_Or_Find_SEED_Locations(str(streams), str(dem), str(seed_file), "LINKNO", "DSLINKNO")
    return sorted((int(linkno), round(point.x), round(point.y)) for linkno, point in zip(seeds.LINKNO, seeds.geometry))


FOUND = [(1, 0, 1000), (2, 0, 0)]


def test_without_a_seed_file_the_seeds_are_found_and_written(tmp_path: Path) -> None:
    streams, dem = write_streams_and_dem(tmp_path)
    assert seeds_from(streams, dem, tmp_path / "seeds.gpkg") == FOUND
    assert (tmp_path / "seeds.gpkg").exists()


def test_an_up_to_date_seed_file_is_used_as_it_is(tmp_path: Path) -> None:
    """A SEED file written (or edited by hand) after the streams and the DEM is kept, whatever it holds."""
    streams, dem = write_streams_and_dem(tmp_path)
    SEEDS.iloc[:1].to_file(tmp_path / "seeds.gpkg", driver="GPKG")
    assert seeds_from(streams, dem, tmp_path / "seeds.gpkg") == [(1, 0, 1000)]


@pytest.mark.parametrize("changed", ["streams", "dem"])
def test_a_seed_file_older_than_the_streams_or_the_dem_is_replaced(tmp_path: Path, changed: str) -> None:
    streams, dem = write_streams_and_dem(tmp_path)
    seed_file = tmp_path / "seeds.gpkg"
    SEEDS.iloc[:1].to_file(seed_file, driver="GPKG")
    written = os.path.getmtime(seed_file)
    os.utime(seed_file, (written - 60.0, written - 60.0))
    os.utime({"streams": streams, "dem": dem}[changed], (written, written))
    assert seeds_from(streams, dem, seed_file) == FOUND
    assert len(gpd.read_file(seed_file)) == 2


def test_a_seed_file_for_other_streams_is_replaced(tmp_path: Path) -> None:
    streams, dem = write_streams_and_dem(tmp_path)
    gpd.GeoDataFrame({"LINKNO": [99]}, geometry=[Point(0, 0)], crs=CRS).to_file(tmp_path / "seeds.gpkg", driver="GPKG")
    assert seeds_from(streams, dem, tmp_path / "seeds.gpkg") == FOUND
