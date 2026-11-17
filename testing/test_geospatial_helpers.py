"""The helper scripts that prepare ARC's inputs and use its outputs, on rasters either way up."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from test_io import PROJECTED, write_raster
from test_pipeline import ROWS, flip_to_south_up, write_channel

from arc import pipeline
from arc import process_geospatial_data as geospatial
from arc.config import Configs
from arc.create_geojson import (
    Cell_Center_Coordinates,
    Write_SEED_Data_To_File_Using_Stream_Raster,
)
from arc.io import Raster


@pytest.mark.parametrize("south_up", [False, True], ids=["north-up", "south-up"])
def test_the_land_cover_and_stream_rasters_are_made_on_the_dem_s_grid(tmp_path: Path, south_up: bool) -> None:
    inputs = write_channel(tmp_path / "site")
    dem = flip_to_south_up(inputs["DEM_File"], tmp_path / "dem_south.tif") if south_up else inputs["DEM_File"]
    minx, miny, maxx, maxy, _, _, ncols, nrows, geotransform, _ = geospatial.Get_Raster_Details(dem)
    land, streams = str(tmp_path / "land.tif"), str(tmp_path / "streams.tif")

    geospatial.Create_ARC_LandRaster(inputs["LU_Raster_SameRes"], land, [minx, maxy, maxx, miny], ncols, nrows)
    geospatial.Match_DEM_Orientation(land, geotransform)
    geospatial.Create_ARC_StrmRaster(inputs["StrmShp_File"], streams, [minx, miny, maxx, maxy], ncols, nrows, "LINKNO")
    geospatial.Match_DEM_Orientation(streams, geotransform)

    assert Raster(land).same_grid(Raster(dem)) and Raster(streams).same_grid(Raster(dem))
    stream_cells = Raster(streams).read_array()
    north_up_cells = stream_cells[::-1] if south_up else stream_cells
    expected = Raster(inputs["Stream_File"]).read_array()
    assert np.array_equal(north_up_cells > 0, expected > 0)


def test_cell_centres_come_from_the_geotransform_whichever_way_up(tmp_path: Path) -> None:
    inputs = write_channel(tmp_path)
    south = flip_to_south_up(inputs["DEM_File"], tmp_path / "dem_south.tif")
    rows, cols = np.array([0, 5, ROWS - 1]), np.array([0, 17, 239])

    x, y = Cell_Center_Coordinates(Raster(inputs["DEM_File"]).geotransform, rows, cols)

    np.testing.assert_allclose(x, 500000.0 + cols + 0.5)  # 1 m cells, the top left corner at (500000, 3800000)
    np.testing.assert_allclose(y, 3800000.0 - rows - 0.5)
    np.testing.assert_allclose(Cell_Center_Coordinates(Raster(south).geotransform, ROWS - 1 - rows, cols), (x, y))


def test_seed_points_are_the_ends_of_streams_and_their_cells_on_the_raster_s_edge(tmp_path: Path) -> None:
    """A stream along the first row is on the edge all along; one along the second-last row isn't, and only its ends
    are SEED points."""
    streams = np.zeros((6, 8), np.int32)
    streams[0, 1:7] = 11
    streams[4, 1:7] = 22
    dem = np.arange(48, dtype=np.float32).reshape(6, 8)
    write_raster(tmp_path / "streams.tif", streams, PROJECTED, 32616)
    write_raster(tmp_path / "dem.tif", dem, PROJECTED, 32616)

    _, _, comids, rows, cols, _, _ = Write_SEED_Data_To_File_Using_Stream_Raster(
        str(tmp_path / "streams.tif"), str(tmp_path / "dem.tif"), str(tmp_path / "seed.txt"))

    seeds = sorted(zip(comids, rows, cols))
    assert seeds == [(11, 0, c) for c in range(1, 7)] + [(22, 4, 1), (22, 4, 6)]


def test_the_starter_input_file_names_the_reach_fields_and_the_water_class(tmp_path: Path) -> None:
    path = tmp_path / "ARC_Input_File.txt"
    geospatial.Create_ARC_Model_Input_File(str(path), "dem.tif", "COMID", "qmax", "qbase", "streams.tif", "land.tif",
                                           "flows.csv", "vdt.csv", "curves.csv", "n.txt", "flood.tif", "depth.tif",
                                           "bathy.tif", "xs.txt", False, False, True, STRM_SHP_File="streams.gpkg",
                                           reach_id_field="COMID", downstream_reach_id_field="NextDownID",
                                           lc_water_value=11)

    configs = Configs.from_file(path)

    assert (configs.reach_id, configs.downstream_reach_id, configs.lc_water_value) == ("COMID", "NextDownID", 11)
    assert configs.bathy_out_file == "bathy.tif" and configs.flow_file_bf == "qbase"


def test_step_2_reads_step_1_s_flow_file_whatever_the_stream_lines_call_their_ids(tmp_path: Path) -> None:
    """Step 1 writes the IDs in a column named COMID, and the GEOGLOWS stream lines hold them in LINKNO: the input
    file Step 2 writes takes each from its own name, and ARC finds the flow file's reaches in the stream raster."""
    import geopandas as gpd

    inputs = write_channel(tmp_path / "site")
    main = tmp_path / "main"
    for folder, source, name in (("DEM", inputs["DEM_File"], "DEM.tif"),
                                 ("LandCover", inputs["LU_Raster_SameRes"], "LandCover.tif")):
        (main / folder).mkdir(parents=True)
        (main / folder / name).write_bytes(Path(source).read_bytes())
    (main / "StrmShp").mkdir()
    lines = gpd.read_file(inputs["StrmShp_File"]).drop(columns="COMID")  # only LINKNO, as GEOGLOWS's
    lines.to_file(main / "StrmShp" / "StreamShapefile.shp")

    geospatial.Process_ARC_Geospatial_Data(str(main), "LINKNO", "qmax", "qbf", inputs["Flow_File"], False, False,
                                           downstream_id_field="DSLINKNO")
    configs = Configs.from_file(main / "ARC_InputFiles" / "ARC_Input_File.txt")

    assert (configs.flow_file_id, configs.reach_id, configs.downstream_reach_id) == ("COMID", "LINKNO", "DSLINKNO")
    flows = pipeline.read_flows(configs)
    streams = Raster(configs.stream_file).read_array()
    assert set(flows.index) == set(np.unique(streams[streams > 0]))
