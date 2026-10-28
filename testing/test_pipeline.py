"""The whole workflow, arc.pipeline, on synthetic channels whose answers are known, and on real sites when their data
are on this machine.

The synthetic channel is a straight trapezoid along a row of 1 m cells: a 10 m wide bed, banks rising 1 m every 2 m
to 2 m high, and floodplains rising 1%, falling 0.001 per metre downstream, in two reaches. Along a row the cross
sections run along columns and every ordinate is a cell centre, so the sampled channel is exactly the trapezoid and
its rating curve is Manning's equation for a trapezoid.
"""
from __future__ import annotations

import math
from dataclasses import fields
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from osgeo import gdal, osr
from scipy.optimize import brentq
from shapely.geometry import LineString

from arc import pipeline
from arc.bathymetry import Banks
from arc.bathymetry.depth import trapezoid_depth
from arc.config import Configs
from arc.hydraulics import DepthRoughness, discharge
from arc.io import Raster
from arc.outputs.rating_curves import WSE, Q, T
from arc.xsection.slope import MIN_SLOPE
from arc.xsection.xsection import XSection

gdal.UseExceptions()

ROWS, COLS, CENTER_ROW = 61, 240, 30
SLOPE = 0.001
BED = 100.0
N = 0.035
N_RASTER = float(np.float32(N))  # Manning's n as the n raster holds it, in float32
Q_MAX = {11: 15.0, 12: 18.0}
BASEFLOW = {11: 2.0, 12: 2.5}
INCREMENTS = 10


def ground(distance: np.ndarray) -> np.ndarray:
    """The channel's height above its bed at a distance from its centre line."""
    return np.where(distance <= 5.0, 0.0, np.where(distance <= 9.0, (distance - 5.0) / 2.0,
                                                   2.0 + 0.01 * (distance - 9.0)))


def trapezoid_discharge(depth: float, slope: float = SLOPE, n: float = N_RASTER) -> float:
    area = depth * (10.0 + 2.0 * depth)
    perimeter = 10.0 + 2.0 * depth * math.sqrt(5.0)
    return area * (area / perimeter) ** (2 / 3) * math.sqrt(slope) / n


def normal_depth(q: float, slope: float = SLOPE, n: float = N_RASTER) -> float:
    return brentq(lambda y: trapezoid_discharge(y, slope, n) - q, 1e-6, 2.0, xtol=1e-14)


def _write_raster(path: Path, values: np.ndarray, geotransform, srs: osr.SpatialReference, gdal_type) -> str:
    ds = gdal.GetDriverByName("GTiff").Create(str(path), values.shape[1], values.shape[0], 1, gdal_type)
    ds.SetGeoTransform(geotransform)
    ds.SetProjection(srs.ExportToWkt())
    ds.GetRasterBand(1).WriteArray(values)
    ds = None
    return str(path)


def _projected() -> tuple[osr.SpatialReference, tuple]:
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(32616)
    return srs, (500000.0, 1.0, 0.0, 3800000.0, 0.0, -1.0)


def write_inputs(folder: Path, dem: np.ndarray, streams: np.ndarray, srs, geotransform, reaches: dict, *,
                 land_cover: np.ndarray | None = None, water_n: float = N, land_n: float = N) -> dict:
    """ARC's inputs for a DEM and stream raster, and an input mapping for them. reaches holds each reach's line, as
    cells (row, col) at its ends, its downstream reach (-1 for none), drainage area, baseflow and maximum flow."""
    folder.mkdir(parents=True, exist_ok=True)
    land_cover = np.ones(dem.shape, dtype=np.uint8) if land_cover is None else land_cover
    _write_raster(folder / "dem.tif", dem, geotransform, srs, gdal.GDT_Float64)
    _write_raster(folder / "streams.tif", streams.astype(np.int32), geotransform, srs, gdal.GDT_Int32)
    _write_raster(folder / "land.tif", land_cover, geotransform, srs, gdal.GDT_Byte)
    (folder / "mannings.txt").write_text(f"LC_ID\tDescription\tManning_n\n1\tGrass\t{land_n}\n80\tWater\t{water_n}\n",
                                         encoding="utf-8")
    ids = list(reaches)
    pd.DataFrame({"COMID": ids, "qbf": [reaches[r]["baseflow"] for r in ids],
                  "qmax": [reaches[r]["q_max"] for r in ids]}).to_csv(folder / "flows.csv", index=False)
    point = lambda cell: (geotransform[0] + (cell[1] + 0.5) * geotransform[1],
                          geotransform[3] + (cell[0] + 0.5) * geotransform[5])
    lines = [LineString([point(reaches[r]["start"]), point(reaches[r]["end"])]) for r in ids]
    gpd.GeoDataFrame({"LINKNO": ids, "COMID": ids, "DSLINKNO": [reaches[r]["downstream"] for r in ids],
                      "DA": [reaches[r].get("area", 50.0) for r in ids]}, geometry=lines,
                     crs=srs.ExportToWkt()).to_file(folder / "streams.gpkg")
    return {"DEM_File": str(folder / "dem.tif"), "Stream_File": str(folder / "streams.tif"),
            "LU_Raster_SameRes": str(folder / "land.tif"), "LU_Manning_n": str(folder / "mannings.txt"),
            "Flow_File": str(folder / "flows.csv"), "Flow_File_ID": "COMID", "Flow_File_BF": "qbf",
            "Flow_File_QMax": "qmax", "StrmShp_File": str(folder / "streams.gpkg"), "reach_id": "LINKNO",
            "downstream_reach_id": "DSLINKNO", "Degree_Manip": 0, "Low_Spot_Range": 0,
            "Gen_Dir_Dist": 10, "Gen_Slope_Dist": 10, "Stream_Slope_Method": "local_average",
            "VDT_Database_NumIterations": INCREMENTS, "Print_VDT_Database": str(folder / "vdt.csv"),
            "Depth_Varying_N": False}


def write_channel(folder: Path, *, geographic: bool = False, water_n: float = N, land_n: float = N,
                  land_cover: np.ndarray | None = None, south=ground, bed=None) -> dict:
    """The synthetic channel's inputs (see the notes above), and an input mapping for them. south gives the ground
    south of the stream (rows below it) if not the same as the north's, and bed its bed elevation at each column
    distance in metres if it doesn't fall evenly."""
    folder.mkdir(parents=True, exist_ok=True)
    if geographic:
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(4326)
        srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        latitude = 34.0
        # Cells of about 1 m: the degrees along each axis that the ellipsoid makes 1 m at this latitude
        from pyproj import Geod
        geod = Geod(ellps="WGS84")
        dy_deg = 1.0 / geod.inv(0.0, latitude, 0.0, latitude + 1.0)[2]
        dx_deg = 1.0 / geod.inv(0.0, latitude, 1.0, latitude)[2]
        geotransform = (-84.0, dx_deg, 0.0, latitude + 0.5 * ROWS * dy_deg, 0.0, -dy_deg)
    else:
        srs, geotransform = _projected()
    _write_raster(folder / "dem.tif", np.zeros((ROWS, COLS)), geotransform, srs, gdal.GDT_Float64)
    dx, dy = Raster(folder / "dem.tif").cell_size_in_metres()

    rows, cols = np.mgrid[0:ROWS, 0:COLS]
    distance = np.abs(rows - CENTER_ROW) * dy
    bed_elevation = BED - SLOPE * cols * dx if bed is None else bed(cols * dx)
    dem = bed_elevation + np.where(rows > CENTER_ROW, south(distance), ground(distance))
    streams = np.zeros((ROWS, COLS), dtype=np.int32)
    streams[CENTER_ROW, 20:120] = 11
    streams[CENTER_ROW, 120:220] = 12
    reaches = {11: {"start": (CENTER_ROW, 20), "end": (CENTER_ROW, 119), "downstream": 12, "area": 50.0,
                    "baseflow": BASEFLOW[11], "q_max": Q_MAX[11]},
               12: {"start": (CENTER_ROW, 120), "end": (CENTER_ROW, 219), "downstream": -1, "area": 60.0,
                    "baseflow": BASEFLOW[12], "q_max": Q_MAX[12]}}
    return write_inputs(folder, dem, streams, srs, geotransform, reaches, land_cover=land_cover, water_n=water_n,
                        land_n=land_n)


def interior(results: pipeline.Results) -> np.ndarray:
    """The cells more than a gen_slope_dist from either end of the stream, where the slope is the bed's."""
    cols = results.cells.cols
    return (cols >= 40) & (cols <= 200)


def used_cells(results: pipeline.Results, inputs: dict) -> np.ndarray:
    """On the DEM's grid, True at the centre cell of each cross section the run used."""
    used = np.zeros(Raster(inputs["DEM_File"]).read_array().shape, dtype=bool)
    for section in results.sections:
        if section is not None and section.usable:
            used[section.row, section.col] = True
    return used


# --- The synthetic channel ----------------------------------------------------------------------------------------


def test_the_rating_curve_is_manning_s_for_the_trapezoid(tmp_path: Path) -> None:
    results = pipeline.run(Configs.from_mapping(write_channel(tmp_path)), quiet=True)
    inside = interior(results)

    assert inside.sum() == 161 and np.allclose(results.slopes[inside], SLOPE, rtol=1e-9)
    for k in np.flatnonzero(inside):
        reach = int(results.cells.comids[k])
        increments = results.curves.increments[k]
        bed = float(results.sections[k].xs.elevations[results.sections[k].xs.elevations.size // 2])
        assert increments[-1, WSE] - bed == pytest.approx(normal_depth(Q_MAX[reach]), abs=1e-9)
        for q, v, t, wse, p in increments:
            depth = wse - bed
            assert q == pytest.approx(trapezoid_discharge(depth), rel=1e-9)
            assert t == pytest.approx(10.0 + 4.0 * depth, rel=1e-12)
            assert p == pytest.approx(10.0 + 2.0 * depth * math.sqrt(5.0), rel=1e-12)
    vdt = pd.read_csv(tmp_path / "vdt.csv")
    assert len(vdt) == results.cells.count == 200
    assert (vdt[f"q_{INCREMENTS}"] == vdt["COMID"].map(Q_MAX)).all()  # rounded to 3 decimals
    assert np.allclose(vdt["XS_Angle"], math.pi / 2)  # the cross sections run along the columns


def test_depth_varying_roughness_and_the_slope_factor_reach_the_rating_curves(tmp_path: Path) -> None:
    inputs = {**write_channel(tmp_path), "Depth_Varying_N": True, "k_decay": 5.0, "shallow_factor": 2.0,
              "deep_factor": 0.9, "slope_adjustment_factor": 1.2}
    results = pipeline.run(Configs.from_mapping(inputs), quiet=True, write=False)
    roughness = DepthRoughness(5.0, 2.0, 0.9)

    for k in np.flatnonzero(interior(results)):
        xs = results.sections[k].xs
        for q, _, _, wse, _ in results.curves.increments[k]:
            assert q == pytest.approx(discharge(xs, SLOPE, wse=wse, roughness=roughness, slope_factor=1.2), rel=1e-9)
        assert results.curves.increments[k, -1, Q] == pytest.approx(Q_MAX[int(results.cells.comids[k])], rel=1e-9)
    assert results.vdt is None  # nothing written


def test_the_water_s_roughness_goes_between_the_banks(tmp_path: Path) -> None:
    """With a rougher land cover, the channel between its banks (±9 m) still takes the water's n."""
    results = pipeline.run(Configs.from_mapping(write_channel(tmp_path, land_n=0.1)), quiet=True, write=False)
    section = results.sections[100]
    offsets = np.arange(section.xs.elevations.size) - section.xs.elevations.size // 2
    on_raster = section.xs.elevations < pipeline.OFF_RASTER_ELEVATION

    assert np.all(section.xs.mannings_n[np.abs(offsets) <= 9] == np.float32(N))
    assert np.all(section.xs.mannings_n[(np.abs(offsets) >= 10) & on_raster] == np.float32(0.1))
    k = 100
    bed = float(section.xs.elevations[section.xs.elevations.size // 2])
    assert results.curves.increments[k, -1, WSE] - bed == pytest.approx(normal_depth(Q_MAX[12]), abs=1e-9)


def test_on_a_geographic_grid_the_cells_are_measured_in_metres(tmp_path: Path) -> None:
    """The cells are about 1 m, measured along the ellipsoid at the raster's middle latitude."""
    inputs = write_channel(tmp_path, geographic=True)
    results = pipeline.run(Configs.from_mapping(inputs), quiet=True, write=False)
    inside = interior(results)
    _, dy = Raster(inputs["DEM_File"]).cell_size_in_metres()

    assert dy == pytest.approx(1.0, rel=1e-3)
    assert np.allclose(results.slopes[inside], SLOPE, rtol=1e-6)
    for k in np.flatnonzero(inside)[::20]:
        section = results.sections[k]
        assert section.xs.ordinate_distance == pytest.approx(dy, rel=1e-12)  # along a column
        bed = float(section.xs.elevations[section.xs.elevations.size // 2])
        # The trapezoid's corners fall a little between the ordinates, 0.9999 m apart
        assert results.curves.increments[k, -1, WSE] - bed == pytest.approx(
            normal_depth(Q_MAX[int(results.cells.comids[k])]), rel=1e-4)


def flip_to_south_up(source: str, target: Path) -> str:
    """The raster with its rows the other way up: its origin at the bottom edge and a positive pixel height."""
    ds = gdal.Open(source)
    x0, dx, rx, y0, ry, dy = ds.GetGeoTransform()
    out = gdal.GetDriverByName("GTiff").Create(str(target), ds.RasterXSize, ds.RasterYSize, 1,
                                                ds.GetRasterBand(1).DataType)
    out.SetGeoTransform((x0, dx, rx, y0 + ds.RasterYSize * dy, ry, -dy))
    out.SetProjection(ds.GetProjection())
    out.GetRasterBand(1).WriteArray(ds.GetRasterBand(1).ReadAsArray()[::-1])
    out = ds = None
    return str(target)


@pytest.mark.parametrize("geographic", [False, True], ids=["projected", "geographic"])
def test_a_south_up_raster_gives_the_same_results_with_its_rows_the_other_way(tmp_path: Path,
                                                                              geographic: bool) -> None:
    """The channel with steeper banks to the south, so that its sides differ. The cross-section angle is measured
    from the columns towards the rows, so it turns the other way too."""
    steep = lambda d: np.where(d <= 5.0, 0.0, np.where(d <= 7.0, d - 5.0, 2.0 + 0.01 * (d - 7.0)))
    north = {**write_channel(tmp_path, south=steep, geographic=geographic),
             "BATHY_Out_File": str(tmp_path / "bathy.tif"), "Degree_Manip": 30, "Degree_Interval": 5,
             "Low_Spot_Range": 2}
    south = {**north, "Print_VDT_Database": str(tmp_path / "south_vdt.csv"),
             "BATHY_Out_File": str(tmp_path / "south_bathy.tif")}
    for key in ("DEM_File", "Stream_File", "LU_Raster_SameRes"):
        south[key] = flip_to_south_up(north[key], tmp_path / f"south_{Path(north[key]).name}")

    north_up = pipeline.run(Configs.from_mapping(north), quiet=True)
    south_up = pipeline.run(Configs.from_mapping(south), quiet=True)

    vdt = pd.read_csv(north["Print_VDT_Database"]).set_index(["Row", "Col"]).sort_index()
    flipped = pd.read_csv(south["Print_VDT_Database"])
    flipped["Row"] = ROWS - 1 - flipped["Row"]
    flipped = flipped.set_index(["Row", "Col"]).sort_index()
    assert len(vdt) == 200 and flipped.index.equals(vdt.index)
    pd.testing.assert_frame_equal(flipped.drop(columns="XS_Angle"), vdt.drop(columns="XS_Angle"))
    assert np.allclose((math.pi - flipped["XS_Angle"]) % math.pi, vdt["XS_Angle"], atol=1e-6)
    np.testing.assert_array_equal(south_up.bathymetry[::-1], north_up.bathymetry)
    assert Raster(south["BATHY_Out_File"]).same_grid(Raster(south["DEM_File"]))


@pytest.mark.parametrize("use_banks", [False, True])
def test_the_bathymetry_carves_a_channel_for_the_baseflow(tmp_path: Path, use_banks: bool) -> None:
    """Without bank elevations the channel is carved below the stream cell, which the gap fill keeps below the DEM.
    With them it is carved below the smoothed bank elevation, here the banks' top, 2 m above the bed."""
    inputs = {**write_channel(tmp_path), "BATHY_Out_File": str(tmp_path / "bathy.tif"), "Bathy_Use_Banks": use_banks,
              "XS_Out_File": str(tmp_path / "xs.txt")}
    results = pipeline.run(Configs.from_mapping(inputs), quiet=True)
    bathymetry = Raster(tmp_path / "bathy.tif").read_array()
    dem = Raster(tmp_path / "dem.tif").read_array()
    carved = np.isfinite(bathymetry)

    assert np.array_equal(np.isfinite(results.bathymetry), carved)
    assert carved[CENTER_ROW, 40:200].all()
    assert not carved[:CENTER_ROW - 10].any() and not carved[CENTER_ROW + 11:].any()  # only between the banks
    if not use_banks:
        assert np.all(bathymetry[carved] <= dem[carved] + 1e-6)
        assert np.all(bathymetry[CENTER_ROW, 40:200] < dem[CENTER_ROW, 40:200])
    for k in np.flatnonzero(interior(results)):
        section = results.sections[k]
        assert section.usable
        bed = section.xs.elevations[section.xs.elevations.size // 2]
        reference = section.bank_elevation if use_banks else section.dem_low_point
        assert bed < reference
        # Each rating curve is on the carved cross section
        assert results.curves.increments[k, -1, Q] == pytest.approx(Q_MAX[int(results.cells.comids[k])], rel=1e-9)
    xs = pd.read_csv(tmp_path / "xs.txt", sep="\t")
    assert len(xs) == results.cells.count
    assert list(xs.columns) == ['COMID', 'Row', 'Col', 'XS1_Profile', 'Ordinate_Dist', 'Manning_N_Raster1',
                                'XS2_Profile', 'Manning_N_Raster2', 'r1', 'c1', 'r2', 'c2']


@pytest.mark.parametrize("reach_average", [False, True], ids=["cells", "reach average"])
@pytest.mark.parametrize("bathymetry", [False, True], ids=["DEM", "bathymetry"])
def test_the_elevation_written_is_the_rating_curve_s_bed(tmp_path: Path, bathymetry: bool,
                                                         reach_average: bool) -> None:
    """The VDT database's Elev and the curve file's BaseElev are both the elevation of the centre of the cross section
    the rating curve was worked out on: the carved bed with bathymetry, and otherwise the DEM. So a water surface
    less it is the water's depth."""
    inputs = {**write_channel(tmp_path), "Print_Curve_File": str(tmp_path / "curves.csv"),
              "Reach_Average_Curve_File": reach_average}
    if bathymetry:
        inputs["BATHY_Out_File"] = str(tmp_path / "bathy.tif")
    results = pipeline.run(Configs.from_mapping(inputs), quiet=True)
    vdt = pd.read_csv(inputs["Print_VDT_Database"]).set_index(["Row", "Col"])
    curves = pd.read_csv(tmp_path / "curves.csv").set_index(["Row", "Col"])
    dem = Raster(inputs["DEM_File"]).read_array()

    assert "DEM_Elev" not in curves.columns and "BaseElev" not in vdt.columns
    for k, section in enumerate(results.sections):
        cell = (section.row, section.col)
        bed = section.xs.elevations[section.xs.elevations.size // 2]
        assert vdt.loc[cell, "Elev"] == pytest.approx(bed, abs=5e-4)
        assert curves.loc[cell, "BaseElev"] == vdt.loc[cell, "Elev"]
        if bathymetry:
            assert bed < dem[cell] - 0.1
            assert results.bathymetry[cell] == pytest.approx(vdt.loc[cell, "Elev"], abs=5e-4)
        else:
            assert bed == dem[cell]
        assert vdt.loc[cell, "wse_1"] > vdt.loc[cell, "Elev"]


def test_the_bathymetry_depth_is_solved_with_the_water_class_s_n(tmp_path: Path) -> None:
    """The channel is carved to carry the baseflow with the n it has between its banks, the water class's, so a
    rougher channel is carved deeper."""
    depths = {}
    for water_n in (0.03, 0.06):
        folder = tmp_path / f"n{water_n}"
        inputs = {**write_channel(folder, water_n=water_n), "BATHY_Out_File": str(folder / "bathy.tif"),
                  "Bathy_Bed_Cap": False}
        results = pipeline.run(Configs.from_mapping(inputs), quiet=True, write=False)
        depths[water_n] = []
        for k in np.flatnonzero(interior(results)):
            section = results.sections[k]
            depth = section.dem_low_point - section.xs.elevations[section.xs.elevations.size // 2]
            top = section.carve_banks.top_width
            # Bathy_Trap_H is 0.2, so the bed is 0.6 of the top width; n is float32 as the Manning's table is read.
            # The bed's running median is a step (1 mm here) off at a reach's ends (arc.bathymetry.bed_smoothing).
            assert depth == pytest.approx(trapezoid_depth(results.baseflow[k], 0.6 * top, top, results.slopes[k],
                                                          float(np.float32(water_n))), abs=2e-3)
            depths[water_n].append(depth)
    assert np.all(np.array(depths[0.06]) > np.array(depths[0.03]))


def test_the_drainage_area_depth_is_carved_as_it_is(tmp_path: Path) -> None:
    """A power-law target depth applies whatever the baseflow, and the bed smoothing then evens it out."""
    inputs = {**write_channel(tmp_path), "BATHY_Out_File": str(tmp_path / "bathy.tif"), "Bathy_Bed_Cap": False,
              "drainage_area_field": "DA", "coefficient_depth": 0.1, "exponent_depth": 0.5}
    results = pipeline.run(Configs.from_mapping(inputs), quiet=True, write=False)

    for k in np.flatnonzero(interior(results)):
        section = results.sections[k]
        depth = section.dem_low_point - section.xs.elevations[section.xs.elevations.size // 2]
        target = 0.1 * (50.0 if results.cells.comids[k] == 11 else 60.0) ** 0.5
        assert depth == pytest.approx(target, abs=0.01)


def test_with_the_standard_step_each_bed_is_its_own_unless_smoothing_is_asked_for(tmp_path: Path) -> None:
    """Unless Bathy_Smooth_Bed is given, a run with the standard step carves each cross section's own bed, as
    Bathy_Smooth_Bed False does, and one without it smooths the beds. A bed that rises and falls a little gives each
    cross section its own slope and so its own depth, which the smoothing changes."""
    bumpy = lambda x: BED - SLOPE * x + 0.05 * np.sin(x / 7.0)

    def beds(folder: Path, **extra) -> np.ndarray:
        inputs = {**write_channel(folder, bed=bumpy), "BATHY_Out_File": str(folder / "bathy.tif"),
                  "Stream_Slope_Method": "local_average", **extra}
        results = pipeline.run(Configs.from_mapping(inputs), quiet=True, write=False)
        return np.array([s.xs.elevations[s.xs.elevations.size // 2] for s in results.sections])

    with_the_step = beds(tmp_path / "step", Standard_Step=True)
    without_it = beds(tmp_path / "normal")

    np.testing.assert_array_equal(with_the_step, beds(tmp_path / "step_off", Standard_Step=True,
                                                      Bathy_Smooth_Bed=False))
    np.testing.assert_array_equal(without_it, beds(tmp_path / "smoothed", Bathy_Smooth_Bed=True))
    assert not np.allclose(with_the_step, without_it)


def representative_inputs(folder: Path, suffix: str = ".csv", **overrides) -> dict:
    """The synthetic channel's inputs for a run that only builds representative cross sections."""
    inputs = {**write_channel(folder), "Build_Representative_Cross_Section": True,
              "Representative_Cross_Section_File": str(folder / f"representative{suffix}"), **overrides}
    del inputs["Print_VDT_Database"]
    return inputs


@pytest.mark.parametrize("suffix", [".csv", ".parquet"])
def test_representative_cross_sections(tmp_path: Path, suffix: str) -> None:
    path = tmp_path / f"representative{suffix}"
    inputs = representative_inputs(tmp_path, suffix)
    results = pipeline.run(Configs.from_mapping(inputs), quiet=True)
    df = pd.read_parquet(path) if suffix == ".parquet" else pd.read_csv(path)

    assert results.curves is None and not (tmp_path / "vdt.csv").exists()
    assert sorted(df["COMID"].unique()) == [11, 12]
    assert df.groupby("COMID").size().tolist() == [250, 250]
    stage = df[(df["COMID"] == 12) & (df["Depth_Stage_Index"] == 10)].iloc[0]
    # 1 m deep in the trapezoid; the slopes at the reach's ends are the bed's too
    assert stage["Mean_Cross_Sectional_Area"] == pytest.approx(12.0, rel=1e-6)
    assert stage["Mean_Discharge"] == pytest.approx(trapezoid_discharge(1.0), rel=1e-4)


@pytest.mark.parametrize("bathymetry", [{}, {"BATHY_Out_File": "bathy.tif"},
                                        {"BATHY_Out_File": "bathy.tif", "drainage_area_field": "DA",
                                         "coefficient_depth": 0.1, "exponent_depth": 0.5}],
                         ids=["no-bathymetry", "baseflow-bathymetry", "power-law-bathymetry"])
def test_representative_cross_sections_beside_the_rating_curves(tmp_path: Path, bathymetry: dict) -> None:
    """Asked for both, a run makes both from the same cross sections, and the rating curves and bathymetry are the
    ones it makes without the representative cross sections. The maximum flows are above bankfull: a run that only
    builds representative cross sections reads no maximum flow, so its banks aren't held below that water, which here
    they then aren't anyway."""
    def run(folder: Path, **extra) -> pipeline.Results:
        inputs = with_max_flows_above_bankfull({**write_channel(folder), **extra})
        if "BATHY_Out_File" in extra:
            inputs["BATHY_Out_File"] = str(folder / extra["BATHY_Out_File"])
        return pipeline.run(Configs.from_mapping(inputs), quiet=True)

    alone = run(tmp_path / "alone", **bathymetry)
    both = run(tmp_path / "both", **bathymetry, Build_Representative_Cross_Section=True,
               Representative_Cross_Section_File=str(tmp_path / "both" / "representative.csv"))

    np.testing.assert_array_equal(both.curves.increments, alone.curves.increments)
    np.testing.assert_array_equal(both.curves.metadata, alone.curves.metadata)
    pd.testing.assert_frame_equal(pd.read_csv(tmp_path / "both" / "vdt.csv"), pd.read_csv(tmp_path / "alone" / "vdt.csv"))
    if bathymetry:
        np.testing.assert_array_equal(both.bathymetry, alone.bathymetry)
    representative = pd.read_csv(tmp_path / "both" / "representative.csv")
    assert sorted(representative["COMID"].unique()) == [11, 12]
    if "coefficient_depth" not in bathymetry:
        # Carving for the baseflow as the rating curves do, it is what a run building only them makes
        only = pipeline.run(Configs.from_mapping(representative_inputs(
            tmp_path / "only", **{k: str(tmp_path / "only" / v) for k, v in bathymetry.items()})), quiet=True)
        pd.testing.assert_frame_equal(both.representative, only.representative, check_exact=True)


def test_with_the_standard_step_the_representative_cross_sections_carry_the_rating_curves(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Each rated cross section's sample has its rating curve's water surfaces and flows. On the synthetic channel the
    outlet reach's water stays at normal depth, and so do its representative cross sections, but for the discharge
    between its rating curves' water surfaces, the chord above normal depth's that the VDT database is read along;
    reach 11's is backed up (by the larger flow below it at the same share of the maximum flow), and its cross sections
    carry less at each stage, at a friction slope below the bed's."""
    samples, original = [], pipeline.representative_cross_section_dataframe

    def capture(given, **kwargs):
        samples.append(list(given))
        return original(given, **kwargs)

    monkeypatch.setattr(pipeline, "representative_cross_section_dataframe", capture)

    def run(folder: Path, **extra) -> pipeline.Results:
        inputs = {**write_channel(folder), "Build_Representative_Cross_Section": True,
                  "Representative_Cross_Section_File": str(folder / "representative.csv"), **extra}
        return pipeline.run(Configs.from_mapping(inputs), quiet=True)

    stepped = run(tmp_path / "step", Standard_Step=True)
    plain = run(tmp_path / "plain")

    usable = [k for k, section in enumerate(stepped.sections) if section is not None and section.usable]
    increments = stepped.curves.increments
    rated = [k for k in usable if np.isfinite(increments[k, -1, Q])]
    assert len(rated) > 100
    for k, sample in zip(usable, samples[0], strict=True):
        if k in rated:
            np.testing.assert_array_equal(sample.curve, increments[k][:, [WSE, Q]])
        else:
            assert sample.curve is None
    assert all(sample.curve is None for sample in samples[1])
    # The cross sections are all alike, so which the two-deviation filter leaves out is down to rounding (and the
    # step's tolerance): the count isn't compared, nor the mean water surface, which their beds' fall moves
    by_reach = [{comid: df[df["COMID"] == comid].drop(columns=["Hydraulic_Sample_Count", "Mean_WSE"])
                 .reset_index(drop=True) for comid in (11, 12)} for df in (stepped.representative, plain.representative)]
    outlet, at_normal = by_reach[0][12], by_reach[1][12]
    hydraulics = ["Mean_Discharge", "Mean_Velocity", "Representative_Velocity", "Mean_Section_Velocity",
                  "Mean_Friction_Slope"]
    pd.testing.assert_frame_equal(outlet.drop(columns=hydraulics), at_normal.drop(columns=hydraulics),
                                  check_exact=False, rtol=1e-5)
    # Above the top of every rating curve, normal depth's; below, the chord's, a few percent at most above it here
    top = max(np.nanmax(sample.curve[:, 0]) - sample.thalweg for sample in samples[0]
              if sample.comid == 12 and sample.curve is not None)
    above = outlet["Depth_Stage_Meters"] > top
    assert above.sum() > 200
    pd.testing.assert_frame_equal(outlet.loc[above, hydraulics], at_normal.loc[above, hydraulics],
                                  check_exact=False, rtol=1e-5)
    chord = (outlet.loc[~above, "Mean_Discharge"] / at_normal.loc[~above, "Mean_Discharge"]).to_numpy()
    assert np.all((chord > 1 - 1e-5) & (chord < 1.05))
    backed_up, normal = by_reach[0][11], by_reach[1][11]
    assert (backed_up["Mean_Discharge"] < 0.95 * normal["Mean_Discharge"]).all()
    assert (backed_up["Mean_Friction_Slope"] < 0.9 * SLOPE).all()
    pd.testing.assert_frame_equal(backed_up[["Mean_Cross_Sectional_Area", "Mean_Top_Width"]],
                                  normal[["Mean_Cross_Sectional_Area", "Mean_Top_Width"]], check_exact=False, rtol=1e-9)


def test_representative_cross_sections_carved_for_the_baseflow_are_of_the_reaches_with_flows(tmp_path: Path) -> None:
    """Carving for the baseflow, a run building only representative cross sections works on the reaches in both the
    flow file and the stream raster."""
    inputs = representative_inputs(tmp_path, BATHY_Out_File=str(tmp_path / "bathy.tif"))
    flows = pd.read_csv(inputs["Flow_File"])
    flows[flows["COMID"] == 11].to_csv(inputs["Flow_File"], index=False)
    configs = Configs.from_mapping(inputs)

    results = pipeline.run(configs, quiet=True)

    assert configs.use_representative_baseflow_bathymetry
    assert set(results.cells.comids.tolist()) == {11}
    assert results.representative["COMID"].unique().tolist() == [11]


def test_the_drainage_area_power_laws_give_each_reach_its_depth_and_width(tmp_path: Path) -> None:
    """depth = coefficient_depth * area ** exponent_depth, and the same for the width, from each reach's first
    feature."""
    layer = gpd.GeoDataFrame({"LINKNO": [101, 202, 101], "DA": [4.0, 9.0, 100.0]},
                             geometry=[LineString([(0.0, 0.0), (1.0, 0.0)])] * 3, crs="EPSG:4326")
    configs = Configs.from_mapping({"StrmShp_File": "streams.gpkg", "reach_id": "LINKNO", "drainage_area_field": "DA",
                                    "coefficient_depth": 2.0, "exponent_depth": 0.5, "coefficient_width": 3.0,
                                    "exponent_width": 1.0})

    targets = pipeline.power_law_targets(configs, layer)

    assert targets == {101: pytest.approx((4.0, 12.0)), 202: pytest.approx((6.0, 27.0))}
    with pytest.raises(ValueError, match="must be positive"):
        pipeline.power_law_targets(configs, layer.assign(DA=[4.0, 0.0, 1.0]))


def test_representative_cross_sections_carved_for_the_baseflow_need_no_maximum_flow(tmp_path: Path) -> None:
    flows = tmp_path / "flows.csv"
    flows.write_text("COMID,qbf\n11,2.5\n", encoding="utf-8")
    configs = Configs.from_mapping({"Flow_File": str(flows), "Flow_File_ID": "COMID", "Flow_File_BF": "qbf",
                                    "Build_Representative_Cross_Section": True,
                                    "Representative_Cross_Section_File": "representative.csv"})

    assert configs.use_representative_baseflow_bathymetry
    assert pipeline.read_flows(configs).to_dict() == {"qbf": {11: 2.5}}


def test_the_manning_s_n_table_must_have_the_water_class(tmp_path: Path) -> None:
    table = tmp_path / "mannings.txt"
    table.write_text("LC_ID\tDescription\tManning_n\n10\tTrees\t0.12\n", encoding="utf-8")
    configs = Configs.from_mapping({"LU_Manning_n": str(table)})

    with pytest.raises(ValueError, match="no entry for the configured water class"):
        pipeline.mannings_n_raster(configs, np.full((2, 2), 10, dtype=np.uint8))
    n, water_n = pipeline.mannings_n_raster(Configs.from_mapping({"LU_Manning_n": str(table), "LC_Water_Value": 10}),
                                            np.full((2, 2), 10, dtype=np.uint8))
    assert water_n == pytest.approx(0.12) and n == pytest.approx(np.full((2, 2), 0.12))


def test_the_command_line_writes_the_outputs(tmp_path: Path) -> None:
    import yaml

    inputs = write_channel(tmp_path)
    config = tmp_path / "ARC_Input_File.yaml"
    config.write_text(yaml.safe_dump(inputs), encoding="utf-8")

    results = pipeline.main([str(config), "--quiet"])

    assert (tmp_path / "vdt.csv").exists()
    used = used_cells(results, inputs)
    assert used.sum() == results.cells.count
    assert used[CENTER_ROW, 20:220].all()


def test_inputs_on_different_grids_are_refused(tmp_path: Path) -> None:
    inputs = write_channel(tmp_path)
    dem = Raster(inputs["DEM_File"])
    shifted = list(dem.geotransform)
    shifted[0] += 5.0
    ds = gdal.Open(inputs["LU_Raster_SameRes"], gdal.GA_Update)
    ds.SetGeoTransform(shifted)
    ds = None

    with pytest.raises(ValueError, match="isn't on the DEM's grid"):
        pipeline.run(Configs.from_mapping(inputs), quiet=True, write=False)


def test_cells_without_dem_data_are_walls(tmp_path: Path) -> None:
    """A strip of nodata beyond the right bank holds the water in, like the raster's edge."""
    inputs = write_channel(tmp_path)
    ds = gdal.Open(inputs["DEM_File"], gdal.GA_Update)
    band = ds.GetRasterBand(1)
    values = band.ReadAsArray()
    values[CENTER_ROW + 12:CENTER_ROW + 15] = -9999.0
    band.WriteArray(values)
    band.SetNoDataValue(-9999.0)
    ds = None

    results = pipeline.run(Configs.from_mapping(inputs), quiet=True, write=False)
    section = results.sections[100]
    strip = (section.rows >= CENTER_ROW + 12) & (section.rows < CENTER_ROW + 15)

    assert strip.sum() == 3 and np.all(section.xs.elevations[strip] == 9999.0)
    assert results.curves.increments[100, -1, Q] == pytest.approx(Q_MAX[12], rel=1e-9)
    # Water 3 m deep stops at the strip on its side, 12 m out, and on the other reaches the section's end, 30 m out
    from arc.hydraulics import top_widths
    center = section.xs.elevations.size // 2
    sides = top_widths(section.xs.elevations, 1.0, section.xs.elevations[center] + 3.0)
    assert sorted(sides) == pytest.approx([11.0, 30.0], abs=0.001)


def test_stream_cells_are_water_for_the_land_cover_banks(tmp_path: Path) -> None:
    """Every stream cell is water in the land cover. With water in a band 2 cells either side of the stream but not
    at the stream cells themselves, the land cover's water has an edge, 2.5 m out, only because of that. Without
    Bathy_Use_Banks it stops the DEM's water (the channel's flat bed, 5 m either side) running more than two cells
    past it, as across a floodplain at the water's level, so the banks are there."""
    land = np.ones((ROWS, COLS), dtype=np.uint8)
    land[CENTER_ROW - 2:CENTER_ROW + 3] = 80
    land[CENTER_ROW] = 1
    inputs = {**write_channel(tmp_path, land_cover=land), "FindBanksBasedOnLandCover": True,
              "BATHY_Out_File": str(tmp_path / "bathy.tif")}
    results = pipeline.run(Configs.from_mapping(inputs), quiet=True, write=False)

    for k in np.flatnonzero(interior(results)):
        banks = results.sections[k].banks
        assert banks.method == "water_edge" and (banks.left, banks.right) == pytest.approx((2.5, 2.5))


def write_water_mask(inputs: dict, water: np.ndarray) -> dict:
    """A water mask (BathyWaterMask) on the synthetic channel's grid, and the inputs with it."""
    srs, geotransform = _projected()
    path = Path(inputs["DEM_File"]).with_name("water_mask.tif")
    _write_raster(path, water.astype(np.uint8), geotransform, srs, gdal.GDT_Byte)
    return {**inputs, "BathyWaterMask": str(path)}


def water_band(cells: int) -> np.ndarray:
    """A water mask of a band of water this many cells either side of the stream, but not at the stream cells."""
    water = np.zeros((ROWS, COLS), dtype=np.uint8)
    water[CENTER_ROW - cells:CENTER_ROW + cells + 1] = 1
    water[CENTER_ROW] = 0
    return water


@pytest.mark.parametrize("mask", [False, True])
@pytest.mark.parametrize("land_cover_banks", [False, True])
def test_without_bathymetry_the_banks_are_the_bank_tops_mask_or_not(tmp_path: Path, mask: bool,
                                                                    land_cover_banks: bool) -> None:
    """Without bathymetry the banks divide the channel from its overbanks, which the divided-channel method does at the
    bank tops: 9 m out, with a water mask or without, with FindBanksBasedOnLandCover or without. Not where the water
    mask's water ends, 3.5 m out, nor the land cover's, 5.5 m out, partway down the bank faces. The maximum flows are
    above bankfull, as a 100-year flood is (for below it, see
    test_no_bank_is_above_the_water_that_carries_the_maximum_flow)."""
    land = np.ones((ROWS, COLS), dtype=np.uint8)
    land[CENTER_ROW - 5:CENTER_ROW + 6] = 80
    inputs = write_channel(tmp_path, land_cover=land)
    if mask:
        inputs = write_water_mask(inputs, water_band(3))
    inputs = with_max_flows_above_bankfull({**inputs, "FindBanksBasedOnLandCover": land_cover_banks})
    results = pipeline.run(Configs.from_mapping(inputs), quiet=True, write=False)

    for k in np.flatnonzero(interior(results)):
        section = results.sections[k]
        assert section.banks.method == "width_to_depth_ratio"
        assert (section.banks.left, section.banks.right) == pytest.approx((9.0, 9.0))
        assert section.hydraulic_banks == section.banks


def test_without_bank_elevations_a_mask_wider_than_the_dem_s_water_doesn_t_widen_the_channel(tmp_path: Path) -> None:
    """Without Bathy_Use_Banks the DEM's water is the channel: where the mask's water widens to 8 cells either side for
    three columns, past the DEM's bed (5 cells either side, its "water"), the banks stay at the DEM's water's edge, half
    way to its first land cell, 11 m apart, and the bank faces the mask takes in aren't carved. The hydraulics divide
    at the bank tops, 18 m apart, mask or not (the maximum flows above bankfull, so that they aren't held below)."""
    wide_cols = range(95, 98)
    water = water_band(3)
    water[CENTER_ROW - 8:CENTER_ROW + 9, wide_cols.start:wide_cols.stop] = 1
    water[CENTER_ROW] = 0
    inputs = {**write_water_mask(write_channel(tmp_path), water), "BATHY_Out_File": str(tmp_path / "bathy.tif")}
    results = pipeline.run(Configs.from_mapping(with_max_flows_above_bankfull(inputs)), quiet=True, write=False)

    for k in np.flatnonzero(interior(results)):
        section = results.sections[k]
        assert section.banks.method == "water_edge" and section.banks.top_width == pytest.approx(11.0)
        assert section.carve_banks == section.banks
        assert section.hydraulic_banks.method == "width_to_depth_ratio"
        assert section.hydraulic_banks.top_width == pytest.approx(18.0)


def test_without_bank_elevations_the_bank_faces_above_the_water_are_channel(tmp_path: Path) -> None:
    """The channel is carved below the DEM's water only, between its edges 5.5 m out, but the hydraulics divide the
    carved cross section at the bank tops, 9 m out, and the bank faces between take the water's n, not the land's: they
    hold the channel's water above the low flow, not the floodplain's."""
    inputs = {**write_channel(tmp_path, water_n=0.03, land_n=0.06), "BATHY_Out_File": str(tmp_path / "bathy.tif")}
    results = pipeline.run(Configs.from_mapping(with_max_flows_above_bankfull(inputs)), quiet=True, write=False)
    dem = Raster(inputs["DEM_File"]).read_array()

    carved_rows = np.flatnonzero((np.isfinite(results.bathymetry) & (results.bathymetry < dem - 1e-9)).any(axis=1))
    assert np.abs(carved_rows - CENTER_ROW).max() <= 5
    for k in np.flatnonzero(interior(results)):
        section = results.sections[k]
        assert section.carve_banks.top_width == pytest.approx(11.0)
        assert (section.xs.left_bank_distance, section.xs.right_bank_distance) == pytest.approx((9.0, 9.0))
        offsets = np.abs(np.arange(section.xs.elevations.size) - section.xs.elevations.size // 2)
        assert np.all(section.xs.mannings_n[offsets <= 9] == np.float32(0.03))
        assert np.all(section.xs.mannings_n[(offsets > 9) & (section.xs.mannings_n < 1.0)] == np.float32(0.06))


def test_the_width_filter_keeps_the_dem_s_water_s_edge(tmp_path: Path) -> None:
    """Without Bathy_Use_Banks, where the channel's bed (its water, on this DEM) widens to 20 m for three columns, those
    cross sections keep their own banks, 21 m apart, though the reach's others are 11 m: the water shows the channel's
    width, and the reach's median width would carve the bank faces beside it."""
    results = pipeline.run(Configs.from_mapping(write_widened_channel(tmp_path)), quiet=True, write=False)

    for k in np.flatnonzero(np.isin(results.cells.cols, WIDE_COLS)):
        section = results.sections[k]
        assert section.carve_banks == section.banks
        assert section.banks.method == "water_edge" and section.banks.top_width == pytest.approx(21.0)
    others = np.flatnonzero(interior(results) & ~np.isin(results.cells.cols, WIDE_COLS))
    assert all(results.sections[k].carve_banks.top_width == pytest.approx(11.0) for k in others)


def test_a_water_mask_off_the_dem_s_grid_is_refused(tmp_path: Path) -> None:
    inputs = write_water_mask(write_channel(tmp_path), water_band(3))
    ds = gdal.Open(inputs["BathyWaterMask"], gdal.GA_Update)
    shifted = list(ds.GetGeoTransform())
    shifted[0] += 5.0
    ds.SetGeoTransform(shifted)
    ds = None

    with pytest.raises(ValueError, match="water mask raster .* isn't on the DEM's grid"):
        pipeline.run(Configs.from_mapping(inputs), quiet=True, write=False)


def write_oblique_channel(folder: Path, degrees: float = 30.0, roughness: float = 0.0, **kwargs) -> dict:
    """The synthetic trapezoid along a line turned `degrees` from the rows, as one reach of 150 columns, with random
    bumps up to `roughness` metres high."""
    shape = (140, 180)
    theta = math.radians(degrees)
    first_row, first_col = 25, 20
    rows, cols = np.mgrid[0:shape[0], 0:shape[1]]
    along = (cols - first_col) * math.cos(theta) + (rows - first_row) * math.sin(theta)
    across = (rows - first_row) * math.cos(theta) - (cols - first_col) * math.sin(theta)
    dem = BED - SLOPE * along + ground(np.abs(across))
    dem += np.random.default_rng(12).uniform(0.0, roughness, shape)
    streams = np.zeros(shape, dtype=np.int32)
    for col in range(first_col, first_col + 150):
        streams[first_row + int(round((col - first_col) * math.tan(theta))), col] = 21
    end = (first_row + int(round(149 * math.tan(theta))), first_col + 149)
    reaches = {21: {"start": (first_row, first_col), "end": end, "downstream": -1, "baseflow": 2.0, "q_max": 15.0}}
    srs, geotransform = _projected()
    return write_inputs(folder, dem, streams, srs, geotransform, reaches, **kwargs)


def test_the_water_s_roughness_between_banks_reaches_other_cross_sections(tmp_path: Path) -> None:
    """The water's n goes in the roughness raster at every cell between each cross section's banks, so every cross
    section crossing those cells takes it too. On an oblique channel the ordinates just beyond a cross
    section's banks are sampled partly from cells of its neighbours' channels, and are smoother than the land. The
    ordinates between its own banks are the water's exactly."""
    from arc.bathymetry import in_bank

    inputs = write_oblique_channel(tmp_path, land_n=0.1)
    results = pipeline.run(Configs.from_mapping(inputs), quiet=True, write=False)
    water = float(np.float32(N))
    blended = 0
    for k in range(20, results.cells.count - 20):
        section = results.sections[k]
        xs = section.xs
        inside = in_bank(xs, section.banks)
        assert section.banks.valid and np.all(xs.mannings_n[inside] == water)
        offsets = np.abs(np.arange(xs.elevations.size) - xs.elevations.size // 2) * xs.ordinate_distance
        near = ~inside & (offsets <= max(section.banks.left, section.banks.right) + 1.5 * xs.ordinate_distance)
        blended += int(np.count_nonzero(xs.mannings_n[near] < 0.1 - 1e-9))
        assert np.all(xs.mannings_n[near] >= water - 1e-12)
    assert blended > 50


def test_the_cross_section_file_s_side_1_runs_towards_the_higher_rows(tmp_path: Path) -> None:
    """With banks twice as steep south of the stream, side 1 is the south side, the higher rows, out to the
    raster's edge."""
    steep = lambda d: np.where(d <= 5.0, 0.0, np.where(d <= 7.0, d - 5.0, 2.0 + 0.01 * (d - 7.0)))
    inputs = {**write_channel(tmp_path, south=steep), "XS_Out_File": str(tmp_path / "xs.txt")}
    pipeline.run(Configs.from_mapping(inputs), quiet=True)
    parse = lambda text: np.array(text.strip("[]").split(), dtype=float)
    xs = pd.read_csv(tmp_path / "xs.txt", sep="\t")

    assert len(xs) == 200
    distance = np.arange(CENTER_ROW + 1, dtype=np.float64)
    for _, row in xs.iterrows():
        xs1, xs2 = parse(row["XS1_Profile"]), parse(row["XS2_Profile"])
        bed = BED - SLOPE * row["Col"]
        assert np.allclose(xs1, bed + steep(distance), atol=2e-6)
        assert np.allclose(xs2, bed + ground(distance), atol=2e-6)
        assert (row["r1"], row["c1"], row["r2"], row["c2"]) == (ROWS - 1, row["Col"], 0, row["Col"])


def test_the_cross_section_file_can_be_parquet(tmp_path: Path) -> None:
    """A .parquet XS_Out_File holds each cross section's profiles and n as the arrays the run used."""
    inputs = {**write_channel(tmp_path), "XS_Out_File": str(tmp_path / "xs.parquet")}
    results = pipeline.run(Configs.from_mapping(inputs), quiet=True)
    xs = pd.read_parquet(tmp_path / "xs.parquet")

    assert len(xs) == len(results.cross_sections) == results.cells.count
    for record, (_, row) in zip(results.cross_sections, xs.iterrows()):
        assert (row["COMID"], row["Row"], row["Col"], row["r1"], row["c2"]) == \
            (record.comid, record.row, record.col, record.r1, record.c2)
        for column, values in (("XS1_Profile", record.xs1_profile), ("Manning_N_Raster1", record.manning_n_raster1),
                               ("XS2_Profile", record.xs2_profile), ("Manning_N_Raster2", record.manning_n_raster2)):
            assert np.array_equal(row[column], values)


def test_without_bank_elevations_the_bathymetry_never_rises_above_the_dem(tmp_path: Path) -> None:
    """On rough ground along an oblique channel the cross sections' elevations are interpolated, so carving them can
    leave a cell above its own ground; the gap fill drops those cells, and doesn't fill above the ground either."""
    inputs = {**write_oblique_channel(tmp_path, roughness=0.3), "BATHY_Out_File": str(tmp_path / "bathy.tif")}
    results = pipeline.run(Configs.from_mapping(inputs), quiet=True, write=False)
    dem = Raster(inputs["DEM_File"]).read_array()
    carved = np.isfinite(results.bathymetry)

    assert carved.sum() > 1000
    assert np.all(results.bathymetry[carved] <= dem[carved].astype(np.float32))  # the raster's float32


WIDE_COLS = range(95, 98)


def with_max_flows_above_bankfull(inputs: dict, reaches=(11, 12), factor: float = 3.0) -> dict:
    """The inputs with these reaches' maximum flow `factor` times the synthetic channel's bankfull flow (its 2 m banks
    full), as a flow file's maximum flow, a 100-year flood, is above bankfull: with Bathy_Use_Banks no bank top is
    then above the water that carries it, where it would be taken for a valley wall."""
    flows = pd.read_csv(inputs["Flow_File"])
    flows.loc[flows["COMID"].isin(reaches), "qmax"] = factor * trapezoid_discharge(2.0)
    flows.to_csv(inputs["Flow_File"], index=False)
    return inputs


def write_widened_channel(tmp_path: Path) -> dict:
    """The synthetic channel with bathymetry, its bed widened to 20 m for three columns."""
    inputs = write_channel(tmp_path)
    dem = Raster(inputs["DEM_File"]).read_array()
    rows = np.arange(ROWS)
    for col in WIDE_COLS:
        distance = np.abs(rows - CENTER_ROW) * 1.0
        dem[:, col] = BED - SLOPE * col + np.where(distance <= 10.0, 0.0, np.where(
            distance <= 14.0, (distance - 10.0) / 2.0, 2.0 + 0.01 * (distance - 14.0)))
    srs, geotransform = _projected()
    _write_raster(Path(inputs["DEM_File"]), dem, geotransform, srs, gdal.GDT_Float64)
    return {**inputs, "BATHY_Out_File": str(tmp_path / "bathy.tif")}


def test_a_cross_section_s_outlying_width_is_rebuilt_at_the_reach_median(tmp_path: Path) -> None:
    """With Bathy_Use_Banks, where the channel widens to a 20 m bed for three columns, those cross sections' bank tops
    are outliers of their reach's widths, so the bathymetry and the hydraulics use banks at the reach's median width
    instead."""
    inputs = {**with_max_flows_above_bankfull(write_widened_channel(tmp_path)), "Bathy_Use_Banks": True}
    results = pipeline.run(Configs.from_mapping(inputs), quiet=True, write=False)

    for k in np.flatnonzero(np.isin(results.cells.cols, WIDE_COLS)):
        section = results.sections[k]
        assert section.banks.top_width == pytest.approx(28.0, abs=1e-6)  # its own banks, at its bank tops
        assert section.hydraulic_banks.method == "target_width"
        assert section.hydraulic_banks.top_width == pytest.approx(18.0, abs=1e-6)
        assert (section.xs.left_bank_distance, section.xs.right_bank_distance) == pytest.approx((9.0, 9.0))


def test_the_water_s_roughness_goes_between_the_banks_the_width_filter_gives(tmp_path: Path) -> None:
    """Where the channel narrows to a 2 m bed for three columns, under a rougher land cover, its banks (±5 m) are
    outliers of the reach's widths and are rebuilt at the median (±9 m). The water's n, which the channel's depth is
    solved with, goes between those, not only between the banks found. (With Bathy_Use_Banks, whose bank tops the
    width filter rebuilds; without it, the DEM's water's edge is kept.)"""
    inputs = {**with_max_flows_above_bankfull(write_channel(tmp_path, land_n=0.1)),
              "BATHY_Out_File": str(tmp_path / "bathy.tif"), "Bathy_Use_Banks": True}
    dem = Raster(inputs["DEM_File"]).read_array()
    distance = np.abs(np.arange(ROWS) - CENTER_ROW) * 1.0
    for col in WIDE_COLS:
        dem[:, col] = BED - SLOPE * col + np.where(distance <= 1.0, 0.0, np.where(
            distance <= 5.0, (distance - 1.0) / 2.0, 2.0 + 0.01 * (distance - 5.0)))
    srs, geotransform = _projected()
    _write_raster(Path(inputs["DEM_File"]), dem, geotransform, srs, gdal.GDT_Float64)
    results = pipeline.run(Configs.from_mapping(inputs), quiet=True, write=False)

    for k in np.flatnonzero(np.isin(results.cells.cols, WIDE_COLS)):
        section = results.sections[k]
        offsets = np.arange(section.xs.elevations.size) - section.xs.elevations.size // 2
        on_raster = section.xs.elevations < pipeline.OFF_RASTER_ELEVATION
        assert section.banks.top_width < 12.0 and section.hydraulic_banks.top_width == pytest.approx(18.0, abs=1e-6)
        assert np.all(section.xs.mannings_n[np.abs(offsets) <= 9] == np.float32(N))
        assert np.all(section.xs.mannings_n[(np.abs(offsets) >= 10) & on_raster] == np.float32(0.1))
        inside = np.abs(section.xs.profile.stations) <= 9.0
        assert np.all(section.xs.profile.mannings_n[inside] == np.float32(N))


def test_a_bank_s_ceiling_is_the_water_that_carries_the_maximum_flow_over_the_dem(tmp_path: Path) -> None:
    """Over a V rising 0.5 m a metre, the ceiling carries the cell's maximum flow, by the rating curves' roughness,
    on the cross section as the DEM has it. Banks from the water mask, banks that aren't valid, and cells without a
    maximum flow have none."""
    offsets = np.arange(-50, 51, dtype=np.float64)

    def section(method: str, valid: bool = True) -> pipeline.Section:
        xs = XSection(BED + 0.5 * np.abs(offsets), np.full(offsets.size, N), 1.0)
        cells = np.zeros(offsets.size, dtype=np.int64)
        result = pipeline.Section(xs, 0, 0, 0.0, 0.0, cells, cells, 1, BED)
        result.banks = Banks(method, 5.0, 5.0, BED + 2.5, BED + 2.5, False, valid)
        return result

    sections = [section("width_to_depth_ratio"), section("land_cover"), section("flat_water", valid=False),
                section("target_width"), section("width_to_depth_ratio")]
    max_flow, slopes = np.array([10.0, 10.0, 10.0, 0.0, 40.0]), np.full(5, SLOPE)
    roughness = DepthRoughness(6.0, 2.0, 1.0)

    ceilings = pipeline.bank_ceilings(sections, range(5), max_flow, slopes, roughness, 1.0)

    assert np.isnan(ceilings[[1, 2, 3]]).all()
    for k in (0, 4):
        assert BED < ceilings[k] < BED + 25.0
        assert discharge(sections[k].xs, SLOPE, wse=ceilings[k], roughness=roughness) == pytest.approx(max_flow[k],
                                                                                                     rel=1e-6)
    assert ceilings[4] > ceilings[0]


def test_without_bank_smoothing_an_outlying_width_keeps_its_own_banks(tmp_path: Path) -> None:
    inputs = {**with_max_flows_above_bankfull(write_widened_channel(tmp_path)), "Bathy_Smooth_Banks": False,
              "Bathy_Use_Banks": True}
    results = pipeline.run(Configs.from_mapping(inputs), quiet=True, write=False)

    for k in np.flatnonzero(np.isin(results.cells.cols, WIDE_COLS)):
        section = results.sections[k]
        assert section.hydraulic_banks == section.banks
        assert section.banks.top_width == pytest.approx(28.0, abs=1e-6)
        assert (section.xs.left_bank_distance, section.xs.right_bank_distance) == pytest.approx((14.0, 14.0))


def write_valley(folder: Path, side, *, q_max: float = 2.0, **power_laws) -> dict:
    """A stream along a row of 10 m cells, falling 0.001 per metre, whose ground rises side(distance) metres either
    side, and power laws of the drainage area with these coefficients and exponents of 0: a constant width and
    depth."""
    srs, _ = _projected()
    geotransform = (500000.0, 10.0, 0.0, 3800000.0, 0.0, -10.0)
    rows, cols = np.mgrid[0:ROWS, 0:COLS]
    dem = BED - SLOPE * cols * 10.0 + side(np.abs(rows - CENTER_ROW) * 10.0)
    streams = np.zeros((ROWS, COLS), dtype=np.int32)
    streams[CENTER_ROW, 20:120] = 11
    streams[CENTER_ROW, 120:220] = 12
    reaches = {11: {"start": (CENTER_ROW, 20), "end": (CENTER_ROW, 119), "downstream": 12, "baseflow": 0.5,
                    "q_max": q_max},
               12: {"start": (CENTER_ROW, 120), "end": (CENTER_ROW, 219), "downstream": -1, "baseflow": 0.5,
                    "q_max": q_max}}
    inputs = {**write_inputs(folder, dem, streams, srs, geotransform, reaches),
              "BATHY_Out_File": str(folder / "bathy.tif"), "Bathy_Bed_Cap": False, "drainage_area_field": "DA"}
    for name, value in power_laws.items():
        inputs.update({f"coefficient_{name}": value, f"exponent_{name}": 0.0})
    return inputs


def test_a_creek_narrower_than_a_cell_is_carved_and_rated_at_its_own_width(tmp_path: Path) -> None:
    """A stream cell 1 m below its neighbours, in a valley whose sides rise 2 m a cell, with a width prior of 4 m in
    its 10 m cells and a depth prior of 1.5 m, and bank elevations. The DEM can't resolve the creek (its width-to-depth
    ratio never rises, and its flat water is 2 m across), so it's a single-cell channel, 4 m wide: the raster gets its
    bed at the stream cell and nothing beside it, and the rating curve, within the channel, is Manning's for the
    4 m trapezoid, where the ordinates alone would have made it a V 20 m across."""
    inputs = write_valley(tmp_path, lambda d: np.where(d == 0.0, 0.0, 1.0 + 0.2 * (d - 10.0)), width=4.0,
                          depth=1.5)
    results = pipeline.run(Configs.from_mapping({**inputs, "Bathy_Use_Banks": True}), quiet=True, write=False)
    carved = np.isfinite(results.bathymetry)

    assert carved[CENTER_ROW, 40:200].all() and carved[:CENTER_ROW].sum() == carved[CENTER_ROW + 1:].sum() == 0
    for k in np.flatnonzero(interior(results)):
        section = results.sections[k]
        center = section.xs.elevations.size // 2
        banks = section.hydraulic_banks
        assert banks.single_cell and (banks.left, banks.right) == (2.0, 2.0)
        assert banks.left_elevation == section.dem_low_point + 1.0  # the ordinate beside the stream cell
        bed = float(section.xs.elevations[center])
        depth = section.bank_elevation - bed
        assert depth == pytest.approx(1.5, abs=0.05)  # as the bed smoothing left it
        assert results.bathymetry[section.row, section.col] == pytest.approx(bed, abs=1e-4)
        assert section.xs.elevations[[center - 1, center + 1]].tolist() == [section.dem_low_point + 1.0] * 2
        for q, _, t, wse, p in results.curves.increments[k]:
            y = wse - bed
            side = 0.8 * y / depth  # each side slopes over 0.2 of the 4 m width, whatever the depth
            area = y * (2.4 + side)
            perimeter = 2.4 + 2 * math.hypot(side, y)
            assert y <= depth
            assert t == pytest.approx(2.4 + 2 * side, rel=1e-9)
            assert p == pytest.approx(perimeter, rel=1e-9)
            assert q == pytest.approx(area * (area / perimeter) ** (2 / 3) * math.sqrt(SLOPE) / N_RASTER, rel=1e-9)


def test_the_dem_s_banks_win_over_a_width_prior(tmp_path: Path) -> None:
    """The same creek in floodplains rising 1%: the width-to-depth ratio is lowest with the water at the neighbours'
    ground, 20 m across, and those banks win over the 4 m prior. The prior sizes only a channel the DEM doesn't
    resolve."""
    inputs = write_valley(tmp_path, lambda d: np.where(d == 0.0, 0.0, 1.0 + 0.01 * (d - 10.0)), q_max=50.0,
                          width=4.0, depth=1.5)
    results = pipeline.run(Configs.from_mapping({**inputs, "Bathy_Use_Banks": True}), quiet=True, write=False)

    for k in np.flatnonzero(interior(results)):
        banks = results.sections[k].hydraulic_banks
        assert banks.method == "width_to_depth_ratio" and banks.top_width == pytest.approx(20.0)


@pytest.mark.parametrize("bathymetry", [True, False], ids=["bank-elevations", "no-bathymetry"])
def test_no_bank_is_above_the_water_that_carries_the_maximum_flow(tmp_path: Path, bathymetry: bool) -> None:
    """The synthetic channel's maximum flows, 15 and 18 m^3/s, stand about a metre deep in it, below its 2 m bank
    tops, which by the maximum flow's reckoning are as much a valley wall as a bank: the banks are never above that
    water. With Bathy_Use_Banks they are the water's edges at the highest stage below it; without bathymetry, its own
    edges, so that the hydraulics don't divide it, and the rating curve is Manning's for the whole trapezoid. Either
    way the search found no bank top below that water (max_flow_water)."""
    inputs = write_channel(tmp_path)
    if bathymetry:
        inputs = {**inputs, "BATHY_Out_File": str(tmp_path / "bathy.tif"), "Bathy_Use_Banks": True}
    results = pipeline.run(Configs.from_mapping(inputs), quiet=True, write=False)

    for k in np.flatnonzero(interior(results)):
        section = results.sections[k]
        height = min(section.banks.left_elevation, section.banks.right_elevation) - section.dem_low_point
        top = normal_depth(Q_MAX[int(results.cells.comids[k])])
        assert section.banks.method == "max_flow_water"
        if bathymetry:
            assert 0.5 < height <= top + 0.05
        else:
            assert height == pytest.approx(top, abs=0.05)


@pytest.mark.parametrize("width", [30.0, None])
def test_a_channel_nothing_resolves_is_carved_at_its_width_prior_or_one_cell_wide(tmp_path: Path,
                                                                                    width: float | None) -> None:
    """A V rising 0.5 m a cell: its width-to-depth ratio never rises, and its flat water is 2 m out, too narrow to
    resolve, so no cross section in the reach has valid banks. A width prior wider than two cells doesn't make it a
    single cell either. The channel is carved at the prior's width, or without one, one cell wide. The hydraulics
    divide it no narrower: at the edges of the maximum flow's water, which the V holds, where those are wider."""
    inputs = write_valley(tmp_path, lambda d: 0.05 * d, **({} if width is None else {"width": width}))
    results = pipeline.run(Configs.from_mapping(inputs), quiet=True, write=False)

    for k in np.flatnonzero(interior(results)):
        section = results.sections[k]
        assert not section.banks.valid
        banks = section.carve_banks
        assert section.hydraulic_banks.top_width >= banks.top_width
        if width is None:
            assert banks.single_cell and banks.top_width == 10.0
        else:
            assert banks.method == "target_width" and banks.top_width == width
        offsets = (np.arange(section.xs.elevations.size) - section.xs.elevations.size // 2) * 10.0
        carved = section.xs.elevations < section.dem_low_point + 0.05 * np.abs(offsets) - 1e-9
        assert np.flatnonzero(carved).tolist() == np.flatnonzero(np.abs(offsets) < banks.top_width / 2).tolist()


def write_channel_and_plateau(folder: Path) -> dict:
    """The synthetic channel, and south of it a flat plateau with a reach of its own that joins nothing, the plateau's
    first row without data, so that neither's cross sections reach into the other's."""
    channel = write_channel(folder / "channel")
    plateau = np.full((ROWS, COLS), BED + 10.0)
    plateau[0] = np.nan
    dem = np.vstack([Raster(channel["DEM_File"]).read_array(), plateau])
    streams = np.vstack([Raster(channel["Stream_File"]).read_array(), np.zeros((ROWS, COLS), dtype=np.int32)])
    streams[ROWS + CENTER_ROW, 20:100] = 31
    reaches = {11: {"start": (CENTER_ROW, 20), "end": (CENTER_ROW, 119), "downstream": 12, "baseflow": BASEFLOW[11],
                    "q_max": Q_MAX[11]},
               12: {"start": (CENTER_ROW, 120), "end": (CENTER_ROW, 219), "downstream": -1, "baseflow": BASEFLOW[12],
                    "q_max": Q_MAX[12]},
               31: {"start": (ROWS + CENTER_ROW, 20), "end": (ROWS + CENTER_ROW, 99), "downstream": -1,
                    "baseflow": 2.0, "q_max": 15.0}}
    srs, geotransform = _projected()
    return write_inputs(folder, dem, streams, srs, geotransform, reaches)


@pytest.mark.parametrize("use_banks", [True, False])
def test_a_reach_without_banks_of_its_own_takes_the_site_s_bank_height(tmp_path: Path, use_banks: bool) -> None:
    """On the plateau no bank stands above the stream, so its reach takes the site's bank height, from the channel's
    reaches, and stays in the bathymetry and the rating curves. With bank elevations its channel, which has no banks
    to give it a width, is carved a cell wide up to that height, above the plateau, so its water spills from the
    channel at its brim, and the rating curves fail their check on the discharge there. (Its baseflow is below
    Bathy_Bankless_Max_Baseflow's default, which would carve it to the plateau instead: off here.)"""
    inputs = {**with_max_flows_above_bankfull(write_channel_and_plateau(tmp_path)),
              "BATHY_Out_File": str(tmp_path / "bathy.tif"), "Bathy_Use_Banks": use_banks,
              "Bathy_Bankless_Max_Baseflow": 0}
    results = pipeline.run(Configs.from_mapping(inputs), quiet=True)
    plateau = np.flatnonzero(results.cells.comids == 31)
    used = used_cells(results, inputs)

    assert all(section.usable for section in results.sections if section is not None)
    heights = [results.sections[k].bank_elevation - (BED + 10.0) for k in plateau]
    np.testing.assert_allclose(heights, heights[0])
    assert heights[0] > 0.0
    assert np.isnan(results.curves.increments[plateau]).all() == use_banks
    assert used[ROWS + CENTER_ROW, 20:100].all()
    assert used[CENTER_ROW, 20:220].all()


def test_a_bankless_cross_section_on_a_small_baseflow_carries_it_at_its_stream_level(tmp_path: Path) -> None:
    """The plateau's cross sections find no banks, and its baseflow, 2 m³/s, is below Bathy_Bankless_Max_Baseflow (20
    by default): with bank elevations its channel carries the baseflow at the plateau, its stream's level, not at the
    site's bank height above it. The channel's cross sections whose banks were found keep their bank elevations. With
    the threshold at the baseflow itself, the plateau takes the site's bank height again."""
    inputs = {**write_channel_and_plateau(tmp_path), "BATHY_Out_File": str(tmp_path / "bathy.tif"),
              "Bathy_Use_Banks": True}
    default = pipeline.run(Configs.from_mapping(inputs), quiet=True, write=False)
    at_baseflow = pipeline.run(Configs.from_mapping({**inputs, "Bathy_Bankless_Max_Baseflow": 2.0}), quiet=True,
                               write=False)
    plateau = np.flatnonzero(default.cells.comids == 31)
    found = [k for k in np.flatnonzero(default.cells.comids != 31)
             if default.sections[k].banks.method not in ("none", "single_cell")]

    assert all(default.sections[k].banks.method == "none" for k in plateau)
    assert [default.sections[k].bank_elevation for k in plateau] == [BED + 10.0] * plateau.size
    assert all(at_baseflow.sections[k].bank_elevation > BED + 10.0 for k in plateau)
    assert len(found) > 100
    assert [default.sections[k].bank_elevation for k in found] == [at_baseflow.sections[k].bank_elevation
                                                                    for k in found]


@pytest.mark.parametrize("use_banks", [True, False])
def test_without_a_bank_anywhere_the_reach_is_dropped_with_bank_elevations(tmp_path: Path, use_banks: bool) -> None:
    """The plateau alone: no bank stands above the stream anywhere, so there's no bank height to take, and the reach
    is dropped from the bathymetry and the rating curves, but only with Bathy_Use_Banks, which needs it."""
    srs, geotransform = _projected()
    streams = np.zeros((ROWS, COLS), dtype=np.int32)
    streams[CENTER_ROW, 20:100] = 31
    reaches = {31: {"start": (CENTER_ROW, 20), "end": (CENTER_ROW, 99), "downstream": -1, "baseflow": 2.0,
                    "q_max": 15.0}}
    inputs = {**write_inputs(tmp_path, np.full((ROWS, COLS), BED), streams, srs, geotransform, reaches),
              "BATHY_Out_File": str(tmp_path / "bathy.tif"), "Bathy_Use_Banks": use_banks}
    results = pipeline.run(Configs.from_mapping(inputs), quiet=True)
    used = used_cells(results, inputs)

    assert all(section.usable != use_banks for section in results.sections if section is not None)
    assert np.isnan(results.curves.increments).all() == use_banks
    assert np.isnan(results.bathymetry).all() == use_banks
    assert (~used[CENTER_ROW, 20:100]).all() == use_banks


@pytest.mark.parametrize("method", ["local_average", "reach_average", "local_average_corrected", "end_points",
                                    "local_fit"])
def test_each_cell_gets_its_reach_s_slope_by_the_method(tmp_path: Path, method: str) -> None:
    """On a bed that rises and falls a little as it falls, each method's slopes, as arc.xsection.slope gives them. The
    local averages take in the other reach's cells too, its neighbour on the main stem."""
    from arc.xsection.slope import (
        corrected_local_slopes,
        local_average_slopes,
        local_fit_slopes,
        reach_median_slope,
    )
    from arc.xsection.stream_path import along_stream_stations

    bumpy = lambda x: BED - SLOPE * x + 0.05 * np.sin(x / 7.0)
    inputs = {**write_channel(tmp_path, bed=bumpy), "Stream_Slope_Method": method, "Slope_Low_Percentile": 20,
              "Slope_High_Percentile": 60}
    results = pipeline.run(Configs.from_mapping(inputs), quiet=True, write=False)
    dem = Raster(inputs["DEM_File"]).read_array()
    both_rows, both_cols = results.cells.rows, results.cells.cols  # reaches 11 and 12
    both_local = local_average_slopes(dem[both_rows, both_cols], both_rows, both_cols,
                                      along_stream_stations(both_rows, both_cols, 1.0, 1.0), 1.0, 1.0, 10)

    for reach in (11, 12):
        cells = results.cells.comids == reach
        rows, cols = results.cells.rows[cells], results.cells.cols[cells]
        stations = along_stream_stations(rows, cols, 1.0, 1.0)
        z = dem[rows, cols]
        local = both_local[cells]
        median, lower, upper = reach_median_slope(z, rows, cols, stations, 1.0, 1.0, 10, 20, 60)
        expected = {"local_average": local, "reach_average": np.full(rows.size, median),
                    "local_average_corrected": corrected_local_slopes(local, lower, upper),
                    "end_points": np.full(rows.size, max(round(abs(z[-1] - z[0]) / 99.0, 8), MIN_SLOPE)),
                    "local_fit": local_fit_slopes(z, stations, 10)[0]}[method]
        np.testing.assert_allclose(results.slopes[cells], expected, rtol=1e-12)
    distinct = len(set(np.round(results.slopes, 12)))
    assert distinct == 2 if method in ("reach_average", "end_points") else distinct > 50


@pytest.mark.parametrize("network", [True, False], ids=["network", "no network"])
def test_local_averages_reach_across_a_junction_only_along_the_stream_network(tmp_path: Path, network: bool) -> None:
    """The bed steepens from 1 to 5 m per km where reach 11 meets reach 12. Reach 11's cells are averaged over both
    reaches' cells with the stream network, so its last ones see the steeper bed, and over its own alone without."""
    from arc.xsection.slope import local_average_slopes
    from arc.xsection.stream_path import along_stream_stations

    steepening = lambda x: BED - SLOPE * x - 0.004 * np.maximum(x - 120.0, 0.0)
    inputs = {**write_channel(tmp_path, bed=steepening), "Stream_Slope_Method": "local_average"}
    if not network:
        inputs["downstream_reach_id"] = "no such field"
    results = pipeline.run(Configs.from_mapping(inputs), quiet=True, write=False)
    dem = Raster(inputs["DEM_File"]).read_array()

    cells = results.cells.comids == 11
    over = np.ones_like(cells) if network else cells
    rows, cols = results.cells.rows[over], results.cells.cols[over]
    local = local_average_slopes(dem[rows, cols], rows, cols, along_stream_stations(rows, cols, 1.0, 1.0), 1.0, 1.0,
                                 10)
    np.testing.assert_allclose(results.slopes[cells], local[cells[over]], rtol=1e-12)
    near = results.cells.cols[cells] >= 113
    assert (results.slopes[cells][near] > 1.01 * SLOPE).all() == network


SLOPE_METHODS =["local_average", "reach_average", "local_average_corrected", "end_points", "local_fit"]


@pytest.mark.parametrize("network", [True, False], ids=["network", "no network"])
@pytest.mark.parametrize("method", SLOPE_METHODS)
def test_a_uniform_reach_has_its_slope_by_every_method(tmp_path: Path, method: str, network: bool) -> None:
    """Every slope along the synthetic channel is the same, so its percentiles are equal: a slope like any other."""
    inputs = {**write_channel(tmp_path), "Stream_Slope_Method": method}
    if not network:
        inputs["downstream_reach_id"] = "no such field"

    results = pipeline.run(Configs.from_mapping(inputs), quiet=True, write=False)

    assert results.cells.count == 200
    np.testing.assert_allclose(results.slopes, SLOPE, rtol=1e-6)


def write_channel_with_a_one_cell_reach(folder: Path) -> dict:
    """The synthetic channel with reach 13 too: a single stream cell just above reach 11, flowing into it. No method
    can measure a slope on one cell, or along its line, which is a point."""
    inputs = write_channel(folder / "channel")
    dem = Raster(inputs["DEM_File"]).read_array()
    streams = Raster(inputs["Stream_File"]).read_array()
    streams[CENTER_ROW, 19] = 13
    reaches = {13: {"start": (CENTER_ROW, 19), "end": (CENTER_ROW, 19), "downstream": 11, "area": 40.0,
                    "baseflow": BASEFLOW[11], "q_max": Q_MAX[11]},
               11: {"start": (CENTER_ROW, 20), "end": (CENTER_ROW, 119), "downstream": 12, "area": 50.0,
                    "baseflow": BASEFLOW[11], "q_max": Q_MAX[11]},
               12: {"start": (CENTER_ROW, 120), "end": (CENTER_ROW, 219), "downstream": -1, "area": 60.0,
                    "baseflow": BASEFLOW[12], "q_max": Q_MAX[12]}}
    srs, geotransform = _projected()
    return write_inputs(folder, dem, streams, srs, geotransform, reaches)


@pytest.mark.parametrize("method", SLOPE_METHODS)
def test_a_reach_without_a_slope_takes_its_neighbour_s_by_every_method(tmp_path: Path, method: str) -> None:
    inputs = {**write_channel_with_a_one_cell_reach(tmp_path), "Stream_Slope_Method": method}

    results = pipeline.run(Configs.from_mapping(inputs), quiet=True, write=False)

    one_cell = np.flatnonzero(results.cells.comids == 13)
    assert one_cell.size == 1 and results.cells.cols[one_cell[0]] == 19
    reach_11 = results.slopes[results.cells.comids == 11]
    assert results.slopes[one_cell[0]] == pytest.approx(np.median(reach_11), rel=1e-12)
    assert results.slopes[one_cell[0]] == pytest.approx(SLOPE, rel=1e-6)


@pytest.mark.parametrize("method", SLOPE_METHODS)
def test_a_reach_s_slope_doesn_t_depend_on_which_others_the_run_rates(tmp_path: Path, method: str) -> None:
    """Slopes are measured for the run's reaches, and for the others on the stream raster only when one of the run's
    has none: reach 13 alone still takes reach 11's slope, and reach 11 alone gets the slopes it gets with the
    others."""
    inputs = {**write_channel_with_a_one_cell_reach(tmp_path), "Stream_Slope_Method": method}
    every = pipeline.run(Configs.from_mapping(inputs), quiet=True, write=False)
    flows = pd.read_csv(inputs["Flow_File"])
    for reach in (13, 11):
        alone = tmp_path / f"flows_{reach}.csv"
        flows[flows["COMID"] == reach].to_csv(alone, index=False)

        results = pipeline.run(Configs.from_mapping({**inputs, "Flow_File": str(alone)}), quiet=True, write=False)

        assert set(results.cells.comids.tolist()) == {reach}
        np.testing.assert_array_equal(results.slopes, every.slopes[every.cells.comids == reach])


@pytest.mark.parametrize("method", SLOPE_METHODS)
def test_without_a_network_a_reach_without_a_slope_is_skipped(tmp_path: Path, method: str) -> None:
    inputs = {**write_channel_with_a_one_cell_reach(tmp_path), "Stream_Slope_Method": method,
              "downstream_reach_id": "no such field"}

    results = pipeline.run(Configs.from_mapping(inputs), quiet=True, write=False)

    assert 13 not in results.cells.comids and results.cells.count == 200


def test_a_cell_off_its_reach_s_others_takes_the_reach_s_slope(tmp_path: Path) -> None:
    """With Gen_Slope_Dist 1, reach 11's cell at column 18 has no other cell of its reach within a cell of it (column
    19 is reach 13's, and 20 is two away), so it takes reach 11's median slope."""
    inputs = write_channel_with_a_one_cell_reach(tmp_path)
    streams = Raster(inputs["Stream_File"]).read_array()
    streams[CENTER_ROW, 18] = 11
    _write_raster(Path(inputs["Stream_File"]), streams.astype(np.int32), *_projected()[::-1], gdal.GDT_Int32)
    inputs = {**inputs, "Stream_Slope_Method": "local_average", "Gen_Slope_Dist": 1}

    results = pipeline.run(Configs.from_mapping(inputs), quiet=True, write=False)

    k = np.flatnonzero((results.cells.comids == 11) & (results.cells.cols == 18))
    others = results.slopes[(results.cells.comids == 11) & (results.cells.cols != 18)]
    assert k.size == 1 and results.slopes[k[0]] == pytest.approx(np.median(others), rel=1e-12)


def write_manual_sections(folder: Path, inputs: dict, cols: list[int]) -> dict:
    """Manual cross sections, in the manual cross-section file's format, that are the synthetic channel's own along
    columns, one per ID from 1001, each on its reach's stream; and inputs using them."""
    dem = Raster(inputs["DEM_File"]).read_array()
    records = []
    for k, col in enumerate(cols):
        south, north = np.arange(CENTER_ROW, ROWS), np.arange(CENTER_ROW, -1, -1)
        records.append({"COMID": 1001 + k, "Row": CENTER_ROW, "Col": col, "Ordinate_Dist": 1.0, "XS_Angle": math.pi / 2,
                        "XS1_Profile": json_list(dem[south, col]), "XS2_Profile": json_list(dem[north, col]),
                        "LC1_Profile": json_list(np.ones(south.size, dtype=int)),
                        "LC2_Profile": json_list(np.ones(north.size, dtype=int)),
                        "XS1_Row": json_list(south), "XS1_Col": json_list(np.full(south.size, col)),
                        "XS2_Row": json_list(north), "XS2_Col": json_list(np.full(north.size, col)),
                        "Source_Stream_ID": 11 if col < 120 else 12})
    pd.DataFrame(records).to_csv(folder / "manual.tsv", sep="\t", index=False)
    pd.DataFrame({"COMID": [1001 + k for k in range(len(cols))], "qbf": 2.0,
                  "qmax": [Q_MAX[11 if c < 120 else 12] for c in cols]}).to_csv(folder / "manual_flows.csv", index=False)
    return {**inputs, "Manual_Cross_Sections_File": str(folder / "manual.tsv"),
            "Flow_File": str(folder / "manual_flows.csv"), "Print_VDT_Database": str(folder / "manual_vdt.csv")}


def json_list(values: np.ndarray) -> str:
    import json

    return json.dumps(np.asarray(values).tolist())


def test_manual_cross_sections_give_the_rating_curves_their_sampled_twins_give(tmp_path: Path) -> None:
    inputs = write_channel(tmp_path)
    cols = [45, 80, 150, 190]
    manual_inputs = write_manual_sections(tmp_path, inputs, cols)

    manual = pipeline.run(Configs.from_mapping(manual_inputs), quiet=True)
    sampled = pipeline.run(Configs.from_mapping(inputs), quiet=True, write=False)

    assert manual.cells.comids.tolist() == [1001, 1002, 1003, 1004]
    assert manual.cells.reaches.tolist() == [11, 11, 12, 12]
    for k, col in enumerate(cols):
        twin = int(np.flatnonzero(sampled.cells.cols == col)[0])
        assert manual.slopes[k] == sampled.slopes[twin]
        np.testing.assert_allclose(manual.curves.increments[k], sampled.curves.increments[twin], rtol=1e-12)
    vdt = pd.read_csv(tmp_path / "manual_vdt.csv")
    assert vdt["COMID"].tolist() == [1001, 1002, 1003, 1004]


@pytest.mark.parametrize("mask", [True, False])
def test_manual_cross_sections_take_the_water_mask_at_their_cells(tmp_path: Path, mask: bool) -> None:
    """Their stream cell is burned 0.6 m below the channel's bed, and their own land cover has no water. The mask's
    water at their cells is the bed, which gives the water level, so the bank tops are found above it, 9 m out. Without
    the mask, the search starts at the burned stream cell, and the bed beside it is the burn's banks, 1 m out. The
    maximum flows are above bankfull, so that the banks aren't held below their water."""
    inputs = write_channel(tmp_path)
    dem = Raster(inputs["DEM_File"]).read_array()
    dem[CENTER_ROW] -= 0.6
    srs, geotransform = _projected()
    _write_raster(Path(inputs["DEM_File"]), dem, geotransform, srs, gdal.GDT_Float64)
    if mask:
        inputs = write_water_mask(inputs, water_band(3))
    inputs = with_max_flows_above_bankfull(write_manual_sections(tmp_path, inputs, [45, 150]), reaches=(1001, 1002))
    manual = pipeline.run(Configs.from_mapping(inputs), quiet=True, write=False)

    for section in manual.sections:
        assert section.banks.method == "width_to_depth_ratio"
        expected = 9.0 if mask else 1.0
        assert (section.banks.left, section.banks.right) == pytest.approx((expected, expected))


# --- Real sites ---------------------------------------------------------------------------------------------------

SITES = Path(r"C:\Users\lrr43\Documents\masters\fim_sites_nencarta")
# Sweetwater Creek is in the repository (testing/data), so it runs everywhere; the others only where their data are
BUNDLED_SITE = Path(__file__).parent / "data" / "sweetwater_creek"
REAL_SITES = ["sweetwater_creek", "Salt_Creek_at_Wood_Dale_(2010)", "Blue_River_nr_Stanley_(2012)"]


def real_inputs(site: str, folder: Path) -> dict:
    import yaml

    if site == BUNDLED_SITE.name:
        inputs = yaml.safe_load((BUNDLED_SITE / "inputs.yaml").read_text(encoding="utf-8"))
        inputs = {key: str(BUNDLED_SITE / value) if isinstance(value, str) and (BUNDLED_SITE / value).is_file()
                  else value for key, value in inputs.items()}
    else:
        path = SITES / site / "ARC_InputFiles" / "GEOGLOWS_ARC_Input_fabdem_Bathy.yaml"
        if not path.exists():
            pytest.skip(f"{site}'s data aren't on this machine")
        inputs = yaml.safe_load(path.read_text(encoding="utf-8"))
    # The file was shared with curve2flood: ARC's keys only (AROutBATHY is set below)
    arc_keys = {f.name for f in fields(Configs) if f.init}
    inputs = {key: value for key, value in inputs.items() if key.lower() in arc_keys}
    # The site's own choice of Standard_Step left out: each test below says whether it wants it
    inputs.pop("Standard_Step", None)
    inputs.update({"Print_VDT_Database": str(folder / "vdt.parquet"), "BATHY_Out_File": str(folder / "bathy.tif"),
                   "AROutBATHY": str(folder / "bathy.tif"), "XS_Out_File": str(folder / "xs.txt"),
                   "Print_AP_Database": str(folder / "ap.parquet")})
    return inputs


@pytest.mark.parametrize("site", REAL_SITES)
def test_a_real_site_s_rating_curves_are_physically_consistent(tmp_path: Path, site: str) -> None:
    """Whatever the cross sections, banks and bathymetry, each written rating curve is Manning's discharge on its
    own cross section, rises with the water, and reaches the maximum flow unless the water spills there."""
    inputs = real_inputs(site, tmp_path)
    configs = Configs.from_mapping(inputs)
    results = pipeline.run(configs, quiet=True)
    roughness = DepthRoughness(configs.k_decay, configs.shallow_factor, configs.deep_factor) \
        if configs.depth_varying_n else None
    vdt = pd.read_parquet(tmp_path / "vdt.parquet")
    ap = pd.read_parquet(tmp_path / "ap.parquet")
    n = configs.vdt_database_numiterations

    assert len(vdt) > 0.8 * results.cells.count
    reached = 0
    for k in range(results.cells.count):
        increments = results.curves.increments[k]
        if np.isnan(increments[-1, Q]):
            continue
        section = results.sections[k]
        q, wse = increments[:, Q], increments[:, WSE]
        assert np.all(np.diff(q) >= 0.0) and np.all(np.diff(wse) >= 0.0)
        for i in range(1, n):  # the first increment's discharge can be the baseflow's
            assert q[i] == pytest.approx(discharge(section.xs, results.slopes[k], wse=wse[i], roughness=roughness,
                                                   slope_factor=configs.slope_adjustment_factor), rel=1e-9)
        assert 0.5 * results.max_flow[k] <= q[-1] <= 1.5 * results.max_flow[k]
        reached += abs(q[-1] / results.max_flow[k] - 1.0) < 1e-6
        assert np.all(increments[:, T] > 0.0)
    assert reached > 0.5 * len(vdt)
    # The AP database's area is discharge over velocity, which is the cross section's area
    merged = vdt.merge(ap, on=["COMID", "Row", "Col"], suffixes=("", "_ap"))
    assert np.allclose(merged[f"q_{n}"], merged[f"q_{n}_ap"])
    assert np.all(merged[[f"a_{i}" for i in range(1, n + 1)]].to_numpy() > 0.0)
    # The written tables and rasters line up with the DEM
    dem = Raster(configs.dem_file)
    bathymetry = Raster(tmp_path / "bathy.tif")
    assert bathymetry.same_grid(dem)
    assert vdt["Row"].between(0, dem.nrows - 1).all() and vdt["Col"].between(0, dem.ncols - 1).all()
    # The bathymetry raster's bed at each rating curve's cell is the one the curve was worked out on, wherever other
    # cross sections cross the cell
    beds = bathymetry.read_array(np.float64)[vdt["Row"].to_numpy(), vdt["Col"].to_numpy()]
    assert np.allclose(beds, vdt["Elev"], rtol=0.0, atol=1e-3)
    xs = pd.read_csv(tmp_path / "xs.txt", sep="\t")
    assert len(xs) == sum(1 for s in results.sections if s is not None and s.usable)


@pytest.mark.parametrize("site", REAL_SITES)
def test_a_real_site_s_standard_step_curves_rise_with_the_flow(tmp_path: Path, site: str) -> None:
    """With Standard_Step, each rating curve is at its reach's shares of the maximum flow, and its water surface and
    top width rise with the flow, above the bed."""
    from arc.standard_step import flows_for

    inputs = {**real_inputs(site, tmp_path), "Standard_Step": True}
    results = pipeline.run(Configs.from_mapping(inputs), quiet=True, write=False)
    n = results.curves.number_of_increments
    rated = 0
    for k in range(results.cells.count):
        increments = results.curves.increments[k]
        if np.isnan(increments[-1, Q]):
            continue
        rated += 1
        bed = results.sections[k].xs.elevations[results.sections[k].xs.elevations.size // 2]
        np.testing.assert_allclose(increments[:, Q], flows_for(results.max_flow[k], results.baseflow[k], n))
        assert np.all(np.diff(increments[:, WSE]) >= -1e-9) and np.all(increments[:, WSE] > bed)
        assert np.all(np.diff(increments[:, T]) >= -1e-9) and np.all(increments[:, T] > 0.0)
    assert rated > 0.7 * results.cells.count
