from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import LineString

from arc.Automated_Rating_Curve_Generator import (
    build_bathymetry_geometry_dict,
    read_flow_file,
    read_main_input_file,
)
from arc.cross_section import CrossSection, compute_stream_derivatives, _find_nonzero_interior_bounds
from arc.hydraulic_data import build_representative_cross_section_dataframe


def _build_representative_cross_section_record(
    comid: int,
    slope: float,
    thalweg: float,
    inflect_curve: list[float],
) -> dict:
    """Create a compact representative-cross-section input record for tests."""
    profile = [0.00, 0.02, 0.04, 0.06, 0.08, 0.10]
    mannings = [0.03] * len(profile)
    return {
        "COMID": comid,
        "Row": 0,
        "Col": 0,
        "XS1_Profile": profile,
        "Ordinate_Dist": 1.0,
        "Manning_N_Raster1": mannings,
        "XS2_Profile": profile,
        "Manning_N_Raster2": mannings,
        "r1": 0,
        "c1": 0,
        "r2": 0,
        "c2": 0,
        "Slope": slope,
        "Thalweg": thalweg,
        "Inflect_D2W_Dy2": inflect_curve,
    }


def _build_test_cross_section() -> CrossSection:
    """Create a minimal reusable cross section for bank-selection tests."""
    params = {
        "d_x_section_distance": 10.0,
        "b_FindBanksBasedOnLandCover": False,
        "i_lc_water_value": 80,
        "d_bathymetry_trapzoid_height": 0.1,
        "b_bathy_use_banks": False,
        "d_degree_manipulation": 0.0,
        "d_degree_interval": 0.0,
        "i_boundary_number": 0,
        "nrows": 5,
        "ncols": 5,
    }
    return CrossSection(
        1.0,
        1.0,
        np.zeros((5, 5), dtype=np.float64),
        np.zeros((5, 5), dtype=np.uint8),
        params,
    )


def test_read_main_input_file_keeps_bathymetry_enabled_without_baseflow(tmp_path: Path) -> None:
    """A complete power-law configuration should bypass ``Flow_File_BF`` cleanly."""
    dummy_vector = tmp_path / "stream_network.gpkg"
    gdf = gpd.GeoDataFrame(
        {"COMID": [1], "DA": [12.0]},
        geometry=[LineString([(0.0, 0.0), (1.0, 1.0)])],
        crs="EPSG:4326",
    )
    gdf.to_file(dummy_vector, driver="GPKG")

    params = read_main_input_file(
        "",
        {
            "DEM_File": "dem.tif",
            "Stream_File": "stream.tif",
            "LU_Raster_SameRes": "land.tif",
            "LU_Manning_n": "mannings.txt",
            "Flow_File": "flows.csv",
            "Flow_File_ID": "COMID",
            "Flow_File_QMax": "qmax",
            "StrmShp_File": str(dummy_vector),
            "AROutBATHY": "bathy.tif",
            "drainage_area_field": "DA",
            "coefficient_depth": 0.5,
            "exponent_depth": 0.25,
            "coefficient_width": 3.0,
            "exponent_width": 0.4,
        },
    )

    assert params["s_flow_file_baseflow"] == ""
    assert params["b_use_bathymetry_powerlaw"] is True
    assert params["s_output_bathymetry_path"] == "bathy.tif"
    assert params["s_bathymetry_drainage_area_field"] == "DA"


def test_read_main_input_file_rejects_partial_powerlaw_configuration() -> None:
    """The new parameters are all-or-nothing so ARC never guesses intent."""
    with pytest.raises(ValueError, match="requires all five optional parameters together"):
        read_main_input_file(
            "",
            {
                "DEM_File": "dem.tif",
                "Stream_File": "stream.tif",
                "LU_Raster_SameRes": "land.tif",
                "LU_Manning_n": "mannings.txt",
                "Flow_File": "flows.csv",
                "Flow_File_ID": "COMID",
                "Flow_File_QMax": "qmax",
                "StrmShp_File": "stream_network.gpkg",
                "AROutBATHY": "bathy.tif",
                "drainage_area_field": "DA",
                "coefficient_depth": 0.5,
                "exponent_depth": 0.25,
            },
        )


def test_read_main_input_file_requires_representative_output_path() -> None:
    """Representative cross sections need an explicit output path."""
    with pytest.raises(
        ValueError,
        match="Build_Representative_Cross_Section requires Representative_Cross_Section_File",
    ):
        read_main_input_file(
            "",
            {
                "DEM_File": "dem.tif",
                "Stream_File": "stream.tif",
                "LU_Raster_SameRes": "land.tif",
                "LU_Manning_n": "mannings.txt",
                "Flow_File": "flows.csv",
                "Flow_File_ID": "COMID",
                "Flow_File_QMax": "qmax",
                "Build_Representative_Cross_Section": True,
            },
        )


def test_representative_baseflow_inputs_take_precedence_over_powerlaw() -> None:
    """A complete baseflow trio should select hydraulic bathymetry without QMax."""
    params = read_main_input_file(
        "",
        {
            "DEM_File": "dem.tif",
            "Stream_File": "stream.tif",
            "LU_Raster_SameRes": "land.tif",
            "LU_Manning_n": "mannings.txt",
            "Flow_File": "flows.csv",
            "Flow_File_ID": "COMID",
            "Flow_File_BF": "baseflow",
            "StrmShp_File": "stream_network.gpkg",
            "reach_id": "COMID",
            "downstream_reach_id": "DSCOMID",
            "AROutBATHY": "bathy.tif",
            "Build_Representative_Cross_Section": True,
            "Representative_Cross_Section_File": "representative.csv",
            "drainage_area_field": "DA",
            "coefficient_depth": 0.5,
            "exponent_depth": 0.25,
            "coefficient_width": 3.0,
            "exponent_width": 0.4,
        },
    )

    assert params["b_use_representative_baseflow_bathymetry"] is True
    assert params["b_use_bathymetry_powerlaw"] is False
    assert params["s_flow_file_qmax"] == ""


def test_read_flow_file_supports_baseflow_without_qmax(tmp_path: Path) -> None:
    """Representative bathymetry should load only its ID and baseflow columns."""
    flow_path = tmp_path / "flows.csv"
    flow_path.write_text("COMID,baseflow\n1,2.5\n", encoding="utf-8")

    flow_data = read_flow_file(str(flow_path), "COMID", "baseflow", "")

    assert flow_data == {1: {"baseflow": 2.5}}


def test_reach_scale_inflect_bank_depth_maps_back_to_local_bank_indices() -> None:
    """Reach-average INFLECT bank depth should map to local side widths."""
    x_section = _build_test_cross_section()
    x_section.da_xs_profile1[:5] = np.array([0.00, 0.01, 0.02, 0.05, 0.09], dtype=np.float64)
    x_section.da_xs_profile2[:5] = np.array([0.00, 0.015, 0.03, 0.06, 0.09], dtype=np.float64)
    x_section.xs1_n = 5
    x_section.xs2_n = 5
    x_section.d_ordinate_dist = 1.0
    x_section.set_reach_scale_inflect_bank_index(3)

    bank_1_index, bank_2_index = x_section._find_bank_using_reach_scale_inflection()

    assert (bank_1_index, bank_2_index) == (3, 2)


def test_compute_stream_derivatives_uses_windowed_regression_smoothing() -> None:
    """INFLECT derivatives should be smoothed by a local regression window."""
    width_array = np.square(np.arange(30, dtype=np.float64)) + 1.0

    dW_dy, d2W_dy2 = compute_stream_derivatives(width_array, 1.0)

    assert np.allclose(dW_dy[:5], 0.0)
    assert np.allclose(dW_dy[-5:], 0.0)
    assert np.all(dW_dy[5:25] > 0.0)
    assert np.all(d2W_dy2[10:20] > 0.0)


def test_find_nonzero_interior_bounds_skips_padded_derivative_edges() -> None:
    """The second derivative should ignore padded zero edges in ``dW_dy``."""
    dW_dy = np.array([0.0, 0.0, 1.5, 2.0, 2.5, 0.0, 0.0], dtype=np.float64)

    start, end = _find_nonzero_interior_bounds(dW_dy)

    assert (start, end) == (2, 5)


def test_build_representative_cross_section_dataframe_uses_inflect_terrace_depth() -> None:
    """Representative staging should stop at the reach-average INFLECT terrace."""
    cross_section_data = [
        _build_representative_cross_section_record(7, 0.001, 0.00, [2.0, 1.0, 0.0, -1.0, -2.0]),
        _build_representative_cross_section_record(7, 0.001, 0.01, [4.0, 2.0, 0.0, -2.0, -4.0]),
    ]

    df = build_representative_cross_section_dataframe(cross_section_data)

    assert df.shape == (5, 21)
    assert df["COMID"].tolist() == [7] * 5
    assert df["Depth_Stage_Index"].tolist() == [1, 2, 3, 4, 5]
    assert df["Depth_Stage_Meters"].tolist() == pytest.approx([0.01, 0.02, 0.03, 0.04, 0.05])
    assert df["Stream_Slope"].tolist() == pytest.approx([0.001] * 5)
    assert df["Reach_Inflect_Terrace_Depth"].tolist() == pytest.approx([0.05] * 5)
    assert df["Cross_Section_Count"].tolist() == [2] * 5
    assert df["Hydraulic_Sample_Count"].tolist() == [2] * 5
    assert df["Representative_Thalweg_Elevation"].tolist() == pytest.approx([0.005] * 5)
    assert df["Mean_Depth"].tolist() == pytest.approx([0.01, 0.02, 0.03, 0.04, 0.05])
    assert np.all(df["Mean_Discharge"].to_numpy(dtype=np.float64) > 0.0)


def test_build_representative_cross_section_dataframe_derives_dimensions_from_width_and_area() -> None:
    """Representative dimensions should satisfy the staged trapezoidal area relation."""
    cross_section_data = [
        _build_representative_cross_section_record(9, 0.001, 0.00, [2.0, 1.0, 0.0, -1.0, -2.0]),
        _build_representative_cross_section_record(9, 0.002, 0.00, [2.5, 1.5, 0.5, -0.5, -3.0]),
    ]

    df = build_representative_cross_section_dataframe(cross_section_data)

    representative_area = df["Representative_Cross_Sectional_Area"].to_numpy(dtype=np.float64)
    representative_width = df["Representative_Top_Width"].to_numpy(dtype=np.float64)
    representative_depth_increment = df["Representative_Depth_Increment"].to_numpy(dtype=np.float64)
    representative_depth = df["Representative_Depth"].to_numpy(dtype=np.float64)

    assert df["Stream_Slope"].tolist() == pytest.approx([0.0015] * df.shape[0])
    assert np.all(np.diff(representative_area) >= -1e-9)
    assert np.all(np.diff(representative_width) >= -1e-9)
    assert np.all(np.diff(representative_depth) >= -1e-9)
    assert df["Representative_Left_Station"].tolist() == pytest.approx((-0.5 * representative_width).tolist())
    assert df["Representative_Right_Station"].tolist() == pytest.approx((0.5 * representative_width).tolist())
    assert df["Representative_Stage_Elevation"].tolist() == pytest.approx(
        (df["Representative_Thalweg_Elevation"] + df["Representative_Depth"]).tolist()
    )

    previous_width = 0.0
    previous_area = 0.0
    previous_depth = 0.0
    for i in range(df.shape[0]):
        delta_area = representative_area[i] - previous_area
        expected_delta_area = 0.5 * (previous_width + representative_width[i]) * representative_depth_increment[i]
        assert delta_area == pytest.approx(expected_delta_area, abs=1e-6)
        assert representative_depth[i] == pytest.approx(previous_depth + representative_depth_increment[i], abs=1e-6)
        previous_width = representative_width[i]
        previous_area = representative_area[i]
        previous_depth = representative_depth[i]


def test_build_representative_cross_section_dataframe_skips_invalid_cross_sections() -> None:
    """Cross sections with invalid slope should not contribute to staged medians."""
    cross_section_data = [
        _build_representative_cross_section_record(11, 0.001, 0.00, [2.0, 1.0, 0.0, -1.0, -2.0]),
        _build_representative_cross_section_record(11, 0.0, 0.00, [2.0, 1.0, 0.0, -1.0, -2.0]),
    ]

    df = build_representative_cross_section_dataframe(cross_section_data)

    assert df.shape == (5, 21)
    assert df["Stream_Slope"].tolist() == pytest.approx([0.001] * 5)
    assert df["Cross_Section_Count"].tolist() == [2] * 5
    assert df["Hydraulic_Sample_Count"].tolist() == [1] * 5
    assert np.all(df["Mean_Discharge"].to_numpy(dtype=np.float64) > 0.0)


def test_build_bathymetry_geometry_dict_uses_drainage_area_power_laws(tmp_path: Path) -> None:
    """Per-reach target depth and width should be computed from the vector field."""
    stream_path = tmp_path / "stream_network.gpkg"
    gdf = gpd.GeoDataFrame(
        {"COMID": [101, 202], "DrainArea": [4.0, 9.0]},
        geometry=[
            LineString([(0.0, 0.0), (1.0, 0.0)]),
            LineString([(0.0, 1.0), (1.0, 1.0)]),
        ],
        crs="EPSG:4326",
    )
    gdf.to_file(stream_path, driver="GPKG")

    bathy_dict = build_bathymetry_geometry_dict(
        str(stream_path),
        "COMID",
        "DrainArea",
        coefficient_depth=2.0,
        exponent_depth=0.5,
        coefficient_width=3.0,
        exponent_width=1.0,
    )

    assert bathy_dict[101]["depth"] == pytest.approx(4.0)
    assert bathy_dict[101]["width"] == pytest.approx(12.0)
    assert bathy_dict[202]["depth"] == pytest.approx(6.0)
    assert bathy_dict[202]["width"] == pytest.approx(27.0)
