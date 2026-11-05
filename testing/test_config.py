from __future__ import annotations

import logging
from pathlib import Path

import pytest

from arc.config import Configs

BASE_INPUTS = {
    "DEM_File": "dem.tif",
    "Stream_File": "stream.tif",
    "LU_Raster_SameRes": "land.tif",
    "LU_Manning_n": "mannings.txt",
    "Flow_File": "flows.csv",
    "Flow_File_ID": "COMID",
    "Flow_File_QMax": "qmax",
    "StrmShp_File": "stream_network.gpkg",
    "reach_id": "COMID",
    "downstream_reach_id": "DSCOMID",
}
DEPTH_POWERLAW = {"drainage_area_field": "DA", "coefficient_depth": 0.5, "exponent_depth": 0.25}
WIDTH_POWERLAW = {"drainage_area_field": "DA", "coefficient_width": 3.0, "exponent_width": 0.4}
REPRESENTATIVE = {
    "Build_Representative_Cross_Section": True,
    "Representative_Cross_Section_File": "representative.csv",
}


def test_text_yaml_and_mapping_inputs_build_equal_configs(tmp_path: Path) -> None:
    """All three input styles parse to the same, correctly typed values."""
    text_file = tmp_path / "ARC_Input_File.txt"
    text_file.write_text(
        "# ARC Inputs\n"
        "DEM_File\tinputs/my dem.tif\n"
        "Stream_File\tinputs/stream.tif\n"
        "\n"
        "Degree_Manip\t40\n"
        "Low_Spot_Range\t2\n"
        "k_decay\t4.5\n"
        "Bathy_Use_Banks\tTrue\n"
        "FindBanksBasedOnLandCover\tfalse\n"
        "Stream_Slope_Method\treach_average\n",
        encoding="utf-8",
    )
    yaml_file = tmp_path / "ARC_Input_File.yaml"
    yaml_file.write_text(
        "DEM_File: inputs/my dem.tif\n"
        "Stream_File: inputs/stream.tif\n"
        "Degree_Manip: 40\n"
        "Low_Spot_Range: 2\n"
        "k_decay: 4.5\n"
        "Bathy_Use_Banks: true\n"
        "FindBanksBasedOnLandCover: false\n"
        "Stream_Slope_Method: reach_average\n",
        encoding="utf-8",
    )
    mapping = {
        "dem_file": "inputs/my dem.tif",
        "stream_file": "inputs/stream.tif",
        "degree_manip": 40,
        "low_spot_range": 2,
        "k_decay": 4.5,
        "bathy_use_banks": True,
        "findbanksbasedonlandcover": False,
        "stream_slope_method": "reach_average",
    }

    configs = Configs.from_file(text_file)

    assert configs == Configs.from_file(yaml_file) == Configs.from_mapping(mapping)
    assert configs.dem_file == "inputs/my dem.tif"
    assert configs.degree_manip == 40.0 and isinstance(configs.degree_manip, float)
    assert configs.low_spot_range == 2 and isinstance(configs.low_spot_range, int)
    assert configs.bathy_use_banks is True
    assert configs.findbanksbasedonlandcover is False
    assert configs.flow_file is None
    assert configs.gen_dir_dist == 10


def test_the_bed_cap_is_on_unless_turned_off(tmp_path: Path) -> None:
    """Bathy_Bed_Cap switches the limit on how fast the smoothed channel bed rises or falls (MAX_SLOPE per
    metre)."""
    text_file = tmp_path / "ARC_Input_File.txt"
    text_file.write_text("DEM_File\tdem.tif\nBathy_Bed_Cap\tFalse\n", encoding="utf-8")
    yaml_file = tmp_path / "ARC_Input_File.yaml"
    yaml_file.write_text("DEM_File: dem.tif\nBathy_Bed_Cap: false\n", encoding="utf-8")

    assert Configs.from_mapping(BASE_INPUTS).bathy_bed_cap is True
    assert Configs.from_file(text_file).bathy_bed_cap is False
    assert Configs.from_file(yaml_file).bathy_bed_cap is False
    assert Configs.from_mapping({**BASE_INPUTS, "BATHY_BED_CAP": "true"}).bathy_bed_cap is True


def test_depth_varying_roughness_is_off_unless_turned_on(tmp_path: Path) -> None:
    """Depth_Varying_N switches the scaling of Manning's n by k_decay, shallow_factor and deep_factor."""
    text_file = tmp_path / "ARC_Input_File.txt"
    text_file.write_text("DEM_File\tdem.tif\nDepth_Varying_N\tTrue\n", encoding="utf-8")
    yaml_file = tmp_path / "ARC_Input_File.yaml"
    yaml_file.write_text("DEM_File: dem.tif\nDepth_Varying_N: true\nk_decay: 3.0\n", encoding="utf-8")

    assert Configs.from_mapping(BASE_INPUTS).depth_varying_n is False
    assert Configs.from_file(text_file).depth_varying_n is True
    configs = Configs.from_file(yaml_file)
    assert configs.depth_varying_n is True and configs.k_decay == 3.0
    # The parameters are checked whether the scaling is on or off
    with pytest.raises(ValueError, match="k_decay must be finite and positive"):
        Configs.from_mapping({**BASE_INPUTS, "Depth_Varying_N": False, "k_decay": 0})


def test_older_input_file_keys_are_accepted_and_left_out() -> None:
    """Older ARC input files carry keys ARC doesn't use."""
    inputs = {**BASE_INPUTS, "Spatial_Units": "deg", "Str_Limit_Val": 1, "Print_VDT": "vdt.txt"}

    configs = Configs.from_mapping(inputs)

    assert configs == Configs.from_mapping(BASE_INPUTS)


@pytest.mark.parametrize("key", ["Comid_Flow_File", "mapper", "TW_MultFact", "FSOutBATHY", "Flood_Bathy_Mask",
                                 "MPI_K", "FLDPLN_Median_Filter_Size", "OutSHP", "FS_ADJUST_FLOW_BY_FRACTION"])
def test_curve2flood_s_keys_are_rejected(key) -> None:
    """An input file is ARC's alone: curve2flood gets its own."""
    with pytest.raises(ValueError, match="Unrecognized config key"):
        Configs.from_mapping({**BASE_INPUTS, key: 1})


def test_check_keys_checks_names_and_types_without_a_run() -> None:
    # The standard step needs the stream network to run, but its key is checked alone
    assert Configs.check_keys({"Standard_Step": "true", "Degree_Manip": "6.1", "aroutbathy": "bathy.tif"}) == \
        {"standard_step": True, "degree_manip": 6.1, "bathy_out_file": "bathy.tif"}
    with pytest.raises(ValueError, match=r"'tw_multfact'"):
        Configs.check_keys({"TW_MultFact": 1.5})
    with pytest.raises(ValueError, match="gen_dir_dist must be an integer"):
        Configs.check_keys({"Gen_Dir_Dist": "ten"})


def test_the_bathymetry_smoothing_is_on_unless_turned_off() -> None:
    configs = Configs.from_mapping(BASE_INPUTS)
    assert configs.bathy_smooth_banks is True and configs.bathy_smooth_bed is True
    configs = Configs.from_mapping({**BASE_INPUTS, "Bathy_Smooth_Banks": "False", "Bathy_Smooth_Bed": False})
    assert configs.bathy_smooth_banks is False and configs.bathy_smooth_bed is False


@pytest.mark.parametrize(("standard_step", "use_banks", "given", "expected"),
                         [(True, False, None, False), (False, False, None, True), (False, True, None, False),
                          (True, True, None, False), (True, False, "True", True), (False, True, "True", True),
                          (False, False, "False", False), (True, False, "", False)])
def test_the_bed_is_smoothed_by_default_only_without_the_standard_step_and_bank_elevations(standard_step, use_banks,
                                                                                            given, expected) -> None:
    inputs = {**BASE_INPUTS, "Standard_Step": standard_step, "Bathy_Use_Banks": use_banks}
    if given is not None:
        inputs["Bathy_Smooth_Bed"] = given

    configs = Configs.from_mapping(inputs)

    assert configs.bathy_smooth_bed is expected
    assert configs.bathy_smooth_banks is True


def test_the_water_mask_is_an_input() -> None:
    """curve2flood reads BathyWaterMask too, but ARC finds the banks in it."""
    assert Configs.from_mapping(BASE_INPUTS).bathywatermask is None
    assert Configs.from_mapping({**BASE_INPUTS, "BathyWaterMask": "mask.tif"}).bathywatermask == "mask.tif"


def test_unrecognized_key_is_rejected_with_a_suggestion() -> None:
    with pytest.raises(ValueError, match=r"'degree_manipulation' \(did you mean 'degree_manip'\?\)"):
        Configs.from_mapping({**BASE_INPUTS, "Degree_Manipulation": 40})


def test_x_section_dist_is_ignored_with_a_warning(caplog) -> None:
    """Cross sections have no set length any more, so an input file that gives one still loads."""
    with caplog.at_level("WARNING"):
        configs = Configs.from_mapping({**BASE_INPUTS, "X_Section_Dist": 5000})
    assert configs == Configs.from_mapping(BASE_INPUTS)
    assert "x_section_dist is no longer used" in caplog.text


def test_aroutflood_is_ignored_with_a_warning(caplog) -> None:
    """ARC no longer writes the flood raster, so an input file that names one still loads."""
    with caplog.at_level("WARNING"):
        configs = Configs.from_mapping({**BASE_INPUTS, "AROutFLOOD": "flood.tif"})
    assert configs == Configs.from_mapping(BASE_INPUTS)
    assert "aroutflood is no longer used" in caplog.text


@pytest.mark.parametrize("key", ["alpha_low", "alpha_boost", "f_min"])
def test_obsolete_roughness_keys_name_their_replacement(key: str) -> None:
    with pytest.raises(ValueError, match=f"{key} has been replaced by shallow_factor and deep_factor"):
        Configs.from_mapping({**BASE_INPUTS, key: 1.0})


def test_the_roughness_slope_and_output_defaults() -> None:
    configs = Configs.from_mapping(BASE_INPUTS)

    assert (configs.k_decay, configs.shallow_factor, configs.deep_factor) == (6.0, 2.0, 1.0)
    assert configs.slope_adjustment_factor == 1.0
    assert configs.stream_slope_method == "local_average_corrected"
    assert configs.compression == "ZSTD"
    assert configs.bathy_bankless_max_baseflow == 20.0
    assert Configs.from_mapping({**BASE_INPUTS, "Bathy_Bankless_Max_Baseflow": "0"}).bathy_bankless_max_baseflow == 0.0
    assert (configs.valley_sections, configs.valley_dir_widths, configs.valley_dir_dist) == (False, 16.0, 40)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        pytest.param({"StrmShp_File": "", "Stream_Slope_Method": "end_points"}, "'end_points' stream slope method requires a strmshp_file", id="end-points-needs-stream-vector"),
        pytest.param({"Stream_Slope_Method": "steepest"}, "stream_slope_method must be one of", id="unknown-slope-method"),
        pytest.param({"Build_Representative_Cross_Section": True}, "requires a representative_cross_section_file", id="representative-needs-output-file"),
        pytest.param({"coefficient_depth": 0.5, "exponent_depth": 0.25}, "drainage_area_field is required", id="power-law-needs-drainage-area"),
        pytest.param({"drainage_area_field": "DA", "coefficient_depth": 0.5}, "requires coefficient_depth and exponent_depth together", id="unpaired-depth-power-law"),
        pytest.param({**WIDTH_POWERLAW, "StrmShp_File": ""}, "StrmShp_File is required", id="power-law-needs-stream-vector"),
        pytest.param({"Flow_File_BF": "baseflow", "AROutBATHY": "bathy.tif", "reach_id": ""}, "requires both reach_id and downstream_reach_id", id="bathymetry-needs-reach-ids"),
        pytest.param({"Degree_Manip": "wide"}, "degree_manip must be a number", id="non-numeric-float"),
        pytest.param({"Low_Spot_Range": "2.5"}, "low_spot_range must be an integer", id="non-integer-int"),
        pytest.param({"k_decay": 0}, "k_decay must be finite and positive", id="k-decay"),
        pytest.param({"shallow_factor": 0.5}, "shallow_factor must be finite and >= 1", id="shallow-factor"),
        pytest.param({"deep_factor": 1.5}, r"deep_factor must be finite and in \(0, 1\]", id="deep-factor"),
        pytest.param({"slope_adjustment_factor": "nan"}, "slope_adjustment_factor must be finite and positive", id="slope-adjustment-factor"),
        pytest.param({"Bathy_Bankless_Max_Baseflow": -1}, "Bathy_Bankless_Max_Baseflow must be 0 or more", id="bankless-max-baseflow"),
        pytest.param({"Valley_Sections": True, "Valley_Dir_Dist": 1}, "Valley_Dir_Dist must be at least 2 cells", id="valley-dir-dist"),
        pytest.param({"Valley_Sections": True, "Valley_Dir_Widths": -1}, "Valley_Dir_Widths must be finite and 0 or more", id="valley-dir-widths"),
    ],
)
def test_invalid_options_are_rejected(overrides: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        Configs.from_mapping({**BASE_INPUTS, **overrides})


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        pytest.param({"Flow_File_BF": "baseflow", "AROutBATHY": "bathy.tif"}, (False, False, False, "bathy.tif"), id="baseflow"),
        pytest.param({"AROutBATHY": "bathy.tif"}, (False, False, False, None), id="no-source-disables-bathymetry"),
        pytest.param({**DEPTH_POWERLAW, "AROutBATHY": "bathy.tif"}, (False, True, False, "bathy.tif"), id="depth-power-law"),
        pytest.param({**WIDTH_POWERLAW, "AROutBATHY": "bathy.tif"}, (False, True, True, None), id="width-power-law-has-no-depth"),
        pytest.param({**REPRESENTATIVE, **WIDTH_POWERLAW, "AROutBATHY": "bathy.tif"}, (False, True, True, None), id="representative-width-power-law-has-no-depth"),
        pytest.param({**DEPTH_POWERLAW, **WIDTH_POWERLAW, "AROutBATHY": "bathy.tif"}, (False, True, True, "bathy.tif"), id="depth-and-width-power-laws"),
        pytest.param({**REPRESENTATIVE, "Flow_File_BF": "baseflow", **DEPTH_POWERLAW, **WIDTH_POWERLAW, "AROutBATHY": "bathy.tif"}, (True, False, True, "bathy.tif"), id="representative-baseflow-beats-power-law"),
        pytest.param({**REPRESENTATIVE, "Flow_File_ID": "", "Flow_File_BF": "baseflow", **DEPTH_POWERLAW, "AROutBATHY": "bathy.tif"}, (False, True, False, "bathy.tif"), id="representative-partial-baseflow-falls-back"),
        pytest.param({**REPRESENTATIVE, "AROutBATHY": "bathy.tif"}, (False, False, False, None), id="representative-without-source"),
        pytest.param({"Flow_File_BF": "baseflow", "AROutBATHY": "arc.tif", "BATHY_Out_File": "other.tif"}, (False, False, False, "arc.tif"), id="aroutbathy-beats-bathy-out-file"),
    ],
)
def test_bathymetry_options_are_derived_like_read_main_input_file(overrides: dict, expected: tuple) -> None:
    """Expected values are what read_main_input_file produces for the same inputs, but that a width law alone, which
    gives no depth to carve, doesn't turn the bathymetry on (it wrote an empty bathymetry raster)."""
    configs = Configs.from_mapping({**BASE_INPUTS, **overrides})

    assert (
        configs.use_representative_baseflow_bathymetry,
        configs.use_bathymetry_powerlaw,
        configs.use_bathymetry_powerlaw_width,
        configs.bathy_out_file,
    ) == expected


RATING_CURVE_OUTPUTS = {"vdt": {"Print_VDT_Database": "vdt.csv"}, "ap": {"Print_AP_Database": "ap.csv"},
                        "curve-file": {"Print_Curve_File": "curve.csv"}}


def test_rating_curves_are_made_unless_only_representative_cross_sections_are_asked_for() -> None:
    assert Configs.from_mapping(BASE_INPUTS).makes_rating_curves
    assert not Configs.from_mapping({**BASE_INPUTS, **REPRESENTATIVE}).makes_rating_curves
    for output in RATING_CURVE_OUTPUTS.values():
        assert Configs.from_mapping({**BASE_INPUTS, **REPRESENTATIVE, **output}).makes_rating_curves


@pytest.mark.parametrize("output", RATING_CURVE_OUTPUTS.values(), ids=RATING_CURVE_OUTPUTS.keys())
@pytest.mark.parametrize(
    "bathymetry",
    [
        pytest.param({"Flow_File_BF": "baseflow"}, id="baseflow"),
        pytest.param({"Flow_File_BF": "baseflow", **DEPTH_POWERLAW, **WIDTH_POWERLAW}, id="baseflow-and-power-laws"),
        pytest.param({**WIDTH_POWERLAW}, id="width-power-law"),
        pytest.param({}, id="no-source"),
    ],
)
def test_beside_rating_curves_representative_cross_sections_carve_as_the_rating_curves_do(output: dict,
                                                                                          bathymetry: dict) -> None:
    inputs = {**BASE_INPUTS, **bathymetry, **output, "AROutBATHY": "bathy.tif"}
    alone = Configs.from_mapping(inputs)
    both = Configs.from_mapping({**inputs, **REPRESENTATIVE})

    derived = ("use_representative_baseflow_bathymetry", "use_bathymetry_powerlaw", "use_bathymetry_powerlaw_width",
               "bathy_out_file", "makes_rating_curves")
    assert [getattr(both, name) for name in derived] == [getattr(alone, name) for name in derived]


def test_beside_rating_curves_a_flow_file_without_baseflow_is_no_fallback(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger="arc._log"):
        Configs.from_mapping({**BASE_INPUTS, **REPRESENTATIVE, **RATING_CURVE_OUTPUTS["vdt"]})

    assert "Falling back" not in caplog.text


def test_bathymetry_fallbacks_are_logged(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger="arc._log"):
        Configs.from_mapping({**BASE_INPUTS, **REPRESENTATIVE, "AROutBATHY": "bathy.tif"})

    assert "Missing: Flow_File_BF. Falling back to drainage-area power-law bathymetry." in caplog.text
    assert "disabling bathymetry estimation" in caplog.text


def test_reach_average_curve_file_needs_a_curve_file() -> None:
    inputs = {**BASE_INPUTS, "Reach_Average_Curve_File": "True"}

    assert Configs.from_mapping(inputs).reach_average_curve_file is False
    assert Configs.from_mapping({**inputs, "Print_Curve_File": "curve.csv"}).reach_average_curve_file is True
