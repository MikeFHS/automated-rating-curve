"""The decision log, the QA flags, the log file and the outputs' metadata (arc.outputs.decisions, arc.outputs.tables),
on the real site in testing/data."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest
import yaml
from osgeo import gdal

import arc
from arc import pipeline
from arc.config import Configs
from arc.outputs.decisions import QA, decision_log_dataframe, qa_names
from arc.pipeline import Cells

SITE = Path(__file__).parent / "data" / "sweetwater_creek"


def site_inputs(folder: Path, **more) -> dict:
    inputs = yaml.safe_load((SITE / "inputs.yaml").read_text(encoding="utf-8"))
    inputs = {key: str(SITE / value) if isinstance(value, str) and (SITE / value).is_file() else value
              for key, value in inputs.items()}
    inputs.update({"Print_VDT_Database": str(folder / "vdt.parquet"), "Print_AP_Database": str(folder / "ap.parquet"),
                   "Print_Curve_File": str(folder / "curve.parquet"), "BATHY_Out_File": str(folder / "bathy.tif"),
                   "XS_Out_File": str(folder / "xs.parquet"), "Print_Decision_Log": str(folder / "decisions.csv"),
                   "Log_File": str(folder / "arc.log"), **more})
    return inputs


@pytest.fixture(scope="module")
def site_run(tmp_path_factory) -> tuple[Path, pipeline.Results]:
    folder = tmp_path_factory.mktemp("site")
    return folder, pipeline.run(Configs.from_mapping(site_inputs(folder)), quiet=True)


def test_the_decision_log_has_a_row_for_each_stream_cell(site_run) -> None:
    folder, results = site_run
    log = pd.read_csv(folder / "decisions.csv", keep_default_na=False, na_values=[""])

    assert len(log) == results.cells.count
    np.testing.assert_array_equal(log["Stream_Row"], results.cells.rows)
    np.testing.assert_array_equal(log["Stream_Col"], results.cells.cols)
    np.testing.assert_array_equal(log["QA_Flag"], results.qa_flags)
    assert set(log["Slope_Source"]) <= {"cell", "reach", "neighbours"}
    assert set(log["Carve"]) <= {"none", "baseflow", "depth_law", "stream_level"}
    assert set(log["Rating_Curve"]) <= {"rated", "no_cross_section", "reach_dropped", "edge", "no_curve"}
    assert (log["QA"].fillna("") == [qa_names(flag) for flag in log["QA_Flag"]]).all()


def test_the_rated_cells_are_the_vdt_database_s_with_the_same_qa_flags(site_run) -> None:
    folder, _ = site_run
    log = pd.read_csv(folder / "decisions.csv")
    rated = log[log["Rating_Curve"] == "rated"]
    for name in ("vdt", "ap", "curve"):
        table = pd.read_parquet(folder / f"{name}.parquet")
        assert table.columns[-1] == "QA_Flag"
        merged = table.merge(rated, on=["COMID", "Row", "Col"], suffixes=("", "_log"))
        assert len(merged) == len(table)
        np.testing.assert_array_equal(merged["QA_Flag"], merged["QA_Flag_log"])
    assert len(pd.read_parquet(folder / "vdt.parquet")) == len(rated)
    # A cell without a rating curve says why
    assert (log.loc[log["Rating_Curve"] == "no_curve", "QA_Flag"] & int(QA.NO_RATING_CURVE)).all()


def test_the_flags_follow_the_decisions(site_run) -> None:
    folder, results = site_run
    log = pd.read_csv(folder / "decisions.csv", keep_default_na=False, na_values=[""])
    flags = log["QA_Flag"].to_numpy()

    not_found = log["Bank_Method"].isin(["none", "single_cell", "target_width"]).to_numpy()
    np.testing.assert_array_equal((flags & int(QA.NO_BANKS_FOUND)) > 0, not_found)
    np.testing.assert_array_equal((flags & int(QA.CARVED_TO_STREAM)) > 0, log["Carve"] == "stream_level")
    np.testing.assert_array_equal((flags & int(QA.STEP_HELD)) > 0, log["Step_Held"] > 0)
    np.testing.assert_array_equal((flags & int(QA.STEP_CRITICAL)) > 0, log["Step_Critical"] > 0)
    np.testing.assert_array_equal((flags & int(QA.SLOPE_FROM_NEIGHBOURS)) > 0, log["Slope_Source"] == "neighbours")
    # The standard step's counts for each cell add up to its totals
    assert results.step.held_at.sum() == results.step.held and results.step.critical_at.sum() == results.step.critical


def test_cells_skipped_for_want_of_a_slope_are_logged_too(site_run) -> None:
    _, results = site_run
    skipped = Cells(np.array([1, 2]), np.array([3, 4]), np.array([99, 99]), np.array([99, 99]))
    log = decision_log_dataframe(pipeline.Results(**{**results.__dict__, "skipped": skipped}))

    assert len(log) == results.cells.count + 2
    tail = log.tail(2)
    assert (tail["Rating_Curve"] == "no_slope").all() and (tail["QA_Flag"] == int(QA.NO_SLOPE)).all()
    assert tail["Row"].tolist() == [1, 2] and tail["Col"].tolist() == [3, 4]


def test_parquet_outputs_and_the_bathymetry_raster_say_what_made_them(site_run) -> None:
    folder, results = site_run
    for name in ("vdt", "ap", "curve", "xs"):
        metadata = pq.read_metadata(folder / f"{name}.parquet").metadata
        assert metadata[b"arc_version"].decode() == arc.__version__
        assert Configs.from_mapping(json.loads(metadata[b"arc_configs"])) == results.configs
    tags = gdal.Open(str(folder / "bathy.tif")).GetMetadata()
    assert tags["arc_version"] == arc.__version__
    assert Configs.from_mapping(json.loads(tags["arc_configs"])) == results.configs


def test_the_log_file_has_the_version_the_inputs_and_the_info_messages(site_run) -> None:
    folder, results = site_run
    text = (folder / "arc.log").read_text(encoding="utf-8")

    assert f"ARC {arc.__version__}" in text
    assert "Inputs: " + json.dumps(results.configs.as_mapping(), default=str) in text
    assert "Standard step:" in text and "QA flags over" in text
    # The console's level is left as it was, and the file is let go
    assert arc.LOG.level == logging.WARNING
    assert not any(isinstance(h, logging.FileHandler) for h in arc.LOG.handlers)


def test_the_inputs_make_themselves_again() -> None:
    configs = Configs.from_mapping(site_inputs(Path("out")))
    assert Configs.from_mapping(configs.as_mapping()) == configs


def test_subcell_channels_reconstruct_the_narrow_channels_and_clear_their_flag(tmp_path: Path) -> None:
    results = pipeline.run(Configs.from_mapping(site_inputs(tmp_path, Subcell_Channels=True)), quiet=True)
    log = pd.read_csv(tmp_path / "decisions.csv")

    assert log["Subcell"].sum() > 0.5 * len(log)
    assert not (log.loc[log["Subcell"], "QA_Flag"] & int(QA.NARROW_CHANNEL)).any()
    assert all(s.xs.subcell == flag for s, flag in zip(results.sections, log["Subcell"]) if s is not None)
