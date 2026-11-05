from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from arc.bathymetry import carve_channel, find_banks
from arc.hydraulics import hydraulic_geometry
from arc.outputs import (
    REPRESENTATIVE_CROSS_SECTION_COLUMNS,
    XS_EXPORT_COLUMNS,
    CrossSectionRecord,
    RatingCurves,
    RepresentativeSample,
    format_array,
    representative_cross_section_dataframe,
    vdt_dataframe,
    write_ap,
    write_cross_sections,
    write_curve_file,
    write_reach_average_curve_file,
    write_representative_cross_sections,
    write_table,
    write_vdt,
)
from arc.outputs.rating_curves import P, Q, V
from arc.outputs.representative import DEPTH_INCREMENT
from arc.xsection.xsection import XSection

INCREMENTS = 6
FITTED = ["depth_a", "depth_b", "tw_a", "tw_b", "vel_a", "vel_b"]


def random_curves(rng, cells: int = 60) -> RatingCurves:
    """Rating curves shaped like ARC's, with cells lacking curves, repeated rows, negative values, partial rows."""
    curves = RatingCurves.empty(cells, INCREMENTS)
    for k in range(cells):
        comid = 1000 + k // 12
        base = 100.0 + rng.uniform(0, 30)
        curves.metadata[k] = [comid, rng.integers(0, 500), rng.integers(0, 500), base, rng.uniform(0.5, 20),
                              rng.uniform(1e-4, 0.02), rng.uniform(0, math.pi)]
        q_top = rng.uniform(5, 500)
        depth = np.linspace(0.3, 1.0, INCREMENTS) * rng.uniform(1, 6)
        q = q_top * (depth / depth[-1]) ** rng.uniform(1.3, 2.2)
        top = rng.uniform(5, 30) * depth ** rng.uniform(0.2, 0.6)
        area = top * depth * rng.uniform(0.5, 0.8)
        curves.increments[k] = np.column_stack([q, q / area, top, base + depth, top + 2 * depth])
    curves.metadata[3] = np.nan  # no rating curve
    curves.increments[3] = np.nan
    curves.increments[5, 2, 0] = np.nan  # an incomplete one
    curves.increments[7, 1, 1] = -0.5  # a negative velocity
    curves.metadata[9] = curves.metadata[8]  # a repeat of the cell before
    curves.increments[9] = curves.increments[8]
    return curves


DROPPED = [3, 5, 7, 9]  # random_curves' cells that the VDT and AP databases leave out


def power_law_curves(coefficients: dict[int, tuple], cells_per_reach: int = 3, extra_cells: int = 0) -> RatingCurves:
    """Rating curves whose depths, top widths and velocities are exact power laws of the discharge, with each
    reach's coefficients (depth_a, depth_b, tw_a, tw_b, vel_a, vel_b) and each cell's bed at its own elevation.
    The extra cells at the end are left empty."""
    curves = RatingCurves.empty(len(coefficients) * cells_per_reach + extra_cells, INCREMENTS)
    q = np.linspace(2.0, 60.0, INCREMENTS)
    k = 0
    for comid, (depth_a, depth_b, tw_a, tw_b, vel_a, vel_b) in coefficients.items():
        for cell in range(cells_per_reach):
            bed = 100.0 + 3.0 * cell
            curves.metadata[k] = [comid, 10 + k, 20 + k, bed, 0.0, 0.001, 1.0]
            top_width = tw_a * q ** tw_b
            curves.increments[k] = np.column_stack([q, vel_a * q ** vel_b, top_width, bed + depth_a * q ** depth_b,
                                                    top_width + 1.0])
            k += 1
    return curves


COEFFICIENTS = {1000: (0.3, 0.45, 12.0, 0.3, 0.8, 0.25), 1001: (0.2, 0.5, 20.0, 0.35, 1.1, 0.2)}


def test_the_vdt_database_keeps_each_complete_rating_curve_once(tmp_path: Path) -> None:
    """Without the wetted perimeters, and without the cells that have no curve, an incomplete one, a negative value,
    or the same values as the row before."""
    curves = random_curves(np.random.default_rng(1))

    written = write_vdt(curves, tmp_path / "vdt.csv")

    assert written.index.tolist() == [k for k in range(60) if k not in DROPPED]
    assert list(written.columns) == ['COMID', 'Row', 'Col', 'Elev', 'QBaseflow', 'Slope', 'XS_Angle'] + \
        [f"{field}_{i}" for i in range(1, INCREMENTS + 1) for field in ("q", "v", "t", "wse")]
    for column in written.columns:
        decimals = {"Slope": 8, "XS_Angle": 7}.get(column, 3)
        assert np.array_equal(written[column], written[column].round(decimals)), column
    pd.testing.assert_frame_equal(pd.read_csv(tmp_path / "vdt.csv"), written.reset_index(drop=True),
                                  check_dtype=False)


def test_the_vdt_database_s_text(tmp_path: Path) -> None:
    curves = RatingCurves.empty(1, 2)
    curves.metadata[0] = [1001, 4, 5, 99.5, 2.5, 0.0012345678912, 1.23456789123]
    curves.increments[0] = [[1.23456, 0.5, 10.0, 100.1234, 11.0], [5.0, 0.75, 12.5, 100.5, 13.0]]

    write_vdt(curves, tmp_path / "vdt.csv")

    assert (tmp_path / "vdt.csv").read_text().splitlines() == [
        "COMID,Row,Col,Elev,QBaseflow,Slope,XS_Angle,q_1,v_1,t_1,wse_1,q_2,v_2,t_2,wse_2",
        "1001,4,5,99.5,2.5,0.00123457,1.2345679,1.235,0.5,10.0,100.123,5.0,0.75,12.5,100.5"]


def test_valley_slopes_take_a_column_of_the_vdt_database_but_not_of_the_ap_database(tmp_path: Path) -> None:
    """With Valley_Sections the VDT database gives each cell's valley slope after XS_Angle, rounded to 8 decimals as
    Slope is."""
    curves = RatingCurves.empty(1, 2, valley_slopes=True)
    curves.metadata[0] = [1001, 4, 5, 99.5, 2.5, 0.0012345678912, 1.23456789123]
    curves.valley_slopes[0] = 0.0018765432109
    curves.increments[0] = [[1.23456, 0.5, 10.0, 100.1234, 11.0], [5.0, 0.75, 12.5, 100.5, 13.0]]

    write_vdt(curves, tmp_path / "vdt.csv")
    ap = write_ap(curves, tmp_path / "ap.csv")

    assert (tmp_path / "vdt.csv").read_text().splitlines() == [
        "COMID,Row,Col,Elev,QBaseflow,Slope,XS_Angle,Valley_Slope,q_1,v_1,t_1,wse_1,q_2,v_2,t_2,wse_2",
        "1001,4,5,99.5,2.5,0.00123457,1.2345679,0.00187654,1.235,0.5,10.0,100.123,5.0,0.75,12.5,100.5"]
    assert list(ap.columns) == ['COMID', 'Row', 'Col', 'q_1', 'a_1', 'p_1', 'q_2', 'a_2', 'p_2']


def test_the_vdt_database_can_be_parquet(tmp_path: Path) -> None:
    curves = random_curves(np.random.default_rng(2))

    write_vdt(curves, tmp_path / "vdt.parquet")

    pd.testing.assert_frame_equal(pd.read_parquet(tmp_path / "vdt.parquet"),
                                  vdt_dataframe(curves).reset_index(drop=True))


def test_parquet_tables_take_a_dictionary_only_where_values_repeat(tmp_path: Path) -> None:
    """Brotli, and dictionary encoding only on the columns where under 30% of the values differ."""
    rng = np.random.default_rng(12)
    df = pd.DataFrame({"COMID": np.repeat([7, 8, 9], 100), "Row": np.arange(300),
                       "v_1": rng.choice([0.5, 0.75, 1.125], 300), "wse_1": np.round(rng.uniform(100, 200, 300), 3)})

    write_table(df, tmp_path / "table.parquet")

    row_group = pq.ParquetFile(tmp_path / "table.parquet").metadata.row_group(0)
    columns = {row_group.column(i).path_in_schema: row_group.column(i) for i in range(row_group.num_columns)}
    assert {name for name, column in columns.items() if 'RLE_DICTIONARY' in column.encodings} == {"COMID", "v_1"}
    assert {column.compression for column in columns.values()} == {'BROTLI'}
    pd.testing.assert_frame_equal(pd.read_parquet(tmp_path / "table.parquet"), df)


def test_the_ap_database_has_each_increment_s_discharge_area_and_wetted_perimeter(tmp_path: Path) -> None:
    """The area is the discharge over the velocity, or 0 where there's no velocity."""
    curves = random_curves(np.random.default_rng(3))
    curves.increments[11, 2, V] = 0.0

    written = write_ap(curves, tmp_path / "ap.csv")

    assert written.index.tolist() == [k for k in range(60) if k not in DROPPED]
    assert list(written.columns) == ['COMID', 'Row', 'Col'] + \
        [f"{field}_{i}" for i in range(1, INCREMENTS + 1) for field in ("q", "a", "p")]
    q, v, p = (curves.increments[11, :, field] for field in (Q, V, P))
    row = written.loc[11]
    assert row[[f"q_{i}" for i in range(1, INCREMENTS + 1)]].tolist() == np.round(q, 3).tolist()
    assert row[[f"a_{i}" for i in range(1, INCREMENTS + 1)]].tolist() == \
        np.round(np.divide(q, v, out=np.zeros_like(q), where=v != 0), 3).tolist()
    assert row[[f"p_{i}" for i in range(1, INCREMENTS + 1)]].tolist() == np.round(p, 3).tolist()
    assert row["a_3"] == 0
    pd.testing.assert_frame_equal(pd.read_csv(tmp_path / "ap.csv"), written.reset_index(drop=True),
                                  check_dtype=False)


def test_the_curve_file_fits_each_cell_s_power_laws(tmp_path: Path) -> None:
    curves = power_law_curves(COEFFICIENTS)
    qmax = pd.Series({1000: 75.0, 1001: 90.0})

    written = write_curve_file(curves, qmax, tmp_path / "curve.csv")

    assert list(written.columns) == ['COMID', 'Row', 'Col', 'BaseElev', 'QMax', 'Slope', 'XS_Angle', *FITTED]
    assert written["Row"].tolist() == [10, 11, 12, 13, 14, 15]
    assert written["BaseElev"].tolist() == [100.0, 103.0, 106.0] * 2  # each cell's bed, the elevation in the VDT
    for _, row in written.iterrows():
        assert tuple(row[FITTED]) == pytest.approx(COEFFICIENTS[row["COMID"]], abs=1e-9)
        assert row["QMax"] == qmax[row["COMID"]]
    pd.testing.assert_frame_equal(pd.read_csv(tmp_path / "curve.csv"), written, check_dtype=False)


def test_the_curve_file_leaves_out_cells_without_metadata_or_a_positive_fit(tmp_path: Path) -> None:
    curves = random_curves(np.random.default_rng(4))
    curves.metadata[11, 3] = 200.0  # a bed above the water, so the depths are negative and so is depth_a
    qmax = pd.Series({1000 + k: 100.0 + k for k in range(5)})

    written = write_curve_file(curves, qmax, tmp_path / "curve.csv")

    cells = list(zip(curves.metadata[:, 1], curves.metadata[:, 2]))
    assert list(zip(written["Row"], written["Col"])) == [cell for k, cell in enumerate(cells) if k not in (3, 11)]
    assert (written[["depth_a", "tw_a", "vel_a"]] > 0).all().all()


def test_the_reach_average_curve_file_fits_each_reach_s_power_laws(tmp_path: Path) -> None:
    """Fitted to all of a reach's rating curves in the VDT database, and given to each of its cells, those without a
    rating curve too."""
    curves = power_law_curves(COEFFICIENTS, extra_cells=1)
    curves.metadata[6] = [1000, 7, 8, 101.0, np.nan, 0.001, 1.0]  # a cell without a rating curve
    qmax = pd.Series({1000: 75.0, 1001: 90.0})

    written = write_reach_average_curve_file(curves, vdt_dataframe(curves), qmax, tmp_path / "curve.csv")

    assert sorted(written["Row"]) == [7, 10, 11, 12, 13, 14, 15]
    for _, row in written.iterrows():
        # the VDT database's values are rounded to 3 decimals, so the fit is only close to the coefficients
        assert tuple(row[FITTED]) == pytest.approx(COEFFICIENTS[row["COMID"]], abs=2e-3)
        assert row["QMax"] == qmax[row["COMID"]]
    pd.testing.assert_frame_equal(pd.read_csv(tmp_path / "curve.csv"), written.reset_index(drop=True),
                                  check_dtype=False)


@pytest.mark.parametrize("failure", ["not converging", "a negative depth coefficient"])
def test_the_reach_average_curve_file_leaves_out_reaches_without_a_positive_fit(tmp_path: Path, monkeypatch,
                                                                               failure: str) -> None:
    """As the curve file leaves out cells: a reach whose fit doesn't converge (-9999.9), or gives a coefficient that
    isn't positive."""
    from arc.outputs import curve_files
    curves = power_law_curves(COEFFICIENTS)
    if failure == "not converging":
        fit = curve_files.fit_power_law

        def failing(x, y, init_guess=(1.0, 1.0)):
            if np.allclose(y, 20.0 * x ** 0.35, atol=1e-3):  # reach 1001's top widths
                return curve_files.FAILED_FIT, curve_files.FAILED_FIT, curve_files.FAILED_FIT
            return fit(x, y, init_guess)

        monkeypatch.setattr(curve_files, "fit_power_law", failing)
    else:
        curves.metadata[curves.metadata[:, 0] == 1001, 3] = 1000.0  # beds above the water, so the depths are negative
    qmax = pd.Series({1000: 75.0, 1001: 90.0})

    written = write_reach_average_curve_file(curves, vdt_dataframe(curves), qmax, tmp_path / "curve.csv")

    assert written["COMID"].tolist() == [1000, 1000, 1000]
    assert (written[["depth_a", "tw_a", "vel_a"]] > 0).all().all()


# --- The cross-section file ------------------------------------------------------------------------------------


def random_records(rng, count: int = 40) -> list[CrossSectionRecord]:
    records = []
    for k in range(count):
        side1, side2 = int(rng.integers(1, 90)), int(rng.integers(1, 90))
        profile1 = 150.0 + np.cumsum(rng.uniform(-0.2, 1.5, side1))
        profile2 = np.concatenate([[profile1[0]], 150.0 + np.cumsum(rng.uniform(-0.2, 1.5, side2 - 1))]) \
            if side2 > 1 else profile1[:1].copy()
        n1 = rng.choice([0.03, 0.035, 0.12], side1).astype(np.float32).astype(np.float64)
        n2 = rng.choice([0.03, 0.035, 0.12], side2).astype(np.float32).astype(np.float64)
        records.append(CrossSectionRecord(1000 + k, int(rng.integers(0, 900)), int(rng.integers(0, 900)), profile1,
                                          float(rng.uniform(20, 35)), n1, profile2, n2, int(rng.integers(0, 900)),
                                          int(rng.integers(0, 900)), int(rng.integers(0, 900)),
                                          int(rng.integers(0, 900))))
    return records


def test_the_cross_section_file_is_tab_separated_with_the_arrays_as_numpy_prints_them(tmp_path: Path) -> None:
    records = random_records(np.random.default_rng(6))

    write_cross_sections(records + [None], tmp_path / "xs.txt")

    lines = (tmp_path / "xs.txt").read_text().splitlines()
    assert lines[0].split("\t") == XS_EXPORT_COLUMNS
    assert len(lines) == len(records) + 1  # a cell without a cross section has no row
    for line, record in zip(lines[1:], records):
        for text, value in zip(line.split("\t"), record):
            assert text == (format_array(value) if isinstance(value, np.ndarray) else str(value))


def test_an_empty_cross_section_file_has_only_its_header(tmp_path: Path) -> None:
    write_cross_sections([], tmp_path / "xs.txt")

    assert (tmp_path / "xs.txt").read_text().splitlines() == ["\t".join(XS_EXPORT_COLUMNS)]


ARRAY_COLUMNS = ('XS1_Profile', 'Manning_N_Raster1', 'XS2_Profile', 'Manning_N_Raster2')


def test_the_cross_section_file_can_be_parquet(tmp_path: Path) -> None:
    """In Parquet the profiles and n are lists of float64, the records' arrays exactly, which the text file prints."""
    records = random_records(np.random.default_rng(6))
    text = write_cross_sections(records + [None], tmp_path / "xs.txt")

    returned = write_cross_sections(records + [None], tmp_path / "xs.parquet")

    df = pd.read_parquet(tmp_path / "xs.parquet")
    assert list(df.columns) == list(returned.columns) == XS_EXPORT_COLUMNS
    assert len(df) == len(returned) == len(records)
    for column, field in zip(XS_EXPORT_COLUMNS, CrossSectionRecord._fields):
        expected = [getattr(record, field) for record in records]
        if column in ARRAY_COLUMNS:
            for value, array in zip(df[column], expected):
                assert value.dtype == np.float64 and np.array_equal(value, array)
            assert [format_array(value) for value in df[column]] == list(text[column])
        else:
            assert df[column].dtype == (np.float64 if column == 'Ordinate_Dist' else np.int64)
            assert df[column].tolist() == expected


def test_an_empty_cross_section_file_can_be_parquet(tmp_path: Path) -> None:
    write_cross_sections([], tmp_path / "xs.parquet")

    schema = pq.read_schema(tmp_path / "xs.parquet")
    assert schema.names == XS_EXPORT_COLUMNS
    for column in ARRAY_COLUMNS:
        assert pa.types.is_list(schema.field(column).type) and schema.field(column).type.value_type == pa.float64()
    assert pd.read_parquet(tmp_path / "xs.parquet").empty


def test_parquet_cross_sections_split_the_elevations_bytes_and_keep_a_dictionary_of_n(tmp_path: Path) -> None:
    """The layout that makes the file small. pyarrow silently ignores a list column named without its
    '.list.element' path, which would undo it."""
    write_cross_sections(random_records(np.random.default_rng(9)), tmp_path / "xs.parquet")

    row_group = pq.ParquetFile(tmp_path / "xs.parquet").metadata.row_group(0)
    columns = {row_group.column(i).path_in_schema: row_group.column(i) for i in range(row_group.num_columns)}
    for side in (1, 2):
        assert 'BYTE_STREAM_SPLIT' in columns[f'XS{side}_Profile.list.element'].encodings
        assert 'RLE_DICTIONARY' in columns[f'Manning_N_Raster{side}.list.element'].encodings
    assert {column.compression for column in columns.values()} == {'ZSTD'}


def _array2string(values):
    return np.array2string(values, precision=6, max_line_width=np.inf, threshold=np.inf, floatmode='fixed')


@pytest.mark.parametrize("values", [
    np.array([170.472009, 171.74819, 99999.9]), np.array([-3.5, 12.25, 0.0, -0.0]), np.array([]),
    np.array([0.035, 0.12], dtype=np.float32), np.array([np.nan, 1.5, np.inf, -np.inf]), np.array([np.nan, np.nan]),
    np.array([0.00001, 250.0]), np.array([1.0, 1e9]), np.array([0.5, 600.0]), np.array([1e7], dtype=np.float32),
    np.array([3, 4]), np.array([[1.5, 2.5]])])
def test_arrays_print_as_numpy_prints_them(values: np.ndarray) -> None:
    assert format_array(values) == _array2string(values)


def test_arrays_print_as_numpy_prints_them_at_random() -> None:
    rng = np.random.default_rng(7)
    for _ in range(5000):
        size = int(rng.integers(1, 60))
        values = rng.uniform(-50, 3000, size) if rng.random() < 0.7 else 10 ** rng.uniform(-6, 9, size)
        if rng.random() < 0.3:
            values = np.round(values, int(rng.integers(0, 7)))
        if rng.random() < 0.2:
            values = values.astype(np.float32)
        assert format_array(values) == _array2string(values)


# --- Representative cross sections ------------------------------------------------------------------------------


def channel_sample(comid: int, half: int, spacing: float, bed: float, slope: float) -> RepresentativeSample:
    """A flat bed `half` spacings either side of the stream cell, between walls that rise 30 m over one spacing."""
    elevations = np.full(2 * half + 41, bed + 30.0)
    center = elevations.size // 2
    elevations[center - half:center + half + 1] = bed
    return RepresentativeSample(comid, XSection(elevations, np.full(elevations.size, 0.035), spacing), bed, slope)


def flat_bed_sample(rng, comid: int) -> RepresentativeSample:
    """channel_sample at a random width, spacing, bed elevation and slope."""
    return channel_sample(comid, int(rng.integers(3, 10)), float(rng.uniform(5, 15)),
                          100.0 + float(np.round(rng.uniform(-2, 2), 3)), float(rng.uniform(0.0005, 0.005)))


def test_cross_sections_alike_average_to_their_own_manning_hydraulics() -> None:
    """Three copies of a 40 m wide flat bed whose walls rise 30 m over a 10 m spacing: at each 0.1 m stage the water
    is d deep over the bed and spreads d / 3 m up each wall."""
    samples = [channel_sample(7, 2, 10.0, 100.0, 0.002) for _ in range(3)]

    df = representative_cross_section_dataframe(samples)

    d = 0.1 * np.arange(1, 251)
    spread = d / 3.0
    area = 40.0 * d + d * spread
    perimeter = 40.0 + 2.0 * np.hypot(d, spread)
    discharge = area * (area / perimeter) ** (2 / 3) * math.sqrt(0.002) / 0.035
    assert list(df.columns) == REPRESENTATIVE_CROSS_SECTION_COLUMNS
    assert df["Depth_Stage_Meters"].to_numpy() == pytest.approx(d, rel=1e-12)
    assert (df["Cross_Section_Count"] == 3).all() and (df["Hydraulic_Sample_Count"] == 3).all()
    assert df["Mean_Cross_Sectional_Area"].to_numpy() == pytest.approx(area, rel=1e-12)
    assert df["Mean_Top_Width"].to_numpy() == pytest.approx(40.0 + 2.0 * spread, rel=1e-12)
    assert df["Mean_Discharge"].to_numpy() == pytest.approx(discharge, rel=1e-12)
    assert df["Mean_Velocity"].to_numpy() == pytest.approx(discharge / area, rel=1e-12)
    assert df["Mean_WSE"].to_numpy() == pytest.approx(100.0 + d, rel=1e-12)
    assert (df["Stream_Slope"] == 0.002).all() and (df["Representative_Thalweg_Elevation"] == 100.0).all()


def test_the_representative_shape_grows_its_width_linearly_between_stages_to_hold_each_area() -> None:
    """From no width at the thalweg, each stage's depth increment is the one that holds the stage's extra area in a
    trapezoid between the stage below's top width and its own."""
    rng = np.random.default_rng(9)
    samples = [flat_bed_sample(rng, 9) for _ in range(5)]

    df = representative_cross_section_dataframe(samples)

    area = df["Representative_Cross_Sectional_Area"].to_numpy()
    width = df["Representative_Top_Width"].to_numpy()
    increment = df["Representative_Depth_Increment"].to_numpy()
    depth = df["Representative_Depth"].to_numpy()
    assert np.all(np.diff(area) >= 0.0) and np.all(np.diff(width) >= 0.0) and np.all(np.diff(depth) >= 0.0)
    previous_width = previous_area = previous_depth = 0.0
    for i in range(len(df)):
        assert area[i] - previous_area == pytest.approx(0.5 * (previous_width + width[i]) * increment[i], abs=1e-9)
        assert depth[i] == pytest.approx(previous_depth + increment[i], abs=1e-12)
        previous_width, previous_area, previous_depth = width[i], area[i], depth[i]
    assert df["Representative_Left_Station"].tolist() == pytest.approx((-0.5 * width).tolist())
    assert df["Representative_Right_Station"].tolist() == pytest.approx((0.5 * width).tolist())
    assert df["Representative_Stage_Elevation"].tolist() == \
        pytest.approx((df["Representative_Thalweg_Elevation"] + depth).tolist())


def test_cross_sections_without_a_slope_count_but_are_not_averaged() -> None:
    samples = [channel_sample(11, 2, 10.0, 100.0, 0.001), channel_sample(11, 2, 10.0, 100.0, 0.0)]

    df = representative_cross_section_dataframe(samples)

    assert (df["Cross_Section_Count"] == 2).all() and (df["Hydraulic_Sample_Count"] == 1).all()
    assert (df["Stream_Slope"] == 0.001).all()
    pd.testing.assert_frame_equal(df.drop(columns="Cross_Section_Count"),
                                  representative_cross_section_dataframe(samples[:1]).drop(
                                      columns="Cross_Section_Count"))


def test_representative_cross_sections_use_the_channel_s_velocity_where_there_are_banks() -> None:
    rng = np.random.default_rng(10)
    sample = flat_bed_sample(rng, 7)
    sample.xs.left_bank_distance = sample.xs.right_bank_distance = sample.xs.ordinate_distance

    banked = representative_cross_section_dataframe([sample])
    sample.xs.left_bank_distance = sample.xs.right_bank_distance = -1.0
    unbanked = representative_cross_section_dataframe([sample])

    assert (banked["Mean_Discharge"] >= 0).all()
    assert not np.allclose(banked["Mean_Velocity"], unbanked["Mean_Velocity"])
    assert np.allclose(unbanked["Mean_Velocity"], unbanked["Mean_Discharge"] / unbanked["Mean_Cross_Sectional_Area"])


def test_representative_cross_sections_use_a_carved_channel_s_own_shape() -> None:
    """A 4 m channel carved into 10 m cells: at each stage the area is its profile's, not the ordinates'."""
    elevations = 100.0 + np.array([6.0, 4.0, 2.0, 0.0, 2.0, 4.0, 6.0])
    xs = XSection(elevations.copy(), np.full(7, 0.035), 10.0)
    carve_channel(xs, find_banks(xs, target_width=4.0), 1.0, trapezoid_height=0.2, bank_elevation=101.0)
    sample = RepresentativeSample(8, xs, float(xs.elevations[3]), 0.001)

    df = representative_cross_section_dataframe([sample])

    for _, stage in df[df["Depth_Stage_Meters"] <= 1.0 + 1e-9].iterrows():
        assert stage["Mean_Cross_Sectional_Area"] == pytest.approx(
            hydraulic_geometry(xs, depth=stage["Depth_Stage_Meters"]).area, rel=1e-9)
        assert stage["Mean_Top_Width"] <= 4.0 + 1e-9  # within the channel, narrower than a cell


def test_representative_cross_sections_are_written_rounded_to_6_decimals(tmp_path: Path) -> None:
    rng = np.random.default_rng(11)
    samples = [flat_bed_sample(rng, 3) for _ in range(3)]
    df = representative_cross_section_dataframe(samples)

    write_representative_cross_sections(df, tmp_path / "representative.csv")
    back = pd.read_csv(tmp_path / "representative.csv")

    assert np.allclose(back["Mean_Discharge"], df["Mean_Discharge"].round(6))
    write_representative_cross_sections(None, tmp_path / "empty.csv")
    assert pd.read_csv(tmp_path / "empty.csv").empty


def test_representative_cross_sections_can_be_parquet(tmp_path: Path) -> None:
    rng = np.random.default_rng(11)
    samples = [flat_bed_sample(rng, 3) for _ in range(3)]
    df = representative_cross_section_dataframe(samples)

    write_representative_cross_sections(df, tmp_path / "representative.csv")
    written = write_representative_cross_sections(df, tmp_path / "representative.parquet")

    back = pd.read_parquet(tmp_path / "representative.parquet")
    pd.testing.assert_frame_equal(back, written)
    pd.testing.assert_frame_equal(back, pd.read_csv(tmp_path / "representative.csv"), check_exact=False, rtol=1e-12)
    write_representative_cross_sections(None, tmp_path / "empty.parquet")
    empty = pd.read_parquet(tmp_path / "empty.parquet")
    assert empty.empty and list(empty.columns) == REPRESENTATIVE_CROSS_SECTION_COLUMNS
    assert empty["COMID"].dtype == np.int64 and empty["Mean_Discharge"].dtype == np.float64


def test_a_reach_stops_at_the_first_stage_whose_hydraulics_are_not_finite() -> None:
    """Water reaching a NaN in one cross section ends its reach: the stages before stand, and say where it ended."""
    rng = np.random.default_rng(12)
    samples = [flat_bed_sample(rng, 4) for _ in range(3)] + [flat_bed_sample(rng, 5)]
    elevations, bed = samples[1].xs.elevations, samples[1].thalweg
    center = elevations.size // 2
    wall = center + int(np.argmax(elevations[center:] > bed))
    elevations[wall:] = bed + 1.25  # a right overbank 1.25 m up, to a NaN at its end
    elevations[-1] = np.nan

    df = representative_cross_section_dataframe(samples)

    ended = df[df["COMID"] == 4]
    assert ended["Depth_Stage_Index"].tolist() == list(range(1, 13))  # at 1.3 m the water reaches the NaN
    assert (ended["Reach_Inflect_Terrace_Depth"] == ended["Depth_Stage_Meters"].iloc[-1]).all()
    assert (df["COMID"] == 5).sum() == 250


def stage_wse(sample: RepresentativeSample, stages) -> np.ndarray:
    """The water surfaces of stages, as the representative cross sections work them out."""
    return np.array([sample.thalweg + stage * DEPTH_INCREMENT for stage in stages])


def test_a_rating_curve_at_normal_depth_is_normal_depth_at_its_water_surfaces() -> None:
    """A cross section whose rating curve is its own normal depth carries normal depth's discharge, at its own slope,
    at each of the curve's water surfaces and beyond its ends. Between them its discharge is the curve's, linear in the
    water surface as the VDT database is read: the chord above Manning's discharge, which curves upward."""
    sample = channel_sample(7, 3, 10.0, 100.0, 0.002)
    plain = representative_cross_section_dataframe([sample]).set_index("Depth_Stage_Index")
    stages = np.arange(5, 120, 10)
    curve = np.column_stack([stage_wse(sample, stages), plain.loc[stages, "Mean_Discharge"]])

    linked = representative_cross_section_dataframe([sample._replace(curve=curve)]).set_index("Depth_Stage_Index")

    assert (plain["Mean_Friction_Slope"] == 0.002).all()
    within = (linked.index >= stages[0]) & (linked.index <= stages[-1])
    outside = linked.index[~within].union(stages)
    pd.testing.assert_frame_equal(linked.loc[outside], plain.loc[outside], check_exact=False, rtol=1e-12)
    chord = np.interp(linked.loc[within, "Mean_WSE"], curve[:, 0], curve[:, 1])
    assert linked.loc[within, "Mean_Discharge"].to_numpy() == pytest.approx(chord, rel=1e-12)
    assert np.all(linked.loc[within, "Mean_Discharge"] >= plain.loc[within, "Mean_Discharge"] * (1 - 1e-12))
    # The friction slope is the one the discharge implies; the cross section's hydraulics are its own
    assert linked["Mean_Friction_Slope"].to_numpy() == pytest.approx(
        0.002 * (linked["Mean_Discharge"] / plain["Mean_Discharge"]).to_numpy() ** 2, rel=1e-12)
    for column in ("Mean_Cross_Sectional_Area", "Mean_Top_Width", "Mean_WSE", "Representative_Depth"):
        np.testing.assert_array_equal(linked[column], plain[column])


def test_a_standard_step_curve_s_flows_are_the_discharges_at_its_water_surfaces() -> None:
    """Water 0.5 m above normal depth: at each of its rating curve's water surfaces, the cross section carries the
    curve's flow, at the friction slope (Q / K)^2, below its own slope; between them, the curve's flow linear in the
    water surface, at the friction slope that implies; below and above the curve, the friction slope at its end."""
    sample = channel_sample(7, 3, 10.0, 100.0, 0.002)
    plain = representative_cross_section_dataframe([sample]).set_index("Depth_Stage_Index")
    normal = np.arange(5, 105, 10)  # the curve's flows are normal depth's at these stages...
    raised = normal + 5  # ...and its water surfaces 0.5 m higher
    flows = plain.loc[normal, "Mean_Discharge"].to_numpy()

    linked = representative_cross_section_dataframe(
        [sample._replace(curve=np.column_stack([stage_wse(sample, raised), flows]))]).set_index("Depth_Stage_Index")

    assert linked.loc[raised, "Mean_Discharge"].to_numpy() == pytest.approx(flows, rel=1e-12)
    friction = 0.002 * (flows / plain.loc[raised, "Mean_Discharge"].to_numpy()) ** 2
    assert np.all(friction < 0.002)
    assert linked.loc[raised, "Mean_Friction_Slope"].to_numpy() == pytest.approx(friction, rel=1e-12)
    between = linked.loc[raised[0] + 3, "Mean_Discharge"]
    assert between == pytest.approx(flows[0] + 0.3 * (flows[1] - flows[0]), rel=1e-12)
    assert linked.loc[raised[0] + 3, "Mean_Friction_Slope"] == pytest.approx(
        0.002 * (between / plain.loc[raised[0] + 3, "Mean_Discharge"]) ** 2, rel=1e-12)
    below, above = np.arange(1, raised[0]), np.arange(raised[-1] + 1, 251)
    assert linked.loc[below, "Mean_Friction_Slope"].to_numpy() == pytest.approx(friction[0], rel=1e-12)
    assert linked.loc[above, "Mean_Discharge"].to_numpy() == pytest.approx(
        plain.loc[above, "Mean_Discharge"].to_numpy() * math.sqrt(friction[-1] / 0.002), rel=1e-12)
    for column in ("Mean_Velocity", "Mean_Section_Velocity"):  # without banks, the channel's is the section's
        assert linked[column].to_numpy() == pytest.approx(
            (linked["Mean_Discharge"] / linked["Mean_Cross_Sectional_Area"]).to_numpy(), rel=1e-12)
    for column in ("Mean_Cross_Sectional_Area", "Mean_Top_Width", "Mean_WSE", "Representative_Depth"):
        np.testing.assert_array_equal(linked[column], plain[column])


def test_a_cross_section_s_discharge_never_falls_as_its_stage_rises() -> None:
    """A 40 m channel a metre deep opens onto a 600 m flat overbank: just over its banks the wetted perimeter jumps and
    normal depth's discharge would fall, so the discharge below is held there, at the friction slope that implies."""
    elevations = np.full(81, 130.0)
    elevations[10:71] = 101.0
    elevations[38:43] = 100.0
    sample = RepresentativeSample(9, XSection(elevations, np.full(elevations.size, 0.035), 10.0), 100.0, 0.002)

    df = representative_cross_section_dataframe([sample]).set_index("Depth_Stage_Index")

    rises = np.diff(df["Mean_Discharge"].to_numpy())
    assert np.all(rises >= 0.0)
    held = df.index[1:][rises == 0.0]
    assert held.size > 0
    assert np.all(df.loc[held, "Mean_Friction_Slope"] > 0.002)
    assert (df.drop(held)["Mean_Friction_Slope"] == 0.002).all()


def test_of_water_held_at_one_level_the_highest_flow_counts() -> None:
    """Water held at the baseflow level over two flows carries the higher; a water surface below the one before, or
    with no flow, is left out."""
    sample = channel_sample(7, 3, 10.0, 100.0, 0.002)
    curve = np.column_stack([stage_wse(sample, [10, 10, 8, 20, 20]), [1.0, 2.0, 3.0, np.nan, 4.0]])

    linked = representative_cross_section_dataframe([sample._replace(curve=curve)]).set_index("Depth_Stage_Index")

    assert linked.loc[[10, 20], "Mean_Discharge"].tolist() == pytest.approx([2.0, 4.0], rel=1e-12)
    assert linked.loc[8, "Mean_Friction_Slope"] == linked.loc[10, "Mean_Friction_Slope"]


def test_cross_sections_without_a_rating_curve_stay_at_normal_depth_beside_those_with_one() -> None:
    plain_sample = channel_sample(7, 3, 10.0, 100.0, 0.002)
    plain = representative_cross_section_dataframe([plain_sample])
    curve = np.column_stack([stage_wse(plain_sample, [10, 20]), [5.0, 20.0]])
    linked_alone = representative_cross_section_dataframe([plain_sample._replace(curve=curve)])

    both = representative_cross_section_dataframe([plain_sample, plain_sample._replace(curve=curve)])

    for column in ("Mean_Discharge", "Mean_Friction_Slope"):
        assert both[column].to_numpy() == pytest.approx(0.5 * (plain[column] + linked_alone[column]).to_numpy(),
                                                        rel=1e-12)


@pytest.mark.parametrize("count", [0, 1, 7, 8, 9, 127, 128, 129, 255, 256, 257, 1000, 4099])
def test_representative_means_sum_as_numpy_does(count: int) -> None:
    """The compiled stages' sums are numpy's to the bit, so their means and deviations are numpy's too."""
    from arc.outputs.representative import _sum

    rng = np.random.default_rng(count)
    values = rng.standard_normal(count + 3) * 10.0 ** rng.integers(-3, 6, count + 3)

    assert _sum(values, 3, count) == np.sum(values[3:])


@pytest.mark.filterwarnings("ignore:Covariance of the parameters could not be estimated")
def test_the_power_law_fits_are_curve_fit_s() -> None:
    """The curve file's fits, which call MINPACK's lmdif as scipy's curve_fit does, give curve_fit's coefficients bit
    for bit, and fail where it fails: on noisy power laws, flat lines, and data no power law fits."""
    from scipy.optimize import curve_fit

    from arc.outputs.curve_files import FAILED_FIT, _power, fit_power_law
    rng = np.random.default_rng(68)
    failed = 0
    for _ in range(400):
        size = int(rng.integers(2, 31))
        q = np.sort(rng.uniform(0.1, 500.0, size))
        y = float(rng.uniform(0.1, 50.0)) * q ** float(rng.uniform(-0.5, 1.0)) * rng.normal(1.0, 0.05, size)
        if rng.random() < 0.1:
            y = np.full(size, 3.0)
        if rng.random() < 0.1:
            y = rng.normal(0.0, 1.0, size)
        guess = tuple(float(g) for g in rng.uniform(0.1, 2.0, 2))
        try:
            expected = tuple(curve_fit(_power, q, y, p0=list(guess))[0])
        except RuntimeError:
            expected = (FAILED_FIT, FAILED_FIT)
            failed += 1
        got = fit_power_law(q, y, guess)[:2]
        assert np.array_equal(np.array(got).view(np.uint64), np.array(expected, dtype=np.float64).view(np.uint64))
    assert failed  # some fits fail
