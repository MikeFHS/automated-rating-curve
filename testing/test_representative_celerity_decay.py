"""Celerity K coefficient generation, length configuration, and CSV export."""

from unittest.mock import patch

import numpy as np
import pandas as pd
import unittest
import tempfile
from pathlib import Path

from arc.Automated_Rating_Curve_Generator import read_main_input_file
from arc.hydraulic_data import (
    HydraulicData, REPRESENTATIVE_CROSS_SECTION_COLUMNS,
    REPRESENTATIVE_CROSS_SECTION_COEFFICIENT_COLUMNS,
    representative_cross_section_coefficients_path,
    build_representative_cross_section_dataframe, fit_celerity_k_power_law,
)


def config(**overrides):
    return {"Build_Representative_Cross_Section": True,
            "Representative_Cross_Section_File": "representative.csv",
            "StrmShp_File": "streams.gpkg", "Stream_Length_m": "custom_length",
            **overrides}


def staged_rows(comid, length, amplitude, exponent, q=None):
    if q is None:
        q = np.array([1., 5., 10., 25., 50., 100.])
    velocity = length / ((5. / 3.) * amplitude * q ** exponent)
    return [{"COMID": comid, "Cross_Section_Count": 1, "Hydraulic_Sample_Count": 1,
             "Depth_Stage_Index": i, "Depth_Stage_Meters": i * .1,
             "Stream_Slope": .001, "Reach_Inflect_Terrace_Depth": .6,
             "Representative_Thalweg_Elevation": 0., "Mean_Discharge": discharge,
             "Mean_Depth": i * .1, "Mean_Velocity": v,
             "Mean_Top_Width": 10., "Mean_Cross_Sectional_Area": i,
             "Mean_WSE": i * .1}
            for i, (discharge, v) in enumerate(zip(q, velocity), 1)]


class RepresentativeCelerityPowerLawTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.tmp_path = Path(directory.name)

    def test_length_argument_required_only_for_representative_output(self):
        tmp_path = self.tmp_path
        for value in ("", "   ", None):
            with self.assertRaisesRegex(ValueError, "requires Stream_Length_m"):
                read_main_input_file("", config(Stream_Length_m=value))
        missing_length = config()
        del missing_length["Stream_Length_m"]
        with self.assertRaisesRegex(ValueError, "requires Stream_Length_m"):
            read_main_input_file("", missing_length)
        with self.assertRaisesRegex(ValueError, "requires StrmShp_File"):
            read_main_input_file("", config(StrmShp_File=""))
        read_main_input_file("", {})
        assert read_main_input_file("", config())["s_stream_length_field"] == "custom_length"
        for suffix, separator in ((".yaml", ": "), (".txt", "\t")):
            path = tmp_path / ("inputs" + suffix)
            path.write_text("".join(f"{key}{separator}{value}\n" for key, value in config().items()))
            assert read_main_input_file(str(path), {})["s_stream_length_field"] == "custom_length"


    def test_independent_coefficients_repeated_on_every_comid_row(self):
        expected = {1: (1000., 5000., -.5), 2: (3500., 12000., .8)}
        rows = {comid: staged_rows(comid, *values) for comid, values in expected.items()}
        with patch("arc.hydraulic_data._build_representative_hydraulic_rows_for_reach",
                   side_effect=lambda comid, *args: rows[comid]):
            frame = build_representative_cross_section_dataframe(
                [{"COMID": 1}, {"COMID": 2}], reach_lengths_m={1: 1000., 2: 3500.},
            )
        for comid, (_, amplitude, exponent) in expected.items():
            group = frame.loc[frame.COMID == comid]
            np.testing.assert_allclose(group.Celerity_K_Power_Law_A, amplitude, rtol=1e-5)
            np.testing.assert_allclose(group.Celerity_K_Power_Law_p, exponent, rtol=1e-5)
            assert group.Celerity_K_Power_Law_A.nunique() == 1
            assert group.Celerity_K_Power_Law_p.nunique() == 1
        assert list(frame.columns) == (
            REPRESENTATIVE_CROSS_SECTION_COLUMNS + REPRESENTATIVE_CROSS_SECTION_COEFFICIENT_COLUMNS[1:]
        )
        assert list(build_representative_cross_section_dataframe([]).columns) == list(frame.columns)


    def test_invalid_or_insufficient_curves_return_nan_without_losing_rows(self):
        assert np.isnan(fit_celerity_k_power_law([1, 2], [1, 2], 1000.)).all()
        assert np.isnan(fit_celerity_k_power_law([1, 2, 3], [1, 2, 3], -1.)).all()
        rows = staged_rows(1, 1000., 5000., .01)
        with patch("arc.hydraulic_data._build_representative_hydraulic_rows_for_reach", return_value=rows):
            frame = build_representative_cross_section_dataframe([{"COMID": 1}])
        assert len(frame) == len(rows)
        assert frame[["Celerity_K_Power_Law_A", "Celerity_K_Power_Law_p"]].isna().all().all()


    def test_nonpositive_discharge_does_not_contribute_to_fit(self):
        q = np.array([1., 5., 10., 25., 50., 100.])
        velocity = 1000. / ((5. / 3.) * 5000. * q ** (-.5))
        expected = fit_celerity_k_power_law(q, velocity, 1000.)
        actual = fit_celerity_k_power_law(
            np.r_[-100., -1., 0., q], np.r_[.001, 1000., .001, velocity], 1000.,
        )
        np.testing.assert_allclose(actual, expected, rtol=0, atol=0)
        self.assertTrue(np.isnan(fit_celerity_k_power_law(
            [-1., 0., 1., 2.], [1., 1., 1., 1.], 1000.,
        )).all())

    def test_fitted_k_is_nonnegative_across_discharge_range(self):
        q = np.array([1., 5., 10., 25., 50., 100.])
        for velocity, expected_exponent in ((np.ones_like(q), 0.), (q ** 0.5, -.5), (1. / q, 1.)):
            amplitude, exponent = fit_celerity_k_power_law(q, velocity, 1000.)
            self.assertGreater(amplitude, 0.)
            self.assertAlmostEqual(exponent, expected_exponent)
            predicted = amplitude * np.r_[1e-6, q, 1e6] ** exponent
            self.assertTrue(np.isfinite(predicted).all())
            self.assertTrue((predicted >= 0).all())

    def test_csv_uses_configured_length_and_preserves_small_exponent(self):
        tmp_path = self.tmp_path
        output = tmp_path / "representative.csv"
        writer = HydraulicData({**read_main_input_file("", config(Representative_Cross_Section_File=str(output))),
                                "b_modified_dem": False})
        writer.xs_data = [{"COMID": 1}]
        q = np.array([1e6, 5e6, 1e7, 5e7, 1e8])
        rows = staged_rows(1, 1000., 5000., 1e-8, q=q)
        streams = pd.DataFrame({"COMID": [1], "custom_length": [1000.], "LengthGeodesicMeters": [9999.]})
        with patch("geopandas.read_file", return_value=streams), patch(
            "arc.hydraulic_data._build_representative_hydraulic_rows_for_reach", return_value=rows,
        ):
            writer.save_representative_cross_section_file()
        exported = pd.read_csv(output)
        assert list(exported.columns) == REPRESENTATIVE_CROSS_SECTION_COLUMNS
        coefficients = pd.read_csv(tmp_path / 'representative_exponents.csv')
        assert list(coefficients.columns) == REPRESENTATIVE_CROSS_SECTION_COEFFICIENT_COLUMNS
        assert coefficients.COMID.tolist() == [1]
        np.testing.assert_allclose(coefficients.Celerity_K_Power_Law_A, 5000., rtol=1e-5)
        np.testing.assert_allclose(coefficients.Celerity_K_Power_Law_p, 1e-8, rtol=1e-5, atol=0)

    def test_companion_file_has_one_record_per_comid(self):
        output = self.tmp_path / 'representative.csv'
        writer = HydraulicData({**read_main_input_file('', config(Representative_Cross_Section_File=str(output))),
                                'b_modified_dem': False})
        writer.xs_data = [{'COMID': 1}, {'COMID': 2}]
        rows = {1: staged_rows(1, 1000., 5000., -.5), 2: staged_rows(2, 2000., 7000., .3)}
        streams = pd.DataFrame({'COMID': [1, 2], 'custom_length': [1000., 2000.]})
        with patch('geopandas.read_file', return_value=streams), patch(
            'arc.hydraulic_data._build_representative_hydraulic_rows_for_reach',
            side_effect=lambda comid, *args: rows[comid],
        ):
            writer.save_representative_cross_section_file()
        coefficients = pd.read_csv(representative_cross_section_coefficients_path(output))
        assert coefficients.COMID.tolist() == [1, 2]
        np.testing.assert_allclose(coefficients.Celerity_K_Power_Law_A, [5000., 7000.])
        np.testing.assert_allclose(coefficients.Celerity_K_Power_Law_p, [-.5, .3])
        assert len(pd.read_csv(output)) == 12

    def test_empty_export_writes_both_schemas_and_preserves_filename(self):
        output = self.tmp_path / 'representative.sections.csv'
        writer = HydraulicData({**read_main_input_file('', config(Representative_Cross_Section_File=str(output))),
                                'b_modified_dem': False})
        writer.xs_data = []
        writer.save_representative_cross_section_file()
        coefficients_path = representative_cross_section_coefficients_path(output)
        assert coefficients_path.name == 'representative.sections_exponents.csv'
        assert pd.read_csv(output).empty
        assert list(pd.read_csv(output).columns) == REPRESENTATIVE_CROSS_SECTION_COLUMNS
        assert pd.read_csv(coefficients_path).empty
        assert list(pd.read_csv(coefficients_path).columns) == REPRESENTATIVE_CROSS_SECTION_COEFFICIENT_COLUMNS


    def test_stream_length_lookup_rejects_missing_invalid_or_duplicate_records(self):
        writer = HydraulicData({**read_main_input_file("", config()), "b_modified_dem": False})
        for streams in (
            pd.DataFrame({"COMID": [1], "other_length": [1000.]}),
            pd.DataFrame({"COMID": [2], "custom_length": [1000.]}),
            pd.DataFrame({"COMID": [1], "custom_length": [0.]}),
            pd.DataFrame({"COMID": [1, 1], "custom_length": [1000., 1000.]}),
        ):
            with patch("geopandas.read_file", return_value=streams), self.assertRaises(ValueError):
                writer._representative_reach_lengths([{"COMID": 1}])


if __name__ == "__main__":
    unittest.main()
