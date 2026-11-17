"""Celerity K coefficient generation, length configuration, and CSV export."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from arc.config import Configs
from arc.outputs import (
    REPRESENTATIVE_CROSS_SECTION_COEFFICIENT_COLUMNS,
    REPRESENTATIVE_CROSS_SECTION_COLUMNS,
    fit_celerity_k_power_law,
    reach_lengths_from_network,
    representative_cross_section_coefficients_path,
    representative_cross_section_dataframe,
    write_representative_cross_sections,
)
from arc.outputs.representative import with_celerity_k

COEFFICIENTS = ['Celerity_K_Power_Law_A', 'Celerity_K_Power_Law_p']


def config(**overrides):
    return {"Build_Representative_Cross_Section": True,
            "Representative_Cross_Section_File": "representative.csv",
            "StrmShp_File": "streams.gpkg", "Stream_Length_m": "custom_length",
            **overrides}


def staged_rows(comid, length, amplitude, exponent, q=None):
    """A reach's stages whose velocities give K = amplitude Q^exponent over a reach this long."""
    if q is None:
        q = np.array([1., 5., 10., 25., 50., 100.])
    velocity = length / ((5. / 3.) * amplitude * q ** exponent)
    rows = pd.DataFrame({column: np.zeros(q.size) for column in REPRESENTATIVE_CROSS_SECTION_COLUMNS})
    stages = np.arange(1, q.size + 1)
    rows["COMID"] = comid
    rows["Cross_Section_Count"] = rows["Hydraulic_Sample_Count"] = 1
    rows["Depth_Stage_Index"] = stages
    rows["Depth_Stage_Meters"] = rows["Mean_Depth"] = stages * .1
    rows["Mean_Discharge"] = q
    rows["Mean_Velocity"] = rows["Representative_Velocity"] = velocity
    rows["Mean_Top_Width"] = 10.
    rows["Mean_Cross_Sectional_Area"] = stages.astype(float)
    return rows


def stages(*reaches):
    return pd.concat(reaches, ignore_index=True)


class RepresentativeCelerityPowerLawTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.tmp_path = Path(directory.name)

    def test_length_argument_required_only_for_representative_output(self):
        tmp_path = self.tmp_path
        for value in ("", "   ", None):
            with self.assertRaisesRegex(ValueError, "requires Stream_Length_m"):
                Configs.from_mapping(config(Stream_Length_m=value))
        missing_length = config()
        del missing_length["Stream_Length_m"]
        with self.assertRaisesRegex(ValueError, "requires Stream_Length_m"):
            Configs.from_mapping(missing_length)
        with self.assertRaisesRegex(ValueError, "requires StrmShp_File"):
            Configs.from_mapping(config(StrmShp_File=""))
        Configs.from_mapping({})
        assert Configs.from_mapping(config()).stream_length_m == "custom_length"
        alias = config()
        alias["s_stream_length_field"] = alias.pop("Stream_Length_m")
        assert Configs.from_mapping(alias).stream_length_m == "custom_length"
        for suffix, separator in ((".yaml", ": "), (".txt", "\t")):
            path = tmp_path / ("inputs" + suffix)
            path.write_text("".join(f"{key}{separator}{value}\n" for key, value in config().items()))
            assert Configs.from_file(path).stream_length_m == "custom_length"

    def test_independent_coefficients_repeated_on_every_comid_row(self):
        expected = {1: (1000., 5000., -.5), 2: (3500., 12000., .8)}
        frame = with_celerity_k(stages(*(staged_rows(comid, *values) for comid, values in expected.items())),
                                {1: 1000., 2: 3500.})
        for comid, (_, amplitude, exponent) in expected.items():
            group = frame.loc[frame.COMID == comid]
            np.testing.assert_allclose(group.Celerity_K_Power_Law_A, amplitude, rtol=1e-5)
            np.testing.assert_allclose(group.Celerity_K_Power_Law_p, exponent, rtol=1e-5)
            assert group.Celerity_K_Power_Law_A.nunique() == 1
            assert group.Celerity_K_Power_Law_p.nunique() == 1
        assert list(frame.columns) == REPRESENTATIVE_CROSS_SECTION_COLUMNS + COEFFICIENTS
        assert list(representative_cross_section_dataframe([]).columns) == list(frame.columns)

    def test_invalid_or_insufficient_curves_return_nan_without_losing_rows(self):
        assert np.isnan(fit_celerity_k_power_law([1, 2], [1, 2], 1000.)).all()
        assert np.isnan(fit_celerity_k_power_law([1, 2, 3], [1, 2, 3], -1.)).all()
        rows = staged_rows(1, 1000., 5000., .01)
        for lengths in (None, {}, {1: np.nan}):
            frame = with_celerity_k(rows, lengths)
            assert len(frame) == len(rows)
            assert frame[COEFFICIENTS].isna().all().all()

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
        output = self.tmp_path / "representative.csv"
        streams = pd.DataFrame({"COMID": [1], "custom_length": [1000.], "LengthGeodesicMeters": [9999.]})
        lengths = reach_lengths_from_network(streams, [1], "custom_length", "LINKNO")
        q = np.array([1e6, 5e6, 1e7, 5e7, 1e8])
        write_representative_cross_sections(with_celerity_k(staged_rows(1, 1000., 5000., 1e-8, q=q), lengths), output)
        exported = pd.read_csv(output)
        assert list(exported.columns) == REPRESENTATIVE_CROSS_SECTION_COLUMNS
        coefficients = pd.read_csv(self.tmp_path / 'representative_exponents.csv')
        assert list(coefficients.columns) == REPRESENTATIVE_CROSS_SECTION_COEFFICIENT_COLUMNS
        assert coefficients.COMID.tolist() == [1]
        assert coefficients.Cross_Section_Count.tolist() == [1]
        np.testing.assert_allclose(coefficients.Celerity_K_Power_Law_A, 5000., rtol=1e-5)
        np.testing.assert_allclose(coefficients.Celerity_K_Power_Law_p, 1e-8, rtol=1e-5, atol=0)

    def test_companion_file_has_one_record_per_comid(self):
        for suffix, read in ((".csv", pd.read_csv), (".parquet", pd.read_parquet)):
            output = self.tmp_path / f'representative{suffix}'
            frame = with_celerity_k(stages(staged_rows(1, 1000., 5000., -.5), staged_rows(2, 2000., 7000., .3)),
                                    {1: 1000., 2: 2000.})
            write_representative_cross_sections(frame, output)
            coefficients = read(representative_cross_section_coefficients_path(output))
            assert coefficients.COMID.tolist() == [1, 2]
            np.testing.assert_allclose(coefficients.Celerity_K_Power_Law_A, [5000., 7000.])
            np.testing.assert_allclose(coefficients.Celerity_K_Power_Law_p, [-.5, .3])
            assert len(read(output)) == 12

    def test_empty_export_writes_both_schemas_and_preserves_filename(self):
        output = self.tmp_path / 'representative.sections.csv'
        write_representative_cross_sections(None, output)
        coefficients_path = representative_cross_section_coefficients_path(output)
        assert coefficients_path.name == 'representative.sections_exponents.csv'
        assert pd.read_csv(output).empty
        assert list(pd.read_csv(output).columns) == REPRESENTATIVE_CROSS_SECTION_COLUMNS
        assert pd.read_csv(coefficients_path).empty
        assert list(pd.read_csv(coefficients_path).columns) == REPRESENTATIVE_CROSS_SECTION_COEFFICIENT_COLUMNS

    def test_stream_length_lookup_rejects_missing_invalid_or_duplicate_records(self):
        for streams in (
            pd.DataFrame({"COMID": [1], "other_length": [1000.]}),
            pd.DataFrame({"COMID": [2], "custom_length": [1000.]}),
            pd.DataFrame({"COMID": [1], "custom_length": [0.]}),
            pd.DataFrame({"COMID": [1, 1], "custom_length": [1000., 1000.]}),
        ):
            with self.assertRaises(ValueError):
                reach_lengths_from_network(streams, [1], "custom_length", "LINKNO")
        with self.assertRaises(ValueError):
            reach_lengths_from_network(None, [1], "custom_length", "LINKNO")
        assert reach_lengths_from_network(None, [], "custom_length", "LINKNO") == {}
        # Without a COMID column, the reach IDs are the stream lines' reach_id
        streams = pd.DataFrame({"LINKNO": [1, 2], "custom_length": [1000., 2000.]})
        assert reach_lengths_from_network(streams, [2], "custom_length", "LINKNO") == {2: 2000.}


if __name__ == "__main__":
    unittest.main()
