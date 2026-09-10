import tempfile
import unittest
from pathlib import Path

import numpy as np

from arc.Automated_Rating_Curve_Generator import read_main_input_file, flood_increments, find_wse
from arc.cross_section import _calculate_all, calculate_discharge_from_wse, _adjust_n_by_depth
from arc.hydraulic_data import build_representative_cross_section_dataframe


class DepthRoughnessTests(unittest.TestCase):
    def test_api_and_input_files(self):
        defaults = read_main_input_file('', {})
        self.assertEqual((defaults['k_decay'], defaults['shallow_factor'], defaults['deep_factor']), (6.0, 2.0, 1.0))
        values = {'k_decay': 2.5, 'shallow_factor': 1.2, 'deep_factor': 0.7, 'slope_adjustment_factor': 1.4}
        self.assertEqual(defaults['slope_adjustment_factor'], 1.0)
        parsed = read_main_input_file('', values)
        self.assertEqual([parsed[k] for k in values], list(values.values()))
        with tempfile.TemporaryDirectory() as directory:
            for suffix, separator in (('.txt', '\t'), ('.yaml', ': ')):
                path = Path(directory) / ('inputs' + suffix)
                path.write_text(''.join(f'{k}{separator}{v}\n' for k, v in values.items()))
                parsed = read_main_input_file(str(path), {})
                self.assertEqual([parsed[k] for k in values], list(values.values()))
        for key in values:
            for value in (-1, float('nan'), float('inf'), 'invalid'):
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    read_main_input_file('', {key: value})

    def test_parameter_limits_and_legacy_parameters(self):
        parsed = read_main_input_file('', {'shallow_factor': 1., 'deep_factor': 1.})
        self.assertEqual(parsed['shallow_factor'], 1.)
        for key, value in (('k_decay', 0.), ('shallow_factor', 0.9), ('deep_factor', 0.), ('deep_factor', 1.1)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                read_main_input_file('', {key: value})
        for obsolete in ('f_min', 'alpha_boost', 'alpha_low'):
            with self.subTest(obsolete=obsolete), self.assertRaisesRegex(ValueError, 'has been replaced'):
                read_main_input_file('', {obsolete: 0.6})

    def test_slope_adjustment_scales_discharge_once(self):
        with self.assertRaisesRegex(ValueError, 'slope_adjustment_factor'):
            read_main_input_file('', {'slope_adjustment_factor': 0.})
        profile = np.array([0., 0., 1., 2.])
        roughness = np.full(4, 0.03)
        args = (profile, 4, roughness, profile, 4, roughness, 1.)
        base = _calculate_all(*args, 0.5, 0.03, 6., 2., 1.)
        base_q = calculate_discharge_from_wse(0.5, 0.03, *args, 6., 2., 1.)
        for factor in (0.5, 1., 2.):
            scaled = _calculate_all(*args, 0.5, 0.03, 6., 2., 1., factor)
            np.testing.assert_allclose(np.array(scaled)[[0, 1, 4, 5]], np.array(base)[[0, 1, 4, 5]])
            self.assertAlmostEqual(scaled[3], base[3] * factor, delta=0.002)
            self.assertAlmostEqual(scaled[2], base[2] * factor, delta=0.002)
            searched = calculate_discharge_from_wse(0.5, 0.03, *args, 6., 2., 1., factor)
            self.assertAlmostEqual(searched, base_q * factor)
            wse, q, accepted = find_wse(100, 0., 0.01, searched, args + (6., 2., 1., factor), 0.0009)
            self.assertTrue(accepted)
            self.assertAlmostEqual(wse, 0.5, delta=0.01)
            output = np.zeros((1, 18))
            flood_increments(3, 0.5, args, 0., 0.0009, 1e6, output, 0, False, 6., 2., 1., factor)
            self.assertAlmostEqual(output[0, 8], scaled[3], delta=0.002)

    def test_adjustment_endpoints_and_input_preservation(self):
        n = np.full(4, 0.03)
        depth = np.array([-1., 0., 0.5, 100.])
        expected = n * (0.6 + (2.5 - 0.6) * np.exp(-2. * np.maximum(depth, 0.)))
        adjusted = _adjust_n_by_depth(n, depth, 2.5, 0.6, 2.)
        np.testing.assert_allclose(adjusted, expected)
        np.testing.assert_array_equal(n, np.full(4, 0.03))
        np.testing.assert_array_equal(depth, [-1., 0., 0.5, 100.])
        np.testing.assert_allclose(_adjust_n_by_depth(n, depth, 1., 1., 2.), n)

    def test_uniform_depth_roughness_fraction(self):
        profile = np.zeros(4)
        roughness = np.full(4, 0.03)
        args = (profile, 4, roughness, profile, 4, roughness, 1.)
        for depth in (0.1, 0.5, 2.):
            base = _calculate_all(*args, depth, 0.03, 6., 1., 1.)
            reduced = _calculate_all(*args, depth, 0.03, 6., 2., 1.)
            multiplier = 1.0 + np.exp(-6. * depth)
            self.assertAlmostEqual(reduced[5] / base[5], multiplier, delta=0.005)
            np.testing.assert_allclose(np.array(reduced)[[0, 1, 4]], np.array(base)[[0, 1, 4]])

    def test_discharge_and_flood_staging_use_parameters(self):
        profile = np.array([0., 0., 1., 2.])
        roughness = np.full(4, 0.03)
        args = (profile, 4, roughness, profile, 4, roughness, 1.0)
        baseline = _calculate_all(*args, 0.5, 0.03, 6., 1., 1.)
        no_decay = _calculate_all(*args, 0.5, 0.03, 2., 1., 1.)
        np.testing.assert_allclose(baseline, no_decay)
        decayed = _calculate_all(*args, 0.5, 0.03, 6., 2., 1.)
        self.assertLess(decayed[3], baseline[3])
        np.testing.assert_allclose(_calculate_all(*args, 0.5, 0.03), decayed)
        padded_n = np.full(8, 0.03)
        padded_args = (profile, 4, padded_n, profile, 4, padded_n, 1.0)
        np.testing.assert_allclose(_calculate_all(*padded_args, 0.5, 0.03), decayed)
        self.assertAlmostEqual(
            calculate_discharge_from_wse(0.5, 0.03, *padded_args), decayed[3], delta=0.01,
        )
        for decay, shallow, deep in ((0.5, 2., 1.), (6., 1., 1.), (6., 2., 1.), (2., 1.2, 0.7)):
            expected = _calculate_all(*args, 0.5, 0.03, decay, shallow, deep)[3]
            searched = calculate_discharge_from_wse(0.5, 0.03, *args, decay, shallow, deep)
            self.assertAlmostEqual(searched, expected, delta=0.01)
            wse, discharge, accepted = find_wse(
                100, 0., 0.01, searched, args + (decay, shallow, deep), 0.0009,
            )
            self.assertTrue(accepted)
            self.assertAlmostEqual(wse, 0.5, delta=0.01)
            self.assertAlmostEqual(discharge, searched, delta=0.01)
            output = np.zeros((1, 18))
            flood_increments(3, 0.5, args, 0., 0.0009, 1e6, output, 0, False, decay, shallow, deep)
            self.assertAlmostEqual(output[0, 8], expected, delta=0.01)

    def test_representative_export_uses_parameters(self):
        record = {'COMID': 1, 'Thalweg': 0., 'Slope': 0.001,
                  'XS1_Profile': [0., 0., 1., 2.], 'XS2_Profile': [0., 0., 1., 2.],
                  'Manning_N_Raster1': [0.03] * 4, 'Manning_N_Raster2': [0.03] * 4,
                  'Ordinate_Dist': 1.}
        base = build_representative_cross_section_dataframe([record], 6., 1., 1.)
        boosted = build_representative_cross_section_dataframe([record], 6., 2., 1.)
        self.assertFalse(base.empty)
        self.assertTrue((boosted.Mean_Discharge <= base.Mean_Discharge).all())
        self.assertLess(boosted.Mean_Discharge.iloc[0], base.Mean_Discharge.iloc[0])
        default = build_representative_cross_section_dataframe([record])
        np.testing.assert_allclose(default.Mean_Discharge, boosted.Mean_Discharge)
        scaled = build_representative_cross_section_dataframe([record], 6., 2., 1., 1.5)
        np.testing.assert_allclose(scaled.Mean_Discharge, default.Mean_Discharge * 1.5, atol=0.002)



if __name__ == '__main__':
    unittest.main()
