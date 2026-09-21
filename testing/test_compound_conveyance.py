import unittest
import numpy as np
from arc.cross_section import (
    CrossSection, _compound_section_conveyance, _calculate_all,
    calculate_discharge_from_wse,
)
from arc.Automated_Rating_Curve_Generator import find_wse, flood_increments
from arc.hydraulic_data import build_representative_cross_section_dataframe


class CompoundConveyanceTests(unittest.TestCase):
    def setUp(self):
        self.profile = np.array([0., 1., 1., 3.])
        self.n = np.full(4, 0.03)
        self.args = (self.profile, 4, self.n, self.profile, 4, self.n, 1.)
        self.parameters = (1., 1., 1., 1., 1, 1)

    def test_analytic_compound_section(self):
        area, perimeter, width, conveyance = _compound_section_conveyance(
            self.profile, self.n, 1, self.profile, self.n, 1, 2., 1., 1., 1., 1.)
        channel_area, channel_p = 3., 2. * np.sqrt(2.)
        overbank_area, overbank_p = 1.25, 1. + np.sqrt(1.25)
        expected = (channel_area * (channel_area / channel_p)**(2./3.)
                    + 2. * overbank_area * (overbank_area / overbank_p)**(2./3.)) / 0.03
        self.assertAlmostEqual(area, 5.5)
        self.assertAlmostEqual(perimeter, channel_p + 2. * overbank_p)
        self.assertAlmostEqual(width, 5.)
        self.assertAlmostEqual(conveyance, expected)

    def test_below_banks_matches_original_and_transition_is_continuous(self):
        original = calculate_discharge_from_wse(.5, .03, *self.args, 1., 1., 1.)
        divided = calculate_discharge_from_wse(.5, .03, *self.args, *self.parameters)
        self.assertAlmostEqual(original, divided)
        at_bank = calculate_discharge_from_wse(1., .03, *self.args, *self.parameters)
        above = calculate_discharge_from_wse(1. + 1e-8, .03, *self.args, *self.parameters)
        self.assertAlmostEqual(at_bank, above, places=6)

    def test_overbank_roughness_does_not_reduce_channel_conveyance(self):
        _, _, _, base = _compound_section_conveyance(
            self.profile, self.n, 1, self.profile, self.n, 1, 2., 1., 1., 1., 1.)
        rough = np.array([.03, .03, 3., 3.])
        _, _, _, reduced = _compound_section_conveyance(
            self.profile, rough, 1, self.profile, rough, 1, 2., 1., 1., 1., 1.)
        channel = 3. * (3. / (2. * np.sqrt(2.)))**(2./3.) / .03
        self.assertLess(reduced, base)
        self.assertGreater(reduced, channel)

    def test_wse_flood_and_representative_paths(self):
        q = calculate_discharge_from_wse(2., .03, *self.args, *self.parameters)
        staged = _calculate_all(*self.args, 2., .03, *self.parameters)
        self.assertAlmostEqual(staged[3], q, delta=.001)
        wse, _, accepted = find_wse(300, 0., .01, q, self.args + self.parameters, .0009)
        self.assertTrue(accepted)
        self.assertAlmostEqual(wse, 2., delta=.01)
        output = np.zeros((1, 18))
        flood_increments(3, 2., self.args, 0., .0009, 1e6, output, 0, False, *self.parameters)
        self.assertAlmostEqual(output[0, 8], q, delta=.001)
        record = dict(COMID=1, Thalweg=0., Slope=.0009,
                      XS1_Profile=self.profile, XS2_Profile=self.profile,
                      Manning_N_Raster1=self.n, Manning_N_Raster2=self.n,
                      Ordinate_Dist=1., Bank_Index1=1, Bank_Index2=1)
        frame = build_representative_cross_section_dataframe([record], 1., 1., 1.)
        self.assertAlmostEqual(frame.Mean_Discharge.iloc[19], q, delta=.001)

    def test_representative_velocity_is_filtered_mean_channel_velocity(self):
        record = dict(COMID=1, Thalweg=0., Slope=.0009,
                      XS1_Profile=self.profile, XS2_Profile=self.profile,
                      Manning_N_Raster1=self.n, Manning_N_Raster2=self.n,
                      Ordinate_Dist=1., Bank_Index1=1, Bank_Index2=1)
        slow = {**record, 'Manning_N_Raster1': self.n * 2., 'Manning_N_Raster2': self.n * 2.}
        outlier = {**record, 'Slope': 100.}
        frame = build_representative_cross_section_dataframe([record] * 8 + [slow, outlier], 1., 1., 1.)
        row = frame.iloc[19]
        channel_v = (3. / (2. * np.sqrt(2.)))**(2./3.) / .03 * .03
        expected = (8. * np.round(channel_v, 3) + np.round(channel_v / 2., 3)) / 9.
        self.assertEqual(row.Hydraulic_Sample_Count, 9)
        self.assertAlmostEqual(row.Mean_Velocity, expected)
        self.assertAlmostEqual(row.Representative_Velocity, expected)
        full = _calculate_all(*self.args, 2., .03, *self.parameters)
        channel = _calculate_all(*self.args, 2., .03, *self.parameters, channel_velocity=True)
        np.testing.assert_equal(np.array(full)[[0, 1, 3, 4, 5]], np.array(channel)[[0, 1, 3, 4, 5]])
        self.assertGreater(channel[2], full[2])
        self.assertAlmostEqual(row.Mean_Cross_Sectional_Area, full[0])
        unbanked = _calculate_all(*self.args, 2., .03)
        np.testing.assert_equal(unbanked, _calculate_all(*self.args, 2., .03, channel_velocity=True))

    def test_asymmetric_banks_and_dry_barriers(self):
        from arc.cross_section import _bank_subsection_geometry
        low_bank = np.array([0., 1., 1., 3.])
        high_bank = np.array([0., 2., 0., 3.])
        left = _bank_subsection_geometry(low_bank, self.n, 1, 1.5, 1., 1., 1., 1.)
        right = _bank_subsection_geometry(high_bank, self.n, 1, 1.5, 1., 1., 1., 1.)
        self.assertGreater(left[1, 0], 0.)
        self.assertEqual(right[1, 0], 0.)
        dry = _compound_section_conveyance(low_bank, self.n, 1, high_bank, self.n, 1,
                                          0., 1., 1., 1., 1.)
        np.testing.assert_equal(dry, (0., 0., 0., 0.))
        # Fully submerged profiles retain every actual bed segment.
        wet = _bank_subsection_geometry(low_bank, self.n, 1, 4., 1., 1., 1., 1.)
        self.assertAlmostEqual(wet[:, 1].sum(), np.sqrt(2.) + 1. + np.sqrt(5.))

    def test_invalid_bank_fallback_and_reset(self):
        section = object.__new__(CrossSection)
        section.xs1_n = section.xs2_n = 4
        section.set_hydraulic_banks(dict(is_valid=True, i_bank_1_index=1, i_bank_2_index=2))
        self.assertEqual(section.hydraulic_bank_indices, (1, 2))
        section.set_hydraulic_banks(None)
        self.assertEqual(section.hydraulic_bank_indices, (-1, -1))
        baseline = _calculate_all(*self.args, 2., .03)
        np.testing.assert_equal(baseline, _calculate_all(*self.args, 2., .03, bank_index1=99, bank_index2=1))


if __name__ == '__main__':
    unittest.main()
