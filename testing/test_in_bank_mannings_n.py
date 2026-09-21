import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import arc.Automated_Rating_Curve_Generator as generator
from arc.cross_section import CrossSection


class InBankManningsTests(unittest.TestCase):
    def test_bank_cells_use_table_water_value_and_survive_resampling(self):
        raster = np.full((3, 9), 0.12, dtype=np.float32)
        record = {
            'xs1_profile': np.zeros(5), 'xs2_profile': np.zeros(5),
            'xs1_row': np.ones(5, dtype=int), 'xs2_row': np.ones(5, dtype=int),
            'xs1_col': np.arange(4, -1, -1), 'xs2_col': np.arange(4, 9),
            'bank_search_result': {'is_valid': True, 'i_bank_1_index': 2, 'i_bank_2_index': 1},
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'manning.txt'
            path.write_text('LC_ID\tDescription\tManning_n\n42\tWater\t0.4\n10\tTrees\t0.12\n')
            params = {'s_input_mannings_path': str(path), 'i_lc_water_value': 42}
            invalid = {**record, 'bank_search_result': {'is_valid': False}}
            out_of_range = {**record, 'bank_search_result': {
                'is_valid': True, 'i_bank_1_index': 99, 'i_bank_2_index': 1}}
            with patch.object(generator, '_MANNINGS_N', raster):
                generator._set_in_bank_mannings_n_to_water([None, invalid, out_of_range], params)
                np.testing.assert_allclose(raster, .12)
                generator._set_in_bank_mannings_n_to_water([record], params)
            expected = np.full((3, 9), .12, dtype=np.float32)
            expected[1, 2:6] = .4
            np.testing.assert_array_equal(raster, expected)
            section = object.__new__(CrossSection)
            section.xs1_n = section.xs2_n = 5
            section.ia_xc_row1_index_main = record['xs1_row']
            section.ia_xc_column1_index_main = record['xs1_col']
            section.ia_xc_row2_index_main = record['xs2_row']
            section.ia_xc_column2_index_main = record['xs2_col']
            section.set_mannings_n_values(raster)
            np.testing.assert_allclose(section.mannings_n1, [.4, .4, .4, .12, .12])
            np.testing.assert_allclose(section.mannings_n2, [.4, .4, .12, .12, .12])

    def test_missing_water_class_has_clear_error(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'manning.txt'
            path.write_text('LC_ID\tDescription\tManning_n\n10\tTrees\t0.12\n')
            with self.assertRaisesRegex(ValueError, 'no entry'):
                generator._set_in_bank_mannings_n_to_water([], {
                    's_input_mannings_path': str(path), 'i_lc_water_value': 80})


if __name__ == '__main__':
    unittest.main()
