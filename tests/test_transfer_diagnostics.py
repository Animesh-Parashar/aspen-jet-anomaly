import unittest
import numpy as np
from scripts.validation.transfer_diagnostics import histogram, constituent_values, correlation, rotation_check


class TransferDiagnosticsTests(unittest.TestCase):
    def test_histogram_tails_and_last_edge(self):
        h = histogram([-1, 0, 1, 2, 3], [0, 1, 2], [1, 2, 3, 4, 5])
        np.testing.assert_allclose(h['probability'], [2/15, 7/15])
        self.assertAlmostEqual(sum(h['probability'])+h['underflow']+h['overflow'], 1)

    def test_each_jet_has_equal_particle_weight(self):
        x = np.zeros((2, 3, 4)); x[0, :, 3] = [1, 999, 999]; x[1, :, 3] = [2, 3, 4]
        mask = np.array([[False, True, True], [False, False, False]])
        values, weights = constituent_values(x, mask)
        np.testing.assert_array_equal(values, [1, 2, 3, 4])
        np.testing.assert_allclose(weights, [1, 1/3, 1/3, 1/3])

    def test_tied_spearman_and_constant(self):
        self.assertAlmostEqual(correlation([1, 1, 2, 3], [2, 2, 1, 0]), -1)
        self.assertIsNone(correlation([1, 1, 1], [1, 2, 3]))
        with self.assertRaises(ValueError): correlation([1, np.nan], [1, 2])

    def test_rotation_bound_and_padding(self):
        x = np.zeros((3, 2, 4)); x[:, 0, :2] = [.8, .8]; x[:, 1, :2] = [100, 100]
        mask = np.array([[False, True]]*3)
        r = rotation_check(x, mask)
        self.assertTrue(r['rejection_impossible_for_all_angles'])
        self.assertEqual(r['one_angle_mc_rejection_fraction'], 0)
        x[:, 0, 0] = 4
        self.assertFalse(rotation_check(x, mask)['rejection_impossible_for_all_angles'])


if __name__ == '__main__': unittest.main()
