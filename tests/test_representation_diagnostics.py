import unittest
import numpy as np
from scripts.validation.representation_diagnostics import geometry, rho


class GeometryTests(unittest.TestCase):
    def test_collapse(self):
        result = geometry(np.ones((10, 4)))
        self.assertEqual(result['participation_rank'], 0)
        self.assertEqual(result['entropy_rank'], 0)
        self.assertIsNone(result['top_component_fraction'])

    def test_isotropic_and_rank_one(self):
        x = np.concatenate([np.eye(4), -np.eye(4)])
        self.assertAlmostEqual(geometry(x)['participation_rank'], 4)
        self.assertAlmostEqual(geometry(x)['entropy_rank'], 4)
        x[:, 1:] = 0
        self.assertAlmostEqual(geometry(x)['participation_rank'], 1)
        self.assertAlmostEqual(geometry(x)['entropy_rank'], 1)

    def test_rank_invariance(self):
        x = np.random.default_rng(8).normal(size=(50, 4))
        self.assertAlmostEqual(geometry(x)['participation_rank'], geometry(3*x+7)['participation_rank'])
        self.assertIsNone(rho(np.ones(4), np.arange(4)))
        self.assertAlmostEqual(rho(np.arange(4), -np.arange(4)), -1)

    def test_invalid(self):
        with self.assertRaises(ValueError):
            geometry(np.full((4, 4), np.nan))


if __name__ == '__main__':
    unittest.main()
