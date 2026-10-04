import unittest
import numpy as np
from scripts.validation.scorer_robustness import all_contrasts,SEEDS
class ContrastTests(unittest.TestCase):
 def test_all_planned_comparisons_and_no_offset_effect(self):
  cs=all_contrasts();self.assertEqual(len(cs),27)
  for c in cs.values():self.assertAlmostEqual(sum(c.values()),0.)
 def test_paired_seed_mean_matches_coefficient_mean(self):
  rng=np.random.default_rng(12)
  for coeff in all_contrasts().values():
   values={n:float(rng.uniform()) for n in coeff}
   paired=[sum(c*values[n]*(3 if '_seed' in n else 1) for n,c in coeff.items() if '_seed' not in n or f'_seed{s}_' in n) for s in SEEDS]
   self.assertAlmostEqual(float(np.mean(paired)),sum(c*values[n] for n,c in coeff.items()))
 def test_same_seed_random_and_fixed_control(self):
  cs=all_contrasts()
  self.assertEqual(cs['aspen_white_knn_minus_random']['aspen_seed29_white_knn'],1/3)
  self.assertEqual(cs['aspen_white_knn_minus_random']['random_seed29_white_knn'],-1/3)
  self.assertEqual(cs['lhco_mahalanobis_minus_girth_high']['girth_high'],-1.)
if __name__=='__main__':unittest.main()
