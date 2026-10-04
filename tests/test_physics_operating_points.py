import unittest
import numpy as np
from scipy.stats import ks_2samp
from scipy.spatial.distance import jensenshannon
from scripts.validation.physics_operating_points import threshold,Shape,quantile_edges,wilson,bin_rates
class PhysicsOperatingTests(unittest.TestCase):
 def test_threshold_order_and_ties(self):
  a=np.arange(100.)
  self.assertEqual(threshold(a,.1),90.)
  self.assertEqual(int((a>=threshold(a,.05)).sum()),5)
  self.assertEqual(float((np.ones(100)>=threshold(np.ones(100),.05)).mean()),1.)
 def test_bootstrap_counts_equal_explicit_duplicate_rows(self):
  rng=np.random.default_rng(27);scores=rng.integers(0,20,100).astype(float)
  for _ in range(20):
   ix=rng.integers(100,size=100);counts=np.bincount(ix,minlength=100)
   for target in [.05,.1]:self.assertEqual(threshold(scores,target,counts),threshold(scores[ix],target))
 def test_invalid_calibration(self):
  for s in [[],[1,np.nan]]:
   with self.assertRaises(ValueError):threshold(s,.1)
  with self.assertRaises(ValueError):threshold([1,2],.1,[0,0])
  with self.assertRaises(ValueError):threshold([1,2],.1,[-1,3])
 def test_shape_independent_ks_js_and_duplicate_bootstrap(self):
  rng=np.random.default_rng(1);m=rng.integers(0,40,100).astype(float);edges=quantile_edges(m);shape=Shape(m,edges)
  selected=m>20
  for _ in range(10):
   ix=rng.integers(100,size=100);counts=np.bincount(ix,minlength=100);ks,js=shape.evaluate(selected,counts)
   inc=m[ix];passed=inc[selected[ix]]
   self.assertAlmostEqual(ks,ks_2samp(inc,passed).statistic,places=12)
   a=np.histogram(inc,edges)[0];b=np.histogram(passed,edges)[0]
   self.assertAlmostEqual(js,jensenshannon(a,b)**2,places=12)
 def test_unchanged_and_empty_shapes(self):
  m=np.array([1.,1.,2.,2.]);s=Shape(m,np.array([-np.inf,1.5,np.inf]))
  np.testing.assert_allclose(s.evaluate(np.ones(4,bool)),[0,0],atol=1e-15)
  np.testing.assert_allclose(s.evaluate(np.array([1,0,1,0],bool)),[0,0],atol=1e-15)
  self.assertTrue(np.isnan(s.evaluate(np.zeros(4,bool))).all())
 def test_bins_include_outside_calibration_support(self):
  edges=quantile_edges(np.arange(100.));v=np.array([-100.,5.,200.]);r=bin_rates(v,edges,np.array([1,0,1],bool))
  self.assertEqual(sum(a['total'] for a in r),3);self.assertEqual(sum(a['selected'] for a in r),2)
 def test_wilson_not_zero_uncertainty_at_endpoints(self):
  self.assertGreater(wilson(0,100)[1],0);self.assertLess(wilson(100,100)[0],1);self.assertIsNone(wilson(0,0))
  lo,hi=wilson(50,100);self.assertAlmostEqual(lo,1-hi)
 def test_calibration_independent_of_signal_and_assessment(self):
  cal=np.arange(100.);before=cal.copy();cut=threshold(cal,.05)
  for sample in [np.arange(500.)*1e5,np.zeros(1000)]:
   selected=sample>=cut;self.assertEqual(threshold(cal,.05),cut)
  np.testing.assert_array_equal(before,cal)
if __name__=='__main__':unittest.main()
