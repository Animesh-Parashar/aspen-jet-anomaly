import unittest,sys
from pathlib import Path
import numpy as np
from scipy.spatial.distance import cdist
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/validation'))
from background_metric import fit_metric,transform,knn

class MetricTests(unittest.TestCase):
 def test_diagonal_example(self):
  r=np.array([[2.,0.],[-2.,0.],[0.,4.],[0.,-4.]])
  s=fit_metric(r)
  np.testing.assert_allclose(s['covariance'],np.diag([2.,8.]))
  np.testing.assert_allclose(s['regularized_covariance'],np.diag([2.3,7.7]))
  q=np.array([[1.,3.]])
  np.testing.assert_allclose(np.sum(transform(q,s)**2),1/2.3+9/7.7)
 def test_orthogonal_invariance_and_independent_solve(self):
  rng=np.random.default_rng(44);r=rng.normal(size=(80,5));q=rng.normal(size=(12,5));o=np.linalg.qr(rng.normal(size=(5,5)))[0]
  a=fit_metric(r);b=fit_metric(r@o)
  da=cdist(transform(q,a),transform(r,a));db=cdist(transform(q@o,b),transform(r@o,b))
  np.testing.assert_allclose(da,db,rtol=1e-11,atol=1e-11)
  delta=q-a['mean'];direct=np.einsum('ij,ji->i',delta,np.linalg.solve(a['regularized_covariance'],delta.T))
  np.testing.assert_allclose(np.sum(transform(q,a)**2,1),direct)
  np.testing.assert_allclose(knn(transform(r,a),transform(q,a)),np.partition(da,9,axis=1)[:,9])
 def test_rank_deficiency_regularized_not_empirically_white(self):
  r=np.column_stack([np.arange(30.),np.arange(30.),np.zeros(30)])
  s=fit_metric(r);z=transform(r,s)
  self.assertTrue(np.isfinite(z).all());self.assertLess(s['identity_error'],1e-10)
  self.assertFalse(np.allclose(z.T@z/len(z),np.eye(3)))
  with self.assertRaises(ValueError):fit_metric(np.zeros((10,3)))
  with self.assertRaises(ValueError):fit_metric(np.full((10,3),np.nan))
 def test_queries_do_not_mutate_fit(self):
  rng=np.random.default_rng(2);r=rng.normal(size=(40,4));s=fit_metric(r);before={k:np.copy(v) for k,v in s.items()}
  transform(rng.normal(size=(6,4))*1e4,s)
  for k in s:np.testing.assert_array_equal(s[k],before[k])
  with self.assertRaises(ValueError):transform(np.full((2,4),np.inf),s)
if __name__=='__main__':unittest.main()
