import unittest
import numpy as np
from scripts.validation.signal_kinematics import weighted_quantile, pair_auc, distribution


class SignalKinematicsTests(unittest.TestCase):
    def test_zero_weights_ties_and_endpoints(self):
        np.testing.assert_array_equal(weighted_quantile([0,1,1,4,99], [0,1,1,2,0], [0,.5,.75,1]), [1,1,4,4])

    def test_auc_matches_explicit_weighted_pairs(self):
        y=np.array([0,1,0,1,1]); x=np.array([1,1,4,5,0]); w=np.array([2,.5,1,3,0])
        numerator=sum(w[i]*w[j]*((x[i]>x[j])+.5*(x[i]==x[j]))
                      for i in range(len(y)) if y[i]==1 for j in range(len(y)) if y[j]==0)
        self.assertAlmostEqual(pair_auc(y,x,w), numerator/(w[y==1].sum()*w[y==0].sum()))

    def test_histogram_endpoint_and_tails(self):
        r=distribution([-1,0,1,2,3],np.ones(5),[0,1,2])
        np.testing.assert_allclose(r['probability'],[.2,.4]);self.assertEqual(r['underflow'],.2);self.assertEqual(r['overflow'],.2)

    def test_invalid_weights(self):
        with self.assertRaises(ValueError): weighted_quantile([1,2],[0,0])
        with self.assertRaises(ValueError): weighted_quantile([1,2],[1,-1])


if __name__ == '__main__': unittest.main()
