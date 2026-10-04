import sys,unittest
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/validation'))
from physics_ablation_comparison import weighted_auc_sorted
from gate_crosscheck import common_weights

class ComparisonTests(unittest.TestCase):
    def test_weighted_ties_and_duplicate_bootstrap(self):
        rng=np.random.default_rng(14);y=np.r_[np.zeros(100),np.ones(100)];s=rng.integers(0,8,200);pt=rng.uniform(0,1,200)
        fn=weighted_auc_sorted(y,s)
        for _ in range(12):
            ix=np.r_[rng.integers(0,100,100),rng.integers(100,200,100)]
            w,_=common_weights(y[ix],pt[ix],np.array([0,.5,1]))
            agg=np.bincount(ix,weights=w,minlength=200)
            self.assertAlmostEqual(fn(agg),roc_auc_score(y[ix],s[ix],sample_weight=w),places=12)
    def test_all_tied(self):
        self.assertEqual(weighted_auc_sorted(np.array([0,1]),np.ones(2))(np.array([3.,5.])),.5)
if __name__=='__main__':unittest.main()
