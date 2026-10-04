import unittest
import numpy as np
from scripts.preprocess.three_prong_v2 import particles_from_values,event_identity,NAMESPACE
from src.data.corrected_preprocessing import partition
class ThreeProngTests(unittest.TestCase):
 def test_labeled_and_unlabeled_layouts(self):
  a=np.zeros((2,2100));a[:,0]=10
  np.testing.assert_array_equal(particles_from_values(a),particles_from_values(np.c_[a,np.ones(2)]))
  self.assertEqual(particles_from_values(a).shape,(2,700,3))
 def test_reject_wrong_label_shape_and_negative_pt(self):
  for a in [np.zeros((2,2101)),np.zeros((2,2099)),np.full((2,2100),np.nan),np.full((2,2100),-1.)]:
   with self.assertRaises(ValueError):particles_from_values(a)
 def test_source_namespaces_and_split_stability(self):
  ids=[tuple(event_identity(i)) for i in range(1000)]
  self.assertEqual(len(set(ids)),1000);self.assertTrue(all(e[0]<0 for e in ids))
  splits={e:partition(NAMESPACE,e,1) for e in ids}
  self.assertEqual(splits,{e:partition(NAMESPACE,e,1) for e in ids[::-1]})
  self.assertEqual(set(splits.values()),{'validation','test'})
if __name__=='__main__':unittest.main()
