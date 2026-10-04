import json
from pathlib import Path
import tempfile
import unittest
import h5py
import numpy as np
from scripts.validation.frozen_topology_eval import load_partition,weighted_points
from src.data.corrected_preprocessing import partition,FEATURE_NAMES
class FinalLoaderTests(unittest.TestCase):
 def test_partition_guard(self):
  with self.assertRaises(ValueError):load_partition(Path('/not-accessed'),'signal','train',2,1)
  with self.assertRaises(ValueError):load_partition(Path('/not-accessed'),'signal','reference',2,1)
 def test_deterministic_samples_and_hash_membership(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);(root/'background').mkdir()
   meta=dict(schema='jet-preprocessing-v2.0',features=FEATURE_NAMES,namespace='LHCO_RD_v2',seed=20260925)
   (root/'COMPLETE.json').write_text(json.dumps({'metadata':meta}))
   (root/'VERIFIED.json').write_text(json.dumps({'status':'passed'}))
   ids=np.array([[i] for i in range(1000) if partition('LHCO_RD_v2',(i,),0)=='test'])
   n=len(ids)
   with h5py.File(root/'background/test.h5','w') as f:
    f.attrs['partition']='background/test';f['event_id']=ids;f['constituents']=np.zeros((n,50,4),np.float32);f['mask']=np.zeros((n,50),bool);f['label']=np.zeros(n,np.int8);f['jet_kinematics']=np.ones((n,4));f['n_constituents']=np.full(n,50)
   a=load_partition(root,'background','test',10,20261004);b=load_partition(root,'background','test',10,20261004)
   np.testing.assert_array_equal(a['event_id'],b['event_id'])
   with h5py.File(root/'background/test.h5','r+') as f:f.attrs['partition']='background/train'
   with self.assertRaises(ValueError):load_partition(root,'background','test',10,20261004)
 def test_tied_operating_points(self):
  r=weighted_points(np.array([0,1]),np.array([1.,1.]),np.ones(2))
  self.assertEqual(r[0]['background_efficiency'],1.)
if __name__=='__main__':unittest.main()
